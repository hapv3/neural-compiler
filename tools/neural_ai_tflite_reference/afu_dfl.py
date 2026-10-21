"""Independent deployed AFU DFL16 arithmetic and an opt-in TFLite graph adapter.

No compiler/firmware code, NAI LUT, or RTL output is used to generate predictions.
The original model and the default hybrid interpreter are never modified.
"""

import copy
import hashlib
import math
from pathlib import Path

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema

CONTRACT = "afu_dfl16_deployed_signed_delta_v1"


def quantize_scale(real):
    """Select a positive 16-bit multiplier and right shift, independently."""
    if not math.isfinite(real) or real <= 0:
        raise ValueError("DFL scale must be finite and positive")
    choices = []
    for shift in range(32):
        scaled = math.ldexp(real, shift)
        if not 1 <= scaled <= 65535:
            continue
        multiplier = math.floor(scaled + 0.5)
        if multiplier <= 65535:
            choices.append((abs(math.ldexp(multiplier, -shift) - real), -shift, multiplier))
    if not choices:
        raise ValueError("DFL scale cannot be represented by the deployed requant contract")
    _, negative_shift, multiplier = min(choices)
    return multiplier, -negative_shift


def tables(input_scale):
    if not math.isfinite(input_scale) or input_scale <= 0:
        raise ValueError("DFL input scale must be finite and positive")
    # Deliberately model the deployed compiler, not an idealized correction:
    # indices 1..127 are interpreted as positive signed-byte deltas and clamp
    # exp to 1. Real differences -255..-129 wrap into this half of the table.
    exp = np.array([max(1, min(32768, math.floor(math.exp(min(0, (i if i < 128 else i - 256) * input_scale))
                                                  * 32768 + 0.5))) for i in range(256)], np.int64)
    reciprocal = np.array([((1 << 37) + (513 + 2 * i) // 2) // (513 + 2 * i)
                           for i in range(256)], np.int64)
    return exp, reciprocal


def expectation_q8(logits, exp, reciprocal):
    """INT8 [...,16] -> unsigned Q8 expectation, with AFU integer rounding."""
    logits = np.asarray(logits)
    if logits.dtype != np.int8 or not logits.size or logits.shape[-1] != 16:
        raise ValueError("DFL16 requires nonempty INT8 distributions with 16 bins")
    exp, reciprocal = np.asarray(exp), np.asarray(reciprocal)
    if (exp.shape != (256,) or reciprocal.shape != (256,) or
            exp.dtype.kind not in 'iu' or reciprocal.dtype.kind not in 'iu' or
            np.any(exp < 1) or np.any(exp > 32768) or
            np.any(reciprocal < 0) or np.any(reciprocal > (1 << 28))):
        raise ValueError("Invalid DFL exp or reciprocal table")
    x = logits.astype(np.int64)
    delta = x - x.max(axis=-1, keepdims=True)
    probability = exp.astype(np.int64)[delta & 255]
    total = probability.sum(axis=-1)
    weighted = (probability * np.arange(16)).sum(axis=-1)
    shifts = np.array([int(v).bit_length() - 1 for v in total.flat], np.int64).reshape(total.shape)
    normalized = (total << 8) >> shifts
    reciprocal_value = reciprocal.astype(np.int64)[np.clip(normalized, 256, 511) & 255]
    # Max product < 2**59, so signed int64 is sufficient (RTL uses 63 bits).
    product = (weighted << 8) * reciprocal_value
    q8 = np.minimum((product + (1 << (27 + shifts))) >> (28 + shifts), 65535)
    return q8, {"distributions": int(total.size), "min_delta": int(delta.min()),
                "wrapped_delta_elements": int(np.count_nonzero(delta < -128)),
                "wrapped_delta_distributions": int(np.count_nonzero(np.any(delta < -128, axis=-1)))}


def requantize_q8(q8, multiplier, shift, zero_point=-128, low=-128, high=127):
    q8 = np.asarray(q8)
    if (q8.dtype.kind not in 'iu' or np.any(q8 < 0) or np.any(q8 > 65535) or
            not 1 <= multiplier <= 65535 or not 0 <= shift <= 31 or
            not -128 <= zero_point <= 127 or not -128 <= low <= high <= 127):
        raise ValueError("Invalid DFL output requantization")
    value = q8.astype(np.int64) * multiplier
    if shift:
        value = (value + (1 << (shift - 1))) >> shift
    return np.clip(value + zero_point, low, high).astype(np.int8)


def _quant(tensor):
    q = tensor.quantization
    if (tensor.type != schema.TensorType.INT8 or q is None or q.scale is None or q.zeroPoint is None or
            len(q.scale) != 1 or len(q.zeroPoint) != 1 or not math.isfinite(float(q.scale[0])) or
            q.scale[0] <= 0 or not -128 <= q.zeroPoint[0] <= 127 or
            tensor.shape is None or any(int(d) <= 0 for d in tensor.shape) or tensor.sparsity is not None):
        raise ValueError("DFL graph requires static, per-tensor quantized INT8 tensors")
    return float(q.scale[0]), int(q.zeroPoint[0])


def _constant(model, tensor):
    data = model.buffers[tensor.buffer].data
    if data is None or not len(data):
        raise ValueError("Expected a constant DFL parameter")
    dtype = {schema.TensorType.INT8: np.int8, schema.TensorType.INT32: '<i4'}.get(tensor.type)
    if dtype is None:
        raise ValueError("Unsupported DFL constant type")
    return np.frombuffer(bytes(data), dtype).reshape(tensor.shape)


class DFLPlan:
    """Recognize a private Softmax -> bin projection -> views -> Mul -> view.

    Dimensions and tensor IDs come from graph edges, never model-specific IDs.
    Reject unsupported patterns instead of silently falling back to TFLite DFL.
    """

    def __init__(self, content):
        self.content = bytes(content)
        if self.content[4:8] != b'TFL3':
            raise ValueError("Expected a TFLite model")
        self.model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(self.content, 0))
        if len(self.model.subgraphs) != 1:
            raise ValueError("DFL reference requires one static subgraph")
        self.graph = self.model.subgraphs[0]
        self.ops = self.graph.operators
        self.codes = [max(int(c.builtinCode), int(c.deprecatedBuiltinCode)) for c in self.model.operatorCodes]
        self.readers = {}
        self.writers = {}
        for index, op in enumerate(self.ops):
            for t in op.inputs:
                self.readers.setdefault(int(t), []).append(index)
            for t in op.outputs:
                if int(t) in self.writers:
                    raise ValueError("Multiple tensor writers are unsupported")
                self.writers[int(t)] = index
        candidates = [i for i, op in enumerate(self.ops) if self.code(op) == schema.BuiltinOperator.SOFTMAX]
        if len(candidates) != 1:
            raise ValueError("DFL reference requires exactly one supported Softmax chain")
        first = candidates[0]
        softmax = self.ops[first]
        if len(softmax.inputs) != 1 or len(softmax.outputs) != 1 or softmax.builtinOptions.beta != 1:
            raise ValueError("DFL requires beta=1 Softmax")
        self.input = int(softmax.inputs[0])
        x = self.graph.tensors[self.input]
        shape = tuple(x.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 4 or shape[3] != 16:
            raise ValueError("DFL reference requires [1,4,locations,16] logits")
        self.locations = int(shape[2])
        self.input_scale, _ = _quant(x)
        if (_quant(self.graph.tensors[int(softmax.outputs[0])]) != (1 / 256, -128) or
                tuple(self.graph.tensors[int(softmax.outputs[0])].shape) != shape):
            raise ValueError("Unsupported Softmax output contract")
        projection_index, projection = self.next(int(softmax.outputs[0]), schema.BuiltinOperator.CONV_2D)
        if len(projection.inputs) != 3 or len(projection.outputs) != 1:
            raise ValueError("DFL requires a constant 16-bin projection with bias")
        p = projection.builtinOptions
        if (p.strideH != 1 or p.strideW != 1 or p.dilationHFactor != 1 or p.dilationWFactor != 1 or
                p.padding not in (schema.Padding.VALID, schema.Padding.SAME) or
                p.fusedActivationFunction != schema.ActivationFunctionType.NONE):
            raise ValueError("Unsupported DFL projection convolution")
        weights = self.graph.tensors[int(projection.inputs[1])]
        weight_scale, weight_zp = _quant(weights)
        real_weights = (_constant(self.model, weights).astype(np.float64) - weight_zp) * weight_scale
        if (tuple(weights.shape) != (1, 1, 1, 16) or
                np.any(np.abs(real_weights.reshape(-1) - np.arange(16)) > weight_scale * 0.51) or
                np.any(_constant(self.model, self.graph.tensors[int(projection.inputs[2])]) != 0)):
            raise ValueError("DFL projection must approximate bins 0..15 with zero bias")
        current = int(projection.outputs[0])
        if tuple(self.graph.tensors[current].shape) != (1, 4, self.locations, 1):
            raise ValueError("Unsupported DFL projection output shape")
        _quant(self.graph.tensors[current])
        selected = [first, projection_index]
        while self.peek_code(current) == schema.BuiltinOperator.RESHAPE:
            current = self.view(current, selected)
        mul_index, mul = self.next(current, schema.BuiltinOperator.MUL)
        if (len(mul.inputs) != 2 or len(mul.outputs) != 1 or
                mul.builtinOptions.fusedActivationFunction != schema.ActivationFunctionType.NONE):
            raise ValueError("DFL requires an unclamped constant box-scale Mul")
        scale_tensor = self.graph.tensors[int(mul.inputs[1] if mul.inputs[0] == current else mul.inputs[0])]
        scale, zp = _quant(scale_tensor)
        raw = _constant(self.model, scale_tensor)
        try:
            scales = np.broadcast_to(raw, self.graph.tensors[int(mul.outputs[0])].shape).reshape(4, self.locations)
        except ValueError as error:
            raise ValueError("DFL scale cannot be broadcast over four sides") from error
        if not np.all(scales == scales[0]):
            raise ValueError("DFL requires the same per-location scale on all four sides")
        self.box_scales = (scales[0].astype(np.float64) - zp) * scale
        if np.any(self.box_scales <= 0):
            raise ValueError("DFL box scales must be positive")
        selected.append(mul_index)
        current = int(mul.outputs[0])
        while self.peek_code(current) == schema.BuiltinOperator.RESHAPE:
            current = self.view(current, selected)
        self.output = current
        if tuple(self.graph.tensors[current].shape) != (1, 4, self.locations):
            raise ValueError("DFL reference requires [1,4,locations] distances")
        self.output_scale, self.zero_point = _quant(self.graph.tensors[current])
        if self.zero_point != -128:
            raise ValueError("Deployed DFL requires output zero point -128")
        if selected != list(range(first, selected[-1] + 1)):
            raise ValueError("Interleaved DFL operators are not supported")
        self.selected = selected
        self.exp, self.reciprocal = tables(self.input_scale)
        self.groups = []
        start = 0
        while start < self.locations:
            end = start + 1
            while end < self.locations and self.box_scales[end] == self.box_scales[start]:
                end += 1
            multiplier, shift = quantize_scale(self.box_scales[start] / (256 * self.output_scale))
            self.groups.append(dict(first=start, locations=end - start, multiplier=multiplier, shift=shift,
                                    box_scale=float(self.box_scales[start])))
            start = end
        self.middle_inputs = self.external_inputs(selected)
        self.tail_nodes = list(range(selected[-1] + 1, len(self.ops)))
        self.tail_inputs = self.external_inputs(self.tail_nodes)
        if self.output not in self.tail_inputs and self.output not in self.graph.outputs:
            raise ValueError("DFL output must reach the model output")
        self.prefix_outputs = list(dict.fromkeys(self.middle_inputs + [i for i in self.tail_inputs if i != self.output]))
        wanted = set(self.prefix_outputs)
        self.prefix_nodes = []
        for i in range(first - 1, -1, -1):
            if wanted.intersection(map(int, self.ops[i].outputs)):
                self.prefix_nodes.append(i)
                wanted.update(map(int, self.ops[i].inputs))
        self.prefix_nodes.reverse()
        if set(self.external_inputs(self.prefix_nodes)) - set(map(int, self.graph.inputs)):
            raise ValueError("DFL prefix depends on an unavailable intermediate")

    def code(self, op):
        return self.codes[op.opcodeIndex]

    def peek_code(self, tensor):
        readers = self.readers.get(tensor, [])
        return self.code(self.ops[readers[0]]) if len(readers) == 1 else None

    def next(self, tensor, code):
        readers = self.readers.get(tensor, [])
        if len(readers) != 1 or tensor in self.graph.outputs or self.code(self.ops[readers[0]]) != code:
            raise ValueError("DFL internal tensors must have one private supported consumer")
        return readers[0], self.ops[readers[0]]

    def view(self, tensor, selected):
        index, op = self.next(tensor, schema.BuiltinOperator.RESHAPE)
        if len(op.outputs) != 1:
            raise ValueError("Invalid DFL view")
        result = int(op.outputs[0])
        if (_quant(self.graph.tensors[result]) != _quant(self.graph.tensors[tensor]) or
                int(np.prod(self.graph.tensors[result].shape)) != 4 * self.locations):
            raise ValueError("DFL views must preserve quantization and elements")
        selected.append(index)
        return result

    def external_inputs(self, nodes):
        produced = {int(t) for i in nodes for t in self.ops[i].outputs}
        return list(dict.fromkeys(int(t) for i in nodes for t in self.ops[i].inputs
                                  if t >= 0 and int(t) not in produced and
                                  not self.has_constant_data(int(t))))

    def has_constant_data(self, tensor):
        data = self.model.buffers[self.graph.tensors[tensor].buffer].data
        return data is not None and len(data) > 0

    def slice(self, nodes, inputs, outputs):
        model = copy.deepcopy(self.model)
        graph = model.subgraphs[0]
        graph.operators = [graph.operators[i] for i in nodes]
        graph.inputs, graph.outputs = list(inputs), list(outputs)
        model.signatureDefs = None
        builder = flatbuffers.Builder(1024)
        builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
        return bytes(builder.Output())

    def compute(self, logits):
        if np.asarray(logits).shape != (1, 4, self.locations, 16):
            raise ValueError("DFL logits shape changed")
        q8, stats = expectation_q8(logits, self.exp, self.reciprocal)
        result = np.empty((1, 4, self.locations), np.int8)
        for group in self.groups:
            region = slice(group['first'], group['first'] + group['locations'])
            result[:, :, region] = requantize_q8(q8[:, :, region], group['multiplier'], group['shift'], self.zero_point)
        return result, stats

    def description(self):
        return dict(contract=CONTRACT, source_sha256=hashlib.sha256(self.content).hexdigest(),
                    source_nodes=self.selected, input_tensor=self.input, output_tensor=self.output,
                    input_scale=self.input_scale, output_scale=self.output_scale, output_zero_point=self.zero_point,
                    groups=self.groups, exp_sha256=hashlib.sha256(self.exp.astype('<u4').tobytes()).hexdigest(),
                    reciprocal_sha256=hashlib.sha256(self.reciprocal.astype('<u4').tobytes()).hexdigest(),
                    limitation="Deployed exp LUT signed-delta alias at differences below -128 is modeled, not fixed")


class AFUDFLInterpreter:
    """Interpreter-compatible opt-in host reference. No RTL timing is modeled."""

    def __init__(self, model, base_kind, preserve, threads, factory):
        self.plan = DFLPlan(model if isinstance(model, bytes) else Path(model).read_bytes())
        self.base_kind, self.factory, self.preserve, self.threads = base_kind, factory, preserve, threads
        p = self.plan
        self.prefix = factory(p.slice(p.prefix_nodes, p.graph.inputs, p.prefix_outputs), base_kind, preserve, threads)
        self.suffix = factory(p.slice(p.tail_nodes, p.tail_inputs, p.graph.outputs), base_kind, preserve, threads)
        self.middle = None
        self.distances = None
        self.last_dfl_stats = None

    def get_input_details(self):
        return self.prefix.get_input_details()

    def get_output_details(self):
        return self.suffix.get_output_details()

    def set_tensor(self, index, value):
        if index not in [int(t['index']) for t in self.get_input_details()]:
            raise ValueError("Set only public inputs on the DFL reference interpreter")
        self.prefix.set_tensor(index, value)

    def get_tensor(self, index):
        if index == self.plan.output:
            if self.distances is None:
                raise ValueError("Invoke DFL reference first")
            return self.distances.copy()
        suffix_tensors = set(self.plan.tail_inputs) | {int(t) for i in self.plan.tail_nodes for t in self.plan.ops[i].outputs}
        if index in suffix_tensors:
            return self.suffix.get_tensor(index)
        omitted = {int(t) for i in self.plan.selected for t in self.plan.ops[i].outputs}
        if index in omitted:
            raise ValueError("This intermediate does not exist in fused AFU DFL")
        return self.prefix.get_tensor(index)

    def invoke(self):
        self.prefix.invoke()
        self.distances, self.last_dfl_stats = self.plan.compute(self.prefix.get_tensor(self.plan.input))
        self._tail(self.distances)

    def _tail(self, distances):
        for tensor in self.plan.tail_inputs:
            self.suffix.set_tensor(tensor, distances if tensor == self.plan.output else self.prefix.get_tensor(tensor))
        self.suffix.invoke()

    def invoke_comparison(self):
        """One identical prefix, source DFL and AFU DFL, then identical tail.

        The source middle is a control, not an approximation of the AFU kernel.
        It allows every image's control output to be checked against an ordinary
        unchanged hybrid invocation without recomputing its Conv prefix twice.
        """
        p = self.plan
        if self.middle is None:
            self.middle = self.factory(p.slice(p.selected, p.middle_inputs, [p.output]),
                                       self.base_kind, self.preserve, self.threads)
        self.prefix.invoke()
        for tensor in p.middle_inputs:
            self.middle.set_tensor(tensor, self.prefix.get_tensor(tensor))
        self.middle.invoke()
        source_distances = self.middle.get_tensor(p.output)
        self._tail(source_distances)
        baseline = self.suffix.get_tensor(self.get_output_details()[0]['index'])
        self.distances, self.last_dfl_stats = p.compute(self.prefix.get_tensor(p.input))
        self._tail(self.distances)
        actual = self.suffix.get_tensor(self.get_output_details()[0]['index'])
        return baseline, actual, source_distances

    def reference_coverage(self):
        from runtime import coverage
        prefix, suffix = coverage(self.prefix), coverage(self.suffix)
        counts = {}
        for part in (prefix, suffix):
            for name, count in part['neural_ai'].items():
                counts[name] = counts.get(name, 0) + count
        return dict(neural_ai=counts,
                    mac_fallback_nodes=[(part, i) for part, value in (('prefix', prefix), ('suffix', suffix))
                                        for i in value['mac_fallback_nodes']],
                    prefix=prefix, suffix=suffix, afu_dfl=self.plan.description(),
                    scope="AFU DFL16 + selected MAC reference; remaining operators use TFLite reference")
