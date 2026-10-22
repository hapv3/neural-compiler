"""Independent deployed AFU Mean arithmetic; opt-in, without changing old oracles.

Only static batch-one INT8 spatial/token reductions to one value per channel
are recognized. Predictions use source quantization, never compiled qparams or
saved RTL bytes. Layouts, queues, cycles and overflowing accumulators are not modeled.
"""

import copy
import hashlib
import math
from pathlib import Path

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema

CONTRACT = "afu_mean_int8_fused_single_round_v1"


def parameters(input_scale, output_scale, spatial, input_zero_point, output_zero_point):
    """Independently select Q31 scale, fold the divisor, and bound the accumulator."""
    if (not all(math.isfinite(s) and s > 0 for s in (input_scale, output_scale)) or
            type(spatial) is not int or spatial <= 0 or
            not isinstance(input_zero_point, (int, np.integer)) or
            not isinstance(output_zero_point, (int, np.integer)) or
            not -128 <= input_zero_point <= 127 or not -128 <= output_zero_point <= 127):
        raise ValueError("Invalid AFU Mean quantization or spatial size")
    ratio = input_scale / output_scale
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Unrepresentable AFU Mean scale")
    significand, exponent = math.frexp(ratio)
    multiplier, shift = math.floor(significand * (1 << 31) + 0.5), 31 - exponent
    if not 0 < multiplier < (1 << 31) or not 0 <= shift <= 62:
        raise ValueError("AFU Mean ratio is outside the deployed Q31 contract")
    divisor_shift = min(spatial.bit_length() - 1, 32, 62 - shift)
    multiplier = (multiplier << divisor_shift) // spatial
    shift += divisor_shift
    offset = -input_zero_point * spatial
    if (not 0 < multiplier < (1 << 31) or not 0 <= shift <= 63 or
            not -(1 << 31) <= offset < (1 << 31) or
            offset - 128 * spatial < -(1 << 31) or
            offset + 127 * spatial >= (1 << 31)):
        raise ValueError("AFU Mean requant or accumulator is outside the deployed contract")
    return dict(output_multiplier=multiplier, output_shift=shift,
                input_offset=offset, output_zero_point=int(output_zero_point), double_round_shift=0)


def fused_mean(values, axes, output_shape, quant):
    """Signed sum, zero-point correction, wide multiply, one half-up shift, clamp.

    AFU adds a positive half-unit before arithmetic shift for both signs: negative
    ties go toward +infinity, unlike the MAC reference's ties-away-from-zero.
    """
    values = np.asarray(values)
    if values.dtype != np.int8 or not values.size:
        raise ValueError("AFU Mean requires nonempty INT8 data")
    if (not 0 < quant['output_multiplier'] < (1 << 31) or not 0 <= quant['output_shift'] <= 63 or
            not -(1 << 31) <= quant['input_offset'] < (1 << 31) or
            not -128 <= quant['output_zero_point'] <= 127):
        raise ValueError("Invalid deployed Mean requant parameters")
    sums = values.astype(np.int64).sum(axis=tuple(axes)) + quant['input_offset']
    if np.any(sums < -(1 << 31)) or np.any(sums >= (1 << 31)):
        raise ValueError("AFU Mean accumulator overflow is not modeled")
    product = sums * quant['output_multiplier']
    shift = quant['output_shift']
    if quant['double_round_shift'] != 0:
        raise ValueError("This version models compiler-emitted single-round Mean only")
    if shift:
        product = (product + (1 << (shift - 1))) >> shift
    return np.clip(product + quant['output_zero_point'], -128, 127).astype(np.int8).reshape(output_shape)


def _quant(tensor):
    q = tensor.quantization
    # Exporters commonly retain a symbolic batch while resolving it to one.
    # No resize API is exposed; any other dynamic dimension is unsupported.
    signature = tensor.shapeSignature
    dynamic_spatial = signature is not None and any(int(d) < 0 for d in signature[1:])
    if (tensor.type != schema.TensorType.INT8 or tensor.isVariable or tensor.sparsity is not None or
            tensor.shape is None or any(int(d) <= 0 for d in tensor.shape) or
            dynamic_spatial or
            q is None or q.scale is None or q.zeroPoint is None or
            len(q.scale) != 1 or len(q.zeroPoint) != 1 or
            not math.isfinite(float(q.scale[0])) or q.scale[0] <= 0 or not -128 <= q.zeroPoint[0] <= 127):
        raise ValueError("AFU Mean requires static per-tensor quantized INT8 tensors")
    return float(q.scale[0]), int(q.zeroPoint[0])


class MeanPlan:
    """One recognized Mean anywhere in a static graph, including bypass edges.

    Multiple Mean nodes are deliberately rejected rather than partially modeled.
    The original FlatBuffer is never edited; interpreter slices are in-memory copies.
    """

    def __init__(self, content):
        self.content = content
        self.model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(content, 0))
        if len(self.model.subgraphs) != 1:
            raise ValueError("AFU Mean reference requires one static subgraph")
        self.graph = self.model.subgraphs[0]
        self.ops = self.graph.operators
        matches = [i for i, op in enumerate(self.ops)
                   if self.model.operatorCodes[op.opcodeIndex].builtinCode == schema.BuiltinOperator.MEAN]
        if len(matches) != 1:
            raise ValueError("AFU Mean reference requires exactly one supported Mean node")
        self.node = matches[0]
        op = self.ops[self.node]
        if len(op.inputs) != 2 or len(op.outputs) != 1 or op.builtinOptionsType != schema.BuiltinOptions.ReducerOptions:
            raise ValueError("Unsupported Mean signature")
        self.input, self.output = int(op.inputs[0]), int(op.outputs[0])
        x, y = (self.graph.tensors[i] for i in (self.input, self.output))
        self.input_scale, self.input_zero_point = _quant(x)
        self.output_scale, self.output_zero_point = _quant(y)
        self.input_shape, self.output_shape = tuple(map(int, x.shape)), tuple(map(int, y.shape))
        rank = len(self.input_shape)
        if rank not in (3, 4) or self.input_shape[0] != 1:
            raise ValueError("Mean must be batch-one NHWC or [1,tokens,channels]")
        axis = self.graph.tensors[int(op.inputs[1])]
        data = self.model.buffers[axis.buffer].data
        if (axis.type != schema.TensorType.INT32 or axis.isVariable or data is None or len(data) == 0 or
                len(data) % 4 or axis.shape is None or math.prod(axis.shape) * 4 != len(data)):
            raise ValueError("Mean axes must be constant INT32")
        raw_axes = np.frombuffer(bytes(data), '<i4').tolist()
        if any(a < -rank or a >= rank for a in raw_axes):
            raise ValueError("Mean axis out of range")
        self.axes = tuple(sorted({a % rank for a in raw_axes}))
        spatial_axes = set(range(1, rank - 1))
        if (not self.axes or len(self.axes) != len(raw_axes) or not set(self.axes) <= spatial_axes or
                any(self.input_shape[a] != 1 for a in spatial_axes - set(self.axes))):
            raise ValueError("Mean must reduce all non-unit spatial/token axes, not batch or channels")
        expected_shape = tuple(1 if a in self.axes else d for a, d in enumerate(self.input_shape))
        if not op.builtinOptions.keepDims:
            expected_shape = tuple(d for a, d in enumerate(self.input_shape) if a not in self.axes)
        if self.output_shape != expected_shape:
            raise ValueError("Mean output shape does not match its axes/keep_dims")
        self.spatial = math.prod(self.input_shape[a] for a in self.axes)
        self.quant = parameters(self.input_scale, self.output_scale, self.spatial,
                                self.input_zero_point, self.output_zero_point)
        # Validate topological availability, including public outputs and bypasses.
        available = set(map(int, self.graph.inputs)) | {
            i for i in range(len(self.graph.tensors)) if self.constant(i)}
        for item in self.ops:
            if any(int(t) not in available for t in item.inputs if t >= 0):
                raise ValueError("Mean graph has unavailable/non-topological inputs")
            if any(int(t) in available for t in item.outputs):
                raise ValueError("Mean graph reuses an existing tensor producer")
            available.update(map(int, item.outputs))
        if not set(map(int, self.graph.outputs)) <= available:
            raise ValueError("Mean graph output has no producer")
        self.prefix_nodes, self.tail_nodes = list(range(self.node)), list(range(self.node + 1, len(self.ops)))
        tail_produced = {int(t) for i in self.tail_nodes for t in self.ops[i].outputs}
        self.tail_inputs = self.external_inputs(self.tail_nodes)
        self.tail_inputs += [int(t) for t in self.graph.outputs
                             if int(t) not in tail_produced and int(t) not in self.tail_inputs]
        self.prefix_outputs = list(dict.fromkeys([self.input] + [t for t in self.tail_inputs if t != self.output]))

    def constant(self, tensor):
        data = self.model.buffers[self.graph.tensors[tensor].buffer].data
        return data is not None and len(data) > 0

    def external_inputs(self, nodes):
        produced = {int(t) for i in nodes for t in self.ops[i].outputs}
        return list(dict.fromkeys(int(t) for i in nodes for t in self.ops[i].inputs
                                  if t >= 0 and int(t) not in produced and not self.constant(int(t))))

    def slice(self, nodes, inputs, outputs):
        model = copy.deepcopy(self.model)
        graph = model.subgraphs[0]
        graph.operators = [graph.operators[i] for i in nodes]
        graph.inputs, graph.outputs = list(inputs), list(outputs)
        model.signatureDefs = None
        builder = flatbuffers.Builder(1024)
        builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
        return bytes(builder.Output())

    def compute(self, values):
        if np.asarray(values).shape != self.input_shape:
            raise ValueError("AFU Mean input shape changed")
        return fused_mean(values, self.axes, self.output_shape, self.quant)

    def description(self):
        return dict(contract=CONTRACT, source_sha256=hashlib.sha256(self.content).hexdigest(),
                    source_node=self.node, input_tensor=self.input, output_tensor=self.output,
                    input_shape=self.input_shape, output_shape=self.output_shape, axes=self.axes,
                    spatial=self.spatial, input_scale=self.input_scale, output_scale=self.output_scale,
                    input_zero_point=self.input_zero_point, **self.quant)


class AFUMeanInterpreter:
    """Opt-in interpreter; source-Mean control is retained for paired comparisons."""

    def __init__(self, model, base_kind, preserve, threads, factory):
        self.plan = MeanPlan(model if isinstance(model, bytes) else Path(model).read_bytes())
        self.base_kind, self.factory, self.preserve, self.threads = base_kind, factory, preserve, threads
        p = self.plan
        self.prefix = factory(p.slice(p.prefix_nodes, p.graph.inputs, p.prefix_outputs), base_kind, preserve, threads)
        self.suffix = factory(p.slice(p.tail_nodes, p.tail_inputs, p.graph.outputs), base_kind, preserve, threads)
        self.middle, self.mean = None, None

    def get_input_details(self):
        return self.prefix.get_input_details()

    def get_output_details(self):
        return self.suffix.get_output_details()

    def set_tensor(self, index, value):
        if index not in [int(t['index']) for t in self.get_input_details()]:
            raise ValueError("Set only public inputs on the AFU Mean interpreter")
        self.prefix.set_tensor(index, value)

    def get_tensor(self, index):
        if index == self.plan.output:
            if self.mean is None:
                raise ValueError("Invoke AFU Mean reference first")
            return self.mean.copy()
        tail = set(self.plan.tail_inputs) | {int(t) for i in self.plan.tail_nodes for t in self.plan.ops[i].outputs}
        return (self.suffix if index in tail else self.prefix).get_tensor(index)

    def _tail(self, values):
        for tensor in self.plan.tail_inputs:
            self.suffix.set_tensor(tensor, values if tensor == self.plan.output else self.prefix.get_tensor(tensor))
        self.suffix.invoke()

    def invoke(self):
        self.prefix.invoke()
        self.mean = self.plan.compute(self.prefix.get_tensor(self.plan.input))
        self._tail(self.mean)

    def invoke_comparison(self):
        """Identical prefix, original versus deployed Mean, then identical tails."""
        p = self.plan
        if self.middle is None:
            self.middle = self.factory(p.slice([p.node], [p.input], [p.output]),
                                       self.base_kind, self.preserve, self.threads)
        self.prefix.invoke()
        values = self.prefix.get_tensor(p.input)
        self.middle.set_tensor(p.input, values)
        self.middle.invoke()
        source_mean = self.middle.get_tensor(p.output)
        self._tail(source_mean)
        control = self.suffix.get_tensor(self.get_output_details()[0]['index'])
        self.mean = p.compute(values)
        self._tail(self.mean)
        actual = self.suffix.get_tensor(self.get_output_details()[0]['index'])
        return control, actual, source_mean

    def reference_coverage(self):
        from runtime import coverage
        prefix, suffix = coverage(self.prefix), coverage(self.suffix)
        counts = {}
        for part in (prefix, suffix):
            for name, count in part['neural_ai'].items():
                counts[name] = counts.get(name, 0) + count
        return dict(neural_ai=counts, afu_mean=self.plan.description(), prefix=prefix, suffix=suffix,
                    mac_fallback_nodes=[(part, i) for part, report in (('prefix', prefix), ('suffix', suffix))
                                        for i in report['mac_fallback_nodes']],
                    scope="One deployed AFU Mean + selected MAC reference; other ops remain TFLite reference")
