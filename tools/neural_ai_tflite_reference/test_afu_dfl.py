"""AFU DFL arithmetic, fail-closed graph matching and hybrid-control tests."""

import copy
import unittest

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema

from afu_dfl import DFLPlan, expectation_q8, quantize_scale, requantize_q8, tables
from runtime import coverage, make_interpreter


def fixture(locations=7):
    model = schema.ModelT()
    model.version = 3
    model.buffers = [schema.BufferT()]
    graph = schema.SubGraphT()
    graph.tensors, graph.operators = [], []
    model.operatorCodes = []

    def tensor(shape, scale=0.1, zp=-128, data=None, dtype=schema.TensorType.INT8):
        t = schema.TensorT()
        t.shape, t.type, t.buffer = shape, dtype, 0
        t.name = f'tensor_{len(graph.tensors)}'.encode()
        t.quantization = schema.QuantizationParametersT()
        t.quantization.scale, t.quantization.zeroPoint = [scale], [zp]
        if data is not None:
            t.buffer = len(model.buffers)
            b = schema.BufferT()
            b.data = np.frombuffer(np.asarray(data).tobytes(), np.uint8)
            model.buffers.append(b)
        graph.tensors.append(t)
        return len(graph.tensors) - 1

    def op(code, inputs, output, options, version=1):
        c = schema.OperatorCodeT()
        c.builtinCode = c.deprecatedBuiltinCode = code
        c.version = version
        o = schema.OperatorT()
        o.opcodeIndex = len(model.operatorCodes)
        o.inputs, o.outputs = inputs, [output]
        o.builtinOptions = options
        o.builtinOptionsType = getattr(schema.BuiltinOptions, type(options).__name__[:-1])
        graph.operators.append(o)
        model.operatorCodes.append(c)

    logits = tensor([1, 4, locations, 16], 0.2654646337, 63)
    classes = tensor([1, 80, locations], 0.00352863618)
    probabilities = tensor([1, 4, locations, 16], 1 / 256)
    softmax = schema.SoftmaxOptionsT()
    softmax.beta = 1
    op(schema.BuiltinOperator.SOFTMAX, [logits], probabilities, softmax, 2)
    ws = 15 / 127
    weights = tensor([1, 1, 1, 16], ws, 0, np.rint(np.arange(16) / ws).astype(np.int8))
    bias = tensor([1], ws / 256, 0, np.array([0], np.int32), schema.TensorType.INT32)
    projected = tensor([1, 4, locations, 1], 0.04498727247)
    conv = schema.Conv2DOptionsT()
    conv.padding = schema.Padding.VALID
    conv.strideH = conv.strideW = conv.dilationHFactor = conv.dilationWFactor = 1
    op(schema.BuiltinOperator.CONV_2D, [probabilities, weights, bias], projected, conv, 3)

    def reshape(source, shape, scale):
        shape_tensor = tensor([len(shape)], 1, 0, np.array(shape, np.int32), schema.TensorType.INT32)
        out = tensor(shape, scale)
        options = schema.ReshapeOptionsT()
        options.newShape = shape
        op(schema.BuiltinOperator.RESHAPE, [source, shape_tensor], out, options)
        return out

    view = reshape(projected, [1, 1, 4, locations], 0.04498727247)
    raw_scales = np.full((1, 4, locations), -64, np.int8)
    raw_scales[:, :, locations // 2:] = 0
    scale = tensor([1, 4, locations], 0.1 / 255, -128, raw_scales)
    multiplied = tensor([1, 1, 4, locations], 0.00352863618)
    op(schema.BuiltinOperator.MUL, [view, scale], multiplied, schema.MulOptionsT(), 2)
    distances = reshape(multiplied, [1, 4, locations], 0.00352863618)
    out = tensor([1, 84, locations], 0.00352863618)
    concat = schema.ConcatenationOptionsT()
    concat.axis = 1
    op(schema.BuiltinOperator.CONCATENATION, [distances, classes], out, concat, 2)
    graph.inputs, graph.outputs = [logits, classes], [out]
    model.subgraphs = [graph]
    return model


def pack(model):
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    return bytes(builder.Output())


class ArithmeticTest(unittest.TestCase):
    def test_deployed_lut_including_wrapped_half(self):
        exp, recip = tables(0.2654646337)
        np.testing.assert_array_equal(exp[:128], np.full(128, 32768))
        self.assertEqual(exp[128], 1)
        self.assertEqual(exp[240], 469)
        self.assertEqual(exp[254], 19269)
        self.assertEqual(exp[255], 25128)
        for i in range(256):
            divisor = 513 + 2 * i
            self.assertEqual(recip[i], ((1 << 37) + divisor // 2) // divisor)

    def test_integer_pipeline_against_scalar_and_extreme_logits(self):
        rng = np.random.default_rng(753)
        x = rng.integers(-128, 128, (257, 16), dtype=np.int8)
        x[0] = 0
        x[1] = -128
        x[1, -1] = 127
        exp, reciprocal = tables(0.2654646337)
        result, stats = expectation_q8(x, exp, reciprocal)
        expected = []
        for row in x.tolist():
            p = [int(exp[(v - max(row)) % 256]) for v in row]
            total, weighted = sum(p), sum(i * v for i, v in enumerate(p))
            shift = total.bit_length() - 1
            address = min(511, max(256, (total * 256) // (1 << shift))) - 256
            product = weighted * 256 * int(reciprocal[address])
            expected.append(min(65535, (product + 2 ** (27 + shift)) // 2 ** (28 + shift)))
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result[0], 1916)
        self.assertGreater(stats['wrapped_delta_distributions'], 0)
        self.assertEqual(stats['min_delta'], -255)

    def test_requant_ties_shift_zero_clamps_and_bad_scales(self):
        np.testing.assert_array_equal(requantize_q8(np.array([0, 1, 3, 65535]), 1, 1, 0), [0, 1, 2, 127])
        np.testing.assert_array_equal(requantize_q8(np.array([0, 1, 65535]), 1, 0), [-128, -127, 127])
        np.testing.assert_array_equal(requantize_q8(np.array([0, 10, 20]), 1, 0, 0, -2, 7), [0, 7, 7])
        for scale in (0, -1, float('nan'), float('inf'), 1e-20, 1e10):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                quantize_scale(scale)
        self.assertEqual(quantize_scale(0.5), (32768, 16))
        for logits in (np.zeros((1, 16)), np.zeros((1, 15), np.int8), np.zeros((0, 16), np.int8)):
            with self.assertRaises(ValueError):
                expectation_q8(logits, *tables(0.2))


class GraphTest(unittest.TestCase):
    def test_recognizes_edges_and_nonstandard_locations(self):
        for count in (1, 7, 33):
            plan = DFLPlan(pack(fixture(count)))
            self.assertEqual(plan.locations, count)
            self.assertEqual(plan.selected, [0, 1, 2, 3, 4])
            self.assertEqual(sum(g['locations'] for g in plan.groups), count)
            self.assertEqual(plan.input, 0)

    def test_nonzero_empty_buffer_ids_are_still_variable_tensors(self):
        model = fixture()
        for tensor in model.subgraphs[0].tensors:
            if tensor.buffer == 0:
                tensor.buffer = len(model.buffers)
                model.buffers.append(schema.BufferT())
        plan = DFLPlan(pack(model))
        self.assertEqual(plan.middle_inputs, [0])
        self.assertEqual(plan.tail_inputs, [11, 1])

    def test_rejects_unsupported_contracts_and_shared_intermediates(self):
        original = fixture()
        mutations = [lambda m: setattr(m.subgraphs[0].operators[0].builtinOptions, 'beta', 0.5),
                     lambda m: setattr(m.subgraphs[0].operators[1].builtinOptions, 'strideW', 2),
                     lambda m: m.subgraphs[0].outputs.append(2),
                     lambda m: setattr(m.subgraphs[0].tensors[0], 'shape', [1, 4, 7, 8]),
                     lambda m: setattr(m.buffers[1], 'data', np.zeros(16, np.uint8)),
                     lambda m: setattr(m.subgraphs[0].tensors[11].quantization, 'zeroPoint', [-127])]
        for change in mutations:
            model = copy.deepcopy(original)
            change(model)
            with self.subTest(change=change), self.assertRaises(ValueError):
                DFLPlan(pack(model))

    def test_split_control_matches_unchanged_hybrid_and_preserves_classes(self):
        model = pack(fixture())
        for base, kind in [('neural_ai_reference', 'neural_ai_afu_dfl_reference'),
                           ('tflite_reference', 'tflite_afu_dfl_reference')]:
            plain = make_interpreter(model, base, preserve=True)
            afu = make_interpreter(model, kind, preserve=True)
            rng = np.random.default_rng(57)
            for _ in range(2):
                for detail in plain.get_input_details():
                    x = rng.integers(-128, 128, tuple(detail['shape']), dtype=np.int8)
                    plain.set_tensor(detail['index'], x)
                    afu.set_tensor(detail['index'], x)
                plain.invoke()
                baseline, actual, _ = afu.invoke_comparison()
                np.testing.assert_array_equal(baseline, plain.get_tensor(plain.get_output_details()[0]['index']))
                np.testing.assert_array_equal(actual[:, 4:], baseline[:, 4:])
                np.testing.assert_array_equal(actual, afu.get_tensor(afu.get_output_details()[0]['index']))
                afu.invoke()
                np.testing.assert_array_equal(actual, afu.get_tensor(afu.get_output_details()[0]['index']))
            self.assertEqual(coverage(afu)['afu_dfl']['contract'], 'afu_dfl16_deployed_signed_delta_v1')
            with self.assertRaises(ValueError):
                afu.get_tensor(2)
            del plain, afu


if __name__ == '__main__':
    unittest.main()
