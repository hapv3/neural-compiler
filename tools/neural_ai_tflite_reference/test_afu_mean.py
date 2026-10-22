"""Deployed Mean numerical/graph controls, without simulation or a converter."""

import copy
from fractions import Fraction
import unittest

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema

from afu_mean import MeanPlan, fused_mean, parameters
from runtime import coverage, make_interpreter


def fixture(shape=(1, 2, 3, 33), axes=(1, 2), keep=False, tail=True, bypass=False,
            input_scale=0.125, output_scale=0.25, input_zp=-3, output_zp=-7, prefix=False):
    model = schema.ModelT()
    model.version = 3
    model.buffers = [schema.BufferT()]
    graph = schema.SubGraphT()
    graph.tensors, graph.operators = [], []
    model.operatorCodes = []

    def tensor(dims, dtype=schema.TensorType.INT8, scale=1.0, zp=0, data=None):
        item = schema.TensorT()
        item.name = f'tensor_{len(graph.tensors)}'.encode()
        item.shape, item.type, item.buffer = list(dims), dtype, 0
        item.quantization = schema.QuantizationParametersT()
        item.quantization.scale, item.quantization.zeroPoint = [scale], [zp]
        if data is not None:
            buf = schema.BufferT()
            buf.data = np.frombuffer(data.tobytes(), np.uint8)
            item.buffer = len(model.buffers)
            model.buffers.append(buf)
        graph.tensors.append(item)
        return len(graph.tensors) - 1

    def op(code, inputs, output, options, version):
        c = schema.OperatorCodeT()
        c.builtinCode = c.deprecatedBuiltinCode = code
        c.version = version
        item = schema.OperatorT()
        item.opcodeIndex = len(model.operatorCodes)
        item.inputs, item.outputs = inputs, [output]
        item.builtinOptions = options
        item.builtinOptionsType = getattr(schema.BuiltinOptions, type(options).__name__[:-1])
        model.operatorCodes.append(c)
        graph.operators.append(item)

    x = tensor(shape, scale=input_scale, zp=input_zp)
    mean_input = x
    if prefix:
        mean_input = tensor(shape, scale=input_scale, zp=input_zp)
        options = schema.ReshapeOptionsT()
        options.newShape = list(shape)
        op(schema.BuiltinOperator.RESHAPE, [x], mean_input, options, 1)
    axis = tensor([len(axes)], schema.TensorType.INT32, data=np.array(axes, np.int32))
    normalized = {a % len(shape) for a in axes}
    yshape = [1 if i in normalized else d for i, d in enumerate(shape)]
    if not keep:
        yshape = [d for i, d in enumerate(shape) if i not in normalized]
    y = tensor(yshape, scale=output_scale, zp=output_zp)
    options = schema.ReducerOptionsT()
    options.keepDims = keep
    op(schema.BuiltinOperator.MEAN, [mean_input, axis], y, options, 2)
    out = y
    if tail:
        weights = tensor([3, shape[-1]], scale=0.5,
                         data=(np.arange(3 * shape[-1]) % 7 - 3).astype(np.int8).reshape(3, shape[-1]))
        out = tensor([1, 3], scale=output_scale * 8, zp=2)
        op(schema.BuiltinOperator.FULLY_CONNECTED, [y, weights, -1], out, schema.FullyConnectedOptionsT(), 4)
    graph.inputs, graph.outputs = [x], [out, x] if bypass else [out]
    model.subgraphs = [graph]
    return model


def pack(model):
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    return bytes(builder.Output())


class ArithmeticTest(unittest.TestCase):
    def test_known_mobile_and_small_block_parameters(self):
        q = parameters(0.0235294122248888, 0.018716562539339066, 49, -128, -128)
        self.assertEqual(q, dict(output_multiplier=881533370, output_shift=35,
                                 input_offset=6272, output_zero_point=-128, double_round_shift=0))
        q = parameters(0.125, 0.25, 6, -3, -7)
        self.assertEqual(q['output_multiplier'], 715827882)
        self.assertEqual(q['output_shift'], 33)
        self.assertEqual(q['input_offset'], 18)

    def test_negative_ties_are_half_up_not_mac_round_away(self):
        x = np.array([-5, -3, -1, 1, 3, 5], np.int8).reshape(1, 1, 6)
        q = parameters(0.5, 1.0, 1, 0, 0)
        np.testing.assert_array_equal(fused_mean(x, (1,), (1, 6), q), [[-2, -1, 0, 1, 2, 3]])

    def test_all_byte_values_scalar_rational_control_and_saturation(self):
        rng = np.random.default_rng(625)
        for count in (1, 2, 6, 7, 49, 129):
            x = rng.integers(-128, 128, (1, count, 256), dtype=np.int8)
            x[0, 0] = np.arange(-128, 128, dtype=np.int8)
            for ratio, izp, ozp in ((0.5, 0, 0), (1.2571438892923814, -128, -128),
                                    (2.0, 127, 127), (0.03125, -3, 2)):
                q = parameters(ratio, 1.0, count, izp, ozp)
                actual = fused_mean(x, (1,), (1, 256), q)
                expected = []
                for channel in range(256):
                    acc = sum(int(v) - izp for v in x[0, :, channel])
                    exact = Fraction(acc * q['output_multiplier'], 1 << q['output_shift']) + Fraction(1, 2)
                    rounded = exact.numerator // exact.denominator
                    expected.append(max(-128, min(127, rounded + ozp)))
                np.testing.assert_array_equal(actual, [expected])

    def test_shift_zero_and_63_and_invalid_contracts(self):
        q = dict(output_multiplier=1, output_shift=0, input_offset=0, output_zero_point=0, double_round_shift=0)
        x = np.array([[[-128, 127]]], np.int8)
        np.testing.assert_array_equal(fused_mean(x, (1,), (1, 2), q), [[-128, 127]])
        q['output_shift'] = 63
        np.testing.assert_array_equal(fused_mean(x, (1,), (1, 2), q), [[0, 0]])
        for scale in (0, -1, float('nan'), float('inf'), 1e-30, 1e30):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                parameters(scale, 1.0, 7, 0, 0)
        for count in (0, -1, 1 << 30):
            with self.subTest(count=count), self.assertRaises(ValueError):
                parameters(1.0, 1.0, count, -128, 0)
        q['double_round_shift'] = 1
        with self.assertRaises(ValueError):
            fused_mean(x, (1,), (1, 2), q)
        with self.assertRaises(ValueError):
            fused_mean(x.astype(float), (1,), (1, 2), q)


class GraphTest(unittest.TestCase):
    def test_spatial_token_negative_axes_keep_dims_and_channel_tail(self):
        for shape, axes in (((1, 7, 7, 33), (1, 2)), ((1, 13, 65), (-2,)),
                            ((1, 1, 7, 3), (2,)), ((1, 1, 7, 3), (1, 2))):
            for keep in (False, True):
                p = MeanPlan(pack(fixture(shape, axes, keep, tail=False)))
                self.assertEqual(p.spatial, np.prod(shape[1:-1]))
                self.assertEqual(p.input, 0)

    def test_rejects_partial_channel_batch_dynamic_and_nonconstant_mean(self):
        for shape, axes in (((1, 2, 3, 4), (1,)), ((1, 2, 3), (2,)),
                            ((2, 2, 3), (1,)), ((1, 2, 3), (0, 1)), ((1, 2, 3), (1, 1))):
            with self.subTest(shape=shape, axes=axes), self.assertRaises(ValueError):
                MeanPlan(pack(fixture(shape, axes, tail=False)))
        mutations = [lambda m: setattr(m.buffers[1], 'data', None),
                     lambda m: setattr(m.subgraphs[0].tensors[0], 'shapeSignature', [1, -1, 3, 33]),
                     lambda m: setattr(m.subgraphs[0].tensors[2].quantization, 'scale', [0.1, 0.2]),
                     lambda m: setattr(m.subgraphs[0].tensors[2], 'shape', [1, 1, 33]),
                     lambda m: m.subgraphs[0].operators.append(copy.deepcopy(m.subgraphs[0].operators[0]))]
        for mutate in mutations:
            model = fixture(tail=False)
            mutate(model)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                MeanPlan(pack(model))

    def test_split_control_matches_unchanged_runtime_and_bypass(self):
        for tail in (False, True):
            for base, kind in (('neural_ai_reference', 'neural_ai_afu_mean_reference'),
                               ('tflite_reference', 'tflite_afu_mean_reference')):
                model = pack(fixture(tail=tail, bypass=True))
                plain = make_interpreter(model, base, preserve=True)
                afu = make_interpreter(model, kind, preserve=True)
                rng = np.random.default_rng(128)
                for _ in range(2):
                    x = rng.integers(-128, 128, (1, 2, 3, 33), dtype=np.int8)
                    plain.set_tensor(0, x); afu.set_tensor(0, x)
                    plain.invoke()
                    control, actual, source_mean = afu.invoke_comparison()
                    np.testing.assert_array_equal(control, plain.get_tensor(plain.get_output_details()[0]['index']))
                    np.testing.assert_array_equal(source_mean, plain.get_tensor(2))
                    np.testing.assert_array_equal(afu.get_tensor(0), x)
                    np.testing.assert_array_equal(actual, afu.get_tensor(afu.get_output_details()[0]['index']))
                    afu.invoke()
                    np.testing.assert_array_equal(actual, afu.get_tensor(afu.get_output_details()[0]['index']))
                self.assertEqual(coverage(afu)['afu_mean']['contract'], 'afu_mean_int8_fused_single_round_v1')
                if base == 'neural_ai_reference':
                    self.assertFalse(coverage(afu)['mac_fallback_nodes'])
                with self.assertRaises(ValueError):
                    afu.set_tensor(2, source_mean)
                with self.assertRaises(ValueError):
                    afu.plan.compute(x[:, :, :2])
                del plain, afu

    def test_nonempty_prefix_and_unmodeled_public_bypass(self):
        model = pack(fixture(prefix=True, bypass=True))
        plain = make_interpreter(model, preserve=True)
        afu = make_interpreter(model, 'neural_ai_afu_mean_reference', preserve=True)
        x = np.arange(198, dtype=np.int16).astype(np.int8).reshape(1, 2, 3, 33)
        plain.set_tensor(0, x); afu.set_tensor(0, x)
        plain.invoke()
        control, _, _ = afu.invoke_comparison()
        np.testing.assert_array_equal(control, plain.get_tensor(plain.get_output_details()[0]['index']))
        np.testing.assert_array_equal(afu.get_tensor(1), x)
        np.testing.assert_array_equal(afu.get_tensor(0), x)
        del plain, afu

    def test_symbolic_batch_is_resolved_to_one_without_resize(self):
        model = fixture(tail=False)
        for tensor in (model.subgraphs[0].tensors[0], model.subgraphs[0].tensors[2]):
            tensor.shapeSignature = [-1] + tensor.shape[1:]
        self.assertEqual(MeanPlan(pack(model)).input_shape, (1, 2, 3, 33))


if __name__ == '__main__':
    unittest.main()
