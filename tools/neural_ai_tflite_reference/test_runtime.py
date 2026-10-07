"""Small native-kernel tests; no RTL simulation or TensorFlow converter required."""

import ctypes
import unittest

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema

from runtime import ROOT, coverage, make_interpreter


def model_bytes(code, xshape, weight, yshape, params, scales, bias=None, activation=None,
                input_scale=0.125, output_scale=0.25, input_zp=-7, output_zp=3,
                adjacent=False):
    """Build tiny, deterministic operator graphs without training/conversion."""
    model = schema.ModelT()
    model.version = 3
    model.buffers = [schema.BufferT()]
    sub = schema.SubGraphT()
    sub.tensors = []

    def tensor(shape, dtype, scale, zp, data=None, axis=0):
        item = schema.TensorT()
        item.name = f"tensor_{len(sub.tensors)}".encode()
        item.shape = shape
        item.type = dtype
        item.buffer = 0
        item.quantization = schema.QuantizationParametersT()
        item.quantization.scale = scale
        item.quantization.zeroPoint = zp
        item.quantization.quantizedDimension = axis
        if data is not None:
            buf = schema.BufferT()
            buf.data = np.frombuffer(data.tobytes(), np.uint8)
            item.buffer = len(model.buffers)
            model.buffers.append(buf)
        sub.tensors.append(item)
        return len(sub.tensors) - 1

    x = tensor(xshape, schema.TensorType.INT8, [input_scale], [input_zp])
    w = tensor(list(weight.shape), schema.TensorType.INT8, scales, [0] * len(scales), weight,
               3 if code == schema.BuiltinOperator.DEPTHWISE_CONV_2D else 0)
    b = -1 if bias is None else tensor(list(bias.shape), schema.TensorType.INT32,
                                      [s * input_scale for s in scales], [0] * len(scales), bias)
    y = tensor(yshape, schema.TensorType.INT8, [output_scale], [output_zp])
    if activation is not None:
        params.fusedActivationFunction = activation
    op = schema.OperatorT()
    op.opcodeIndex = 0
    op.inputs = [x, w, b]
    op.outputs = [y]
    op.builtinOptions = params
    op.builtinOptionsType = getattr(schema.BuiltinOptions, type(params).__name__[:-1])
    sub.inputs = [x]
    sub.outputs = [y]
    sub.operators = [op]
    if adjacent:
        # Square FC with identical input/output quantization; preserve intermediate.
        z = tensor(yshape, schema.TensorType.INT8, [output_scale], [output_zp])
        second = schema.OperatorT()
        second.opcodeIndex = 0
        second.inputs = [y, w, b]
        second.outputs = [z]
        second.builtinOptions = params
        second.builtinOptionsType = op.builtinOptionsType
        sub.operators.append(second)
        sub.outputs = [z]
    opcode = schema.OperatorCodeT()
    opcode.builtinCode = code
    opcode.deprecatedBuiltinCode = code
    opcode.version = 3 if code in (3, 4) else 4
    model.operatorCodes = [opcode]
    model.subgraphs = [sub]
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def round_away(values, shift):
    values = np.asarray(values, np.int64)
    rounded = (np.abs(values) + (1 << (shift - 1))) >> shift
    return np.where(values < 0, -rounded, rounded)


class RuntimeTest(unittest.TestCase):
    def run_model(self, model, values):
        runtime = make_interpreter(model, preserve=True)
        report = coverage(runtime)
        self.assertFalse(report["mac_fallback_nodes"], report)
        runtime.set_tensor(runtime.get_input_details()[0]["index"], values)
        runtime.invoke()
        return runtime.get_tensor(runtime.get_output_details()[0]["index"]), runtime

    def test_requant_negative_ties_and_saturation(self):
        lib = ctypes.CDLL(str(ROOT / "build/libneural_ai_reference.so"))
        requant = lib.neural_ai_reference_requant
        requant.argtypes = [ctypes.c_int64, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        requant.restype = ctypes.c_int
        for value in range(-255, 256):
            self.assertEqual(requant(value, 0.5, 0, -128, 127), int(np.clip(round_away(value, 1), -128, 127)))
        for accumulator, result in [(4155, 11), (5434, 13), (-29088, -40)]:
            self.assertEqual(requant(accumulator, 3359085 / 2**31, 5, -128, 127), result)
        self.assertEqual(requant(2**32, 1.0, 3, -128, 127), 127)
        self.assertEqual(requant(-2**32, 1.0, 3, -128, 127), -128)
        self.assertEqual(requant(7, 1.0, 0, -2, 2), 2)
        self.assertEqual(requant(1, 1e-20, 0, -128, 127), -2**31)

    def test_fully_connected_per_channel_and_bias(self):
        x = np.array([[-8, -6, 10], [50, -50, 100]], np.int8)
        w = np.array([[1, -2, 3], [-3, 1, -1]], np.int8)
        bias = np.array([2, -3], np.int32)
        model = model_bytes(9, list(x.shape), w, [2, 2], schema.FullyConnectedOptionsT(), [0.5, 0.25], bias)
        actual, _ = self.run_model(model, x)
        acc = (x.astype(np.int64) + 7) @ w.astype(np.int64).T + bias
        expected = np.stack([round_away(acc[:, 0], 2), round_away(acc[:, 1], 3)], axis=-1)
        np.testing.assert_array_equal(actual, np.clip(expected + 3, -128, 127))

    def test_adjacent_fully_connected_and_optional_bias(self):
        x = np.array([[1, -1]], np.int8)
        w = np.array([[1, 0], [0, 1]], np.int8)
        model = model_bytes(9, [1, 2], w, [1, 2], schema.FullyConnectedOptionsT(), [0.5],
                            input_scale=0.25, input_zp=0, output_zp=0, adjacent=True)
        actual, runtime = self.run_model(model, x)
        np.testing.assert_array_equal(actual, [[1, -1]])
        self.assertEqual(coverage(runtime)["neural_ai"]["FULLY_CONNECTED"], 2)
        np.testing.assert_array_equal(runtime.get_tensor(2), [[1, -1]])

    def test_conv_padding_stride_dilation_activation(self):
        rng = np.random.default_rng(123)
        for padding, stride, dilation in [(schema.Padding.SAME, 2, 1), (schema.Padding.VALID, 1, 2)]:
            for activation in [schema.ActivationFunctionType.NONE, schema.ActivationFunctionType.RELU,
                               schema.ActivationFunctionType.RELU6, schema.ActivationFunctionType.RELU_N1_TO_1]:
                with self.subTest(padding=padding, stride=stride, dilation=dilation, activation=activation):
                    x = rng.integers(-12, 8, (2, 5, 5, 2), dtype=np.int8)
                    w = rng.integers(-3, 4, (3, 3, 3, 2), dtype=np.int8)
                    bias = np.array([2, -5, 7], np.int32)
                    size = 3 if padding == schema.Padding.SAME else 1
                    pad = 1 if padding == schema.Padding.SAME else 0
                    p = schema.Conv2DOptionsT()
                    p.padding = padding; p.strideH = stride; p.strideW = stride
                    p.dilationHFactor = dilation; p.dilationWFactor = dilation
                    model = model_bytes(3, list(x.shape), w, [2, size, size, 3], p, [0.5, 0.25, 0.125], bias, activation)
                    actual, _ = self.run_model(model, x)
                    expected = np.empty_like(actual)
                    for n, oy, ox, c in np.ndindex(expected.shape):
                        acc = int(bias[c])
                        for ky, kx, ic in np.ndindex(3, 3, 2):
                            iy, ix = oy * stride + ky * dilation - pad, ox * stride + kx * dilation - pad
                            if 0 <= iy < 5 and 0 <= ix < 5:
                                acc += (int(x[n, iy, ix, ic]) + 7) * int(w[c, ky, kx, ic])
                        low, high = {0: (-128, 127), 1: (3, 127), 3: (3, 27), 2: (-1, 7)}[activation]
                        expected[n, oy, ox, c] = np.clip(round_away(acc, c + 2) + 3, low, high)
                    np.testing.assert_array_equal(actual, expected)

    def test_depthwise_channel_multiplier(self):
        x = np.arange(-12, 20, dtype=np.int8).reshape(1, 4, 4, 2)
        w = np.arange(-8, 8, dtype=np.int8).reshape(1, 2, 2, 4)
        p = schema.DepthwiseConv2DOptionsT()
        p.padding = schema.Padding.VALID; p.strideH = 1; p.strideW = 1
        p.dilationHFactor = 1; p.dilationWFactor = 1; p.depthMultiplier = 2
        model = model_bytes(4, list(x.shape), w, [1, 3, 3, 4], p, [0.25] * 4, np.zeros(4, np.int32))
        actual, _ = self.run_model(model, x)
        expected = np.empty_like(actual)
        for n, oy, ox, c in np.ndindex(expected.shape):
            acc = sum((int(x[n, oy + ky, ox + kx, c // 2]) + 7) * int(w[0, ky, kx, c])
                      for ky, kx in np.ndindex(2, 2))
            expected[n, oy, ox, c] = np.clip(round_away(acc, 3) + 3, -128, 127)
        np.testing.assert_array_equal(actual, expected)

    def test_unsupported_scale_reports_fallback(self):
        model = model_bytes(9, [1, 1], np.array([[1]], np.int8), [1, 1],
                            schema.FullyConnectedOptionsT(), [1e-15])
        runtime = make_interpreter(model)
        self.assertEqual(coverage(runtime)["mac_fallback_nodes"], [0])


if __name__ == "__main__":
    unittest.main()
