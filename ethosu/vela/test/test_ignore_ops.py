# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import flatbuffers
import pytest

from ethosu.vela import vela
from ethosu.vela.tflite import AddOptions
from ethosu.vela.tflite import Buffer
from ethosu.vela.tflite import Model
from ethosu.vela.tflite import MulOptions
from ethosu.vela.tflite import Operator
from ethosu.vela.tflite import OperatorCode
from ethosu.vela.tflite import QuantizationParameters
from ethosu.vela.tflite import SubGraph
from ethosu.vela.tflite import Tensor
from ethosu.vela.tflite.BuiltinOperator import BuiltinOperator
from ethosu.vela.tflite.BuiltinOptions import BuiltinOptions
from ethosu.vela.tflite.Model import Model as TfliteModel
from ethosu.vela.tflite.TensorType import TensorType


def _vector_i32(builder, values):
    builder.StartVector(4, len(values), 4)
    for value in reversed(values):
        builder.PrependInt32(value)
    return builder.EndVector()


def _vector_i64(builder, values):
    builder.StartVector(8, len(values), 8)
    for value in reversed(values):
        builder.PrependInt64(value)
    return builder.EndVector()


def _vector_f32(builder, values):
    builder.StartVector(4, len(values), 4)
    for value in reversed(values):
        builder.PrependFloat32(value)
    return builder.EndVector()


def _vector_uoffset(builder, values, start):
    start(builder, len(values))
    for value in reversed(values):
        builder.PrependUOffsetTRelative(value)
    return builder.EndVector()


def _create_quantization(builder):
    scale = _vector_f32(builder, [1.0])
    zero_point = _vector_i64(builder, [0])

    QuantizationParameters.Start(builder)
    QuantizationParameters.AddScale(builder, scale)
    QuantizationParameters.AddZeroPoint(builder, zero_point)
    return QuantizationParameters.End(builder)


def _create_tensor(builder, name, buffer_index):
    name_offset = builder.CreateString(name)
    shape = _vector_i32(builder, [1, 1, 1, 1])
    quantization = _create_quantization(builder)

    Tensor.Start(builder)
    Tensor.AddShape(builder, shape)
    Tensor.AddType(builder, TensorType.INT8)
    Tensor.AddBuffer(builder, buffer_index)
    Tensor.AddName(builder, name_offset)
    Tensor.AddQuantization(builder, quantization)
    return Tensor.End(builder)


def _create_mul_add_model():
    builder = flatbuffers.Builder(1024)

    Buffer.Start(builder)
    buffers = [Buffer.End(builder)]
    tensors = [
        _create_tensor(builder, "ifm0", 0),
        _create_tensor(builder, "ifm1", 0),
        _create_tensor(builder, "ifm2", 0),
        _create_tensor(builder, "mul_ofm", 0),
        _create_tensor(builder, "ofm", 0),
    ]

    MulOptions.Start(builder)
    mul_options = MulOptions.End(builder)
    mul_inputs = _vector_i32(builder, [0, 1])
    mul_outputs = _vector_i32(builder, [3])

    Operator.Start(builder)
    Operator.AddOpcodeIndex(builder, 0)
    Operator.AddInputs(builder, mul_inputs)
    Operator.AddOutputs(builder, mul_outputs)
    Operator.AddBuiltinOptionsType(builder, BuiltinOptions.MulOptions)
    Operator.AddBuiltinOptions(builder, mul_options)
    mul_op = Operator.End(builder)

    AddOptions.Start(builder)
    add_options = AddOptions.End(builder)
    add_inputs = _vector_i32(builder, [3, 2])
    add_outputs = _vector_i32(builder, [4])

    Operator.Start(builder)
    Operator.AddOpcodeIndex(builder, 1)
    Operator.AddInputs(builder, add_inputs)
    Operator.AddOutputs(builder, add_outputs)
    Operator.AddBuiltinOptionsType(builder, BuiltinOptions.AddOptions)
    Operator.AddBuiltinOptions(builder, add_options)
    add_op = Operator.End(builder)

    OperatorCode.Start(builder)
    OperatorCode.AddDeprecatedBuiltinCode(builder, BuiltinOperator.MUL)
    OperatorCode.AddVersion(builder, 1)
    OperatorCode.AddBuiltinCode(builder, BuiltinOperator.MUL)
    mul_op_code = OperatorCode.End(builder)

    OperatorCode.Start(builder)
    OperatorCode.AddDeprecatedBuiltinCode(builder, BuiltinOperator.ADD)
    OperatorCode.AddVersion(builder, 1)
    OperatorCode.AddBuiltinCode(builder, BuiltinOperator.ADD)
    add_op_code = OperatorCode.End(builder)

    tensors_vector = _vector_uoffset(builder, tensors, SubGraph.StartTensorsVector)
    graph_inputs = _vector_i32(builder, [0, 1, 2])
    graph_outputs = _vector_i32(builder, [4])
    operators_vector = _vector_uoffset(builder, [mul_op, add_op], SubGraph.StartOperatorsVector)

    SubGraph.Start(builder)
    SubGraph.AddTensors(builder, tensors_vector)
    SubGraph.AddInputs(builder, graph_inputs)
    SubGraph.AddOutputs(builder, graph_outputs)
    SubGraph.AddOperators(builder, operators_vector)
    subgraph = SubGraph.End(builder)

    op_codes_vector = _vector_uoffset(builder, [mul_op_code, add_op_code], Model.StartOperatorCodesVector)
    subgraphs_vector = _vector_uoffset(builder, [subgraph], Model.StartSubgraphsVector)
    buffers_vector = _vector_uoffset(builder, buffers, Model.StartBuffersVector)

    Model.Start(builder)
    Model.AddVersion(builder, 3)
    Model.AddOperatorCodes(builder, op_codes_vector)
    Model.AddSubgraphs(builder, subgraphs_vector)
    Model.AddBuffers(builder, buffers_vector)
    model = Model.End(builder)

    builder.Finish(model, file_identifier=b"TFL3")
    return bytes(builder.Output())


def _operator_builtin_codes(path):
    model = TfliteModel.GetRootAs(path.read_bytes(), 0)
    codes = []
    for graph_index in range(model.SubgraphsLength()):
        graph = model.Subgraphs(graph_index)
        for op_index in range(graph.OperatorsLength()):
            op = graph.Operators(op_index)
            codes.append(model.OperatorCodes(op.OpcodeIndex()).BuiltinCode())
    return codes


def _vela_args(model_path, extra_args=None):
    return (
        [
            "--config",
            "Arm/vela.ini",
            "--accelerator-config",
            "ethos-u55-128",
            "--system-config",
            "Ethos_U55_High_End_Embedded",
            "--memory-mode",
            "Shared_Sram",
        ]
        + (extra_args or [])
        + [str(model_path)]
    )


def test_ignore_ops_cli_keeps_requested_operator_in_output(tmp_path):
    model_path = tmp_path / "mul_add.tflite"
    baseline_dir = tmp_path / "baseline"
    ignore_dir = tmp_path / "ignore"
    model_path.write_bytes(_create_mul_add_model())

    assert vela.main(_vela_args(model_path, ["--output-dir", str(baseline_dir)])) == 0
    baseline_codes = _operator_builtin_codes(baseline_dir / "mul_add_vela.tflite")
    assert BuiltinOperator.CUSTOM in baseline_codes
    assert BuiltinOperator.ADD not in baseline_codes

    assert vela.main(_vela_args(model_path, ["--output-dir", str(ignore_dir), "--ignore-ops", "add,arg_max"])) == 0
    ignore_codes = _operator_builtin_codes(ignore_dir / "mul_add_vela.tflite")
    assert BuiltinOperator.CUSTOM in ignore_codes
    assert BuiltinOperator.ADD in ignore_codes


@pytest.mark.parametrize("ignore_ops", [" ", "ADD, RELU", "ADD;RELU", "A" * (vela.MAX_IGNORE_OPS_LENGTH + 1)])
def test_ignore_ops_cli_rejects_bad_input(tmp_path, ignore_ops):
    model_path = tmp_path / "mul_add.tflite"
    model_path.write_bytes(_create_mul_add_model())

    with pytest.raises(SystemExit):
        vela.main(_vela_args(model_path, ["--ignore-ops", ignore_ops]))
