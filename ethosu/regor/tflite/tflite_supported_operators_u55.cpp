//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the License); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "tflite_supported_operators_u55.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "compiler/op_type.hpp"

namespace regor
{

TfLiteSupportedOperatorsU55::TfLiteSupportedOperatorsU55(IArchitectureConstraints *constraints) :
        TfLiteSupportedOperators(constraints)
{
    _supportedOpTypes = {
        // clang-format off
        OpType::Add,
        OpType::AvgPool,
        OpType::BatchMatMul,
        OpType::Concat,
        OpType::Conv2D,
        OpType::DepthwiseConv2D,
        OpType::FullyConnected,
        OpType::Sigmoid,
        OpType::MaxPool,
        OpType::Mul,
        OpType::Relu,
        OpType::ReluN1To1,
        OpType::Relu6,
        OpType::Reshape,
        OpType::Softmax,
        OpType::Tanh,
        OpType::Pad,
        OpType::BatchToSpaceND,
        OpType::SpaceToBatchND,
        OpType::Transpose,
        OpType::Mean,
        OpType::Sub,
        OpType::Squeeze,
        OpType::StridedSlice,
        OpType::Exp,
        OpType::Split,
        OpType::Prelu,
        OpType::Maximum,
        OpType::ArgMax,
        OpType::Minimum,
        OpType::PadV2,
        OpType::Slice,
        OpType::TransposeConv2D,
        OpType::Tile,
        OpType::ExpandDims,
        OpType::ReduceSum,
        OpType::Rsqrt,
        OpType::Pack,
        OpType::Unpack,
        OpType::LeakyRelu,
        OpType::SquaredDifference,
        OpType::MirrorPad,
        OpType::Abs,
        OpType::SplitV,
        OpType::Quantize,
        OpType::HardSwish,
        // clang-format on
    };
    _supportedDataTypes = {
        // clang-format off
        DataType::UInt8,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32
        // clang-format on
    };
    _checks = {
        &TfLiteSupportedOperatorsU55::ConstraintBroadcastShapes,
        &TfLiteSupportedOperatorsU55::ConstraintReverse,
        &TfLiteSupportedOperatorsU55::Constraint32bitOps,
    };
}

bool TfLiteSupportedOperatorsU55::Check(const Operation *op)
{
    for ( auto &check : _genericChecks )
    {
        if ( !((this->*check)(op)) ) return false;
    }
    for ( auto &check : _checks )
    {
        if ( !((this->*check)(op)) ) return false;
    }
    return true;
}

bool TfLiteSupportedOperatorsU55::ConstraintBroadcastShapes(const Operation *op)
{
    const char *constraint = "One input-tensor must match the shape of the output-tensor.";
    if ( !IsElementwise(op->Type()) )
    {
        // only applied to elementwise ops
        return true;
    }
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto ifm2Conn = op->Input(TensorUsage::IFM1);
    auto ofmConn = op->Output(TensorUsage::OFM);
    assert(ifmConn);
    assert(ofmConn);
    Shape ifmShape = ifmConn->shape;
    Shape ofmShape = ofmConn->shape;
    Shape ifm2Shape = ifm2Conn ? ifm2Conn->shape : Shape();
    if ( ifmShape != ofmShape && (ifm2Shape == false || ifm2Shape != ofmShape) )
    {
        Failure(op, "Operation has invalid broadcast.", constraint);
        return false;
    }
    return true;
}

bool TfLiteSupportedOperatorsU55::ConstraintReverse(const Operation *op)
{
    if ( op->Type() != OpType::Reverse && op->Type() != OpType::ReverseV2 )
    {
        return true;
    }
    auto params = op->Input(TensorUsage::Params);
    assert(params);
    if ( !params->tensor->IsConstant() )
    {
        return false;
    }
    auto ofmConn = op->Output(TensorUsage::OFM);
    assert(ofmConn);
    auto view = params->tensor->View();
    Shape axes = Shape(view.Buffer()->Data<int32_t>(), view.ViewShape().Elements());
    auto mask = ToReverseMask(axes, ofmConn->shape.Size());
    if ( mask != ReverseType::None )
    {
        Failure(op, fmt::format("Reverse is not supported"), "");
        return false;
    }
    return true;
}

bool TfLiteSupportedOperatorsU55::Constraint32bitOps(const Operation *op)
{
    static const std::unordered_set<OpType> supported = {
        OpType::ReduceSum,
        OpType::Shape,
        OpType::ArgMax,
        OpType::Transpose,
        OpType::MirrorPad,
        OpType::Add,
        OpType::Mul,
        OpType::Sub,
        OpType::BatchMatMul,
        OpType::FullyConnected,
    };

    OpType opType = op->Type();

    if ( supported.count(opType) > 0 )
    {
        return true;
    }

    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &[usage, conn] : list->pairs() )
        {
            auto type = conn.tensor->Type();
            if ( type == DataType::Int32 && (IsIFM(usage) || IsOFM(usage)) )
            {
                Failure(op, "Operation does not support Int32 inputs/outputs", "");
                return false;
            }
        }
    }
    return true;
}
}  // namespace regor
