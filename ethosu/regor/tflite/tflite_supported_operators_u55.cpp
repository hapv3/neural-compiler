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
    _maxWeightSum8Bit = 127 * (1 << 16);
    _maxWeightSum16Bit = 127 * (1 << 16);
    _maxBias = (1LL << 40) - 1;
    _checks = {
        &TfLiteSupportedOperatorsU55::ConstraintBroadcastShapes,
        &TfLiteSupportedOperatorsU55::ConstraintReverse,
        &TfLiteSupportedOperatorsU55::Constraint32bitOps,
        &TfLiteSupportedOperatorsU55::ConstraintKernelStride,
        &TfLiteSupportedOperatorsU55::ConstraintUnrolledKernelStride,
        &TfLiteSupportedOperatorsU55::ConstraintMatmul,
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
    static const char *constraint = "One input-tensor must match the shape of the output-tensor.";
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

bool TfLiteSupportedOperatorsU55::ConstraintKernelStride(const Operation *op)
{
    const auto kernel = op->Kernel();
    assert(kernel);
    const int32_t stride_w = kernel->Stride().x;
    const int32_t stride_h = kernel->Stride().y;
    if ( op->Type() == OpType::Conv2D )
    {
        // Conv2D is handled by ConstraintUnrolledKernelStride
        return true;
    }
    if ( stride_w > 3 || stride_h > 3 )
    {
        Failure(op, fmt::format("Unsupported kernel stride: {}, {}", stride_w, stride_h), "kernel stride must be in the range (1,3)");
        return false;
    }
    return true;
}

bool TfLiteSupportedOperatorsU55::ConstraintUnrolledKernelStride(const Operation *op)
{
    // Constraints for UnrollConv
    const static char *constraint =
        "Stride >3 is only supported when:\n"
        "\t * kernel dilation = 1\n"
        "\t * IFM and OFM are not sliced\n"
        "\t * padding = VALID\n";
    const auto ifmConn = op->Input(TensorUsage::IFM);
    const auto ofmConn = op->Output(TensorUsage::OFM);
    const auto kernel = op->Kernel();
    assert(ifmConn);
    assert(ofmConn);
    assert(kernel);
    if ( op->Type() != OpType::Conv2D )
    {
        return true;
    }
    const int32_t stride_w = kernel->Stride().x;
    const int32_t stride_h = kernel->Stride().y;
    if ( stride_w <= 3 && stride_h <= 3 )
    {
        // always supported
        return true;
    }
    // stride > 3 requires unrolling, check unroll conditions
    const bool hasPadding = !kernel->Padding().IsZero();
    const bool hasIfmSlice = ifmConn->slice.shape.IsValid() || ifmConn->slice.offset.IsValid();
    const bool hasOfmSlice = ofmConn->slice.shape.IsValid() || ofmConn->slice.offset.IsValid();
    const int32_t dilation_h = kernel->Dilation().y;
    const int32_t dilation_w = kernel->Dilation().x;
    const bool canUnroll = !hasPadding && !hasIfmSlice && !hasOfmSlice && (dilation_h == 1) && (dilation_w == 1);
    if ( !canUnroll )
    {
        Failure(op, fmt::format("Unsupported kernel stride: {}, {}", stride_w, stride_h), constraint);
    }
    return true;
}

bool TfLiteSupportedOperatorsU55::ConstraintMatmul(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType != OpType::BatchMatMul && opType != OpType::FullyConnected )
    {
        return true;
    }
    auto ifmConn = op->Input(TensorUsage::IFM0);
    auto ofmConn = op->Output(TensorUsage::OFM);
    assert(ifmConn);
    assert(ofmConn);
    auto ifmShape = ifmConn->shape;
    auto ofmShape = ofmConn->shape;

    bool adj_x = false;
    if ( opType == OpType::FullyConnected )
    {
        auto wConn = op->Input(TensorUsage::Weights);
        assert(wConn);
        if ( wConn->tensor->IsConstant() )
        {
            // Non-dynamic weights, not a matmul
            return true;
        }
    }
    else
    {
        const tflite::Operator *const passthrough = static_cast<const tflite::Operator *>(op->Passthrough());
        const auto options = passthrough->builtin_options_as_BatchMatMulOptions();
        if ( options )
        {
            adj_x = options->adj_x();
        }
    }

    if ( adj_x )
    {
        // NHWC-transpose ifm-shape
        ifmShape = ifmShape.Permute(0x3201);
    }
    // OFM-depth and the reduced axis (ifmShape.Depth()) is constrained to 16-bits
    const static int maxAxis = 1 << 16;
    if ( ifmShape.Depth() > maxAxis )
    {
        static const std::string constraint = fmt::format("The reduced axis must be less than or equal to {}", maxAxis);
        Failure(op, fmt::format("The reduced Axis is: {}", ifmShape.Depth()), constraint);
        return false;
    }
    if ( ofmShape.Depth() > maxAxis )
    {
        static const std::string constraint = fmt::format("The OFM depth must be less than or equal to {}", maxAxis);
        Failure(op, fmt::format("OFM channel: {}", ofmShape.Depth()), constraint);
        return false;
    }
    if ( ifmConn->tensor->Type() != DataType::Int8 )
    {
        Failure(op, fmt::format("IFM has datatype: {}", DataTypeToString(ifmConn->tensor->Type())), "IFM must be Int8");
        return false;
    }
    return true;
}

}  // namespace regor
