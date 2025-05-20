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

#include "tflite_supported_operators_u85.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "compiler/op_type.hpp"

#include <unordered_set>

namespace regor
{

TfLiteSupportedOperatorsU85::TfLiteSupportedOperatorsU85(IArchitectureConstraints *constraints) :
        TfLiteSupportedOperators(constraints)
{
    _supportedOpTypes = {
        // clang-format off
        OpType::Add,
        OpType::AvgPool,
        OpType::Concat,
        OpType::Conv2D,
        OpType::DepthwiseConv2D,
        OpType::FullyConnected,
        OpType::Sigmoid,
        OpType::MaxPool,
        OpType::Mul,
        OpType::Relu,
        OpType::Relu0To1,
        OpType::Relu6,
        OpType::ReluN1To1,
        OpType::Reshape,
        OpType::ResizeBilinear,
        OpType::Softmax,
        OpType::Tanh,
        OpType::Pad,
        OpType::GatherV2,
        OpType::Transpose,
        OpType::Mean,
        OpType::Sub,
        OpType::Div,
        OpType::Squeeze,
        OpType::StridedSlice,
        OpType::Exp,
        OpType::Split,
        OpType::Cast,
        OpType::Prelu,
        OpType::Maximum,
        OpType::ArgMax,
        OpType::Minimum,
        OpType::PadV2,
        OpType::Select,
        OpType::Greater,
        OpType::GreaterEqual,
        OpType::LessEqual,
        OpType::Slice,
        OpType::TransposeConv2D,
        OpType::Tile,
        OpType::ExpandDims,
        OpType::Equal,
        OpType::NotEqual,
        OpType::ReduceSum,
        OpType::Rsqrt,
        OpType::ReduceMax,
        OpType::Pack,
        OpType::Unpack,
        OpType::ReduceMin,
        OpType::ReduceAny,
        OpType::LogicalOr,
        OpType::LogicalAnd,
        OpType::LogicalNot,
        OpType::ResizeNearestNeighbor,
        OpType::LeakyRelu,
        OpType::SquaredDifference,
        OpType::MirrorPad,
        OpType::Abs,
        OpType::SplitV,
        OpType::ReverseV2,
        OpType::Quantize,
        OpType::HardSwish,
        OpType::SelectV2,
        OpType::ScatterNd,
        OpType::SelectV2,
        OpType::BatchMatMul,
        OpType::ReduceAll,
        OpType::MemoryCopy,
        OpType::Log,
        OpType::UnidirectionalSequenceLstm,
        // clang-format on
    };
    _supportedDataTypes = {
        // clang-format off
        DataType::UInt8,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::Bool,
        DataType::Bool8
        // clang-format on
    };
    _maxWeightSum8Bit = 127 * (1 << 16);
    _maxWeightSum16Bit = 127 * (1 << 24);
    _maxBias = (1LL << 48) - 1;
    _checks = {
        &TfLiteSupportedOperatorsU85::ConstraintResizeCommon,
        &TfLiteSupportedOperatorsU85::ConstraintResizeBilinear,
        &TfLiteSupportedOperatorsU85::ConstraintGather,
        &TfLiteSupportedOperatorsU85::ConstraintScatter,
    };
}

bool TfLiteSupportedOperatorsU85::Check(const Operation *op)
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

bool TfLiteSupportedOperatorsU85::ConstraintResizeCommon(const Operation *op)
{
    static const char *constraint =
        "ALIGN_CORNERS and HALF_PIXEL_CENTERS are mutually exclusive.\n"
        "if ALIGN_CORNERS:\n"
        "\tScale-factor can be maximum 2048\n"
        "else if HALF_PIXEL_CENTERS:\n"
        "\tScale-factor can be maximum 1024\n";
    OpType opType = op->Type();
    if ( opType != OpType::ResizeBilinear && opType != OpType::ResizeNearestNeighbor )
    {
        return true;
    }
    auto ifmConn = op->Input(TensorUsage::IFM0);
    auto ofmConn = op->Output(TensorUsage::OFM);
    assert(ifmConn);
    assert(ofmConn);
    int width_n = ofmConn->shape.Width();
    int width_d = ifmConn->shape.Width();
    int height_n = ofmConn->shape.Height();
    int height_d = ifmConn->shape.Height();
    bool halfPixelCenters = false;
    bool alignCorners = false;
    const tflite::Operator *passthrough = static_cast<const tflite::Operator *>(op->Passthrough());
    assert(passthrough);

    if ( opType == OpType::ResizeBilinear )
    {
        const auto *opt = passthrough->builtin_options_as_ResizeBilinearOptions();
        assert(opt);
        alignCorners = opt->align_corners();
        halfPixelCenters = opt->half_pixel_centers();
    }
    else
    {
        assert(opType == OpType::ResizeNearestNeighbor);
        const auto *opt = passthrough->builtin_options_as_ResizeNearestNeighborOptions();
        assert(opt);
        alignCorners = opt->align_corners();
        // Use half-pixel-centers if align-corners is false.
        // This aligns with reference kernels
        halfPixelCenters = !alignCorners || opt->half_pixel_centers();
    }

    if ( alignCorners && halfPixelCenters )
    {
        Failure(op, "Operation with both align_corners=true and half_pixel_centers=true (these are mutually exclusive)", constraint);
        return false;
    }

    if ( alignCorners )
    {
        if ( width_d > 1 )
        {
            width_n -= 1;
            width_d -= 1;
        }
        if ( height_d > 1 )
        {
            height_n -= 1;
            height_d -= 1;
        }
    }

    auto ConstrainScaleFactor = [&](int num, int den, const char *axis) -> bool
    {
        if ( num == 0 || den == 0 )
        {
            Failure(op, fmt::format("unsupported {} scale-factor ({}/{})", axis, num, den), constraint);
            return false;
        }
        int scaleFactor = num / den;
        if ( halfPixelCenters && scaleFactor > 1024 )
        {
            Failure(op, fmt::format("halfPixelCenters {} scaleFactor exceeds 1024: {}/{}", axis, num, den), constraint);
            return false;
        }
        else if ( scaleFactor > 2048 )
        {
            Failure(op, fmt::format("{} scaleFactor exceeds 2048: {}/{}", axis, num, den), constraint);
            return false;
        }
        return true;
    };
    if ( !(ConstrainScaleFactor(width_n, width_d, "width") && ConstrainScaleFactor(height_n, height_d, "height")) )
    {
        return false;
    }
    return true;
}

bool TfLiteSupportedOperatorsU85::ConstraintResizeBilinear(const Operation *op)
{
    static const char *constraint =
        "if IFM HxW > 1x1\n"
        "\tand ALIGN_CORNERS:\n"
        "\t\tOFM W-1 and H-1 must be a power-of-two integer-multiple of IFM W-1 and H-1\n"
        "\telse:\n"
        "\t\tOFM W and H must be a power-of-two integer-multiple of IFM W and H\n";
    OpType opType = op->Type();
    if ( opType != OpType::ResizeBilinear )
    {
        return true;
    }
    auto ifmConn = op->Input(TensorUsage::IFM0);
    auto ofmConn = op->Output(TensorUsage::OFM);
    assert(ifmConn);
    assert(ofmConn);
    int width_n = ofmConn->shape.Width();
    int width_d = ifmConn->shape.Width();
    int height_n = ofmConn->shape.Height();
    int height_d = ifmConn->shape.Height();
    bool alignCorners = false;
    const tflite::Operator *passthrough = static_cast<const tflite::Operator *>(op->Passthrough());
    assert(passthrough);

    if ( width_d == 1 && height_d == 1 )
    {
        return true;
    }

    const auto *opt = passthrough->builtin_options_as_ResizeBilinearOptions();
    assert(opt);
    alignCorners = opt->align_corners();

    if ( alignCorners )
    {
        if ( width_d > 1 )
        {
            width_n -= 1;
            width_d -= 1;
        }
        if ( height_d > 1 )
        {
            height_n -= 1;
            height_d -= 1;
        }
    }

    auto ConstrainScaleFactor = [&](int num, int den, const char *axis) -> bool
    {
        assert(num > 0 && den > 0);
        int scaleFactor = num / den;
        if ( num % den != 0 )
        {
            Failure(op, fmt::format("{} scale-factor must be integer. scale-factor: ({}/{})", axis, num, den), constraint);
            return false;
        }
        if ( !IsPowerOfTwo(scaleFactor) )
        {
            Failure(op, fmt::format("{} scale-factor must be power of two. scale-factor: ({}/{})", axis, num, den), constraint);
            return false;
        }
        return true;
    };
    if ( !(ConstrainScaleFactor(width_n, width_d, "width") && ConstrainScaleFactor(height_n, height_d, "height")) )
    {
        return false;
    }
    return true;
}

bool TfLiteSupportedOperatorsU85::ConstraintGather(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType != OpType::GatherV2 )
    {
        return true;
    }
    const tflite::Operator *const passthrough = static_cast<const tflite::Operator *>(op->Passthrough());
    const auto options = passthrough->builtin_options_as_GatherOptions();
    auto *params = op->Input(TensorUsage::IFM0);
    assert(params);
    int paramsRank = params->shape.Size();
    int batchDimsParam = 0;
    int axisParam = 0;
    if ( options )
    {
        axisParam = options->axis();
        if ( axisParam < 0 ) axisParam = paramsRank - (-axisParam);
        batchDimsParam = options->batch_dims();
    }

    if ( axisParam != batchDimsParam )
    {
        Failure(op, fmt::format("axis: {} != batch_dims: {}", axisParam, batchDimsParam), "axis must be equal to batch_dims");
        return false;
    }
    return true;
}

bool TfLiteSupportedOperatorsU85::ConstraintScatter(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType != OpType::ScatterNd )
    {
        return true;
    }
    auto *idxConn = op->Input(TensorUsage::IFM0);
    auto *shapeConn = op->Input(TensorUsage::Params);
    assert(idxConn);
    assert(shapeConn);
    // index tensor must have C == 1
    if ( idxConn->shape[-1] != 1 )
    {
        Failure(op, fmt::format("index shape: {}", idxConn->shape.ToString()), "Channel must be 1 for ScatterNd index tensor");
        return false;
    }
    // index tensor must be constant
    if ( !idxConn->tensor->IsConstant() )
    {
        Failure(op, "non-constant index tensor", "index tensor must be constant");
        return false;
    }
    // shape tensor must be constant
    if ( !shapeConn->tensor->IsConstant() )
    {
        Failure(op, "non-constant shape tensor", "shape tensor must be constant");
        return false;
    }
    // Can not support duplicates in the index tensor
    const auto idxs = idxConn->tensor->View().Values<int32_t>();
    const std::unordered_set<int32_t> uniqueIdxs(idxs.begin(), idxs.end());
    if ( idxConn->tensor->View().Elements() != int(uniqueIdxs.size()) )
    {
        Failure(op, "index tensor contains duplicates", "index tensor elements must be unique");
        return false;
    }
    return true;
}
}  // namespace regor
