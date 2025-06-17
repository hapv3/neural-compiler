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

#include <set>
#include <unordered_set>

namespace regor
{

static const std::set<OpType> s_supportedOpTypes = {
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
    OpType::BatchMatMul,
    OpType::ReduceAll,
    OpType::MemoryCopy,
    OpType::Log,
    OpType::UnidirectionalSequenceLstm,
};

static const std::set<DataType> s_supportedDataTypes = {
    DataType::UInt8,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::Bool,
};

// zeroPoints are ignored for the following OpTypes to align with reference
static const std::unordered_set<OpType> s_ignoreZeroPoints = {
    OpType::AvgPool,
    OpType::ResizeBilinear,
    OpType::ResizeNearestNeighbor,
    OpType::Div,
    OpType::UnidirectionalSequenceLstm,
};

namespace
{
// Align-corners and half-pixel-centers are mutually exclusive
bool ResizeACHPC(const Operation *op)
{
    const tflite::Operator *passthrough = static_cast<const tflite::Operator *>(op->Passthrough());
    bool alignCorners = false;
    bool halfPixelCenters = false;
    if ( op->Type() == OpType::ResizeBilinear )
    {
        const auto *opt = passthrough->builtin_options_as_ResizeBilinearOptions();
        assert(opt);
        alignCorners = opt->align_corners();
        halfPixelCenters = opt->half_pixel_centers();
    }
    else
    {
        assert(op->Type() == OpType::ResizeNearestNeighbor);
        const auto *opt = passthrough->builtin_options_as_ResizeNearestNeighborOptions();
        assert(opt);
        alignCorners = opt->align_corners();
        // Use half-pixel-centers if align-corners is false.
        // This aligns with reference kernels
        halfPixelCenters = !alignCorners || opt->half_pixel_centers();
    }
    if ( alignCorners && halfPixelCenters )
    {
        Failure(op, "both align_corners=true and half_pixel_centers=true");
        return false;
    }
    return true;
}

// constraints on scale-factor for resize operations
bool ResizeScaleFactor(const Operation *op)
{
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
    if ( op->Type() == OpType::ResizeBilinear )
    {
        const auto *opt = passthrough->builtin_options_as_ResizeBilinearOptions();
        assert(opt);
        alignCorners = opt->align_corners();
        halfPixelCenters = opt->half_pixel_centers();
    }
    else
    {
        assert(op->Type() == OpType::ResizeNearestNeighbor);
        const auto *opt = passthrough->builtin_options_as_ResizeNearestNeighborOptions();
        assert(opt);
        alignCorners = opt->align_corners();
        // Use half-pixel-centers if align-corners is false.
        // This aligns with reference kernels
        halfPixelCenters = !alignCorners || opt->half_pixel_centers();
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
            Failure(op, fmt::format("unsupported {} scale-factor ({}/{})", axis, num, den));
            return false;
        }
        int scaleFactor = num / den;
        if ( halfPixelCenters && scaleFactor > 1024 )
        {
            Failure(op, fmt::format("halfPixelCenters {} scaleFactor exceeds 1024: {}/{}", axis, num, den));
            return false;
        }
        else if ( scaleFactor > 2048 )
        {
            Failure(op, fmt::format("{} scaleFactor exceeds 2048: {}/{}", axis, num, den));
            return false;
        }
        return true;
    };
    if ( !ConstrainScaleFactor(width_n, width_d, "width") || !ConstrainScaleFactor(height_n, height_d, "height") )
    {
        return false;
    }
    return true;
}

// ResizeBilinear must have a scale-factor that is a power of two
bool ResizeBilinearPowerOfTwo(const Operation *op)
{
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
        if ( (num % den != 0) || !IsPowerOfTwo(int(num / den)) )
        {
            Failure(op, fmt::format("{} scale-factor must be an integer power of two. scale-factor: ({}/{})", axis, num, den));
            return false;
        }
        return true;
    };
    if ( !ConstrainScaleFactor(width_n, width_d, "width") || !ConstrainScaleFactor(height_n, height_d, "height") )
    {
        return false;
    }
    return true;
}

// Scatter shape-tensors must be constant
bool ScatterShapeConst(const Operation *op)
{
    auto *shapeConn = op->Input(TensorUsage::Params);
    assert(shapeConn);
    if ( !shapeConn->tensor->IsConstant() )
    {
        Failure(op, "non-constant shape tensor");
        return false;
    }
    return true;
}

// Scatter index-tensors must be constant
bool ScatterIndexConst(const Operation *op)
{
    auto *idxConn = op->Input(TensorUsage::IFM0);
    assert(idxConn);
    if ( !idxConn->tensor->IsConstant() )
    {
        Failure(op, "non-constant index tensor");
        return false;
    }
    return true;
}

// Scatter index-tensors must have unit channel
bool ScatterIndexUnitChannel(const Operation *op)
{
    auto *idxConn = op->Input(TensorUsage::IFM0);
    assert(idxConn);
    if ( idxConn->shape[-1] != 1 )
    {
        Failure(op, fmt::format("index shape: {}", idxConn->shape.ToString()));
        return false;
    }
    return true;
}

// Scatter index tensors must not contain duplicate elements
bool ScatterIndexUniqueElements(const Operation *op)
{
    auto *idxConn = op->Input(TensorUsage::IFM0);
    assert(idxConn);
    // Can not support duplicates in the index tensor
    const auto idxs = idxConn->tensor->View().Values<int32_t>();
    const std::unordered_set<int32_t> uniqueIdxs(idxs.begin(), idxs.end());
    if ( idxConn->tensor->View().Elements() != int(uniqueIdxs.size()) )
    {
        Failure(op, "index tensor contains duplicates");
        return false;
    }
    return true;
}

// Gather axis must be equal to batch_dims
bool GatherAxis(const Operation *op)
{
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
        Failure(op, fmt::format("axis: {} != batch_dims: {}", axisParam, batchDimsParam));
        return false;
    }
    return true;
}

// Zero-point constraints for IFM
bool IFMZeroPoints(const Operation *op)
{
    OpType opType = op->Type();
    for ( const auto &[usage, conn] : op->Inputs().pairs() )
    {
        DataType dType = conn.tensor->Type();
        for ( auto zp : conn.quantization.zeroPoints )
        {
            if ( IsIFM(usage) )
            {
                switch ( dType )
                {
                    case DataType::Int8:
                        if ( zp < -128 || zp > 127 )
                        {
                            Failure(op, "IFM ZeroPoint out of range");
                            return false;
                        }
                        break;
                    case DataType::UInt8:
                        if ( zp < 0 || zp > 255 )
                        {
                            Failure(op, "IFM ZeroPoint out of range");
                            return false;
                        }
                        break;
                    case DataType::UInt16:
                        if ( zp != 0 && zp != 32768 )
                        {
                            Failure(op, "IFM ZeroPoint out of range");
                            return false;
                        }
                        break;
                    default:
                        if ( zp != 0 )
                        {
                            Failure(op, "IFM ZeroPoint out of range");
                            return false;
                        }
                        break;
                }
            }
        }
    }
    return true;
}

// zero-point constraints for OFM
bool OFMZeroPoints(const Operation *op)
{
    OpType opType = op->Type();
    for ( const auto &[usage, conn] : op->Outputs().pairs() )
    {
        DataType dType = conn.tensor->Type();
        for ( auto zp : conn.quantization.zeroPoints )
        {
            if ( IsOFM(usage) )
            {
                if ( IsSignedInteger(dType) )
                {
                    if ( zp < -128 || zp > 127 )
                    {
                        Failure(op, "OFM zeroPoint out of range");
                        return false;
                    }
                }
                else
                {
                    if ( zp < 0 || (zp > 255 && zp != 32768) )
                    {
                        Failure(op, "OFM zeroPoint out of range");
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

}  // namespace

TfLiteSupportedOperatorsU85::TfLiteSupportedOperatorsU85() :
        TfLiteSupportedOperators(127 * (1 << 16),  // maxWeightSum8Bit
            127 * (1 << 24),                       // maxWeightSum16bit
            (1LL << 48) - 1,                       // maxBias
            s_supportedDataTypes, s_supportedOpTypes)
{
    resizeACHPC = {&ResizeACHPC, "align_corners and half_pixel_centers are mutually exclusive."};
    resizeScaleFactor = {&ResizeScaleFactor,
        "Scale-factor:\n"
        "  - IF align_corners:\n"
        "    - Scale-factor can be maximum 2048\n"
        "  - ELSE:\n"
        "    - Scale-factor can be maximum 1024."};
    resizeBilinearPowerOfTwo = {&ResizeBilinearPowerOfTwo, "Scale-factor must be an integer power of two."};
    scatterShapeConst = {&ScatterShapeConst, "Shape tensor must be constant"};
    scatterIndexConst = {&ScatterIndexConst, "Index tensor must be constant"};
    scatterIndexUnitChannel = {&ScatterIndexUnitChannel, "Index tensor must have channel=1"};
    scatterIndexUniqueElements = {&ScatterIndexUniqueElements, "Index tensor elements must be unique"};
    gatherAxis = {&GatherAxis, "axis must be equal to batch_dims"};
    ifmZeroPoints = {&IFMZeroPoints,
        "IFM Zero-points must be in the range:\n"
        "  - Int8: (-128 <= zp <= 127).\n"
        "  - UInt8: (0 <= zp <= 255).\n"
        "  - UInt16: zp must be 0 or 32768.\n"
        "  - Other: zp must be 0."};
    ofmZeroPoints = {&OFMZeroPoints,
        "OFM Zero-points must be in the range:\n"
        "  - Signed-integer: (-128 <= zp <= 127)\n"
        "  - Unsigned-integer: (0 <= zp <= 255) OR zp == 32768."};

    for ( OpType type : s_supportedOpTypes )
    {
        if ( s_ignoreZeroPoints.count(type) == 0 )
        {
            opConstraints[type].push_back(&ifmZeroPoints);
            opConstraints[type].push_back(&ofmZeroPoints);
        }
    }

    opConstraints[OpType::ScatterNd].push_back(&scatterShapeConst);
    opConstraints[OpType::ScatterNd].push_back(&scatterIndexConst);
    opConstraints[OpType::ScatterNd].push_back(&scatterIndexUnitChannel);
    opConstraints[OpType::ScatterNd].push_back(&scatterIndexUniqueElements);
    opConstraints[OpType::GatherV2].push_back(&gatherAxis);
    opConstraints[OpType::ResizeNearestNeighbor].push_back(&resizeACHPC);
    opConstraints[OpType::ResizeNearestNeighbor].push_back(&resizeScaleFactor);
    opConstraints[OpType::ResizeBilinear].push_back(&resizeACHPC);
    opConstraints[OpType::ResizeBilinear].push_back(&resizeScaleFactor);
    opConstraints[OpType::ResizeBilinear].push_back(&resizeBilinearPowerOfTwo);
}

}  // namespace regor
