//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "compiler/operation_util.hpp"
#include "compiler/shape_util.hpp"

#include <set>
#include <unordered_set>

namespace regor
{

static const std::set<OpType> s_supportedOpTypes = {
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
    OpType::Relu0To1,
    OpType::Relu6,
    OpType::ReluN1To1,
    OpType::Reshape,
    OpType::ReverseV2,
    OpType::Softmax,
    OpType::Tanh,
    OpType::Pad,
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
    OpType::ReduceMax,
    OpType::ReduceMin,
    OpType::ResizeBilinear,
    OpType::ResizeNearestNeighbor,
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
    OpType::MemoryCopy,
    OpType::Log,
    OpType::UnidirectionalSequenceLstm,
    OpType::Neg,
};

// generic dataType support
static const std::set<DataType> s_supportedDataTypes = {
    DataType::UInt8,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
};

static const std::unordered_set<OpType> s_supports32Bit = {
    OpType::ReduceSum,
    OpType::ArgMax,
    OpType::Transpose,
    OpType::MirrorPad,
    OpType::Add,
    OpType::Mul,
    OpType::Sub,
    OpType::BatchMatMul,
    OpType::FullyConnected,
    OpType::Reshape,
    OpType::QuantizedReshape,
    OpType::Squeeze,
    OpType::ExpandDims,
    OpType::Identity,
    OpType::MemoryCopy,
    OpType::Neg,
};

static const std::unordered_set<OpType> s_supports64Bit = {
    OpType::ArgMax,
    OpType::MemoryCopy,
};

// zeroPoints are ignored for the following OpTypes to align with reference
static const std::unordered_set<OpType> s_ignoreZeroPoints = {
    OpType::AvgPool,
    OpType::ResizeBilinear,
    OpType::ResizeNearestNeighbor,
    OpType::UnidirectionalSequenceLstm,
};

namespace
{

// Axis attributes are not supported
bool ReverseMask(const Operation *op)
{
    assert(op->Type() == OpType::ReverseV2);
    auto params = op->Input(TensorUsage::Params);
    assert(params);
    auto ofmConn = op->Output(TensorUsage::OFM);
    assert(ofmConn);
    auto view = params->tensor->View();
    Shape axes = Shape(view.Buffer()->Data<int32_t>(), view.ViewShape().Elements());
    auto mask = ToReverseMask(axes, ofmConn->shape.Size());
    if ( mask != ReverseType::None )
    {
        Failure(op, fmt::format("Unsupported reverse mask."));
        return false;
    }
    return true;
}

// Kernel stride must be in the range (1,3)
bool KernelStride(const Operation *op)
{
    const auto kernel = op->Kernel();
    assert(kernel);
    const int32_t stride_w = kernel->Stride().x;
    const int32_t stride_h = kernel->Stride().y;
    if ( stride_w > 3 || stride_h > 3 )
    {
        Failure(op, fmt::format("Unsupported kernel stride: {}, {}", stride_w, stride_h));
        return false;
    }
    return true;
}

// Stride > 3 is only supported IF:
//    * dilation == 1
//    * padding == VALID
bool UnrolledKernelStride(const Operation *op)
{
    // Constraints for UnrollConv
    const auto ifmConn = op->Input(TensorUsage::IFM);
    const auto ofmConn = op->Output(TensorUsage::OFM);
    const auto kernel = op->Kernel();
    assert(ifmConn);
    assert(ofmConn);
    assert(kernel);
    const int32_t stride_w = kernel->Stride().x;
    const int32_t stride_h = kernel->Stride().y;
    if ( stride_w <= 3 && stride_h <= 3 )
    {
        // always supported
        return true;
    }
    // stride > 3 requires unrolling, check unroll conditions
    const bool hasPadding = !kernel->Padding().IsZero();
    const int32_t dilation_h = kernel->Dilation().y;
    const int32_t dilation_w = kernel->Dilation().x;
    const bool canUnroll = !hasPadding && (dilation_h == 1) && (dilation_w == 1);
    if ( !canUnroll )
    {
        Failure(op, fmt::format("Unsupported kernel stride: {}, {}", stride_w, stride_h));
        return false;
    }
    return true;
}

// Reduced axis must be less than or equal to 2^16
bool MatmulReducedAxis(const Operation *op)
{
    auto ifmConn = op->Input(TensorUsage::IFM0);
    auto opType = op->Type();
    assert(ifmConn);
    auto ifmShape = ifmConn->shape;
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
    if ( ifmShape.Depth() > (1 << 16) )
    {
        Failure(op, fmt::format("Reduced axis is too large: {}", ifmShape.Depth()));
        return false;
    }
    return true;
}

// OFM Depth must be less than or equal to 2^16
bool MatmulOFMDepth(const Operation *op)
{
    auto opType = op->Type();
    auto ofmConn = op->Output(TensorUsage::OFM);
    assert(ofmConn);
    assert(ofmConn->shape);
    int ofmDepth = ofmConn->shape.Depth();
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
    if ( ofmDepth > (1 << 16) )
    {
        Failure(op, fmt::format("OFM depth is too large: {}", ofmDepth));
        return false;
    }
    return true;
}

// IFM precision must be INT8
bool MatmulIFMPrecision(const Operation *op)
{
    auto opType = op->Type();
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
    auto ifmConn = op->Input(TensorUsage::IFM);
    assert(ifmConn);
    auto ifmType = ifmConn->tensor->Type();
    if ( ifmType != DataType::Int8 )
    {
        Failure(op, "Matmul with unsupported IFM type");
        return false;
    }
    return true;
}

// IFM depth must be less than or equal to 127
bool ArgMaxDepth(const Operation *op)
{
    int depth = op->Input(TensorUsage::IFM)->shape.Depth();
    if ( depth > 127 )
    {
        Failure(op, fmt::format("The depth of the argmax: {}, is over the limit: 127.", depth));
        return false;
    }
    return true;
}

// Operation must be performed along the depth axis
bool ArgMaxAxis(const Operation *op)
{
    // Get the input tensor's shape
    auto ifmConn = op->Input(TensorUsage::IFM);
    assert(ifmConn);
    int inp_dims = ifmConn->shape.Size();

    // Get the axis tensor
    auto axisConn = op->Input(TensorUsage::Params);
    assert(axisConn);
    auto axisTensor = axisConn->tensor.get();
    auto axisView = axisTensor->View();
    int axis = axisView.Buffer()->Data<int32_t>()[0];

    // Axis must be last dimension (inp_dims - 1) or -1
    if ( axis != (inp_dims - 1) && axis != -1 )
    {
        Failure(op, fmt::format("Axis is {}, but number of input dimensions is {}. Only last axis or -1 is supported.", axis, inp_dims));
        return false;
    }
    return true;
}

// Shape constraints for 32-bit transpose
bool Transpose32Bit(const Operation *op)
{
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto ifmShape = Shape::PadAxes(ifmConn->shape, 4, 1);
    auto ifmType = ifmConn->tensor->Type();
    auto *params = op->Input(TensorUsage::Params);
    assert(params);
    Shape perm = TensorToShape(params->tensor.get(), params->shape.Depth());
    auto transposeMask = TransposeTypeFromShape(perm);
    if ( ifmType != DataType::Int32 )
    {
        return true;
    }
    if ( ifmShape.Size() > 4 )
    {
        Failure(op, fmt::format("32-bit transpose with rank > 4: {}", ifmShape.ToString()));
        return false;
    }
    switch ( transposeMask )
    {
        case TransposeType::None:
            // 32-bit NHWC: C-axis must be 0->32768
            if ( ifmShape.Depth() > (1 << 15) )
            {
                Failure(op, fmt::format("32-bit NHWC transpose with depth > 32768: {}", ifmShape.ToString()));
                return false;
            }
            break;
        case TransposeType::NWHC:
        {
            // 32-bit NWHC: max-shape (1,65536,65536,16384)
            const static Shape maxShape = Shape(1, (1 << 16), (1 << 16), (1 << 14));
            if ( ifmShape.GreaterMask(maxShape) > 0 )
            {
                Failure(op, fmt::format("32-bit NWHC transpose with shape out of range: {}", ifmShape.ToString()));
                return false;
            }
        }
        break;
        case TransposeType::NHCW:
        {
            // 32-bit NHCW: (N*H: 65536, W: 65536, C: 65536)
            const static Shape maxShape = Shape((1 << 16), (1 << 16), (1 << 16));
            Shape ifmSquashed = ifmShape.WithHeight(ifmShape.Height() * ifmShape.Batch()).WithBatch(1);
            if ( ifmSquashed.GreaterMask(maxShape) > 0 )
            {
                Failure(op, fmt::format("32-bit NHCW transpose with shape out of range: {}", ifmSquashed.ToString()));
                return false;
            }
        }
        break;
        default:
            Failure(op, "Unsupported transpose-type");
            return false;
    }
    return true;
}

// Shape constraints for 8- and 16-bit transpose
bool Transpose8And16Bit(const Operation *op)
{
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto ifmShape = Shape::PadAxes(ifmConn->shape, 4, 1);
    auto ifmType = ifmConn->tensor->Type();
    auto *params = op->Input(TensorUsage::Params);
    assert(params);
    Shape perm = TensorToShape(params->tensor.get(), params->shape.Depth());
    auto transposeMask = TransposeTypeFromShape(perm);
    if ( DataTypeSizeBits(ifmType) > 16 )
    {
        return true;
    }
    if ( transposeMask == TransposeType::None )
    {
        // NHWC: any size is supported
        return true;
    }
    if ( (ifmShape.Size() <= 4) &&
         (transposeMask == TransposeType::NWHC || transposeMask == TransposeType::NHCW || transposeMask == TransposeType::NCWH ||
             transposeMask == TransposeType::NWCH || transposeMask == TransposeType::NCHW) )
    {
        // Directly HW-supported transpose-masks
        // NWHC/NHCW/NCWH: (N*H: 65536, W: 65536, C: 65536)
        // Indirectly HW-supported transpose-masks through decomposition
        // NWCH/NCHW: (N*H: 65536, W: 65536, C: 65536)
        const static Shape maxShape = Shape((1 << 16), (1 << 16), (1 << 16));
        Shape ifmSquashed = ifmShape.WithHeight(ifmShape.Height() * ifmShape.Batch()).WithBatch(1);
        if ( ifmSquashed.GreaterMask(maxShape) > 0 )
        {
            Failure(op,
                fmt::format("Transpose with permutation {} has shape out of range: {}", EnumToString(transposeMask),
                    ifmSquashed.ToString()));
            return false;
        }
    }
    else
    {
        // Decomposed transpose-masks
        // Axis product must be less or equal to 65536
        if ( ifmShape.Elements64() > (1 << 16) )
        {
            Failure(op,
                fmt::format("Transpose with permutation {} has shape out of range: {}", perm.ToString(), ifmShape.ToString()));
            return false;
        }
    }
    return true;
}

// Scale-factor constraints for Resize operations
bool Resize(const Operation *op)
{
    OpType opType = op->Type();
    assert(opType == OpType::ResizeBilinear || opType == OpType::ResizeNearestNeighbor);
    bool halfPixelCentersRB = false;
    bool alignCorners = false;
    const auto *passthrough = static_cast<const tflite::Operator *>(op->Passthrough());
    assert(passthrough);
    if ( opType == OpType::ResizeBilinear )
    {
        const auto *opt = passthrough->builtin_options_as_ResizeBilinearOptions();
        assert(opt);
        alignCorners = opt->align_corners();
        halfPixelCentersRB = opt->half_pixel_centers();
    }
    else
    {
        const auto *opt = passthrough->builtin_options_as_ResizeNearestNeighborOptions();
        assert(opt);
        alignCorners = opt->align_corners();
    }
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto ofmConn = op->Output(TensorUsage::OFM);
    assert(ifmConn);
    assert(ofmConn);
    Shape ifmShape = Shape::PadAxes(ifmConn->shape, 4, 1);
    Shape ofmShape = Shape::PadAxes(ofmConn->shape, 4, 1);
    if ( ifmShape.Height() == 1 && ifmShape.Width() == 1 )
    {
        return true;
    }
    if ( ifmShape.Height() == ofmShape.Height() && ifmShape.Width() == ofmShape.Width() )
    {
        return !halfPixelCentersRB;
    }
    float hUpscale;
    float wUpscale;
    if ( alignCorners )
    {
        hUpscale = ofmShape.Height() == 1 ? 1 : float(ofmShape.Height() - 1) / (ifmShape.Height() - 1);
        wUpscale = ofmShape.Width() == 1 ? 1 : float(ofmShape.Width() - 1) / (ifmShape.Width() - 1);
    }
    else
    {
        hUpscale = float(ofmShape.Height()) / ifmShape.Height();
        wUpscale = float(ofmShape.Width()) / ifmShape.Width();
    }
    if ( hUpscale != wUpscale )
    {
        if ( !((ofmShape.Height() == 1 && hUpscale == 1) || (ofmShape.Width() == 1 && wUpscale == 1)) )
        {
            Failure(op,
                fmt::format("HW scale-factor is not equal and operation has unsupported parameter combination ofm h={}, h scale-factor={}, ofm w={}, w scale-factor={}.",
                    ofmShape.Height(), hUpscale, ofmShape.Width(), wUpscale));
            return false;
        }
        else if ( halfPixelCentersRB )
        {
            Failure(op, fmt::format("HW scale-factor is not equal and Resize Bilinear has half pixel centers, h scale-factor={}, w scale-factor={}.", hUpscale, wUpscale));
            return false;
        }
    }
    auto maxUpscale = halfPixelCentersRB ? 2 : 8;
    auto upscale = std::max(hUpscale, wUpscale);
    if ( !((ifmShape.Height() == 1 && ifmShape.Width() == 1) ||
             (std::trunc(upscale) == upscale && IsPowerOfTwo(int(upscale)) && upscale > 1 && upscale <= maxUpscale)) )
    {
        Failure(op, fmt::format("Scaling matches and operation has unsupported scale-factor={}", upscale));
        return false;
    }
    return true;
}

// Used to explicitly reject some dataTypes for certain optypes
bool RejectPrecision(const Operation *op, int precision)
{
    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &[usage, conn] : list->pairs() )
        {
            auto type = conn.tensor->Type();

            if ( DataTypeSizeBits(type) == precision && (IsIFM(usage) || IsOFM(usage)) )
            {
                Failure(op, fmt::format("Unsupported dataType {}", DataTypeToString(type)));
                return false;
            }
        }
    }
    return true;
}

// Zero points must be:
// * 0 for Int32
// * non-negative for unsigned dataTypes
bool ConstrainZeroPoints(const Operation *op)
{
    OpType opType = op->Type();
    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &[usage, conn] : list->pairs() )
        {
            DataType dType = conn.tensor->Type();
            for ( auto zp : conn.quantization.zeroPoints )
            {
                if ( IsOFM(usage) || IsIFM(usage) )
                {
                    // must be 0 for 32-bit featureMaps
                    if ( DataTypeSizeBits(dType) == 32 && zp != 0 )
                    {
                        Failure(op, fmt::format("Unsupported zeroPoint {} for dataType: {}", zp, DataTypeToString(dType)));
                        return false;
                    }
                    // must be non-negative for unsigned featureMaps
                    if ( !IsSignedInteger(dType) && zp < 0 )
                    {
                        Failure(op, fmt::format("Unsupported zeroPoint {} for dataType: {}", zp, DataTypeToString(dType)));
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

// Stride values must be 1x1 or 2x2
// or 2x1 if IFM height and kernel height is 1
// or 1x2 if IFM width and kernel width is 1
bool TransposeConvStrides(const Operation *op)
{
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto kernel = op->Kernel();
    assert(ifmConn);
    assert(kernel);
    const auto &ifmShape = ifmConn->shape;
    auto [kw, kh] = kernel->Size();
    auto stride = kernel->Stride();
    if ( stride.x < 1 || stride.x > 2 || stride.y < 1 || stride.y > 2 )
    {
        Failure(op, fmt::format("stride out of range: ({},{})", stride.x, stride.y));
        return false;
    }
    if ( stride == Point2i(1, 2) && !(ifmShape.Width() == 1 && kw == 1) )
    {
        Failure(op, fmt::format("unsupported stride combination: ({},{})", stride.x, stride.y));
        return false;
    }
    if ( stride == Point2i(2, 1) && !(ifmShape.Height() == 1 && kh == 1) )
    {
        Failure(op, fmt::format("unsupported stride combination: ({},{})", stride.x, stride.y));
        return false;
    }
    return true;
}

}  // namespace

TfLiteSupportedOperatorsU55::TfLiteSupportedOperatorsU55() :
        TfLiteSupportedOperators(127 * (1 << 16),  // maxWeightSum8Bit
            127 * (1 << 16),                       // maxWeightSum16Bit
            (1LL << 40) - 1,                       // maxBias
            s_supportedDataTypes, s_supportedOpTypes)
{
    // create constraint objects
    reverseMask = {&ReverseMask, "Axis attribute is not supported."};
    kernelStride = {&KernelStride, "Kernel stride must be in the range (1,3)."};
    unrolledKernelStride = {&UnrolledKernelStride, "Stride>3 is only supported IF (dilation = 1 AND padding = VALID)"};
    matmulReducedAxis = {&MatmulReducedAxis, "IF Matmul or (FC with dynamic weights): Reduced axis must be less than or equal to 2^16"};
    matmulOFMDepth = {&MatmulOFMDepth, "IF Matmul or (FC with dynamic weights): OFM depth must be less than or equal to 2^16"};
    matmulIFMPrecision = {&MatmulIFMPrecision, "IF Matmul or (FC with dynamic weights): IFM precision must be Int8"};
    argMaxDepth = {&ArgMaxDepth, "IFM depth must be less than or equal to 127."};
    argMaxAxis = {&ArgMaxAxis, "Operation must be performed along the depth axis."};
    transpose32Bit = {&Transpose32Bit,
        "IF IFM is Int32:\n"
        "  - Rank must be less than or equal to 4\n"
        "  - NHWC: C <= 2^16\n"
        "  - NWHC: N ==1, H <= 2^16, W <= 2^16, C <= 2^14\n"
        "  - NHCW: N*H <= 2^16, W <= 2^16, C <= 2^16\n"
        "  - Any other permutation vector is unsupported"};

    transpose8And16Bit = {&Transpose8And16Bit,
        "IF IFM is Int8 or Int16\n"
        "  - NHWC: no shape constraints\n"
        "  - ELSE IF Rank <= 4D and permutation is: NWHC/NHCW/NCWH/NWCH/NCHW:\n"
        "    - (N*H, W, C) <= (2^16, 2^16, 2^16)\n"
        "  - ELSE:\n"
        "    - Product of elements must be less than or equal to 2^16."};

    resizeBilinear = {&Resize,
        "IF not (IFM H == IFM W == 1) AND not IFM Shape == OFM Shape:\n"
        "  - IF W Scale-factor != H Scale-factor:\n"
        "    - OFM W or H must be 1, and scaling in the dim that is must also be 1\n"
        "  - IF align corners:\n"
        "    - Scale-factor is defined as OFM H-1 / IFM H-1\n"
        "  - ELSE:\n"
        "    - Scale-factor is defined as OFM H/IFM H\n"
        "  - IF half pixel centers:\n"
        "    - Scale-factor needs to be 2x\n"
        "  - Else:\n"
        "    - Scale-factor needs to be one of: 2x/4x/8x\n"};

    resizeNearest = {&Resize,
        "IF NOT (IFM H == IFM W == 1) AND NOT IFM Shape == OFM Shape:\n"
        "  - IF W Scale-factor != H Scale-factor:\n"
        "    - OFM W or H must be 1, and scaling in the dim that is must also be 1\n"
        "  - IF align corners:\n"
        "    - Scale-factor is defined as OFM H-1 / IFM H-1\n"
        "  - ELSE:\n"
        "    - Scale-factor is defined as OFM H / IFM H\n"
        "  - Scale-factor must be one of: 2x/4x/8x\n"};

    reject32BitPrecision = {[](const Operation *op) { return RejectPrecision(op, 32); }, "32-bit feature-maps are not supported."};

    reject64BitPrecision = {[](const Operation *op) { return RejectPrecision(op, 64); }, "64-bit feature-maps are not supported."};

    constrainZeroPoints = {&ConstrainZeroPoints, "Zero-points must be 0 for Int32 and non-negative for unsigned dataTypes"};

    transposeConvStrides = {&TransposeConvStrides,
        "Stride values WxH must be: 1x1 OR 2x2 OR 2x1 (if IFM height and kernel height is 1) OR 1x2 (if IFM width and kernel width is 1)"};

    // populate opConstraints
    for ( OpType type : s_supportedOpTypes )
    {
        if ( s_ignoreZeroPoints.count(type) == 0 )
        {
            opConstraints[type].push_back(&constrainZeroPoints);
        }
        if ( s_supports32Bit.count(type) == 0 )
        {
            opConstraints[type].push_back(&reject32BitPrecision);
        }
        if ( s_supports64Bit.count(type) == 0 )
        {
            opConstraints[type].push_back(&reject64BitPrecision);
        }
        if ( IsConvolution(type) || IsPooling(type) || type == OpType::FullyConnected )
        {
            if ( type == OpType::Conv2D || type == OpType::AvgPool || type == OpType::MaxPool )
            {
                opConstraints[type].push_back(&unrolledKernelStride);
            }
            else if ( type != OpType::TransposeConv2D )
            {
                opConstraints[type].push_back(&kernelStride);
            }
        }
    }
    opConstraints[OpType::ReverseV2].push_back(&reverseMask);
    opConstraints[OpType::FullyConnected].push_back(&matmulReducedAxis);
    opConstraints[OpType::FullyConnected].push_back(&matmulOFMDepth);
    opConstraints[OpType::FullyConnected].push_back(&matmulIFMPrecision);
    opConstraints[OpType::BatchMatMul].push_back(&matmulReducedAxis);
    opConstraints[OpType::BatchMatMul].push_back(&matmulOFMDepth);
    opConstraints[OpType::BatchMatMul].push_back(&matmulIFMPrecision);
    opConstraints[OpType::ArgMax].push_back(&argMaxDepth);
    opConstraints[OpType::ArgMax].push_back(&argMaxAxis);
    opConstraints[OpType::Transpose].push_back(&transpose32Bit);
    opConstraints[OpType::Transpose].push_back(&transpose8And16Bit);
    opConstraints[OpType::ResizeBilinear].push_back(&resizeBilinear);
    opConstraints[OpType::ResizeNearestNeighbor].push_back(&resizeNearest);
    opConstraints[OpType::TransposeConv2D].push_back(&transposeConvStrides);
}
}  // namespace regor
