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
#include "compiler/operation_util.hpp"
#include "compiler/shape_util.hpp"

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
        OpType::Relu0To1,
        OpType::Relu6,
        OpType::ReluN1To1,
        OpType::Reshape,
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
        // clang-format on
    };
    _supportedDataTypes = {
        // clang-format off
        DataType::UInt8,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        // clang-format on
    };
    _maxWeightSum8Bit = 127 * (1 << 16);
    _maxWeightSum16Bit = 127 * (1 << 16);
    _maxBias = (1LL << 40) - 1;
    _checks = {
        &TfLiteSupportedOperatorsU55::ConstraintBroadcastShapes,
        &TfLiteSupportedOperatorsU55::ConstraintReverse,
        &TfLiteSupportedOperatorsU55::Constraint32bitOps,
        &TfLiteSupportedOperatorsU55::ConstraintArgMaxDepth,
        &TfLiteSupportedOperatorsU55::ConstraintArgMaxAxis,
        &TfLiteSupportedOperatorsU55::ConstraintKernelStride,
        &TfLiteSupportedOperatorsU55::ConstraintUnrolledKernelStride,
        &TfLiteSupportedOperatorsU55::ConstraintMatmul,
        &TfLiteSupportedOperatorsU55::ConstraintTranspose,
        &TfLiteSupportedOperatorsU55::ConstraintResize,
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

bool TfLiteSupportedOperatorsU55::ConstraintResize(const Operation *op)
{
    OpType opType = op->Type();
    if ( !(opType == OpType::ResizeBilinear || opType == OpType::ResizeNearestNeighbor) )
    {
        return true;
    }
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
    if ( ifmShape.Height() == ofmShape.Height() && ifmShape.Height() == ofmShape.Height() )
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
    std::string constraint =
        "If not (IFM H == IFM W == 1) and not IFM Shape == OFM Shape:\n"
        "\tIf W upScale != H upScale:\n"
        "\t\tOFM W or H must be 1, and scaling in the dim that is must also be 1\n"
        "\tIf align corners:\n"
        "\t\tupScale is definied as OFM H-1 / IFM H - 1\n"
        "\tElse:\n"
        "\t\tupScale is defined as OFM H/IFM H\n"
        "\t\tIF Resize Bilinear and half pixel centers:\n"
        "\t\t\tupscale needs to be 2x\n"
        "\t\tElse:\n"
        "\t\t\tupScale needs to be one of: 2x/4x/8x\n";


    if ( hUpscale != wUpscale )
    {
        if ( !((ofmShape.Height() == 1 && hUpscale == 1) || (ofmShape.Width() == 1 && wUpscale == 1)) )
        {
            Failure(op,
                fmt::format("HW upScaling is not equal and operation has unsupported parameter combination ofm h={}, h up-scale={}, ofm w={}, w up-scale={}.",
                    ofmShape.Height(), hUpscale, ofmShape.Width(), wUpscale),
                constraint);
            return false;
        }
        else if ( halfPixelCentersRB )
        {
            Failure(op, fmt::format("HW upScaling is not equal and Resize Bilinear has half pixel centers, h up-scale={}, w up-scale={}.", hUpscale, wUpscale),
                constraint);
            return false;
        }
    }
    auto maxUpscale = halfPixelCentersRB ? 2 : 8;

    auto upscale = std::max(hUpscale, wUpscale);
    if ( !((ifmShape.Height() == 1 && ifmShape.Width() == 1) ||
             (std::trunc(upscale) == upscale && IsPowerOfTwo(int(upscale)) && upscale > 1 && upscale <= maxUpscale)) )
    {
        Failure(op, fmt::format("Scaling matches and operation has unsupported upScaling={}", upscale), constraint);
        return false;
    }
    return true;
}


bool TfLiteSupportedOperatorsU55::ConstraintReverse(const Operation *op)
{
    if ( op->Type() != OpType::ReverseV2 )
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

bool TfLiteSupportedOperatorsU55::ConstraintTranspose(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType != OpType::Transpose )
    {
        return true;
    }
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto ofmConn = op->Output(TensorUsage::OFM);
    auto ifmShape = Shape::PadAxes(ifmConn->shape, 4, 1);
    auto ifmType = ifmConn->tensor->Type();
    auto *params = op->Input(TensorUsage::Params);
    assert(params);
    Shape perm = TensorToShape(params->tensor.get(), params->shape.Depth());
    auto transposeMask = TransposeTypeFromShape(perm);
    if ( ifmType == DataType::Int32 )
    {
        static const char *constraint =
            "IFM Shape constraints for 32-bit Transpose:\n"
            "  * Rank must be less than or equal to 4\n"
            "  * Max shape based on permutation:\n"
            "     NHWC: C <= 2^16\n"
            "     NWHC: N ==1, H <= 2^16, W <= 2^16, C <= 2^14\n"
            "     NHCW: N*H <= 2^16, W <= 2^16, C <= 2^16\n"
            "     Any other permutation vector is unsupported";
        if ( ifmShape.Size() > 4 )
        {
            Failure(op, fmt::format("32-bit transpose with rank > 4: {}", ifmShape.ToString()), constraint);
            return false;
        }
        switch ( transposeMask )
        {
            case TransposeType::None:
                // 32-bit NHWC: C-axis must be 0->32768
                if ( ifmShape.Depth() > (1 << 15) )
                {
                    Failure(op, fmt::format("32-bit NHWC transpose with depth > 32768: {}", ifmShape.ToString()), constraint);
                    return false;
                }
                break;
            case TransposeType::NWHC:
            {
                // 32-bit NWHC: max-shape (1,65536,65536,16384)
                const static Shape maxShape = Shape(1, (1 << 16), (1 << 16), (1 << 14));
                if ( ifmShape.GreaterMask(maxShape) > 0 )
                {
                    Failure(op, fmt::format("32-bit NWHC transpose with shape out of range: {}", ifmShape.ToString()), constraint);
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
                    Failure(op, fmt::format("32-bit NHCW transpose with shape out of range: {}", ifmSquashed.ToString()), constraint);
                    return false;
                }
            }
            break;
            default:
                Failure(op, "Unsupported transpose-type", constraint);
                return false;
        }
    }
    else
    {
        static const char *constraint =
            "IFM shape constraints for 8 or 16-bit Transpose:\n"
            "  * Max shape based on permutation:\n"
            "    NHWC: no shape constraints\n"
            "    ELSE IF Rank <= 4D and permutation is: NWHC/NHCW/NCWH:\n"
            "      (N*H, W, C) <= (2^16, 2^16, 2^16)\n"
            "    ELSE:\n"
            "      Product of elements must be less than or equal to 2^16.";
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
                        ifmSquashed.ToString()),
                    constraint);
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
                    fmt::format("Transpose with permutation {} has shape out of range: {}", perm.ToString(), ifmShape.ToString()), constraint);
                return false;
            }
        }
    }
    return true;
}

bool TfLiteSupportedOperatorsU55::Constraint32bitOps(const Operation *op)
{
    static const std::unordered_set<OpType> supported = {
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


// Check that depth is not greater than 127
bool TfLiteSupportedOperatorsU55::ConstraintArgMaxDepth(const Operation *op)
{
    if ( op->Type() != OpType::ArgMax )
    {
        return true;
    }
    int depth = op->Input(TensorUsage::IFM)->shape.Depth();
    if ( depth > 127 )
    {
        Failure(op, fmt::format("The depth of the argmax: {}, is over the limit: 127.", depth));
        return false;
    }
    return true;
}

// Check that the operations are performed along the depth axis
bool TfLiteSupportedOperatorsU55::ConstraintArgMaxAxis(const Operation *op)
{
    if ( op->Type() != OpType::ArgMax )
    {
        return true;
    }
    auto axis = op->Attribute<axis_attr_t>()->axis;
    const int noAxes = op->Input(TensorUsage::IFM)->shape.Size();
    if ( axis != noAxes - 1 )
    {
        Failure(op, fmt::format("The axis of the argmax: {}, is not equal to the index of the depth axis: {} ", axis, noAxes - 1));
        return false;
    }
    return true;
}

bool TfLiteSupportedOperatorsU55::ConstraintKernelStride(const Operation *op)
{
    const auto kernel = op->Kernel();
    assert(kernel);
    const int32_t stride_w = kernel->Stride().x;
    const int32_t stride_h = kernel->Stride().y;
    if ( op->Type() == OpType::Conv2D || op->Type() == OpType::AvgPool || op->Type() == OpType::MaxPool )
    {
        // Conv2D and Pooling is handled by ConstraintUnrolledKernelStride
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
    // Constraints for UnrollKernelStrides
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
    if ( !(op->Type() == OpType::Conv2D || op->Type() == OpType::AvgPool || op->Type() == OpType::MaxPool) )
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
        return false;
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
