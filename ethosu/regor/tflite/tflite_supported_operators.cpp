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

#include "tflite_supported_operators.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "compiler/op_type.hpp"
#include "compiler/operation_util.hpp"
#include "tflite_supported_operators_u55.hpp"
#include "tflite_supported_operators_u85.hpp"

#include "include/regor.h"

namespace regor
{

bool TfLiteSupportedOperators::ConstraintOpType(const Operation *op)
{
    OpType opType = op->Type();
    if ( _supportedOpTypes.count(opType) == 0 )
    {
        Failure(op, "OpType is not supported", "");
        return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintTensDtypes(const Operation *op)
{
    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &item : list->pairs() )
        {
            auto usage = item.first;
            const auto &conn = item.second;
            auto type = conn.tensor->Type();
            if ( (IsIFM(usage) || IsOFM(usage)) && _supportedDataTypes.count(type) == 0 )
            {
                Failure(op, fmt::format("Operation has tensor with unsupported DataType {}", DataTypeToString(type)), "");
                return false;
            }
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintNumSplits(const Operation *op)
{
    const char *constraint = "num_splits must match the number of outputs";
    const tflite::Operator *passthrough = static_cast<const tflite::Operator *>(op->Passthrough());
    OpType opType = op->Type();
    int numSplits = 0;
    if ( opType == OpType::Split )
    {
        assert(passthrough);
        const auto *opt = passthrough->builtin_options_as_SplitOptions();
        assert(opt);
        numSplits = opt->num_splits();
    }
    else if ( opType == OpType::SplitV )
    {
        assert(passthrough);
        const auto *opt = passthrough->builtin_options_as_SplitVOptions();
        assert(opt);
        numSplits = opt->num_splits();
    }
    else
    {
        return true;
    }
    int numOutputs = op->Outputs().size();
    if ( numSplits != numOutputs )
    {
        Failure(op, fmt::format("num_splits: {} does not match the number of outputs: {}", numSplits, numOutputs), constraint);
        return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintMustHaveIFM(const Operation *op)
{
    const char *constraint = "Operations must have at least one IFM.";
    for ( const auto item : op->Inputs().pairs() )
    {
        auto usage = item.first;
        if ( IsIFM(usage) )
        {
            return true;
        }
    }
    Failure(op, "Operation without IFM", constraint);
    return false;
}

bool TfLiteSupportedOperators::ConstraintMustHaveOFM(const Operation *op)
{
    const char *constraint = "Operations must have at least one OFM.";
    for ( const auto item : op->Outputs().pairs() )
    {
        auto usage = item.first;
        if ( IsOFM(usage) )
        {
            return true;
        }
    }
    Failure(op, "Operation without OFM", constraint);
    return false;
}

bool TfLiteSupportedOperators::ConstraintTensMustHaveShape(const Operation *op)
{
    const char *constraint = "Tensors must have constant shape.";
    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &item : list->pairs() )
        {
            auto usage = item.first;
            const auto &conn = item.second;
            if ( !conn.shape )
            {
                Failure(op, "Operation has shapeless tensor", constraint);
                return false;
            }
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintTensQuantized(const Operation *op)
{
    const char *constraint = "Input(s), Output and Weight tensors must have quantization parameters";
    // Exceptions for this check
    switch ( op->Type() )
    {
        case OpType::ArgMax:
        case OpType::MirrorPad:
        case OpType::Quantize:
        case OpType::Shape:
        case OpType::Transpose:
        case OpType::GatherNd:
        case OpType::GatherV2:
        case OpType::Select:
        case OpType::SelectV2:
        case OpType::ScatterNd:
        case OpType::Pad:
        case OpType::PadV2:
        case OpType::ReduceAll:
        case OpType::ReduceAny:
        case OpType::ExpandDims:
        case OpType::MemoryCopy:
            return true;
        default:
            break;
    }
    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &item : list->pairs() )
        {
            auto usage = item.first;
            const auto &conn = item.second;
            if ( IsIFM(usage) || IsOFM(usage) || usage == TensorUsage::Weights )
            {
                const Quantization &quant = conn.quantization;
                if ( quant.scales.empty() || quant.zeroPoints.empty() )
                {
                    Failure(op, fmt::format("Operation has tensor {} with missing quantization parameters", conn.tensor->Name()), constraint);
                    return false;
                }
            }
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintFCWeightShape(const Operation *op)
{
    const char *constraint = "FullyConnected weights must be on the form O,1,1,..,1,I";
    if ( op->Type() != OpType::FullyConnected )
    {
        return true;
    }
    auto weights = op->Input(TensorUsage::Weights);
    assert(weights);
    assert(weights->tensor);
    const auto &shape = weights->shape;
    // Total elements must be equal to first-dim * last-dim
    if ( shape.Size() < 2 || (shape.Elements() != (shape[0] * shape[-1])) )
    {
        Failure(op, fmt::format("Unsupported weights shape: {}", shape.ToString()), constraint);
        return false;
    }

    return true;
}

bool TfLiteSupportedOperators::ConstraintPerAxisQuant(const Operation *op)
{
    OpType opType = op->Type();
    if ( IsConvolution(opType) || opType == OpType::FullyConnected )
    {
        return true;
    }

    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &[usage, conn] : list->pairs() )
        {
            if ( conn.quantization.scales.size() > 1 || conn.quantization.zeroPoints.size() > 1 )
            {
                Failure(op, "Operation does not support per-axis quantization", "");
                return false;
            }
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintMatchingQuantization(const Operation *op)
{
    const char *constraint = "Both Input quantization parameters must match OFM quantization parameters";

    OpType opType = op->Type();

    if ( opType != OpType::Minimum && opType != OpType::Maximum )
    {
        return true;
    }

    const auto ofmConn = op->Output(TensorUsage::OFM);
    const auto ifmConn = op->Input(TensorUsage::IFM);
    const auto ifm2Conn = op->Input(TensorUsage::IFM1);
    assert(ofmConn);
    assert(ifmConn);
    assert(ifm2Conn);
    const auto &ofmQuant = ofmConn->quantization;
    const auto &ifmQuant = ifmConn->quantization;
    const auto &ifm2Quant = ifm2Conn->quantization;
    if ( ifmQuant != ofmQuant || ifm2Quant != ofmQuant )
    {
        Failure(op, "Operation has mismatching quantization parameters.", constraint);
        return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintZeroPoints(const Operation *op)
{
    OpType opType = op->Type();
    // zeroPoints are ignored for the following operations to align with reference
    if ( opType == OpType::AvgPool || opType == OpType::Resize || opType == OpType::CLZ || opType == OpType::SHL ||
         opType == OpType::Div || opType == OpType::UnidirectionalSequenceLstm )
    {
        return true;
    }
    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &[usage, conn] : list->pairs() )
        {
            DataType dType = conn.tensor->Type();
            for ( auto zp : conn.quantization.zeroPoints )
            {
                if ( !_archConstraints->SupportedZeroPoint(zp, usage, dType, opType) )
                {
                    Failure(op, fmt::format("tensor {} has unsupported zeroPoint: {}", conn.tensor->Name(), zp));
                    return false;
                }
            }
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintWeightsPrecision(const Operation *op)
{
    const char *constraint = "Weight tensors must be 8-bit precision";
    const auto wconn = op->Input(TensorUsage::Weights);
    if ( !wconn )
    {
        return true;
    }
    const auto type = wconn->tensor->Type();
    if ( DataTypeSizeBits(type) != 8 )
    {
        Failure(op, fmt::format("Weights tensor with precision: {}", DataTypeToString(type)), constraint);
        return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintWeightSum(const Operation *op)
{
    static const std::string constraint = fmt::format(
        "The sum of absolute weights cannot exceed:\n"
        "\t{} for 8-bit IFM\n"
        "\t{} for 16-bit IFM",
        _maxWeightSum8Bit, _maxWeightSum16Bit);

    auto wConn = op->Input(TensorUsage::Weights);
    auto ifmConn = op->Input(TensorUsage::IFM);
    if ( !wConn || !ifmConn )
    {
        return true;
    }
    if ( !wConn->tensor->IsConstant() )
    {
        return true;
    }

    auto view = wConn->tensor->View();
    auto zeroPoints = wConn->quantization.zeroPoints;
    auto ifmType = ifmConn->tensor->Type();
    int ifmBits = DataTypeSizeBits(ifmType);
    int64_t maxWeightSum = ifmBits == 8 ? _maxWeightSum8Bit : _maxWeightSum16Bit;
    auto reader = view.Values<int>(wConn->tensor->Type());
    AxisOrder order = wConn->tensor->AxisOrder();
    Shape readShape = wConn->tensor->StorageShape();
    assert(readShape.Size() == 4);
    assert(order == AxisOrder::OHWI || order == AxisOrder::IHWO);

    int outChannels = readShape.Depth();
    int inChannels = readShape.Batch();
    if ( order == AxisOrder::OHWI )
    {
        std::swap(outChannels, inChannels);
    }
    // abort early if the readShape of the weights tensor guarantees no overflow.
    if ( (255 * readShape.Elements64() / outChannels) < maxWeightSum )
    {
        return true;
    }
    // Accumulate the weights in slices of output-channels
    // Fail if any slice overflows maxWeightSum
    for ( int out = 0; out < outChannels; out++ )
    {
        int64_t zeroPoint = 0;
        if ( !zeroPoints.empty() )
        {
            zeroPoint = zeroPoints.size() > 1 ? zeroPoints[out] : zeroPoints[0];
        }
        int64_t sum = 0;
        for ( int in = 0; in < inChannels; in++ )
        {
            for ( int h = 0; h < readShape.Height(); h++ )
            {
                for ( int w = 0; w < readShape.Width(); w++ )
                {
                    int64_t v;
                    if ( order == AxisOrder::OHWI )
                    {
                        v = reader[{out, h, w, in}];
                    }
                    else
                    {
                        v = reader[{in, h, w, out}];
                    }
                    sum += std::abs(v - zeroPoint);
                }
            }
        }
        if ( sum > maxWeightSum )
        {
            Failure(op, fmt::format("The absolute sum of weight-tensor elements: {} exceeds {}", sum, maxWeightSum), constraint);
            return false;
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintBias(const Operation *op)
{
    auto bConn = op->Input(TensorUsage::Scales);
    if ( !bConn )
    {
        return true;
    }
    auto bShape = bConn->shape;
    if ( bShape.Elements() > bShape.Depth() )
    {
        Failure(op, fmt::format("Bias shape: {}", bShape.ToString()), "bias-values must be stored in channel axis");
        return false;
    }
    if ( !bConn->tensor->IsConstant() )
    {
        Failure(op, "Operation has non-constant bias tensor.", "The bias tensor must be constant");
        return false;
    }
    auto type = bConn->tensor->Type();
    if ( type != DataType::Int32 && type != DataType::Int64 )
    {
        Failure(op, fmt::format("Operation has bias with type:{}", DataTypeToString(type)), "The bias tensor precision must be Int32 or Int64");
        return false;
    }
    if ( type == DataType::Int64 )
    {
        // read bias values
        auto view = bConn->tensor->View();
        auto values = view.Values<int64_t>();
        for ( int64_t bias : values )
        {
            if ( bias > _maxBias )
            {
                static const std::string constraint = fmt::format("Int64 bias must be smaller than {}", _maxBias);
                Failure(op, fmt::format("Bias is out of range: {} > {}", bias, _maxBias), constraint);
                return false;
            }
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintAvgPool(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType != OpType::AvgPool )
    {
        return true;
    }
    auto kernel = op->Kernel();
    assert(kernel);
    auto [w, h] = kernel->Size();
    if ( kernel->Padding().IsZero() )
    {
        // VALID padding
        if ( h > 256 || h < 1 )
        {
            Failure(op, fmt::format("kernel height: {} out of range", h), "When padding=VALID, kernel-height must be in the range (1,256)");
            return false;
        }
        if ( h * w > 256 * 256 )
        {
            Failure(op, fmt::format("kernel product: {} out of range", h * w),
                "When padding=VALID, kernel product (H*W) must be in the range (1, 256*256)");
            return false;
        }
    }
    else
    {
        // SAME padding
        if ( w > 8 || w < 1 )
        {
            // kernel width out of range
            Failure(op, fmt::format("kernel width: {} out of range", w), "When padding=SAME, kernel width must be in the range (1,8)");
            return false;
        }
        if ( h > 8 || h < 1 )
        {
            Failure(op, fmt::format("kernel height: {} out of range", h), "When padding=SAME, kernel height must be in the range (1,8)");
            return false;
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintMaxPool(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType != OpType::MaxPool )
    {
        return true;
    }
    auto kernel = op->Kernel();
    assert(kernel);
    auto [w, h] = kernel->Size();
    auto [sw, sh] = kernel->Stride();
    if ( h > 256 || h < 1 )
    {
        Failure(op, fmt::format("kernel height: {} out of range", h), "Kernel height must be in the range (1, 256)");
        return false;
    }
    if ( h * w > 256 * 256 )
    {
        Failure(op, fmt::format("kernel product: {} out of range", h * w), "Kernel product must be in the range (1, 256 * 256)");
        return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintTCStrides(const Operation *op)
{
    static const std::string constraint =
        "Stride values WxH must be:\n"
        "\t1x1 OR 2x2\n"
        "\tOR 2x1 if ifm height and kernel height = 1\n"
        "\tOR 1x2 if ifm width and kernel width = 1";
    OpType opType = op->Type();
    if ( opType != OpType::TransposeConv2D )
    {
        return true;
    }
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto kernel = op->Kernel();
    assert(ifmConn);
    assert(kernel);
    const auto &ifmShape = ifmConn->shape;
    auto [kw, kh] = kernel->Size();
    auto stride = kernel->Stride();

    if ( stride.x < 1 || stride.x > 2 || stride.y < 1 || stride.y > 2 )
    {
        Failure(op, fmt::format("stride out of range: ({},{})", stride.x, stride.y), constraint);
        return false;
    }
    if ( stride == Point2i(1, 2) && !(ifmShape.Width() == 1 && kw == 1) )
    {
        Failure(op, fmt::format("unsupported stride combination: ({},{})", stride.x, stride.y), constraint);
        return false;
    }
    if ( stride == Point2i(2, 1) && !(ifmShape.Height() == 1 && kh == 1) )
    {
        Failure(op, fmt::format("unsupported stride combination: ({},{})", stride.x, stride.y), constraint);
        return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintTCShapes(const Operation *op)
{
    static const std::string constraint =
        "if PADDING=SAME\n"
        "\tOFM must equal IFM * stride\n"
        "if PADDING=VALID\n"
        "\tOFM must equal IFM * stride + (kernel - stride)";
    OpType opType = op->Type();
    if ( opType != OpType::TransposeConv2D )
    {
        return true;
    }
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto ofmConn = op->Output(TensorUsage::OFM);
    auto kernel = op->Kernel();
    assert(ifmConn);
    assert(ofmConn);
    assert(kernel);
    const auto &ifmShape = ifmConn->shape;
    const auto &ofmShape = ofmConn->shape;
    auto stride = kernel->Stride();
    assert(op->Passthrough());
    const tflite::Operator *passthrough = static_cast<const tflite::Operator *>(op->Passthrough());
    const auto *opt = passthrough->builtin_options_as<tflite::TransposeConvOptions>();
    assert(opt);
    Point2i ifmWH(ifmShape.Width(), ifmShape.Height());
    Point2i ofmWH(ofmShape.Width(), ofmShape.Height());
    if ( opt->padding() == tflite::Padding::SAME )
    {
        if ( ifmWH * stride != ofmWH )
        {
            Failure(op,
                fmt::format("(Padding::SAME) Unsupported IFM/OFM shapes. ifm:({},{}), ofm:({},{}), stride:({},{})",
                    ifmWH.x, ifmWH.y, ofmWH.x, ofmWH.y, stride.x, stride.y),
                constraint);
            return false;
        }
    }
    else
    {
        Point2i diff = Point2i::Max((kernel->Size() - stride), Point2i(0, 0));
        if ( (ifmWH * stride + diff) != ofmWH )
        {
            Failure(op,
                fmt::format("(Padding::VALID) Unsupported IFM/OFM shapes. ifm:({},{}) ofm:({},{}), stride:({},{}), kernel:({},{})",
                    ifmWH.x, ifmWH.y, ofmWH.x, ofmWH.y, stride.x, stride.y, kernel->Size().x, kernel->Size().y),
                constraint);
            return false;
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintRsqrt(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType != OpType::Rsqrt )
    {
        return true;
    }
    auto ifmConn = op->Input(TensorUsage::IFM);
    assert(ifmConn);
    auto ifmType = ifmConn->tensor->Type();
    if ( ifmType != DataType::Int8 && ifmType != DataType::Int16 )
    {
        Failure(op, fmt::format("{} IFM", DataTypeToString(ifmType)), "IFM must be Int8 or Int16");
        return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintConstParams(const Operation *op)
{
    OpType opType = op->Type();
    switch ( opType )
    {
        case OpType::Slice:
        case OpType::StridedSlice:
        case OpType::Mean:
        case OpType::Pad:
        case OpType::PadV2:
        case OpType::MirrorPad:
        case OpType::Transpose:
            break;
        default:
            return true;
    }

    for ( const auto item : op->Inputs().pairs() )
    {
        auto usage = item.first;
        auto &conn = item.second;
        if ( IsParams(usage) && !conn.tensor->IsConstant() )
        {
            assert(conn.tensor);
            Failure(op, fmt::format("non-constant tensor {}", conn.tensor->Name()), "Parameter tensors must be constant");
            return false;
        }
    }

    return true;
}

bool TfLiteSupportedOperators::ConstraintMean(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType != OpType::Mean )
    {
        return true;
    }
    static constexpr int MAX_MEAN_KERNEL_SIZE = 64 * 64;
    static constexpr int MAX_MEAN_ELEMENTS_INT8 = 1 << 24;   // 2²⁴ x 2⁷  = 2³¹
    static constexpr int MAX_MEAN_ELEMENTS_UINT8 = 1 << 23;  // 2²³ x 2⁸  = 2³¹
    static constexpr int MAX_MEAN_ELEMENTS_INT16 = 1 << 16;  // 2¹⁶ x 2¹⁵ = 2³¹
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto params = op->Input(TensorUsage::Params);
    assert(ifmConn);
    assert(params);
    auto ifmShape = ifmConn->shape;
    auto axisTens = params->tensor;
    auto axisCount = axisTens->StorageShape().IsEmpty() ? 1 : axisTens->StorageShape().Depth();
    auto axisValues = axisTens->View().Values<int32_t>();

    auto axisMask = ifmShape.WithZeros();
    for ( int i = 0; i < axisCount; i++ )
    {
        axisMask[axisValues[i]] = 1;
    }

    axisMask = Shape::PadAxes(axisMask, 4, 0);
    Shape ifmShape4D = Shape::PadAxes(ifmShape, 4, 1);

    auto ifmType = ifmConn->tensor->Type();

    // Constrain IFM-Batch to 1
    if ( ifmShape4D.Batch() > 1 )
    {
        Failure(op, fmt::format("Batch > 1: {}", ifmShape4D.ToString()), "Batch > 1 is not supported");
        return false;
    }

    // Reduced depth is only supported if any of IFM H,W,C is 1
    if ( axisMask.Depth() )
    {
        bool supported = false;
        for ( int i = 1; i < 4; i++ )
        {
            if ( ifmShape4D[i] == 1 )
            {
                supported = true;
                break;
            }
        }
        if ( !supported )
        {
            Failure(op, fmt::format("Unsupported depth-reduction.  IFM: {}", ifmShape4D.ToString()), "Depth is only supported if any of h,w,c == 1");
            return false;
        }
    }

    // Reduced axes are represented with their IFM-value
    // Non reduced axes are represented by 0
    // e.g. IFM (5,8,7,9) with axis=H,C -> (0,8,0,9)
    Shape reducedAxes = ifmShape4D * axisMask;
    // Constrain kernel-size
    if ( reducedAxes.GreaterMask(Shape(nullptr, 4, MAX_MEAN_KERNEL_SIZE)) != 0 )
    {
        static const std::string constraint = fmt::format("Reduced axis must be less than {}", MAX_MEAN_KERNEL_SIZE);
        Failure(op, "Reduced axis is too large", constraint);
        return false;
    }

    // Constrain reduced elements
    int elements = 1;
    for ( int i = 0; i < axisMask.Size(); i++ )
    {
        elements *= axisMask[i] ? ifmShape4D[i] : 1;
    }
    switch ( ifmConn->tensor->Type() )
    {
        case DataType::Int8:
            if ( elements > MAX_MEAN_ELEMENTS_INT8 )
            {
                static const std::string constraint = fmt::format("max elements (int8) = {}", MAX_MEAN_ELEMENTS_INT8);
                Failure(op, fmt::format("Too many reduced elements: {}", elements), constraint);
                return false;
            }
            break;
        case DataType::UInt8:
            if ( elements > MAX_MEAN_ELEMENTS_UINT8 )
            {
                static const std::string constraint = fmt::format("max elements (uint8) = {}", MAX_MEAN_ELEMENTS_UINT8);
                Failure(op, fmt::format("Too many reduced elements: {}", elements), constraint);
                return false;
            }
            break;
        case DataType::Int16:
            if ( elements > MAX_MEAN_ELEMENTS_INT16 )
            {
                static const std::string constraint = fmt::format("max elements (int16) = {}", MAX_MEAN_ELEMENTS_INT16);
                Failure(op, fmt::format("Too many reduced elements: {}", elements), constraint);
                return false;
            }
            break;
        default:
            Failure(op, fmt::format("Unsupported Mean IFM type {}", DataTypeToString(ifmType)));
            return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintSoftmax(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType != OpType::Softmax )
    {
        return true;
    }
    auto ifmConn = op->Input(TensorUsage::IFM);
    assert(ifmConn);
    static constexpr int maxProd = 1 << 16;
    const auto ifmShape = Shape::PadAxes(ifmConn->shape, 4, 1);
    if ( ifmShape.ElementsWH() > maxProd )
    {
        Failure(op, fmt::format("ifmShape: ({}), W * H = {}", ifmShape.ToString(), ifmShape.ElementsWH()),
            "The product of IFM width and height must be less than 65536");
        return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintPad(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType != OpType::Pad && opType != OpType::PadV2 && opType != OpType::MirrorPad )
    {
        return true;
    }
    auto params = op->Input(TensorUsage::Params);
    assert(params);
    const auto &pType = params->tensor->Type();
    if ( pType != DataType::Int32 && pType != DataType::Int64 )
    {
        Failure(op, fmt::format("Params tensor with datatype: {}", DataTypeToString(pType)), "Params tensor must be Int32 or Int64.");
        return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintTransposeDims(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType != OpType::Transpose )
    {
        return true;
    }
    auto params = op->Input(TensorUsage::Params);
    assert(params);
    if ( params->shape.Depth() > 8 )
    {
        Failure(op, "Unsupported transpose-shape", "tensor dimension must be <= 8");
        return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintStridedSlice(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType == OpType::StridedSlice )
    {
        const auto *ifmConn = op->Input(TensorUsage::IFM);
        const auto *ofmConn = op->Output(TensorUsage::OFM);
        const auto *beginParmConn = op->Input(TensorUsage::Params0);
        const auto *endParamConn = op->Input(TensorUsage::Params1);
        const auto *stridesParamConn = op->Input(TensorUsage::Params2);

        // Read StridedSlice attributes
        int32_t begin_mask = 0;
        int32_t ellipsis_mask = 0;
        int32_t end_mask = 0;
        int32_t new_axis_mask = 0;
        int32_t shrink_axis_mask = 0;
        const tflite::Operator *const passthrough = static_cast<const tflite::Operator *>(op->Passthrough());

        if ( passthrough )
        {
            const auto options = passthrough->builtin_options_as_StridedSliceOptions();
            if ( options )
            {
                begin_mask = options->begin_mask();
                ellipsis_mask = options->ellipsis_mask();
                end_mask = options->end_mask();
                new_axis_mask = options->new_axis_mask();
                shrink_axis_mask = options->shrink_axis_mask();
            }
        }

        const Shape beginAttr = TensorToShape(beginParmConn->tensor.get(), beginParmConn->shape.Elements());
        const Shape endAttr = TensorToShape(endParamConn->tensor.get(), endParamConn->shape.Elements());

        const Shape stridesAttr = TensorToShape(stridesParamConn->tensor.get(), stridesParamConn->shape.Elements());
        const int specShapeSize = std::min({beginAttr.Size(), endAttr.Size(), stridesAttr.Size()});

        // Start off with the full IFM
        const int ifmShapeSize = ifmConn->shape.Size();
        Shape sliceOffset(nullptr, ifmShapeSize, 0);
        Shape sliceShape(ifmConn->shape);
        Shape sliceStride(nullptr, ifmShapeSize, 1);

        // Process each spec
        for ( int specIndex = 0, ifmIndex = 0; specIndex < specShapeSize; specIndex++ )
        {
            const bool isBegin = (begin_mask & (1 << specIndex)) != 0;
            const bool isEllipsis = (ellipsis_mask & (1 << specIndex)) != 0;
            const bool isEnd = (end_mask & (1 << specIndex)) != 0;
            const bool isNewAxis = (new_axis_mask & (1 << specIndex)) != 0;
            const bool isShrink = (shrink_axis_mask & (1 << specIndex)) != 0;

            if ( isEllipsis )
            {
                // Skip to the end
                ifmIndex = ifmShapeSize - (specShapeSize - specIndex - 1);
                assert(ifmIndex >= 0);
                assert(ifmIndex <= ifmShapeSize);
            }
            else
            {
                if ( !isBegin || isShrink )
                {
                    // Handle the begin value
                    int begin = beginAttr[specIndex];
                    if ( begin < 0 ) begin = ifmConn->shape[ifmIndex] + begin;
                    begin = std::clamp(begin, 0, ifmConn->shape[ifmIndex] - 1);
                    sliceOffset[ifmIndex] = begin;
                    sliceShape[ifmIndex] = isShrink ? 1 : ifmConn->shape[ifmIndex] - begin;
                }

                if ( !isEnd && !isShrink )
                {
                    // Handle the end value
                    int end = endAttr[specIndex];
                    if ( end < 0 ) end = ifmConn->shape[ifmIndex] + end;
                    end = std::clamp(end, 1, ifmConn->shape[ifmIndex]);
                    assert(end > sliceOffset[ifmIndex]);
                    sliceShape[ifmIndex] = end - sliceOffset[ifmIndex];
                }

                // Handle the stride value
                sliceStride[ifmIndex] = stridesAttr[specIndex];

                // Go to next dimension
                ifmIndex++;
            }
        }

        // TODO MLBEDSW-10165: Handle stride < 0 and other dimensions than H and W
        if ( sliceStride.WithHeight(1).WithWidth(1).Elements64() != 1 )
        {
            Failure(op, "StridedSlice with unsupported stride axis", "Stride must be over H or W");
            return false;
        }
        if ( sliceStride.LessMask(sliceStride.WithZeros()) )
        {
            Failure(op, "StridedSlice with unsupported negative stride", "Negative stride is not supported");
            return false;
        }
        if ( !sliceShape.GreaterMask(sliceShape.WithZeros()) )
        {
            Failure(op, fmt::format("StridedSlice with invalid sliceShape: {}", sliceShape.ToString()), "sliceShape must be a volume");
            return false;
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintLog(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType != OpType::Log )
    {
        return true;
    }
    auto ifmConn = op->Input(TensorUsage::IFM);
    assert(ifmConn);
    auto ofmConn = op->Output(TensorUsage::OFM);
    assert(ofmConn);
    auto ifmType = ifmConn->tensor->Type();
    auto ofmType = ofmConn->tensor->Type();
    const auto &ifmShape = ifmConn->shape;
    const auto &ofmShape = ofmConn->shape;
    if ( ifmType != DataType::Int8 && ifmType != DataType::Int16 )
    {
        Failure(op, fmt::format("{} IFM", DataTypeToString(ifmType)), "IFM must be Int8 or Int16");
        return false;
    }
    if ( ifmType != ofmType )
    {
        Failure(op, fmt::format("{} IFM {} OFM", DataTypeToString(ifmType), DataTypeToString(ofmType)), "IFM and OFM data types must match");
        return false;
    }
    if ( ifmShape != ofmShape )
    {
        Failure(op, fmt::format("{} IFM {} OFM", ifmShape.ToString(), ofmShape.ToString()), "IFM and OFM shape must match");
        return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintLSTM(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType != OpType::UnidirectionalSequenceLstm )
    {
        return true;
    }

    for ( int i = 0; i <= 7; i++ )
    {
        // Check that all the gate weights are present. If they are not it's either invalid or using Couple
        // Input and Forget Gate (CIFG), where the input gate is computed implicitly from the forget gate,
        // which is not supported.
        if ( op->Input(MakeTensorUsage(TensorUsage::Weights, i)) == nullptr )
        {
            Failure(op, "Missing gate weight tensor", "LSTM with implicit gate calculation is not supported");
            return false;
        }
    }

    for ( int i = 8; i <= 10; i++ )
    {
        if ( op->Input(MakeTensorUsage(TensorUsage::Weights, i)) )
        {
            Failure(op, "Peephole weight tensor present", "Peephole LSTM variant is not supported");
            return false;
        }
    }

    if ( op->Input(MakeTensorUsage(TensorUsage::Weights, 11)) || op->Input(MakeTensorUsage(TensorUsage::Scales, 4)) )
    {
        Failure(op, "Projection weight or bias tensor present", "LSTM with projection is not supported");
        return false;
    }

    for ( int i = 5; i <= 8; i++ )
    {
        if ( op->Input(MakeTensorUsage(TensorUsage::Scales, i)) )
        {
            Failure(op, "Normalization coefficient tensor present", "LSTM with gate normalization is not supported");
            return false;
        }
    }

    return true;
}

void TfLiteSupportedOperators::Failure(const Operation *op, const std::string &message, const std::string &constraint)
{
    assert(op);
    auto ofmConn = op->Output(TensorUsage::OFM);
    const char *name = "N/A";
    OpType opType = op->Type();
    if ( ofmConn && ofmConn->tensor )
    {
        name = ofmConn->tensor->Name().c_str();
    }
    std::string type = OpTypeToString(op->Type());
    if ( opType != OpType::None && opType != OpType::Passthrough )
    {
        auto tfLiteType = TfLiteMapping::OpTypeToBuiltinOperator(opType);
        type = TfLiteMapping::BuiltinOperatorToString(tfLiteType);
    }
    assert(message.size() || constraint.size());
    LOG_WARN("\nWarning (supported operators) operator:{} ofm:{}\n", std::move(type), name);
    if ( message.size() )
    {
        LOG_WARN("Reason: {}\n", message);
    }
    if ( constraint.size() )
    {
        LOG_WARN("Constraint: {}\n", constraint);
    }
}

TfLiteSupportedOperators::TfLiteSupportedOperators(IArchitectureConstraints *constraints) :
        _archConstraints(constraints)
{
    _maxWeightSum8Bit = 0;
    _maxWeightSum16Bit = 0;
    _maxBias = 0;
    _genericChecks = {
        &TfLiteSupportedOperators::ConstraintOpType,
        &TfLiteSupportedOperators::ConstraintTensDtypes,
        &TfLiteSupportedOperators::ConstraintNumSplits,
        &TfLiteSupportedOperators::ConstraintMustHaveIFM,
        &TfLiteSupportedOperators::ConstraintMustHaveOFM,
        &TfLiteSupportedOperators::ConstraintTensMustHaveShape,
        &TfLiteSupportedOperators::ConstraintFCWeightShape,
        &TfLiteSupportedOperators::ConstraintTensQuantized,
        &TfLiteSupportedOperators::ConstraintPerAxisQuant,
        &TfLiteSupportedOperators::ConstraintMatchingQuantization,
        &TfLiteSupportedOperators::ConstraintZeroPoints,
        &TfLiteSupportedOperators::ConstraintWeightsPrecision,
        &TfLiteSupportedOperators::ConstraintWeightSum,
        &TfLiteSupportedOperators::ConstraintBias,
        &TfLiteSupportedOperators::ConstraintAvgPool,
        &TfLiteSupportedOperators::ConstraintMaxPool,
        &TfLiteSupportedOperators::ConstraintTCStrides,
        &TfLiteSupportedOperators::ConstraintTCShapes,
        &TfLiteSupportedOperators::ConstraintRsqrt,
        &TfLiteSupportedOperators::ConstraintConstParams,
        &TfLiteSupportedOperators::ConstraintMean,
        &TfLiteSupportedOperators::ConstraintSoftmax,
        &TfLiteSupportedOperators::ConstraintPad,
        &TfLiteSupportedOperators::ConstraintTransposeDims,
        &TfLiteSupportedOperators::ConstraintStridedSlice,
        &TfLiteSupportedOperators::ConstraintLog,
        &TfLiteSupportedOperators::ConstraintLSTM,
    };
}

std::unique_ptr<TfLiteSupportedOperators> MakeSupportedOpsChecker(const std::string &target, IArchitectureConstraints *constraints)
{
    if ( target == REGOR_ARCH_ETHOSU85 )
    {
        return std::make_unique<TfLiteSupportedOperatorsU85>(constraints);
    }
    else
    {
        assert(target == REGOR_ARCH_ETHOSU55 || target == REGOR_ARCH_ETHOSU65);
        return std::make_unique<TfLiteSupportedOperatorsU55>(constraints);
    }
}

}  // namespace regor
