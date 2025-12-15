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

#include <unordered_set>

#include "include/regor.h"

namespace regor
{

static const std::unordered_set<OpType> s_noQuant = {
    OpType::ArgMax,
    OpType::MirrorPad,
    OpType::Quantize,
    OpType::Shape,
    OpType::Transpose,
    OpType::GatherNd,
    OpType::GatherV2,
    OpType::Select,
    OpType::SelectV2,
    OpType::ScatterNd,
    OpType::Pad,
    OpType::PadV2,
    OpType::ReduceAll,
    OpType::ReduceAny,
    OpType::ExpandDims,
    OpType::MemoryCopy,
};

constexpr int MAX_MEAN_KERNEL_SIZE = 64 * 64;
constexpr int MAX_MEAN_ELEMENTS_INT8 = 1 << 24;   // 2²⁴ x 2⁷  = 2³¹
constexpr int MAX_MEAN_ELEMENTS_UINT8 = 1 << 23;  // 2²³ x 2⁸  = 2³¹
constexpr int MAX_MEAN_ELEMENTS_INT16 = 1 << 16;  // 2¹⁶ x 2¹⁵ = 2³¹

void Failure(const Operation *op, const std::string &extra)
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
    if ( opType != OpType::None && opType != OpType::Passthrough && opType != OpType::MemoryCopy )
    {
        auto tfLiteType = TfLiteMapping::OpTypeToBuiltinOperator(opType);
        type = TfLiteMapping::BuiltinOperatorToString(tfLiteType);
    }
    LOG_WARN("\nWarning (supported operators) operator:{} ofm:{}\n", std::move(type), name);
    if ( extra.size() )
    {
        LOG_WARN("Reason: {}\n", extra);
    }
}

bool TfLiteSupportedOperators::Check(const Operation *op)
{
    if ( !opConstraints.contains(op->Type()) )
    {
        Failure(op, "Unsupported opType");
        return false;
    }
    for ( auto &check : opConstraints[op->Type()] )
    {
        if ( !check->Check(op) )
        {
            LOG_WARN("{}\n", check->Documentation());
            return false;
        }
    }
    return true;
}

ordered_map<OpType, std::vector<std::string>> TfLiteSupportedOperators::Documentation()
{
    ordered_map<OpType, std::vector<std::string>> documentation;
    for ( const auto &[opType, constraints] : opConstraints.pairs() )
    {
        if ( opType == OpType::None || opType == OpType::Passthrough || opType == OpType::MemoryCopy )
        {
            // some internal opTypes end up here
            // as pattern-matching runs before supported-operator checks
            // skip emitting documentation for internal opTypes
            continue;
        }
        for ( auto &check : constraints )
        {
            documentation[opType].push_back(check->Documentation());
        }
    }
    return documentation;
}

namespace
{

// Operations must have at least one IFM
bool MustHaveIFM(const Operation *op)
{
    if ( !op->Input(TensorUsage::IFM0) )
    {
        Failure(op, "Operation without IFM");
        return false;
    }
    return true;
}

// Operations must have at least one OFM
bool MustHaveOFM(const Operation *op)
{
    if ( !op->Output(TensorUsage::OFM) )
    {
        Failure(op, "Operation without OFM");
        return false;
    }
    return true;
}
// Only the first dimension of tensors can be dynamic.
bool TensDimMustBeStatic(const Operation *op)
{
    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &item : list->pairs() )
        {
            const auto &shape = item.second.shape;
            for ( int i = 0; i < shape.Size(); ++i )
            {
                if ( shape[i] < 0 )
                {
                    Failure(op,
                        fmt::format("Dynamic non-batch dimension at axis {} on tensor {}", i, item.second.tensor->Name()));
                    return false;
                }
            }
        }
    }
    return true;
}

// Tensors must have a shape
bool TensMustHaveShape(const Operation *op)
{
    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &item : list->pairs() )
        {
            auto usage = item.first;
            const auto &conn = item.second;
            if ( !conn.shape )
            {
                Failure(op, "Operation has shapeless tensor");
                return false;
            }
        }
    }
    return true;
}

// Tensors must have quantization parameters
// Does not apply to opTypes in the noQuant set
bool TensQuantized(const Operation *op)
{
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
                    Failure(op, fmt::format("Operation has tensor {} with missing quantization parameters", conn.tensor->Name()));
                    return false;
                }
            }
        }
    }
    return true;
}

// Quantization scale shift must be positive
bool QuantizationScaleShiftMustBePositive(const Operation *op)
{
    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &item : list->pairs() )
        {
            for ( const auto &scale : item.second.quantization.scales )
            {
                if ( scale.shift < 0 ) return false;
            }
        }
    }
    return true;
}

// FullyConnected constraint on weight-shape
bool FCWeightShape(const Operation *op)
{
    assert(op->Type() == OpType::FullyConnected);
    auto weights = op->Input(TensorUsage::Weights);
    assert(weights);
    assert(weights->tensor);
    const auto &shape = weights->shape;
    // Total elements must be equal to first-dim * last-dim
    if ( shape.Size() < 2 || (shape.Elements() != (shape[0] * shape[-1])) )
    {
        Failure(op, fmt::format("Unsupported weights shape: {}", shape.ToString()));
        return false;
    }
    return true;
}

// Rejects operations that don't support per-axis quantization
bool PerAxisQuant(const Operation *op)
{
    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &[usage, conn] : list->pairs() )
        {
            if ( conn.quantization.scales.size() > 1 || conn.quantization.zeroPoints.size() > 1 )
            {
                Failure(op, "Operation does not support per-axis quantization");
                return false;
            }
        }
    }
    return true;
}

// Minimum/Maximum specific
// ifm/ifm2 quantization must match ofm quantization
bool MatchingQuantization(const Operation *op)
{
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
        Failure(op, "Operation has mismatching quantization parameters.");
        return false;
    }
    return true;
}

// Weight-tensors must be 8-bit precision
bool WeightsPrecision(const Operation *op)
{
    const auto wconn = op->Input(TensorUsage::Weights);
    assert(wconn);
    const auto type = wconn->tensor->Type();
    if ( DataTypeSizeBits(type) != 8 )
    {
        Failure(op, fmt::format("Weights tensor with precision: {}", DataTypeToString(type)));
        return false;
    }
    return true;
}

// Constraints the sum of weights based on IFM precision
bool WeightSum(const Operation *op, int64_t maxWeightSum8Bit, int64_t maxWeightSum16Bit)
{
    auto wConn = op->Input(TensorUsage::Weights);
    auto ifmConn = op->Input(TensorUsage::IFM);
    assert(wConn);
    assert(ifmConn);
    if ( !wConn->tensor->IsConstant() )
    {
        return true;
    }
    auto view = wConn->tensor->View();
    auto zeroPoints = wConn->quantization.zeroPoints;
    auto ifmType = ifmConn->tensor->Type();
    int ifmBits = DataTypeSizeBits(ifmType);
    int64_t maxWeightSum = ifmBits == 8 ? maxWeightSum8Bit : maxWeightSum16Bit;

    auto reader = view.Values<int>(wConn->tensor->Type());
    Shape readShape = wConn->tensor->StorageShape();
    assert(readShape.Size() == 4);

    int outChannels;
    int inChannels;
    int64_t maxAccumulation = 0;
    bool depthwise = op->Type() == OpType::DepthwiseConv2D;
    if ( depthwise )
    {
        inChannels = 1;
        outChannels = readShape.Depth();
        maxAccumulation = int64_t(255) * readShape.ElementsWH();
    }
    else
    {
        inChannels = readShape.Depth();
        outChannels = readShape.Batch();
        maxAccumulation = int64_t(255) * readShape.Elements64() / outChannels;
    }

    // Abort early if the readShape of the weights tensor guarantees no overflow.
    if ( maxAccumulation < maxWeightSum )
    {
        return true;
    }

    // Accumulate the weights in slices of output-channels
    // Fail if any slice overflows maxWeightSum
    for ( int out = 0; out < outChannels; out++ )
    {
        int64_t sum = 0;
        int64_t zeroPoint = 0;
        if ( !zeroPoints.empty() )
        {
            zeroPoint = zeroPoints.size() > 1 ? zeroPoints[out] : zeroPoints[0];
        }
        for ( int in = 0; in < inChannels; in++ )
        {
            for ( int h = 0; h < readShape.Height(); h++ )
            {
                for ( int w = 0; w < readShape.Width(); w++ )
                {
                    int64_t v;
                    if ( depthwise )
                    {
                        v = reader[{0, h, w, out}];
                    }
                    else
                    {
                        v = reader[{out, h, w, in}];
                    }
                    sum += std::abs(v - zeroPoint);
                }
            }
        }
        if ( sum > maxWeightSum )
        {
            Failure(op, fmt::format("The absolute sum of weight-tensor elements: {} exceeds limit: {}", sum, maxWeightSum));
            return false;
        }
    }
    return true;
}

// Bias values must be stored in the channel axis
bool BiasShape(const Operation *op)
{
    auto bConn = op->Input(TensorUsage::Scales);
    if ( !bConn ) return true;
    auto bShape = bConn->shape;
    if ( bShape.Elements() > bShape.Depth() )
    {
        Failure(op, fmt::format("Bias shape: {}", bShape.ToString()));
        return false;
    }
    return true;
}

// bias tensor must be constant
bool BiasConstant(const Operation *op)
{
    auto bConn = op->Input(TensorUsage::Scales);
    if ( !bConn ) return true;
    if ( !bConn->tensor->IsConstant() )
    {
        Failure(op, "Operation has non-constant bias tensor.");
        return false;
    }
    return true;
}

// Bias tensors must be Int32 or Int64
bool BiasPrecision(const Operation *op)
{
    auto bConn = op->Input(TensorUsage::Scales);
    if ( !bConn ) return true;
    auto type = bConn->tensor->Type();
    if ( type != DataType::Int32 && type != DataType::Int64 )
    {
        Failure(op, fmt::format("Operation has bias with type:{}", DataTypeToString(type)));
        return false;
    }
    return true;
}

// Constrain max-value for 64-bit bias
bool Bias64BitRange(const Operation *op, int64_t maxBias)
{
    auto bConn = op->Input(TensorUsage::Scales);
    if ( !bConn ) return true;
    auto type = bConn->tensor->Type();
    if ( type == DataType::Int64 )
    {
        // read bias values
        auto view = bConn->tensor->View();
        auto values = view.Values<int64_t>();
        for ( int64_t bias : values )
        {
            if ( bias > maxBias )
            {
                Failure(op, fmt::format("Bias is out of range: {} > {}", bias, maxBias));
                return false;
            }
        }
    }
    return true;
}

// AvgPool padding constraints:
// IF padding=VALID
//   Kernel product must be less than 256*256 and kernel height must be less than 256
// ELSE
//   kernel width and height must be less than 8
bool AvgPoolPad(const Operation *op)
{
    auto kernel = op->Kernel();
    assert(kernel);
    auto [w, h] = kernel->Size();
    if ( kernel->Padding().IsZero() )
    {
        if ( h > 256 || h < 1 )
        {
            Failure(op, fmt::format("kernel height: {} out of range", h));
            return false;
        }
        if ( h * w > 256 * 256 )
        {
            Failure(op, fmt::format("kernel product: {} out of range", h * w));
            return false;
        }
    }
    else
    {
        // SAME padding
        if ( w > 8 || w < 1 )
        {
            // kernel width out of range
            Failure(op, fmt::format("kernel width: {} out of range", w));
            return false;
        }
        if ( h > 8 || h < 1 )
        {
            Failure(op, fmt::format("kernel height: {} out of range", h));
            return false;
        }
    }
    return true;
}

// Kernel product must be less than or equal to 256*256
// and kernel height must be less than or equal to 256
bool MaxPool(const Operation *op)
{
    auto kernel = op->Kernel();
    assert(kernel);
    auto [w, h] = kernel->Size();
    auto [sw, sh] = kernel->Stride();
    if ( h > 256 || h < 1 )
    {
        Failure(op, fmt::format("kernel height: {} out of range", h));
        return false;
    }
    if ( h * w > 256 * 256 )
    {
        Failure(op, fmt::format("kernel product: {} out of range", h * w));
        return false;
    }
    return true;
}

// Shape constraints for TransposeConv
// IF padding::SAME:
//     OFM must be equal to IFM * stride
// ELSE (padding::VALID)
//     OFM must be equal to IFM * stride + (kernel - stride)
bool TransposeConvShape(const Operation *op)
{
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
                    ifmWH.x, ifmWH.y, ofmWH.x, ofmWH.y, stride.x, stride.y));
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
                    ifmWH.x, ifmWH.y, ofmWH.x, ofmWH.y, stride.x, stride.y, kernel->Size().x, kernel->Size().y));
            return false;
        }
    }
    return true;
}

// IFM must be Int8 or Int16
bool RsqrtIFMPrecision(const Operation *op)
{
    auto ifmConn = op->Input(TensorUsage::IFM);
    assert(ifmConn);
    auto ifmType = ifmConn->tensor->Type();
    if ( ifmType != DataType::Int8 && ifmType != DataType::Int16 )
    {
        Failure(op, fmt::format("{} IFM", DataTypeToString(ifmType)));
        return false;
    }
    return true;
}

// Parameter tensors must be constant
bool ConstParams(const Operation *op)
{
    for ( const auto item : op->Inputs().pairs() )
    {
        auto usage = item.first;
        auto &conn = item.second;
        if ( IsParams(usage) && !conn.tensor->IsConstant() )
        {
            assert(conn.tensor);
            Failure(op, fmt::format("non-constant parameter-tensor {}", conn.tensor->Name()));
            return false;
        }
    }
    return true;
}

// W*H must be less than or equal to 2^16
bool SoftmaxOverflow(const Operation *op)
{
    auto ifmConn = op->Input(TensorUsage::IFM);
    assert(ifmConn);
    static constexpr int maxProd = 1 << 16;
    const auto ifmShape = Shape::PadAxes(ifmConn->shape, 4, 1);
    if ( ifmShape.ElementsWH() > maxProd )
    {
        Failure(op, fmt::format("Unsupported ifmShape: ({}), W * H = {}", ifmShape.ToString(), ifmShape.ElementsWH()));
        return false;
    }
    return true;
}

// Pad parameters must be Int32 or Int64
bool PadParams(const Operation *op)
{
    auto params = op->Input(TensorUsage::Params);
    assert(params);
    const auto &pType = params->tensor->Type();
    if ( pType != DataType::Int32 && pType != DataType::Int64 )
    {
        Failure(op, fmt::format("Unsupported params tensor with datatype: {}", DataTypeToString(pType)));
        return false;
    }
    return true;
}

// Transpose-mask can be maximum 8D
bool TransposeDims(const Operation *op)
{
    auto params = op->Input(TensorUsage::Params);
    assert(params);
    if ( params->shape.Depth() > 8 )
    {
        Failure(op, "Unsupported transpose-shape");
        return false;
    }
    return true;
}

// IFM and OFM shapes must match
bool LogShapes(const Operation *op)
{
    auto ifmConn = op->Input(TensorUsage::IFM);
    assert(ifmConn);
    auto ofmConn = op->Output(TensorUsage::OFM);
    assert(ofmConn);
    const auto &ifmShape = ifmConn->shape;
    const auto &ofmShape = ofmConn->shape;
    if ( ifmShape != ofmShape )
    {
        Failure(op, fmt::format("Mismatching shapes: IFM={}, OFM={}", ifmShape.ToString(), ofmShape.ToString()));
        return false;
    }
    return true;
}

// IFM and OFM datatypes must match.
// IFM and OFM datatypes must be Int8 or Int16
bool LogPrecision(const Operation *op)
{
    auto ifmConn = op->Input(TensorUsage::IFM);
    assert(ifmConn);
    auto ofmConn = op->Output(TensorUsage::OFM);
    assert(ofmConn);
    auto ifmType = ifmConn->tensor->Type();
    auto ofmType = ofmConn->tensor->Type();
    if ( ifmType != DataType::Int8 && ifmType != DataType::Int16 )
    {
        Failure(op, fmt::format("Unsupported IFM type: {}", DataTypeToString(ifmType)));
        return false;
    }
    if ( ifmType != ofmType )
    {
        Failure(op, fmt::format("Mismatching dataTypes: IFM={}, OFM={}", DataTypeToString(ifmType), DataTypeToString(ofmType)));
        return false;
    }
    return true;
}

// Batch must be 1.
bool UnitBatch(const Operation *op)
{
    auto ifmConn = op->Input(TensorUsage::IFM);
    assert(ifmConn);
    Shape ifmShape4D = Shape::PadAxes(ifmConn->shape, 4, 1);
    if ( ifmShape4D.Batch() > 1 )
    {
        Failure(op, fmt::format("Batch > 1: {}", ifmShape4D.ToString()));
        return false;
    }
    return true;
}

// Reduction over depth is only supported if any of h,w,c == 1
bool MeanDepth(const Operation *op)
{
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto params = op->Input(TensorUsage::Params);
    assert(ifmConn);
    assert(params);
    Shape ifmShape4D = Shape::PadAxes(ifmConn->shape, 4, 1);
    auto axisTens = params->tensor;
    auto axisCount = axisTens->StorageShape().IsEmpty() ? 1 : axisTens->StorageShape().Depth();
    auto axisValues = axisTens->View().Values<int32_t>();
    auto axisMask = ifmConn->shape.WithZeros();
    for ( int i = 0; i < axisCount; i++ )
    {
        axisMask[axisValues[i]] = 1;
    }
    axisMask = Shape::PadAxes(axisMask, 4, 0);
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
            Failure(op, fmt::format("Unsupported depth-reduction.  IFM: {}", ifmShape4D.ToString()));
            return false;
        }
    }
    return true;
}

// Reduced axis must be less than MEAN_KERNEL_SIZE
bool MeanAxisSize(const Operation *op)
{
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto params = op->Input(TensorUsage::Params);
    assert(ifmConn);
    assert(params);
    Shape ifmShape4D = Shape::PadAxes(ifmConn->shape, 4, 1);
    auto axisTens = params->tensor;
    auto axisCount = axisTens->StorageShape().IsEmpty() ? 1 : axisTens->StorageShape().Depth();
    auto axisValues = axisTens->View().Values<int32_t>();
    auto axisMask = ifmConn->shape.WithZeros();
    for ( int i = 0; i < axisCount; i++ )
    {
        axisMask[axisValues[i]] = 1;
    }
    axisMask = Shape::PadAxes(axisMask, 4, 0);
    // Reduced axes are represented with their IFM-value
    // Non reduced axes are represented by 0
    // e.g. IFM (5,8,7,9) with axis=H,C -> (0,8,0,9)
    Shape reducedAxes = ifmShape4D * axisMask;
    // Constrain kernel-size
    if ( reducedAxes.GreaterMask(Shape(nullptr, 4, MAX_MEAN_KERNEL_SIZE)) != 0 )
    {
        Failure(op, "Reduced axis is too large");
        return false;
    }
    return true;
}

// IFM datatype must be Int8, UInt8 or Int16
bool MeanDataType(const Operation *op)
{
    auto ifmConn = op->Input(TensorUsage::IFM);
    assert(ifmConn);
    auto ifmType = ifmConn->tensor->Type();
    if ( ifmType != DataType::Int8 && ifmType != DataType::UInt8 && ifmType != DataType::Int16 )
    {
        Failure(op, "Unsupported ifm type");
        return false;
    }
    return true;
}

// Constraints for mean total elements
bool MeanTotalElements(const Operation *op)
{
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto params = op->Input(TensorUsage::Params);
    assert(ifmConn);
    assert(params);
    Shape ifmShape4D = Shape::PadAxes(ifmConn->shape, 4, 1);
    auto axisTens = params->tensor;
    auto axisCount = axisTens->StorageShape().IsEmpty() ? 1 : axisTens->StorageShape().Depth();
    auto axisValues = axisTens->View().Values<int32_t>();
    auto axisMask = ifmConn->shape.WithZeros();
    for ( int i = 0; i < axisCount; i++ )
    {
        axisMask[axisValues[i]] = 1;
    }
    axisMask = Shape::PadAxes(axisMask, 4, 0);
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
                Failure(op, fmt::format("Too many reduced elements: {}", elements));
                return false;
            }
            break;
        case DataType::UInt8:
            if ( elements > MAX_MEAN_ELEMENTS_UINT8 )
            {
                Failure(op, fmt::format("Too many reduced elements: {}", elements));
                return false;
            }
            break;
        case DataType::Int16:
            if ( elements > MAX_MEAN_ELEMENTS_INT16 )
            {
                Failure(op, fmt::format("Too many reduced elements: {}", elements));
                return false;
            }
            break;
        default:
            break;
    }
    return true;
}

// Decoded slice must represent a volume
// Stride must be non-negative (and over W or H)
bool StridedSlice(const Operation *op)
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
    if ( sliceStride.WithHW(1, 1).Elements64() != 1 )
    {
        Failure(op, "StridedSlice with unsupported stride axis");
        return false;
    }
    if ( sliceStride.LessMask(sliceStride.WithZeros()) )
    {
        Failure(op, "StridedSlice with unsupported negative stride");
        return false;
    }
    if ( !sliceShape.GreaterMask(sliceShape.WithZeros()) )
    {
        Failure(op, fmt::format("StridedSlice with invalid sliceShape: {}", sliceShape.ToString()));
        return false;
    }
    return true;
}

// Implicit gate calc is not supported
bool LSTMImplicitGateCalc(const Operation *op)
{
    assert(op->Type() == OpType::UnidirectionalSequenceLstm);
    for ( int i = 0; i <= 7; i++ )
    {
        // Check that all the gate weights are present. If they are not it's either invalid or using Couple
        // Input and Forget Gate (CIFG), where the input gate is computed implicitly from the forget gate,
        // which is not supported.
        if ( op->Input(MakeTensorUsage(TensorUsage::Weights, i)) == nullptr )
        {
            Failure(op, "Missing gate weight tensor");
            return false;
        }
    }
    return true;
}

// Peephole is not supported
bool LSTMPeephole(const Operation *op)
{
    assert(op->Type() == OpType::UnidirectionalSequenceLstm);
    for ( int i = 8; i <= 10; i++ )
    {
        if ( op->Input(MakeTensorUsage(TensorUsage::Weights, i)) )
        {
            Failure(op, "Peephole weight tensor present");
            return false;
        }
    }
    return true;
}

// Projection is not supported
bool LSTMProjection(const Operation *op)
{
    assert(op->Type() == OpType::UnidirectionalSequenceLstm);
    if ( op->Input(MakeTensorUsage(TensorUsage::Weights, 11)) || op->Input(MakeTensorUsage(TensorUsage::Scales, 4)) )
    {
        Failure(op, "Projection weight or bias tensor present");
        return false;
    }
    return true;
}

// Gate normalization is not supported
bool LSTMGateNorm(const Operation *op)
{
    assert(op->Type() == OpType::UnidirectionalSequenceLstm);
    for ( int i = 5; i <= 8; i++ )
    {
        if ( op->Input(MakeTensorUsage(TensorUsage::Scales, i)) )
        {
            Failure(op, "Normalization coefficient tensor present");
            return false;
        }
    }
    return true;
}

// num_splits must match the number of outputs
bool SplitsMatchOutputs(const Operation *op)
{
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
        Failure(op, fmt::format("num_splits: {} does not match the number of outputs: {}", numSplits, numOutputs));
        return false;
    }
    return true;
}

// IFM and OFM dataTypes must be contained in supportedDataTypes
bool SupportedDTypes(const Operation *op, const std::set<DataType> &supportedDataTypes)
{
    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &item : list->pairs() )
        {
            auto usage = item.first;
            const auto &conn = item.second;
            auto type = conn.tensor->Type();
            if ( (IsIFM(usage) || IsOFM(usage)) && supportedDataTypes.count(type) == 0 )
            {
                Failure(op, fmt::format("Operation has tensor with unsupported DataType {}", DataTypeToString(type)));
                return false;
            }
        }
    }
    return true;
}

}  // namespace

TfLiteSupportedOperators::TfLiteSupportedOperators(int64_t maxWeightSum8Bit, int64_t maxWeightSum16Bit, int64_t maxBias,
    const std::set<DataType> &supportedDataTypes, const std::set<OpType> &supportedOpTypes)
{
    // Define constraint objects
    std::set<std::string> dTypes;
    for ( DataType type : supportedDataTypes )
    {
        auto tflType = TfLiteMapping::DataTypeToTensorType(type);
        dTypes.insert(TfLiteMapping::BuiltinTensorTypeToString(tflType));
    }
    supportedDTypes = {[&supportedDataTypes](const Operation *op) -> bool
        { return SupportedDTypes(op, supportedDataTypes); },
        fmt::format("Feature-map dataTypes must be one of {}", dTypes)};

    mustHaveIFM = {&MustHaveIFM, "Operations must have at least one IFM."};
    mustHaveOFM = {&MustHaveOFM, "Operations must have at least one OFM."};
    tensMustHaveShape = {&TensMustHaveShape, "Tensors must have constant shape."};
    tensDimMustBeStatic = {&TensDimMustBeStatic, "Only the first dimension of tensors can be dynamic."};
    tensQuantized = {&TensQuantized, "Input(s), Output and Weight tensors must have quantization parameters."};
    quantizationScaleShiftPositive = {&QuantizationScaleShiftMustBePositive, "Quantization scale of any tensor must be positive."};
    fcWeightShape = {&FCWeightShape, "FullyConnected weights must be on the form O,1,1,..,1,I."};
    perAxisQuant = {&PerAxisQuant, "Per-axis quantization is not supported."};
    matchingQuantization = {&MatchingQuantization, "Both Input quantization parameters must match OFM quantization parameters"};
    weightsPrecision = {&WeightsPrecision, "Weight tensors must be 8-bit precision"};
    weightSum = {[maxWeightSum8Bit, maxWeightSum16Bit](const Operation *op) -> bool
        { return WeightSum(op, maxWeightSum8Bit, maxWeightSum16Bit); },
        fmt::format(
            "Weigth sum constraints:\n"
            "  - IF 8-bit IFM:\n"
            "    - The sum of absolute weights cannot exceed {}\n"
            "  - IF 16-bit IFM:\n"
            "    - The sum of absolute weights cannot exceed {}",
            maxWeightSum8Bit, maxWeightSum16Bit)};
    biasShape = {&BiasShape, "Bias values must be stored in the channel axis"};
    biasConstant = {&BiasConstant, "Bias tensors must be constant"};
    biasPrecision = {&BiasPrecision, "Bias tensors precision must be Int32 or Int64"};
    bias64BitRange = {[maxBias](const Operation *op) -> bool { return Bias64BitRange(op, maxBias); },
        fmt::format("Int64 bias must be smaller than {}", maxBias)};
    avgPoolPad = {&AvgPoolPad,
        "Kernel constraints:\n"
        "  - IF padding=VALID:\n"
        "    - Kernel product must be less than or equal to 256*256\n"
        "    - Kernel height must be less than or equal to 256.\n"
        "  - ELSE:\n"
        "    - Kernel width and height must be less than or equal to 8"};
    maxPool = {&MaxPool, "kernel product must be less than or equal to 256*256 and kernel height must be less than or equal to 256."};
    transposeConvShape = {&TransposeConvShape,
        "Shape constraints:\n"
        "  - IF PADDING=SAME:\n"
        "    - OFM must be equal to IFM * stride\n"
        "  - ELSE:\n"
        "    - OFM must be equal to IFM * stride + (kernel - stride)"};
    rsqrtIFMPrecision = {&RsqrtIFMPrecision, "IFM must be Int8 or Int16"};
    constParams = {&ConstParams, "Parameter tensors must be constant"};
    softmaxOverflow = {&SoftmaxOverflow, "The product of IFM width and height must be less than 65536"};
    padParams = {&PadParams, "Params tensor must be Int32 or Int64"};
    transposeDims = {&TransposeDims, "Tensor dimension must be <= 8"};
    logShapes = {&LogShapes, "IFM and OFM shapes must match"};
    logPrecision = {&LogPrecision, "IFM and OFM datatypes must match, and must be Int8 or Int16"};
    unitBatch = {&UnitBatch, "Batch must be 1."};
    meanDepth = {&MeanDepth, "Reduction over depth is only supported if any of h,w,c == 1"};
    meanAxisSize = {&MeanAxisSize, fmt::format("Reduced axis must be less than {}", MAX_MEAN_KERNEL_SIZE)};
    meanTotalElements = {&MeanTotalElements,
        fmt::format("Amount of reduced elements cannot exceed: {}(Int8), {}(UInt8), {}(Int16).", MAX_MEAN_ELEMENTS_INT8,
            MAX_MEAN_ELEMENTS_UINT8, MAX_MEAN_ELEMENTS_INT16)};
    meanDataType = {&MeanDataType, "IFM precision must be one of: (Int8, UInt8, Int16)"};
    stridedSlice = {&StridedSlice, "Slice must represent a volume. Stride must be non-negative (and over W or H)."};
    lstmImplicitGateCalc = {&LSTMImplicitGateCalc, "Implicit gate calculation is not supported."};
    lstmPeephole = {&LSTMPeephole, "Peephole variant is not supported."};
    lstmProjection = {&LSTMProjection, "Projection is not supported."};
    lstmGateNorm = {&LSTMGateNorm, "Gate normalization is not supported."};
    splitsMatchOutputs = {&SplitsMatchOutputs, "num_splits must match the number of outputs."};

    // Map opTypes to constraint objects
    for ( OpType type : supportedOpTypes )
    {
        // generic constraints
        opConstraints[type].push_back(&supportedDTypes);
        opConstraints[type].push_back(&mustHaveIFM);
        opConstraints[type].push_back(&mustHaveOFM);
        opConstraints[type].push_back(&tensMustHaveShape);
        opConstraints[type].push_back(&tensDimMustBeStatic);
        // quantization constraints
        if ( s_noQuant.find(type) == std::end(s_noQuant) )
        {
            opConstraints[type].push_back(&tensQuantized);
            opConstraints[type].push_back(&quantizationScaleShiftPositive);
        }
        if ( type != OpType::FullyConnected && !IsConvolution(type) )
        {
            opConstraints[type].push_back(&perAxisQuant);
        }
        // weight/bias constraints
        if ( type == OpType::FullyConnected || IsConvolution(type) )
        {
            opConstraints[type].push_back(&weightsPrecision);
            opConstraints[type].push_back(&weightSum);
            opConstraints[type].push_back(&biasShape);
            opConstraints[type].push_back(&biasConstant);
            opConstraints[type].push_back(&biasPrecision);
            opConstraints[type].push_back(&bias64BitRange);
        }
    }
    // op-specific constraints
    opConstraints[OpType::Slice].push_back(&constParams);
    opConstraints[OpType::StridedSlice].push_back(&constParams);
    opConstraints[OpType::Mean].push_back(&constParams);
    opConstraints[OpType::Pad].push_back(&constParams);
    opConstraints[OpType::PadV2].push_back(&constParams);
    opConstraints[OpType::MirrorPad].push_back(&constParams);
    opConstraints[OpType::Transpose].push_back(&constParams);
    opConstraints[OpType::Minimum].push_back(&matchingQuantization);
    opConstraints[OpType::Maximum].push_back(&matchingQuantization);
    opConstraints[OpType::FullyConnected].push_back(&fcWeightShape);
    opConstraints[OpType::AvgPool].push_back(&avgPoolPad);
    opConstraints[OpType::MaxPool].push_back(&maxPool);
    opConstraints[OpType::TransposeConv2D].push_back(&transposeConvShape);
    opConstraints[OpType::Rsqrt].push_back(&rsqrtIFMPrecision);
    opConstraints[OpType::Softmax].push_back(&softmaxOverflow);
    opConstraints[OpType::Transpose].push_back(&transposeDims);
    opConstraints[OpType::Pad].push_back(&padParams);
    opConstraints[OpType::PadV2].push_back(&padParams);
    opConstraints[OpType::MirrorPad].push_back(&padParams);
    opConstraints[OpType::Log].push_back(&logShapes);
    opConstraints[OpType::Log].push_back(&logPrecision);
    opConstraints[OpType::Mean].push_back(&unitBatch);
    opConstraints[OpType::Mean].push_back(&meanDepth);
    opConstraints[OpType::Mean].push_back(&meanAxisSize);
    opConstraints[OpType::Mean].push_back(&meanDataType);
    opConstraints[OpType::Mean].push_back(&meanTotalElements);
    opConstraints[OpType::StridedSlice].push_back(&stridedSlice);
    opConstraints[OpType::UnidirectionalSequenceLstm].push_back(&lstmImplicitGateCalc);
    opConstraints[OpType::UnidirectionalSequenceLstm].push_back(&lstmPeephole);
    opConstraints[OpType::UnidirectionalSequenceLstm].push_back(&lstmProjection);
    opConstraints[OpType::UnidirectionalSequenceLstm].push_back(&lstmGateNorm);
    opConstraints[OpType::Split].push_back(&splitsMatchOutputs);
    opConstraints[OpType::SplitV].push_back(&splitsMatchOutputs);
};

std::unique_ptr<TfLiteSupportedOperators> MakeSupportedOpsChecker(const std::string &target)
{
    if ( target == REGOR_ARCH_ETHOSU85 )
    {
        return std::make_unique<TfLiteSupportedOperatorsU85>();
    }
    else
    {
        assert(target == REGOR_ARCH_ETHOSU55 || target == REGOR_ARCH_ETHOSU65);
        return std::make_unique<TfLiteSupportedOperatorsU55>();
    }
}

}  // namespace regor
