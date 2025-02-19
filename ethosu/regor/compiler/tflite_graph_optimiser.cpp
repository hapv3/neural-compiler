//
// SPDX-FileCopyrightText: Copyright 2021-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "compiler/tflite_graph_optimiser.hpp"

#include "common/logging.hpp"

#include "architecture/architecture.hpp"
#include "architecture/architecture_constraints.hpp"
#include "common/scaling.hpp"
#include "common/transpose_type.hpp"
#include "graph.hpp"
#include "graph_optimiser.hpp"
#include "op_type.hpp"
#include "operation.hpp"
#include "optimiser_utils.hpp"
#include "softmax.hpp"
#include "tensor.hpp"
#include "tflite/tflite_schema_generated.hpp"

#include <fixedpoint/fixedpoint.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <unordered_set>
#include <vector>

namespace regor
{

using namespace GraphOptimisation;

// Multiplies int with QuantizedScale with rounding.
int TFLiteGraphOptimiser::MultiplyByQuantizedMultiplier(int x, QuantizedScale quantScale)
{
    // Multiplies x (int32) by QuantizedScale (scale, shift), returns rounded result.
    // Expects the QuantizedScale to be left-shift positive.
    const int leftShift = quantScale.shift > 0 ? quantScale.shift : 0;
    const int rightShift = quantScale.shift < 0 ? -quantScale.shift : 0;
    const std::int32_t mul = gemmlowp::SaturatingRoundingDoublingHighMul(x * (1 << leftShift), quantScale.scale);
    return gemmlowp::RoundingDivideByPOT<std::int32_t>(mul, rightShift);
}

Operation *TFLiteGraphOptimiser::MakeMulWithConstTensor(const std::string &name, const TensorConnection &ifmConn,
    const TensorConnection &ofmConn, const std::shared_ptr<Tensor> &constTens, const Quantization &quantization)
{
    auto ofm = ofmConn.tensor;
    auto op = std::make_shared<Operation>(OpType::Mul);

    op->CopyInput(TensorUsage::IFM0, ifmConn);
    op->ConnectInput(TensorUsage::IFM1, constTens).Set(quantization);

    auto ofmName = ofm->Name();
    ofmName.append("_");
    ofmName.append(name);

    std::shared_ptr<Tensor> cloneOfm = ofm->Clone();
    cloneOfm->SetName(ofmName);
    op->ConnectOutput(TensorUsage::OFM, cloneOfm).Set(ofmConn.shape).Set(ofmConn.quantization).Set(ofmConn.slice).Set(RoundMode::DBL);

    return op.get();
}

Operation *TFLiteGraphOptimiser::MakeOperation(
    OpType opType, const TensorConnection *ifm0Conn, const TensorConnection *ifm1Conn, const TensorConnection *ofmConn)
{
    auto op = std::make_shared<Operation>(opType);
    assert(ifm0Conn != nullptr);
    assert(ofmConn != nullptr);
    op->CopyInput(TensorUsage::IFM0, *ifm0Conn);
    op->CopyOutput(TensorUsage::OFM, *ofmConn);
    if ( ifm1Conn != nullptr )
    {
        op->CopyInput(TensorUsage::IFM1, *ifm1Conn);
    }
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    return op.get();
}

// Converts LeakyReLU to
// if 0 <= alpha <= 1
//     Maximum(alpha * IFM, identity * IFM)
// else
//     Relu(IFM)   Minimum(IFM, 0)
//        \         /
//         \    Mul(alpha)
//          \     /
//            Add
Operation *TFLiteGraphOptimiser::ConvertLeakyRelu16bit(TensorConnection &ifmConn, TensorConnection &ofmConn, Operation *operation)
{
    Operation *returnOp = operation;

    auto ifm = ifmConn.tensor.get();
    auto ofm = ofmConn.tensor.get();
    auto params = operation->Input(TensorUsage::Params);
    auto *attr = operation->Attribute<leaky_relu_attr_t>();
    float alpha = attr->alpha;
    int64_t scalar = 1;
    auto alphaQuant = ifmConn.quantization;
    alphaQuant.quantMin = {0};
    alphaQuant.quantMax = {int64_t(alpha * IntegerMax(ifmConn.tensor->Type()))};
    alphaQuant.zeroPoints[0] = 0;
    alphaQuant.scales[0] = QuantizedScale(alpha);

    if ( alpha < 0 )
    {
        // For negative alpha we move the sign to the scalar instead.
        scalar = -1;
        alphaQuant.scales[0].scale *= -1;
    }

    if ( params != nullptr )
    {
        // If alpha comes in a params-tensor (e.g. converted PReLU)
        // the alpha-value also has quantization-parameters
        assert(params->tensor->IsConstant());
        assert(params->tensor->Type() == DataType::Int16);
        assert(params->quantization.zeroPoints.size() > 0);
        auto view = params->tensor->View();
        // Set scalar and alphaQuant accordingly
        scalar = int64_t(view.Values<int16_t>()[0]) - params->quantization.zeroPoints[0];
        alphaQuant = params->quantization;
    }

    if ( alpha >= 0 && alpha <= 1 )
    {
        // Lower to:
        //     Maximum(alpha * IFM, identity * IFM)
        auto fmAlpha = CreateConstTensor("lrelu_alpha", int16_t(scalar));
        auto alphaMulOp = MakeMulWithConstTensor("alpha", ifmConn, ofmConn, fmAlpha, alphaQuant);
        RecordOptimisation(operation, alphaMulOp);

        TensorConnection *identityConn = &ifmConn;
        if ( !IsScalingValidAndEqual(ifmConn, ofmConn) )
        {
            // Identity operation is introduced to handle rescaling of the IFM
            auto identityQuant = ifmConn.quantization;
            identityQuant.quantMin = {0};
            identityQuant.quantMax = {int64_t(IntegerMax(ifmConn.tensor->Type()))};
            identityQuant.zeroPoints[0] = 0;
            identityQuant.scales[0] = {1, 0};
            auto fmIdentity = CreateConstTensor("lrelu_ident", int16_t(1));
            auto identityMulOp = MakeMulWithConstTensor("identity", ifmConn, ofmConn, fmIdentity, identityQuant);
            RecordOptimisation(operation, identityMulOp);
            identityConn = identityMulOp->Output(TensorUsage::OFM);
        }

        // Merge scaled and unscaled values with a Maximum
        // Maximum(negative * alpha, negative) = negative * alpha
        // Maximum(positive * alpha, positive) = positive
        auto maxOp = MakeOperation(OpType::Maximum, alphaMulOp->Output(TensorUsage::OFM), identityConn, &ofmConn);
        maxOp->Input(TensorUsage::IFM)->Set(ofmConn.quantization);
        RecordOptimisation(operation, maxOp);
        returnOp = maxOp;
    }
    else
    {
        // Lower to:
        //     Relu(IFM)   Minimum(IFM, 0)
        //        \         /
        //         \    Mul(alpha)
        //          \     /
        //            Add

        // Create Minimum(IFM, 0)
        std::shared_ptr<Tensor> zeroTens = CreateConstTensor("zero_const", ifmConn.tensor->Type(), 0);
        std::shared_ptr<Tensor> fmNegative = ifmConn.tensor->Clone();
        auto minOp = std::make_shared<Operation>(OpType::Minimum);
        minOp->CopyInput(TensorUsage::IFM0, ifmConn);
        minOp->ConnectInput(TensorUsage::IFM1, zeroTens).Set(ifmConn.quantization);
        minOp->ConnectOutput(TensorUsage::OFM, fmNegative).Set(ifmConn.quantization).Set(RoundMode::DBL);
        RecordOptimisation(operation, minOp.get());

        // create Mul(alpha)
        auto fmAlpha = CreateConstTensor("lrelu_alpha", int16_t(scalar));
        auto alphaMulOp = MakeMulWithConstTensor("alpha", *minOp->Output(TensorUsage::OFM), ofmConn, fmAlpha, alphaQuant);
        RecordOptimisation(operation, alphaMulOp);

        // create ReLU(IFM) to Select (and scale) values > 0
        std::shared_ptr<Tensor> fmScaled = ofmConn.tensor->Clone();
        auto reluOp = std::make_shared<Operation>(OpType::Relu);
        reluOp->CopyInput(TensorUsage::IFM0, ifmConn);
        reluOp->ConnectOutput(TensorUsage::OFM, fmScaled).Set(ofmConn.quantization);
        reluOp->Output(TensorUsage::OFM)->quantization.quantMin.push_back(ofmConn.quantization.zeroPoints[0]);
        reluOp->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
        RecordOptimisation(operation, reluOp.get());

        // Create Add(Relu, Mul) to add scaled and alpha-multiplied values
        auto addOp = std::make_shared<Operation>(OpType::Add);
        addOp->CopyInput(TensorUsage::IFM0, *reluOp->Output(TensorUsage::OFM));
        addOp->CopyInput(TensorUsage::IFM1, *alphaMulOp->Output(TensorUsage::OFM));
        addOp->CopyOutput(TensorUsage::OFM, ofmConn);
        RecordOptimisation(operation, addOp.get());
        returnOp = addOp.get();
    }
    return returnOp;
}


// Get axis parameter for operator
int TFLiteGraphOptimiser::GetAxis(const Operation *const operation)
{
    auto opType = operation->Type();
    int axis = 0;

    switch ( opType )
    {
        case OpType::Pack:
        case OpType::Unpack:
            axis = operation->Attribute<axis_attr_t>()->axis;
            break;
        case OpType::Split:
        {
            auto *paramConn = operation->Input(TensorUsage::Params);
            axis = paramConn->tensor->View().Values<int>()[0];
            break;
        }
        case OpType::SplitV:
        {
            auto usage = MakeTensorUsage(TensorUsage::Params, 1);
            auto *paramConn = operation->Input(usage);
            axis = paramConn->tensor->View().Values<int>()[0];
            break;
        }
        default:
            break;
    }
    return axis;
}


// Creates MemoryCopy operation for the given ifm/ofm and write offset.
std::shared_ptr<Operation> TFLiteGraphOptimiser::MakeMemoryCopyForConcat(
    const TensorConnection *const ofmConn, const TensorConnection *const ifmConn, const Shape &writeOffset)
{
    auto op = std::make_shared<Operation>(OpType::MemoryCopy);

    op->CopyInput(TensorUsage::IFM0, *ifmConn);
    op->ConnectOutput(TensorUsage::OFM, ofmConn->tensor)
        .Set(ofmConn->shape)
        .Set(ofmConn->quantization)
        .Set({writeOffset, ifmConn->shape})
        .Set(RoundMode::NATURAL);

    return op;
}


Operation *TFLiteGraphOptimiser::MakeDepthwiseMeanOp(const TensorConnection *ifmConn, const Shape &ifmShape4D, const Shape &readShape,
    const Shape &readOffset, const Shape &ofmShape4D, int w, int h, const std::string &name, std::shared_ptr<Tensor> &weightTensor,
    std::shared_ptr<Tensor> biasTensor, const Quantization &ifmQuant, const Quantization &weightQuant, const Quantization &ofmQuant)
{
    auto ifm = ifmConn->tensor;
    auto op = std::make_shared<Operation>(OpType::DepthwiseConv2D);
    op->SetKernel(std::make_unique<Kernel>(Point2i(w, h), Point2i(1, 1), Point2i(1, 1)));

    if ( weightTensor == nullptr )
    {
        Shape weightShape(ifmShape4D.Batch(), h, w, ifmShape4D.Depth());
        std::vector<uint8_t> ones(weightShape.Elements(), 1);
        auto onesBuf = std::make_shared<Buffer>(std::move(ones));
        weightTensor = std::make_shared<Tensor>(name + "_weights", DataType::UInt8, weightShape, onesBuf);
        weightTensor->SetAxisOrder(AxisOrder::IHWO);
    }

    if ( biasTensor == nullptr )
    {
        DataType biasType;
        std::shared_ptr<Buffer> buf;
        auto elems = ifmShape4D.Depth();
        if ( ifm->Type() == DataType::Int16 )
        {
            biasType = DataType::Int64;
            std::vector<int64_t> data(ToUnsigned(elems));
            buf = std::make_shared<Buffer>(std::move(data));
        }
        else
        {
            biasType = DataType::Int32;
            std::vector<int32_t> data(ToUnsigned(elems));
            buf = std::make_shared<Buffer>(std::move(data));
        }
        biasTensor = std::make_shared<Tensor>(name + "bias", biasType, Shape(ifmShape4D.Depth()), buf);
    }

    auto ifmQuantZp0 = ifmQuant;
    ifmQuantZp0.zeroPoints.clear();
    ifmQuantZp0.zeroPoints.push_back(0);
    op->ConnectInput(TensorUsage::IFM, ifm).Set(ifmShape4D).Set(ifmQuant).Set({readOffset, readShape});
    op->ConnectInput(TensorUsage::Weights, weightTensor).Set(weightQuant);
    op->ConnectInput(TensorUsage::Scales, biasTensor).Set(ifmQuantZp0);

    auto ofm = std::make_shared<Tensor>(name + "_intermediate", DataType::Int32);
    ofm->SetStorageShape(ofmShape4D);
    op->ConnectOutput(TensorUsage::OFM, ofm).Set(ofmQuant);

    auto rounding = ifm->Type() == DataType::Int16 ? RoundMode::NATURAL : RoundMode::DBL;
    op->Output(TensorUsage::OFM)->Set(rounding);

    return op.get();
}


// Upcast to int32
Operation *TFLiteGraphOptimiser::CreateCastToInt32(const TensorConnection *ifmConn)
{
    assert(ifmConn->tensor->Type() != DataType::Int32);

    auto noScaleQuantZp0 = ifmConn->quantization;
    noScaleQuantZp0.scales.clear();
    noScaleQuantZp0.zeroPoints.clear();
    noScaleQuantZp0.zeroPoints.push_back(0);

    auto ofmShape4D = Shape::PadAxes(ifmConn->shape, 4, 1);
    auto op = std::make_shared<Operation>(OpType::MemoryCopy);
    op->CopyInput(TensorUsage::IFM0, *ifmConn);
    auto ofm = std::make_shared<Tensor>(ifmConn->tensor->Name() + "_32bit", DataType::Int32);
    ofm->SetStorageShape(ofmShape4D);
    op->ConnectOutput(TensorUsage::OFM, ofm).Set(noScaleQuantZp0).Set(RoundMode::NATURAL);
    return op.get();
}


// Converts op to int8/uint8 LUT which is generated with the given function.
template<typename FUNC>
static Operation *ConvertToLUT8(Operation *op, FUNC func, const std::string &name)
{
    auto ifmConn = op->Input(TensorUsage::IFM0);
    auto ofmConn = op->Output(TensorUsage::OFM);
    auto ifm = ifmConn->tensor;
    auto ofm = ofmConn->tensor;

    if ( (ifm->Type() != DataType::Int8 && ifm->Type() != DataType::UInt8) || ifm->Type() != ofm->Type() )
    {
        return op;
    }

    // Generate LUT
    double ifmScale(ifmConn->quantization.scales[0].Dequantize());
    double ofmScale(ofmConn->quantization.scales[0].Dequantize());
    auto zpIn = ifmConn->quantization.zeroPoints[0];
    auto zpOut = ofmConn->quantization.zeroPoints[0];
    int qMin = ifm->Type() == DataType::Int8 ? -128 : 0;
    int qMax = ifm->Type() == DataType::Int8 ? 127 : 255;

    std::vector<uint8_t> lut;
    lut.reserve(256);
    for ( int x = qMin; x <= qMax; ++x )
    {
        auto xReal = ifmScale * double(x - zpIn);
        auto yReal = func(xReal);
        int lutVal = int(std::round(double(zpOut) + yReal / ofmScale));
        lutVal = std::min(qMax, std::max(qMin, lutVal));
        lut.push_back(uint8_t(lutVal));
    }
    auto lutTens = CreateConstTensor(name, ifmConn->tensor->Type(), std::make_shared<Buffer>(std::move(lut)));
    // The LUT must be applied without any preceding rescaling (the LUT itself performs the rescale),
    // so even if the OFM has a different scale than the IFM, the generated OFM scale instructions
    // should be the same as the IFM
    auto returnOp = CreateLUT(ifmConn->tensor, lutTens, ifmConn->quantization, ifmConn->quantization, lutTens->Type(),
        &ifmConn->shape, ofmConn->tensor, ifmConn->slice, ofmConn->slice);
    returnOp->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
    return returnOp;
}

// Converts op to int16 interpolating LUT which is generated with the given function.
template<typename FUNC>
static Operation *ConvertToInterpolatingLUT16(Operation *op, FUNC func, const std::string &name)
{
    auto ifmConn = op->Input(TensorUsage::IFM0);
    auto ofmConn = op->Output(TensorUsage::OFM);
    auto ifm = ifmConn->tensor;
    auto ofm = ofmConn->tensor;

    if ( (ifm->Type() != DataType::Int16) || ifm->Type() != ofm->Type() )
    {
        return op;
    }

    float ifmScale = float(ifmConn->quantization.scales[0].Dequantize());
    float ofmScale = float(ofmConn->quantization.scales[0].Dequantize());
    auto zpIn = ifmConn->quantization.zeroPoints[0];
    auto zpOut = ofmConn->quantization.zeroPoints[0];
    float qMin = std::numeric_limits<int16_t>::min();
    float qMax = std::numeric_limits<int16_t>::max();
    float inputMin = ifmScale * (qMin - zpIn);
    float inputMax = ifmScale * (qMax - zpIn);
    float outputMin = ofmScale * (qMin - zpOut);
    float outputMax = ofmScale * (qMax - zpOut);
    const int steps = 512;
    float step = (inputMax - inputMin) / steps;
    float halfStep = step / 2.0f;
    float outputScalingInv = (qMax - qMin + 1) / (outputMax - outputMin);

    // Create 32-bit LUT represented by a 16-bit base and 16-bit slope.
    auto lut = std::make_unique<uint32_t[]>(512);
    float prevLutResult = 0;
    for ( int i = 0; i < steps; i++ )
    {
        float val = func(inputMin + i * step);
        float valMidpoint = func(inputMin + i * step + halfStep);
        float valNext = func(inputMin + (i + 1) * step);
        float sampleVal = std::round(val * outputScalingInv);

        float midpointInterpVal = std::round((valNext * outputScalingInv + sampleVal) / 2.0f);
        float midpointVal = std::round(valMidpoint * outputScalingInv);
        float midpointErr = midpointInterpVal - midpointVal;
        float bias = std::round(midpointErr / 2.0f);

        float lutResult = std::clamp(sampleVal - bias, qMin, qMax);

        if ( i > 0 )
        {
            uint32_t base = uint32_t(prevLutResult);
            uint32_t slope = uint32_t(lutResult - prevLutResult);
            lut[i - 1] = base + (slope << 16);
        }
        prevLutResult = lutResult;
    }
    float val = float(std::round(func(inputMax) * outputScalingInv));
    float lutResult = std::clamp(val, qMin, qMax);
    uint32_t base = uint32_t(prevLutResult);
    uint32_t slope = uint32_t(lutResult - prevLutResult);
    lut[steps - 1] = base + (slope << 16);

    auto lutTens = CreateConstTensor(name, DataType::Int32, std::make_shared<Buffer>(std::move(lut), 512));
    // The LUT must be applied without any preceding rescaling (the LUT itself performs the rescale),
    // so even if the OFM has a different scale than the IFM, the generated OFM scale instructions
    // should be the same as the IFM
    auto returnOp = CreateLUT(ifmConn->tensor, lutTens, ifmConn->quantization, ifmConn->quantization, lutTens->Type(),
        &ifmConn->shape, ofmConn->tensor, ifmConn->slice, ofmConn->slice);
    returnOp->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
    return returnOp;
}

Operation *TFLiteGraphOptimiser::ConvertTanhSigmoidToLUT16(Operation *const op)
{
    auto ifmConn = op->Input(TensorUsage::IFM0);
    auto ofmConn = op->Output(TensorUsage::OFM);
    auto ifm = ifmConn->tensor;
    auto ofm = ofmConn->tensor;

    if ( ifm->Type() != DataType::Int16 || ifm->Type() != ofm->Type() )
    {
        return op;
    }

    // clang-format off
    // Table of sigmoid(i/24)*65536
    static const uint16_t SIGMOID_TABLE[256] =
    {
        32768, 33451, 34133, 34813, 35493, 36169, 36843, 37513,
        38180, 38841, 39498, 40149, 40794, 41432, 42064, 42688,
        43304, 43912, 44511, 45102, 45683, 46255, 46817, 47369,
        47911, 48443, 48964, 49475, 49975, 50464, 50942, 51409,
        51865, 52311, 52745, 53169, 53581, 53983, 54374, 54755,
        55125, 55485, 55834, 56174, 56503, 56823, 57133, 57433,
        57724, 58007, 58280, 58544, 58800, 59048, 59288, 59519,
        59743, 59959, 60168, 60370, 60565, 60753, 60935, 61110,
        61279, 61441, 61599, 61750, 61896, 62036, 62172, 62302,
        62428, 62549, 62666, 62778, 62886, 62990, 63090, 63186,
        63279, 63368, 63454, 63536, 63615, 63691, 63765, 63835,
        63903, 63968, 64030, 64090, 64148, 64204, 64257, 64308,
        64357, 64405, 64450, 64494, 64536, 64576, 64614, 64652,
        64687, 64721, 64754, 64786, 64816, 64845, 64873, 64900,
        64926, 64950, 64974, 64997, 65019, 65039, 65060, 65079,
        65097, 65115, 65132, 65149, 65164, 65179, 65194, 65208,
        65221, 65234, 65246, 65258, 65269, 65280, 65291, 65301,
        65310, 65319, 65328, 65337, 65345, 65352, 65360, 65367,
        65374, 65381, 65387, 65393, 65399, 65404, 65410, 65415,
        65420, 65425, 65429, 65433, 65438, 65442, 65445, 65449,
        65453, 65456, 65459, 65462, 65465, 65468, 65471, 65474,
        65476, 65479, 65481, 65483, 65485, 65488, 65489, 65491,
        65493, 65495, 65497, 65498, 65500, 65501, 65503, 65504,
        65505, 65507, 65508, 65509, 65510, 65511, 65512, 65513,
        65514, 65515, 65516, 65517, 65517, 65518, 65519, 65520,
        65520, 65521, 65522, 65522, 65523, 65523, 65524, 65524,
        65525, 65525, 65526, 65526, 65526, 65527, 65527, 65528,
        65528, 65528, 65529, 65529, 65529, 65529, 65530, 65530,
        65530, 65530, 65531, 65531, 65531, 65531, 65531, 65532,
        65532, 65532, 65532, 65532, 65532, 65533, 65533, 65533,
        65533, 65533, 65533, 65533, 65533, 65534, 65534, 65534,
        65534, 65534, 65534, 65534, 65534, 65534, 65534, 65535
        // clang-format on
    };

    auto lut = std::make_unique<uint32_t[]>(512);
    for ( int i = -256; i < 256; ++i )
    {
        int j0, j1, v0, v1;
        if ( i >= 0 )
        {
            j0 = i;
            j1 = i == 255 ? 255 : i + 1;
            v0 = SIGMOID_TABLE[j0] - 0x8000;
            v1 = SIGMOID_TABLE[j1] - 0x8000;
        }
        else
        {
            j0 = i == -256 ? 255 : -i;
            if ( op->Type() == OpType::Sigmoid )
            {
                j1 = j0 - 1;
            }
            else
            {
                j1 = i == -256 ? 255 : j0 - 1;
            }

            v0 = 0x8000 - SIGMOID_TABLE[j0];
            v1 = 0x8000 - SIGMOID_TABLE[j1];
        }

        uint32_t base = v0 & 0xffff;

        uint32_t slope = 0;
        if ( v1 - v0 > 0 ) slope = v1 - v0;

        lut[256 + i] = (slope << 16) | (base);
    }

    auto lutTens = CreateConstTensor("LUT", ifmConn->tensor->Type(), std::make_shared<Buffer>(std::move(lut), 512));
    op->ConnectInput(TensorUsage::LUT, lutTens);
    return op;
}


// Rewrite functions

// Convert EXP operations to LUT
Operation *TFLiteGraphOptimiser::ConvertExpToLUT(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    OpType type = operation->Type();
    if ( type != OpType::Exp )
    {
        return returnOp;
    }
    const auto &ifmConn = operation->Input(TensorUsage::IFM0);
    DataType ifmType = ifmConn->tensor->Type();
    if ( (ifmType & DataType::Bits8) == DataType::Bits8 )
    {
        returnOp = ConvertToLUT8(
            operation, [](double x) -> float { return expf(float(x)); }, "Exp");
        RecordOptimisation(operation, returnOp);
        operation->Disconnect();
    }
    else if ( ifmType == DataType::Int16 )
    {
        returnOp = ConvertToInterpolatingLUT16(
            operation, [](double x) -> float { return expf(float(x)); }, "Exp16(interp)");
        RecordOptimisation(operation, returnOp);
        operation->Disconnect();
    }
    return returnOp;
}

// Convert TFLite Pack into TOSA Concat
Operation *TFLiteGraphOptimiser::RewritePack(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType == OpType::Pack )
    {
        auto *ofmConn = operation->Output(TensorUsage::OFM);
        const auto axis = GetAxis(operation);

        // Create a new CONCAT op
        auto concatOp = std::make_shared<Operation>(OpType::Concat);
        concatOp->CopyOutput(TensorUsage::OFM, *ofmConn);
        concatOp->Attribute<axis_attr_t>()->axis = axis;
        for ( auto [usage, ifmConn] : operation->Inputs().pairs() )
        {
            if ( !IsIFM(usage) ) continue;

            concatOp->CopyInput(usage, ifmConn);
            concatOp->Input(usage)->shape = ifmConn.shape.Insert(axis, 1);
        }
        returnOp = concatOp.get();
        operation->Disconnect();
    }
    return returnOp;
}

// Convert TFLite Slice into TOSA Slice
Operation *TFLiteGraphOptimiser::RewriteSlice(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType == OpType::Slice )
    {
        const auto *ifmConn = operation->Input(TensorUsage::IFM);
        const auto *ofmConn = operation->Output(TensorUsage::OFM);
        const auto *beginParamConn = operation->Input(TensorUsage::Params0);
        const auto *sizeParamConn = operation->Input(TensorUsage::Params1);

        // Skip this op if missing param tensors
        if ( !beginParamConn || !sizeParamConn )
        {
            return returnOp;
        }

        // Convert param tensors to attributes
        Shape sliceOffset = TensorToShape(beginParamConn->tensor.get(), beginParamConn->shape.Elements());
        Shape sliceShape = TensorToShape(sizeParamConn->tensor.get(), sizeParamConn->shape.Elements());
        for ( int i = 0; i < sliceShape.Size(); i++ )
        {
            // Fixup elements that are -1
            if ( sliceShape[i] == -1 ) sliceShape[i] = ifmConn->shape[i] - sliceOffset[i];
        }
        auto *attr = operation->Attribute<slice_attr_t>();
        assert(sliceOffset + sliceShape <= ifmConn->shape);
        assert(sliceOffset >= ifmConn->shape.WithZeros());
        assert(sliceShape == ofmConn->shape);
        attr->size = sliceShape;
        attr->begin = sliceOffset;
    }
    return returnOp;
}


// Convert TFLite StridedSlice into TOSA Slice
Operation *TFLiteGraphOptimiser::RewriteStridedSlice(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    auto *returnOp = operation;
    const auto opType = operation->Type();
    if ( opType == OpType::StridedSlice )
    {
        const auto *ifmConn = operation->Input(TensorUsage::IFM);
        const auto *ofmConn = operation->Output(TensorUsage::OFM);
        const auto *beginParmConn = operation->Input(TensorUsage::Params0);
        const auto *endParamConn = operation->Input(TensorUsage::Params1);
        const auto *stridesParamConn = operation->Input(TensorUsage::Params2);

        // Read StridedSlice attributes
        int32_t begin_mask = 0;
        int32_t ellipsis_mask = 0;
        int32_t end_mask = 0;
        int32_t new_axis_mask = 0;
        int32_t shrink_axis_mask = 0;
        const tflite::Operator *const passthrough = static_cast<const tflite::Operator *>(operation->Passthrough());
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

        // TODO MLBEDSW-10165: Handle stride != 1
        if ( sliceStride != sliceStride.WithOnes() )
        {
            returnOp->SetPassthroughOp();
            return returnOp;
        }

        // Adjust resulting shape for stride
        sliceShape = Shape::DivRoundUp(sliceShape, sliceStride);

        // Create a new SLICE op
        auto sliceOp = std::make_shared<Operation>(OpType::Slice);
        sliceOp->CopyInput(TensorUsage::IFM, *ifmConn);
        sliceOp->CopyOutput(TensorUsage::OFM, *ofmConn);
        sliceOp->Output(TensorUsage::OFM)->Set(sliceShape);
        auto *attr = sliceOp->Attribute<slice_attr_t>();
        assert(sliceOffset + sliceShape <= ifmConn->shape);
        assert(sliceOffset >= ifmConn->shape.WithZeros());
        attr->size = sliceShape;
        attr->begin = sliceOffset;
        RecordOptimisation(operation, sliceOp.get());
        returnOp = sliceOp.get();

        // Remove original op
        operation->Disconnect();
    }
    return returnOp;
}


// Convert TFLite Unpack/Split/SplitV into one or more TOSA Slice
Operation *TFLiteGraphOptimiser::RewriteUnpack(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType == OpType::Unpack || opType == OpType::Split || opType == OpType::SplitV )
    {
        const auto *ifmConn = operation->Input(TensorUsage::IFM);
        auto axis = GetAxis(operation);
        if ( axis < 0 ) axis = ifmConn->shape.Size() + axis;

        // Offset of first slice
        Shape sliceOffset = ifmConn->shape.WithZeros();

        for ( auto [usage, ofmConn] : operation->Outputs().pairs() )
        {
            if ( !IsOFM(usage) ) continue;

            // Shape of next slice
            Shape sliceShape;
            if ( opType == OpType::Unpack ) sliceShape = ofmConn.shape.Insert(axis, 1);
            else sliceShape = Shape::PadAxes(ofmConn.shape, ifmConn->shape.Size(), 1);

            // Create a new SLICE op
            auto sliceOp = std::make_shared<Operation>(OpType::Slice);
            sliceOp->CopyInput(TensorUsage::IFM, *ifmConn);
            sliceOp->CopyOutput(TensorUsage::OFM, ofmConn);
            sliceOp->Output(TensorUsage::OFM)->Set(sliceShape);
            auto *attr = sliceOp->Attribute<slice_attr_t>();
            assert(sliceOffset + sliceShape <= ifmConn->shape);
            assert(sliceOffset >= ifmConn->shape.WithZeros());
            attr->size = sliceShape;
            attr->begin = sliceOffset;
            RecordOptimisation(operation, sliceOp.get());
            returnOp = sliceOp.get();

            // Offset of next slice
            sliceOffset[axis] += sliceShape[axis];
        }

        // Remove original op
        operation->Disconnect();
    }
    return returnOp;
}


Operation *TFLiteGraphOptimiser::RemoveReshape(Graph *const graph, Operation *const operation)
{
    Operation *returnOp = operation;
    OpType opType = operation->Type();

    if ( IsReshape(opType) )
    {
        auto *ifmConn = operation->Input(TensorUsage::IFM0);
        auto *ofmConn = operation->Output(TensorUsage::OFM);
        auto *ifm = ifmConn->tensor.get();
        auto *ofm = ofmConn->tensor.get();

        // Check if ifm/ofm are network ifm/ofm or constant
        bool isIfmConst = ifm->IsConstant();
        bool isIfmSgIfm = IsTensorInVector(graph->Inputs(), ifm);
        bool isOfmSgOfm = IsTensorInVector(graph->Outputs(), ofm);
        bool isIfmSgOfm = IsTensorInVector(graph->Outputs(), ifm);

        // Check if ifm/ofm is produced/consumed by a CPU operation
        auto isPassthroughOp = [](const std::shared_ptr<Operation> &op) { return op->Type() == OpType::Passthrough; };
        const bool isOfmCpuIfm =
            std::find_if(ofm->Readers().begin(), ofm->Readers().end(), isPassthroughOp) != ofm->Readers().end();
        const bool isIfmCpuOfm =
            std::find_if(ifm->Writers().begin(), ifm->Writers().end(), isPassthroughOp) != ifm->Writers().end();

        // Inserts a copy op if needed before removing reshapes.
        if ( ((isIfmSgIfm || isIfmSgOfm || isIfmConst) && (isOfmSgOfm)) || (isIfmCpuOfm && isOfmCpuIfm) )
        {
            auto copyOp = InsertCopyOpAfterTensor(ifmConn->tensor, ifmConn->quantization);
            copyOp->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);

            // reset the ifm to reflect the reshape's new ifm
            ifmConn = operation->Input(TensorUsage::IFM0);
            ifm = ifmConn->tensor.get();
            returnOp = copyOp.get();
            RecordOptimisation(operation, returnOp);
            // Reshape still needs to be removed.
        }

        // Remove the reshape and one of the tensors.
        if ( isOfmSgOfm || isOfmCpuIfm )
        {
            // The OFM is in graph outputs, do not remove this tensor.
            // Bypass by replacing ifm with ofm.
            // Set OFM as output for IFM producers
            ReplaceProducerOutput(ifm->Writers(), ifm, ofmConn->tensor);

            // Set OFM as input to other IFM consumers.
            ReplaceConsumerInput(operation, ifm->Readers(), ifm, ofmConn->tensor);
        }
        else
        {
            // Bypass by replacing ofm with ifm.
            // Set IFM as input to OFM consumers.
            ReplaceConsumerInput(nullptr, ofm->Readers(), ofm, ifmConn->tensor);
        }
        // Remove the reshape from ifm readers and ofm writers.
        // Note the Inputs/Outputs on operation should still be intact to not break the traversal.
        ifm->RemoveReader(operation->shared_from_this());
        ofm->RemoveWriter(operation->shared_from_this());
    }

    return returnOp;
}

// Convert ReverseV2 into TOSA Reverse
// ReverseV2 supports a vector of axes, while TOSA reverse only supports one axis
// If there is more than one reversed axis, convert to a sequence of Reverse operations.
//
// ReverseV2(Axis 1,2,3) is converted to:
//     Reverse(axis: 1) -> Reverse(axis: 2) -> Reverse(axis: 3)
//
Operation *TFLiteGraphOptimiser::ConvertReverse(Graph *const graph, Operation *const operation)
{
    auto returnOp = operation;

    if ( operation->Type() == OpType::ReverseV2 )
    {
        auto ifmConn = operation->Input(TensorUsage::IFM);
        auto paramsConn = operation->Input(TensorUsage::Params);
        auto ofmConn = operation->Output(TensorUsage::OFM);
        auto ofm = ofmConn->tensor;

        // We can only handle constant axis vectors
        if ( !paramsConn->tensor->IsConstant() ) return returnOp;
        assert(paramsConn->tensor->Type() == DataType::Int32);
        assert(paramsConn->shape.Size() == 1);

        int numAxes = paramsConn->shape.Depth();
        if ( numAxes == 0 ) return returnOp;

        // Create one Reverse operation for every element in axis
        auto inputConn = ifmConn;
        std::shared_ptr<Tensor> outTens;
        for ( int i = 0; i < numAxes; i++ )
        {
            int32_t axis = paramsConn->tensor->View().Values<int32_t>()[i];
            outTens = ofm;
            // If this is not the final axis, we need to create an intermediate tensor
            if ( i < (numAxes - 1) )
            {
                std::string name(fmt::format("{}_reverse_axis_{}", ofm->Name(), axis));
                outTens = std::make_shared<Tensor>(name, ofm->Type(), ofm->StorageShape());
            }
            auto reverseOp = std::make_shared<Operation>(OpType::Reverse);
            reverseOp->ConnectInput(TensorUsage::IFM, inputConn->tensor).Set(ofmConn->shape);
            reverseOp->ConnectOutput(TensorUsage::OFM, outTens).Set(ofmConn->shape);
            auto *attr = reverseOp->Attribute<axis_attr_t>();
            attr->axis = axis;
            inputConn = reverseOp->Output(TensorUsage::OFM);
            RecordOptimisation(operation, reverseOp.get());
            returnOp = reverseOp.get();
        }

        // quantization is set on the final Reverse operation, the others have unit-scaling
        returnOp->Input(TensorUsage::IFM)->quantization = ifmConn->quantization;
        returnOp->Output(TensorUsage::OFM)->quantization = ofmConn->quantization;
        operation->Disconnect();
    }

    return returnOp;
}

// Replace TFLite GatherV2 and GatherNd with GraphIR Gather, if possible.
Operation *TFLiteGraphOptimiser::ConvertGather(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);

    Operation *returnOp = operation;

    OpType opType = operation->Type();

    ExecutionQuery query{};
    query.opType = OpType::Gather;

    if ( opType == OpType::GatherV2 && _constraints->CanExecute(query) )
    {
        auto *paramsConn = operation->Input(TensorUsage::IFM0);
        auto *idxConn = operation->Input(TensorUsage::IFM1);
        auto *ofmConn = operation->Output(TensorUsage::OFM);
        assert(paramsConn);
        assert(idxConn);
        assert(ofmConn);

        auto paramsRank = paramsConn->shape.Size();
        auto idxRank = idxConn->shape.Size();

        // TFLite Gather attributes
        int axisParam = 0;
        int batchDimsParam = 0;
        const tflite::Operator *const passthrough = static_cast<const tflite::Operator *>(operation->Passthrough());
        if ( passthrough )
        {
            const auto options = passthrough->builtin_options_as_GatherOptions();
            if ( options )
            {
                axisParam = options->axis();
                if ( axisParam < 0 ) axisParam = paramsRank - (-axisParam);
                batchDimsParam = options->batch_dims();
                // TODO: convert below asserts to TFLite semantic checks
                assert(axisParam >= 0);
                assert(axisParam < paramsRank);
                assert(batchDimsParam >= 0);
                assert(batchDimsParam < paramsRank);
                assert(batchDimsParam < idxRank);
                assert(batchDimsParam <= axisParam);
            }
        }
        // TODO: Convert to supported ops check
        // TODO: MLBEDSW-10279 Investigate if constraint can be relaxed
        if ( axisParam != batchDimsParam )
        {
            return returnOp;
        }

        // Calculate GraphIR Gather N dim
        int N = 1;
        for ( int i = 0; i < batchDimsParam; i++ )
        {
            N *= paramsConn->shape[i];
        }

        // Calculate GraphIR Gather W dim
        int W = 1;
        for ( int i = batchDimsParam; i < idxRank; i++ )
        {
            W *= idxConn->shape[i];
        }

        // Calculate GraphIR Gather K dim
        int K = paramsConn->shape[axisParam];

        // Calculate GraphIR Gather C dim
        int C = 1;
        for ( int i = axisParam + 1; i < paramsRank; i++ )
        {
            C *= paramsConn->shape[i];
        }

        // Calculate the remaining dims (must be 1)
        int S = 1;
        for ( int i = batchDimsParam; i < axisParam; i++ )
        {
            S *= paramsConn->shape[i];
        }

        if ( S == 1 )
        {
            // Rebuild shapes
            paramsConn->shape = Shape(1, N, K, C);
            paramsConn->tensor->SetName("values");
            idxConn->shape = Shape(1, 1, N, W);
            idxConn->tensor->SetName("indices");
            ofmConn->shape = Shape(1, N, W, C);
            ofmConn->tensor->SetName("output");

            if ( idxConn->tensor->Type() == DataType::Int16 )
            {
                // Create new op that casts indices to int32
                auto idxCastOp = CreateCastToInt32(idxConn);

                // Use the casted indicies
                auto idxCastConn = idxCastOp->Output(TensorUsage::OFM);
                idxCastConn->shape = Shape(1, 1, N, W);
                idxCastConn->tensor->SetName("indices-int32");
                operation->CopyInput(TensorUsage::IFM1, *idxCastConn);
            }

            // Replace TFLite GatherV2 with GraphIR Gather
            auto gatherOp = std::make_shared<Operation>(OpType::Gather);
            ReplaceOperation(operation, gatherOp.get());
            gatherOp->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
            RecordOptimisation(operation, gatherOp.get());

            returnOp = gatherOp.get();
        }
    }

    return returnOp;
}

// Replace TFLite ScatterNd with GraphIR Scatter, if possible.
Operation *TFLiteGraphOptimiser::ConvertScatter(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);

    Operation *returnOp = operation;

    OpType opType = operation->Type();

    ExecutionQuery query{};
    query.opType = OpType::Scatter;

    if ( opType == OpType::ScatterNd && _constraints->CanExecute(query) )
    {
        auto *idxConn = operation->Input(TensorUsage::IFM0);
        auto *updatesConn = operation->Input(TensorUsage::IFM1);
        auto *shapeConn = operation->Input(TensorUsage::Params);
        auto *ofmConn = operation->Output(TensorUsage::OFM);
        assert(idxConn);
        assert(updatesConn);
        assert(shapeConn);
        assert(ofmConn);

        // Can only support this op when last dimension is 1
        if ( idxConn->shape[-1] != 1 )
        {
            return returnOp;
        }

        // Can only support constant index tensors
        if ( !idxConn->tensor->IsConstant() )
        {
            return operation;
        }
        // Can not support duplicates in the index tensor
        std::unordered_set<int> unique_idxs;
        for ( int i = 0; i < idxConn->tensor->View().Elements(); i++ )
        {
            int idx = idxConn->tensor->View().RawData<int32_t>()[i];
            if ( unique_idxs.find(idx) != unique_idxs.end() )
            {
                return operation;
            }
            unique_idxs.insert(idx);
        }

        assert(shapeConn->tensor->IsConstant());
        assert(shapeConn->shape.Size() == 1);

        // Calculate GraphIR Scatter N dim
        int N = 1;

        // Calculate GraphIR Scatter K dim
        int K = shapeConn->tensor->View().Values<int32_t>()[0];

        // Calculate GraphIR Scatter W dim
        int W = 1;
        for ( int i = 0; i < idxConn->shape.Size() - 1; i++ )
        {
            W *= idxConn->shape[i];
        }

        // Calculate GraphIR Scatter C dim
        int C = 1;
        for ( int i = 1; i < shapeConn->shape.Depth(); i++ )
        {
            C *= shapeConn->tensor->View().Values<int32_t>()[i];
        }

        // Reshape tensors to follow GraphIR Scatter convention
        idxConn->shape = Shape(1, 1, N, W);
        idxConn->tensor->SetName("indices");
        updatesConn->shape = Shape(1, N, W, C);
        updatesConn->tensor->SetName("input");
        ofmConn->shape = Shape(1, N, K, C);
        ofmConn->tensor->SetName("values_out");

        // Generate a constant zeroed tensor as the GraphIR Scatter values_in tensor with same shape as values_out
        auto dtype = ofmConn->tensor->Type();
        std::vector<uint8_t> zeroVector(DataTypeStorageSizeBytes(dtype, ofmConn->shape.Elements()), 0);
        auto zeroBuffer = std::make_shared<Buffer>(std::move(zeroVector));
        auto zeroTensor = CreateConstTensor("values_in", dtype, zeroBuffer, &ofmConn->shape);

        // Add GraphIR Scatter op
        auto scatterOp = std::make_shared<Operation>(OpType::Scatter);
        scatterOp->ConnectInput(TensorUsage::IFM0, zeroTensor);  // GraphIR Scatter values_in
        scatterOp->CopyInput(TensorUsage::IFM1, *idxConn);       // GraphIR Scatter indices
        scatterOp->CopyInput(TensorUsage::IFM2, *updatesConn);   // GraphIR Scatter input
        scatterOp->CopyOutput(TensorUsage::OFM, *ofmConn);       // GraphIR Scatter values_out
        scatterOp->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);

        // Remove TFLite ScatterNd op
        operation->Disconnect();
        RecordOptimisation(operation, scatterOp.get());

        returnOp = scatterOp.get();
    }

    return returnOp;
}

// Replace TFLite ResizeBilinear or ResizeNearestNeighbor with Resize
Operation *TFLiteGraphOptimiser::ConvertResize(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    OpType opType = operation->Type();

    if ( opType == OpType::ResizeBilinear || opType == OpType::ResizeNearestNeighbor )
    {
        auto ifmConn = operation->Input(TensorUsage::IFM);
        auto ofmConn = operation->Output(TensorUsage::OFM);
        assert(ifmConn);
        assert(ofmConn);

        // Get numerators(n) and denominators(d) for the scale fractions
        int width_n = ofmConn->shape.Width();
        int width_d = ifmConn->shape.Width();
        int height_n = ofmConn->shape.Height();
        int height_d = ifmConn->shape.Height();
        int heightOffset = 0;
        int widthOffset = 0;

        const tflite::Operator *passthrough = static_cast<const tflite::Operator *>(operation->Passthrough());
        assert(passthrough);
        bool halfPixelCenters = false;
        bool alignCorners = false;
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

        // Compute scaling fractions
        // align-corners use a scale-factor of (n-1)/(d-1)
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

        // reduce scaling fractions with gcd
        int gcd_w = std::gcd(width_n, width_d);
        width_n = (width_n / gcd_w);
        width_d = (width_d / gcd_w);

        int gcd_h = std::gcd(height_n, height_d);
        height_n = (height_n / gcd_h);
        height_d = (height_d / gcd_h);

        if ( halfPixelCenters )
        {
            // make sure fractions are evenly divisible by 2
            width_n = width_n * 2;
            width_d = width_d * 2;
            height_n = height_n * 2;
            height_d = height_d * 2;
            // adjust offset for half-pixel-centers
            widthOffset = (width_d / 2) - (width_n / 2);
            heightOffset = (height_d / 2) - (height_n / 2);
        }

        // set up op-support query
        ResizeSupportQuery resizeQuery;
        resizeQuery.scaleX = {int16_t(width_n), int16_t(width_d)};
        resizeQuery.scaleY = {int16_t(height_n), int16_t(height_d)};
        resizeQuery.offsetX = widthOffset;
        resizeQuery.offsetY = heightOffset;
        resizeQuery.ifmShape = ifmConn->shape;

        if ( opType == OpType::ResizeBilinear )
        {
            resizeQuery.mode = ArchResizeMode::Bilinear;
        }
        else
        {
            resizeQuery.mode = ArchResizeMode::Nearest;
        }

        ExecutionQuery query{};
        query.opType = opType;
        query.resizeQuery = resizeQuery;

        if ( _constraints->CanExecute(query) )
        {
            // Replace ResizeBilinear or ResizeNearestNeighbor with a Resize op
            auto resizeOp = std::make_shared<Operation>(OpType::Resize);
            resizeOp->CopyInput(TensorUsage::IFM, *ifmConn);
            resizeOp->CopyOutput(TensorUsage::OFM, *ofmConn);
            resizeOp->Output(TensorUsage::OFM)->Set(RoundMode::SYMMETRIC);

            // write operator attributes
            auto *attr = resizeOp->Attribute<resize_attr_t>();
            attr->scaleX = {width_n, width_d};
            attr->scaleY = {height_n, height_d};
            attr->offset = {widthOffset, heightOffset};
            attr->border = {0, 0};
            attr->mode = (opType == OpType::ResizeBilinear) ? tosa::ResizeMode::BILINEAR : tosa::ResizeMode::NEAREST;

            int shift = 0;
            if ( opType == OpType::ResizeBilinear && (ifmConn->shape.Width() > 1 || ifmConn->shape.Height() > 1) )
            {
                // ResizeBilinear is post-scaled with
                // 1 / (height_n * width_n)
                // as the scale-factor is a power of two, we can use shift
                shift = IntLog2(width_n * height_n);
            }

            // Set explicit scaling
            Quantization ofmQuant = ofmConn->quantization;
            ofmQuant.scales.clear();
            ofmQuant.zeroPoints.clear();
            ofmQuant.scales.emplace_back(QuantizedScale(1, shift));
            ofmQuant.zeroPoints.emplace_back(0);
            ofmQuant.type = QuantizationType::EXPLICIT;
            resizeOp->Output(TensorUsage::OFM)->Set(ofmQuant);

            RecordOptimisation(operation, resizeOp.get());
            returnOp = resizeOp.get();
            operation->Disconnect();
        }
    }
    return returnOp;
}

// Convert TFLite Transpose into TOSA Transpose
Operation *TFLiteGraphOptimiser::ConvertTranspose(Graph *const graph, Operation *const operation)
{
    Operation *returnOp = operation;
    OpType opType = operation->Type();
    if ( opType == OpType::Transpose )
    {
        auto *paramsConn = operation->Input(TensorUsage::Params);
        auto *attr = operation->Attribute<transpose_attr_t>();

        // We can only handle permutation vectors up to 8 elements
        if ( paramsConn->shape.Depth() > 8 ) return returnOp;

        // We can only handle constant permutation vectors
        if ( !paramsConn->tensor->IsConstant() ) return returnOp;

        // Decode the permutation vector into a shape
        std::vector<int32_t> perm;
        for ( int i = 0; i < paramsConn->shape.Depth(); i++ )
        {
            perm.push_back(paramsConn->tensor->View().Values<int32_t>()[i]);
        }
        attr->perm = Shape::FromVector(perm);
    }
    return returnOp;
}

// Convert TFLite REDUCE_{MIN,MAX,ANY,ALL} to one or more TOSA REDUCE_{MIN,MAX,ANY,ALL}
Operation *TFLiteGraphOptimiser::ConvertReduceMinMaxAnyAll(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    const auto opType = operation->Type();
    if ( opType == OpType::ReduceMin || opType == OpType::ReduceMax || opType == OpType::ReduceAny || opType == OpType::ReduceAll )
    {
        auto *ifmConn = operation->Input(TensorUsage::IFM);
        auto *paramsConn = operation->Input(TensorUsage::Params);
        auto *ofmConn = operation->Output(TensorUsage::OFM);

        // Probably already a TOSA op
        if ( !paramsConn ) return returnOp;

        // Get the axis values from the constant params tensor
        assert(paramsConn->shape.Size() == 1);
        assert(paramsConn->shape.Depth() > 0);
        assert(paramsConn->tensor->IsConstant());
        assert(paramsConn->tensor->Type() == DataType::Int32);
        const auto paramsValues = paramsConn->tensor->View().Values<int32_t>();
        std::vector<int32_t> axes;
        for ( int i = 0; i < paramsConn->shape.Depth(); i++ )
        {
            int axis = paramsValues[i];
            if ( axis < 0 ) axis = ifmConn->shape.Size() + axis;
            assert(axis >= 0);
            assert(axis < ifmConn->shape.Size());
            axes.push_back(axis);
        }

        // Break down TFLite op into one or more TOSA ops, one per reduced dimension
        Operation *prevOp = operation;
        TensorConnection *prevConn = ifmConn;
        for ( int axis : axes )
        {
            auto reduceOp = std::make_shared<Operation>(opType);
            auto *reduceOpAttr = reduceOp->Attribute<axis_attr_t>();
            reduceOpAttr->axis = axis;
            reduceOp->CopyInput(TensorUsage::IFM, *prevConn);
            const auto ofmName = prevConn->tensor->Name() + "_reduce" + std::to_string(axis);
            const auto ofmType = prevConn->tensor->Type();
            const auto ofmShape = prevConn->shape.With(axis, 1);
            const auto ofmTensor = std::make_shared<Tensor>(ofmName, ofmType, ofmShape);
            reduceOp->ConnectOutput(TensorUsage::OFM, ofmTensor).Set(prevConn->quantization).Set(RoundMode::NATURAL);
            RecordOptimisation(operation, reduceOp.get());
            returnOp = reduceOp.get();

            prevOp = reduceOp.get();
            prevConn = reduceOp->Output(TensorUsage::OFM);
        }

        // Adjust the last op so it connects to the original OFM
        prevOp->ConnectOutput(TensorUsage::OFM, ofmConn->tensor).Set(prevConn->quantization).Set(prevConn->shape);

        // Remove TFLite op
        operation->Disconnect();
    }
    return returnOp;
}

Operation *TFLiteGraphOptimiser::CreateTransposeForMatMul(const std::shared_ptr<Tensor> &ifm, const Shape &ofmShape)
{
    auto op = std::make_shared<Operation>(OpType::Transpose);

    int32_t permutation[] = {0, 1, 3, 2};
    auto buf = std::make_shared<Buffer>(4, std::move(permutation), false);

    // IFM should have the untransposed shape
    op->ConnectInput(TensorUsage::IFM, ifm).Set(Shape(1, ofmShape.Height(), ofmShape.Depth(), ofmShape.Width()));
    op->ConnectInput(TensorUsage::Params, std::make_shared<Tensor>("perm", DataType::Int32, Shape(4), buf));

    auto ofm = std::make_shared<Tensor>(ifm->Name() + "/" + OpTypeToString(op->Type()), ifm->Type());
    ofm->SetStorageShape(ofmShape);

    op->ConnectOutput(TensorUsage::OFM, ofm);
    return op.get();
}

// Convert TFLite BatchMatmul to GraphIR Matmul
// Transpose inputs (NHCW) based on adj_x/y to align
// with the TOSA/GraphIR representation of Matmul:
//    IFM should be transposed if adj_x is true
//    IFM2 should be transposed if adj_y is true
Operation *TFLiteGraphOptimiser::RewriteBatchMatMul(Graph *const, Operation *const operation)
{
    Operation *returnOp = operation;
    ExecutionQuery query{};
    query.opType = OpType::MatMul;
    if ( operation->Type() == OpType::BatchMatMul && _constraints->CanExecute(query) )
    {
        const auto ifm = operation->Input(TensorUsage::IFM0);
        const auto ifm2 = operation->Input(TensorUsage::IFM1);
        const auto ofm = operation->Output(TensorUsage::OFM);

        bool transposeIfm = false;
        bool transposeIfm2 = false;
        const tflite::Operator *const passthrough = static_cast<const tflite::Operator *>(operation->Passthrough());
        if ( passthrough )
        {
            const auto options = passthrough->builtin_options_as_BatchMatMulOptions();
            if ( options )
            {
                // TOSA/GraphIR Matmul representation aligns with adj_x/adj_y == false.
                // Transpose inputs if necessary
                transposeIfm = options->adj_x();
                transposeIfm2 = options->adj_y();
            }
        }

        auto ofmShape = Shape::PadAxes(ofm->shape, 4, 1);
        auto ifmShape = Shape::PadAxes(ifm->shape, 4, 1);
        auto ifm2Shape = Shape::PadAxes(ifm2->shape, 4, 1);

        int n = ofmShape.Batch() * ofmShape.Height();

        // IFM handling - Reshape ifm N,H,W,C -> 1,NxH,W,C
        auto ifmReshaped = Shape(1, n, ifmShape.Width(), ifmShape.Depth());
        auto ifmTensor = ifm->tensor;
        if ( transposeIfm )
        {
            // Add Transpose op, ifm:  1,n,W,C -> 1,n,C,W
            ifmReshaped = Shape(1, ifmReshaped.Height(), ifmReshaped.Depth(), ifmReshaped.Width());
            auto op = CreateTransposeForMatMul(ifm->tensor, ifmReshaped);
            RecordOptimisation(operation, op);
            ifmTensor = op->Output(TensorUsage::OFM)->tensor;
        }

        // IFM2 handling - Reshape ifm2 N,H,W,C -> 1,NxH,W,C
        auto ifm2Reshaped = Shape(1, n, ifm2Shape.Width(), ifm2Shape.Depth());
        auto ifm2Tensor = ifm2->tensor;
        if ( transposeIfm2 )
        {
            // Add Transpose op, ifm2: 1,n,W,C -> 1,n,C,W
            ifm2Reshaped = Shape(1, ifm2Reshaped.Height(), ifm2Reshaped.Depth(), ifm2Reshaped.Width());
            auto op = CreateTransposeForMatMul(ifm2->tensor, ifm2Reshaped);
            RecordOptimisation(operation, op);
            ifm2Tensor = op->Output(TensorUsage::OFM)->tensor;
        }

        auto ofmReshaped = Shape(1, n, ofmShape.Width(), ofmShape.Depth());

        auto rounding = ifm->tensor->Type() == DataType::Int16 ? RoundMode::NATURAL : RoundMode::DBL;
        auto newOp = std::make_shared<Operation>(OpType::MatMul);
        newOp->ConnectInput(TensorUsage::IFM0, ifmTensor).Set(ifmReshaped).Set(ifm->quantization);
        newOp->ConnectInput(TensorUsage::IFM1, ifm2Tensor).Set(ifm2Reshaped).Set(ifm2->quantization);
        newOp->CopyOutput(TensorUsage::OFM, *ofm);
        newOp->Output(TensorUsage::OFM)->Set(ofmReshaped).Set(rounding);
        returnOp = newOp.get();
        RecordOptimisation(operation, returnOp);
        operation->Disconnect();
    }
    return returnOp;
}


Operation *TFLiteGraphOptimiser::RewriteFullyConnectDynamic(Graph *const, Operation *const operation)
{
    Operation *returnOp = operation;
    auto ifm2 = operation->Input(TensorUsage::Weights);
    ExecutionQuery query{};
    query.opType = OpType::MatMul;
    if ( operation->Type() == OpType::FullyConnected && !ifm2->tensor->IsConstant() && _constraints->CanExecute(query) )
    {
        const auto ifm = operation->Input(TensorUsage::IFM0);
        const auto ofm = operation->Output(TensorUsage::OFM);

        auto ofmShape = Shape::PadAxes(ofm->shape, 4, 1);
        auto ifmShape = Shape::PadAxes(ifm->shape, 4, 1);
        auto ifm2Shape = Shape::PadAxes(ifm2->shape, 4, 1);

        // Add NHCW Transpose op, to convert to GraphIR/TOSA Matmul representation
        auto ifm2Reshaped = Shape(ifm2Shape.Batch(), ifm2Shape.Height(), ifm2Shape.Depth(), ifm2Shape.Width());
        auto transposeOp = CreateTransposeForMatMul(ifm2->tensor, ifm2Reshaped);
        RecordOptimisation(operation, transposeOp);
        auto ifm2Tensor = transposeOp->Output(TensorUsage::OFM)->tensor;

        auto matMulOp = std::make_shared<Operation>(OpType::MatMul);
        auto rounding = ifm->tensor->Type() == DataType::Int16 ? RoundMode::NATURAL : RoundMode::DBL;

        matMulOp->ConnectInput(TensorUsage::IFM0, ifm->tensor).Set(ifmShape).Set(ifm->quantization).Set(ifm->slice);
        matMulOp->ConnectInput(TensorUsage::IFM1, ifm2Tensor).Set(ifm2Reshaped).Set(ifm2->quantization).Set(ifm2->slice);
        matMulOp->ConnectOutput(TensorUsage::OFM, ofm->tensor).Set(ofmShape).Set(ofm->quantization).Set(ofm->slice).Set(rounding);

        RecordOptimisation(operation, matMulOp.get());
        returnOp = matMulOp.get();

        operation->Disconnect();
    }
    return returnOp;
}


Operation *TFLiteGraphOptimiser::RewriteSquaredDifference(Graph *const, Operation *const operation)
{
    Operation *returnOp = operation;
    if ( operation->Type() == OpType::SquaredDifference )
    {
        const auto ifmConn = operation->Input(TensorUsage::IFM0);
        const auto ifm2Conn = operation->Input(TensorUsage::IFM1);
        const auto ofmConn = operation->Output(TensorUsage::OFM);

        const double ifmScale = ifmConn->quantization.scales[0].Dequantize();
        const double ifm2Scale = ifm2Conn->quantization.scales[0].Dequantize();
        const double ofmScale = ofmConn->quantization.scales[0].Dequantize();

        auto oneScaleQuant = ifmConn->quantization;
        oneScaleQuant.scales[0] = {1, 0};
        oneScaleQuant.zeroPoints.clear();

        auto noScaleQuant = ifmConn->quantization;
        noScaleQuant.scales.clear();
        noScaleQuant.zeroPoints.clear();

        // All the calculations same as reference kernel
        const double twiceMaxInputScale = 2.0 * std::max(ifmScale, ifm2Scale);
        const double realInput1Multiplier = ifmScale / twiceMaxInputScale;
        const double realInput2Multiplier = ifm2Scale / twiceMaxInputScale;

        int leftShift = ifmConn->tensor->Type() == DataType::Int16 ? 0 : 7;

        double realOutputMultiplier = (twiceMaxInputScale * twiceMaxInputScale) / ((1 << (leftShift * 2)) * ofmScale);

        auto quantizedRealInput1 = QuantizedScale(realInput1Multiplier);
        auto quantizedRealInput2 = QuantizedScale(realInput2Multiplier);
        auto quantizedRealOutput = QuantizedScale(realOutputMultiplier);
        quantizedRealInput1.scale = std::max(quantizedRealInput1.scale, 1);
        quantizedRealInput2.scale = std::max(quantizedRealInput2.scale, 1);
        quantizedRealOutput.scale = std::max(quantizedRealOutput.scale, 1);

        auto input1MultiplierConst = CreateConstTensor(
            ifmConn->tensor->Name() + "_input1_multiplier", quantizedRealInput1.scale);
        auto input2MultiplierConst = CreateConstTensor(
            ifm2Conn->tensor->Name() + "_input2_multiplier", quantizedRealInput2.scale);
        auto outputMultiplierConst = CreateConstTensor(
            ofmConn->tensor->Name() + "_output_multiplier", quantizedRealOutput.scale);

        // Convert ifm to 32 bit
        auto castOp = CreateCastToInt32(ifmConn);
        // Use explicit scaling (multiplier) for the left shift
        castOp->Output(TensorUsage::OFM)->quantization.scales.clear();
        castOp->Output(TensorUsage::OFM)->quantization.scales.push_back(QuantizedScale(1 << leftShift, 0));
        castOp->Output(TensorUsage::OFM)->quantization.type = QuantizationType::EXPLICIT;

        // Scale/shift ifm (for 32-bit operations, scale is not applied but shift is)
        auto mulOp = CreateMul(castOp->Output(TensorUsage::OFM)->tensor, input1MultiplierConst, noScaleQuant, noScaleQuant, noScaleQuant);
        mulOp->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
        mulOp->Output(TensorUsage::OFM)->quantization.scales.clear();
        mulOp->Output(TensorUsage::OFM)->quantization.scales.push_back(QuantizedScale(1, quantizedRealInput1.shift));
        mulOp->Output(TensorUsage::OFM)->quantization.type = QuantizationType::EXPLICIT;
        auto ifmScaled = mulOp->Output(TensorUsage::OFM);
        RecordOptimisation(operation, mulOp);

        // Convert ifm2 to 32 bit
        castOp = CreateCastToInt32(ifm2Conn);
        // Use explicit scaling (multiplier) for the left shift
        castOp->Output(TensorUsage::OFM)->quantization.scales.clear();
        castOp->Output(TensorUsage::OFM)->quantization.scales.push_back(QuantizedScale(1 << leftShift, 0));
        castOp->Output(TensorUsage::OFM)->quantization.type = QuantizationType::EXPLICIT;
        RecordOptimisation(operation, castOp);

        // Scale/shift ifm2 (for 32-bit operations, scale is not applied but shift is)
        mulOp = CreateMul(castOp->Output(TensorUsage::OFM)->tensor, input2MultiplierConst, noScaleQuant, noScaleQuant, noScaleQuant);
        mulOp->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
        mulOp->Output(TensorUsage::OFM)->quantization.scales.clear();
        mulOp->Output(TensorUsage::OFM)->quantization.scales.push_back(QuantizedScale(1, quantizedRealInput2.shift));
        mulOp->Output(TensorUsage::OFM)->quantization.type = QuantizationType::EXPLICIT;
        auto ifm2Scaled = mulOp->Output(TensorUsage::OFM);
        RecordOptimisation(operation, mulOp);

        // Calculate the raw diff
        auto subOp = CreateSub(ifmScaled->tensor, ifm2Scaled->tensor, noScaleQuant, noScaleQuant, noScaleQuant);
        subOp->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
        auto rawDiff = subOp->Output(TensorUsage::OFM);
        RecordOptimisation(operation, subOp);

        // Calculate the squared diff
        mulOp = CreateMul(rawDiff->tensor, rawDiff->tensor, noScaleQuant, noScaleQuant, noScaleQuant);
        mulOp->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
        auto squaredRaw = mulOp->Output(TensorUsage::OFM);
        RecordOptimisation(operation, mulOp);

        // Scale/shift ofm ((for 32-bit operations, scale is not applied but shift is)
        returnOp = CreateMul(squaredRaw->tensor, outputMultiplierConst, noScaleQuant, noScaleQuant, ofmConn->quantization);
        returnOp->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
        returnOp->ConnectOutput(TensorUsage::OFM, ofmConn->tensor);
        returnOp->Output(TensorUsage::OFM)->quantization.scales.clear();
        returnOp->Output(TensorUsage::OFM)->quantization.scales.push_back(QuantizedScale(1, quantizedRealOutput.shift));
        returnOp->Output(TensorUsage::OFM)->quantization.type = QuantizationType::EXPLICIT;
        RecordOptimisation(operation, returnOp);

        operation->Disconnect();
    }
    return returnOp;
}


Operation *TFLiteGraphOptimiser::RewriteSpaceToBatchConvBatchToSpace(Graph *const, Operation *const operation)
{
    auto opType = operation->Type();
    if ( opType == OpType::DepthwiseConv2D || opType == OpType::Conv2D )
    {
        auto prevOp = operation->IFM(0)->Writers().empty() ? nullptr : operation->IFM(0)->Writers().front().get();
        auto nextOp = operation->OFM()->Readers().empty() ? nullptr : operation->OFM()->Readers().front().get();
        if ( prevOp && prevOp->Type() == OpType::SpaceToBatchND &&  // Previous op is SpaceToBatchND
             nextOp && nextOp->Type() == OpType::BatchToSpaceND &&  // Next op is BatchToSpaceND
             operation->IFM(0)->Readers().size() == 1 &&            // No other consumers of SpaceToBatchND output
             operation->OFM()->Readers().size() == 1                // No other consumers of BatchToSpaceND input
        )
        {
            // Go ahead and short-circuit the SpaceToBatchND and BatchToSpaceND ops
            operation->ConnectInput(TensorUsage::IFM0, prevOp->Input(TensorUsage::IFM0)->tensor);
            operation->ConnectOutput(TensorUsage::OFM, nextOp->Output(TensorUsage::OFM)->tensor);
            // Set new kernel dilation
            auto blockShape = prevOp->Input(TensorUsage::Params);
            int count = blockShape->shape[0];
            assert(count == operation->IFM(0)->StorageShape().Size() - 2);
            assert(blockShape->tensor->IsConstant());
            auto values = blockShape->tensor->View().Values<int32_t>();
            Point2i dilation(values[0], values[count > 1 ? 1 : 0]);
            Kernel dilatedKernel = operation->Kernel()->WithDilation(std::move(dilation));
            // Calculate padding for new kernel
            Point2i dilatedWH = dilatedKernel.DilatedWH();
            auto &stride = dilatedKernel.Stride();
            auto &inputShape = operation->IFM(0)->StorageShape();
            int xpad = NeededTotalPadding(inputShape.Width(), stride.x, dilatedWH.x);
            int ypad = NeededTotalPadding(inputShape.Height(), stride.y, dilatedWH.y);
            Margin pad = Margin(ypad / 2, xpad / 2, (ypad + 1) / 2, (xpad + 1) / 2);
            // Set the new kernel with updated dilation and padding
            operation->SetKernel(std::make_unique<Kernel>(dilatedKernel.WithPadding(pad)));
            // Disconnect the SpaceToBatchND and BatchToSpaceND ops
            prevOp->Disconnect();
            nextOp->Disconnect();
        }
    }
    return operation;
}

// Fixup Conv2D and DepthwiseConv2D to allow dilation greater than 2.
// TODO: Replace with kernel decomposition for supported architectures
Operation *TFLiteGraphOptimiser::FixupDilationGT2(Graph *const, Operation *const operation)
{
    auto returnOp = operation;
    if ( operation->Type() == OpType::Conv2D || operation->Type() == OpType::DepthwiseConv2D )
    {
        auto dilation = operation->Kernel()->Dilation();
        // If dilation in either axis is greater than that supported by hardware then we must manually dilate the kernel
        if ( dilation.x > 2 || dilation.y > 2 )
        {
            // If the dilation is a multiple of 2 then the hardware dilation can be enabled to provide that multiple
            // of 2. This allows the kernel size to be reduced (via the scaled dilation) by half in that dimension.
            int hwDilationH = (dilation.y % 2 == 0) ? 2 : 1;
            int hwDilationW = (dilation.x % 2 == 0) ? 2 : 1;
            int manualDilationH = dilation.y / hwDilationH;
            int manualDilationW = dilation.x / hwDilationW;

            auto *weightConn = operation->Input(TensorUsage::Weights);
            assert(weightConn);
            assert(weightConn->tensor->IsConstant());
            auto weights = weightConn->tensor->View().Values<int8_t>();
            const auto &weightShape = weightConn->shape;

            // Create new empty kernel with dilated size
            auto origKernelSize = operation->Kernel()->Size();
            auto dilatedKernelSize = operation->Kernel()->WithDilation({manualDilationW, manualDilationH}).DilatedWH();
            Kernel dilatedKernel = operation->Kernel()->WithDilation({hwDilationW, hwDilationH}).WithSize(dilatedKernelSize);
            const int newKernelBufferSize = weightShape.Batch() * dilatedKernel.ElementsWH() * weightShape.Depth();
            operation->SetKernel(std::make_unique<Kernel>(std::move(dilatedKernel)));

            // Copy the original kernel values into the new sparse kernel
            // Width and depth stride same for original and new kernel
            auto strideC = 1;
            auto strideW = weightShape.Depth();
            auto newStrideH = strideW * dilatedKernelSize.x;
            auto newStrideO = newStrideH * dilatedKernelSize.y;

            auto newKernelVals = std::make_unique<int8_t[]>(newKernelBufferSize);
            for ( int oc = 0; oc < weightShape.Batch(); oc++ )
            {
                for ( int h = 0; h < origKernelSize.y; ++h )
                {
                    for ( int w = 0; w < origKernelSize.x; ++w )
                    {
                        for ( int c = 0; c < weightShape.Depth(); c++ )
                        {
                            auto newKernelIdx = c * strideC + w * strideW * manualDilationW + h * newStrideH * manualDilationH + oc * newStrideO;
                            assert(newKernelIdx >= 0 && newKernelIdx < newKernelBufferSize);
                            newKernelVals[newKernelIdx] = weights[{oc, h, w, c}];
                        }
                    }
                }
            }
            weightConn->tensor->SetBuffer(std::make_shared<Buffer>(std::move(newKernelVals), newKernelBufferSize));
            Shape newShape = weightShape.WithHW(dilatedKernelSize.y, dilatedKernelSize.x);
            weightConn->tensor->SetStorageShape(newShape);
            weightConn->Set(newShape);
        }
    }
    return returnOp;
}

// If conv op without bias tensor, create one with zeroes
Operation *TFLiteGraphOptimiser::FixupBias(Graph *const, Operation *const operation)
{
    if ( IsConvolution(operation->Type()) && operation->CountInputs(TensorUsage::Scales) == 0 )
    {
        auto ifmConn = operation->Input(TensorUsage::IFM);
        auto ofmConn = operation->Output(TensorUsage::OFM);

        // Create bias tensor with zeroes
        DataType biasType;
        std::shared_ptr<Buffer> biasBuffer;
        auto biasElements = ofmConn->shape.Depth();
        if ( ifmConn->tensor->Type() == DataType::Int16 )
        {
            biasType = DataType::Int64;
            biasBuffer = std::make_shared<Buffer>(std::make_unique<int64_t[]>(biasElements), biasElements);
        }
        else
        {
            biasType = DataType::Int32;
            biasBuffer = std::make_shared<Buffer>(std::make_unique<int32_t[]>(biasElements), biasElements);
        }
        auto biasTensor = CreateConstTensor("bias", biasType, biasBuffer);
        operation->ConnectInput(TensorUsage::Scales, biasTensor);
    }
    return operation;
}

// Check that no reshape like operations remain in graph.
Operation *TFLiteGraphOptimiser::CheckReshapeOpsRemoved(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    OpType opType = operation->Type();
    if ( IsReshape(opType) )
    {
        LOG_ERROR("Reshape-like operation type {0} expected to have been removed, still remains.\n", OpTypeToString(opType));
        assert(false);
    }
    return operation;
}

Operation *TFLiteGraphOptimiser::ConvertSoftmaxOps(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    return _softmax->ConvertOp(operation);
}

static bool MeanOpSupported(Operation *const operation, Shape &reduceAxis, Shape &ifmShape4D)
{
    auto ifmConn = operation->Input(TensorUsage::IFM0);
    auto ifm = ifmConn->tensor;
    auto axis = operation->Input(TensorUsage::Params)->tensor;
    auto axisValues = axis->View().Values<int32_t>();
    auto axisCount = axis->StorageShape().IsEmpty() ? 1 : axis->StorageShape().Depth();
    auto ifmDims = ifmShape4D.Size();

    // Max kernel size
    static constexpr int MAX_MEAN_KERNEL_SIZE = 64 * 64;
    // Max size to avoid overflow INT32
    static constexpr int MAX_MEAN_ELEMENTS_INT8 = 2 << 23;   // 2²⁴ x 2⁷  = 2³¹
    static constexpr int MAX_MEAN_ELEMENTS_UINT8 = 2 << 22;  // 2²³ x 2⁸  = 2³¹
    static constexpr int MAX_MEAN_ELEMENTS_INT16 = 2 << 15;  // 2¹⁶ x 2¹⁵ = 2³¹

    bool supported = false;

    // Compute total number of elements
    int elements = 1;
    for ( int i = 0; i < ifmDims; ++i )
    {
        elements *= reduceAxis[i] ? ifmShape4D[i] : 1;
    }

    // Make sure overflow can not occur
    switch ( ifm->Type() )
    {
        case DataType::Int8:
            supported = elements <= MAX_MEAN_ELEMENTS_INT8;
            break;

        case DataType::UInt8:
            supported = elements <= MAX_MEAN_ELEMENTS_UINT8;
            break;

        case DataType::Int16:
            supported = elements <= MAX_MEAN_ELEMENTS_INT16;
            break;

        default:
            supported = false;
            break;
    }

    // Only support batch 1
    supported = supported && (ifmShape4D.Batch() == 1);

    // Reduced axis must be no greater than MAX_MEAN_KERNEL_SIZE
    supported = supported && (reduceAxis.Depth() * ifmShape4D.Depth() <= MAX_MEAN_KERNEL_SIZE);
    supported = supported && (reduceAxis.Width() * ifmShape4D.Width() <= MAX_MEAN_KERNEL_SIZE);
    supported = supported && (reduceAxis.Height() * ifmShape4D.Height() <= MAX_MEAN_KERNEL_SIZE);

    // Depth is supported if any of h,w,c == 1
    if ( supported && reduceAxis.Depth() )
    {
        supported = false;
        for ( int i = 1; i < 4; i++ )
        {
            if ( ifmShape4D[i] == 1 )
            {
                supported = true;
                break;
            }
        }
    }
    return supported;
}

Operation *TFLiteGraphOptimiser::ConvertMeanOps(Graph *const, Operation *const operation)
{
    auto returnOp = operation;
    if ( operation->Type() == OpType::Mean )
    {
        auto ifmConn = operation->Input(TensorUsage::IFM0);
        auto ofmConn = operation->Output(TensorUsage::OFM);
        auto axis = operation->Input(TensorUsage::Params)->tensor;
        auto axisValues = axis->View().Values<int32_t>();
        auto axisCount = axis->StorageShape().IsEmpty() ? 1 : axis->StorageShape().Depth();
        auto &ifmShape = ifmConn->shape;
        auto &ofmShape = ofmConn->shape;
        auto ifmDims = ifmShape.Size();
        auto ofmDims = ofmShape.Size();
        auto &ifmQuant = ifmConn->quantization;
        auto &ofmQuant = ofmConn->quantization;
        static constexpr int MAX_MEAN_HEIGHT = 64;
        static constexpr int MAX_MEAN_KERNEL_SIZE = 64 * 64;

        Shape reduceAxis = ifmShape.WithZeros();
        for ( int i = 0; i < axisCount; ++i )
        {
            reduceAxis[axisValues[i]] = 1;
        }
        // Create a 4D shape to indicate which axis that will be reduced
        Shape reduceAxis4D = Shape::PadAxes(reduceAxis, 4, 0);

        Shape ifmShape4D = Shape::PadAxes(ifmShape, 4, 1);

        // Check if it is possible to convert the MEAN
        if ( !MeanOpSupported(operation, reduceAxis4D, ifmShape4D) )
        {
            return operation;
        }

        // Fix intermediateShape when keep_dims is false
        // e.g. IFM=1xHxWxC axis=2 OFM=1xHxC, the intermediateShape should be 1xHx1xC
        Shape intermediateShape = ofmConn->shape;
        if ( ofmDims < ifmDims )
        {
            for ( int i = 0; i < ifmDims; i++ )
            {
                // Note: do not use reduceAxis4D here since we are using org dims
                if ( reduceAxis[i] )
                {
                    intermediateShape = intermediateShape.Insert(i, 1);
                }
            }
        }
        intermediateShape = Shape::PadAxes(intermediateShape, 4, 1);

        // Support mean over depth-axis by left-shifting the C channel
        // From operator checks we can assume that one of H,W,C has shape 1
        if ( reduceAxis4D.Depth() && ifmShape4D.Depth() > 1 )
        {
            // If W=1 reshape NxHx1xC -> NxHxCx1, else reshape Nx1xWxC -> NxWxCx1
            int idxToDelete = ifmShape.Width() == 1 ? 2 : 1;

            // Delete axis with size 1
            reduceAxis4D = reduceAxis4D.Erase(idxToDelete);
            ifmShape4D = ifmShape4D.Erase(idxToDelete);
            intermediateShape = intermediateShape.Erase(idxToDelete);

            // Add another element to set channel-axis to one
            reduceAxis4D = reduceAxis4D.Insert(3, 0);
            ifmShape4D = ifmShape4D.Insert(3, 1);
            intermediateShape = intermediateShape.Insert(3, 1);
        }

        // Compute kernel sizes for our convolutions
        int h = reduceAxis4D.Height() ? ifmShape4D.Height() : 1;
        int w = reduceAxis4D.Width() ? ifmShape4D.Width() : 1;

        assert(CheckSafeMul(w, h));
        int num_elements_in_axis = h * w;

        // If one convolution is enough, but height is greater than max kernel height
        // reshape from HxW to 1x(HxW)
        // This can only be done if the mean is computed over both H and W
        if ( h > MAX_MEAN_HEIGHT && num_elements_in_axis <= MAX_MEAN_KERNEL_SIZE && reduceAxis4D.Height() &&
             reduceAxis4D.Width() )
        {
            ifmShape4D = Shape(ifmShape4D.Batch(), 1, h * w, ifmShape4D.Depth());
            w = h * w;
            h = 1;
        }

        // When h x w <= 4096     When h x w > 4096 there is a need to split into several ops.
        //                        Do this by splitting up h and change the read_offset/shape.
        //                        Below is an example where ifm is 1x190x64x1
        //     MEAN                                       MEAN
        //       |                    +---------------------|---------------------+
        // DepthwiseConv2D    1_DepthwiseConv2D     2_DepthwiseConv2D     3_DepthwiseConv2D
        //       |                    |                     |                     |
        //      MUL                   +---------ADD---------+                     |
        //                                       |                                |
        //                                       +--------------ADD---------------+
        //                                                       |
        //                                                      MUL
        //       1_DepthwiseConv2DBias: read_offset [0, 0, 0, 0]> read_shape [1,  64, 64, 1]>
        //       2_DepthwiseConv2DBias: read_offset [0, 64, 0, 0]> read_shape [1,  64, 64, 1]>
        //       3_DepthwiseConv2DBias: read_offset [0, 128, 0, 0]> read_shape [1,  62, 64, 1]>


        int heightPerConv = std::min(MAX_MEAN_KERNEL_SIZE / w, h);
        heightPerConv = std::min(heightPerConv, MAX_MEAN_HEIGHT);
        int opCount = (h + heightPerConv - 1) / heightPerConv;
        Quantization oneScaleQuant = ifmConn->quantization;
        oneScaleQuant.scales.clear();
        oneScaleQuant.scales.push_back({1, 0});
        Quantization oneScaleQuantZp0 = oneScaleQuant;
        oneScaleQuantZp0.zeroPoints.clear();
        oneScaleQuantZp0.zeroPoints.push_back(0);

        std::shared_ptr<Tensor> accTensor = nullptr;

        // Reuse weight tensor if more ops are needed
        std::shared_ptr<Tensor> weightTensor = nullptr;
        std::shared_ptr<Tensor> biasTensor = nullptr;

        // set weight quantization
        Quantization weightQuant = ifmConn->quantization;
        weightQuant.quantMin = {0};
        weightQuant.quantMax = {255};
        weightQuant.scales.clear();
        weightQuant.zeroPoints.clear();
        weightQuant.scales.push_back({1, 0});
        weightQuant.zeroPoints.push_back(0);

        for ( int i = 0; i < opCount; ++i )
        {
            bool isLastOp = (i == (opCount - 1));

            // Compute height for the kernel
            int kh = heightPerConv;
            if ( isLastOp && h % heightPerConv != 0 )
            {
                kh = h % heightPerConv;
                // New kernel shape so new weight tensor is needed
                weightTensor = nullptr;
                biasTensor = nullptr;
            }

            // Calculate read and offset shape
            int readShapeH = reduceAxis4D.Height() ? kh : ifmShape4D.Height();
            int readShapeW = reduceAxis4D.Width() ? w : ifmShape4D.Width();

            Shape readOffset(0, i * heightPerConv, 0, 0);
            Shape readShape = ifmShape4D.WithHW(readShapeH, readShapeW);

            auto op = MakeDepthwiseMeanOp(ifmConn, ifmShape4D, readShape, readOffset, intermediateShape, w, kh,
                ofmConn->tensor->Name(), weightTensor, biasTensor, oneScaleQuant, weightQuant, oneScaleQuantZp0);
            RecordOptimisation(operation, op);

            if ( i > 0 )
            {
                // Add result to accumulator tensor
                Quantization accQuant = op->Output(TensorUsage::OFM)->quantization;
                op = CreateAdd(accTensor, op->Output(TensorUsage::OFM)->tensor, oneScaleQuantZp0, oneScaleQuantZp0, oneScaleQuantZp0);
                op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
                op->Output(TensorUsage::OFM)->quantization.scales.clear();
                op->Output(TensorUsage::OFM)->quantization.scales.push_back(QuantizedScale(1, 0));
                op->Output(TensorUsage::OFM)->quantization.type = QuantizationType::EXPLICIT;
                RecordOptimisation(operation, op);
            }
            accTensor = op->Output(TensorUsage::OFM)->tensor;
        }
        QuantizedScale quant(ifmQuant.scales[0].Dequantize() / ofmQuant.scales[0].Dequantize());

        // Convert to left shift-positive notation
        auto outputShift = 31 - quant.shift;

        // Below calculation same as in reference to avoid any risk of overflow,
        // clamping the shift value at the price of some precision loss.
        // IntLog2 same as 63 - CountLeadingZeros(num_elements_in_axis)
        int shift = IntLog2(num_elements_in_axis);
        shift = std::min(shift, 32);
        shift = std::min(shift, 31 + outputShift);
        // Multiplier should be 32bit
        int32_t outputMultiplier = int32_t((int64_t(quant.scale) << shift) / num_elements_in_axis);

        // Convert to right-shift
        outputShift = 31 - (outputShift - shift);

        // For int32 scaling is not supported so instead multiply with the scale
        auto scalar = CreateConstTensor(ofmConn->tensor->Name() + "_scalar", outputMultiplier);
        auto op = CreateMul(accTensor, scalar, oneScaleQuantZp0, oneScaleQuantZp0, oneScaleQuantZp0);
        op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);

        // Apply the shift
        QuantizedScale scale(1, outputShift);
        Quantization outQuant = ofmConn->quantization;
        outQuant.scales.clear();
        outQuant.scales.push_back({1, outputShift});
        outQuant.type = QuantizationType::EXPLICIT;
        op->ConnectOutput(TensorUsage::OFM, ofmConn->tensor).Set(intermediateShape).Set(outQuant);
        RecordOptimisation(operation, op);
        operation->Disconnect();
        returnOp = op;
    }

    return returnOp;
}

// Converts int8/uint8 Sigmoid and Tanh to a LUT based solution
Operation *TFLiteGraphOptimiser::ConvertTanhSigmoidToLUT(Graph *const, Operation *const operation)
{
    auto returnOp = operation;
    auto opType = operation->Type();
    auto ifmConn = operation->Input(TensorUsage::IFM0);
    auto ifm = ifmConn->tensor.get();

    if ( !(opType == OpType::Sigmoid || opType == OpType::Tanh) )
    {
        return returnOp;
    }

    ArchOperatorQuery query;
    Set(query.ifm[0], ifm);
    Set(query.ofm, operation->OFM());
    ArchRequirements req;
    auto qresult = _constraints->OperatorQuery(opType, &query, &req);
    assert(qresult.Any(QueryResult::Native));

    if ( qresult.Any(QueryResult::HasRequirements) )
    {
        if ( req.req.Any(ArchRequirement::OpSubstitution) && (req.substitution == OpType::LUT) )
        {
            if ( ifm->Type() == DataType::Int16 )
            {
                returnOp = ConvertTanhSigmoidToLUT16(operation);
            }
            else
            {
                if ( opType == OpType::Tanh )
                {
                    returnOp = ConvertToLUT8(
                        operation, [](double x) -> double { return std::tanh(x); }, "tanh");
                }
                else
                {
                    returnOp = ConvertToLUT8(operation, ClampSigmoid8, "sigmoid");
                }
            }
        }
    }

    if ( operation != returnOp )
    {
        RecordOptimisation(operation, returnOp);
        operation->Disconnect();
    }

    return returnOp;
}


Operation *TFLiteGraphOptimiser::ConvertPrelu(Graph *const graph, Operation *const operation)
{
    // Lowering of PReLU
    // if all alpha values are equal:
    //     convert to LeakyReLU
    // else if all alpha values are < 1:
    //     convert to max(alpha * IFM, identity * IFM)
    // else:
    //     Convert to Minimum + Mul + ReLU + Add
    UNUSED(graph);
    auto returnOp = operation;
    auto opType = operation->Type();
    const auto ifmConn = operation->Input(TensorUsage::IFM0);
    const auto params = operation->Input(TensorUsage::Params);
    const auto ofmConn = operation->Output(TensorUsage::OFM);

    if ( opType == OpType::Prelu && ifmConn && ofmConn && params )
    {
        Quantization ofmQuant = ofmConn->quantization;
        Quantization ifmQuant = ifmConn->quantization;
        Quantization alphaQuant = params->quantization;

        Quantization noScaleQuant = Quantization::Unit();
        noScaleQuant.scales.clear();
        noScaleQuant.zeroPoints.clear();

        Quantization unitQuantOfmZp = Quantization::Unit();
        unitQuantOfmZp.zeroPoints.clear();
        unitQuantOfmZp.zeroPoints.push_back(ofmQuant.zeroPoints[0]);
        unitQuantOfmZp.type = QuantizationType::EXPLICIT;

        if ( params->tensor->IsConstant() )
        {
            // Special-cases for constant alpha-tensor
            auto alpha = params->tensor->View();
            int alphaSize = alpha.ViewShape().Elements();

            if ( alphaSize > 0 )
            {
                float alphaScale = 1.0f;
                int64_t alphaZp = 0;
                int alphaMin = 0;
                int alphaMax = 0;
                BufferReader<int> reader;
                switch ( params->tensor->Type() )
                {
                    case DataType::Int8:
                        reader = alpha.Values<int8_t, int>();
                        break;
                    case DataType::UInt8:
                        reader = alpha.Values<uint8_t, int>();
                        break;
                    case DataType::Int16:
                        reader = alpha.Values<int16_t, int>();
                        break;
                    default:
                        assert(false);
                };
                auto alphaMinMax = std::minmax_element(reader.begin(), reader.end());
                alphaMin = *alphaMinMax.first;
                alphaMax = *alphaMinMax.second;
                if ( alphaQuant.zeroPoints.size() )
                {
                    alphaZp = alphaQuant.zeroPoints[0];
                }
                if ( alphaQuant.scales.size() )
                {
                    alphaScale = float(alphaQuant.scales[0].Dequantize());
                }

                // rescale Min/Max
                float scaledAlphaMin = (alphaMin - alphaZp) * alphaScale;
                float scaledAlphaMax = (alphaMax - alphaZp) * alphaScale;

                if ( alphaMin == alphaMax )
                {
                    // If all alpha values are equal, we can convert to LeakyReLU
                    auto lreluOp = std::make_shared<Operation>(OpType::LeakyRelu);
                    lreluOp->CopyInput(TensorUsage::IFM, *ifmConn);
                    lreluOp->CopyInput(TensorUsage::Params, *params);
                    lreluOp->CopyOutput(TensorUsage::OFM, *ofmConn);
                    auto *attr = lreluOp->Attribute<leaky_relu_attr_t>();
                    attr->alpha = scaledAlphaMin;
                    returnOp = lreluOp.get();
                    RecordOptimisation(operation, returnOp);
                    operation->Disconnect();
                    return returnOp;
                }
                else if ( scaledAlphaMax <= 1 )
                {
                    // If all alpha values are <= 1
                    // We can convert to Max(alpha * IFM, identity * IFM)
                    //
                    //   IFM           IFM
                    //     \          /
                    //  Mul(alpha)  Mul(identity) - if ofmScale != ifmScale
                    //       \      /
                    //        Maximum
                    //
                    //

                    std::shared_ptr<Tensor> mulAlphaTens = ofmConn->tensor->Clone();
                    auto mulAlpha = std::make_shared<Operation>(OpType::Mul);
                    mulAlpha->CopyInput(TensorUsage::IFM0, *ifmConn);
                    mulAlpha->CopyInput(TensorUsage::IFM1, *params);
                    mulAlpha->Input(TensorUsage::IFM1)->Set(params->tensor->StorageShape());
                    mulAlpha->CopyOutput(TensorUsage::OFM, *ofmConn);
                    mulAlpha->ConnectOutput(TensorUsage::OFM, mulAlphaTens)
                        .Set(ofmConn->shape)
                        .Set(ofmConn->quantization)
                        .Set(ofmConn->slice)
                        .Set(RoundMode::DBL);
                    RecordOptimisation(operation, mulAlpha.get());

                    TensorConnection *alphaConn = mulAlpha->Output(TensorUsage::OFM);
                    TensorConnection *identityConn = ifmConn;
                    if ( ifmConn->quantization != ofmConn->quantization )
                    {
                        // If OFM/IFM quantization differ, we need to introduce
                        // an identity Mul operation to handle scaling.
                        std::shared_ptr<Tensor> oneTens;
                        if ( ifmConn->tensor->Type() == DataType::Int16 )
                        {
                            oneTens = CreateConstTensor("one_const", int16_t(1));
                        }
                        else
                        {
                            oneTens = CreateConstTensor("one_const", int8_t(1));
                        }
                        auto mulIdentity = MakeMulWithConstTensor("rescaled", *ifmConn, *ofmConn, oneTens, Quantization::Unit());
                        RecordOptimisation(operation, mulIdentity);
                        identityConn = mulIdentity->Output(TensorUsage::OFM);
                    }
                    // Create Maximum operation that combines identity and alphaConn
                    auto maxOp = std::make_shared<Operation>(OpType::Maximum);
                    maxOp->CopyInput(TensorUsage::IFM0, *alphaConn);
                    maxOp->CopyInput(TensorUsage::IFM1, *identityConn);
                    maxOp->CopyOutput(TensorUsage::OFM, *ofmConn);
                    maxOp->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
                    RecordOptimisation(operation, maxOp.get());
                    returnOp = maxOp.get();
                    operation->Disconnect();
                    return returnOp;
                }
            }
        }

        // Generic catch-all case
        // Convert to Minimum + Mul + ReLU + Add
        //
        //   x>0      x <= 0
        //   ReLU      Minimum(x, 0)
        //     \       /
        //      \     Mul(alpha)
        //       \   /
        //        Add
        //
        // ReLU is used for positive input values
        // Minimum(x,0) + Mul(alpha) is used for negative input values
        // Add sums the two cases

        std::shared_ptr<Tensor> zeroTens = CreateConstTensor("zero_const", ifmConn->tensor->Type(), 0);
        std::shared_ptr<Tensor> fmNegative = ifmConn->tensor->Clone();
        std::shared_ptr<Tensor> fmAlpha = ofmConn->tensor->Clone();
        std::shared_ptr<Tensor> fmScaled = ofmConn->tensor->Clone();

        // Select values < 0
        auto minOp = std::make_shared<Operation>(OpType::Minimum);
        minOp->CopyInput(TensorUsage::IFM0, *ifmConn);
        minOp->ConnectInput(TensorUsage::IFM1, zeroTens).Set(noScaleQuant);
        minOp->ConnectOutput(TensorUsage::OFM, fmNegative).Set(ifmConn->quantization);
        minOp->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
        RecordOptimisation(operation, minOp.get());

        // and multiply with alpha tensor
        auto mulAlpha = std::make_shared<Operation>(OpType::Mul);
        mulAlpha->CopyInput(TensorUsage::IFM0, *minOp->Output(TensorUsage::OFM));
        mulAlpha->CopyInput(TensorUsage::IFM1, *params);
        mulAlpha->ConnectOutput(TensorUsage::OFM, fmAlpha).Set(ofmConn->quantization).Set(RoundMode::DBL);
        RecordOptimisation(operation, mulAlpha.get());

        // Select (and scale) values > 0
        auto reluOp = std::make_shared<Operation>(OpType::Relu);
        reluOp->CopyInput(TensorUsage::IFM0, *ifmConn);
        reluOp->ConnectOutput(TensorUsage::OFM, fmScaled).Set(ofmConn->quantization);
        reluOp->Output(TensorUsage::OFM)->quantization.quantMin.push_back(ofmConn->quantization.zeroPoints[0]);
        reluOp->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
        RecordOptimisation(operation, reluOp.get());

        // Add scaled and alpha multiplied values
        auto addOp = std::make_shared<Operation>(OpType::Add);
        addOp->ConnectInput(TensorUsage::IFM0, fmAlpha).Set(unitQuantOfmZp);
        addOp->ConnectInput(TensorUsage::IFM1, fmScaled).Set(unitQuantOfmZp);
        addOp->CopyOutput(TensorUsage::OFM, *ofmConn);
        addOp->Output(TensorUsage::OFM)->Set(unitQuantOfmZp).Set(RoundMode::DBL);
        RecordOptimisation(operation, addOp.get());
        returnOp = addOp.get();
        operation->Disconnect();
    }
    return returnOp;
}

// Converts LeakyReLU
//
// alpha == 0
//   converted to ReLU
// alpha == -1
//   converted to Abs
// 8-bit LeakyReLU
//   converted to a LUT if unsupported by arch
// 16-bit LeakyReLU:
//   alpha > 1
//       Converted to Mul + (Mul) + Min if unsupported by arch
//       The extra Mul is needed if ifmQuant != ofmQuant
//   alpha <= 1
//       Converted to Mul + (Mul) + Max if unsupported by arch
Operation *TFLiteGraphOptimiser::ConvertLeakyRelu(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    auto returnOp = operation;
    auto opType = operation->Type();
    auto ifmConn = operation->Input(TensorUsage::IFM0);
    auto params = operation->Input(TensorUsage::Params);
    auto ofmConn = operation->Output(TensorUsage::OFM);

    // TODO MLBEDSW-8770: Investigate performance of leakyReLU optimisations
    if ( opType == OpType::LeakyRelu && ifmConn != nullptr && ofmConn != nullptr )
    {
        bool isConvertedPrelu = (params != nullptr);  // converted Prelu has params tensor
        const auto *attr = operation->Attribute<leaky_relu_attr_t>();
        float alpha = attr->alpha;
        auto ifm = ifmConn->tensor.get();
        auto ofm = ofmConn->tensor.get();

        if ( alpha == 0 || std::isinf(1 / alpha) )
        {
            // alpha == 0 can be converted to ReLU
            auto reluOp = MakeOperation(OpType::Relu, ifmConn, nullptr, ofmConn);
            reluOp->Output(TensorUsage::OFM)->quantization.quantMin.push_back(ofmConn->quantization.zeroPoints[0]);
            RecordOptimisation(operation, reluOp);
            returnOp = reluOp;
        }
        else if ( alpha == -1 )
        {
            // alpha == -1 can be converted to Abs
            auto absOp = MakeOperation(OpType::Abs, ifmConn, nullptr, ofmConn);
            RecordOptimisation(operation, absOp);
            returnOp = absOp;
        }
        else if ( (ifm->Type() == DataType::Int8 || ifm->Type() == DataType::UInt8) )
        {
            // convert to 8-bit LUT
            assert(ifm->Type() == ofm->Type());
            returnOp = Convert8bitLeakyReluToLUT(graph, operation, alpha);
            RecordOptimisation(operation, returnOp);
        }
        else if ( alpha < 0 || isConvertedPrelu || !_constraints->SupportsLeakyRelu(!IsScalingValidAndEqual(*ifmConn, *ofmConn), ifm->Type()) )
        {
            // Use 16-bit lowering to Mul + Max or Min + Mul + Relu + Add
            returnOp = ConvertLeakyRelu16bit(*ifmConn, *ofmConn, operation);
        }
    }

    if ( operation != returnOp )
    {
        operation->Disconnect();
    }

    return returnOp;
}

Operation *TFLiteGraphOptimiser::Convert8bitLeakyReluToLUT(Graph *const graph, Operation *const operation, float alpha)
{
    UNUSED(graph);
    auto returnOp = operation;
    auto opType = operation->Type();

    auto ifmConn = operation->Input(TensorUsage::IFM0);
    auto ofmConn = operation->Output(TensorUsage::OFM);
    auto params = operation->Input(TensorUsage::Params);
    auto ifm = ifmConn->tensor;
    auto ofm = ofmConn->tensor;
    const double ifmScale = ifmConn->quantization.scales.size() ? ifmConn->quantization.scales[0].Dequantize() : 1.0;
    const double ofmScale = ofmConn->quantization.scales.size() ? ofmConn->quantization.scales[0].Dequantize() : 1.0;
    const auto zpIn = ifmConn->quantization.zeroPoints.size() ? ifmConn->quantization.zeroPoints[0] : 0;
    const auto zpOut = ofmConn->quantization.zeroPoints.size() ? ofmConn->quantization.zeroPoints[0] : 0;
    int64_t scalar = 1;

    assert(opType == OpType::LeakyRelu);
    assert(DataTypeSizeBits(ifm->Type()) == 8);
    assert(ifm->Type() == ofm->Type());

    QuantizedScale identityScale = ElementwiseMulScale(ifmScale, 1.0, ofmScale);
    QuantizedScale alphaScale = ElementwiseMulScale(ifmScale, alpha, ofmScale);

    if ( params != nullptr )
    {
        // If alpha comes in as a params-tensor (e.g. converted PReLU)
        // the alpha-value also has quantization-parameters
        assert(params->tensor->IsConstant());
        assert(params->quantization.scales.size() > 0);
        assert(params->quantization.zeroPoints.size() > 0);
        auto view = params->tensor->View();
        QuantizedScale alphaQuant = QuantizedScale(alpha);
        auto alphaZp = params->quantization.zeroPoints[0];
        if ( params->tensor->Type() == DataType::Int8 )
        {
            scalar = int64_t(view.Values<int8_t>()[0]) - alphaZp;
            alphaQuant = params->quantization.scales[0];
        }
        else if ( params->tensor->Type() == DataType::UInt8 )
        {
            scalar = int64_t(view.Values<uint8_t>()[0]) - alphaZp;
            alphaQuant = params->quantization.scales[0];
        }
        alphaScale = ElementwiseMulScale(ifmScale, alphaQuant.Dequantize(), ofmScale);
    }

    // convert to left shift-positive notation
    identityScale.shift = 31 - identityScale.shift;
    alphaScale.shift = 31 - alphaScale.shift;

    int qMin = ifm->Type() == DataType::Int8 ? -128 : 0;
    int qMax = ifm->Type() == DataType::Int8 ? 127 : 255;

    std::vector<int8_t> lut;
    lut.reserve(256);
    for ( int x = qMin; x <= qMax; ++x )
    {
        int lutResult;
        if ( x < zpIn )
        {
            lutResult = int(zpOut + MultiplyByQuantizedMultiplier(int(scalar * (x - zpIn)), alphaScale));
        }
        else
        {
            lutResult = int(zpOut + MultiplyByQuantizedMultiplier(int(x - zpIn), identityScale));
        }
        lutResult = std::min(qMax, std::max(qMin, lutResult));
        lut.push_back(int8_t(lutResult));
    }
    auto lutTens = CreateConstTensor("lrelu", ifmConn->tensor->Type(), std::make_shared<Buffer>(std::move(lut)));
    // The LUT must be applied without any preceding rescaling (the LUT itself performs the rescale),
    // so even if the OFM has a different scale than the IFM, the generated OFM scale instructions
    // should be the same as the IFM
    returnOp = CreateLUT(ifmConn->tensor, lutTens, ifmConn->quantization, ifmConn->quantization, lutTens->Type(),
        &ifmConn->shape, ofmConn->tensor, ifmConn->slice, ofmConn->slice);
    returnOp->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
    return returnOp;
}

// Converts RSqrt to a LUT based solution.
Operation *TFLiteGraphOptimiser::ConvertRSqrtToLUT(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    auto returnOp = operation;
    auto opType = operation->Type();
    auto ifmConn = operation->Input(TensorUsage::IFM0);
    auto ofmConn = operation->Output(TensorUsage::OFM);

    // LUT has been generated by printing the output from the reference.
    // clang-format off
    static const int32_t kRSqrtLut[] =
    {
        0x00000000, 0x00100000, 0x000b504e, 0x00093cd4, 0x00080000, 0x000727c9, 0x0006882f, 0x00060c24,
        0x0005a827, 0x00055555, 0x00050f45, 0x0004d2fe, 0x00049e6a, 0x00047007, 0x000446b4, 0x00042195,
        0x00040000, 0x0003e16d, 0x0003c570, 0x0003abb0, 0x000393e5, 0x00037dd2, 0x00036945, 0x00035613,
        0x00034418, 0x00033333, 0x0003234b, 0x00031447, 0x00030612, 0x0002f89c, 0x0002ebd3, 0x0002dfaa,
        0x0002d414, 0x0002c906, 0x0002be75, 0x0002b45a, 0x0002aaab, 0x0002a161, 0x00029875, 0x00028fe3,
        0x000287a2, 0x00027fb0, 0x00027807, 0x000270a2, 0x0002697f, 0x00026298, 0x00025bec, 0x00025577,
        0x00024f35, 0x00024925, 0x00024343, 0x00023d8e, 0x00023803, 0x000232a1, 0x00022d65, 0x0002284e,
        0x0002235a, 0x00021e87, 0x000219d5, 0x00021541, 0x000210cb, 0x00020c70, 0x00020831, 0x0002040c,
        0x00020000, 0x0001fc0c, 0x0001f82f, 0x0001f468, 0x0001f0b7, 0x0001ed1a, 0x0001e991, 0x0001e61b,
        0x0001e2b8, 0x0001df67, 0x0001dc26, 0x0001d8f7, 0x0001d5d8, 0x0001d2c8, 0x0001cfc8, 0x0001ccd6,
        0x0001c9f2, 0x0001c71c, 0x0001c454, 0x0001c198, 0x0001bee9, 0x0001bc46, 0x0001b9af, 0x0001b723,
        0x0001b4a3, 0x0001b22d, 0x0001afc2, 0x0001ad61, 0x0001ab0a, 0x0001a8bc, 0x0001a678, 0x0001a43e,
        0x0001a20c, 0x00019fe3, 0x00019dc2, 0x00019baa, 0x0001999a, 0x00019791, 0x00019590, 0x00019397,
        0x000191a5, 0x00018fbb, 0x00018dd7, 0x00018bfa, 0x00018a23, 0x00018853, 0x0001868a, 0x000184c6,
        0x00018309, 0x00018152, 0x00017fa0, 0x00017df4, 0x00017c4e, 0x00017aad, 0x00017911, 0x0001777b,
        0x000175e9, 0x0001745d, 0x000172d6, 0x00017153, 0x00016fd5, 0x00016e5b, 0x00016ce7, 0x00016b76,
        0x00016a0a, 0x000168a2, 0x0001673e, 0x000165de, 0x00016483, 0x0001632b, 0x000161d7, 0x00016087,
        0x00015f3b, 0x00015df2, 0x00015cad, 0x00015b6b, 0x00015a2d, 0x000158f2, 0x000157bb, 0x00015686,
        0x00015555, 0x00015427, 0x000152fd, 0x000151d5, 0x000150b0, 0x00014f8f, 0x00014e70, 0x00014d54,
        0x00014c3b, 0x00014b24, 0x00014a11, 0x00014900, 0x000147f1, 0x000146e5, 0x000145dc, 0x000144d5,
        0x000143d1, 0x000142cf, 0x000141d0, 0x000140d3, 0x00013fd8, 0x00013ee0, 0x00013de9, 0x00013cf5,
        0x00013c03, 0x00013b14, 0x00013a26, 0x0001393b, 0x00013851, 0x0001376a, 0x00013684, 0x000135a1,
        0x000134bf, 0x000133e0, 0x00013302, 0x00013226, 0x0001314c, 0x00013074, 0x00012f9e, 0x00012ec9,
        0x00012df6, 0x00012d25, 0x00012c55, 0x00012b87, 0x00012abb, 0x000129f1, 0x00012928, 0x00012860,
        0x0001279a, 0x000126d6, 0x00012613, 0x00012552, 0x00012492, 0x000123d4, 0x00012317, 0x0001225c,
        0x000121a2, 0x000120e9, 0x00012032, 0x00011f7c, 0x00011ec7, 0x00011e14, 0x00011d62, 0x00011cb1,
        0x00011c02, 0x00011b54, 0x00011aa7, 0x000119fb, 0x00011950, 0x000118a7, 0x000117ff, 0x00011758,
        0x000116b3, 0x0001160e, 0x0001156b, 0x000114c8, 0x00011427, 0x00011387, 0x000112e8, 0x0001124a,
        0x000111ad, 0x00011111, 0x00011076, 0x00010fdc, 0x00010f44, 0x00010eac, 0x00010e15, 0x00010d7f,
        0x00010cea, 0x00010c56, 0x00010bc4, 0x00010b32, 0x00010aa0, 0x00010a10, 0x00010981, 0x000108f3,
        0x00010865, 0x000107d9, 0x0001074d, 0x000106c2, 0x00010638, 0x000105af, 0x00010527, 0x0001049f,
        0x00010419, 0x00010393, 0x0001030e, 0x0001028a, 0x00010206, 0x00010183, 0x00010102, 0x00010080
    };
    // clang-format on

    if ( opType == OpType::Rsqrt && ifmConn->tensor->Type() == DataType::Int8 && ofmConn->tensor->Type() == DataType::Int8 )
    {
        const int kShift = 20;
        const int qMin = -128;
        const int qMax = 127;
        const auto zpIn = ifmConn->quantization.zeroPoints[0];
        const auto zpOut = ofmConn->quantization.zeroPoints[0];
        const auto ifmScale = ifmConn->quantization.scales[0].Dequantize();
        const auto ofmScale = ofmConn->quantization.scales[0].Dequantize();
        double scale = 1.0 / double(std::sqrt(float(ifmScale)) * float(ofmScale));
        QuantizedScale qScale = QuantizedScale(scale);
        // convert to left shift-positive notation
        qScale.shift = 31 - qScale.shift - kShift;

        std::vector<uint8_t> lut;
        lut.reserve(256);
        lut.push_back(qMax);
        for ( int x = qMin + 1; x <= qMax; ++x )
        {
            int index = std::max(0, x - int(zpIn));
            auto value = zpOut + MultiplyByQuantizedMultiplier(kRSqrtLut[index], qScale);
            lut.push_back(uint8_t(std::min(qMax, std::max(qMin, int(value)))));
        }

        auto lutTens = CreateConstTensor("rsqrt", ifmConn->tensor->Type(), std::make_shared<Buffer>(std::move(lut)));
        returnOp = CreateLUT(ifmConn->tensor, lutTens, ifmConn->quantization, ifmConn->quantization, lutTens->Type(),
            &ifmConn->shape, ofmConn->tensor, ifmConn->slice, ofmConn->slice);
        returnOp->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
    }
    else if ( opType == OpType::Rsqrt && ifmConn->tensor->Type() == DataType::Int16 && ofmConn->tensor->Type() == DataType::Int16 )
    {
        float ofmScale = float(operation->Output(TensorUsage::OFM)->quantization.scales[0].Dequantize());
        returnOp = ConvertToInterpolatingLUT16(
            operation,
            [&ofmScale](float x) -> float
            {
                if ( x <= 0.0f )
                {
                    return IntegerMax(DataType::Int16) * ofmScale;
                }
                else
                {
                    return 1.0f / std::sqrt(x);
                }
            },
            "Rsqrt16(interp)");
    }

    if ( operation != returnOp )
    {
        RecordOptimisation(operation, returnOp);
        operation->Disconnect();
    }

    return returnOp;
}

int TFLiteGraphOptimiser::GetPadValue(BufferReader<int> &padValues, int numPadValues, PadAxis axis)
{
    int index = numPadValues - static_cast<int>(axis);
    return index < 0 ? 0 : padValues[index];
}

BufferReader<int> TFLiteGraphOptimiser::GetPadValuesFromTensor(const std::shared_ptr<Tensor> tensor)
{
    BufferReader<int> padValues;
    if ( tensor->Type() == DataType::Int32 )
    {
        padValues = tensor->View().Values<int32_t, int>();
    }
    else
    {
        assert(tensor->Type() == DataType::Int64);
        padValues = tensor->View().Values<int64_t, int>();
    }
    return padValues;
}

// Based on explicit padding provided in a PAD operation, returns adjusted value for
// padAfter that provides equivalent results when used with explicit padding
int TFLiteGraphOptimiser::CalcPadAfter(int inputSize, int stride, int filterSize, int padBefore, int padAfter)
{
    int totalPadding = NeededTotalPadding(inputSize, stride, filterSize);
    // The bottom/right padding might need downward adjustment depending on stride/input size
    int remainderDiff = padAfter % stride - (totalPadding - padBefore) % stride;
    return std::max(0, padAfter - remainderDiff - (remainderDiff >= 0 ? 0 : stride));
}

// Tries to completely remove a PAD operator by using explicit padding.
// E.g. a PAD operation that pads 1, followed by a CONV with VALID padding and kernel size 3
// is rewritten such that the PAD is removed, and the CONV uses explicit padding.
// Converts tens1 -> PAD -> tens2 -> CONV to tens1 -> CONV
// This is the most efficient way to implement PAD, but cannot be done for all pad sizes.
Operation *TFLiteGraphOptimiser::ReplacePadByExplicitPadding(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    if ( IsConvolution(operation->Type()) && operation->Type() != OpType::TransposeConv2D &&
         operation->Kernel()->Padding().IsZero() && operation->IFM(0)->Writers().size() == 1 )
    {
        // Potential for future optimization: in certain cases also Pad+AvgPool can be handled
        // by changing to Depthwise.
        auto padOp = operation->IFM(0)->Writers()[0].get();
        if ( padOp->Type() != OpType::Pad || padOp->Attribute<pad_attr_t>()->pad_const != 0 )
        {
            return operation;
        }
        auto padIfmConn = padOp->Input(TensorUsage::IFM0);
        auto padOfmConn = padOp->Output(TensorUsage::OFM);
        const auto &padIfm = padOp->IFM(0);
        const auto &padOfm = padOp->OFM();

        if ( padIfm->Type() != padOfm->Type() || !IsScalingValidAndEqual(*padIfmConn, *padOfmConn) )
        {
            return operation;
        }
        auto padTensor = padOp->Input(TensorUsage::Params)->tensor;
        BufferReader<int> padValues = GetPadValuesFromTensor(padTensor);
        int numPadValues = padTensor->View().Elements();
        int top = GetPadValue(padValues, numPadValues, PadAxis::Top);
        int bottom = GetPadValue(padValues, numPadValues, PadAxis::Bottom);
        int left = GetPadValue(padValues, numPadValues, PadAxis::Left);
        int right = GetPadValue(padValues, numPadValues, PadAxis::Right);

        const auto &k = operation->Kernel();
        const auto &kwh = k->DilatedWH();
        if ( left + right >= kwh.x || top + bottom >= kwh.y )
        {
            // Too much padding
            return operation;
        }
        const auto &ifmShape = padOp->Input(TensorUsage::IFM0)->shape;
        int bottomPad = CalcPadAfter(ifmShape.Height(), k->Stride().y, kwh.y, top, bottom);
        int rightPad = CalcPadAfter(ifmShape.Width(), k->Stride().x, kwh.x, left, right);
        // Adjust the padding attributes of the convolution operator
        auto kernel = std::make_unique<Kernel>(
            Kernel(k->Size(), k->Stride(), k->Dilation(), k->DepthMultiplier(), Margin(top, left, bottomPad, rightPad)));
        operation->SetKernel(std::move(kernel));
        operation->CopyInput(TensorUsage::IFM0, *(padOp->Input(TensorUsage::IFM0)));
        if ( padOfm->Readers().empty() )
        {
            // Bypass the PAD operator
            padOp->Disconnect();
        }
    }
    return operation;
}

// Lower PadV2 to TOSA Pad
Operation *TFLiteGraphOptimiser::ConvertPadV2(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    if ( operation->Type() == OpType::PadV2 )
    {
        auto padOp = std::make_shared<Operation>(OpType::Pad);
        padOp->CopyInput(TensorUsage::IFM, *operation->Input(TensorUsage::IFM));
        padOp->CopyInput(TensorUsage::Params, *operation->Input(TensorUsage::Params0));
        const auto &ofmConn = operation->Output(TensorUsage::OFM);
        padOp->CopyOutput(TensorUsage::OFM, *ofmConn);
        const auto &attr = padOp->Attribute<pad_attr_t>();
        const auto padConstTens = operation->Input(TensorUsage::Params1)->tensor;
        // This is undoing the existing zero point adjustment to counteract the zero point adjustment
        // which is done in GraphIR lowering of Pad.
        uint8_t zeroPoint = ofmConn->quantization.IsValid() ? uint8_t(ofmConn->quantization.zeroPoints[0]) : 0;
        attr->pad_const = padConstTens->View().Values<int>(padConstTens->Type())[0] - zeroPoint;

        RecordOptimisation(operation, padOp.get());
        operation->Disconnect();
        return padOp.get();
    }
    return operation;
}

void TFLiteGraphOptimiser::MakeMemoryCopyForMirrorPad(const Operation *operation, TensorConnection *ifmConn, const Shape &readShape,
    const Shape &readOffset, TensorConnection *ofmConn, const Shape &writeShape, const Shape &writeOffset, ReverseType reverseAxis)
{
    auto op = std::make_shared<Operation>(OpType::MemoryCopy);

    op->ConnectInput(TensorUsage::IFM0, ifmConn->tensor).Set(ifmConn->shape).Set(ifmConn->quantization).Set({readOffset, readShape});

    op->ConnectOutput(TensorUsage::OFM, ofmConn->tensor)
        .Set(ofmConn->shape)
        .Set(ofmConn->quantization)
        .Set({writeOffset, writeShape})
        .Set(RoundMode::NATURAL)
        .Set(reverseAxis);

    RecordOptimisation(operation, op.get());
}

Operation *TFLiteGraphOptimiser::ConvertMirrorPad(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    if ( operation->Type() != OpType::MirrorPad )
    {
        return operation;
    }
    const auto &ifmConn = operation->Input(TensorUsage::IFM0);
    const auto &ifmShape = ifmConn->shape;
    const auto &ofmConn = operation->Output(TensorUsage::OFM);
    const auto &ofmShape = ofmConn->shape;
    const auto &paramsConn = operation->Input(TensorUsage::Params);

    BufferReader<int> padValues = GetPadValuesFromTensor(paramsConn->tensor);
    int numPadValues = paramsConn->tensor->View().Elements();
    int top = GetPadValue(padValues, numPadValues, PadAxis::Top);
    int bottom = GetPadValue(padValues, numPadValues, PadAxis::Bottom);
    int left = GetPadValue(padValues, numPadValues, PadAxis::Left);
    int right = GetPadValue(padValues, numPadValues, PadAxis::Right);

    auto *attr = operation->Attribute<mirror_pad_mode_attr_t>();
    assert((attr->mode >= tflite::MirrorPadMode::MIN) && (attr->mode <= tflite::MirrorPadMode::MAX));
    int modeOffset = (attr->mode == tflite::MirrorPadMode::REFLECT) ? 1 : 0;

    // Create MemoryCopy op that copies IFM to the right place inside the OFM
    Shape zeroShape = ofmShape.WithZeros();
    auto mainOp = MakeMemoryCopyForConcat(ofmConn, ifmConn, zeroShape.WithHeight(top).WithWidth(left));
    RecordOptimisation(operation, mainOp.get());

    // Add operations that fill the borders of the OFM
    if ( top > 0 )
    {
        Shape shape = ifmShape.WithHeight(top);
        Shape readOffset = zeroShape.WithHeight(modeOffset);
        Shape writeOffset = zeroShape.WithWidth(left);
        MakeMemoryCopyForMirrorPad(operation, ifmConn, shape, readOffset, ofmConn, shape, writeOffset, ReverseType::H);
    }
    if ( bottom > 0 )
    {
        Shape shape = ifmShape.WithHeight(bottom);
        Shape readOffset = zeroShape.WithHeight(ifmShape.Height() - bottom - modeOffset);
        Shape writeOffset = zeroShape.WithWidth(left).WithHeight(ofmShape.Height() - bottom);
        MakeMemoryCopyForMirrorPad(operation, ifmConn, shape, readOffset, ofmConn, shape, writeOffset, ReverseType::H);
    }
    if ( left > 0 )
    {
        Shape shape = ofmShape.WithWidth(left);
        Shape readOffset = zeroShape.WithWidth(left + modeOffset);
        Shape writeOffset = zeroShape;
        MakeMemoryCopyForMirrorPad(operation, ofmConn, shape, readOffset, ofmConn, shape, writeOffset, ReverseType::W);
    }
    if ( right > 0 )
    {
        Shape shape = ofmShape.WithWidth(right);
        Shape readOffset = zeroShape.WithWidth(ifmShape.Width() + left - right - modeOffset);
        Shape writeOffset = zeroShape.WithWidth(ofmShape.Width() - right);
        MakeMemoryCopyForMirrorPad(operation, ofmConn, shape, readOffset, ofmConn, shape, writeOffset, ReverseType::W);
    }

    operation->Disconnect();
    return mainOp.get();
}

Operation *TFLiteGraphOptimiser::ConvertZeroPoint(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    auto opType = operation->Type();
    if ( opType == OpType::Passthrough )
    {
        return operation;
    }

    bool zeroPoint0ForType =
        opType == OpType::AvgPool || opType == OpType::Resize || opType == OpType::CLZ || opType == OpType::SHL || opType == OpType::Div;
    for ( auto [usage, ifmConn] : operation->Inputs().pairs() )
    {
        if ( IsIFM(usage) )
        {
            if ( zeroPoint0ForType || DataTypeSizeBits(ifmConn.tensor->Type()) >= 32 )
                ifmConn.quantization.zeroPoints.clear();
        }
    }
    for ( auto [usage, ofmConn] : operation->Outputs().pairs() )
    {
        if ( IsOFM(usage) )
        {
            if ( zeroPoint0ForType || opType == OpType::ArgMax ) ofmConn.quantization.zeroPoints.clear();
        }
    }
    return operation;
}

// Return a slice of a tensor
template<typename TYPE>
static std::shared_ptr<Tensor>
SliceConstTensor(const TensorConnection *conn, const Shape &sliceShape, const Shape &sliceOffset, const std::string &Name)
{
    assert((sliceShape.Size() == 4) && (sliceOffset.Size() == 4));

    // Create a sub-view to read only a slice of the tensor
    auto subBufferView = conn->tensor->View().SubView(sliceOffset, sliceShape);
    BufferReader<TYPE> values = subBufferView.Values<TYPE, TYPE>();

    // Create a new buffer to hold the slice
    int size = sliceShape.Elements();
    auto newBuffer = std::make_shared<Buffer>(std::make_unique<TYPE[]>(size), size);
    BufferView newBufferView(newBuffer, 0, 8 * sizeof(TYPE), sliceShape, {});
    auto newValues = newBufferView.WritableValues<TYPE>();

    // Copy the values over to the new buffer
    for ( int n = 0; n < sliceShape.Batch(); n++ )
    {
        for ( int h = 0; h < sliceShape.Height(); h++ )
        {
            for ( int w = 0; w < sliceShape.Width(); w++ )
            {
                for ( int c = 0; c < sliceShape.Depth(); c++ )
                {
                    Shape pos({n, h, w, c}, sliceShape.Size());
                    newValues[pos] = values[pos];
                }
            }
        }
    }

    return std::make_shared<Tensor>(Name, conn->tensor->Type(), sliceShape, std::move(newBuffer));
}

// Converts a convolution group with N groups into N * Conv2D ops each operating on a 1/N part of
// the original channels. Finally, all of the individual results will be concatenated depth-wise into
// the OFM tensor.
Operation *TFLiteGraphOptimiser::ConvertConvolutionGroup(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    if ( operation->Type() != OpType::Conv2D )
    {
        return operation;
    }

    const auto &ifmConn = operation->Input(TensorUsage::IFM0);
    const auto &ifmShape = ifmConn->shape;
    const auto &weightConn = operation->Input(TensorUsage::Weights);
    const auto &weightShape = weightConn->shape;
    const auto &biasConn = operation->Input(TensorUsage::Scales);
    const auto &biasShape = biasConn->shape;
    const auto &ofmConn = operation->Output(TensorUsage::OFM);
    const auto &ofmShape = ofmConn->shape;

    // Calculate the number of convolution groups based of the shape of the IFM read by
    // the convolution, accounting for partial reads of the IFM.
    auto ifmReadShape = ifmConn->slice.shape.IsEmpty() ? ifmShape : ifmConn->slice.shape;
    auto numGroups = ifmReadShape.Depth() / weightShape.Depth();
    if ( numGroups == 1 )
    {
        return operation;
    }

    // Create final Concat operation
    auto concatOp = std::make_shared<Operation>(OpType::Concat);
    concatOp->CopyOutput(TensorUsage::OFM, *ofmConn);
    concatOp->Attribute<axis_attr_t>()->axis = -1;

    // Create 'numGroups' number of convolutions, each reading a depth-wise slice of the IFM.
    int kernelsPerGroup = weightShape.Batch() / numGroups;
    Shape zeroShape = ifmReadShape.WithZeros();
    Shape ifmSlice = ifmReadShape.WithDepth(ifmReadShape.Depth() / numGroups);
    Shape ofmSlice = ofmShape.WithDepth(ofmShape.Depth() / numGroups);
    Shape weightSlice = weightShape.WithBatch(kernelsPerGroup);
    Shape biasSlice = biasShape.WithDepth(kernelsPerGroup);

    const auto &weightName = weightConn->tensor->Name();
    const auto &ofmName = ofmConn->tensor->Name();
    Operation *finalOp = nullptr;
    for ( int i = 0; i < numGroups; i++ )
    {
        // Create Convolution and connect the IFM sliced and offset
        auto convGroupOp = std::make_shared<Operation>(OpType::Conv2D);
        convGroupOp->ConnectInput(TensorUsage::IFM0, ifmConn->tensor)
            .Set(ifmReadShape)
            .Set(ifmConn->quantization)
            .Set({zeroShape.WithDepth(i * ifmSlice.Depth()), ifmSlice});

        // Create and connect intermediate OFM
        auto ofmConvGroup = std::make_shared<Tensor>(ofmName + "_convgroup_output" + std::to_string(i), ofmConn->tensor->Type(), ofmSlice);
        convGroupOp->ConnectOutput(TensorUsage::OFM, ofmConvGroup).Set(ofmSlice).Set(ofmConn->quantization).Set(ofmConn->rounding);

        // Copy the kernel from the original operation
        convGroupOp->SetKernel(std::make_unique<Kernel>(*operation->Kernel()));

        // Extract a slice out of the weight tensor
        assert(weightConn->tensor->Type() & DataType::Bits8);
        Shape weightOffset = zeroShape.WithBatch(i * weightSlice.Batch());
        auto weightSubTensor =
            weightConn->tensor->Type() == DataType::UInt8 ?
                SliceConstTensor<uint8_t>(weightConn, weightSlice, weightOffset, weightName + "weights" + std::to_string(i)) :
                SliceConstTensor<int8_t>(weightConn, weightSlice, weightOffset, weightName + "weights" + std::to_string(i));

        // Slice quantization info for weights and bias
        Quantization newWeightQuant = weightConn->quantization;
        newWeightQuant.scales.clear();
        newWeightQuant.zeroPoints.clear();
        Quantization newBiasQuant = biasConn->quantization;
        newBiasQuant.scales.clear();
        newBiasQuant.zeroPoints.clear();
        for ( int j = 0; j < kernelsPerGroup; j++ )
        {
            newWeightQuant.scales.push_back(weightConn->quantization.scales[j + (i * kernelsPerGroup)]);
            newWeightQuant.zeroPoints.push_back(weightConn->quantization.zeroPoints[j + (i * kernelsPerGroup)]);
            newBiasQuant.scales.push_back(biasConn->quantization.scales[j + (i * kernelsPerGroup)]);
            newBiasQuant.zeroPoints.push_back(biasConn->quantization.zeroPoints[j + (i * kernelsPerGroup)]);
        }

        // Connect weights slice
        convGroupOp->ConnectInput(TensorUsage::Weights, weightSubTensor).Set(weightShape).Set(newWeightQuant);

        // Connect the bias and scales slice
        convGroupOp->ConnectInput(TensorUsage::Scales, biasConn->tensor)
            .Set(biasShape)
            .Set(newBiasQuant)
            .Set({zeroShape.WithDepth(i * biasSlice.Depth()), biasSlice});

        // Connect intermediate OFM to Concat op
        concatOp->ConnectInput(MakeTensorUsage(TensorUsage::IFM, i), ofmConvGroup)
            .Set(ofmSlice)
            .Set(convGroupOp->Output(TensorUsage::OFM)->quantization);

        RecordOptimisation(operation, convGroupOp.get());
    }

    RecordOptimisation(operation, concatOp.get());
    operation->Disconnect();
    return concatOp.get();
}

TFLiteGraphOptimiser::TFLiteGraphOptimiser(IArchitectureConstraints *constraints, const GraphOptimiserOptions &options, OptimiserDatabase *db) :
        GraphOptimiser(constraints, options, db)
{
    _softmax = std::make_unique<Softmax>(db);
}

void TFLiteGraphOptimiser::OptimiseGraph(Graph *graph)
{
    for ( auto iOpt = GraphOptimisationSteps().begin(); iOpt != GraphOptimisationSteps().end(); ++iOpt )
    {
        LOG_TRACE1("GraphOptimiser {0}/{1}\n", std::distance(GraphOptimisationSteps().begin(), iOpt) + 1,
            GraphOptimisationSteps().size());
        // Check if function lists are empty. Do not call for step that only contain disabled debug functions.
        if ( !iOpt->opFunction.empty() || !iOpt->tensorFunction.empty() )
        {
            RewriteGraph<TFLiteGraphOptimiser>(graph, *iOpt);
        }
    }
}

}  // namespace regor
