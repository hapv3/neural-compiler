//
// SPDX-FileCopyrightText: Copyright 2021-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "compiler/softmax.hpp"

#include "common/numeric_util.hpp"
#include "common/scaling.hpp"
#include "lut_util.hpp"
#include "operation.hpp"
#include "operation_util.hpp"

#include <fixedpoint/fixedpoint.h>
#include <algorithm>
#include <vector>

namespace regor
{

Softmax::Softmax(OptimiserDatabase *db, IArchitectureConstraints *constraints, double int16NegExpRange) :
        _db(db), _constraints(constraints), _negExpRange(int16NegExpRange)
{
    assert(int16NegExpRange > 0 && int16NegExpRange < 65535);
}

Operation *Softmax::ConvertOp(Operation *const operation)
{
    auto returnOp = operation;

    if ( OpType::Softmax == operation->Type() )
    {
        auto ifmConn = operation->Input(TensorUsage::IFM0);
        auto ofmConn = operation->Output(TensorUsage::OFM);
        auto ifm = ifmConn->tensor.get();
        auto ofm = ofmConn->tensor.get();

        if ( ifm->Type() == ofm->Type() || (ifm->Type() == DataType::Int8 && ofm->Type() == DataType::Int16) )
        {
            // Reshape if needed
            auto fullShape = Shape::PadAxes(ifmConn->shape, 4, 1);
            if ( fullShape.Batch() > 1 )
            {
                fullShape = fullShape.WithHeight(fullShape.Batch() * fullShape.Height()).WithBatch(1);
            }
            ifmConn->shape = fullShape;
            ofmConn->shape = std::move(fullShape);

            if ( ifm->Type() == DataType::Int8 || ifm->Type() == DataType::UInt8 )
            {
                returnOp = GetGraph8Bit(operation, ifmConn, ofmConn);
            }
            else if ( ifm->Type() == DataType::Int16 )
            {
                returnOp = GetGraphInt16(operation, ifmConn, ofmConn);
            }
        }
        if ( operation != returnOp )
        {
            operation->Disconnect();
        }
    }

    return returnOp;
}


void Softmax::RecordOptimisation(Operation *const operation, Operation *op)
{
    if ( _db )
    {
        _db->AddOptimised(*operation, op);
    }
}

Operation *Softmax::CreateTransposeMaxpool(Operation *const operation, TensorConnection *ifmConn, const Shape &ifmShape,
    const Shape &transposePerm, const Shape &transposeOFMShape, const Shape &maxPoolOFMStorageShape, const Quantization &noScaleQuant)
{
    std::shared_ptr transposeTens = ifmConn->tensor->Clone();
    transposeTens->SetName(ifmConn->tensor->Name() + "_transpose");
    transposeTens->SetStorageShape(transposeOFMShape);

    auto transposeOp = std::make_shared<Operation>(OpType::Transpose);
    auto transposeAttr = transposeOp->Attribute<transpose_attr_t>();
    transposeAttr->perm = transposePerm;
    transposeOp->CopyInput(TensorUsage::IFM, *ifmConn);
    transposeOp->Input(TensorUsage::IFM)->Set(ifmShape).Set(Quantization::Unit());
    const auto &transposeConn = transposeOp->ConnectOutput(TensorUsage::OFM, transposeTens);
    RecordOptimisation(operation, transposeOp.get());

    auto maxOp = std::make_shared<Operation>(OpType::MaxPool);
    auto transposeShape = transposeConn.shape;
    int height = transposeShape.Height();
    auto kernel = std::make_unique<Kernel>(Point2i(1, height), Point2i(1, 1), Point2i(1, 1));
    auto ofm = std::make_shared<Tensor>(transposeTens->Name() + "/maxpool", transposeTens->Type());
    ofm->SetStorageShape(maxPoolOFMStorageShape);
    maxOp->SetKernel(std::move(kernel));
    maxOp->ConnectInput(TensorUsage::IFM, transposeTens).Set(ifmConn->quantization);
    maxOp->Input(TensorUsage::IFM)->shape = transposeShape;
    maxOp->ConnectOutput(TensorUsage::OFM, ofm).Set(noScaleQuant);
    maxOp->Output(TensorUsage::OFM)->shape = Shape(1, 1, transposeShape.Width(), transposeShape.Depth());

    return maxOp.get();
}

Operation *Softmax::GetGraph8Bit(Operation *const operation, TensorConnection *ifmConn, TensorConnection *ofmConn)
{
    const auto &ifmQuant = ifmConn->quantization;
    auto *softmax = operation->Attribute<softmax_attr_t>();
    auto expTable = GenerateExpTable(double(softmax->beta), ifmQuant.scales[0].Dequantize());
    auto noScaleQuant = ifmConn->quantization;
    noScaleQuant.scales.clear();
    auto noScaleQuantZp0 = noScaleQuant;
    noScaleQuantZp0.zeroPoints[0] = 0;
    auto oneScaleQuant = ifmConn->quantization;
    oneScaleQuant.scales[0] = {1, 0};
    oneScaleQuant.zeroPoints[0] = 0;
    auto twoScaleQuant = oneScaleQuant;
    twoScaleQuant.scales[0] = {2, 0};

    // Base name and index to use for new tensors
    const auto &tensorBaseName = ifmConn->tensor->Name();
    int tensorIdx = 0;

    const Shape ifmShape3D = ReshapeTo3D(ifmConn->shape, {2, 1, 1}, 1);

    // PASS 0 - Depthwise Maxpool
    Operation *op;
    auto queryResult = _constraints->OperatorQuery(OpType::Transpose);
    bool hasFastTransposeSupport = queryResult.Any(QueryResult::Native) && !queryResult.Any(QueryResult::Emulated);
    Shape transposeOFMShape;
    Shape transposePerm;
    Shape maxPoolOFMStorageShape;
    // Determine transpose candidates and corresponding shapes
    if ( ifmShape3D.Width() > 1 )
    {
        transposePerm = {2, 0, 1};
        transposeOFMShape = ifmShape3D.Extract(2, 0, 1);
        maxPoolOFMStorageShape = Shape(1, transposeOFMShape.Width(), transposeOFMShape.Depth(), 1);
    }
    else
    {
        transposePerm = {2, 1, 0};
        transposeOFMShape = ifmShape3D.Extract(2, 1, 0);
        maxPoolOFMStorageShape = Shape(1, transposeOFMShape.Depth(), 1, 1);
    }
    // Insert transpose when it has native support if not a no-op and depth becomes large enough to benefit
    if ( hasFastTransposeSupport && ifmConn->shape.Depth() != ifmConn->shape.Elements() && transposeOFMShape.Depth() >= 16 )
    {
        op = CreateTransposeMaxpool(operation, ifmConn, ifmShape3D, transposePerm, transposeOFMShape, maxPoolOFMStorageShape, noScaleQuant);
    }
    else
    {
        op = CreateDepthwiseMaxpool(ifmConn->tensor, ifmConn->shape, ifmConn->quantization, noScaleQuant);
    }
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    RecordOptimisation(operation, op);
    auto ifmMax = op->Output(TensorUsage::OFM)->tensor;
    ifmMax->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));

    // PASS 1 - Sub
    auto subQuant = oneScaleQuant;
    subQuant.zeroPoints[0] = 127;
    op = CreateSub(ifmConn->tensor, ifmMax, ifmConn->quantization, noScaleQuant, subQuant, DataType::Int8, &ifmConn->shape);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto ifm_sub = op->Output(TensorUsage::OFM)->tensor;
    ifm_sub->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 1.5 - LUT(exp)
    auto expLut = CreateConstTensor("exp_lut", DataType::Int32, std::make_shared<Buffer>(std::move(expTable)));
    op = CreateLUT(ifm_sub, expLut, subQuant, subQuant);
    auto ifm_exp = op->Output(TensorUsage::OFM)->tensor;
    ifm_exp->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    RecordOptimisation(operation, op);

    // PASS 2 - ASR
    auto right_shift12 = CreateConstTensor("right_shift12", 12);
    op = CreateAsr(ifm_exp, right_shift12, subQuant, noScaleQuant, noScaleQuantZp0);
    op->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
    op->Attribute<asr_attr_t>()->round = true;
    auto rescaled_exp = op->Output(TensorUsage::OFM)->tensor;
    rescaled_exp->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 3 - Reduce sum
    op = CreateReduceSum(rescaled_exp, noScaleQuantZp0, noScaleQuantZp0);
    op->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
    auto sum_of_exp = op->Output(TensorUsage::OFM)->tensor;
    sum_of_exp->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 4 - CLZ
    op = CreateClz(sum_of_exp, noScaleQuantZp0, noScaleQuantZp0);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto headroom_plus_one = op->Output(TensorUsage::OFM)->tensor;
    headroom_plus_one->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 5 - Sub
    auto headroom_offset = CreateConstTensor("headroom_offset", 12 + 31 - DataTypeSizeBits(ofmConn->tensor->Type()));
    op = CreateSub(headroom_offset, headroom_plus_one, noScaleQuantZp0, noScaleQuantZp0, noScaleQuantZp0);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto right_shift = op->Output(TensorUsage::OFM)->tensor;
    right_shift->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 6 - Sub
    auto one = CreateConstTensor("one_const", 1);
    op = CreateSub(headroom_plus_one, one, noScaleQuantZp0, noScaleQuant, noScaleQuantZp0);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto headroom = op->Output(TensorUsage::OFM)->tensor;
    headroom->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 7 - SHL
    op = CreateShl(sum_of_exp, headroom, noScaleQuantZp0, noScaleQuantZp0, oneScaleQuant);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto half_denominator = op->Output(TensorUsage::OFM)->tensor;
    half_denominator->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 8 - Multiply
    auto neg_32_over_17 = CreateConstTensor("neg_32_over_17", -int32_t((32ULL << 29U) / 17U));
    op = CreateMul(half_denominator, neg_32_over_17, oneScaleQuant, oneScaleQuant, twoScaleQuant);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto rescaled = op->Output(TensorUsage::OFM)->tensor;
    rescaled->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 9 - Add
    auto const_48_over_17 = CreateConstTensor("const_48_over_17", int32_t((48ULL << 29U) / 17U));
    op = CreateAdd(rescaled, const_48_over_17, twoScaleQuant, noScaleQuant, oneScaleQuant);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto rescale_w_offset = op->Output(TensorUsage::OFM)->tensor;
    rescale_w_offset->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 10 - 24
    auto nr_x = std::move(rescale_w_offset);
    auto F2_one = CreateConstTensor("F2_one", 1 << 29);
    auto four = CreateConstTensor("four", 4);
    for ( int i = 0; i < 3; ++i )
    {
        // PASS 10, 15, 20 - MUL
        op = CreateMul(nr_x, half_denominator, oneScaleQuant, oneScaleQuant, twoScaleQuant);
        op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
        auto half_denominator_times_x = op->Output(TensorUsage::OFM)->tensor;
        half_denominator_times_x->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
        RecordOptimisation(operation, op);

        // PASS 11, 16, 21 - SUB
        op = CreateSub(F2_one, half_denominator_times_x, noScaleQuant, twoScaleQuant, oneScaleQuant);
        op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
        auto one_minus_half_denominator_times_x = op->Output(TensorUsage::OFM)->tensor;
        one_minus_half_denominator_times_x->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
        RecordOptimisation(operation, op);

        // PASS 12, 17, 22 - MUL
        op = CreateMul(nr_x, one_minus_half_denominator_times_x, oneScaleQuant, oneScaleQuant, twoScaleQuant);
        op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
        auto to_rescale = op->Output(TensorUsage::OFM)->tensor;
        to_rescale->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
        RecordOptimisation(operation, op);

        // PASS 13, 18, 23 - MUL
        op = CreateMul(to_rescale, four, twoScaleQuant, noScaleQuant, noScaleQuantZp0);
        op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
        auto to_add = op->Output(TensorUsage::OFM)->tensor;
        to_add->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
        RecordOptimisation(operation, op);

        // PASS 14, 19, 24 - ADD
        op = CreateAdd(nr_x, to_add, oneScaleQuant, noScaleQuantZp0, oneScaleQuant);
        op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
        nr_x = op->Output(TensorUsage::OFM)->tensor;
        nr_x->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
        RecordOptimisation(operation, op);
    }

    // PASS 25 - Multiply
    op = CreateMul(ifm_exp, nr_x, oneScaleQuant, oneScaleQuant, oneScaleQuant);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto scaled_exp = op->Output(TensorUsage::OFM)->tensor;
    scaled_exp->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 26 - ASR
    auto shrOp = std::make_shared<Operation>(OpType::Asr);
    op = shrOp.get();
    op->Attribute<asr_attr_t>()->round = true;
    op->ConnectInput(TensorUsage::IFM, scaled_exp).Set(oneScaleQuant);
    op->ConnectInput(TensorUsage::IFM1, right_shift).Set(noScaleQuantZp0);
    if ( ifmConn->tensor->Type() == DataType::Int8 && ofmConn->tensor->Type() == DataType::Int16 )
    {  // Special case for int16 output zero point correction
        std::string name(op->IFM(0)->Name() + "/" + OpTypeToString(op->Type()));
        auto shr = std::make_shared<Tensor>(name, op->IFM(0)->Type());
        shr->SetStorageShape(op->Input(TensorUsage::IFM)->shape);
        shr->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
        op->ConnectOutput(TensorUsage::OFM, shr).Set(oneScaleQuant);
        RecordOptimisation(operation, op);

        // PASS 27 - ADD
        int32_t zp = int32_t(ofmConn->quantization.zeroPoints[0]);
        assert(zp == std::numeric_limits<int16_t>::min());
        auto addOp = std::make_shared<Operation>(OpType::Add);
        op = addOp.get();
        op->ConnectInput(TensorUsage::IFM, shr).Set(oneScaleQuant);
        op->ConnectInput(TensorUsage::IFM1, CreateConstTensor("zeroPoint", zp)).Set(noScaleQuantZp0);
        op->ConnectOutput(TensorUsage::OFM, ofmConn->tensor).Set(oneScaleQuant).Set(ofmConn->shape);
    }
    else
    {
        op->ConnectOutput(TensorUsage::OFM, ofmConn->tensor).Set(ofmConn->quantization).Set(ofmConn->shape);
    }
    op->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
    RecordOptimisation(operation, op);

    return op;
}

Operation *Softmax::GetGraphInt16(Operation *const operation, TensorConnection *ifmConn, TensorConnection *ofmConn)
{
    constexpr int32_t range = std::numeric_limits<int16_t>::max() - std::numeric_limits<int16_t>::min();
    auto noScaleQuant = ifmConn->quantization;
    noScaleQuant.scales.clear();

    // Base name and index to use for new tensors
    const auto &tensorBaseName = ifmConn->tensor->Name();
    int tensorIdx = 0;

    // PASS 0 - Depthwise Maxpool
    auto op = CreateDepthwiseMaxpool(ifmConn->tensor, ifmConn->shape, ifmConn->quantization, noScaleQuant);
    op->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
    auto ifmMax = op->Output(TensorUsage::OFM)->tensor;
    ifmMax->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 1 - Sub
    op = CreateSub(ifmConn->tensor, ifmMax, ifmConn->quantization, noScaleQuant, ifmConn->quantization, DataType::Int32,
        &ifmConn->shape);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto sub1_ofm = op->Output(TensorUsage::OFM)->tensor;
    sub1_ofm->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 2 - Mul
    auto *softmax = operation->Attribute<softmax_attr_t>();
    double beta = double(softmax->beta);
    double exp_scale = _negExpRange / range;
    auto quant = ElementwiseMulScale<double>(ifmConn->quantization.scales[0].Dequantize(), beta, exp_scale);
    auto scale_quant = ifmConn->quantization;
    scale_quant.scales[0] = QuantizedScale(beta);
    auto mul2_quant = ofmConn->quantization;
    mul2_quant.scales[0] = QuantizedScale(exp_scale);
    auto scale = CreateConstTensor("mul2_scale", quant.scale);
    op = CreateMul(sub1_ofm, scale, ifmConn->quantization, scale_quant, mul2_quant);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto mul2_ofm = op->Output(TensorUsage::OFM)->tensor;
    mul2_ofm->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 3 - Add
    auto const_add = CreateConstTensor("add3_const", 32767);
    op = CreateAdd(mul2_ofm, const_add, mul2_quant, noScaleQuant, mul2_quant, DataType::Int16);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto ifm_add = op->Output(TensorUsage::OFM)->tensor;
    ifm_add->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 3.5 - LUT(exp)
    auto expLut = GenerateInterpolatingLUT16(
        [](double x) -> double { return std::exp(x); }, exp_scale, 2.0 / range, std::numeric_limits<int16_t>::max(), 0);
    auto expBuf = std::make_shared<Buffer>(std::move(expLut), 512);
    auto expTens = CreateConstTensor("exp_lut", DataType::Int32, expBuf);
    op = CreateLUT(ifm_add, expTens, mul2_quant, mul2_quant, DataType::Int16);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto ifm_exp = op->Output(TensorUsage::OFM)->tensor;
    ifm_exp->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 4 - Reduce sum
    op = CreateReduceSum(ifm_exp, mul2_quant, noScaleQuant);
    op->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
    auto sum_of_exp = op->Output(TensorUsage::OFM)->tensor;
    sum_of_exp->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 5 - CLZ
    op = CreateClz(sum_of_exp, noScaleQuant, noScaleQuant);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto headroom_plus_one = op->Output(TensorUsage::OFM)->tensor;
    headroom_plus_one->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 6 - Sub
    auto const_31 = CreateConstTensor("const_31", 31);
    op = CreateSub(const_31, headroom_plus_one, noScaleQuant, noScaleQuant, noScaleQuant);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto reciprocal_right_shift = op->Output(TensorUsage::OFM)->tensor;
    reciprocal_right_shift->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 7 - SHL
    auto one = CreateConstTensor("one_const", 1);
    op = CreateShl(one, reciprocal_right_shift, noScaleQuant, noScaleQuant, noScaleQuant);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto constant_one = op->Output(TensorUsage::OFM)->tensor;
    RecordOptimisation(operation, op);

    // PASS 8 - Sub
    op = CreateSub(sum_of_exp, constant_one, noScaleQuant, noScaleQuant, noScaleQuant);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto sum_of_exps_minus_one = op->Output(TensorUsage::OFM)->tensor;
    sum_of_exps_minus_one->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // # PASS 9 - SHL
    op = CreateShl(sum_of_exps_minus_one, headroom_plus_one, noScaleQuant, noScaleQuant, noScaleQuant);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto shifted_sum_minus_one = op->Output(TensorUsage::OFM)->tensor;
    shifted_sum_minus_one->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 10 - ASR
    auto shift = CreateConstTensor("shift_const", 15);
    op = CreateAsr(shifted_sum_minus_one, shift, noScaleQuant, noScaleQuant, noScaleQuant);
    op->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
    op->Attribute<asr_attr_t>()->round = true;
    auto shifted_sum_minus_one_16 = op->Output(TensorUsage::OFM)->tensor;
    shifted_sum_minus_one_16->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 11 - Sub
    auto sub11_const = CreateConstTensor("sub11_const", 32768);
    op = CreateSub(shifted_sum_minus_one_16, sub11_const, noScaleQuant, noScaleQuant, noScaleQuant, DataType::Int16);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto reciprocal_scale = op->Output(TensorUsage::OFM)->tensor;
    reciprocal_scale->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 11.5 - LUT(one over one plus x)
    auto oneOverOnePlusXLut = GenerateInterpolatingLUT16([](double x) -> double { return 1.0 / (1.0 + x); },
        1.0 / range, 2.0 / range, std::numeric_limits<int16_t>::min(), 0);
    auto oneOverOnePlusXBuf = std::make_shared<Buffer>(std::move(oneOverOnePlusXLut), 512);
    auto oneOverOnePlusXTens = CreateConstTensor("one_over_one_plus_x_lut", DataType::Int32, oneOverOnePlusXBuf);
    op = CreateLUT(reciprocal_scale, oneOverOnePlusXTens, noScaleQuant, noScaleQuant, DataType::Int16);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    reciprocal_scale = op->Output(TensorUsage::OFM)->tensor;
    RecordOptimisation(operation, op);

    // # PASS 12 - Multiply
    op = CreateMul(ifm_exp, reciprocal_scale, noScaleQuant, noScaleQuant, noScaleQuant, DataType::Int32);
    op->Output(TensorUsage::OFM)->Set(RoundMode::DBL);
    auto mul_ofm = op->Output(TensorUsage::OFM)->tensor;
    mul_ofm->SetName(tensorBaseName + "_softmax_" + std::to_string(tensorIdx++));
    RecordOptimisation(operation, op);

    // PASS 13 - ASR
    auto shrOp = std::make_shared<Operation>(OpType::Asr);
    op = shrOp.get();
    op->Attribute<asr_attr_t>()->round = true;
    op->ConnectInput(TensorUsage::IFM, mul_ofm).Set(noScaleQuant);
    op->ConnectInput(TensorUsage::IFM1, reciprocal_right_shift).Set(noScaleQuant);
    op->ConnectOutput(TensorUsage::OFM, ofmConn->tensor).Set(ofmConn->quantization).Set(ofmConn->shape);
    op->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
    RecordOptimisation(operation, op);

    return op;
}

std::vector<int32_t> Softmax::GenerateExpTable(double beta, double inputScale)
{
    const int kTableSize = 256;
    const int kIntegerBits = 5;
    const int kSignedBits = 31;
    std::vector<int32_t> expTable(kTableSize);
    using FixedPoint = gemmlowp::FixedPoint<int32_t, kIntegerBits>;

    const double realBeta = std::min(beta * inputScale * (1 << (kSignedBits - kIntegerBits)), (1ll << kSignedBits) - 1.0);
    const auto quant = QuantizedScale(realBeta);
    const int leftShift = 31 - quant.shift;
    const int diffMin = -int(std::floor(1.0 * ((1 << kIntegerBits) - 1) * (1 << (kSignedBits - kIntegerBits)) / (1U << leftShift)));

    for ( int x = 0; x < kTableSize; ++x )
    {
        int inputDiff = x - 255;
        if ( inputDiff >= diffMin )
        {
            const int32_t inputDiffRescaled = gemmlowp::SaturatingRoundingDoublingHighMul(
                ClampToType<int32_t>(inputDiff * (1LL << leftShift)), quant.scale);
            expTable[x] = gemmlowp::exp_on_negative_values(FixedPoint::FromRaw(inputDiffRescaled)).raw();
        }
        else
        {
            expTable[x] = 0;
        }
    }

    return expTable;
}

}  // namespace regor
