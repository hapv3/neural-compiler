//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_graph_optimiser.hpp"

#include "common/buffer_view.hpp"
#include "kernel.hpp"
#include "operation_util.hpp"

#include <algorithm>
#include <limits>

namespace regor
{

namespace
{

int64_t ScalarZeroPoint(const Quantization &quantization)
{
    return quantization.zeroPoints.empty() ? 0 : quantization.zeroPoints[0];
}

bool HasScalarQuantization(const Quantization &quantization)
{
    return quantization.scales.size() == 1 && quantization.zeroPoints.size() <= 1;
}

bool IsRawAddScale(const TensorConnection *lhs, const TensorConnection *rhs, const TensorConnection *ofm)
{
    if ( lhs == nullptr || rhs == nullptr || ofm == nullptr ||
         !HasScalarQuantization(lhs->quantization) || !HasScalarQuantization(rhs->quantization) ||
         !HasScalarQuantization(ofm->quantization) ) return false;
    return lhs->quantization.scales[0] == QuantizedScale(32768.0) &&
        rhs->quantization.scales[0] == QuantizedScale(32768.0) &&
        ofm->quantization.scales[0] == QuantizedScale(1.0 / 32768.0) &&
        (ofm->quantization.quantMin.empty() || ofm->quantization.quantMin[0] <= -128) &&
        (ofm->quantization.quantMax.empty() || ofm->quantization.quantMax[0] >= 127);
}

bool ShiftClamp(Quantization &quantization, int64_t delta)
{
    if ( quantization.quantMin.size() != 1 || quantization.quantMax.size() != 1 ) return false;
    const int64_t shiftedMin = quantization.quantMin[0] + delta;
    const int64_t shiftedMax = quantization.quantMax[0] + delta;
    if ( shiftedMin < -128 || shiftedMax > 127 || shiftedMin > shiftedMax ) return false;
    quantization.quantMin[0] = shiftedMin;
    quantization.quantMax[0] = shiftedMax;
    return true;
}

}  // namespace

void NeuralAIGraphOptimiser::CanonicalizeAsymmetricConv(Graph *graph, Operation *operation)
{
    const OpType opType = operation->Type();
    if ( opType != OpType::Conv2D && opType != OpType::DepthwiseConv2D ) return;

    TensorConnection *ifm = operation->Input(TensorUsage::IFM0);
    TensorConnection *weights = operation->Input(TensorUsage::Weights);
    TensorConnection *bias = operation->Input(TensorUsage::Scales);
    TensorConnection *ofm = operation->Output(TensorUsage::OFM);
    const Kernel *kernel = operation->Kernel();
    if ( ifm == nullptr || weights == nullptr || bias == nullptr || ofm == nullptr || kernel == nullptr ||
         ifm->tensor->Type() != DataType::Int8 || weights->tensor->Type() != DataType::Int8 ||
         bias->tensor->Type() != DataType::Int32 || ifm->quantization.zeroPoints.size() != 1 ||
         !weights->tensor->IsConstant() || !bias->tensor->IsConstant() )
        return;

    const int64_t ifmZeroPoint = ifm->quantization.zeroPoints[0];
    if ( ifmZeroPoint == 0 || std::any_of(weights->quantization.zeroPoints.begin(),
                                  weights->quantization.zeroPoints.end(), [](int64_t value) { return value != 0; }) )
        return;

    const Shape weightShape = weights->shape.IsEmpty() ? weights->tensor->StorageShape() : weights->shape;
    const Shape ofmShape = ofm->shape.IsEmpty() ? ofm->tensor->StorageShape() : ofm->shape;
    if ( weightShape.Size() != 4 || ofmShape.Depth() <= 0 || bias->tensor->View().Elements() != ofmShape.Depth() ) return;

    const int outputChannels = ofmShape.Depth();
    const auto weightValues = weights->tensor->View().Values<int>(DataType::Int8);
    const auto biasValues = bias->tensor->View().Values<int32_t>();
    std::vector<int32_t> adjustedBias;
    adjustedBias.reserve(outputChannels);
    for ( int oc = 0; oc < outputChannels; ++oc )
    {
        int64_t weightSum = 0;
        if ( opType == OpType::Conv2D )
        {
            if ( weightShape.Batch() != outputChannels ) return;
            for ( int ky = 0; ky < weightShape.Height(); ++ky )
                for ( int kx = 0; kx < weightShape.Width(); ++kx )
                    for ( int ic = 0; ic < weightShape.Depth(); ++ic )
                        weightSum += weightValues[Shape(oc, ky, kx, ic)];
        }
        else if ( weightShape.Batch() == 1 && weightShape.Depth() == outputChannels )
        {
            for ( int ky = 0; ky < weightShape.Height(); ++ky )
                for ( int kx = 0; kx < weightShape.Width(); ++kx )
                    weightSum += weightValues[Shape(0, ky, kx, oc)];
        }
        else if ( weightShape.Batch() == outputChannels && weightShape.Depth() == 1 )
        {
            for ( int ky = 0; ky < weightShape.Height(); ++ky )
                for ( int kx = 0; kx < weightShape.Width(); ++kx )
                    weightSum += weightValues[Shape(oc, ky, kx, 0)];
        }
        else
        {
            return;
        }

        const int64_t corrected = int64_t(biasValues[oc]) - ifmZeroPoint * weightSum;
        if ( corrected < std::numeric_limits<int32_t>::min() || corrected > std::numeric_limits<int32_t>::max() ) return;
        adjustedBias.push_back(int32_t(corrected));
    }

    Operation *explicitPad = nullptr;
    if ( kernel->Padding().IsZero() && ifm->tensor->Writers().size() == 1 )
    {
        Operation *producer = ifm->tensor->Writers().front().get();
        if ( producer->Type() == OpType::Pad && producer->Output(TensorUsage::OFM)->tensor == ifm->tensor &&
             producer->Attribute<pad_attr_t>()->pad_const == 0 )
            explicitPad = producer;
    }

    Quantization zeroPointQuantization = ifm->quantization;
    zeroPointQuantization.zeroPoints = {0};
    if ( !kernel->Padding().IsZero() )
    {
        const Margin padding = kernel->Padding();
        const std::vector<int32_t> paddingValues{
            0, 0, padding.Top(), padding.Bottom(), padding.Left(), padding.Right(), 0, 0};
        const Shape paddingShape(int(paddingValues.size()));
        auto paddingTensor = CreateConstTensor("neural_ai_asymmetric_padding", DataType::Int32,
            std::make_shared<Buffer>(std::vector<int32_t>(paddingValues)), &paddingShape);
        const Shape ifmShape = ifm->shape.IsEmpty() ? ifm->tensor->StorageShape() : ifm->shape;
        const Shape paddedShape = ifmShape.WithHW(ifmShape.WH() + padding.TL() + padding.BR());
        auto paddedTensor = std::make_shared<Tensor>(
            ifm->tensor->Name() + "/zero_point_padding", ifm->tensor->Type(), paddedShape);
        auto pad = std::make_shared<Operation>(OpType::Pad);
        pad->CopyInput(TensorUsage::IFM0, *ifm);
        pad->ConnectInput(TensorUsage::Params, paddingTensor);
        pad->ConnectOutput(TensorUsage::OFM, paddedTensor).Set(paddedShape).Set(zeroPointQuantization);
        pad->Attribute<pad_attr_t>()->pad_const = ifmZeroPoint;
        operation->ConnectInput(TensorUsage::IFM0, paddedTensor).Set(paddedShape).Set(zeroPointQuantization);
        operation->SetKernel(std::make_unique<Kernel>(kernel->WithPadding({})));
        RecordOptimisation(*operation, pad.get());
    }
    else if ( graph->IsInput(ifm->tensor.get()) )
    {
        const TensorConnection original = *ifm;
        auto nativeTensor = std::shared_ptr<Tensor>(original.tensor->Clone().release());
        nativeTensor->SetName(original.tensor->Name() + "/zero_point_staging");
        auto copy = std::make_shared<Operation>(OpType::MemoryCopy);
        copy->CopyInput(TensorUsage::IFM, original);
        copy->ConnectOutput(TensorUsage::OFM, nativeTensor)
            .Set(original.shape)
            .Set(original.slice)
            .Set(zeroPointQuantization)
            .Set(original.rounding);
        operation->ConnectInput(TensorUsage::IFM0, nativeTensor)
            .Set(original.shape)
            .Set(original.slice)
            .Set(zeroPointQuantization)
            .Set(original.reverse)
            .Set(original.rounding);
        RecordOptimisation(*operation, copy.get());
    }
    else
    {
        ifm->Set(zeroPointQuantization);
        if ( explicitPad != nullptr )
        {
            explicitPad->Output(TensorUsage::OFM)->Set(zeroPointQuantization);
            explicitPad->Attribute<pad_attr_t>()->pad_const = ifmZeroPoint;
        }
    }

    auto adjustedBiasTensor = std::shared_ptr<Tensor>(bias->tensor->Clone().release());
    adjustedBiasTensor->SetName(bias->tensor->Name() + "/zero_point_compensated");
    adjustedBiasTensor->SetBuffer(std::make_shared<Buffer>(std::move(adjustedBias)));
    const TensorConnection originalBias = *bias;
    operation->ConnectInput(TensorUsage::Scales, adjustedBiasTensor)
        .Set(originalBias.shape)
        .Set(originalBias.slice)
        .Set(originalBias.quantization)
        .Set(originalBias.reverse)
        .Set(originalBias.rounding);
}

void NeuralAIGraphOptimiser::CanonicalizeConvAdd(Graph *graph, Operation *operation)
{
    if ( operation->Type() != OpType::Add ) return;
    TensorConnection *lhs = operation->Input(TensorUsage::IFM0);
    TensorConnection *rhs = operation->Input(TensorUsage::IFM1);
    TensorConnection *ofm = operation->Output(TensorUsage::OFM);
    if ( !IsRawAddScale(lhs, rhs, ofm) ) return;

    const int64_t delta = ScalarZeroPoint(ofm->quantization) -
        ScalarZeroPoint(lhs->quantization) - ScalarZeroPoint(rhs->quantization);
    if ( delta == 0 ) return;

    const auto isPrivateConvOutput = [graph](const TensorConnection *connection)
    {
        if ( connection == nullptr || !connection->tensor || graph->IsOutput(connection->tensor.get()) ||
             connection->tensor->Writers().size() != 1 || connection->tensor->Readers().size() != 1 ) return false;
        const Operation *writer = connection->tensor->Writers().front().get();
        return writer != nullptr && writer->Type() == OpType::Conv2D;
    };
    const bool lhsCandidate = isPrivateConvOutput(lhs);
    const bool rhsCandidate = isPrivateConvOutput(rhs);
    if ( lhsCandidate == rhsCandidate ) return;

    TensorConnection *candidate = lhsCandidate ? lhs : rhs;
    Operation *producer = candidate->tensor->Writers().front().get();
    TensorConnection *producerOfm = producer->Output(TensorUsage::OFM);
    if ( producerOfm == nullptr || producerOfm->tensor != candidate->tensor ||
         !HasScalarQuantization(producerOfm->quantization) ||
         ScalarZeroPoint(producerOfm->quantization) != ScalarZeroPoint(candidate->quantization) ||
         producerOfm->quantization.quantMin != candidate->quantization.quantMin ||
         producerOfm->quantization.quantMax != candidate->quantization.quantMax ) return;
    const int64_t shiftedZeroPoint = ScalarZeroPoint(producerOfm->quantization) + delta;
    if ( shiftedZeroPoint < -128 || shiftedZeroPoint > 127 ) return;

    Quantization shiftedProducer = producerOfm->quantization;
    Quantization shiftedConsumer = candidate->quantization;
    if ( !ShiftClamp(shiftedProducer, delta) || !ShiftClamp(shiftedConsumer, delta) ) return;
    shiftedProducer.zeroPoints = {shiftedZeroPoint};
    shiftedConsumer.zeroPoints = {ScalarZeroPoint(candidate->quantization) + delta};
    producerOfm->Set(shiftedProducer);
    candidate->Set(shiftedConsumer);
}

void NeuralAIGraphOptimiser::InsertInputConversion(Graph *graph, Operation *operation, TensorUsage usage)
{
    const TensorConnection original = *operation->Input(usage);
    if ( !graph->IsInput(original.tensor.get()) ) return;

    auto nativeTensor = std::shared_ptr<Tensor>(original.tensor->Clone().release());
    nativeTensor->SetName(original.tensor->Name() + "/row32");
    auto copy = std::make_shared<Operation>(OpType::MemoryCopy);
    copy->CopyInput(TensorUsage::IFM, original);
    copy->ConnectOutput(TensorUsage::OFM, nativeTensor)
        .Set(original.shape)
        .Set(original.slice)
        .Set(original.quantization)
        .Set(original.rounding);
    operation->ConnectInput(usage, nativeTensor)
        .Set(original.shape)
        .Set(original.slice)
        .Set(original.quantization)
        .Set(original.reverse)
        .Set(original.rounding);
    RecordOptimisation(*operation, copy.get());
}

void NeuralAIGraphOptimiser::InsertOutputConversion(Graph *graph, Operation *operation)
{
    const TensorConnection original = *operation->Output(TensorUsage::OFM);
    if ( !graph->IsOutput(original.tensor.get()) ) return;

    auto nativeTensor = std::shared_ptr<Tensor>(original.tensor->Clone().release());
    nativeTensor->SetName(original.tensor->Name() + "/row32");
    operation->ConnectOutput(TensorUsage::OFM, nativeTensor)
        .Set(original.shape)
        .Set(original.slice)
        .Set(original.quantization)
        .Set(original.reverse)
        .Set(original.rounding);
    auto copy = std::make_shared<Operation>(OpType::MemoryCopy);
    copy->ConnectInput(TensorUsage::IFM, nativeTensor)
        .Set(original.shape)
        .Set(original.slice)
        .Set(original.quantization)
        .Set(original.reverse)
        .Set(original.rounding);
    copy->CopyOutput(TensorUsage::OFM, original);
    RecordOptimisation(*operation, copy.get());
}

void NeuralAIGraphOptimiser::OptimiseGraph(Graph *graph)
{
    std::vector<std::shared_ptr<Operation>> operations;
    graph->GetAllOperations(operations);
    if ( _stage == NeuralAIGraphOptimiserStage::Prepare )
    {
        for ( const auto &operation : operations ) CanonicalizeAsymmetricConv(graph, operation.get());
        return;
    }
    for ( const auto &operation : operations ) CanonicalizeConvAdd(graph, operation.get());
    for ( const auto &operation : operations )
    {
        if ( operation->Type() != OpType::FullyConnected && operation->Type() != OpType::MatMul &&
             operation->Type() != OpType::Conv2D && operation->Type() != OpType::DepthwiseConv2D &&
             operation->Type() != OpType::LUT && operation->Type() != OpType::Add &&
             operation->Type() != OpType::AvgPool && operation->Type() != OpType::Resize &&
             operation->Type() != OpType::MaxPool && operation->Type() != OpType::Concat &&
             operation->Type() != OpType::Transpose ) continue;
        InsertInputConversion(graph, operation.get(), TensorUsage::IFM0);
        if ( operation->Type() == OpType::Add || operation->Type() == OpType::Concat )
            InsertInputConversion(graph, operation.get(), TensorUsage::IFM1);
        InsertOutputConversion(graph, operation.get());
    }
}

}  // namespace regor
