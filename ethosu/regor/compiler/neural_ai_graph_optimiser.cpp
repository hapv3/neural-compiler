//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_graph_optimiser.hpp"

#include "kernel.hpp"

namespace regor
{

namespace
{

bool IsDirectRgbStem(const Operation *operation)
{
    if ( operation == nullptr || operation->Type() != OpType::Conv2D || operation->Kernel() == nullptr ) return false;
    const TensorConnection *ifm = operation->Input(TensorUsage::IFM0);
    const TensorConnection *ofm = operation->Output(TensorUsage::OFM);
    if ( ifm == nullptr || ofm == nullptr || !ifm->tensor || !ofm->tensor ) return false;
    const Shape ifmShape = ifm->shape.IsEmpty() ? ifm->tensor->StorageShape() : ifm->shape;
    const Shape ofmShape = ofm->shape.IsEmpty() ? ofm->tensor->StorageShape() : ofm->shape;
    const Kernel *kernel = operation->Kernel();
    return ifmShape.Depth() == 3 && ofmShape.Depth() == 32 &&
        kernel->Size() == Point2i(3, 3) && kernel->Stride() == Point2i(2, 2) &&
        kernel->Dilation() == Point2i(1, 1) && kernel->Padding().Top() == 1 &&
        kernel->Padding().Left() == 1 && kernel->Padding().Bottom() == 1 &&
        kernel->Padding().Right() == 1 && ifm->tensor->Writers().empty();
}

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
    for ( const auto &operation : operations ) CanonicalizeConvAdd(graph, operation.get());
    for ( const auto &operation : operations )
    {
        if ( operation->Type() != OpType::FullyConnected && operation->Type() != OpType::MatMul &&
             operation->Type() != OpType::Conv2D && operation->Type() != OpType::DepthwiseConv2D &&
             operation->Type() != OpType::LUT && operation->Type() != OpType::Add &&
             operation->Type() != OpType::AvgPool && operation->Type() != OpType::Resize &&
             operation->Type() != OpType::MaxPool && operation->Type() != OpType::Concat ) continue;
        if ( !IsDirectRgbStem(operation.get()) ) InsertInputConversion(graph, operation.get(), TensorUsage::IFM0);
        if ( operation->Type() == OpType::Add || operation->Type() == OpType::Concat )
            InsertInputConversion(graph, operation.get(), TensorUsage::IFM1);
        InsertOutputConversion(graph, operation.get());
    }
}

}  // namespace regor
