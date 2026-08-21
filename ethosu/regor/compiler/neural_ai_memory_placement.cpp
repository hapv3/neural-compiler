//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_memory_placement.hpp"

#include "scheduler_operation.hpp"

#include <unordered_map>
#include <unordered_set>

namespace regor
{

namespace
{

bool IsC32FeatureMap(const SchedulerConnection *connection)
{
    return connection != nullptr && connection->tensor != nullptr &&
           connection->tensor->format == TensorFormat::C32Blocked &&
           connection->SliceShape().Size() == 4 && connection->SliceShape().Batch() == 1 &&
           connection->SliceShape().Depth() > 0 && connection->SliceShape().Depth() % 32 == 0;
}

bool IsCompactC16FeatureMap(const SchedulerConnection *connection)
{
    return connection != nullptr && connection->tensor != nullptr &&
           connection->tensor->format == TensorFormat::CompactNHWC &&
           connection->Type() == DataType::Int8 && connection->shape.Size() == 4 &&
           connection->shape.Batch() == 1 && connection->shape.Depth() == 16;
}

bool IsAlignedBinary(const SchedulerOperation *operation)
{
    return IsC32FeatureMap(operation->TryIFM(0)) &&
           IsC32FeatureMap(operation->TryIFM(1)) &&
           IsC32FeatureMap(operation->outputs.try_ref(TensorUsage::OFM));
}

bool IsReviewedCompactBinary(const SchedulerOperation *operation)
{
    const SchedulerConnection *lhs = operation->TryIFM(0);
    const SchedulerConnection *rhs = operation->TryIFM(1);
    const SchedulerConnection *ofm = operation->outputs.try_ref(TensorUsage::OFM);
    return IsCompactC16FeatureMap(lhs) && IsCompactC16FeatureMap(rhs) &&
           IsCompactC16FeatureMap(ofm) && lhs->shape == rhs->shape && lhs->shape == ofm->shape;
}

bool IsReviewedConv(const SchedulerOperation *operation)
{
    const Point2i size = operation->Kernel()->Size();
    const bool reviewedKernel = size == Point2i(1, 1) || size == Point2i(3, 3);
    return reviewedKernel && IsC32FeatureMap(operation->TryIFM(0)) &&
           IsC32FeatureMap(operation->outputs.try_ref(TensorUsage::OFM));
}

bool IsReviewedDepthwise(const SchedulerOperation *operation)
{
    return operation->Kernel()->Size() == Point2i(3, 3) &&
           IsC32FeatureMap(operation->TryIFM(0)) &&
           IsC32FeatureMap(operation->outputs.try_ref(TensorUsage::OFM));
}

bool IsReviewedConcatInput(const SchedulerOperation *operation)
{
    return operation->TryIFM(2) == nullptr && IsC32FeatureMap(operation->TryIFM(0)) &&
           IsC32FeatureMap(operation->TryIFM(1)) &&
           IsC32FeatureMap(operation->outputs.try_ref(TensorUsage::OFM));
}

bool IsReviewedCompactCspConcatInput(const SchedulerOperation *operation)
{
    const SchedulerConnection *ifm0 = operation->TryIFM(0);
    const SchedulerConnection *ifm1 = operation->TryIFM(1);
    const SchedulerConnection *ifm2 = operation->TryIFM(2);
    const SchedulerConnection *ofm = operation->outputs.try_ref(TensorUsage::OFM);
    return IsCompactC16FeatureMap(ifm0) && IsCompactC16FeatureMap(ifm1) &&
           IsCompactC16FeatureMap(ifm2) && ofm != nullptr &&
           ofm->tensor->format == TensorFormat::C32Blocked && ofm->Type() == DataType::Int8 &&
           ofm->shape.Size() == 4 && ofm->shape.Batch() == 1 && ofm->shape.Depth() == 48 &&
           ifm0->shape.WithDepth(48) == ofm->shape &&
           ifm1->shape == ifm0->shape && ifm2->shape == ifm0->shape;
}

bool IsReviewedCompactCopyOutput(const SchedulerOperation *operation)
{
    const SchedulerConnection *ifm = operation->TryIFM(0);
    const SchedulerConnection *ofm = operation->outputs.try_ref(TensorUsage::OFM);
    return ifm != nullptr && ifm->tensor->format == TensorFormat::C32Blocked &&
           IsCompactC16FeatureMap(ofm) && ifm->Type() == DataType::Int8 &&
           ifm->SliceShape() == ofm->SliceShape();
}

bool IsReviewedCompactLinebufferInput(
    const SchedulerOperation *operation, const SchedulerTensor *tensor)
{
    const SchedulerConnection *ifm = operation->TryIFM(0);
    const SchedulerConnection *ofm = operation->outputs.try_ref(TensorUsage::OFM);
    return operation->Type() == OpType::Conv2D && operation->Kernel()->Size() == Point2i(3, 3) &&
           ifm != nullptr && ifm->tensor.get() == tensor && IsCompactC16FeatureMap(ifm) &&
           ofm != nullptr && ofm->tensor->format == TensorFormat::C32Blocked &&
           ofm->Type() == DataType::Int8 && ofm->shape.Size() == 4 && ofm->shape.Batch() == 1 &&
           ofm->shape.Height() > 0 && ofm->shape.Width() > 0 && ofm->shape.Depth() == 16;
}

bool SupportsL2Output(const SchedulerOperation *operation, const SchedulerTensor *tensor)
{
    const SchedulerConnection *ofm = operation->outputs.try_ref(TensorUsage::OFM);
    if ( ofm == nullptr || ofm->tensor.get() != tensor ) return false;
    switch ( operation->Type() )
    {
    case OpType::Add:
    case OpType::Sub:
        return IsAlignedBinary(operation) || IsReviewedCompactBinary(operation);
    case OpType::Conv2D: return IsReviewedConv(operation);
    case OpType::DepthwiseConv2D: return IsReviewedDepthwise(operation);
    case OpType::MemoryCopy: return IsReviewedCompactCopyOutput(operation);
    default: return false;
    }
}

bool SupportsL2Input(const SchedulerOperation *operation, const SchedulerTensor *tensor)
{
    bool consumesTensor = false;
    for ( const auto &[usage, connection] : operation->inputs.pairs() )
    {
        UNUSED(usage);
        consumesTensor = consumesTensor || connection.tensor.get() == tensor;
    }
    if ( !consumesTensor )
        return false;
    switch ( operation->Type() )
    {
    case OpType::Add:
    case OpType::Sub:
        return IsAlignedBinary(operation) || IsReviewedCompactBinary(operation);
    case OpType::Conv2D:
        return IsReviewedConv(operation) || IsReviewedCompactLinebufferInput(operation, tensor);
    case OpType::DepthwiseConv2D: return IsReviewedDepthwise(operation);
    case OpType::Concat:
        return IsReviewedConcatInput(operation) || IsReviewedCompactCspConcatInput(operation);
    default: return false;
    }
}

}  // namespace

bool IsNeuralAIL2SpillCandidate(const SchedulerTensor *tensor)
{
    if ( tensor == nullptr || tensor->IsConstant() ||
         (tensor->format != TensorFormat::C32Blocked &&
             tensor->format != TensorFormat::CompactNHWC) ||
         tensor->isGraphInput || tensor->isGraphOutput || tensor->isPersistent ||
         tensor->hasCPUReaders || tensor->hasCPUWriters ||
         tensor->producers.empty() || tensor->consumers.empty() )
        return false;

    return std::all_of(tensor->producers.begin(), tensor->producers.end(),
               [tensor](const SchedulerOperation *operation) { return SupportsL2Output(operation, tensor); }) &&
           std::all_of(tensor->consumers.begin(), tensor->consumers.end(),
               [tensor](const SchedulerOperation *operation) { return SupportsL2Input(operation, tensor); });
}

NeuralAIMemoryPlacementStats ApplyNeuralAIMemoryPlacement(
    const std::vector<std::unique_ptr<SchedulerOperation>> &operations,
    const MemArea &l2Memory, const MemArea &tcdmMemory)
{
    std::unordered_map<UniqueId, std::vector<SchedulerTensor *>> equivalenceGroups;
    std::unordered_set<SchedulerTensor *> collected;
    const auto collectTensor = [&](const SchedulerConnection &connection)
    {
        SchedulerTensor *tensor = connection.tensor.get();
        if ( tensor == nullptr || tensor->IsConstant() || !collected.insert(tensor).second ) return;
        equivalenceGroups[tensor->equivalenceId].push_back(tensor);
    };
    const auto collectOperation = [&](const SchedulerOperation *operation)
    {
        for ( const auto &[usage, connection] : operation->inputs.pairs() )
        {
            UNUSED(usage);
            collectTensor(connection);
        }
        for ( const auto &[usage, connection] : operation->outputs.pairs() )
        {
            UNUSED(usage);
            collectTensor(connection);
        }
    };
    for ( const auto &operation : operations )
    {
        collectOperation(operation.get());
        for ( const auto &subOperation : operation->SubOps() ) collectOperation(subOperation.get());
    }

    NeuralAIMemoryPlacementStats stats;
    for ( auto &[equivalenceId, tensors] : equivalenceGroups )
    {
        UNUSED(equivalenceId);
        const bool spill = std::all_of(tensors.begin(), tensors.end(), IsNeuralAIL2SpillCandidate);
        const MemArea &target = spill ? l2Memory : tcdmMemory;
        for ( SchedulerTensor *tensor : tensors )
        {
            tensor->memArea = target;
            if ( spill ) ++stats.l2Tensors;
            else ++stats.tcdmTensors;
        }
    }
    return stats;
}

}  // namespace regor
