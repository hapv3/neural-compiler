//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_memory_placement.hpp"

#include "scheduler_operation.hpp"

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

bool IsAlignedBinary(const SchedulerOperation *operation)
{
    return IsC32FeatureMap(operation->TryIFM(0)) &&
           IsC32FeatureMap(operation->TryIFM(1)) &&
           IsC32FeatureMap(operation->outputs.try_ref(TensorUsage::OFM));
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

bool SupportsL2Output(const SchedulerOperation *operation, const SchedulerTensor *tensor)
{
    const SchedulerConnection *ofm = operation->outputs.try_ref(TensorUsage::OFM);
    if ( ofm == nullptr || ofm->tensor.get() != tensor ) return false;
    switch ( operation->Type() )
    {
    case OpType::Add:
    case OpType::Sub: return IsAlignedBinary(operation);
    case OpType::Conv2D: return IsReviewedConv(operation);
    case OpType::DepthwiseConv2D: return IsReviewedDepthwise(operation);
    default: return false;
    }
}

bool SupportsL2Input(const SchedulerOperation *operation, const SchedulerTensor *tensor)
{
    const SchedulerConnection *ifm0 = operation->TryIFM(0);
    const SchedulerConnection *ifm1 = operation->TryIFM(1);
    if ( (ifm0 == nullptr || ifm0->tensor.get() != tensor) &&
         (ifm1 == nullptr || ifm1->tensor.get() != tensor) )
        return false;
    switch ( operation->Type() )
    {
    case OpType::Add:
    case OpType::Sub: return IsAlignedBinary(operation);
    case OpType::Conv2D: return IsReviewedConv(operation);
    case OpType::DepthwiseConv2D: return IsReviewedDepthwise(operation);
    case OpType::Concat: return IsReviewedConcatInput(operation);
    default: return false;
    }
}

}  // namespace

bool IsNeuralAIL2SpillCandidate(const SchedulerTensor *tensor)
{
    if ( tensor == nullptr || tensor->IsConstant() ||
         tensor->format != TensorFormat::C32Blocked ||
         tensor->isGraphInput || tensor->isGraphOutput || tensor->isPersistent ||
         tensor->hasCPUReaders || tensor->hasCPUWriters ||
         tensor->producers.empty() || tensor->consumers.empty() )
        return false;

    return std::all_of(tensor->producers.begin(), tensor->producers.end(),
               [tensor](const SchedulerOperation *operation) { return SupportsL2Output(operation, tensor); }) &&
           std::all_of(tensor->consumers.begin(), tensor->consumers.end(),
               [tensor](const SchedulerOperation *operation) { return SupportsL2Input(operation, tensor); });
}

}  // namespace regor
