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

}  // namespace

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
    for ( const auto &operation : operations )
    {
        if ( operation->Type() != OpType::FullyConnected && operation->Type() != OpType::MatMul &&
             operation->Type() != OpType::Conv2D && operation->Type() != OpType::DepthwiseConv2D &&
             operation->Type() != OpType::LUT && operation->Type() != OpType::Add &&
             operation->Type() != OpType::AvgPool && operation->Type() != OpType::Resize ) continue;
        if ( !IsDirectRgbStem(operation.get()) ) InsertInputConversion(graph, operation.get(), TensorUsage::IFM0);
        if ( operation->Type() == OpType::Add )
            InsertInputConversion(graph, operation.get(), TensorUsage::IFM1);
        InsertOutputConversion(graph, operation.get());
    }
}

}  // namespace regor
