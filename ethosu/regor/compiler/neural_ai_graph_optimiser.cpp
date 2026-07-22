//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_graph_optimiser.hpp"

namespace regor
{

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
        if ( operation->Type() != OpType::FullyConnected && operation->Type() != OpType::MatMul ) continue;
        InsertInputConversion(graph, operation.get(), TensorUsage::IFM0);
        InsertOutputConversion(graph, operation.get());
    }
}

}  // namespace regor
