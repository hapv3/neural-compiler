//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph_optimiser.hpp"

namespace regor
{

enum class NeuralAIGraphOptimiserStage
{
    Prepare,
    Finalize,
};

class NeuralAIGraphOptimiser final : public GraphOptimiser
{
public:
    NeuralAIGraphOptimiser(IArchitectureConstraints *constraints, const GraphOptimiserOptions &options,
        OptimiserDatabase *db, NeuralAIGraphOptimiserStage stage = NeuralAIGraphOptimiserStage::Finalize) :
            GraphOptimiser(constraints, options, db), _stage(stage)
    {
    }

    void OptimiseGraph(Graph *graph) override;

private:
    NeuralAIGraphOptimiserStage _stage;

    void CanonicalizeAsymmetricConv(Graph *graph, Operation *operation);
    void CanonicalizeConvAdd(Graph *graph, Operation *operation);
    void InsertInputConversion(Graph *graph, Operation *operation, TensorUsage usage);
    void InsertOutputConversion(Graph *graph, Operation *operation);
    void MaterializeStructuralCspConcatInputs(Operation *operation);
    void FuseStructuralHeadPack(Graph *graph, Operation *operation);
    void DistributeStructuralHeadMerge(Graph *graph, Operation *operation);
};

}  // namespace regor
