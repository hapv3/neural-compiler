//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph_optimiser.hpp"

namespace regor
{

class NeuralAIGraphOptimiser final : public GraphOptimiser
{
public:
    NeuralAIGraphOptimiser(IArchitectureConstraints *constraints, const GraphOptimiserOptions &options,
        OptimiserDatabase *db) : GraphOptimiser(constraints, options, db)
    {
    }

    void OptimiseGraph(Graph *graph) override;

private:
    void InsertInputConversion(Graph *graph, Operation *operation, TensorUsage usage);
    void InsertOutputConversion(Graph *graph, Operation *operation);
};

}  // namespace regor
