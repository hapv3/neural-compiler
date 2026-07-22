//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "neural_ai_writer.hpp"
#include "scheduler.hpp"

namespace regor
{

class NeuralAICommandGenerator
{
public:
    bool Generate(const Graph *graph, const std::vector<std::unique_ptr<SchedulerOperation>> &operations,
        const Schedule *schedule, CompiledNeuralAIArtifact &artifact, std::string &error);
};

}  // namespace regor
