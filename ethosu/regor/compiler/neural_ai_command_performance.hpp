//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compiler/network_performance.hpp"
#include "compiler/neural_ai_writer.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace regor
{

class Architecture;
class Database;

struct NeuralAICommandPerformance
{
    int index = 0;
    uint16_t type = 0;
    uint32_t layerId = 0;
    uint32_t tileId = 0;
    std::string name;
    int64_t computeCycles = 0;
    int64_t memoryCycles = 0;
    int64_t totalCycles = 0;
    int64_t modelReadBytes = 0;
    int64_t l2ReadBytes = 0;
    int64_t l2WriteBytes = 0;
    int64_t tcdmReadBytes = 0;
    int64_t tcdmWriteBytes = 0;
};

struct NeuralAICommandPerformanceResult
{
    PerformanceResult performance;
    std::vector<NeuralAICommandPerformance> commands;
};

/// Estimates the serialized Neural-AI command stream after command lowering.
/// This is intentionally a linear ABI scan and is not used by the scheduler.
NeuralAICommandPerformanceResult MeasureNeuralAICommandPerformance(
    const CompiledNeuralAIArtifact &artifact, Architecture *architecture, Database *db = nullptr);

}  // namespace regor
