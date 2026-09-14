//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "neural_ai_command_ir.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace regor::neuralai
{

struct GlobalCommandScheduleStats
{
    uint32_t asynchronousSubmits = 0;
    uint32_t insertedWaits = 0;
    uint32_t reorderedCommands = 0;
    uint32_t bankConflictDeferrals = 0;
    uint32_t qparamPreloads = 0;
    int64_t estimatedCycles = 0;
};

struct QParamResidencyStats
{
    uint32_t inputLoads = 0;
    uint32_t retainedLoads = 0;
    uint32_t redundantLoads = 0;
};

struct AFULutResidencyStats
{
    uint32_t commands = 0;
    uint32_t loads = 0;
    uint32_t reused = 0;
    uint32_t invalidations = 0;
};

struct DMAResidencyStats
{
    uint32_t inputLoads = 0;
    uint32_t retainedLoads = 0;
    uint32_t redundantConstantLoads = 0;
    uint32_t redundantFeatureLoads = 0;
    uint64_t redundantBytes = 0;
};

// Retains the currently programmed qparam block across systolic commands and
// removes a reload only when its immutable QParams reference, count, and
// logical block identity are unchanged. The pass operates before dependency
// construction so all state generations describe the compacted stream.
bool OptimizeQParamResidency(std::vector<SemanticCommand> &commands,
    QParamResidencyStats &stats, std::string &error);

// Marks AFU LUT commands as resident hits only while the referenced immutable
// table remains active. DFL16 and GlobalAvgPool repurpose the shared LUT banks
// and therefore invalidate the tracked table.
bool OptimizeAFULutResidency(std::vector<SemanticCommand> &commands,
    AFULutResidencyStats &stats, std::string &error);

// Removes an external-to-TCDM transfer only when the exact source-to-
// destination mapping is still resident and neither range has been written.
// Bounding-span invalidation is deliberately conservative for strided copies.
bool OptimizeDMAResidency(std::vector<SemanticCommand> &commands,
    DMAResidencyStats &stats, std::string &error);

// Schedules one blocking semantic stream across the two DMA queues, systolic
// pending slot, AFU, and Spatz. Dependencies remain authoritative; the resource
// calendar decides which eligible command can run while asynchronous work is
// pending and inserts waits only at consumers, fences, or queue pressure.
bool ScheduleGlobalCommandStream(const std::vector<SemanticCommand> &commands,
    const CommandDependencyGraph &dependencies, std::vector<uint8_t> &scheduledBytes,
    GlobalCommandScheduleStats &stats, std::string &error);

}  // namespace regor::neuralai
