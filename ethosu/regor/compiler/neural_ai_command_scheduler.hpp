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
    int64_t estimatedCycles = 0;
};

// Schedules one blocking semantic stream across the two DMA queues, systolic
// pending slot, AFU, and Spatz. Dependencies remain authoritative; the resource
// calendar decides which eligible command can run while asynchronous work is
// pending and inserts waits only at consumers, fences, or queue pressure.
bool ScheduleGlobalCommandStream(const std::vector<SemanticCommand> &commands,
    const CommandDependencyGraph &dependencies, std::vector<uint8_t> &scheduledBytes,
    GlobalCommandScheduleStats &stats, std::string &error);

}  // namespace regor::neuralai
