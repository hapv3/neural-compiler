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

constexpr uint32_t MaxAffineLoopPatches = 24;
constexpr uint32_t MaxAffineLoopBodyCommands = 16;
constexpr uint32_t MaxAffineLoopRecordBytes = 2048;

struct CommandCompactionStats
{
    uint32_t logicalCommands = 0;
    uint32_t encodedCommands = 0;
    uint32_t loops = 0;
    uint32_t loopedCommands = 0;
    uint32_t bytesBefore = 0;
    uint32_t bytesAfter = 0;
};

// Compacts repeated affine command sequences after global scheduling. The
// semantic command list remains the authoritative expanded trace/debug map.
bool CompactAffineCommandStream(const std::vector<SemanticCommand> &commands,
    std::vector<uint8_t> &encoded, CommandCompactionStats &stats, std::string &error);

// Expands an ABI stream for tests, tooling, and logical command slicing.
bool ExpandAffineCommandStream(const std::vector<uint8_t> &encoded,
    std::vector<uint8_t> &expanded, uint32_t &logicalCommands, std::string &error);

}  // namespace regor::neuralai
