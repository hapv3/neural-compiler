//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "architecture/neuralai/neural_ai_abi.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace regor::neuralai
{

enum class SemanticAccessMode : uint8_t
{
    Read,
    Write,
};

enum class SemanticResource : uint32_t
{
    None = 0,
    Control = 1U << 0,
    DMAExternalToLocal = 1U << 1,
    DMALocalToExternal = 1U << 2,
    Systolic = 1U << 3,
    AFU = 1U << 4,
    Spatz = 1U << 5,
};

enum class SemanticState : uint8_t
{
    Quantization,
    SystolicSlot,
    AFUEngine,
    AFULut,
    SpatzEngine,
    DMAQueue,
};

enum class DependencyKind : uint8_t
{
    MemoryRAW,
    MemoryWAR,
    MemoryWAW,
    StateRAW,
    StateWAR,
    StateWAW,
    Control,
};

struct SemanticContentSlice
{
    uint64_t begin = 0;
    uint64_t end = 0;
    uint64_t identity = 0;
    uint32_t generation = 0;
};

struct SemanticMemoryAccess
{
    uint16_t region = 0;
    uint16_t index = 0;
    uint64_t begin = 0;
    uint64_t end = 0;
    SemanticAccessMode mode = SemanticAccessMode::Read;
    // Valid for TCDM accesses. UINT16_MAX denotes a non-TCDM access.
    uint16_t bankPhase = UINT16_MAX;
    std::vector<SemanticContentSlice> content;
};

struct SemanticStateAccess
{
    SemanticState state = SemanticState::Quantization;
    uint32_t instance = 0;
    SemanticAccessMode mode = SemanticAccessMode::Read;
    uint64_t valueIdentity = 0;
    uint32_t generation = 0;
};

struct SemanticCommand
{
    CommandType type = CommandType::End;
    uint32_t flags = 0;
    uint32_t layerId = 0;
    uint32_t tileId = 0;
    uint32_t sourceCommandIndex = 0;
    uint32_t resources = 0;
    std::vector<SemanticMemoryAccess> memoryAccesses;
    std::vector<SemanticStateAccess> stateAccesses;
    bool asyncCapable = false;
    bool asynchronous = false;
    bool controlFence = false;
    uint32_t queueCapacity = 0;
    uint32_t queueId = UINT32_MAX;
    int64_t estimatedCycles = 1;
    // Retained until typed command serializers replace the legacy emitters.
    // The final ABI byte stream is always regenerated from these records.
    std::vector<uint8_t> encoding;
};

struct CommandDependency
{
    uint32_t predecessor = 0;
    uint32_t successor = 0;
    DependencyKind kind = DependencyKind::MemoryRAW;
};

struct CommandDependencyGraph
{
    std::vector<CommandDependency> edges;
    std::vector<std::vector<uint32_t>> predecessors;
    std::vector<std::vector<uint32_t>> successors;
};

constexpr uint32_t ResourceMask(SemanticResource resource)
{
    return uint32_t(resource);
}

bool DecodeSemanticCommandStream(const std::vector<uint8_t> &bytes,
    std::vector<SemanticCommand> &commands, std::string &error);

bool BuildCommandDependencyGraph(std::vector<SemanticCommand> &commands,
    CommandDependencyGraph &graph, std::string &error);

bool ValidateCommandDependencyGraph(const std::vector<SemanticCommand> &commands,
    const CommandDependencyGraph &graph, std::string &error);

std::vector<uint8_t> SerializeSemanticCommandStream(
    const std::vector<SemanticCommand> &commands);

}  // namespace regor::neuralai
