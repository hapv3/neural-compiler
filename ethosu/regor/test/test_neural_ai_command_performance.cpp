//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "architecture/neuralai/neural_ai.hpp"
#include "compiler/database.hpp"
#include "compiler/neural_ai_command_performance.hpp"

#include <catch_all.hpp>

#include <algorithm>

using namespace regor;

namespace
{

template<typename TYPE>
void AppendCommand(CompiledNeuralAIArtifact &artifact, const TYPE &command)
{
    const auto *bytes = reinterpret_cast<const uint8_t *>(&command);
    artifact.commands.insert(artifact.commands.end(), bytes, bytes + sizeof(command));
    ++artifact.commandCount;
}

}  // namespace

TEST_CASE("neural_ai command performance measures exact DMA3D repetitions")
{
    ArchNeuralAI architecture;
    REQUIRE(architecture.ConfigureExternalMemory(1.0f, 10, 5, 64, 1, 64, 32));

    neuralai::CommandDMA3DV2 command{};
    command.header.type = uint16_t(neuralai::CommandType::DMA3D);
    command.header.sizeBytes = sizeof(command);
    command.header.layerId = 7;
    command.header.tileId = 3;
    command.source = {uint16_t(neuralai::Region::ModelConstants), 0, 128};
    command.destination = {uint16_t(neuralai::Region::TCDMScratch), 0, 256};
    command.length = 64;
    command.repetitions2 = 2;
    command.repetitions3 = 3;

    CompiledNeuralAIArtifact artifact;
    AppendCommand(artifact, command);
    const auto result = MeasureNeuralAICommandPerformance(artifact, &architecture);

    REQUIRE(result.commands.size() == 1);
    CHECK(result.commands[0].name == "DMA3D");
    CHECK(result.commands[0].layerId == 7);
    CHECK(result.commands[0].tileId == 3);
    CHECK(result.commands[0].modelReadBytes == 384);
    CHECK(result.commands[0].tcdmWriteBytes == 384);
    CHECK(result.commands[0].memoryCycles == 22);
    CHECK(result.performance.npuCycles == 22);
    CHECK(result.performance.memory.at(architecture.ReadonlyMemory().memory)
              .access.at(AccessType::Weights).bytesRead == 384);
}

TEST_CASE("neural_ai command performance accounts AFU traffic and emits command table")
{
    ArchNeuralAI architecture;
    neuralai::CommandAFUBinaryV2 binary{};
    binary.header.type = uint16_t(neuralai::CommandType::AFUBinary);
    binary.header.sizeBytes = sizeof(binary);
    binary.header.layerId = 19;
    binary.lhs = {uint16_t(neuralai::Region::TCDMScratch), 0, 0};
    binary.rhs = {uint16_t(neuralai::Region::TCDMScratch), 0, 512};
    binary.ofm = {uint16_t(neuralai::Region::TCDMScratch), 0, 1024};
    binary.length = 320;
    binary.mode = uint32_t(neuralai::AFUBinaryMode::AddI8);

    neuralai::CommandHeaderV2 end{};
    end.type = uint16_t(neuralai::CommandType::End);
    end.sizeBytes = 32;

    CompiledNeuralAIArtifact artifact;
    AppendCommand(artifact, binary);
    const auto *endBytes = reinterpret_cast<const uint8_t *>(&end);
    artifact.commands.insert(artifact.commands.end(), endBytes, endBytes + sizeof(end));
    artifact.commands.resize(artifact.commands.size() + 16);
    ++artifact.commandCount;

    Database database;
    const auto result = MeasureNeuralAICommandPerformance(artifact, &architecture, &database);

    REQUIRE(result.commands.size() == 2);
    CHECK(result.commands[0].computeCycles == 10);
    CHECK(result.commands[0].memoryCycles == 30);
    CHECK(result.commands[0].totalCycles == 30);
    CHECK(result.commands[0].tcdmReadBytes == 640);
    CHECK(result.commands[0].tcdmWriteBytes == 320);
    CHECK(result.commands[1].name == "END");
    CHECK(result.commands[1].totalCycles == 1);
    CHECK(result.performance.npuCycles == 31);

    auto *tables = database.Tables();
    REQUIRE(tables->Next());
    CHECK(tables->Name() == "nai_command_perf");
    CHECK(tables->Rows() == 2);
    CHECK(tables->Columns() == 13);
    tables->Release();
}

TEST_CASE("neural_ai command performance hides asynchronous DMA store cycles")
{
    ArchNeuralAI architecture;
    REQUIRE(architecture.ConfigureExternalMemory(1.0f, 10, 5, 64, 1, 64, 32));

    neuralai::CommandDMA1DV2 store{};
    store.header.type = uint16_t(neuralai::CommandType::DMASubmit1D);
    store.header.sizeBytes = sizeof(store);
    store.source = {uint16_t(neuralai::Region::TCDMScratch), 0, 0};
    store.destination = {uint16_t(neuralai::Region::L2TemporaryBinding), 0, 0};
    store.length = 256;
    store.direction = uint32_t(neuralai::DMADirection::LocalToExternal);

    neuralai::CommandDMA1DV2 load{};
    load.header.type = uint16_t(neuralai::CommandType::DMA1D);
    load.header.sizeBytes = sizeof(load);
    load.source = {uint16_t(neuralai::Region::ModelConstants), 0, 0};
    load.destination = {uint16_t(neuralai::Region::TCDMScratch), 0, 1024};
    load.length = 256;
    load.direction = uint32_t(neuralai::DMADirection::ExternalToLocal);

    neuralai::CommandDMAWaitV2 wait{};
    wait.header.type = uint16_t(neuralai::CommandType::DMAWait);
    wait.header.sizeBytes = sizeof(wait);
    wait.direction = uint32_t(neuralai::DMADirection::LocalToExternal);

    CompiledNeuralAIArtifact artifact;
    AppendCommand(artifact, store);
    AppendCommand(artifact, load);
    AppendCommand(artifact, wait);
    const auto result = MeasureNeuralAICommandPerformance(artifact, &architecture);

    REQUIRE(result.commands.size() == 3);
    CHECK(result.commands[0].name == "DMA_SUBMIT_1D");
    CHECK(result.commands[0].totalCycles == 1);
    CHECK(result.commands[0].l2WriteBytes == 256);
    CHECK(result.commands[1].name == "DMA1D");
    CHECK(result.commands[2].name == "DMA_WAIT");
    const int64_t remaining = std::max<int64_t>(0,
        result.commands[0].memoryCycles - result.commands[1].totalCycles);
    CHECK(result.commands[2].memoryCycles == remaining);
    CHECK(result.commands[2].totalCycles == std::max<int64_t>(1, remaining));
    CHECK(result.performance.npuCycles == 1 + result.commands[1].totalCycles +
        result.commands[2].totalCycles);
}

TEST_CASE("neural_ai command performance stops at a truncated command")
{
    ArchNeuralAI architecture;
    CompiledNeuralAIArtifact artifact;
    artifact.commands.resize(sizeof(neuralai::CommandHeaderV2));
    artifact.commands[0] = uint8_t(neuralai::CommandType::DMA1D);
    artifact.commands[2] = uint8_t(sizeof(neuralai::CommandDMA1DV2));

    const auto result = MeasureNeuralAICommandPerformance(artifact, &architecture);
    CHECK(result.commands.empty());
    CHECK(result.performance.totalCycles == 0);
}
