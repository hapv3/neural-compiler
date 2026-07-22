//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/neural_ai_writer.hpp"

#include <catch_all.hpp>

#include <algorithm>
#include <array>

using namespace regor;

namespace
{

uint32_t Read32(const std::vector<uint8_t> &bytes, int offset)
{
    return uint32_t(bytes[offset]) | (uint32_t(bytes[offset + 1]) << 8) |
           (uint32_t(bytes[offset + 2]) << 16) | (uint32_t(bytes[offset + 3]) << 24);
}

neuralai::BindingV1 MakeBinding(neuralai::BindingDirection direction, uint16_t index)
{
    neuralai::BindingV1 binding{};
    binding.direction = uint16_t(direction);
    binding.index = index;
    binding.dataType = uint16_t(neuralai::DataType::Int8);
    binding.layout = uint16_t(neuralai::TensorLayout::NHWC);
    binding.rank = 4;
    binding.dimensions[0] = 1;
    binding.dimensions[1] = 2;
    binding.dimensions[2] = 3;
    binding.dimensions[3] = 5;
    binding.byteSize = 30;
    return binding;
}

}  // namespace

TEST_CASE("neural_ai_writer emits deterministic relocatable package")
{
    CompiledNeuralAIArtifact artifact;
    artifact.commands.resize(32);
    artifact.commands[0] = uint8_t(neuralai::CommandType::End);
    artifact.constants = {1, 2, 3};
    artifact.bindings.push_back(MakeBinding(neuralai::BindingDirection::Input, 0));
    artifact.bindings.push_back(MakeBinding(neuralai::BindingDirection::Output, 0));
    artifact.commandCount = 1;
    artifact.requiredTCDMBytes = 33;

    std::vector<uint8_t> first;
    std::vector<uint8_t> second;
    std::string error;
    REQUIRE(WriteNeuralAIModel(artifact, first, error));
    REQUIRE(WriteNeuralAIModel(artifact, second, error));
    REQUIRE(first == second);
    REQUIRE(first.size() % neuralai::Alignment == 0);
    REQUIRE(Read32(first, 0) == neuralai::ModelMagic);
    REQUIRE(Read32(first, 16) == first.size());
    REQUIRE(Read32(first, 20) == 5);
    REQUIRE(Read32(first, 36) == 64);
    REQUIRE(Read32(first, 44) == 1);
    REQUIRE(Read32(first, 48) == 1);

    // Constants are data only; neither package nor references contain a host address.
    const std::array<uint8_t, 4> absoluteL2Base = {0x00, 0x00, 0x00, 0x80};
    REQUIRE(std::search(first.begin(), first.end(), absoluteL2Base.begin(), absoluteL2Base.end()) == first.end());
}

TEST_CASE("neural_ai_writer rejects native public layout")
{
    CompiledNeuralAIArtifact artifact;
    artifact.commands.resize(32);
    artifact.bindings.push_back(MakeBinding(neuralai::BindingDirection::Input, 0));
    artifact.bindings[0].layout = uint16_t(neuralai::TensorLayout::Row32);

    std::vector<uint8_t> output;
    std::string error;
    REQUIRE_FALSE(WriteNeuralAIModel(artifact, output, error));
    REQUIRE(error.find("NHWC") != std::string::npos);
}
