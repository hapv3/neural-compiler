//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_writer.hpp"

#include "common/numeric_util.hpp"

#include <array>
#include <limits>

namespace regor
{
namespace
{

void Append16(std::vector<uint8_t> &output, uint16_t value)
{
    output.push_back(uint8_t(value));
    output.push_back(uint8_t(value >> 8));
}

void Append32(std::vector<uint8_t> &output, uint32_t value)
{
    output.push_back(uint8_t(value));
    output.push_back(uint8_t(value >> 8));
    output.push_back(uint8_t(value >> 16));
    output.push_back(uint8_t(value >> 24));
}

void AppendTensor(std::vector<uint8_t> &output, const neuralai::TensorV1 &value)
{
    Append32(output, value.tensorId);
    Append32(output, value.flags);
    Append16(output, value.dataType);
    Append16(output, value.layout);
    Append16(output, value.rank);
    Append16(output, value.reserved0);
    for ( uint32_t dimension : value.dimensions ) Append32(output, dimension);
    Append32(output, value.byteSize);
    Append32(output, value.alignment);
    Append32(output, value.scratchOffset);
    Append32(output, value.qparamIndex);
    for ( uint32_t reserved : value.reserved ) Append32(output, reserved);
}

void AppendBinding(std::vector<uint8_t> &output, const neuralai::BindingV1 &value)
{
    Append16(output, value.direction);
    Append16(output, value.index);
    Append16(output, value.dataType);
    Append16(output, value.layout);
    Append16(output, value.rank);
    Append16(output, value.reserved0);
    Append32(output, value.tensorId);
    for ( uint32_t dimension : value.dimensions ) Append32(output, dimension);
    Append32(output, value.byteSize);
    Append32(output, value.scaleBits);
    Append32(output, uint32_t(value.zeroPoint));
    Append32(output, value.flags);
    for ( uint32_t reserved : value.reserved ) Append32(output, reserved);
}

void AppendQParam(std::vector<uint8_t> &output, const neuralai::QParamV1 &value)
{
    Append32(output, uint32_t(value.bias));
    Append32(output, uint32_t(value.multiplier));
    Append32(output, value.shift);
    Append32(output, uint32_t(value.zeroPoint));
    Append32(output, uint32_t(value.clampMin));
    Append32(output, uint32_t(value.clampMax));
    for ( uint32_t reserved : value.reserved ) Append32(output, reserved);
}

void Pad32(std::vector<uint8_t> &data)
{
    data.resize(RoundAway(int(data.size()), int(neuralai::Alignment)), 0);
}

bool ProductBytes(const neuralai::BindingV1 &binding, uint32_t &bytes)
{
    if ( binding.rank == 0 || binding.rank > 4 ) return false;
    uint64_t elements = 1;
    for ( int axis = 0; axis < binding.rank; ++axis )
    {
        if ( binding.dimensions[axis] == 0 ) return false;
        elements *= binding.dimensions[axis];
    }
    const uint32_t elementBytes = binding.dataType == uint16_t(neuralai::DataType::Int8) ? 1U :
        binding.dataType == uint16_t(neuralai::DataType::Int32) ? 4U : 0U;
    const uint64_t result = elements * elementBytes;
    if ( elementBytes == 0 || result > std::numeric_limits<uint32_t>::max() ) return false;
    bytes = uint32_t(result);
    return true;
}

struct SectionPayload
{
    neuralai::SectionType type;
    uint32_t elementCount;
    std::vector<uint8_t> bytes;
};

}  // namespace

bool WriteNeuralAIModel(const CompiledNeuralAIArtifact &artifact, std::vector<uint8_t> &output, std::string &error)
{
    if ( artifact.commands.empty() || artifact.commands.size() % neuralai::Alignment != 0 )
    {
        error = "NeuralAI commands must be non-empty and 32-byte aligned";
        return false;
    }
    if ( artifact.requiredTCDMBytes > 0x0007F000U )
    {
        error = "NeuralAI TCDM requirement exceeds the 508 KiB allocatable arena";
        return false;
    }

    uint32_t inputCount = 0;
    uint32_t outputCount = 0;
    uint32_t l2TemporaryCount = 0;
    for ( const auto &binding : artifact.bindings )
    {
        uint32_t compactBytes = 0;
        if ( !ProductBytes(binding, compactBytes) || binding.byteSize != compactBytes )
        {
            error = "NeuralAI public binding has an invalid compact byte size";
            return false;
        }
        const bool publicIO = binding.direction == uint16_t(neuralai::BindingDirection::Input) ||
                              binding.direction == uint16_t(neuralai::BindingDirection::Output);
        if ( publicIO && binding.rank == 4 &&
             binding.layout != uint16_t(neuralai::TensorLayout::NHWC) )
        {
            error = "NeuralAI ABI v1 rank-4 public bindings must use NHWC";
            return false;
        }
        if ( binding.direction == uint16_t(neuralai::BindingDirection::Input) ) ++inputCount;
        else if ( binding.direction == uint16_t(neuralai::BindingDirection::Output) ) ++outputCount;
        else if ( binding.direction == uint16_t(neuralai::BindingDirection::L2Temporary) &&
                  binding.index == 0 && l2TemporaryCount++ == 0 )
        {
            // The model owns one relocatable L2 arena.  Its runtime base is
            // supplied alongside the public bindings at invocation time.
        }
        else
        {
            error = "NeuralAI binding has an invalid direction or L2 temporary index";
            return false;
        }
    }

    std::array<SectionPayload, 5> payloads = {{
        {neuralai::SectionType::Commands, artifact.commandCount, artifact.commands},
        {neuralai::SectionType::Constants, 0, artifact.constants},
        {neuralai::SectionType::Tensors, uint32_t(artifact.tensors.size()), {}},
        {neuralai::SectionType::Bindings, uint32_t(artifact.bindings.size()), {}},
        {neuralai::SectionType::QParams, uint32_t(artifact.qparams.size()), {}},
    }};
    for ( const auto &tensor : artifact.tensors ) AppendTensor(payloads[2].bytes, tensor);
    for ( const auto &binding : artifact.bindings ) AppendBinding(payloads[3].bytes, binding);
    for ( const auto &qparam : artifact.qparams ) AppendQParam(payloads[4].bytes, qparam);
    for ( auto &payload : payloads ) Pad32(payload.bytes);

    constexpr uint32_t sectionTableOffset = sizeof(neuralai::ModelHeaderV1);
    uint32_t offset = sectionTableOffset + uint32_t(payloads.size() * sizeof(neuralai::SectionV1));
    std::array<neuralai::SectionV1, payloads.size()> sections{};
    for ( size_t index = 0; index < payloads.size(); ++index )
    {
        const auto &payload = payloads[index];
        if ( payload.bytes.size() > std::numeric_limits<uint32_t>::max() - offset )
        {
            error = "NeuralAI package exceeds the 32-bit ABI size";
            return false;
        }
        sections[index] = {uint32_t(payload.type), 0, offset, uint32_t(payload.bytes.size()),
            neuralai::Alignment, payload.elementCount, {0, 0}};
        offset += uint32_t(payload.bytes.size());
    }

    neuralai::ModelHeaderV1 header{};
    header.magic = neuralai::ModelMagic;
    header.abiMajor = neuralai::AbiMajor;
    header.abiMinor = neuralai::AbiMinor;
    header.targetId = neuralai::TargetId;
    header.totalBytes = offset;
    header.sectionCount = uint32_t(sections.size());
    header.sectionTableOffset = sectionTableOffset;
    header.entryCommandOffset = sections[0].offset;
    header.commandCount = artifact.commandCount;
    header.requiredTCDMBytes = RoundAway(artifact.requiredTCDMBytes, neuralai::Alignment);
    header.requiredTCDMAlignment = neuralai::Alignment;
    header.inputCount = inputCount;
    header.outputCount = outputCount;

    output.clear();
    output.reserve(header.totalBytes);
    const auto headerBytes = neuralai::Serialize(header);
    output.insert(output.end(), headerBytes.begin(), headerBytes.end());
    for ( const auto &section : sections )
    {
        const auto sectionBytes = neuralai::Serialize(section);
        output.insert(output.end(), sectionBytes.begin(), sectionBytes.end());
    }
    for ( const auto &payload : payloads ) output.insert(output.end(), payload.bytes.begin(), payload.bytes.end());
    return true;
}

}  // namespace regor
