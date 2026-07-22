//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_abi.hpp"

#include <limits>

namespace regor::neuralai
{
namespace
{

template<size_t SIZE>
void Write16(std::array<uint8_t, SIZE> &output, size_t &offset, uint16_t value)
{
    output[offset++] = uint8_t(value);
    output[offset++] = uint8_t(value >> 8);
}

template<size_t SIZE>
void Write32(std::array<uint8_t, SIZE> &output, size_t &offset, uint32_t value)
{
    output[offset++] = uint8_t(value);
    output[offset++] = uint8_t(value >> 8);
    output[offset++] = uint8_t(value >> 16);
    output[offset++] = uint8_t(value >> 24);
}

}  // namespace

SerializedModelHeaderV1 Serialize(const ModelHeaderV1 &value)
{
    SerializedModelHeaderV1 output{};
    size_t offset = 0;
    Write32(output, offset, value.magic);
    Write16(output, offset, value.abiMajor);
    Write16(output, offset, value.abiMinor);
    Write32(output, offset, value.targetId);
    Write32(output, offset, value.flags);
    Write32(output, offset, value.totalBytes);
    Write32(output, offset, value.sectionCount);
    Write32(output, offset, value.sectionTableOffset);
    Write32(output, offset, value.entryCommandOffset);
    Write32(output, offset, value.commandCount);
    Write32(output, offset, value.requiredTCDMBytes);
    Write32(output, offset, value.requiredTCDMAlignment);
    Write32(output, offset, value.inputCount);
    Write32(output, offset, value.outputCount);
    for ( uint32_t reserved : value.reserved ) Write32(output, offset, reserved);
    return output;
}

SerializedSectionV1 Serialize(const SectionV1 &value)
{
    SerializedSectionV1 output{};
    size_t offset = 0;
    Write32(output, offset, value.type);
    Write32(output, offset, value.flags);
    Write32(output, offset, value.offset);
    Write32(output, offset, value.size);
    Write32(output, offset, value.alignment);
    Write32(output, offset, value.elementCount);
    Write32(output, offset, value.crc32);
    Write32(output, offset, value.reserved);
    return output;
}

SerializedInvocationV1 Serialize(const InvocationV1 &value)
{
    SerializedInvocationV1 output{};
    size_t offset = 0;
    Write32(output, offset, value.magic);
    Write16(output, offset, value.abiMajor);
    Write16(output, offset, value.abiMinor);
    Write32(output, offset, value.totalBytes);
    Write32(output, offset, value.modelBase);
    Write32(output, offset, value.modelBytes);
    Write32(output, offset, value.bindingTableBase);
    Write32(output, offset, value.bindingCount);
    Write32(output, offset, value.flags);
    for ( uint32_t reserved : value.reserved ) Write32(output, offset, reserved);
    return output;
}

bool IsAligned(uint32_t value, uint32_t alignment)
{
    return alignment != 0 && (alignment & (alignment - 1)) == 0 && (value & (alignment - 1)) == 0;
}

bool IsValidRange(uint32_t offset, uint32_t size, uint32_t totalBytes)
{
    return offset <= totalBytes && size <= totalBytes - offset;
}

bool ValidateModelLayout(const ModelHeaderV1 &header, const SectionV1 *sections, size_t sectionCount, std::string &error)
{
    if ( header.magic != ModelMagic )
    {
        error = "invalid model magic";
        return false;
    }
    if ( header.abiMajor != AbiMajor )
    {
        error = "unsupported model ABI major version";
        return false;
    }
    if ( header.targetId != TargetId )
    {
        error = "model target does not match NeuralAI";
        return false;
    }
    if ( header.totalBytes < sizeof(ModelHeaderV1) || !IsAligned(header.totalBytes, Alignment) )
    {
        error = "model size is not a 32-byte-aligned package";
        return false;
    }
    if ( header.sectionCount != sectionCount || (sectionCount != 0 && sections == nullptr) )
    {
        error = "section count does not match the section table";
        return false;
    }
    if ( sectionCount > std::numeric_limits<uint32_t>::max() / sizeof(SectionV1) ||
         !IsValidRange(header.sectionTableOffset, uint32_t(sectionCount * sizeof(SectionV1)), header.totalBytes) ||
         !IsAligned(header.sectionTableOffset, Alignment) )
    {
        error = "section table is out of range or misaligned";
        return false;
    }
    if ( header.requiredTCDMAlignment != Alignment )
    {
        error = "unsupported TCDM alignment";
        return false;
    }

    for ( size_t index = 0; index < sectionCount; ++index )
    {
        const auto &section = sections[index];
        if ( !IsAligned(section.offset, Alignment) || !IsAligned(section.size, Alignment) ||
             section.alignment < Alignment || !IsAligned(section.offset, section.alignment) ||
             !IsValidRange(section.offset, section.size, header.totalBytes) )
        {
            error = "section " + std::to_string(index) + " is out of range or misaligned";
            return false;
        }
        if ( section.reserved != 0 )
        {
            error = "section " + std::to_string(index) + " has non-zero reserved fields";
            return false;
        }
    }
    return true;
}

}  // namespace regor::neuralai
