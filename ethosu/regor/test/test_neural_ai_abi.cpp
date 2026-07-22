//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "architecture/neuralai/neural_ai_abi.hpp"

#include <catch_all.hpp>

#include <array>
#include <limits>
#include <string>

using namespace regor::neuralai;

TEST_CASE("neural_ai_abi model header has deterministic little-endian bytes")
{
    ModelHeaderV1 header{};
    header.magic = ModelMagic;
    header.abiMajor = AbiMajor;
    header.abiMinor = AbiMinor;
    header.targetId = TargetId;
    header.totalBytes = 0x400;
    header.sectionCount = 5;
    header.sectionTableOffset = 0x40;
    header.entryCommandOffset = 0x100;
    header.commandCount = 3;
    header.requiredTCDMBytes = 0x12340;
    header.requiredTCDMAlignment = Alignment;
    header.inputCount = 1;
    header.outputCount = 2;

    const auto bytes = Serialize(header);
    const SerializedModelHeaderV1 expected = {
        0x4E, 0x41, 0x49, 0x4D, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x04, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
        0x40, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x00, 0x40, 0x23, 0x01, 0x00,
        0x20, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    REQUIRE(bytes == expected);
}

TEST_CASE("neural_ai_abi validates ranges without integer overflow")
{
    REQUIRE(IsValidRange(64, 32, 96));
    REQUIRE(IsValidRange(96, 0, 96));
    REQUIRE_FALSE(IsValidRange(97, 0, 96));
    REQUIRE_FALSE(IsValidRange(std::numeric_limits<uint32_t>::max() - 15, 32,
        std::numeric_limits<uint32_t>::max()));
}

TEST_CASE("neural_ai_abi freezes RQ load command layout")
{
    CommandRQLoadV2 command{};
    command.header.type = uint16_t(CommandType::RQLoad);
    command.header.sizeBytes = sizeof(command);
    command.qparamIndex = 64;
    command.qparamCount = 32;
    command.qparamBlock = 2;

    REQUIRE(command.header.type == 5);
    REQUIRE(command.header.sizeBytes == 32);
    REQUIRE(command.qparamIndex == 64);
    REQUIRE(command.qparamCount == 32);
    REQUIRE(command.qparamBlock == 2);
    REQUIRE(command.reserved == 0);
}

TEST_CASE("neural_ai_abi validates section alignment and bounds")
{
    ModelHeaderV1 header{};
    header.magic = ModelMagic;
    header.abiMajor = AbiMajor;
    header.targetId = TargetId;
    header.totalBytes = 160;
    header.sectionCount = 1;
    header.sectionTableOffset = 64;
    header.requiredTCDMAlignment = Alignment;

    SectionV1 section{};
    section.type = uint32_t(SectionType::Commands);
    section.offset = 96;
    section.size = 64;
    section.alignment = Alignment;

    std::string error;
    REQUIRE(ValidateModelLayout(header, &section, 1, error));

    SECTION("misaligned section")
    {
        section.offset = 97;
        REQUIRE_FALSE(ValidateModelLayout(header, &section, 1, error));
        REQUIRE(error.find("misaligned") != std::string::npos);
    }
    SECTION("overflowing range")
    {
        section.offset = 128;
        section.size = std::numeric_limits<uint32_t>::max() - 127;
        REQUIRE_FALSE(ValidateModelLayout(header, &section, 1, error));
        REQUIRE(error.find("out of range") != std::string::npos);
    }
    SECTION("non-zero reserved field")
    {
        section.reserved = 1;
        REQUIRE_FALSE(ValidateModelLayout(header, &section, 1, error));
        REQUIRE(error.find("reserved") != std::string::npos);
    }
}
