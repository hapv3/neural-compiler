//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the License); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "tflite/custom_operator_ethosu.hpp"

#include <catch_all.hpp>

using namespace regor;

static uint32_t read32(std::unique_ptr<Buffer> &buffer, int byteOffset)
{
    assert(byteOffset + 4 <= buffer->Size());
    uint32_t word;
    std::copy_n(&buffer->Data<uint8_t>()[byteOffset], sizeof(word), reinterpret_cast<uint8_t *>(&word));
    return word;
}

static uint64_t read64(std::unique_ptr<Buffer> &buffer, int byteOffset)
{
    assert(byteOffset + 8 <= buffer->Size());
    uint64_t dword;
    std::copy_n(&buffer->Data<uint8_t>()[byteOffset], sizeof(dword), reinterpret_cast<uint8_t *>(&dword));
    return dword;
}

TEST_CASE("Custom Operator Payload 1")
{
    const uint32_t FOURCC_WORD = 0x31504F43;   // COP1
    const uint32_t VERSION_WORD = 0x00100001;  // Version 0.1
    const uint32_t ARCH_CONFIG_WORD = 0x12345678;
    const uint32_t ARCH_VERSION_WORD = 0x23456789;
    const uint32_t NOP_WORD = 0x00000005;
    const uint32_t COMMAND_STREAM_WORD = 0x00010002;  // Size 1

    std::vector<uint32_t> commandStream = {0xDEADBEEF};
    auto buffer = DriverActions::CreateDriverPayload1(commandStream, ARCH_CONFIG_WORD, ARCH_VERSION_WORD);

    REQUIRE(buffer->Size() == 36);
    REQUIRE(read32(buffer, 0) == FOURCC_WORD);
    REQUIRE(read32(buffer, 4) == VERSION_WORD);
    REQUIRE(read32(buffer, 8) == ARCH_CONFIG_WORD);
    REQUIRE(read32(buffer, 12) == ARCH_VERSION_WORD);
    REQUIRE(read32(buffer, 16) == NOP_WORD);
    REQUIRE(read32(buffer, 20) == NOP_WORD);
    REQUIRE(read32(buffer, 24) == NOP_WORD);
    REQUIRE(read32(buffer, 28) == COMMAND_STREAM_WORD);
    REQUIRE(read32(buffer, 32) == commandStream[0]);
}

TEST_CASE("Custom Operator Payload 2")
{
    const uint32_t FOURCC_WORD_1 = 0x32504F43;   // COP2
    const uint32_t FOURCC_WORD_2 = 0x4D554556;   // VEUM
    const uint32_t VERSION_WORD_1 = 0x00000001;  // Version 1.0
    const uint32_t VERSION_WORD_2 = 0x00000001;  // Version 1.0
    const uint32_t ARCH_CONFIG_WORD = 0x12345678;
    const uint32_t ARCH_VERSION_WORD = 0x23456789;
    const uint32_t STAGING_USAGE = 5678;

    SECTION("Direct Drive")
    {
        std::vector<uint32_t> commandStream = {0xDEADBEEF};
        auto buffer = DriverActions::CreateDriverPayload2(commandStream, ARCH_CONFIG_WORD, ARCH_VERSION_WORD, STAGING_USAGE, true);

        REQUIRE(buffer->Size() == 68);
        REQUIRE(read32(buffer, 0) == FOURCC_WORD_1);
        REQUIRE(read32(buffer, 4) == VERSION_WORD_1);
        REQUIRE(read64(buffer, 8) == 68 - 16);
        REQUIRE(read32(buffer, 16) == 1);   // Command Stream type
        REQUIRE(read32(buffer, 20) == 44);  // Command Stream entry data length (bytes)
        REQUIRE(read32(buffer, 24) == 36);  // Metadata length (bytes)
        REQUIRE(read32(buffer, 28) == FOURCC_WORD_2);
        REQUIRE(read32(buffer, 32) == VERSION_WORD_2);
        REQUIRE(read32(buffer, 36) == 20);
        REQUIRE(read32(buffer, 40) == ARCH_CONFIG_WORD);
        REQUIRE(read32(buffer, 44) == ARCH_VERSION_WORD);
        REQUIRE(read32(buffer, 48) == 0x2);            // Feature mask
        REQUIRE(read64(buffer, 52) == STAGING_USAGE);  // Cache memory size
        REQUIRE(read32(buffer, 60) == 0xFFFFFFFF);     // Padding
        REQUIRE(read32(buffer, 64) == commandStream[0]);
    }

    SECTION("ML Island")
    {
        std::vector<uint32_t> commandStream = {0xDEADBEEF};
        auto buffer = DriverActions::CreateDriverPayload2(commandStream, ARCH_CONFIG_WORD, ARCH_VERSION_WORD, STAGING_USAGE, false);

        REQUIRE(buffer->Size() == 68);
        REQUIRE(read32(buffer, 0) == FOURCC_WORD_1);
        REQUIRE(read32(buffer, 4) == VERSION_WORD_1);
        REQUIRE(read64(buffer, 8) == 68 - 16);
        REQUIRE(read32(buffer, 16) == 1);   // Command Stream type
        REQUIRE(read32(buffer, 20) == 44);  // Command Stream entry data length (bytes)
        REQUIRE(read32(buffer, 24) == 36);  // Metadata length (bytes)
        REQUIRE(read32(buffer, 28) == FOURCC_WORD_2);
        REQUIRE(read32(buffer, 32) == VERSION_WORD_2);
        REQUIRE(read32(buffer, 36) == 20);
        REQUIRE(read32(buffer, 40) == ARCH_CONFIG_WORD);
        REQUIRE(read32(buffer, 44) == ARCH_VERSION_WORD);
        REQUIRE(read32(buffer, 48) == 0x1);            // Feature mask
        REQUIRE(read64(buffer, 52) == STAGING_USAGE);  // Cache memory size
        REQUIRE(read32(buffer, 60) == 0xFFFFFFFF);     // Padding
        REQUIRE(read32(buffer, 64) == commandStream[0]);
    }
}
