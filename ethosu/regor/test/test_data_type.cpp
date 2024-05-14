//
// SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/data_type.hpp"

#include <catch_all.hpp>

using namespace regor;

TEST_CASE("DataType sizes")
{
    SECTION("Size bits")
    {
        REQUIRE(DataTypeSizeBits(DataType::Int4) == 4);
        REQUIRE(DataTypeSizeBits(DataType::Int4Packed8) == 4);
        REQUIRE(DataTypeSizeBits(DataType::Int8) == 8);
        REQUIRE(DataTypeSizeBits(DataType::Int16) == 16);
        REQUIRE(DataTypeSizeBits(DataType::Int32) == 32);
        REQUIRE(DataTypeSizeBits(DataType::Int48) == 48);
        REQUIRE(DataTypeSizeBits(DataType::Int64) == 64);
        REQUIRE(DataTypeSizeBits(DataType::UInt8) == 8);
        REQUIRE(DataTypeSizeBits(DataType::UInt16) == 16);
        REQUIRE(DataTypeSizeBits(DataType::UInt32) == 32);
        REQUIRE(DataTypeSizeBits(DataType::UInt48) == 48);
        REQUIRE(DataTypeSizeBits(DataType::UInt64) == 64);
    }

    SECTION("Storage size bits")
    {
        REQUIRE(DataTypeStorageSizeBits(DataType::Int4Packed8) == 8);
        REQUIRE(DataTypeStorageSizeBits(DataType::Int4) == 8);
        REQUIRE(DataTypeStorageSizeBits(DataType::Int8) == 8);
        REQUIRE(DataTypeStorageSizeBits(DataType::Int16) == 16);
        REQUIRE(DataTypeStorageSizeBits(DataType::Int32) == 32);
        REQUIRE(DataTypeStorageSizeBits(DataType::Int48) == 48);
        REQUIRE(DataTypeStorageSizeBits(DataType::Int64) == 64);
        REQUIRE(DataTypeStorageSizeBits(DataType::UInt8) == 8);
        REQUIRE(DataTypeStorageSizeBits(DataType::UInt16) == 16);
        REQUIRE(DataTypeStorageSizeBits(DataType::UInt32) == 32);
        REQUIRE(DataTypeStorageSizeBits(DataType::UInt48) == 48);
        REQUIRE(DataTypeStorageSizeBits(DataType::UInt64) == 64);
    }

    SECTION("Storage size bytes")
    {
        const int kMax = 16;
        for ( int elements = 0; elements <= kMax; ++elements )
        {
            CAPTURE(elements);
            REQUIRE(DataTypeStorageSizeBytes(DataType::Int4Packed8, elements) == 1 * (elements + 1) / 2);
            REQUIRE(DataTypeStorageSizeBytes(DataType::Int4, elements) == 1 * elements);
            REQUIRE(DataTypeStorageSizeBytes(DataType::Int8, elements) == 1 * elements);
            REQUIRE(DataTypeStorageSizeBytes(DataType::Int16, elements) == 2 * elements);
            REQUIRE(DataTypeStorageSizeBytes(DataType::Int32, elements) == 4 * elements);
            REQUIRE(DataTypeStorageSizeBytes(DataType::Int48, elements) == 6 * elements);
            REQUIRE(DataTypeStorageSizeBytes(DataType::Int64, elements) == 8 * elements);
            REQUIRE(DataTypeStorageSizeBytes(DataType::UInt8, elements) == 1 * elements);
            REQUIRE(DataTypeStorageSizeBytes(DataType::UInt16, elements) == 2 * elements);
            REQUIRE(DataTypeStorageSizeBytes(DataType::UInt32, elements) == 4 * elements);
            REQUIRE(DataTypeStorageSizeBytes(DataType::UInt48, elements) == 6 * elements);
            REQUIRE(DataTypeStorageSizeBytes(DataType::UInt64, elements) == 8 * elements);
        }
    }

    SECTION("Integer Min")
    {
        REQUIRE(IntegerMin(DataType::Int4) == -8);
        REQUIRE(IntegerMin(DataType::Int8) == int64_t(std::numeric_limits<int8_t>::min()));
        REQUIRE(IntegerMin(DataType::Int16) == int64_t(std::numeric_limits<int16_t>::min()));
        REQUIRE(IntegerMin(DataType::Int32) == int64_t(std::numeric_limits<int32_t>::min()));
        REQUIRE(IntegerMin(DataType::Int48) == -140737488355328LL);
        REQUIRE(IntegerMin(DataType::Int64) == std::numeric_limits<int64_t>::min());
        REQUIRE(IntegerMin(DataType::UInt8) == 0);
        REQUIRE(IntegerMin(DataType::UInt16) == 0);
        REQUIRE(IntegerMin(DataType::UInt32) == 0);
        REQUIRE(IntegerMin(DataType::UInt48) == 0);
        REQUIRE(IntegerMin(DataType::UInt64) == 0);
    }

    SECTION("Integer Max")
    {
        REQUIRE(IntegerMax(DataType::Int4) == 7);
        REQUIRE(IntegerMax(DataType::Int8) == std::numeric_limits<int8_t>::max());
        REQUIRE(IntegerMax(DataType::Int16) == std::numeric_limits<int16_t>::max());
        REQUIRE(IntegerMax(DataType::Int32) == std::numeric_limits<int32_t>::max());
        REQUIRE(IntegerMax(DataType::Int48) == 140737488355327ULL);
        REQUIRE(IntegerMax(DataType::Int64) == std::numeric_limits<int64_t>::max());
        REQUIRE(IntegerMax(DataType::UInt8) == std::numeric_limits<uint8_t>::max());
        REQUIRE(IntegerMax(DataType::UInt16) == std::numeric_limits<uint16_t>::max());
        REQUIRE(IntegerMax(DataType::UInt32) == std::numeric_limits<uint32_t>::max());
        REQUIRE(IntegerMax(DataType::UInt48) == 281474976710655ULL);
        REQUIRE(IntegerMax(DataType::UInt64) == std::numeric_limits<uint64_t>::max());
    }
}

TEST_CASE("DataType elements")
{
    const int kMax = DataTypeStorageSizeBytes(DataType::Int64, 2);
    for ( int bytes = 0; bytes <= kMax; ++bytes )
    {
        CAPTURE(bytes);
        REQUIRE(DataTypeElements(DataType::Int4Packed8, bytes) == bytes * 2);
        REQUIRE(DataTypeElements(DataType::Int4, bytes) == bytes);
        REQUIRE(DataTypeElements(DataType::Int8, bytes) == bytes);
        REQUIRE(DataTypeElements(DataType::Int16, bytes) == bytes / 2);
        REQUIRE(DataTypeElements(DataType::Int32, bytes) == bytes / 4);
        REQUIRE(DataTypeElements(DataType::Int48, bytes) == bytes / 6);
        REQUIRE(DataTypeElements(DataType::Int64, bytes) == bytes / 8);
        REQUIRE(DataTypeElements(DataType::UInt8, bytes) == bytes);
        REQUIRE(DataTypeElements(DataType::UInt16, bytes) == bytes / 2);
        REQUIRE(DataTypeElements(DataType::UInt32, bytes) == bytes / 4);
        REQUIRE(DataTypeElements(DataType::UInt48, bytes) == bytes / 6);
        REQUIRE(DataTypeElements(DataType::UInt64, bytes) == bytes / 8);
    }
}

TEST_CASE("int48_t")
{
    const int64_t val = 48217395205765;
    int48_t val48 = val;
    int64_t val64 = val48;
    REQUIRE(val64 == val);
    int48_t val48neg = -val;
    val64 = val48neg;
    REQUIRE(val64 == -val);
    int48_t array48[] = {val, -val};
    int48_t *p48 = &array48[0];
    REQUIRE(int64_t(p48[0]) == val);
    REQUIRE(int64_t(p48[1]) == -val);
}
