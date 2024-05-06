//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/common.hpp"

#include "common/hash.hpp"

#include <catch_all.hpp>

using namespace regor;

static std::string hashToHexString(Hash128 &hash)
{
    std::string hex;
    for ( int i = 0; i < hash.Size(); i++ )
    {
        const auto byte = hash.Buffer()[i];
        const auto msb = (byte >> 4) & 0xF;
        hex += msb < 10 ? ('0' + msb) : ('a' + (msb - 10));
        const auto lsb = (byte >> 0) & 0xF;
        hex += lsb < 10 ? ('0' + lsb) : ('a' + (lsb - 10));
    }
    return hex;
}

TEST_CASE("MD5 hash")
{
    MD5 md5;
    Hash128 hash;

    SECTION("Empty")
    {
        md5.Combine(nullptr, 0);
        md5.Get(hash);
        REQUIRE(hashToHexString(hash) == "d41d8cd98f00b204e9800998ecf8427e");
    }

    SECTION("Multiple")
    {
        const auto text1 = "abc";
        const auto text2 = "def";
        const auto text3 = "ghi";
        md5.Combine(reinterpret_cast<const uint8_t *>(text1), 3);
        md5.Combine(reinterpret_cast<const uint8_t *>(text2), 3);
        md5.Combine(reinterpret_cast<const uint8_t *>(text3), 3);
        md5.Get(hash);
        REQUIRE(hashToHexString(hash) == "8aa99b1f439ff71293e95357bac6fd94");
    }

    SECTION("63 bytes")
    {
        const auto text = "000000000011111111112222222222333333333344444444445555555555666";
        md5.Combine(reinterpret_cast<const uint8_t *>(text), 63);
        md5.Get(hash);
        REQUIRE(hashToHexString(hash) == "e099155a34541abba0cb61471b294308");
    }

    SECTION("64 bytes")
    {
        const auto text = "0000000000111111111122222222223333333333444444444455555555556666";
        md5.Combine(reinterpret_cast<const uint8_t *>(text), 64);
        md5.Get(hash);
        REQUIRE(hashToHexString(hash) == "6aa6f5f2002cefe90290ba1ebe79a3f5");
    }

    SECTION("65 bytes")
    {
        const auto text = "00000000001111111111222222222233333333334444444444555555555566666";
        md5.Combine(reinterpret_cast<const uint8_t *>(text), 65);
        md5.Get(hash);
        REQUIRE(hashToHexString(hash) == "b3c0a5ca7d73c8a33711f82cf37d72d6");
    }

    SECTION("Text 1")
    {
        const auto text = "The quick brown fox jumps over the lazy dog";
        md5.Combine(reinterpret_cast<const uint8_t *>(text), 43);
        md5.Get(hash);
        REQUIRE(hashToHexString(hash) == "9e107d9d372bb6826bd81d3542a419d6");
    }

    SECTION("Text 2")
    {
        const auto text = "The quick brown fox jumps over the lazy dog.";
        md5.Combine(reinterpret_cast<const uint8_t *>(text), 44);
        md5.Get(hash);
        REQUIRE(hashToHexString(hash) == "e4d909c290d0fb1ca068ffaddf22cbd0");
    }
}
