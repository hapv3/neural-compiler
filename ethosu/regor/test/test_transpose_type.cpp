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

#include "common/transpose_type.hpp"
#include "randomize.hpp"

#include <catch_all.hpp>
#include <random>
#include <vector>

TEST_CASE("TransposeType IsNone")
{
    REQUIRE(IsNone(TransposeType::None));
    REQUIRE(IsNone(TransposeType::NHWC));

    REQUIRE(IsNone(TransposeType(0x0)));
    REQUIRE(IsNone(TransposeType(0x01)));
    REQUIRE(IsNone(TransposeType(0x012)));
    REQUIRE(IsNone(TransposeType(0x0123)));
    REQUIRE(IsNone(TransposeType(0x01234)));
    REQUIRE(IsNone(TransposeType(0x012345)));
    REQUIRE(IsNone(TransposeType(0x0123456)));
    REQUIRE(IsNone(TransposeType(0x01234567)));

    REQUIRE_FALSE(IsNone(TransposeType::NWHC));
    REQUIRE_FALSE(IsNone(TransposeType::NHCW));
    REQUIRE_FALSE(IsNone(TransposeType::NWCH));
    REQUIRE_FALSE(IsNone(TransposeType::NCHW));
    REQUIRE_FALSE(IsNone(TransposeType::NCWH));
}

TEST_CASE("TransposeType Reduce4To3")
{
    REQUIRE(uint32_t(Reduce4To3(TransposeType::None)) == 0x012);
    REQUIRE(uint32_t(Reduce4To3(TransposeType::NHWC)) == 0x012);
    REQUIRE(uint32_t(Reduce4To3(TransposeType::NWHC)) == 0x102);
    REQUIRE(uint32_t(Reduce4To3(TransposeType::NHCW)) == 0x021);
    REQUIRE(uint32_t(Reduce4To3(TransposeType::NWCH)) == 0x120);
    REQUIRE(uint32_t(Reduce4To3(TransposeType::NCHW)) == 0x201);
    REQUIRE(uint32_t(Reduce4To3(TransposeType::NCWH)) == 0x210);
}
