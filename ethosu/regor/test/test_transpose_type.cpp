//
// SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include <catch_all.hpp>

TEST_CASE("TransposeType IsNone")
{
    using namespace regor;

    REQUIRE(IsNone(TransposeType::None));
    REQUIRE(IsNone(TransposeType(0x76543210)));

    REQUIRE_FALSE(IsNone(TransposeType::NWHC));
    REQUIRE_FALSE(IsNone(TransposeType::NHCW));
    REQUIRE_FALSE(IsNone(TransposeType::NWCH));
    REQUIRE_FALSE(IsNone(TransposeType::NCHW));
    REQUIRE_FALSE(IsNone(TransposeType::NCWH));
}
