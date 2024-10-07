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

#include "compiler/operation_util.hpp"

#include <catch_all.hpp>

using namespace regor;

TEST_CASE("TransposeTypeFromShape")
{
    // 4D identity
    Shape shape1(0, 1, 2, 3);
    auto mask1 = TransposeTypeFromShape(shape1);
    REQUIRE(mask1 == TransposeType::None);

    // 3D WHC
    Shape shape2(1, 0, 2);
    auto mask2 = TransposeTypeFromShape(shape2);
    REQUIRE(mask2 == TransposeType::NWHC);

    // 2D CW
    Shape shape3(1, 0);
    auto mask3 = TransposeTypeFromShape(shape3);
    REQUIRE(mask3 == TransposeType::NHCW);

    // 6D
    int axes4[6] = {0, 4, 5, 3, 1, 2};
    Shape shape4(axes4, 6);
    auto mask4 = TransposeTypeFromShape(shape4);
    REQUIRE(uint32_t(mask4) == 0x76510243);
}
