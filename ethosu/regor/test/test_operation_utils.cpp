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

TEST_CASE("ReshapeTo3D")
{
    Shape shape4D(3, 7, 11, 13);
    auto shape4Da = ReshapeTo3D(shape4D, {2, 0, 2});
    REQUIRE(shape4Da == Shape(3 * 7, 1, 11 * 13));
    auto shape4Db = ReshapeTo3D(shape4D, {2, 0, 2}, 0);
    REQUIRE(shape4Db == Shape(3 * 7, 0, 11 * 13));

    Shape shape3D(3, 7, 11);
    auto shape3Da = ReshapeTo3D(shape3D, {0, 1, 2});
    REQUIRE(shape3Da == Shape(1, 3, 7 * 11));
    auto shape3Db = ReshapeTo3D(shape3D, {0, 1, 2}, 0);
    REQUIRE(shape3Db == Shape(0, 3, 7 * 11));

    Shape shape2D(3, 7);
    auto shape2Da = ReshapeTo3D(shape2D, {1, 1, 0});
    REQUIRE(shape2Da == Shape(3, 7, 1));
    auto shape2Db = ReshapeTo3D(shape2D, {1, 1, 0}, 0);
    REQUIRE(shape2Db == Shape(3, 7, 0));

    Shape shape1D(3);
    auto shape1Da = ReshapeTo3D(shape1D, {0, 1, 0});
    REQUIRE(shape1Da == Shape(1, 3, 1));
    auto shape1Db = ReshapeTo3D(shape1D, {0, 1, 0}, 0);
    REQUIRE(shape1Db == Shape(0, 3, 0));
}

TEST_CASE("ReshapeTo3DAroundAxis")
{
    Shape shape4D(3, 7, 11, 13);
    auto shape4Da = ReshapeTo3DAroundAxis(shape4D, 0);
    REQUIRE(shape4Da == Shape(1, 3, 7 * 11 * 13));
    auto shape4Da0 = ReshapeTo3DAroundAxis(shape4D, 0, 0);
    REQUIRE(shape4Da0 == Shape(0, 3, 7 * 11 * 13));
    auto shape4Db = ReshapeTo3DAroundAxis(shape4D, 1);
    REQUIRE(shape4Db == Shape(3, 7, 11 * 13));
    auto shape4Db0 = ReshapeTo3DAroundAxis(shape4D, 1, 0);
    REQUIRE(shape4Db0 == Shape(3, 7, 11 * 13));
    auto shape4Dc = ReshapeTo3DAroundAxis(shape4D, 2);
    REQUIRE(shape4Dc == Shape(3 * 7, 11, 13));
    auto shape4Dc0 = ReshapeTo3DAroundAxis(shape4D, 2, 0);
    REQUIRE(shape4Dc0 == Shape(3 * 7, 11, 13));
    auto shape4Dd = ReshapeTo3DAroundAxis(shape4D, 3);
    REQUIRE(shape4Dd == Shape(3 * 7 * 11, 13, 1));
    auto shape4Dd0 = ReshapeTo3DAroundAxis(shape4D, 3, 0);
    REQUIRE(shape4Dd0 == Shape(3 * 7 * 11, 13, 0));

    Shape shape3D(3, 7, 11);
    auto shape3Da = ReshapeTo3DAroundAxis(shape3D, 0);
    REQUIRE(shape3Da == Shape(1, 3, 7 * 11));
    auto shape3Da0 = ReshapeTo3DAroundAxis(shape3D, 0, 0);
    REQUIRE(shape3Da0 == Shape(0, 3, 7 * 11));
    auto shape3Db = ReshapeTo3DAroundAxis(shape3D, 1);
    REQUIRE(shape3Db == Shape(3, 7, 11));
    auto shape3Db0 = ReshapeTo3DAroundAxis(shape3D, 1, 0);
    REQUIRE(shape3Db0 == Shape(3, 7, 11));
    auto shape3Dc = ReshapeTo3DAroundAxis(shape3D, 2);
    REQUIRE(shape3Dc == Shape(3 * 7, 11, 1));
    auto shape3Dc0 = ReshapeTo3DAroundAxis(shape3D, 2, 0);
    REQUIRE(shape3Dc0 == Shape(3 * 7, 11, 0));

    Shape shape2D(3, 7);
    auto shape2Da = ReshapeTo3DAroundAxis(shape2D, 0);
    REQUIRE(shape2Da == Shape(1, 3, 7));
    auto shape2Da0 = ReshapeTo3DAroundAxis(shape2D, 0, 0);
    REQUIRE(shape2Da0 == Shape(0, 3, 7));
    auto shape2Db = ReshapeTo3DAroundAxis(shape2D, 1);
    REQUIRE(shape2Db == Shape(3, 7, 1));
    auto shape2Db0 = ReshapeTo3DAroundAxis(shape2D, 1, 0);
    REQUIRE(shape2Db0 == Shape(3, 7, 0));

    Shape shape1D(3);
    auto shape1Da = ReshapeTo3DAroundAxis(shape1D, 0);
    REQUIRE(shape1Da == Shape(1, 3, 1));
    auto shape1Da0 = ReshapeTo3DAroundAxis(shape1D, 0, 0);
    REQUIRE(shape1Da0 == Shape(0, 3, 0));
}

TEST_CASE("ReshapeTo3DAroundEdges")
{
    Shape shape4D(3, 7, 11, 13);
    auto shape4Da = ReshapeTo3DAroundEdges(shape4D);
    REQUIRE(shape4Da == Shape(3, 7 * 11, 13));
    auto shape4Da0 = ReshapeTo3DAroundEdges(shape4D, 0);
    REQUIRE(shape4Da0 == Shape(3, 7 * 11, 13));

    Shape shape3D(3, 7, 11);
    auto shape3Da = ReshapeTo3DAroundEdges(shape3D);
    REQUIRE(shape3Da == Shape(3, 7, 11));
    auto shape3Da0 = ReshapeTo3DAroundEdges(shape3D, 0);
    REQUIRE(shape3Da0 == Shape(3, 7, 11));

    Shape shape2D(3, 7);
    auto shape2Da = ReshapeTo3DAroundEdges(shape2D);
    REQUIRE(shape2Da == Shape(3, 1, 7));
    auto shape2Da0 = ReshapeTo3DAroundEdges(shape2D, 0);
    REQUIRE(shape2Da0 == Shape(3, 0, 7));
}
