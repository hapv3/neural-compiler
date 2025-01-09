//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/reverse_type.hpp"

#include <catch_all.hpp>

TEST_CASE("TransposeType ToReverseMask")
{
    using namespace regor;

    Flags<ReverseType> C = ReverseType::C;
    Flags<ReverseType> W = ReverseType::W;
    Flags<ReverseType> H = ReverseType::H;
    Flags<ReverseType> N = ReverseType::N;
    Flags<ReverseType> B = ReverseType::B;
    Flags<ReverseType> A = ReverseType::A;

    SECTION("ofmRank 1")
    {
        int ofmRank = 1;
        REQUIRE(ToReverseMask(Shape(0), ofmRank) == C);
    }
    SECTION("ofmRank 2")
    {
        int ofmRank = 2;
        REQUIRE(ToReverseMask(Shape(0), ofmRank) == W);
        REQUIRE(ToReverseMask(Shape(1), ofmRank) == C);
        REQUIRE(ToReverseMask(Shape(0, 1), ofmRank) == Flags<ReverseType>(W | C));
    }
    SECTION("ofmRank 3")
    {
        int ofmRank = 3;
        REQUIRE(ToReverseMask(Shape(0), ofmRank) == H);
        REQUIRE(ToReverseMask(Shape(1), ofmRank) == W);
        REQUIRE(ToReverseMask(Shape(2), ofmRank) == C);
        REQUIRE(ToReverseMask(Shape(0, 2), ofmRank) == Flags<ReverseType>(H | C));
        REQUIRE(ToReverseMask(Shape(1, 2), ofmRank) == Flags<ReverseType>(W | C));
        REQUIRE(ToReverseMask(Shape(0, 1, 2), ofmRank) == Flags<ReverseType>(H | W | C));
    }
    SECTION("ofmRank 4")
    {
        int ofmRank = 4;
        REQUIRE(ToReverseMask(Shape(0), ofmRank) == N);
        REQUIRE(ToReverseMask(Shape(1), ofmRank) == H);
        REQUIRE(ToReverseMask(Shape(2), ofmRank) == W);
        REQUIRE(ToReverseMask(Shape(3), ofmRank) == C);
        REQUIRE(ToReverseMask(Shape(0, 3), ofmRank) == Flags<ReverseType>(N | C));
        REQUIRE(ToReverseMask(Shape(1, 2), ofmRank) == Flags<ReverseType>(H | W));
        REQUIRE(ToReverseMask(Shape(0, 1, 2, 3), ofmRank) == Flags<ReverseType>(N | H | W | C));
    }
    SECTION("ofmRank 5")
    {
        int ofmRank = 5;
        REQUIRE(ToReverseMask(Shape(0), ofmRank) == B);
        REQUIRE(ToReverseMask(Shape(1), ofmRank) == N);
        REQUIRE(ToReverseMask(Shape(2), ofmRank) == H);
        REQUIRE(ToReverseMask(Shape(3), ofmRank) == W);
        REQUIRE(ToReverseMask(Shape(4), ofmRank) == C);
        REQUIRE(ToReverseMask(Shape(0, 3), ofmRank) == Flags<ReverseType>(B | W));
        REQUIRE(ToReverseMask(Shape(1, 2), ofmRank) == Flags<ReverseType>(N | H));
        REQUIRE(ToReverseMask(Shape(0, 1, 2, 3), ofmRank) == Flags<ReverseType>(B | N | H | W));
    }
    SECTION("ofmRank 6")
    {
        int32_t elements[6] = {0, 1, 2, 3, 4, 5};
        int ofmRank = 6;
        REQUIRE(ToReverseMask(Shape(0), ofmRank) == A);
        REQUIRE(ToReverseMask(Shape(1), ofmRank) == B);
        REQUIRE(ToReverseMask(Shape(2), ofmRank) == N);
        REQUIRE(ToReverseMask(Shape(3), ofmRank) == H);
        REQUIRE(ToReverseMask(Shape(4), ofmRank) == W);
        REQUIRE(ToReverseMask(Shape(5), ofmRank) == C);
        REQUIRE(ToReverseMask(Shape(5, 3), ofmRank) == Flags<ReverseType>(H | C));
        REQUIRE(ToReverseMask(Shape(0, 1), ofmRank) == Flags<ReverseType>(A | B));
        REQUIRE(ToReverseMask(Shape(elements, 6), ofmRank) == Flags<ReverseType>(A | B | N | H | W | C));
    }
}
