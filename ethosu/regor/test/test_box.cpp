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

#include "common/box.hpp"

#include <catch_all.hpp>

static std::ostream &operator<<(std::ostream &os, const Shape &item)
{
    os << item.ToString();
    return os;
}

static std::ostream &operator<<(std::ostream &os, const Box &item)
{
    os << item.ToString();
    return os;
}

TEST_CASE("Box: tests")
{
    Box a({0, 0}, {3, 5});
    Box b({0, 0}, {2, 2});

    SECTION("Construct")
    {
        REQUIRE(a.Start() == Shape{0, 0});
        REQUIRE(a.SizeShape() == Shape{3, 5});
        REQUIRE(b.Start() == Shape{0, 0});
        REQUIRE(b.SizeShape() == Shape{2, 2});
    }

    SECTION("Move")
    {
        a.Move(Shape{-1, 1});
        REQUIRE(a.Start() == Shape{-1, 1});
        REQUIRE(a.SizeShape() == Shape{3, 5});
        a.Move(Shape{1, 0});
        REQUIRE(a.Start() == Shape{0, 1});
        REQUIRE(a.SizeShape() == Shape{3, 5});
    }

    SECTION("MoveTo")
    {
        a.MoveTo(Shape{5, 6});
        REQUIRE(a.Start() == Shape{5, 6});
        REQUIRE(a.SizeShape() == Shape{3, 5});
        a.MoveTo(Shape{-3, 7});
        REQUIRE(a.Start() == Shape{-3, 7});
        REQUIRE(a.SizeShape() == Shape{3, 5});
        a.MoveTo(Shape{0, 0});
        REQUIRE(a.Start() == Shape{0, 0});
        REQUIRE(a.SizeShape() == Shape{3, 5});
    }

    SECTION("Overlaps")
    {
        REQUIRE(a.Overlaps(b));
        b.MoveTo({3, 5});
        REQUIRE(!a.Overlaps(b));
        b.Move({-1, -1});
        REQUIRE(a.Overlaps(b));
        b.MoveTo({-2, -2});
        REQUIRE(!a.Overlaps(b));
        b.Move({1, 1});
        REQUIRE(a.Overlaps(b));
    }

    SECTION("Intersection")
    {
        REQUIRE(a.Intersection(b) == Box({0, 0}, Box::Size({2, 2})));
        b.MoveTo({3, 5});
        REQUIRE(a.Intersection(b) == Box{});
        b.Move({-1, -1});
        REQUIRE(a.Intersection(b) == Box({2, 4}, Box::Size({1, 1})));
        b.MoveTo({-1, -1});
        REQUIRE(a.Intersection(b) == Box({0, 0}, Box::Size({1, 1})));
        b.Move({2, 2});
        REQUIRE(a.Intersection(b) == Box({1, 1}, Box::Size({2, 2})));
    }
}
