//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/shape.hpp"
#include "randomize.hpp"

#include <catch_all.hpp>
#include <random>
#include <vector>

static const int MAX_TEST_DIMS = 8;
static const int MAX_STATIC_DIMS = 4;

TEST_CASE("Static shape allocation")
{
    Shape a;
    Shape b(10);
    Shape c(10, 20);
    Shape d(10, 20, 30);
    Shape e(10, 20, 30, 40);

    SECTION("static lengths")
    {
        REQUIRE((a.Size() == 0 && !a.IsValid() && !a.IsDynamic()));  // Empty shape
        REQUIRE((b.Size() == 1 && b.IsValid() && !b.IsDynamic()));
        REQUIRE((c.Size() == 2 && c.IsValid() && !c.IsDynamic()));
        REQUIRE((d.Size() == 3 && d.IsValid() && !d.IsDynamic()));
        REQUIRE((e.Size() == 4 && d.IsValid() && !e.IsDynamic()));
    }

    SECTION("NHWC named indexing")
    {
        REQUIRE((b.Depth() == 10));
        REQUIRE((c.Depth() == 20 && c.Width() == 10));
        REQUIRE((d.Depth() == 30 && d.Width() == 20 && d.Height() == 10));
        REQUIRE((e.Depth() == 40 && e.Width() == 30 && e.Height() == 20 && e.Batch() == 10));
    }

    SECTION("NHWC Forward indexing")
    {
        REQUIRE((b[0] == 10));
        REQUIRE((c[1] == 20 && c[0] == 10));
        REQUIRE((d[2] == 30 && d[1] == 20 && d[0] == 10));
        REQUIRE((e[3] == 40 && e[2] == 30 && e[1] == 20 && e[0] == 10));
    }

    SECTION("NHWC Reverse indexing")
    {
        REQUIRE((b[-1] == 10));
        REQUIRE((c[-1] == 20 && c[-2] == 10));
        REQUIRE((d[-1] == 30 && d[-2] == 20 && d[-3] == 10));
        REQUIRE((e[-1] == 40 && e[-2] == 30 && e[-3] == 20 && e[-4] == 10));
    }

    SECTION("Shape Volume")
    {
        REQUIRE(b.Elements() == 10);
        REQUIRE(c.Elements() == 200);
        REQUIRE(d.Elements() == 6000);
        REQUIRE(e.Elements() == 240000);
    }
}

TEST_CASE("Dynamic shape allocation")
{
    int axes = GENERATE(range(4, MAX_TEST_DIMS));

    SECTION("Zero filled")
    {
        Shape a(nullptr, axes);
        REQUIRE(a.Size() == axes);
        REQUIRE(a.IsEmpty());
        REQUIRE((a.Size() > MAX_STATIC_DIMS) == a.IsDynamic());
        REQUIRE((a.Size() <= MAX_STATIC_DIMS) == !a.IsDynamic());
    }

    SECTION("Initialised fill")
    {
        std::vector<int> temp = random_vector<int>(axes, 1, 100);

        Shape b(temp.data(), int(temp.size()));
        REQUIRE(b.Size() == axes);
        int64_t volume = 1;
        for ( int i = 0; i < axes; i++ )
        {
            REQUIRE(b[i] == temp[i]);
            volume *= b[i];
        }
        REQUIRE(volume == b.Elements64());
    }
}

TEST_CASE("Operators (equal length)")
{
    int axes = GENERATE(range(1, MAX_TEST_DIMS));
    std::vector<int> atemp = random_vector<int>(axes, 1, 32767);
    std::vector<int> btemp = random_vector<int>(axes, 1, 32767);
    Shape a(atemp.data(), axes);
    Shape b(btemp.data(), axes);

    SECTION("Addition")
    {
        Shape c = a + b;
        for ( int i = 0; i < axes; i++ )
        {
            REQUIRE(c[i] == a[i] + b[i]);
        }
        // Idempotent subtraction
        c -= b;
        REQUIRE(c == a);
    }

    SECTION("Subtraction")
    {
        Shape c = a - b;
        for ( int i = 0; i < axes; i++ )
        {
            REQUIRE(c[i] == a[i] - b[i]);
        }
        // Idempotent addition
        c += b;
        REQUIRE(c == a);
    }

    SECTION("Modulo")
    {
        Shape c = a % b;
        for ( int i = 0; i < axes; i++ )
        {
            REQUIRE(c[i] == (a[i] % b[i]));
        }
    }

    SECTION("Divide")
    {
        Shape c = a / b;
        for ( int i = 0; i < axes; i++ )
        {
            REQUIRE(c[i] == (a[i] / b[i]));
        }
    }

    SECTION("Scale")
    {
        int s = GENERATE(take(2, random(1, 100)));
        Shape c = a * s;
        Shape d = a / s;
        for ( int i = 0; i < axes; i++ )
        {
            REQUIRE(c[i] == (a[i] * s));
            REQUIRE(d[i] == (a[i] / s));
        }
    }
}

TEST_CASE("Maximum (equal length)")
{
    int axes = GENERATE(range(1, MAX_TEST_DIMS, 2));
    std::vector<int> atemp = random_vector<int>(axes, 1, 32767);
    std::vector<int> btemp = random_vector<int>(axes, 1, 16384);
    Shape a(atemp.data(), axes);
    Shape b(btemp.data(), axes);

    Shape c = Shape::Max(a, b);
    for ( int i = 0; i < axes; i++ )
    {
        REQUIRE((c[i] >= a[i] && c[i] >= b[i]));
    }
}

TEST_CASE("Minimum (equal length)")
{
    int axes = GENERATE(range(1, MAX_TEST_DIMS, 2));
    std::vector<int> atemp = random_vector<int>(axes, 1, 32767);
    std::vector<int> btemp = random_vector<int>(axes, 1, 16384);
    Shape a(atemp.data(), axes);
    Shape b(btemp.data(), axes);

    Shape c = Shape::Min(a, b);
    for ( int i = 0; i < axes; i++ )
    {
        REQUIRE((c[i] <= a[i] && c[i] <= b[i]));
    }
}

TEST_CASE("RoundAway (equal length)")
{
    int axes = GENERATE(range(1, MAX_TEST_DIMS, 2));
    std::vector<int> atemp = random_vector<int>(axes, 1, 32767);
    std::vector<int> btemp = random_vector<int>(axes, 1, 256);
    Shape a(atemp.data(), axes);
    Shape b(btemp.data(), axes);

    Shape c = Shape::RoundAway(a, b);
    for ( int i = 0; i < axes; i++ )
    {
        REQUIRE(c[i] % b[i] == 0);
        REQUIRE(c[i] >= a[i]);
    }
}

TEST_CASE("RoundZero (equal length)")
{
    int axes = GENERATE(range(1, MAX_TEST_DIMS, 2));
    std::vector<int> atemp = random_vector<int>(axes, 1, 32767);
    std::vector<int> btemp = random_vector<int>(axes, 1, 256);
    Shape a(atemp.data(), axes);
    Shape b(btemp.data(), axes);

    Shape c = Shape::RoundZero(a, b);
    for ( int i = 0; i < axes; i++ )
    {
        REQUIRE(c[i] % b[i] == 0);
        REQUIRE(c[i] <= a[i]);
    }
}

TEST_CASE("DivRoundUp (equal length)")
{
    int axes = GENERATE(range(1, MAX_TEST_DIMS, 2));
    std::vector<int> atemp = random_vector<int>(axes, 1, 32767);
    std::vector<int> btemp = random_vector<int>(axes, 1, 256);
    Shape a(atemp.data(), axes);
    Shape b(btemp.data(), axes);

    Shape c = Shape::DivRoundUp(a, b);
    for ( int i = 0; i < axes; i++ )
    {
        int value = a[i] / b[i];
        REQUIRE((c[i] == value || c[i] == value + 1));
    }
}

TEST_CASE("Wrap (equal length)")
{
    int axes = GENERATE(range(1, MAX_TEST_DIMS, 2));
    std::vector<int> atemp = random_vector<int>(axes, 1, 32767);
    std::vector<int> btemp = random_vector<int>(axes, 1, 256);
    Shape a(atemp.data(), axes);
    Shape b(btemp.data(), axes);

    Shape c = Shape::Wrap(a, b);
    for ( int i = 0; i < axes; i++ )
    {
        int value = (a[i] > b[i]) ? (a[i] % b[i]) : a[i];
        REQUIRE(c[i] >= 0);
        REQUIRE(c[i] == value);
    }
}

TEST_CASE("Pad axes (pads left)")
{
    int axes = GENERATE(range(1, MAX_TEST_DIMS, 1));
    int padto = GENERATE(range(1, MAX_TEST_DIMS, 2));
    std::vector<int> atemp = random_vector<int>(axes, 1, 100);
    Shape a(atemp.data(), axes);
    Shape b = Shape::PadAxes(a, padto, 0);
    Shape c = Shape::PadAxes(a, padto, 12345);

    padto = std::max(b.Size(), padto);
    REQUIRE(b.Size() == padto);
    REQUIRE(c.Size() == padto);
    int diff = (padto - axes);
    for ( int i = 0; i < padto; i++ )
    {
        if ( (padto > axes) && i < diff )
        {
            // Padded part
            REQUIRE(b[i] == 0);
            REQUIRE(c[i] == 12345);
        }
        else
        {
            // Original part
            REQUIRE(b[i] == a[i - diff]);
            REQUIRE(c[i] == a[i - diff]);
        }
    }
}

TEST_CASE("AxisProduct")
{
    Shape a(2, 2, 3, 4);
    for ( int i = 0; i < a.Size(); i++ )
    {
        for ( int j = i + 1; j < a.Size(); j++ )
        {
            // compute product of axes from i to j
            int product = 1;
            for ( int x = i; x < j; x++ )
            {
                product *= a[x];
            }
            REQUIRE(a.AxisProduct(i, j) == product);
        }
        REQUIRE(a.AxisProduct(i, i) == 0);
    }
}

TEST_CASE("Copy, Move and Assignment")
{
    Shape invalid;
    Shape a(1, 2);
    Shape b(1, 2, 3);
    Shape c(1, 2, 3, 4);
    Shape d(b);
    REQUIRE(d == b);
    b = c;
    REQUIRE(b.Size() == 4);
    c = invalid;
    REQUIRE(!c.IsValid());
    b = std::move(a);
    REQUIRE(b.Size() == 2);
    REQUIRE(!a.IsValid());
}

TEST_CASE("Is Empty")
{
    int axes = GENERATE(range(1, MAX_TEST_DIMS, 2));
    std::vector<int> atemp = random_vector<int>(axes, 1, 32767);
    std::vector<int> btemp = random_vector<int>(axes, 0, 0);
    Shape a(atemp.data(), axes);
    Shape b(btemp.data(), axes);

    REQUIRE(!a.IsEmpty());
    REQUIRE(b.IsEmpty());
}

TEST_CASE("Extract axes")
{
    Shape a(100, 200, 300, 400);

    Shape aa = a.Extract(0, 1, 2, 3);
    REQUIRE(a == aa);

    Shape b = a.Extract({3});
    REQUIRE((b[0] == 400 && b.Size() == 1));

    Shape c = a.Extract({1, 2});
    REQUIRE((c[0] == 200 && c[1] == 300 && c.Size() == 2));

    Shape d = a.Extract({3, 2, 1, 0});
    REQUIRE((d[0] == 400 && d[1] == 300 && d[2] == 200 && d[3] == 100));

    Shape e = a.Extract({2, 2, 2});
    REQUIRE((e[0] == 300 && e[1] == 300 && e[2] == 300));
}

TEST_CASE("Erase and Insert axes")
{
    Shape a(100, 200, 300, 400);

    REQUIRE(a.Erase(0) == Shape(200, 300, 400));
    REQUIRE(a.Erase(1) == Shape(100, 300, 400));
    REQUIRE(a.Erase(2) == Shape(100, 200, 400));
    REQUIRE(a.Erase(3) == Shape(100, 200, 300));
    REQUIRE(a.Erase(-1) == Shape(100, 200, 300));
    REQUIRE(a.Erase(-2) == Shape(100, 200, 400));
    REQUIRE(a.Erase(-3) == Shape(100, 300, 400));
    REQUIRE(a.Erase(-4) == Shape(200, 300, 400));

    Shape b(100, 200);
    REQUIRE(b.Insert(0, 50) == Shape(50, 100, 200));
    REQUIRE(b.Insert(1, 50) == Shape(100, 50, 200));
    REQUIRE(b.Insert(2, 50) == Shape(100, 200, 50));
    REQUIRE(b.Insert(-1, 50) == Shape(100, 200, 50));
    REQUIRE(b.Insert(-2, 50) == Shape(100, 50, 200));
    REQUIRE(b.Insert(-3, 50) == Shape(50, 100, 200));
}

TEST_CASE("Permute Mask")
{
    Shape a(100, 200, 300, 400);

    SECTION("identity")
    {
        Shape b = a.Permute(0x3210);
        REQUIRE(a == b);
    }

    Shape c = a.Permute(0x0123);
    REQUIRE((c[0] == a[3] && c[1] == a[2] && c[2] == a[1] && c[3] == a[0]));

    Shape d = a.Permute(0x3021);
    REQUIRE((d[0] == 100 && d[1] == 400 && d[2] == 200 && d[3] == 300));
}

TEST_CASE("Unpermute Mask")
{
    Shape a(100, 200, 300, 400);

    SECTION("identity")
    {
        Shape b = a.Unpermute(0x3210);
        REQUIRE(a == b);
    }

    Shape c = a.Unpermute(0x0321);
    REQUIRE((c[0] == a[1] && c[1] == a[2] && c[2] == a[3] && c[3] == a[0]));

    Shape d = a.Permute(0x0321).Unpermute(0x0321);
    REQUIRE(a == d);

    Shape e = a.Unpermute(0x2103);
    REQUIRE((e[0] == 400 && e[1] == 100 && e[2] == 200 && e[3] == 300));

    Shape f = a.Permute(0x2103).Unpermute(0x2103);
    REQUIRE(a == f);
}

TEST_CASE("Get Strides")
{
    Shape a(1, 2, 3, 4);

    Shape b = Shape::GetStridesForShape(a, {1, 1, 1, 1});
    REQUIRE((b[0] == 24 && b[1] == 12 && b[2] == 4 && b[3] == 1));

    Shape c = Shape::GetStridesForShape(a, {1, 1, 1, 4});
    REQUIRE((c[0] == 96 && c[1] == 48 && c[2] == 16 && c[3] == 4));

    Shape d = Shape::GetStridesForShape(a, {128, 1, 1, 4});
    REQUIRE((d[0] == 128 && d[1] == 48 && d[2] == 16 && d[3] == 4));
}

TEST_CASE("Vector Conversion")
{
    std::vector<int> intVec{5, 6, 2, 1};
    Shape a = Shape::FromVector(intVec);
    REQUIRE((a[0] == 5 && a[2] == 2));

    std::vector<uint8_t> charVec{45, 32, 25, 12, 16, 19};
    Shape b = Shape::FromVector(charVec);
    REQUIRE((b[0] == 45 && b[4] == 16 && b[5] == 19));
}

TEST_CASE("To Mask")
{
    Shape shape1(0, 1, 2, 3);
    uint32_t mask1 = shape1.ToMask();
    REQUIRE((mask1 == 0x3210));

    Shape shape2(1, 0, 3, 2);
    uint32_t mask2 = shape2.ToMask();
    REQUIRE((mask2 == 0x2301));
}

TEST_CASE("Is reduced equal")
{
    Shape shape1a(3, 3);
    Shape shape1b(1, 1, 3, 3);
    REQUIRE(Shape::IsReducedEqual(shape1a, shape1b));
    REQUIRE(Shape::IsReducedEqual(shape1b, shape1a));

    Shape shape3a(3, 3);
    Shape shape3b(3, 3, 1);
    REQUIRE_FALSE(Shape::IsReducedEqual(shape3a, shape3b));
    REQUIRE_FALSE(Shape::IsReducedEqual(shape3b, shape3a));

    Shape shape2a(1);
    Shape shape2b(1, 1, 1);
    REQUIRE(Shape::IsReducedEqual(shape2a, shape2b));
    REQUIRE(Shape::IsReducedEqual(shape2b, shape2a));
}
