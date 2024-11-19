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

#include "catch2/catch_all.hpp"
#include "common/buffer_view.hpp"
#include "common/shape.hpp"
#include "randomize.hpp"

#include <memory>
#include <random>
#include <vector>

using namespace regor;

static const std::vector<int8_t> dataInt8{1, 2, 3, 1, 2, 3, -1, -2, -3, 1, 2, 3, 1, 2, 3, -1, -2, -3, 1, 2, 3, 1, 2, 3, -1, -2, -3};
static const std::vector<int> dataInt32{-1000, -900, -800, -700, -600, -500, -400, -300, -200, -100, 0, 0, 100, 200,
    300, 400, 500, 600, 700, 800, 900, 1000};

TEST_CASE("Linear buffer read/write")
{
    auto buffer8 = std::make_shared<Buffer>(std::vector<int8_t>(dataInt8));
    auto buffer32 = std::make_shared<Buffer>(std::vector<int32_t>(dataInt32));

    BufferView view_int8_linear(buffer8, 0, 8, Shape(1, int(dataInt8.size())), Shape());
    BufferView view_int32_linear(buffer32, 0, 32, Shape(1, int(dataInt32.size())), Shape());

    SECTION("Total elements")
    {
        REQUIRE(buffer8->Size() == view_int8_linear.Elements());
        REQUIRE(buffer32->Size() / 4 == view_int32_linear.Elements());
    }

    SECTION("Compare elements during indexed iteration")
    {
        size_t count = 0;
        auto values = view_int8_linear.Values<int8_t>();
        for ( int i = 0; i < view_int8_linear.Elements(); i++ )
        {
            REQUIRE(values[i] == dataInt8[i]);
        }
    }

    SECTION("Compare elements during iterator iteration")
    {
        size_t count = 0;
        auto values = view_int8_linear.Values<int8_t>();
        auto viewpos = values.begin();
        for ( auto pos = dataInt8.begin(); pos != dataInt8.end(); pos++, viewpos++ )
        {
            REQUIRE(*pos == *viewpos);
            count++;
        }
        REQUIRE(count == dataInt8.size());
    }

    SECTION("Compare elements during iterator iteration with type translation")
    {
        size_t count = 0;
        auto values = view_int32_linear.Values<int32_t, int64_t>();
        auto viewpos = values.begin();
        for ( auto pos = dataInt32.begin(); pos != dataInt32.end(); pos++, viewpos++ )
        {
            REQUIRE(int64_t(*pos) == *viewpos);
            count++;
        }
        REQUIRE(count == dataInt32.size());
    }

    SECTION("Polymorphic indexed access with same reader")
    {
        BufferReader<float> values;

        for ( int i = 0; i <= 1; i++ )
        {
            values = (i == 0) ? view_int8_linear.Values<int8_t, float>() : view_int32_linear.Values<int, float>();

            if ( i == 0 )
            {
                REQUIRE(values[0] == 1);
                REQUIRE(values[1] == 2);
                REQUIRE(values[6] == -1);
                REQUIRE(values[8] == -3);
                REQUIRE(values[25] == -2);
                REQUIRE(values[26] == -3);
            }
            else if ( i == 1 )
            {
                REQUIRE(values[0] == -1000);
                REQUIRE(values[10] == 0);
                REQUIRE(values[21] == 1000);
            }
        }
    }
}

TEST_CASE("Nonlinear buffer")
{
    auto buffer8 = std::make_shared<Buffer>(std::vector<int8_t>(dataInt8));
    auto buffer32 = std::make_shared<Buffer>(std::vector<int32_t>(dataInt32));

    BufferView view_int8(buffer8, 0, 8, Shape(1, 3, 3, 3), Shape());
    BufferView view_int32(buffer32, 0, 32, Shape(1, 2, int(dataInt32.size()) / 2), Shape());

    SECTION("Total elements")
    {
        REQUIRE(buffer8->Size() == view_int8.Elements());
        REQUIRE(buffer32->Size() / 4 == view_int32.Elements());
    }

    SECTION("Shape Coordinate reads")
    {
        BufferReader<double> values = view_int8.Values<int8_t, double>();
        REQUIRE(values[{0, 0, 0}] == 1);
        REQUIRE(values[{0, 2, 0}] == -1);
        REQUIRE(values[{0, 0, 2}] == 3);
        REQUIRE(values[{2, 2, 2}] == -3);

        values = view_int32.Values<int, double>();
        REQUIRE(values[{0, 4}] == -600);
        REQUIRE(values[{1, 4}] == 400);
    }

    SECTION("Shape Coordinate write/read")
    {
        BufferReader<int> read = view_int32.Values<int>();
        BufferWriter<int> write = view_int32.WritableValues<int>();

        write[{1, 3}] = 1234;
        // Equivalent indexed reads
        REQUIRE((read[{1, 3}]) == 1234);
        REQUIRE((read[Shape{14}]) == 1234);
    }
}
