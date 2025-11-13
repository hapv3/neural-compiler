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

#include "common/scaling.hpp"

#include <catch_all.hpp>

TEST_CASE("Test quantization")
{
    SECTION("Comparison operators")
    {
        QuantizedScale a = QuantizedScale::Unit();
        QuantizedScale b = {1, 1};
        QuantizedScale c = QuantizedScale::Unit();
        // Equality checks
        REQUIRE_FALSE(a == b);
        REQUIRE(a == c);
        // Not equals
        REQUIRE(a != b);
        REQUIRE_FALSE(a != c);
        // Strictly less than
        REQUIRE(b < a);
        REQUIRE_FALSE(c < a);
        // Less or equal
        REQUIRE(c <= a);
        REQUIRE_FALSE(a <= b);
    }

    SECTION("Find largest scale factor in list of scales")
    {
        std::vector<QuantizedScale> scales = {
            {1, 30},           // 2^(-30)
            {1073741824, 31},  // 0.5
            {8, 5},            // 0.25
            {1073741824, 30},  // 1.0
            {3, 2},            // 0.75
        };

        auto itr = std::max_element(std::begin(scales), std::end(scales));
        REQUIRE(itr != std::end(scales));
        REQUIRE(*itr == QuantizedScale(1073741824, 30));
    }
}
