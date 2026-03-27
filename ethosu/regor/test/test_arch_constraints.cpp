//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "architecture/ethosu55/ethos_u55.hpp"
#include "architecture/ethosu85/ethos_u85.hpp"
#include "compiler/quantization.hpp"
#include "util.hpp"

#include <catch_all.hpp>

using namespace regor;

namespace
{
Quantization MakeQuant(int32_t scale = 1, int shift = 0, int64_t zeroPoint = 0)
{
    Quantization q;
    q.type = QuantizationType::EXPLICIT;
    q.scales.emplace_back(scale, shift);
    q.zeroPoints.emplace_back(zeroPoint);
    return q;
}

Quantization MakePerChannel(int channels, int32_t scale = 1, int shift = 0, int64_t zeroPoint = 0)
{
    Quantization q;
    q.type = QuantizationType::EXPLICIT;
    q.scales.reserve(channels);
    q.zeroPoints.reserve(channels);
    for ( int i = 0; i < channels; ++i )
    {
        q.scales.emplace_back(scale, shift);
        q.zeroPoints.emplace_back(zeroPoint);
    }
    return q;
}
}  // namespace

TEST_CASE("Ethos-U55 SupportsQuantization handles IFM-fusing elementwise constraints")
{
    auto arch = CreateArchDefault<ArchEthosU55>();
    IArchitectureConstraints *constraints = arch->Constraints();

    const Quantization unit = MakeQuant();

    SECTION("accepts Add with unit scaling")
    {
        REQUIRE(constraints->SupportsQuantization(OpType::Add, unit, DataType::Int8, unit, DataType::Int8, unit, DataType::Int8));
    }
    SECTION("accepts Add with 16-bit IFM scaling and 0 shift")
    {
        const Quantization quant16BitMax = MakeQuant(std::numeric_limits<int16_t>::max(), 0, 0);
        REQUIRE(constraints->SupportsQuantization(
            OpType::Add, quant16BitMax, DataType::Int8, quant16BitMax, DataType::Int8, unit, DataType::Int8));
    }
    SECTION("rejects Add when IFM uses per-channel scales")
    {
        const Quantization perChannel = MakePerChannel(2);
        REQUIRE_FALSE(constraints->SupportsQuantization(
            OpType::Add, perChannel, DataType::Int8, unit, DataType::Int8, unit, DataType::Int8));
    }
    SECTION("rejects Mul with non-unit IFM-scaling")
    {
        const Quantization quant = MakeQuant(2, 1, 0);
        REQUIRE_FALSE(constraints->SupportsQuantization(OpType::Mul, quant, DataType::Int8, quant, DataType::Int8, unit, DataType::Int8));
    }
}

TEST_CASE("Ethos-U85 SupportsQuantization handles IFM-fusing elementwise constraints")
{
    auto arch = CreateArchDefault<ArchEthosU85>();
    IArchitectureConstraints *constraints = arch->Constraints();

    const Quantization unit = MakeQuant();

    SECTION("accepts Mul with unit scaling")
    {
        REQUIRE(constraints->SupportsQuantization(OpType::Mul, unit, DataType::Int8, unit, DataType::Int8, unit, DataType::Int8));
    }
    SECTION("rejects Mul and Div with non-unit IFM-scaling")
    {
        const Quantization quant16BitMax = MakeQuant(2, 0, 0);
        REQUIRE_FALSE(constraints->SupportsQuantization(
            OpType::Mul, quant16BitMax, DataType::Int8, quant16BitMax, DataType::Int8, unit, DataType::Int8));
        REQUIRE_FALSE(constraints->SupportsQuantization(
            OpType::Div, quant16BitMax, DataType::Int8, quant16BitMax, DataType::Int8, unit, DataType::Int8));
    }
    SECTION("rejects Add when IFM uses per-channel scales")
    {
        const Quantization perChannel = MakePerChannel(2);
        REQUIRE_FALSE(constraints->SupportsQuantization(
            OpType::Add, perChannel, DataType::Int8, unit, DataType::Int8, unit, DataType::Int8));
    }
    SECTION("rejects 32-bit Add with non-unit IFM scaling")
    {
        const Quantization quant = MakeQuant(2, 1, 0);
        REQUIRE_FALSE(constraints->SupportsQuantization(OpType::Add, quant, DataType::Int32, quant, DataType::Int32, unit, DataType::Int8));
    }
    SECTION("rejects UInt16 Add with non-unit IFM scaling")
    {
        const Quantization quant = MakeQuant(2, 1, 0);
        REQUIRE_FALSE(constraints->SupportsQuantization(
            OpType::Add, quant, DataType::UInt16, quant, DataType::UInt16, unit, DataType::Int8));
    }
    SECTION("accepts Int16 Add with non-unit IFM scaling")
    {
        const Quantization quant = MakeQuant(2, 1, 0);
        REQUIRE(constraints->SupportsQuantization(OpType::Add, quant, DataType::Int16, quant, DataType::Int16, unit, DataType::Int8));
    }
}

TEST_CASE("Ethos-U55 SupportsQuantization enforces zero-point rules")
{
    auto arch = CreateArchDefault<ArchEthosU55>();
    IArchitectureConstraints *constraints = arch->Constraints();
    const Quantization unit = MakeQuant();
    Quantization unitQuantZp1 = MakeQuant(1, 0, 1);

    SECTION("rejects CLZ and SHL when OFM zp != 0")
    {
        REQUIRE_FALSE(constraints->SupportsQuantization(OpType::CLZ, unit, DataType::Int32, unitQuantZp1, DataType::Int32));
        REQUIRE_FALSE(constraints->SupportsQuantization(
            OpType::SHL, unit, DataType::Int32, unit, DataType::Int32, unitQuantZp1, DataType::Int32));
    }
    SECTION("rejects 32-bit OFM with zp != 0")
    {
        REQUIRE_FALSE(constraints->SupportsQuantization(
            OpType::Conv2D, unit, DataType::Int8, unit, DataType::Int8, unitQuantZp1, DataType::Int32));
    }
    SECTION("allows non-32bit OFM with zp != 0")
    {
        REQUIRE(constraints->SupportsQuantization(OpType::Add, unit, DataType::Int8, unit, DataType::Int8, unitQuantZp1, DataType::Int8));
        REQUIRE(constraints->SupportsQuantization(OpType::Add, unit, DataType::Int8, unit, DataType::Int8, unitQuantZp1, DataType::Int16));
    }
    SECTION("rejects UInt8 Add with negative IFM zeroPoint")
    {
        Quantization quant = MakeQuant(1, 0, -1);
        REQUIRE_FALSE(constraints->SupportsQuantization(OpType::Add, quant, DataType::UInt8, quant, DataType::UInt8, unit, DataType::Int32));
    }
    SECTION("rejects UInt8 Add with negative OFM zeroPoint")
    {
        Quantization quant = MakeQuant(1, 0, -1);
        REQUIRE_FALSE(constraints->SupportsQuantization(OpType::Add, unit, DataType::UInt8, unit, DataType::UInt8, quant, DataType::Int32));
    }
}

TEST_CASE("Ethos-U85 SupportsQuantization enforces zero-point rules")
{
    auto arch = CreateArchDefault<ArchEthosU85>();
    IArchitectureConstraints *constraints = arch->Constraints();
    const Quantization unit = MakeQuant();

    SECTION("rejects Int8 Add with out-of-bounds IFM zeroPoint")
    {
        Quantization quant = MakeQuant(1, 0, 255);
        REQUIRE_FALSE(constraints->SupportsQuantization(OpType::Add, quant, DataType::Int8, quant, DataType::Int8, unit, DataType::Int8));
    }
    SECTION("rejects UInt8 Add with negative IFM zeroPoint")
    {
        Quantization quant = MakeQuant(1, 0, -1);
        REQUIRE_FALSE(constraints->SupportsQuantization(OpType::Add, quant, DataType::UInt8, quant, DataType::UInt8, unit, DataType::UInt8));
    }
    SECTION("rejects Int8 Add with out-of-bounds OFM zeroPoint")
    {
        Quantization ofmQuant = MakeQuant(1, 0, 255);
        REQUIRE_FALSE(constraints->SupportsQuantization(OpType::Add, unit, DataType::Int8, unit, DataType::Int8, ofmQuant, DataType::Int8));
    }
    SECTION("allows 32-bit operation with IFM zp == 0")
    {
        REQUIRE(constraints->SupportsQuantization(OpType::Add, unit, DataType::Int32, unit, DataType::Int32, unit, DataType::Int32));
    }
    SECTION("rejects 32-bit operations with IFM zp != 0")
    {
        const Quantization unitQuantZp1 = MakeQuant(1, 0, 1);
        REQUIRE_FALSE(constraints->SupportsQuantization(
            OpType::Add, unitQuantZp1, DataType::Int32, unitQuantZp1, DataType::Int32, unit, DataType::Int32));
    }
    SECTION("allows UInt16 operations with IFM zp == 0")
    {
        const Quantization unitQuantZp0 = MakeQuant(1, 0, 0);
        REQUIRE(constraints->SupportsQuantization(
            OpType::Add, unitQuantZp0, DataType::UInt16, unitQuantZp0, DataType::UInt16, unit, DataType::UInt16));
    }
    SECTION("allows UInt16 operations with IFM zp == 32768")
    {
        const Quantization quant = MakeQuant(1, 0, 32768);
        REQUIRE(constraints->SupportsQuantization(OpType::Add, quant, DataType::UInt16, quant, DataType::UInt16, unit, DataType::UInt16));
    }
    SECTION("rejects UInt16 operations with IFM zp == 1")
    {
        const Quantization quant = MakeQuant(1, 0, 1);
        REQUIRE_FALSE(constraints->SupportsQuantization(
            OpType::Add, quant, DataType::UInt16, quant, DataType::UInt16, unit, DataType::UInt16));
    }
    SECTION("allows UInt16 operations with OFM zp == 32768")
    {
        const Quantization quant = MakeQuant(1, 0, 32768);
        REQUIRE(constraints->SupportsQuantization(OpType::Add, unit, DataType::UInt16, unit, DataType::UInt16, quant, DataType::UInt16));
    }
    SECTION("rejects Int16 operations with IFM zp == 1")
    {
        const Quantization quant = MakeQuant(1, 0, 1);
        REQUIRE_FALSE(constraints->SupportsQuantization(OpType::Add, quant, DataType::Int16, quant, DataType::Int16, unit, DataType::Int16));
    }
    SECTION("allows Int16 operations with OFM zp == 1")
    {
        const Quantization quant = MakeQuant(1, 0, 1);
        REQUIRE(constraints->SupportsQuantization(OpType::Add, unit, DataType::Int16, unit, DataType::Int16, quant, DataType::Int16));
    }
}

TEST_CASE("Ethos-U85 SupportsQuantization limits resize shifts")
{
    auto arch = CreateArchDefault<ArchEthosU85>();
    IArchitectureConstraints *constraints = arch->Constraints();

    const Quantization unit = MakeQuant();

    SECTION("accepts Resize with shift below threshold")
    {
        const Quantization allowed = MakeQuant(1, 47);
        REQUIRE(constraints->SupportsQuantization(OpType::Resize, unit, DataType::Int8, unit, DataType::None, allowed, DataType::Int8));
    }
    SECTION("rejects Resize with shift of 48 or more")
    {
        const Quantization tooLarge = MakeQuant(1, 48);
        REQUIRE_FALSE(constraints->SupportsQuantization(
            OpType::Resize, unit, DataType::Int8, unit, DataType::None, tooLarge, DataType::Int8));
    }
}
