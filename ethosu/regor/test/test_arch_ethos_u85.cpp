//
// SPDX-FileCopyrightText: Copyright 2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "architecture/ethosu85/ethos_u85.hpp"
#include "util.hpp"

#include <catch_all.hpp>

#include "regor.h"

using namespace regor;

TEST_CASE("arch_ethos_u85 GetOpConfig")
{
    auto arch = CreateArchDefault<ArchEthosU85>(1024);
    ArchitectureConfigQuery query{};
    Kernel kernel({1, 1}, {1, 1}, {0, 0});
    query.ifmBits = 8;
    query.lutBytes = 0;
    query.scaled = false;
    query.ifmResampling = ArchResampling::None;
    query.transpose = TransposeType::None;
    query.ofmFormat = TensorFormat::NHCWB16;
    query.accOutputEnabled = true;

    SECTION("No waste")
    {
        OpType type = OpType::Add;
        query.ofmShape = {1, 8, 8, 32};
        query.ifmShape[0] = query.ofmShape;
        query.kernel = &kernel;
        auto archOpConfig = arch->GetOpConfig(type, query);
        EthosU85OpConfig *ethosU85OpConfig = static_cast<EthosU85OpConfig *>(archOpConfig.get());
        REQUIRE(ethosU85OpConfig->OfmUBlock() == Shape(2, 2, 32));
    }

    SECTION("Waste in H")
    {
        OpType type = OpType::Add;
        query.ofmShape = {1, 1, 8, 32};
        query.ifmShape[0] = query.ofmShape;
        query.kernel = &kernel;
        auto archOpConfig = arch->GetOpConfig(type, query);
        EthosU85OpConfig *ethosU85OpConfig = static_cast<EthosU85OpConfig *>(archOpConfig.get());
        REQUIRE(ethosU85OpConfig->OfmUBlock() == Shape(1, 4, 32));
    }

    SECTION("Waste in W")
    {
        OpType type = OpType::Add;
        query.ofmShape = {1, 8, 1, 16};
        query.ifmShape[0] = query.ofmShape;
        query.kernel = &kernel;
        auto archOpConfig = arch->GetOpConfig(type, query);
        EthosU85OpConfig *ethosU85OpConfig = static_cast<EthosU85OpConfig *>(archOpConfig.get());
        REQUIRE(ethosU85OpConfig->OfmUBlock() == Shape(2, 2, 32));
    }

    SECTION("Waste in C")
    {
        OpType type = OpType::Add;
        query.ofmShape = {1, 8, 8, 1};
        query.ifmShape[0] = query.ofmShape;
        query.kernel = &kernel;
        auto archOpConfig = arch->GetOpConfig(type, query);
        EthosU85OpConfig *ethosU85OpConfig = static_cast<EthosU85OpConfig *>(archOpConfig.get());
        REQUIRE(ethosU85OpConfig->OfmUBlock() == Shape(2, 4, 16));
    }
}


TEST_CASE("arch_ethos_u85 UpscaleAndRounding")
{
    auto arch = CreateArchDefault<ArchEthosU85>(1024);
    SECTION("Resampling None")
    {
        int rounding;
        int upscale = arch->UpscaleAndRounding(ArchResampling::None, rounding);
        REQUIRE(rounding == 0);
        REQUIRE(upscale == 1);
    }
    SECTION("Resampling Zero")
    {
        int rounding;
        int upscale = arch->UpscaleAndRounding(ArchResampling::Zeros, rounding);
        REQUIRE(rounding == 0);
        REQUIRE(upscale == 2);
    }
    SECTION("Resampling Nearest")
    {
        int rounding;
        int upscale = arch->UpscaleAndRounding(ArchResampling::Nearest, rounding);
        REQUIRE(rounding == 1);
        REQUIRE(upscale == 2);
    }
}

TEST_CASE("arch_ethos_u85 PerformanceModel ScaleAccounting")
{
    auto arch = CreateArchDefault<ArchEthosU85>();
    auto *sram = arch->StagingMemory().memory;
    auto *flash = arch->ReadonlyMemory().memory;
    REQUIRE(sram != nullptr);
    REQUIRE(flash != nullptr);

    ArchitectureOpGroupQuery opGroupQuery{};
    opGroupQuery.type = OpType::Conv2D;
    opGroupQuery.opId = 1;
    auto opGroup = arch->CreateOpGroup(opGroupQuery);

    PerformanceQuery query{};
    query.type = OpType::Conv2D;
    query.opGroup = opGroup.get();
    query.constMemory = flash;
    query.weightStagingMemory = sram;
    query.scaleStagingMemory = sram;  // Staged together
    query.ifm[0].memory = sram;
    query.ifm[1].memory = sram;
    query.ofm.memory = sram;
    query.combinedWeightsAndScales = true;
    query.encodedWeightSize = 1000;
    query.encodedScaleSize = 100;
    query.firstWeightDMASize = 550;  // 500 weights + 50 scales

    ElementAccess byteAccess{};
    byteAccess.constRead[0] = 1000;
    byteAccess.constRead[1] = 100;

    SECTION("Scale read attribution")
    {
        // Staged case
        {
            auto perf = arch->Performance()->MeasureAccessCycles(query, byteAccess);
            CHECK(perf[sram].scalesAccessCycles > 0);
            CHECK(perf[flash].scalesAccessCycles == 0);
        }

        // Not staged case
        {
            PerformanceQuery query_not_staged = query;
            query_not_staged.scaleStagingMemory = nullptr;
            auto perf = arch->Performance()->MeasureAccessCycles(query_not_staged, byteAccess);
            CHECK(perf[sram].scalesAccessCycles == 0);
            CHECK(perf[flash].scalesAccessCycles > 0);
        }
    }

    SECTION("DMA cost inclusion")
    {
        // With scales in DMA
        auto perf_with_scales = arch->Performance()->MeasureAccessCycles(query, byteAccess);

        // Without scales in DMA (simulated by setting encodedScaleSize to 0)
        PerformanceQuery query_no_scales = query;
        query_no_scales.encodedScaleSize = 0;
        auto perf_no_scales = arch->Performance()->MeasureAccessCycles(query_no_scales, byteAccess);

        // FLASH is the source of the DMA, so its weightsAccessCycles should be larger
        // when scales are included in the DMA.
        CHECK(perf_with_scales[flash].weightsAccessCycles > perf_no_scales[flash].weightsAccessCycles);
    }
}
