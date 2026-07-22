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

TEST_CASE("architecture layout hooks preserve Ethos defaults")
{
    auto arch = CreateArchDefault<ArchEthosU85>(1024);
    const Shape logical(1, 2, 3, 17);

    REQUIRE(arch->AllocationQuantum() == 16);
    REQUIRE(arch->TensorAlignment(TensorUsage::IFM, TensorFormat::NHCWB16) == 16);
    REQUIRE(arch->ModelBindingFormat(TensorUsage::IFM) == TensorFormat::NHWC);
    REQUIRE(arch->DefaultInternalTensorFormat(TensorUsage::IFM, false) == TensorFormat::NHCWB16);
    REQUIRE(arch->DefaultInternalTensorFormat(TensorUsage::OFM, true) == TensorFormat::NHWC);
    REQUIRE(arch->StorageShape(logical, TensorFormat::NHCWB16) == Shape(1, 2, 3, 32));
    REQUIRE(arch->StorageShape(logical, TensorFormat::NHWC) == logical);
    REQUIRE(arch->StorageBytes(logical, TensorFormat::NHCWB16, DataType::Int8) == 192);
    REQUIRE(arch->StorageBytes(logical, TensorFormat::NHWC, DataType::Int8) == 112);
    REQUIRE(arch->TensorStrides(logical, TensorFormat::NHWC, DataType::Int8) == Shape(102, 51, 17, 1));
    REQUIRE(arch->TensorStrides(logical, TensorFormat::NHCWB16, DataType::Int8) == Shape(192, 96, 16, 48));
    REQUIRE(arch->CanAliasDepthOffset(TensorFormat::NHCWB16, 16));
    REQUIRE_FALSE(arch->CanAliasDepthOffset(TensorFormat::NHCWB16, 1));
    REQUIRE(arch->RollingBufferShape(Shape(1, 3, 4, 17), Shape(1, 2, 4, 17),
        TensorFormat::NHCWB16) == Shape(1, 6, 4, 32));
}
