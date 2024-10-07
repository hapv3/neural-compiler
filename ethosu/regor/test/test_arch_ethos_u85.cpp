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
        Kernel kernel({1, 1}, {1, 1}, {0, 0});
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
        Kernel kernel({1, 1}, {1, 1}, {0, 0});
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
        Kernel kernel({1, 1}, {1, 1}, {0, 0});
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
        Kernel kernel({1, 1}, {1, 1}, {0, 0});
        query.ofmShape = {1, 8, 8, 1};
        query.ifmShape[0] = query.ofmShape;
        query.kernel = &kernel;
        auto archOpConfig = arch->GetOpConfig(type, query);
        EthosU85OpConfig *ethosU85OpConfig = static_cast<EthosU85OpConfig *>(archOpConfig.get());
        REQUIRE(ethosU85OpConfig->OfmUBlock() == Shape(2, 4, 16));
    }
}
