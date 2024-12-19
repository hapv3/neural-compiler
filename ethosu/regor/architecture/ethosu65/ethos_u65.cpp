//
// SPDX-FileCopyrightText: Copyright 2021-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "ethos_u65.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "common/numeric_util.hpp"
#include "ethos_u65_register_cs_generator.hpp"

#include <algorithm>
#include <iterator>
#include <limits>

namespace regor
{

static const EthosU55PerfInfo s_EthosU65PerfInfo[] = {
    // Accelerator.Ethos_U65_256
    {{0.625, 1.125, 0.5, 0.375, 0.5, 0.75, 0.125, 0.25}, {1.0, 0.25, 0.0}},
    // Accelerator.Ethos_U65_512
    {{0.3125, 0.5625, 0.25, 0.1875, 0.25, 0.375, 0.0625, 0.125}, {0.5, 0.125, 0.0}}};

static const ArchEthosU55::AcceleratorConfig s_EthosU65Configs[] = {
    // Accelerator.Ethos_U65_256
    {256, 1, Shape(2, 2, 8), Shape(2, 2, 8), 48, {8, 8, 8, 8, 16, 8, 16, 20}, 8, &s_EthosU65PerfInfo[0]},
    // Accelerator.Ethos_U65_512
    {256, 2, Shape(2, 2, 8), Shape(2, 2, 8), 48, {8, 8, 8, 8, 16, 8, 16, 20}, 8, &s_EthosU65PerfInfo[1]},
};

ArchEthosU65::ArchEthosU65()
{
}

bool ArchEthosU65::ParseConfig(IniReader *reader)
{
    // Parse architecture configuration
    std::string key;
    int macs = 0;
    int cores = 0;
    while ( reader->Begin(key) )
    {
        if ( key == "macs" )
        {
            macs = reader->Get<int>();
        }
        else if ( key == "cores" )
        {
            cores = reader->Get<int>();
        }
        reader->End();
    }

    // Find the requested MAC configuration for this accelerator
    auto cfg = std::find_if(s_EthosU65Configs, std::cend(s_EthosU65Configs),
        [&](const AcceleratorConfig &config) { return config.macs == macs && config.cores == cores; });
    if ( cfg == std::cend(s_EthosU65Configs) )
    {
        assert(macs == 256 && ((cores == 1) || (cores == 2)));
        LOG_TRACE0("Unable to find U65 accelerator for macs={} cores={}", macs, cores);
        return false;
    }

    ApplyConfig(cfg);
    _rcsGenerator = std::make_unique<EthosU65RCSGenerator>(this);

    return true;
}

std::vector<uint32_t> ArchEthosU65::ConfigRegisters()
{
    return std::vector<uint32_t>(1, ConfigRegister(1));
}

}  // namespace regor
