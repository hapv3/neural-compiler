//
// SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#pragma once

#include "architecture/ethosu55/ethos_u55.hpp"
#include "common/shape.hpp"

#include <string>

namespace regor
{

/// <summary>
/// EthosU65 specialisation (based on U55)
/// </summary>
class ArchEthosU65 : public ArchEthosU55
{
public:
    ArchEthosU65();

    bool ParseConfig(IniReader *reader) override;
    Address MaxAddress() override { return 1LL << 40; }
    std::vector<uint32_t> ConfigRegisters() override;

private:
    int MaxOutstandingDMAOps() override { return 2; }
};

}  // namespace regor
