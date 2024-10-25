//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "architecture/ethosu55/ethos_u55_register_cs_generator.hpp"
#include "ethos_u65.hpp"

namespace regor
{
/// <summary>
/// Generates register command streams for Ethos-U65.
/// Inherits common parts from Ethos-U55 generator.
/// </summary>
class EthosU65RCSGenerator : public EthosU55RCSGenerator
{
public:
    EthosU65RCSGenerator(ArchEthosU65 *arch);

protected:
    // Converts TILE operations to DMA commands
    void InsertTileDMACommand(const HLCStripe *stripe, Temporaries &temps, std::vector<const HighLevelCommand *> &emitted) override;

    // Generate register commands for DMA operations
    void GenerateDMA(const HLCDMA *dma, AccessTracking &accesses) override;
    void GenerateInitialRegisterSetup() override;

private:
    ArchEthosU65 *_arch;
};

}  // namespace regor
