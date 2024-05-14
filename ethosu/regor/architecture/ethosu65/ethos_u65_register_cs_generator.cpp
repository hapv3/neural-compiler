//
// SPDX-FileCopyrightText: Copyright 2021, 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "ethos_u65_register_cs_generator.hpp"

#include "ethos_u65.hpp"
#define NPU_NAMESPACE ethosu65
#include "ethos_u65_interface.hpp"

namespace regor
{

using namespace ethosu65;

EthosU65RCSGenerator::EthosU65RCSGenerator(ArchEthosU65 *arch) : EthosU55RCSGenerator(arch), _arch(arch)
{
}

void EthosU65RCSGenerator::GenerateInitialRegisterSetup()
{
    auto mode = _arch->_cores <= 1 ? parallel_mode::SINGLE_CORE : parallel_mode::DUAL_CORE_DEPTH;
    Emit(isa::npu_set_parallel_mode_t(mode));
}

}  // namespace regor
