//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/common.hpp"

#include "common/bit_flags.hpp"

#include <cstdint>
#include <unordered_map>
#include <utility>

namespace regor
{

using Address = int64_t;

enum class MemUsage : uint16_t
{
    None = 0,
    ReadOnly = 0x1,
    FeatureMap = 0x2,
    LUT = 0x4,
    Staging = 0x8,
    Input = 0x10,
    Output = 0x20,
};

struct MemUsageAddress
{
    Flags<MemUsage> usage;
    Address address;
};

using TensorAddressMap = std::unordered_map<UniqueId, MemUsageAddress>;

}  // namespace regor
