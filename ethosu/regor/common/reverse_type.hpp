//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "common/bit_flags.hpp"
#include "common/shape.hpp"

namespace regor
{
enum class ReverseType : uint32_t
{
    None = 0x0,
    C = 1 << 0,
    W = 1 << 1,
    H = 1 << 2,
    N = 1 << 3,
    B = 1 << 4,
    A = 1 << 5,
    Dynamic = 1ULL << 31  // used for non-constant axes
};

Flags<ReverseType> ToReverseMask(const Shape &shape, int ofmRank);

}  // namespace regor
