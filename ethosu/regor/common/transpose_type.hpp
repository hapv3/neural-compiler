//
// SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include <cassert>
#include <cstdint>

namespace regor
{
enum class TransposeType : uint32_t
{
    C = 0x0,
    W = 0x1,
    H = 0x2,
    MaskC = 0xF,
    None = 0x76543210,  // NHWC
    NWHC = 0x76543120,
    NHCW = 0x76543201,
    NWCH = 0x76543102,
    NCHW = 0x76543021,
    NCWH = 0x76543012
};

inline constexpr TransposeType operator>>(TransposeType type, uint32_t size)
{
    assert(size < 32);
    return TransposeType(uint32_t(type) >> size);
}

inline constexpr TransposeType operator&(TransposeType a, TransposeType b)
{
    return TransposeType(uint32_t(a) & uint32_t(b));
}

inline constexpr bool IsNone(TransposeType type)
{
    return type == TransposeType::None;
}

}  // namespace regor
