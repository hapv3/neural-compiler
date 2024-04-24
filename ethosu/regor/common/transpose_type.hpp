//
// SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

enum class TransposeType : uint32_t
{
    H = 0x1,
    W = 0x2,
    C = 0x3,
    MaskC = 0xF,
    NHWC = 0x0123,
    NWHC = 0x0213,
    NHCW = 0x0132,
    NWCH = 0x0231,
    NCHW = 0x0312,
    NCWH = 0x0321,
    None = 0x01234567,
};

inline constexpr TransposeType operator>>(TransposeType type, uint32_t size)
{
    return TransposeType(uint32_t(type) >> size);
}

inline constexpr TransposeType operator&(TransposeType a, TransposeType b)
{
    return TransposeType(uint32_t(a) & uint32_t(b));
}

inline constexpr bool IsNone(TransposeType type)
{
    uint32_t offset = (7u - (uint32_t(type) & 7u)) * 4;
    return uint32_t(TransposeType::None) >> offset == uint32_t(type);
}

// Reduce a 4D transpose mask to a 3D transpose mask (f.ex. 0x0123 -> 0x012)
inline TransposeType Reduce4To3(TransposeType type)
{
    if ( IsNone(type) )
    {
        return TransposeType(0x012);
    }

    switch ( type )
    {
        case TransposeType::NHWC:
        case TransposeType::NWHC:
        case TransposeType::NHCW:
        case TransposeType::NWCH:
        case TransposeType::NCHW:
        case TransposeType::NCWH:
        {
            int n = uint32_t(type >> 12) & 0xF;
            assert(n == 0);
            int h = uint32_t(type >> 8) & 0xF;
            assert(h <= 3);
            int w = uint32_t(type >> 4) & 0xF;
            assert(w <= 3);
            int c = uint32_t(type >> 0) & 0xF;
            assert(c <= 3);
            return TransposeType(((h - 1) << 8) | ((w - 1) << 4) | (c - 1));
        }
        default:
            assert(false && "Unsupported transpose type");
            return type;
    }
}
