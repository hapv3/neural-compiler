//
// SPDX-FileCopyrightText: Copyright 2021-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "scaling.hpp"

#include "common/numeric_util.hpp"

#include <cmath>
#include <limits>

bool QuantizedScale::operator==(const QuantizedScale &other) const
{
    return (scale == other.scale) && (shift == other.shift);
}

bool QuantizedScale::operator!=(const QuantizedScale &other) const
{
    return !(*this == other);
}

QuantizedScale::QuantizedScale(double scale_, bool reduced)
{
    int exponent = 0;
    double significand = std::frexp(scale_, &exponent);
    // convert from left to right-shift
    scale = int32_t(std::round(significand * double(1LL << 31)));
    shift = 31 - exponent;
    if ( reduced )
    {
        scale = (scale >> 16) + (scale >> 15 & 1);
        // make sure reduced scale does not overflow
        scale = std::min<int32_t>(scale, 0x7FFF);
        shift -= 16;
    }
    // if shift is out of bounds [0,63], try to get back within bounds
    if ( shift > 63 )
    {
        if ( scale > std::exp2(shift - 63) )
        {
            scale = scale >> (shift - 63);
            shift = 63;
        }
        else
        {
            // Not possible to get back within bounds, set scale and shift to 0
            // as the shift would shift away all relevant bits anyway.
            scale = 0;
            shift = 0;
        }
    }
    else if ( shift < 0 && scale < std::exp2(shift + 32) )
    {
        scale = scale << (0 - shift);
        shift = 0;
    }
}

double QuantizedScale::Dequantize() const
{
    double significand = double(scale);
    // ldexp expects a left-shift
    // so we convert from right to left-shift
    int exp = -shift;
    return std::ldexp(significand, exp);
}

const QuantizedScale &QuantizedScale::Unit()
{
    static const QuantizedScale unitScale(1, 0);
    return unitScale;
}

QuantizedScale QuantizedScale::ReduceScale(const QuantizedScale &qs)
{
    auto scale = qs.scale;
    auto shift = qs.shift;
    while ( scale > 1 && (scale & 0x1) == 0 && shift > 0 )
    {
        scale >>= 1;
        shift--;
    }
    return {scale, shift};
}

// Convert int32_t multiplier to int16_t with rounding.
int16_t DownScaleInt32ToInt16Multiplier(int32_t multiplier)
{
    return ClampToType<int16_t>(((multiplier / 32768) + 1) / 2);
}
