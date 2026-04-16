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

namespace regor
{

// Generates a 16-bit interpolating LUT for a given function.
template<typename FloatT, typename FUNC>
std::unique_ptr<uint32_t[]> GenerateInterpolatingLUT16(FUNC func, FloatT ifmScale, FloatT ofmScale, int64_t zpIn, int64_t zpOut)
{
    constexpr int steps = 512;
    constexpr FloatT qMin = std::numeric_limits<int16_t>::min();
    constexpr FloatT qMax = std::numeric_limits<int16_t>::max();
    const FloatT inputMin = ifmScale * (qMin - zpIn);
    const FloatT inputMax = ifmScale * (qMax - zpIn);
    const FloatT outputMin = ofmScale * (qMin - zpOut);
    const FloatT outputMax = ofmScale * (qMax - zpOut);
    const FloatT step = (inputMax - inputMin) / steps;
    const FloatT halfStep = step / 2;
    const FloatT outputScalingInv = (qMax - qMin + 1) / (outputMax - outputMin);

    // Create 32-bit LUT represented by a 16-bit base and 16-bit slope.
    auto lut = std::make_unique<uint32_t[]>(steps);
    int16_t prevLutResult = 0;
    for ( int i = 0; i < steps; i++ )
    {
        FloatT val = func(inputMin + i * step);
        FloatT valMidpoint = func(inputMin + i * step + halfStep);
        FloatT valNext = func(inputMin + (i + 1) * step);
        FloatT sampleVal = std::round(val * outputScalingInv);

        FloatT midpointInterpVal = std::round((valNext * outputScalingInv + sampleVal) / 2);
        FloatT midpointVal = std::round(valMidpoint * outputScalingInv);
        FloatT midpointErr = midpointInterpVal - midpointVal;
        FloatT bias = std::round(midpointErr / 2);

        FloatT clampedLutResult = std::clamp(sampleVal - bias, qMin, qMax);
        int16_t lutResult = int16_t(clampedLutResult);

        if ( i > 0 )
        {
            int16_t base = prevLutResult;
            int16_t slope = lutResult - prevLutResult;
            lut[i - 1] = uint16_t(base) + (uint16_t(slope) << 16);
        }
        prevLutResult = lutResult;
    }
    FloatT val = FloatT(std::round(func(inputMax) * outputScalingInv));
    FloatT clampedLutResult = std::clamp(val, qMin, qMax);
    int16_t lutResult = int16_t(clampedLutResult);
    uint32_t base = uint32_t(prevLutResult);
    uint32_t slope = uint32_t(lutResult - prevLutResult);
    lut[steps - 1] = base + (slope << 16);

    return lut;
}

}  // namespace regor
