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

#pragma once
#include "common/numeric_util.hpp"

#include <cstdint>

class QuantizedScale
{
public:
    int32_t scale;
    int shift;

public:
    QuantizedScale() = default;
    QuantizedScale(int32_t scale_, int shift_) : scale(scale_), shift(shift_) {}
    /**
     * Creates a quantized scale and shift from a floating-point
     * scale: floating-point representation of the scale
     * reduced: reduces the quantization to 16-bit (default 32)
     */
    QuantizedScale(double scale_, bool reduced = false);
    /*
     * Dequantizes scale into floating-point
     */
    double Dequantize() const;
    bool operator==(const QuantizedScale &other) const;
    bool operator!=(const QuantizedScale &other) const;
    /**
     * Unit scale, i.e. no scaling
     */
    static const QuantizedScale &Unit();
};

/* Calculate elementwise Mul OFM QuantizedScale */
template<typename T = float>
QuantizedScale ElementwiseMulScale(double inputScale, double input2Scale, double outputScale)
{
    // clamp to single-point precision
    T ifm1Scale = ClampToType<T>(inputScale);
    T ifm2Scale = ClampToType<T>(input2Scale);
    T outScale = ClampToType<T>(outputScale);

    T outputRescale = (ifm1Scale * ifm2Scale) / outScale;
    return QuantizedScale(outputRescale);
}

/* Convert int32_t multiplier to int16_t with rounding. */
int16_t DownScaleInt32ToInt16Multiplier(int32_t mul);
