//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compiler/quantization.hpp"

#include <cstdint>
#include <vector>

namespace regor::neuralai
{

// Selects an integer multiplier and right shift for a positive real scale.
// The search is deliberately independent of signed right-shift behaviour so
// that the compiler reference and the RTL model use the same result.
bool CalculateQuantizedMultiplier(double realScale, int32_t &multiplier, int32_t &shift);

bool GenerateQuantizationParameters(const Quantization &ifmQuant, const Quantization &weightQuant,
    const Quantization &ofmQuant, std::vector<QuantizedScale> &outScales);

int64_t RoundDivideAwayFromZero(int64_t value, uint32_t shift);
int32_t Requantize(int64_t accumulator, const QuantizedScale &scale, int32_t zeroPoint,
    int32_t clampMin, int32_t clampMax);

}  // namespace regor::neuralai
