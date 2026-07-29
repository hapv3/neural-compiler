//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_quantization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace regor::neuralai
{

bool CalculateQuantizedMultiplier(double realScale, int32_t &multiplier, int32_t &shift)
{
    if ( !std::isfinite(realScale) || realScale <= 0.0 ) return false;

    double bestError = std::numeric_limits<double>::infinity();
    int32_t bestMultiplier = 0;
    int32_t bestShift = -1;
    for ( int32_t candidateShift = 0; candidateShift <= 31; ++candidateShift )
    {
        const double scaled = std::ldexp(realScale, candidateShift);
        if ( !std::isfinite(scaled) || scaled < 1.0 ||
             scaled > double(std::numeric_limits<int32_t>::max()) ) continue;
        const double rounded = std::round(scaled);
        if ( rounded <= 0.0 || rounded > double(std::numeric_limits<int32_t>::max()) ) continue;
        const int32_t candidateMultiplier = int32_t(rounded);
        const double error = std::abs(std::ldexp(double(candidateMultiplier), -candidateShift) - realScale);
        if ( error < bestError || (error == bestError && candidateShift > bestShift) )
        {
            bestError = error;
            bestMultiplier = candidateMultiplier;
            bestShift = candidateShift;
        }
    }
    if ( bestShift < 0 ) return false;
    multiplier = bestMultiplier;
    shift = bestShift;
    return true;
}

bool GenerateQuantizationParameters(const Quantization &ifmQuant, const Quantization &weightQuant,
    const Quantization &ofmQuant, std::vector<QuantizedScale> &outScales)
{
    const double ifmScale = ifmQuant.scales.empty() ? 1.0 : ifmQuant.scales.front().Dequantize();
    const double ofmScale = ofmQuant.scales.empty() ? 1.0 : ofmQuant.scales.front().Dequantize();
    if ( !std::isfinite(ifmScale) || !std::isfinite(ofmScale) || ifmScale <= 0.0 || ofmScale <= 0.0 )
        return false;

    const int channelCount = std::max(1, int(weightQuant.scales.size()));
    outScales.clear();
    outScales.reserve(channelCount);
    for ( int channel = 0; channel < channelCount; ++channel )
    {
        const double weightScale = weightQuant.scales.empty() ? 1.0 :
            weightQuant.scales[channel % int(weightQuant.scales.size())].Dequantize();
        int32_t multiplier = 0;
        int32_t shift = 0;
        if ( !std::isfinite(weightScale) || weightScale <= 0.0 ||
             !CalculateQuantizedMultiplier(ifmScale * weightScale / ofmScale, multiplier, shift) ) return false;
        outScales.emplace_back(multiplier, shift);
    }
    return true;
}

int64_t RoundDivideAwayFromZero(int64_t value, uint32_t shift)
{
    if ( shift == 0 ) return value;
    if ( shift > 63 ) return 0;
    const uint64_t magnitude = value < 0 ? uint64_t(-(value + 1)) + 1ULL : uint64_t(value);
    const uint64_t half = 1ULL << (shift - 1);
    const uint64_t rounded = (magnitude + half) >> shift;
    if ( value >= 0 ) return rounded > uint64_t(std::numeric_limits<int64_t>::max()) ?
        std::numeric_limits<int64_t>::max() : int64_t(rounded);
    if ( rounded > uint64_t(std::numeric_limits<int64_t>::max()) + 1ULL )
        return std::numeric_limits<int64_t>::min();
    return rounded == uint64_t(std::numeric_limits<int64_t>::max()) + 1ULL ?
        std::numeric_limits<int64_t>::min() : -int64_t(rounded);
}

int32_t Requantize(int64_t accumulator, const QuantizedScale &scale, int32_t zeroPoint,
    int32_t clampMin, int32_t clampMax)
{
    if ( scale.shift < 0 || scale.shift > 31 || clampMin > clampMax ) return 0;
    const __int128 product = __int128(accumulator) * __int128(scale.scale);
    const __int128 half = scale.shift == 0 ? 0 : (__int128(1) << (scale.shift - 1));
    const __int128 magnitude = product < 0 ? -product : product;
    const __int128 roundedMagnitude = scale.shift == 0 ? magnitude :
        ((magnitude + half) >> scale.shift);
    const __int128 scaled = product < 0 ? -roundedMagnitude : roundedMagnitude;
    const __int128 shifted = scaled + zeroPoint;
    return int32_t(std::clamp<__int128>(shifted, clampMin, clampMax));
}

}  // namespace regor::neuralai
