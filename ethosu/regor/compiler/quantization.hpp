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

#include "common/common.hpp"

#include "common/scaling.hpp"

#include <vector>

namespace regor
{

enum class QuantizationType
{
    TFLITE,    // TFLite-specific rescale in backend
    EXPLICIT,  // Explicit scaling
};

class Quantization
{
public:
    QuantizationType type = QuantizationType::EXPLICIT;
    std::vector<QuantizedScale> scales;
    std::vector<int64_t> zeroPoints;
    std::vector<int64_t> quantMin;
    std::vector<int64_t> quantMax;
    int dimension = 0;

public:
    Quantization() = default;
    Quantization(Quantization &&other) noexcept { *this = std::move(other); }
    Quantization(const Quantization &other) { *this = other; }

    static const Quantization &Unit();
    bool operator==(const Quantization &rhs) const;
    bool operator!=(const Quantization &rhs) const;
    std::string ToString() const;
    bool IsValid() const { return !zeroPoints.empty() && !scales.empty(); }
    bool EqualScales(const Quantization &other) const
    {
        return other.scales == scales && other.zeroPoints == zeroPoints;
    }
    bool IsUnitScale() const { return Quantization::Unit().scales == scales || scales.empty(); }
    explicit operator bool() const { return IsValid(); }

    Quantization &operator=(const Quantization &other)
    {
        if ( this != &other )
        {
            type = other.type;
            scales = other.scales;
            zeroPoints = other.zeroPoints;
            quantMin = other.quantMin;
            quantMax = other.quantMax;
            dimension = other.dimension;
        }
        return *this;
    }

    Quantization &operator=(Quantization &&other) noexcept
    {
        if ( this != &other )
        {
            type = other.type;
            scales = std::move(other.scales);
            zeroPoints = std::move(other.zeroPoints);
            quantMin = std::move(other.quantMin);
            quantMax = std::move(other.quantMax);
            dimension = other.dimension;
        }
        return *this;
    }
};

}  // namespace regor
