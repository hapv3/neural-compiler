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

#include "quantization.hpp"

#include "common/common.hpp"

namespace regor
{

std::string Quantization::ToString() const
{
    std::vector<std::string> scale;
    for ( const QuantizedScale &s : scales )
    {
        scale.push_back(fmt::format("(scale:{}, shift:{})", s.scale, s.shift));
    }
    return fmt::format("scale: [{}], zero_point: [{}], quantMin: [{}], quantMax: [{}], dimension: {}",
        fmt::join(scale, ", "), fmt::join(zeroPoints, ", "), fmt::join(quantMin, ", "), fmt::join(quantMax, ", "), dimension);
}

bool Quantization::operator==(const Quantization &rhs) const
{
    return std::tie(scales, zeroPoints, quantMin, quantMax, dimension) ==
           std::tie(rhs.scales, rhs.zeroPoints, rhs.quantMin, rhs.quantMax, rhs.dimension);
}

bool Quantization::operator!=(const Quantization &rhs) const
{
    return !(*this == rhs);
}

const Quantization &Quantization::Unit()
{
    static Quantization unitQuantization;
    if ( unitQuantization.scales.empty() )
    {
        unitQuantization.scales.emplace_back(QuantizedScale{1, 0});
        unitQuantization.zeroPoints.emplace_back(0);
    }
    return unitQuantization;
}

}  // namespace regor
