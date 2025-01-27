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

#include "include/graphapi.hpp"

namespace regor
{

// Aliased by value (using/typedef doesn't work)
enum class AxisOrder : uint16_t
{
    Unknown = int16_t(GraphApi::AxisOrder::Unknown),
    OHWI = int16_t(GraphApi::AxisOrder::OHWI),
    IHWO = int16_t(GraphApi::AxisOrder::IHWO),
    OI = int16_t(GraphApi::AxisOrder::OI),
    HWCM = int16_t(GraphApi::AxisOrder::HWCM),
};

/// <summary>
/// Classification for how a Tensor is consumed by an operator.
/// </summary>
enum class TensorUsage : uint32_t
{
    None = 0,
    IFM = 0x01,
    OFM = 0x02,
    Weights = 0x03,
    Scales = 0x04,
    Params = 0x05,
    LUT = 0x06,
    State = 0x07,
    Scratch = 0x08,
    UserDefined = 0x1E,
    Last,
    TypeMask = 0x1F,
    IndexShift = 8,
    IndexMask = 0xFFFFF00,
    IFM0 = IFM,
    IFM1 = 0x0100 | IFM,
    IFM2 = 0x0200 | IFM,
    Params0 = Params,
    Params1 = 0x100 | Params,
    Params2 = 0x200 | Params,
    Scratch0 = Scratch,
};

DECLARE_ENUM_AS_FLAGS(TensorUsage)

constexpr inline bool IsOFM(TensorUsage usage)
{
    return (usage & TensorUsage::TypeMask) == TensorUsage::OFM;
}

constexpr inline bool IsIFM(TensorUsage usage)
{
    return (usage & TensorUsage::TypeMask) == TensorUsage::IFM;
}

template<typename NUMERIC>
constexpr inline TensorUsage MakeTensorUsage(TensorUsage type, NUMERIC index)
{
    return TensorUsage(uint32_t(type) | (uint32_t(index) << 8));
}

constexpr inline int GetUsageIndex(TensorUsage usage)
{
    return unsigned(usage & TensorUsage::IndexMask) >> unsigned(TensorUsage::IndexShift);
}
constexpr inline TensorUsage GetUsageType(TensorUsage usage)
{
    return (usage & TensorUsage::TypeMask);
}

}  // namespace regor
