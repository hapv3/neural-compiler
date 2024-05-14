//
// SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/scaling.hpp"

#include <cstdint>

namespace regor
{

void QuantizePoolingScale(int kernelElements, double rescale, int rescaleBits, uint32_t &scale, int &shift, int N);

// Max scale precision based on register size N (32 or 31)
void QuantizePoolingScaleMaxPrecision(int kernelElements, double rescale, uint32_t &scale, int &shift, int N);

// Simplified version of calculating elementwise Add/Sub scales
void SimplifiedElementwiseAddSubScale(double input1Scale, double input2Scale, double outputScale, int inputShift,
    double &input1Rescale, double &input2Rescale, QuantizedScale &outScale);

}  // namespace regor
