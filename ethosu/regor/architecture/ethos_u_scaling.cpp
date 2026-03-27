//
// SPDX-FileCopyrightText: Copyright 2021-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "ethos_u_scaling.hpp"

#include "common/numeric_util.hpp"
#include "compiler/quantization.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace regor
{

void QuantizePoolingScale(int kernelElements, double rescale, int rescaleBits, uint32_t &scale, int &shift, int N)
{
    int exp;
    std::frexp(float(kernelElements - 1), &exp);
    // N = scale instruction register size
    int n = (N - 1) - rescaleBits;
    scale = uint32_t(std::ceil(rescale * double(((1ULL << (n + exp)) + (1ULL << exp)) / kernelElements)));
    shift = n + exp;
    assert(unsigned(shift) < 64);
}

void QuantizePoolingScaleMaxPrecision(int kernelElements, double rescale, uint32_t &scale, int &shift, int N)
{
    int rescaleBits = 0;
    // if rescale != 1, scale need to consider the number of bits needed for rescaling
    if ( rescale > 1 )
    {
        rescaleBits = IntLog2(rescale) + 2;
    }
    else if ( rescale < 1 )
    {
        rescaleBits = -IntLog2(1.0 / rescale);
    }
    QuantizePoolingScale(kernelElements, rescale, rescaleBits, scale, shift, N);
}

}  // namespace regor
