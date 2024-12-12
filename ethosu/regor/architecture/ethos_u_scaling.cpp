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

// Simplified version of calculating elementwise Add/Sub scales
void SimplifiedElementwiseAddSubScale(double input1Scale, double input2Scale, double outputScale, int inputShift,
    double &input1Rescale, double &input2Rescale, QuantizedScale &outScale)
{
    auto m = 2 * std::max(input1Scale, input2Scale);
    auto f = double(int64_t(1) << inputShift);
    input1Rescale = input1Scale * f / m;
    input2Rescale = input2Scale * f / m;
    double outputRescale = m / (outputScale * f);
    outScale = QuantizedScale(outputRescale);
}

Quantization RescalePerChannel(const Quantization &ifmQuant, const Quantization &weightQuant,
    const Quantization &ofmQuant, const DataType scaleDataType, const DataType ifmDataType, OpType opType)
{
    if ( ofmQuant.type != QuantizationType::TFLITE )
    {
        // Explicit quantized scale has already been set
        return ofmQuant;
    }

    Quantization quantResult;
    quantResult.type = QuantizationType::EXPLICIT;
    quantResult.zeroPoints = ofmQuant.zeroPoints;
    quantResult.quantMin = ofmQuant.quantMin;
    quantResult.quantMax = ofmQuant.quantMax;
    quantResult.dimension = ofmQuant.dimension;

    if ( !ifmQuant.scales.empty() && !ofmQuant.scales.empty() && !weightQuant.scales.empty() )
    {
        bool reducedScale = (scaleDataType == DataType::Int64 && DataTypeSizeBits(ifmDataType) == 16);

        int modIfm = (ifmQuant.scales.size()) == 1 ? 0 : -1;
        int modOfm = (ofmQuant.scales.size()) == 1 ? 0 : -1;

        quantResult.scales.reserve(weightQuant.scales.size());

        for ( int i = 0; i < int(weightQuant.scales.size()); i++ )
        {
            double v = 1.0;
            float ifmScale = float(ifmQuant.scales[i & modIfm].Dequantize());
            float ofmScale = float(ofmQuant.scales[i & modOfm].Dequantize());
            float weightScale = float(weightQuant.scales[i].Dequantize());
            if ( ifmDataType == DataType::UInt8 || opType == OpType::FullyConnected )
            {
                v = double(ifmScale * weightScale) / double(ofmScale);
            }
            else if ( ifmDataType == DataType::Int8 || ifmDataType == DataType::Int16 )
            {
                v = (double(ifmScale) * double(weightScale)) / double(ofmScale);
            }

            quantResult.scales.emplace_back(v, reducedScale);
        }
    }

    return quantResult;
}

}  // namespace regor
