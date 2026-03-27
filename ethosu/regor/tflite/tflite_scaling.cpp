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

#include "tflite_scaling.hpp"

#include "common/data_type.hpp"
#include "common/numeric_util.hpp"
#include "common/scaling.hpp"
#include "compiler/op_type.hpp"
#include "compiler/quantization.hpp"

#include <cmath>
#include <limits>

namespace regor
{

namespace
{
void CalculateAddSubRescales(double input1Scale, double input2Scale, double outputScale, int inputShift,
    double &input1Rescale, double &input2Rescale, QuantizedScale &outScale)
{
    assert(inputShift >= 0 && inputShift < 64);
    auto m = 2 * std::max(input1Scale, input2Scale);
    auto f = double(int64_t(1) << inputShift);
    input1Rescale = input1Scale * f / m;
    input2Rescale = input2Scale * f / m;
    outScale = QuantizedScale(m / (outputScale * f));
}
}  // namespace

void ElementwiseAddSubScale(double input1Scale, double input2Scale, double outputScale, int bitDepth,
    double &input1Rescale, double &input2Rescale, QuantizedScale &outScale)
{
    if ( input1Scale == input2Scale && bitDepth == 8 )
    {
        CalculateAddSubRescales(input1Scale, input2Scale, outputScale, 16, input1Rescale, input2Rescale, outScale);

        // If the output scale fulfills the following constraint we can guarantee bit exact
        // double rounding behaviour when scaling the inputs only with scale and no shift.
        if ( (outScale.scale & 0xFFF) == 0 )
        {
            return;
        }
    }

    // Different input scales or cannot guarantee double rounding behaviour, need to scale/shift both inputs.
    int inputShift = bitDepth == 8 ? 20 : 15;
    CalculateAddSubRescales(input1Scale, input2Scale, outputScale, inputShift, input1Rescale, input2Rescale, outScale);
}

Quantization RescalePerChannelToExplicit(const Quantization &ifmQuant, const Quantization &weightQuant,
    const Quantization &ofmQuant, const DataType scaleDataType, const DataType ifmDataType, OpType opType)
{
    assert(ofmQuant.type == QuantizationType::TFLITE);

    Quantization quantResult;
    quantResult.type = QuantizationType::EXPLICIT;
    quantResult.zeroPoints = ofmQuant.zeroPoints;
    quantResult.quantMin = ofmQuant.quantMin;
    quantResult.quantMax = ofmQuant.quantMax;
    quantResult.dimension = ofmQuant.dimension;

    if ( !ifmQuant.scales.empty() && !ofmQuant.scales.empty() && !weightQuant.scales.empty() )
    {
        const bool reducedScale = (scaleDataType == DataType::Int64 && DataTypeSizeBits(ifmDataType) == 16);
        const bool globalScale = weightQuant.scales.size() == 1;

        const int modIfm = (ifmQuant.scales.size()) == 1 ? 0 : -1;
        const int modOfm = (ofmQuant.scales.size()) == 1 ? 0 : -1;

        quantResult.scales.reserve(weightQuant.scales.size());

        for ( int i = 0; i < int(weightQuant.scales.size()); i++ )
        {
            double v = 1.0;
            float ifmScale = float(ifmQuant.scales[i & modIfm].Dequantize());
            float ofmScale = float(ofmQuant.scales[i & modOfm].Dequantize());
            float weightScale = float(weightQuant.scales[i].Dequantize());
            if ( ifmDataType == DataType::UInt8 || (opType == OpType::FullyConnected && globalScale) )
            {
                // Fuse the IFM, weight and OFM scales into one scale using single precision
                v = double(ifmScale * weightScale) / double(ofmScale);
            }
            else if ( ifmDataType == DataType::Int8 || ifmDataType == DataType::Int16 )
            {
                // Fuse the IFM, weight and OFM scales into one scale using double precision
                v = (double(ifmScale) * double(weightScale)) / double(ofmScale);
            }

            quantResult.scales.emplace_back(v, reducedScale);
        }
    }

    return quantResult;
}

QuantizedScale TanhSigmoidScale(double ifmScale, OpType opType)
{
    double rescale = 0x3000 * ifmScale;
    // Calculate scale and shift for the output scale of 1/(3*4096)
    double xLog2 = std::log2(ifmScale);
    int roundedLog2 = int(std::round(xLog2));
    bool isPowerOf2 = std::abs(xLog2 - roundedLog2) < 0.001;
    int shift = roundedLog2 + 12;
    int scale = 0;
    if ( isPowerOf2 && ((opType == OpType::Tanh && (shift == 0 || shift == 1)) || (opType == OpType::Sigmoid && shift == 0)) )
    {
        // Special handling if input scale is 1/2048 or 1/4096
        scale = 3 << shift;
        shift = 0;
    }
    else
    {
        shift = 0;
        int maxRescale = 16384;
        while ( rescale < maxRescale && shift <= 30 )
        {
            shift++;
            rescale *= 2;
        }
        scale = uint32_t(rescale);
    }
    return QuantizedScale(scale, shift);
}

}  // namespace regor
