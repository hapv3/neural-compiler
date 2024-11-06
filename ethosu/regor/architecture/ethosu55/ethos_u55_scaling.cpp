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

#include "ethos_u55_scaling.hpp"

#include "architecture/ethos_u_scaling.hpp"
#include "compiler/high_level_command_stream.hpp"
#include "compiler/op_type.hpp"
#include "compiler/quantization.hpp"

namespace regor::ethosU55Scaling
{
namespace
{
void AdvancedElementwiseAddSubScale(double input1Scale, double input2Scale, double outputScale, int bitDepth,
    QuantizedScale &input1Rescale, QuantizedScale &outScale)
{
    auto maxInputScale = std::max(input1Scale, input2Scale);
    auto minInputScale = std::min(input1Scale, input2Scale);
    int inputShift = bitDepth == 8 ? 20 : 15;
    double ifm1Rescale;
    double ifm2Rescale;
    SimplifiedElementwiseAddSubScale(minInputScale, maxInputScale, outputScale, inputShift, ifm1Rescale, ifm2Rescale, outScale);
    input1Rescale = QuantizedScale(ifm1Rescale);
}

float GetScale(const Quantization *quant)
{
    if ( quant != nullptr && quant->scales.size() != 0 )
    {
        return float(quant->scales[0].Dequantize());
    }
    else
    {
        return 1.0f;
    }
}

}  // namespace

void RescaleElementwise(HLCOperation *op)
{
    int ifmCnt = int(op->ifm.size());
    Quantization *ifm1Quant = &op->ifm[0].quantization;
    Quantization *ifm2Quant = ifmCnt == 2 ? &op->ifm[1].quantization : nullptr;
    Quantization *ofmQuant = &op->ofm.quantization;

    if ( ifm1Quant->type == QuantizationType::EXPLICIT && ofmQuant->type == QuantizationType::EXPLICIT &&
         (ifm2Quant == nullptr || ifm2Quant->type == QuantizationType::EXPLICIT) )
    {
        return;
    }

    QuantizedScale outScale(1, 0);

    double ifm1Scale = GetScale(ifm1Quant);
    double ifm2Scale = GetScale(ifm2Quant);
    double ofmScale = GetScale(ofmQuant);

    DataType ifmDataType = op->ifm[0].dataType;
    OpType opType = op->type;

    bool allHaveScale =
        (!ifm1Quant->scales.empty() && !ofmQuant->scales.empty() && ifm2Quant != nullptr && !ifm2Quant->scales.empty());
    if ( opType == OpType::Mul )
    {
        if ( allHaveScale )
        {
            outScale = ElementwiseMulScale(ifm1Scale, ifm2Scale, ofmScale);
        }
    }
    else if ( opType == OpType::LeakyRelu )
    {
        // The alpha-value is used as ofm-scale for LeakyReLU.
        // This is handled in RCS-gen as this is true for all QuantizationTypes
    }
    else if ( opType == OpType::Abs )
    {
        outScale = QuantizedScale(ifm1Scale / ofmScale);
    }
    else if ( opType == OpType::Add || opType == OpType::Sub )
    {
        int bitDepth = DataTypeSizeBits(ifmDataType);
        bool useAdvancedScaling = false;
        uint32_t opaScale = 1;
        uint32_t opbScale = 1;
        int opaShift = 0;
        int opbShift = 0;
        double ifm1Rescale;
        double ifm2Rescale;
        if ( allHaveScale )
        {
            if ( ifm1Scale == ifm2Scale )
            {
                SimplifiedElementwiseAddSubScale(ifm1Scale, ifm2Scale, ofmScale, 16, ifm1Rescale, ifm2Rescale, outScale);
                opaScale = uint32_t(round(ifm1Rescale));
                opbScale = uint32_t(round(ifm2Rescale));
                if ( bitDepth == 16 )
                {
                    // Align the double rounding with that of advanced scaling
                    opaScale /= 2;
                    opbScale /= 2;
                    --outScale.shift;
                }
                else
                {
                    // For 8 bit we can't guarantee double rounding with simplified scaling will always be
                    // the same as with advanced scaling due to different shifts. When the ofm scale fulfils
                    // the following we know that double rounding will have no effect for advanced scaling
                    // no matter the input, so we can safely use simplified scaling with double rounding disabled.
                    useAdvancedScaling = (outScale.scale & 0xFFF) != 0;
                }
            }
            else
            {
                useAdvancedScaling = true;
            }
            if ( useAdvancedScaling )
            {
                // Use advanced implementation only when input/output scales differ,
                // or when we can't guarantee the absence of rounding errors
                QuantizedScale inScale(1, 0);
                AdvancedElementwiseAddSubScale(ifm1Scale, ifm2Scale, ofmScale, bitDepth, inScale, outScale);
                if ( ifm1Scale <= ifm2Scale )
                {
                    opaScale = inScale.scale;
                    opaShift = inScale.shift;
                    opbScale = 0;
                    opbShift = 0;
                }
                else
                {
                    opaScale = 0;
                    opaShift = 0;
                    opbScale = inScale.scale;
                    opbShift = inScale.shift;
                }
            }
        }
        if ( ifm1Quant->type == QuantizationType::TFLITE )
        {
            ifm1Quant->scales.clear();
            ifm1Quant->scales.push_back({int32_t(opaScale), opaShift});
            ifm1Quant->type = QuantizationType::EXPLICIT;
        }
        if ( ifm2Quant != nullptr && ifm2Quant->type == QuantizationType::TFLITE )
        {
            ifm2Quant->scales.clear();
            ifm2Quant->scales.push_back({int32_t(opbScale), opbShift});
            ifm2Quant->type = QuantizationType::EXPLICIT;
        }
    }
    if ( ofmQuant->type == QuantizationType::TFLITE )
    {
        ofmQuant->scales.clear();
        ofmQuant->scales.push_back(outScale);
        ofmQuant->type = QuantizationType::EXPLICIT;
    }
}

void RescalePooling(HLCOperation *op, bool isNoOp)
{

    Quantization *ifm1Quant = &op->ifm[0].quantization;
    Quantization *ofmQuant = &op->ofm.quantization;
    uint32_t scale = 1;
    int shift = 0;
    DataType ifmDataType = op->ifm[0].dataType;
    OpType opType = op->type;

    if ( ofmQuant->type != QuantizationType::TFLITE &&
         // Special case for average pool with no padding
         !(op->type == OpType::AvgPool && ifm1Quant->scales == Quantization::Unit().scales && op->kernel.Padding().IsZero()) )
    {
        // Explicit scaling
        return;
    }

    if ( !ifm1Quant->scales.empty() && !ofmQuant->scales.empty() )
    {
        double ifmScale = GetScale(ifm1Quant);
        double ofmScale = GetScale(ofmQuant);
        auto actType = op->subOps.empty() ? opType : op->subOps[0].type;
        if ( actType == OpType::Sigmoid || actType == OpType::Tanh )
        {
            double rescale = 0x3000 * ifmScale;
            if ( ifmDataType == DataType::Int16 )
            {
                // Calculate scale and shift for the output scale of 1/(3*4096)
                double xLog2 = std::log2(ifmScale);
                int roundedLog2 = int(std::round(xLog2));
                bool isPowerOf2 = std::abs(xLog2 - roundedLog2) < 0.001;
                shift = roundedLog2 + 12;
                if ( isPowerOf2 && ((actType == OpType::Tanh && (shift == 0 || shift == 1)) || (actType == OpType::Sigmoid && (shift == 0))) )
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
            }
            else
            {
                QuantizePoolingScaleMaxPrecision(op->kernel.ElementsWH(), rescale, scale, shift, 32);
            }
        }
        else if ( opType == OpType::MemoryCopy )
        {
            double rescale = ifmScale / ofmScale;
            // In case of concat or other memory operation, rescaling might be needed.
            // The scale is maximised, to get maximum precision
            QuantizePoolingScaleMaxPrecision(op->kernel.ElementsWH(), rescale, scale, shift, 32);
        }
        else if ( opType == OpType::Quantize )
        {
            // Quantize operations need double-precision scaling
            QuantizedScale quantScale(ifmScale / ofmScale);
            scale = uint32_t(quantScale.scale);
            shift = quantScale.shift;
        }
        else if ( isNoOp )
        {
            QuantizedScale quantScale(float(ifmScale) / float(ofmScale));
            scale = uint32_t(quantScale.scale);
            shift = quantScale.shift;
        }
        else
        {
            // Normal pooling operation, without need for special scaling
            double rescale = ifmScale / ofmScale;
            QuantizePoolingScale(op->kernel.ElementsWH(), rescale, 0, scale, shift, 32);
        }
    }
    ofmQuant->scales.clear();
    ofmQuant->scales.push_back({int32_t(scale), shift});
    ofmQuant->type = QuantizationType::EXPLICIT;
}

}  // namespace regor::ethosU55Scaling
