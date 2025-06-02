//
// SPDX-FileCopyrightText: Copyright 2021-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "ethos_u85_scaling.hpp"

#include "common/logging.hpp"

#include "architecture/ethos_u_scaling.hpp"
#include "compiler/high_level_command_stream.hpp"
#include "compiler/op_type.hpp"
#include "compiler/quantization.hpp"

namespace regor::ethosU85Scaling
{

namespace
{
void AdvancedElementwiseAddSubScale(double input1Scale, double input2Scale, double outputScale, int bitDepth,
    QuantizedScale &input1Rescale, QuantizedScale &input2Rescale, QuantizedScale &outScale)
{
    int inputShift = bitDepth == 8 ? 20 : 15;
    double ifm1Rescale;
    double ifm2Rescale;
    SimplifiedElementwiseAddSubScale(input1Scale, input2Scale, outputScale, inputShift, ifm1Rescale, ifm2Rescale, outScale);
    input1Rescale = QuantizedScale(ifm1Rescale);
    input2Rescale = QuantizedScale(ifm2Rescale);
}
}  // namespace

void RescaleConvolution(HLCOperation *op)
{
    int ifmCnt = int(op->ifm.size());
    Quantization *ifm1Quant = &op->ifm[0].quantization;
    Quantization *ifm2Quant = ifmCnt == 2 ? &op->ifm[1].quantization : nullptr;
    Quantization *ofmQuant = &op->ofm.quantization;
    assert(ifm1Quant && ofmQuant);

    if ( ofmQuant->type == QuantizationType::EXPLICIT )
    {
        return;
    }

    QuantizedScale outScale(1, 0);

    double ifm1Scale = ifm1Quant->Scale().Dequantize();
    double ifm2Scale = ifm2Quant ? ifm2Quant->Scale().Dequantize() : 1.0;
    double ofmScale = ofmQuant->Scale().Dequantize();

    DataType ifmDataType = op->ifm[0].dataType;
    OpType opType = op->type;

    bool allHaveScale =
        (!ifm1Quant->scales.empty() && !ofmQuant->scales.empty() && ifm2Quant != nullptr && !ifm2Quant->scales.empty());

    bool reducedScale = DataTypeSizeBits(ifmDataType) != 8;

    // If ifmCnt is 2 then it is a convolution with dynamic weights and global scale is used
    if ( ifmCnt == 2 && allHaveScale )
    {
        if ( reducedScale )
        {
            outScale = QuantizedScale((ifm1Scale * ifm2Scale) / ofmScale, true);
        }
        else
        {
            outScale = ElementwiseMulScale<float, double>(ifm1Scale, ifm2Scale, ofmScale);
        }
    }

    ofmQuant->scales.clear();
    ofmQuant->scales.push_back(outScale);
    ofmQuant->type = QuantizationType::EXPLICIT;
}

void RescaleElementwise(HLCOperation *op)
{
    int ifmCnt = int(op->ifm.size());
    Quantization *ifm1Quant = &op->ifm[0].quantization;
    Quantization *ifm2Quant = ifmCnt == 2 ? &op->ifm[1].quantization : nullptr;
    Quantization *ofmQuant = &op->ofm.quantization;
    assert(ifm1Quant && ofmQuant);

    if ( ifm1Quant->type == QuantizationType::EXPLICIT && ofmQuant->type == QuantizationType::EXPLICIT &&
         (ifm2Quant == nullptr || ifm2Quant->type == QuantizationType::EXPLICIT) )
    {
        return;
    }

    QuantizedScale input1Scale(1, 0);
    QuantizedScale input2Scale(1, 0);
    QuantizedScale outScale(1, 0);

    double ifm1Scale = ifm1Quant->Scale().Dequantize();
    double ifm2Scale = ifm2Quant ? ifm2Quant->Scale().Dequantize() : 1.0;
    double ofmScale = ofmQuant->Scale().Dequantize();

    DataType ifmDataType = op->ifm[0].dataType;
    OpType opType = op->type;


    double effectiveScale = 0;
    if ( !op->subOps.empty() && (op->subOps[0].type == OpType::Sigmoid || op->subOps[0].type == OpType::Tanh) )
    {
        // Adjust for Sigmoid/Tanh effective output scale.
        effectiveScale = 1.0 / 0x3000;
    }

    bool allHaveScale =
        (!ifm1Quant->scales.empty() && !ofmQuant->scales.empty() && ifm2Quant != nullptr && !ifm2Quant->scales.empty());

    int bitDepth = DataTypeSizeBits(ifmDataType);
    if ( opType == OpType::Div )
    {
        // Div scales should always be Unit
    }
    else if ( opType == OpType::Mul )
    {
        if ( allHaveScale )
        {
            ofmScale = effectiveScale ? effectiveScale : ofmScale;
            outScale = ElementwiseMulScale(ifm1Scale, ifm2Scale, ofmScale);
        }
    }
    else if ( opType == OpType::Abs )
    {
        outScale = QuantizedScale(ifm1Scale / ofmScale);
    }
    else if ( opType == OpType::LeakyRelu )
    {
        // input1Scale is used for rescaling and input2Scale is used for alpha.
        input1Scale = QuantizedScale(float(ifm1Scale) / float(ofmScale));
        input2Scale = QuantizedScale(float(op->parameters.leaky_relu.alpha) * float(ifm1Scale) / float(ofmScale));
    }
    else if ( opType == OpType::Add || opType == OpType::Sub )
    {
        if ( allHaveScale )
        {
            ofmScale = effectiveScale ? effectiveScale : ofmScale;
            AdvancedElementwiseAddSubScale(ifm1Scale, ifm2Scale, ofmScale, bitDepth, input1Scale, input2Scale, outScale);
        }
    }

    if ( ifm1Quant->type == QuantizationType::TFLITE )
    {
        ifm1Quant->scales.clear();
        ifm1Quant->scales.push_back(input1Scale);
        ifm1Quant->type = QuantizationType::EXPLICIT;
    }
    if ( ifm2Quant != nullptr && ifm2Quant->type == QuantizationType::TFLITE )
    {
        ifm2Quant->scales.clear();
        ifm2Quant->scales.push_back(input2Scale);
        ifm2Quant->type = QuantizationType::EXPLICIT;
    }
    if ( ofmQuant->type == QuantizationType::TFLITE )
    {
        ofmQuant->scales.clear();
        ofmQuant->scales.push_back(outScale);
        ofmQuant->type = QuantizationType::EXPLICIT;
    }
    if ( opType == OpType::LeakyRelu )
    {
        ifm1Quant->scales.push_back(input2Scale);
    }
}

void RescalePooling(HLCOperation *op, bool isNoOp)
{
    Quantization *ifm1Quant = &op->ifm[0].quantization;
    Quantization *ofmQuant = &op->ofm.quantization;
    assert(ifm1Quant && ofmQuant);
    uint32_t scale = 1;
    int shift = 0;
    DataType ifmDataType = op->ifm[0].dataType;
    OpType opType = op->type;

    if ( ofmQuant->type != QuantizationType::TFLITE )
    {
        // Explicit scaling
        return;
    }

    if ( opType == OpType::MaxPool || opType == OpType::ArgMax || opType == OpType::ReduceMin ||
         opType == OpType::ReduceMax || opType == OpType::ReduceAny || opType == OpType::ReduceAll )
    {
        // Do nothing
    }
    else if ( !ifm1Quant->scales.empty() && !ofmQuant->scales.empty() )
    {
        if ( opType == OpType::Sigmoid || opType == OpType::Tanh )
        {
            double ifmScale = ifm1Quant->Scale().Dequantize();
            assert(ifmDataType == DataType::Int16);
            double rescale = 0x3000 * ifmScale;
            // Calculate scale and shift for the output scale of 1/(3*4096)
            double xLog2 = std::log2(ifmScale);
            int roundedLog2 = int(std::round(xLog2));
            bool isPowerOf2 = std::abs(xLog2 - roundedLog2) < 0.001;
            shift = roundedLog2 + 12;
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
        }
        else if ( opType == OpType::MemoryCopy )
        {
            // In the case of concat or other memory operation, rescaling might be needed.
            // The scale is maximised, to get maximum precision
            QuantizePoolingScaleMaxPrecision(op->kernel.ElementsWH(), GetScaleFactor(op), scale, shift, 31);
        }
        else if ( opType == OpType::Quantize )
        {
            // Quantize operations need double-precision scaling
            QuantizedScale quantScale(GetScaleFactor(op));
            scale = uint32_t(quantScale.scale);
            shift = quantScale.shift;
        }
        else if ( isNoOp )
        {
            QuantizedScale quantScale(GetScaleFactor(op, /* reducedPrecision */ true));
            scale = uint32_t(quantScale.scale);
            shift = quantScale.shift;
        }
        else
        {
            // Normal pooling operation, without need for special scaling
            QuantizePoolingScale(op->kernel.ElementsWH(), GetScaleFactor(op), 0, scale, shift, 31);
        }
    }
    ofmQuant->scales.clear();
    ofmQuant->scales.push_back({int32_t(scale), shift});
    ofmQuant->type = QuantizationType::EXPLICIT;
}

}  // namespace regor::ethosU85Scaling
