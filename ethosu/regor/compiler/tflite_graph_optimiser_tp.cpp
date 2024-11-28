//
// SPDX-FileCopyrightText: Copyright 2021, 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#include "compiler/tflite_graph_optimiser.hpp"

namespace
{

// Implementation from TensorFlow Lite Micro kernel
int16_t SaturatingLeftShift(std::int16_t value, int amount)
{
    int32_t result = value << amount;
    result = std::min<int32_t>(result, std::numeric_limits<int16_t>::max());
    result = std::max<int32_t>(result, std::numeric_limits<int16_t>::min());
    return int16_t(result);
}

// Implementation from TensorFlow Lite Micro kernel
// Similar to ARM instruction SQDMULH.
// Similar to gemmlowp::SaturatingRoundingDoublingHighMul except
// rounding to zero instead of to nearest (SQRDMULH).
int16_t SaturatingDoublingHighMul(int16_t a, int16_t b)
{
    bool overflow = a == b && a == std::numeric_limits<int16_t>::min();
    int32_t a_32(a);
    int32_t b_32(b);
    int32_t ab_32 = a_32 * b_32;
    int16_t ab_x2_high16 = int16_t((ab_32) / (1 << 15));
    return overflow ? std::numeric_limits<int16_t>::max() : ab_x2_high16;
}

}  // namespace

namespace regor
{

// Converts HardSwish to a LUT based solution.
Operation *TFLiteGraphOptimiser::ConvertHardSwishToLUT(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    auto returnOp = operation;
    auto opType = operation->Type();
    auto ifmConn = operation->Input(TensorUsage::IFM0);
    auto ofmConn = operation->Output(TensorUsage::OFM);

    if ( opType == OpType::HardSwish && ifmConn != nullptr && ofmConn != nullptr )
    {
        auto ifm = ifmConn->tensor.get();
        auto ofm = ofmConn->tensor.get();

        // Generate the LUT
        const double ifmScale = ifmConn->quantization.scales[0].Dequantize();
        const double ofmScale = ofmConn->quantization.scales[0].Dequantize();
        const int zpIn = int(ifmConn->quantization.zeroPoints[0]);
        const int zpOut = int(ofmConn->quantization.zeroPoints[0]);
        const int qMin = ifm->Type() == DataType::Int8 ? -128 : 0;
        const int qMax = ifm->Type() == DataType::Int8 ? 127 : 255;

        const double ifmScaleHires = (1.0 / 128.0) * ifmScale;
        const double reluMultiplier = 3.0 / 32768.0;

        QuantizedScale outScale(ifmScaleHires / ofmScale);
        QuantizedScale reluScale(ifmScaleHires / reluMultiplier);
        int16_t outScale16 = DownScaleInt32ToInt16Multiplier(outScale.scale);
        int16_t reluScale16 = DownScaleInt32ToInt16Multiplier(reluScale.scale);
        // convert to left shift-positive notation
        int outShift = 31 - outScale.shift;
        int reluShift = 31 - reluScale.shift;

        std::vector<uint8_t> lut;
        lut.reserve(256);
        for ( int x = qMin; x <= qMax; ++x )
        {
            // Compute the "relu-ish multiplier".
            // This matches the code in TensorFlow Lite Micro kernel
            const int16_t inputValue = int16_t(x - zpIn);

            const int16_t inputValueOnHiresInputScale = int16_t(inputValue << 7);

            const int16_t inputValueOnPreshiftOutputScale = gemmlowp::SaturatingRoundingDoublingHighMul(inputValueOnHiresInputScale, outScale16);

            int16_t reluValue = inputValueOnHiresInputScale;

            if ( reluShift > 0 )
            {
                reluValue = SaturatingLeftShift(reluValue, reluShift - 1);
            }

            reluValue = gemmlowp::SaturatingRoundingDoublingHighMul(reluValue, reluScale16);

            if ( reluShift > 0 )
            {
                reluValue = SaturatingLeftShift(reluValue, 1);
            }

            if ( reluShift < 0 )
            {
                reluValue = gemmlowp::RoundingDivideByPOT(reluValue, -reluShift);
            }
            reluValue = int16_t((reluValue + (1 << 15)) >> 1);

            const int16_t preshiftOutputValue = SaturatingDoublingHighMul(reluValue, inputValueOnPreshiftOutputScale);

            int16_t outputValue = gemmlowp::RoundingDivideByPOT(preshiftOutputValue, -outShift);

            int lutVal = outputValue + zpOut;
            lutVal = std::min(qMax, std::max(qMin, lutVal));
            lut.push_back(uint8_t(lutVal));
        }

        auto lutTens = CreateConstTensor("hardswish", ifmConn->tensor->Type(), std::make_shared<Buffer>(std::move(lut)));
        // The LUT must be applied without any preceding rescaling (the LUT itself performs the rescale),
        // so even if the OFM has a different scale than the IFM, the generated OFM scale instructions
        // should be the same as the IFM
        returnOp = CreateLUT(ifmConn->tensor, lutTens, ifmConn->quantization, ifmConn->quantization, lutTens->Type(),
            &ifmConn->shape, ofmConn->tensor, ifmConn->slice, ofmConn->slice);
        returnOp->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
    }

    if ( operation != returnOp )
    {
        RecordOptimisation(operation, returnOp);
        operation->Disconnect();
    }

    return returnOp;
}

}  // namespace regor
