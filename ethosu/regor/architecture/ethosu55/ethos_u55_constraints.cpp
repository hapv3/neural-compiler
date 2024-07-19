//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "ethos_u55_constraints.hpp"

#include "ethos_u55_register_cs_generator.hpp"

namespace regor
{

bool EthosU55Constraints::SupportsLeakyRelu(bool quantized, DataType type)
{
    return quantized == false && type == DataType::Int16;
}

bool EthosU55Constraints::SupportsMatMul(OpType opType)
{
    UNUSED(opType);
    return false;
}

bool EthosU55Constraints::SupportsTransposeHW(OpType opType, TransposeType transposeType)
{
    UNUSED(opType);
    UNUSED(transposeType);
    return IsNone(transposeType);
}

bool EthosU55Constraints::SupportsTranspose(OpType opType, TransposeType transposeType, Shape ifmShape)
{
    UNUSED(ifmShape);
    if ( SupportsTransposeHW(opType, transposeType) )
    {
        return true;
    }
    return false;
}

bool EthosU55Constraints::SupportsReverse(OpType opType, ReverseType reverseType)
{
    UNUSED(opType);
    UNUSED(reverseType);
    return reverseType == ReverseType::None;
}

bool EthosU55Constraints::SupportsFusedRescale(
    OpType opType, TensorUsage tensorUsage, DataType fromType, DataType toType, const Quantization &quantization)
{
    auto npuOp = ArchEthosU55::GetHWOp(opType);
    bool globalScale = quantization.scales.size() == 1;
    int fromBits = DataTypeSizeBits(fromType);
    int toBits = DataTypeSizeBits(toType);

    if ( tensorUsage == TensorUsage::IFM )
    {
        if ( npuOp == EthosU55NpuOp::Elementwise )
        {
            bool fromTypeSupported = IsInteger(fromType) && (fromBits == 8 || fromBits == 16);
            bool toTypeSupported = (IsInteger(toType) && (toBits == 8 || toBits == 16)) || toType == DataType::Int32;
            // TODO: Only one ifm can have full 32-bit (advanced) rescale, so for now only allow 16-bit (simple) rescale
            auto &qs = quantization.scales.front();
            bool scaleSupported = qs.shift == 0 && static_cast<int16_t>(qs.scale) == qs.scale;
            return (opType == OpType::Add || opType == OpType::Sub) && globalScale && fromTypeSupported && scaleSupported;
        }
    }
    else if ( tensorUsage == TensorUsage::OFM )
    {
        if ( npuOp == EthosU55NpuOp::Convolution || npuOp == EthosU55NpuOp::Depthwise ||
             npuOp == EthosU55NpuOp::Pooling || npuOp == EthosU55NpuOp::VectorProduct )
        {
            return opType != OpType::Rescale;
        }
        if ( npuOp == EthosU55NpuOp::Elementwise && globalScale )
        {
            if ( opType == OpType::Min || opType == OpType::Max || opType == OpType::Asr || opType == OpType::SHL ||
                 opType == OpType::CLZ || opType == OpType::LeakyRelu )
            {
                return false;
            }
            bool fromTypeSupported = (IsInteger(fromType) && (fromBits == 8 || fromBits == 16)) || fromType == DataType::Int32;
            if ( fromTypeSupported && fromType == DataType::Int32 )
            {
                return quantization.scales.front().scale == 1;  // Only shift supported for int32
            }
            return fromTypeSupported;
        }
    }

    return false;
}

bool EthosU55Constraints::SupportsGather(OpType opType)
{
    UNUSED(opType);
    return false;
}

bool EthosU55Constraints::SupportsScatter(OpType opType)
{
    UNUSED(opType);
    return false;
}
bool EthosU55Constraints::SupportsResize(const ResizeSupportQuery &query)
{
    UNUSED(query);
    return false;
}

bool EthosU55Constraints::SupportsSigmoidTanhLutInt16(OpType opType)
{
    UNUSED(opType);
    return false;
}

bool EthosU55Constraints::SupportsArgMax(OpType opType)
{
    UNUSED(opType);
    return false;
}

bool EthosU55Constraints::SupportsCast(OpType opType, DataType ifmType, DataType ofmType)
{
    UNUSED(opType);
    UNUSED(ifmType);
    UNUSED(ofmType);
    return false;
}

}  // namespace regor
