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

#include "ethos_u85_constraints.hpp"

#include "ethos_u85.hpp"
#include "ethos_u85_register_cs_generator.hpp"

namespace regor
{
bool EthosU85Constraints::SupportsLeakyRelu(bool /*quantized*/, DataType /*type*/)
{
    return true;
}

bool EthosU85Constraints::SupportsMatMul(OpType opType)
{
    EthosU85NpuOp npuOp = ArchEthosU85::GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None )
    {
        return false;
    }

    return true;
}

bool EthosU85Constraints::SupportsTransposeHW(OpType opType, TransposeType transposeType)
{
    if ( IsNone(transposeType) ) return true;

    EthosU85NpuOp npuOp = ArchEthosU85::GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None || npuOp == EthosU85NpuOp::Resize || npuOp == EthosU85NpuOp::Dma )
    {
        return false;
    }
    else if ( npuOp == EthosU85NpuOp::Elementwise )
    {
        return transposeType == TransposeType::NHWC || transposeType == TransposeType::NHCW || transposeType == TransposeType::NCHW;
    }

    return transposeType == TransposeType::NHWC || transposeType == TransposeType::NWHC || transposeType == TransposeType::NHCW ||
           transposeType == TransposeType::NWCH || transposeType == TransposeType::NCHW || transposeType == TransposeType::NCWH;
}

bool EthosU85Constraints::SupportsTranspose(OpType opType, TransposeType transposeType, Shape ifmShape)
{
    if ( SupportsTransposeHW(opType, transposeType) )
    {
        return true;
    }
    else
    {
        for ( int i = 0; ifmShape.Size(); i++ )
        {
            if ( ifmShape[i] == 1 )
            {
                return true;
            }
        }
        return false;
    }
}


bool EthosU85Constraints::SupportsReverse(OpType opType, ReverseType reverseType)
{
    if ( reverseType == ReverseType::None ) return true;

    EthosU85NpuOp npuOp = ArchEthosU85::GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None || npuOp == EthosU85NpuOp::Elementwise || npuOp == EthosU85NpuOp::Dma )
    {
        return false;
    }

    return reverseType == ReverseType::H || reverseType == ReverseType::W || reverseType == ReverseType::C;
}

bool EthosU85Constraints::SupportsFusedRescale(
    OpType opType, TensorUsage tensorUsage, DataType fromType, DataType toType, const Quantization &quantization)
{
    auto npuOp = ArchEthosU85::GetHWOp(opType);
    bool globalScale = quantization.scales.size() == 1;
    int fromBits = DataTypeSizeBits(fromType);
    int toBits = DataTypeSizeBits(toType);

    if ( tensorUsage == TensorUsage::IFM )
    {
        if ( npuOp == EthosU85NpuOp::Elementwise )
        {
            bool fromTypeSupported = (IsInteger(fromType) && fromBits == 8) || fromType == DataType::Int16;
            bool toTypeSupported = (IsInteger(toType) && (toBits == 8 || toBits == 16)) || toType == DataType::Int32;
            return opType != OpType::Div && opType != OpType::Mul && globalScale && fromTypeSupported && toTypeSupported;
        }
    }
    else if ( tensorUsage == TensorUsage::OFM )
    {
        if ( npuOp == EthosU85NpuOp::Convolution || npuOp == EthosU85NpuOp::Depthwise ||
             npuOp == EthosU85NpuOp::Pooling || npuOp == EthosU85NpuOp::VectorProduct )
        {
            return opType != OpType::Rescale;
        }
        if ( npuOp == EthosU85NpuOp::Resize && globalScale )
        {
            auto &qs = quantization.scales.front();
            return qs.scale == 1 && qs.shift >= 16;  // Only shift of 16 or more supported
        }
        if ( npuOp == EthosU85NpuOp::Elementwise && globalScale )
        {
            if ( opType == OpType::SHR || opType == OpType::SHL || opType == OpType::Asr || opType == OpType::Div )
            {
                return false;
            }
            bool fromTypeSupported = (IsInteger(fromType) && (fromBits == 8 || fromBits == 16)) || fromType == DataType::Int32;
            if ( opType == OpType::Mul && fromTypeSupported && fromType == DataType::Int32 )
            {
                return quantization.scales.front().scale == 1;  // Only shift supported
            }
            return fromTypeSupported;
        }
    }

    return false;
}

bool EthosU85Constraints::SupportsGather(OpType opType)
{
    EthosU85NpuOp npuOp = ArchEthosU85::GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None )
    {
        return false;
    }

    return true;
}

bool EthosU85Constraints::SupportsScatter(OpType opType)
{
    EthosU85NpuOp npuOp = ArchEthosU85::GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None )
    {
        return false;
    }

    return true;
}

bool EthosU85Constraints::SupportsSigmoidTanhLutInt16(OpType opType)
{
    return (opType == OpType::Sigmoid || opType == OpType::Tanh);
}

bool EthosU85Constraints::SupportsArgMax(OpType opType)
{
    EthosU85NpuOp npuOp = ArchEthosU85::GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None )
    {
        return false;
    }

    return true;
}

bool EthosU85Constraints::SupportsResize(const ResizeSupportQuery &query)
{
    /* Supported operator checks for resize operations
     *
     *  * Scaling numerators must be less than or equal to 2048
     *  * Offsets must be in the range [-numerator, numerator) for each axis
     *  * The following constraints apply to upscale-factors
     *    mode REPLICATE:
     *      Any width and height upscale-factors are supported
     *    mode NEAREST:
     *      Any width and height upscale-factors are supported
     *    mode BILINEAR:
     *      if IFM W*H == 1*1:
     *        Any width and height upscale-factors are supported
     *      else:
     *        The upscale-factors need to be powers-of-two.
     */
    if ( query.ifmShape.Width() == 1 && query.ifmShape.Height() == 1 )
    {
        return true;
    }

    int n_w = query.scaleX.n;
    int d_w = query.scaleX.d;
    int n_h = query.scaleY.n;
    int d_h = query.scaleY.d;
    bool supported = true;

    if ( n_h > 2048 )
    {
        LOG_WARN("Resize height scale numerator ({}) exceeds maximum size (2048).\n", n_h);
        supported = false;
    }
    if ( n_w > 2048 )
    {
        LOG_WARN("Resize width scale numerator ({}) exceeds maximum size (2048).\n", n_w);
        supported = false;
    }
    if ( query.offsetY >= n_h || query.offsetY < -n_h )
    {
        LOG_WARN("Resize height offset: {} is outside the valid range [-height_numerator, height_numerator) = [{}, {})\n",
            query.offsetY, -n_h, n_h);
        supported = false;
    }
    if ( query.offsetX >= n_w || query.offsetX < -n_w )
    {
        LOG_WARN("Resize width offset: {} is outside the valid range [-with_numerator, width_numerator) = [{}, {})\n",
            query.offsetX, -n_w, n_w);
        supported = false;
    }

    if ( query.mode == ArchResizeMode::Bilinear )
    {
        // Get scale fractions and verify that scale-factor is a power of two.

        if ( n_w % d_w != 0 )
        {
            LOG_WARN("ResizeBilinear width scale-factor is not an integer: {}/{}\n", n_w, d_w);
            supported = false;
        }
        if ( n_h % d_h != 0 )
        {
            LOG_WARN("ResizeBilinear height scale-factor is not an integer: {}/{}\n", n_h, d_h);
            supported = false;
        }
        int scale_w = n_w / d_w;
        int scale_h = n_h / d_h;
        if ( !IsPowerOfTwo(scale_w) )
        {
            LOG_WARN("ResizeBilinear width scale-factor is not a power of two: {}\n", double(n_w) / d_w);
            supported = false;
        }
        if ( !IsPowerOfTwo(scale_h) )
        {
            LOG_WARN("ResizeBilinear height scale-factor is not a power of two: {}\n", double(n_h) / d_h);
            supported = false;
        }
        return supported;
    }
    return supported;
}

bool EthosU85Constraints::SupportsCast(OpType opType, DataType ifmType, DataType ofmType)
{
    return !IsFloat(ifmType | ofmType);
}

}  // namespace regor
