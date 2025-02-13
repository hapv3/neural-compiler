//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

// Unsupported operators - must be sorted ascending
static constexpr OpType s_unsupportedU85[] = {OpType::None};

static_assert(is_sorted(s_unsupportedU85), "list must be sorted");


// Short query
static constexpr std::pair<OpType, QueryResult> s_shortU85[] = {
    {OpType::Transpose, QueryResult::Native},
};

static_assert(is_sorted(s_shortU85, [](const auto &a, const auto &b) { return a.first < b.first; }), "list must be sorted");


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

TransposeSupport EthosU85Constraints::SupportsTranspose(OpType opType, TransposeType transposeType)
{
    if ( transposeType == TransposeType::None ) return TransposeSupport::Any;

    EthosU85NpuOp npuOp = ArchEthosU85::GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None || npuOp == EthosU85NpuOp::Resize || npuOp == EthosU85NpuOp::Dma )
    {
        return TransposeSupport::None;
    }
    else if ( npuOp == EthosU85NpuOp::Elementwise )
    {
        if ( transposeType == TransposeType::None || transposeType == TransposeType::NHCW || transposeType == TransposeType::NCHW )
        {
            return TransposeSupport::Any;
        }

        return TransposeSupport::None;
    }

    if ( transposeType == TransposeType::None || transposeType == TransposeType::NWHC || transposeType == TransposeType::NHCW ||
         transposeType == TransposeType::NWCH || transposeType == TransposeType::NCHW || transposeType == TransposeType::NCWH )
        return TransposeSupport::Any;

    return TransposeSupport::None;
}

bool EthosU85Constraints::SupportsReverse(OpType opType, ReverseType reverseTypeMask)
{
    Flags<ReverseType> reverseMask(reverseTypeMask);
    // Do not support non-constant axes
    if ( reverseMask == ReverseType::Dynamic ) return false;

    // All Optypes support reverseType::None
    if ( reverseMask == ReverseType::None ) return true;

    EthosU85NpuOp npuOp = ArchEthosU85::GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None || npuOp == EthosU85NpuOp::Elementwise || npuOp == EthosU85NpuOp::Dma )
    {
        return false;
    }

    return true;
}

bool EthosU85Constraints::SupportsFusedRescale(OpType opType, TensorUsage tensorUsage, DataType rescaleFromType,
    DataType rescaleToType, DataType opFromType, DataType opToType, const Quantization &quantization)
{
    auto npuOp = ArchEthosU85::GetHWOp(opType);
    bool globalScale = quantization.scales.size() <= 1;
    bool isUnitScale = quantization.IsUnitScale();
    int64_t zp = quantization.zeroPoints.size() ? quantization.zeroPoints.front() : 0;

    if ( tensorUsage == TensorUsage::IFM )
    {
        int fromBits = DataTypeSizeBits(rescaleFromType);
        int toBits = DataTypeSizeBits(opToType);
        if ( npuOp == EthosU85NpuOp::Elementwise && globalScale )
        {
            bool fromTypeSupported = (IsInteger(rescaleFromType) && fromBits == 8) || rescaleFromType == DataType::Int16;
            bool toTypeSupported = (IsInteger(opToType) && (toBits == 8 || toBits == 16)) || opToType == DataType::Int32;

            auto &qs = quantization.scales.front();
            // Make sure shift is valid
            if ( qs.shift < 0 || qs.shift > 63 ) return false;
            // Make sure the rescale can be done without clipping
            int64_t value = (zp < 0 ? int64_t(IntegerMax(rescaleFromType)) : IntegerMin(rescaleFromType));
            value = value - zp;
            value = (value * qs.scale) >> qs.shift;
            bool noClipping = value >= IntegerMin(rescaleToType) && value <= int64_t(IntegerMax(rescaleToType));

            if ( opType == OpType::Div || opType == OpType::Mul )
            {
                return fromTypeSupported && toTypeSupported && noClipping && isUnitScale;
            }
            return fromTypeSupported && toTypeSupported && noClipping;
        }
        else if ( npuOp == EthosU85NpuOp::ReduceSum )
        {
            return globalScale && isUnitScale;
        }
    }
    else if ( tensorUsage == TensorUsage::OFM )
    {
        int fromBits = DataTypeSizeBits(opFromType);
        int toBits = DataTypeSizeBits(rescaleToType);
        if ( npuOp == EthosU85NpuOp::Convolution || npuOp == EthosU85NpuOp::Depthwise ||
             npuOp == EthosU85NpuOp::Pooling || npuOp == EthosU85NpuOp::VectorProduct )
        {
            return opType != OpType::Rescale && !IsActivation(opType);
        }
        else if ( npuOp == EthosU85NpuOp::Resize && globalScale )
        {
            auto &qs = quantization.scales.front();
            return qs.scale == 1 && qs.shift >= 16;  // Only shift of 16 or more supported
        }
        else if ( npuOp == EthosU85NpuOp::Elementwise && globalScale )
        {
            bool fromTypeSupported = (IsInteger(opFromType) && (fromBits == 8 || fromBits == 16)) || opFromType == DataType::Int32;
            if ( opType == OpType::Mul && fromTypeSupported && opFromType == DataType::Int32 )
            {
                return quantization.scales.front().scale == 1;  // Only shift supported
            }
            if ( opType == OpType::SHR || opType == OpType::SHL || opType == OpType::Asr || opType == OpType::Div )
            {
                return fromTypeSupported && isUnitScale;
            }
            return fromTypeSupported;
        }
        else if ( npuOp == EthosU85NpuOp::ReduceSum )
        {
            return globalScale;
        }
    }

    return false;
}

bool EthosU85Constraints::SupportsRescale(DataType fromType, DataType toType)
{
    UNUSED(toType);
    return fromType != DataType::UInt16;
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

    if ( n_h > 2048 )
    {
        LOG_WARN("Resize height scale numerator ({}) exceeds maximum size (2048).\n", n_h);
        return false;
    }
    if ( n_w > 2048 )
    {
        LOG_WARN("Resize width scale numerator ({}) exceeds maximum size (2048).\n", n_w);
        return false;
    }
    if ( query.offsetY >= n_h || query.offsetY < -n_h )
    {
        LOG_WARN("Resize height offset: {} is outside the valid range [-height_numerator, height_numerator) = [{}, {})\n",
            query.offsetY, -n_h, n_h);
        return false;
    }
    if ( query.offsetX >= n_w || query.offsetX < -n_w )
    {
        LOG_WARN("Resize width offset: {} is outside the valid range [-with_numerator, width_numerator) = [{}, {})\n",
            query.offsetX, -n_w, n_w);
        return false;
    }

    if ( query.mode == ArchResizeMode::Bilinear )
    {
        if ( d_w == 0 || d_h == 0 )
        {
            LOG_WARN("ResizeBilinear w/h divisors can't be zero\n");
            return false;
        }

        // Get scale fractions and verify that scale-factor is a power of two.
        if ( n_w % d_w != 0 )
        {
            LOG_WARN("ResizeBilinear width scale-factor is not an integer: {}/{}\n", n_w, d_w);
            return false;
        }
        if ( n_h % d_h != 0 )
        {
            LOG_WARN("ResizeBilinear height scale-factor is not an integer: {}/{}\n", n_h, d_h);
            return false;
        }
        int scale_w = n_w / d_w;
        int scale_h = n_h / d_h;
        if ( !IsPowerOfTwo(scale_w) )
        {
            LOG_WARN("ResizeBilinear width scale-factor is not a power of two: {}\n", double(n_w) / d_w);
            return false;
        }
        if ( !IsPowerOfTwo(scale_h) )
        {
            LOG_WARN("ResizeBilinear height scale-factor is not a power of two: {}\n", double(n_h) / d_h);
            return false;
        }
    }

    return true;
}

bool EthosU85Constraints::SupportsCast(OpType opType, DataType ifmType, DataType ofmType)
{
    return !IsFloat(ifmType | ofmType);
}

bool EthosU85Constraints::SupportsNonMatchingShapes(const Shape &ifmShape, const Shape &ifm2Shape, const Shape &ofmShape)
{
    return true;
}


Flags<QueryResult> EthosU85Constraints::OperatorQuery(OpType opType, const ArchOperatorQuery *query, ArchRequirements *req)
{
    // Check unsupported operator list first
    auto posUnsupported = std::equal_range(std::begin(s_unsupportedU85), std::end(s_unsupportedU85), opType);
    if ( posUnsupported.first != std::end(s_unsupportedU85) )
    {
        return QueryResult::Unsupported;
    }

    // Short query (no additional detail)
    if ( !query )
    {
        auto posShort = std::equal_range(std::begin(s_shortU85), std::end(s_shortU85),
            std::pair<OpType, QueryResult>{opType, {}}, [](const auto &a, const auto &b) { return a.first < b.first; });
        if ( posShort.first != std::end(s_shortU85) )
        {
            return posShort.first->second;
        }
        return QueryResult::Native;
    }

    // Float types always unsupported
    if ( (query->ifm[0].shape && IsFloat(query->ifm[0].type)) || (query->ifm[1].shape && IsFloat(query->ifm[1].type)) ||
         (query->ofm.shape && IsFloat(query->ofm.type)) )
    {
        return QueryResult::Unsupported;
    }

    if ( query->transposeMask != TransposeType::None )
    {
        TransposeSupport tmp = SupportsTranspose(opType, query->transposeMask);
        if ( tmp == TransposeSupport::None ) return QueryResult::Unsupported;
    }

    if ( query->reverseMask != ReverseType::None )
    {
        if ( !SupportsReverse(opType, query->reverseMask) ) return QueryResult::Unsupported;
    }

    // Operator specific
    if ( opType == OpType::Resize )
    {
        if ( !query->specific.resize.ifmShape ) return QueryResult::Unsupported;  // TODO: remove from ResizeQuery
        if ( !SupportsResize(query->specific.resize) ) return QueryResult::Unsupported;
    }
    else if ( (opType == OpType::Sigmoid) || (opType == OpType::Tanh) )
    {
        if ( req )
        {
            req->req = ArchRequirement::OpSubstitution;
            req->substitution = OpType::LUT;
        }
        return QueryResult::NativeHasReq;
    }

    return QueryResult::Native;
}

}  // namespace regor
