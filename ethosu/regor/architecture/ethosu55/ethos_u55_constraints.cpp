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

#include "ethos_u55_constraints.hpp"

#include "ethos_u55_register_cs_generator.hpp"

namespace regor
{

// Unsupported operators - must be sorted ascending
static constexpr OpType s_unsupportedU55[] = {
    OpType::None,
    OpType::ArgMax,
    OpType::Gather,
    OpType::Scatter,
    OpType::Resize,
    OpType::Cast,
};

static_assert(is_sorted(s_unsupportedU55), "list must be sorted");

// Short query
static constexpr std::pair<OpType, QueryResult> s_shortU55[] = {
    {OpType::Transpose, QueryResult::NativeConstrained},
};

static_assert(is_sorted(s_shortU55, [](const auto &a, const auto &b) { return a.first < b.first; }), "list must be sorted");


EthosU55Constraints::EthosU55Constraints(ArchEthosU55 *arch) : _arch(arch)
{
}

bool EthosU55Constraints::SupportsLeakyRelu(bool quantized, DataType type)
{
    return quantized == false && type == DataType::Int16;
}

bool EthosU55Constraints::SupportsMatMul(OpType opType)
{
    UNUSED(opType);
    return false;
}

TransposeSupport EthosU55Constraints::SupportsTranspose(OpType opType, TransposeType transposeType)
{
    if ( IsNone(transposeType) ) return TransposeSupport::Any;

    if ( opType == OpType::Transpose )
    {
        if ( transposeType == TransposeType::NWHC || transposeType == TransposeType::NHCW || transposeType == TransposeType::NCWH )
        {
            return TransposeSupport::NHWC;
        }
    }
    return TransposeSupport::None;
}

bool EthosU55Constraints::SupportsReverse(OpType opType, ReverseType reverseTypeMask)
{
    UNUSED(opType);
    return reverseTypeMask == ReverseType::None;
}

bool EthosU55Constraints::SupportsFusedRescale(OpType opType, TensorUsage tensorUsage, DataType rescaleFromType,
    DataType rescaleToType, DataType opFromType, DataType opToType, const Quantization &quantization)
{
    auto npuOp = ArchEthosU55::GetHWOp(opType);
    bool globalScale = quantization.scales.size() <= 1;
    bool isUnitScale = quantization.IsUnitScale();
    int64_t zp = quantization.zeroPoints.size() ? quantization.zeroPoints.front() : 0;

    if ( tensorUsage == TensorUsage::IFM )
    {
        int fromBits = DataTypeSizeBits(rescaleFromType);
        int toBits = DataTypeSizeBits(opToType);
        if ( npuOp == EthosU55NpuOp::Elementwise && globalScale )
        {
            bool fromTypeSupported = IsInteger(rescaleFromType) && (fromBits == 8 || fromBits == 16);
            bool toTypeSupported = (IsInteger(opToType) && (toBits == 8 || toBits == 16)) || opToType == DataType::Int32;

            // TODO MLBEDSW-10115: Support full 32-bit (advanced) rescale (with nonzero shift)
            // For now only allow 16-bit (simple) rescale
            auto &qs = quantization.scales.front();
            bool scaleSupported = qs.shift == 0 && static_cast<int16_t>(qs.scale) == qs.scale;

            // Make sure the rescale can be done without clipping
            int64_t value = (zp < 0 ? int64_t(IntegerMax(rescaleFromType)) : IntegerMin(rescaleFromType));
            value = value - zp;
            value = (value * qs.scale) >> qs.shift;
            bool noClipping = value >= IntegerMin(rescaleToType) && value <= int64_t(IntegerMax(rescaleToType));

            if ( opType == OpType::Add || opType == OpType::Sub )
            {
                return fromTypeSupported && toTypeSupported && scaleSupported && noClipping;
            }
            return fromTypeSupported && toTypeSupported && scaleSupported && noClipping && isUnitScale;
        }
        else if ( npuOp == EthosU55NpuOp::ReduceSum )
        {
            return globalScale && isUnitScale;
        }
    }
    else if ( tensorUsage == TensorUsage::OFM )
    {
        int fromBits = DataTypeSizeBits(opFromType);
        if ( npuOp == EthosU55NpuOp::Convolution || npuOp == EthosU55NpuOp::Depthwise ||
             npuOp == EthosU55NpuOp::Pooling || npuOp == EthosU55NpuOp::VectorProduct )
        {
            return opType != OpType::Rescale && !IsActivation(opType);
        }
        else if ( npuOp == EthosU55NpuOp::Elementwise && globalScale )
        {
            bool fromTypeSupported = (IsInteger(opFromType) && (fromBits == 8 || fromBits == 16)) || opFromType == DataType::Int32;
            if ( opFromType == DataType::Int32 )
            {
                // For 32-bit operations scale is not applied but shift is
                return quantization.scales.front().scale == 1;
            }
            if ( opType == OpType::Minimum || opType == OpType::Maximum || opType == OpType::Asr ||
                 opType == OpType::SHL || opType == OpType::CLZ || opType == OpType::LeakyRelu )
            {
                return fromTypeSupported && isUnitScale;
            }
            return fromTypeSupported;
        }
        else if ( npuOp == EthosU55NpuOp::ReduceSum )
        {
            return globalScale;
        }
    }

    return false;
}

bool EthosU55Constraints::SupportsRescale(DataType fromType, DataType toType)
{
    if ( DataTypeSizeBits(toType) > 16 )
    {
        return false;
    }
    if ( DataTypeSizeBits(fromType) > 16 )
    {
        return false;
    }
    return true;
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

bool EthosU55Constraints::SupportsNonMatchingShapes(const Shape &ifmShape, const Shape &ifm2Shape, const Shape &ofmShape)
{
    return (ifmShape == ofmShape) || (ifm2Shape && (ifm2Shape == ofmShape));
}


Flags<QueryResult> EthosU55Constraints::OperatorQuery(OpType opType, const ArchOperatorQuery *query, ArchRequirements *req)
{
    // Check unsupported operator list before further checks
    auto posUnsupported = std::equal_range(std::begin(s_unsupportedU55), std::end(s_unsupportedU55), opType);
    if ( posUnsupported.first != posUnsupported.second )
    {
        return QueryResult::Unsupported;
    }

    // Short query (no additional detail)
    if ( !query )
    {
        auto posShort = std::equal_range(std::begin(s_shortU55), std::end(s_shortU55),
            std::pair<OpType, QueryResult>{opType, {}}, [](const auto &a, const auto &b) { return a.first < b.first; });
        if ( posShort.first != posShort.second )
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

    // Reverse never supported
    if ( query->reverseMask != ReverseType::None )
    {
        return QueryResult::Unsupported;
    }

    // Detailed operator queries
    if ( !IsNone(query->transposeMask) )
    {
        if ( opType == OpType::Transpose )
        {
            if ( query->transposeMask == TransposeType::NWHC || query->transposeMask == TransposeType::NHCW ||
                 query->transposeMask == TransposeType::NCWH )
            {
                if ( req ) req->ofmFormat = TensorFormat::NHWC;
                return QueryResult::NativeConstrainedHasReq;
            }
        }
        return QueryResult::Unsupported;
    }

    if ( opType == OpType::MatMul )
    {
        if ( req )
        {
            req->req = ArchRequirement::ScratchTensor;
            req->scratch.size = query->ofm.shape;
            req->scratch.type = DataType::Int32;
            req->scratch.format = TensorFormat::NHWC;
        }
        return QueryResult::Unsupported;
    }
    else if ( (opType == OpType::Sigmoid) || (opType == OpType::Tanh) )
    {
        if ( query->ifm[0].type != DataType::Int16 )
        {
            if ( req )
            {
                req->req = ArchRequirement::OpSubstitution;
                req->substitution = OpType::LUT;
            }
            return QueryResult::NativeHasReq;
        }
    }
    return QueryResult::Native;
}


}  // namespace regor
