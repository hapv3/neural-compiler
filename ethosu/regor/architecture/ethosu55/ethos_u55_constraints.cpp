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

// Table of allowed ifm/ofm data type combinations for each HWOp
static const std::array<DataType, 4> s_defaultAllTypes = {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32};
static const std::array<DataType, 1> s_defaultUInt8Only = {DataType::UInt8};
static const std::array<DataType, 1> s_defaultInt8Only = {DataType::Int8};
static const std::array<DataType, 1> s_defaultInt16Only = {DataType::Int16};

static const std::unordered_map<EthosU55NpuOp, std::unordered_map<DataType, readonly_span_t<DataType>>> s_opDataTypeSupport = {
    {EthosU55NpuOp::Convolution,  // HWOp
        {
            // IFM data type  | OFM data type(s)
            {DataType::UInt8, s_defaultAllTypes},
            {DataType::Int8, s_defaultAllTypes},
            {DataType::Int16, s_defaultAllTypes},
        }},
    {EthosU55NpuOp::Depthwise,
        {
            {DataType::UInt8, s_defaultAllTypes},
            {DataType::Int8, s_defaultAllTypes},
            {DataType::Int16, s_defaultAllTypes},
        }},
    {EthosU55NpuOp::VectorProduct,
        {
            {DataType::UInt8, s_defaultAllTypes},
            {DataType::Int8, s_defaultAllTypes},
            {DataType::Int16, s_defaultAllTypes},
        }},
    {EthosU55NpuOp::Pooling,
        {
            {DataType::UInt8, s_defaultUInt8Only},
            {DataType::Int8, s_defaultInt8Only},
            {DataType::Int16, s_defaultInt16Only},
        }},
    {EthosU55NpuOp::ReduceSum,
        {
            {DataType::UInt8, s_defaultAllTypes},
            {DataType::Int8, s_defaultAllTypes},
            {DataType::Int16, s_defaultAllTypes},
            {DataType::Int32, s_defaultAllTypes},
        }},
};

EthosU55Constraints::EthosU55Constraints(ArchEthosU55 *arch) : _arch(arch)
{
}

bool EthosU55Constraints::SupportsElementwiseLeakyRelu(bool quantized, DataType type)
{
    return quantized == false && type == DataType::Int16;
}

TransposeSupport EthosU55Constraints::SupportsFusedTranspose(OpType opType, TransposeType transposeType)
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

bool EthosU55Constraints::SupportsFusedReverse(OpType opType, ReverseType reverseTypeMask)
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
        if ( npuOp == EthosU55NpuOp::Convolution || npuOp == EthosU55NpuOp::Depthwise || npuOp == EthosU55NpuOp::VectorProduct )
        {
            return true;
        }
        else if ( npuOp == EthosU55NpuOp::Pooling )
        {
            if ( opType == OpType::AvgPool )
            {
                return globalScale && isUnitScale;
            }
            else
            {
                return opType != OpType::Rescale && !IsActivation(opType);
            }
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


static const std::array<DataType, 5> s_validAddOfmTypes = {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64};
static const std::array<DataType, 4> s_validAddMulTypes = {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32};
static const std::array<DataType, 4> s_validMaxAbsTypes = {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32};
static const std::array<DataType, 1> s_validClzShlTypes = {DataType::Int32};
static const std::array<DataType, 4> s_validAsrOfmTypes = {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32};
static const std::array<DataType, 3> s_validReverseTypes = {DataType::UInt8, DataType::Int8, DataType::Int16};



bool EthosU55Constraints::SupportedDtypes(OpType opType, DataType ifmType, DataType ifm2Type, DataType ofmType)
{
    auto npuOp = _arch->GetHWOp(opType);
    if ( IsFloat(ifmType | ifm2Type | ofmType) )
    {
        return false;
    }

    if ( _arch->UseAvgPoolNop(opType) || opType == OpType::Rescale )
    {
        // TODO MLBEDSW-10667: The rules for UseAvgPoolNop are not the same as for a Pooling operation, so skip checks
        // for now
        return true;
    }

    if ( npuOp == EthosU55NpuOp::Compound || npuOp == EthosU55NpuOp::Dma )
    {
        return true;
    }

    readonly_span_t<DataType> ofmTypes;

    // Check allowed ifm/ofm type mapping
    if ( npuOp != EthosU55NpuOp::Elementwise )
    {
        auto map = s_opDataTypeSupport.find(npuOp);
        if ( map == s_opDataTypeSupport.end() )
        {
            assert(false && "Data type mapping for HWOp missing");
            return false;
        }
        auto &typeMap = map->second;
        auto ifmEntry = typeMap.find(ifmType);
        if ( ifmEntry == typeMap.end() )
        {
            // Unsupported ifm data type
            return false;
        }
        ofmTypes = ifmEntry->second;
    }
    else
    {
        readonly_span_t<DataType> ifmTypes;
        switch ( opType )
        {
            case OpType::Add:
            {
                ifmTypes = s_validAddMulTypes;
                ofmTypes = s_validAddOfmTypes;
            }
            break;
            case OpType::Sub:
            case OpType::Mul:
            {
                ifmTypes = s_validAddMulTypes;
                ofmTypes = s_validAddMulTypes;
            }
            break;
            case OpType::Minimum:
            case OpType::Maximum:
            case OpType::LeakyRelu:
            case OpType::Abs:
            {
                ifmTypes = s_validMaxAbsTypes;
                ofmTypes = s_validMaxAbsTypes;
            }
            break;
            case OpType::CLZ:
            case OpType::SHL:
            {
                ifmTypes = s_validClzShlTypes;
                ofmTypes = s_validClzShlTypes;
            }
            break;
            case OpType::Asr:
            {
                ifmTypes = s_validClzShlTypes;
                ofmTypes = s_validAsrOfmTypes;
            }
            break;
            case OpType::Reverse:
            {
                ifmTypes = s_validReverseTypes;
                ofmTypes = s_validReverseTypes;
            }
            break;
            default:
                assert(false && "Unkown elementwise type");
                break;
        }
        if ( !std::any_of(ifmTypes.begin(), ifmTypes.end(), [&](auto t) { return t == ifmType; }) )
        {
            // Unsupported ifm data type
            return false;
        }
        if ( IsBinaryElementwise(opType) && ifm2Type != ifmType )
        {
            // ifm2 data type must match ifm data type
            return false;
        }
    }

    if ( !std::any_of(ofmTypes.begin(), ofmTypes.end(), [&](auto t) { return t == ofmType; }) )
    {  // Unsupported ofm data type
        return false;
    }

    return true;
}

// Validate that zero-points are supported
bool EthosU55Constraints::SupportedZeroPoint(int64_t zp, TensorUsage usage, DataType dType, OpType opType)
{
    if ( !IsSignedInteger(dType) && zp < 0 )
    {
        // must be non-negative for unsigned data types
        return false;
    }

    if ( IsIFM(usage) )
    {
        // must be zero for 32-bit IFM and for CLZ or SHL operations
        if ( DataTypeSizeBits(dType) == 32 || opType == OpType::CLZ || opType == OpType::SHL )
        {
            return zp == 0;
        }
    }
    else if ( IsOFM(usage) )
    {
        // must be zero for CLZ or SHL operations
        if ( opType == OpType::CLZ || opType == OpType::SHL )
        {
            return zp == 0;
        }
        // must be zero for 32-bit OFM unless op is an activation
        if ( DataTypeSizeBits(dType) == 32 && !IsActivation(opType) )
        {
            return zp == 0;
        }
    }
    return true;
}

namespace
{

thread_local std::array<ArchTensorRequirement, 4> s_extraTensorReq;

ArchTensorRequirement *NextTensor(ArchTensorRequirement *tr, unsigned &used)
{
    assert(used < s_extraTensorReq.size());
    ArchTensorRequirement *info = &s_extraTensorReq[used++];
    tr->next = info;
    return info;
};

}  // namespace

Flags<QueryResult> EthosU55Constraints::OperatorQuery(OpType opType, const ArchOperatorQuery *query, ArchRequirements *req)
{
    static constexpr int32_t MAX_AXIS = (1 << 16);

    unsigned usedTensors = 0;
    if ( opType == OpType::Resize )
    {
        if ( query->ifm[0].shape.ElementsWH() == 1 )
        {
            return QueryResult::Unsupported;
        }
        if ( req )
        {
            req->req.Set(ArchRequirement::OpSubstitution, ArchRequirement::Decompose);
        }
        return QueryResult::NativeHasReq;
    }
    // TransposeConv2D and Conv3D are legalized during decomposition
    else if ( opType == OpType::TransposeConv2D || opType == OpType::Conv3D )
    {
        // Check for supported weight format
        if ( query && query->weightFormat != WeightFormat::Default )
        {
            return QueryResult::Unsupported;
        }
        if ( req )
        {
            req->req.Set(ArchRequirement::Decompose);
        }
        return QueryResult::NativeConstrainedHasReq;
    }

    // Check direct native support of the opType
    auto npuOp = _arch->GetHWOp(opType);
    if ( npuOp == EthosU55NpuOp::None )
    {
        return QueryResult::Unsupported;
    }
    else if ( npuOp == EthosU55NpuOp::Dma )
    {
        return QueryResult::Native;
    }

    // Short query (no additional detail)
    if ( !query )
    {
        // More detailed query might fail (constrained)
        return QueryResult::NativeConstrained;
    }

    // Check for supported weight format for convolution type ops
    if ( opType == OpType::DepthwiseConv2D || opType == OpType::Conv2D || opType == OpType::FullyConnected )
    {
        if ( query->weightFormat != WeightFormat::Default )
        {
            return QueryResult::Unsupported;
        }
    }

    Flags<QueryResult> result = QueryResult::Native;

    if ( npuOp == EthosU55NpuOp::ReduceSum )
    {
        // unsupported reduce axis (only C supported)
        if ( query->axis != -1 /* C */ )
        {
            if ( req )
            {
                req->req.Set(ArchRequirement::Decompose);
                req->decomposeProps.Set(ArchProperty::ReduceAxis);
            }
            result.Set(QueryResult::HasRequirements);
        }
    }
    // Check required substitutions first
    else if ( (opType == OpType::Sigmoid) || (opType == OpType::Tanh) )
    {
        if ( query->ifm[0].type != DataType::Int16 )
        {
            if ( req )
            {
                req->req.Set(ArchRequirement::OpSubstitution);
                req->substitution = OpType::LUT;
            }
            result.Set(QueryResult::HasRequirements);
        }
    }

    const auto &ifmShape = query->ifm[0].shape;
    const auto &ifm2Shape = query->ifm[1].shape;
    const auto &ofmShape = query->ofm.shape;
    auto ifmType = query->ifm[0].type;
    auto ifm2Type = query->ifm[1].type;
    auto ofmType = query->ofm.type;
    bool typeInfo = (ifmType != DataType::None && ofmType != DataType::None);
    bool shapeInfo = (ifmShape && ofmShape);

    if ( !typeInfo || !shapeInfo || !query->kernel )
    {
        // missing detail, more detailed queries might fail
        result.Set(QueryResult::Constrained);
    }

    // Validate DataTypes
    if ( typeInfo && !SupportedDtypes(opType, query->ifm[0].type, query->ifm[1].type, query->ofm.type) )
    {
        return QueryResult::Unsupported;
    }

    // Validate tensor-shapes
    if ( shapeInfo )
    {
        for ( const auto &s : {ifmShape, ifm2Shape, ofmShape} )
        {
            if ( !s ) continue;
            auto shape = Shape::PadAxes(s, 4, 1);
            // validate that leading dimensions are unit
            for ( int i = 0; i < shape.Size() - 3; i++ )
            {
                if ( shape[i] > 1 )
                {
                    if ( req )
                    {
                        req->req.Set(ArchRequirement::Decompose);
                        req->decomposeProps.Set(ArchProperty::TensorDims);
                    }
                    result.Set(QueryResult::HasRequirements);
                }
            }
            // validate that HWC are within valid range
            for ( int i = shape.Size() - 3; i < shape.Size(); i++ )
            {
                if ( shape[i] > MAX_AXIS )
                {
                    if ( req )
                    {
                        req->req.Set(ArchRequirement::Decompose);
                        req->decomposeProps.Set(ArchProperty::TensorAxis);
                    }
                    result.Set(QueryResult::HasRequirements);
                }
            }
        }
    }

    // Detailed operator queries
    if ( opType == OpType::Transpose )
    {
        // TODO MLBEDSW-10668: Transpose-implementation does not support large-axis decomposition
        if ( req && req->decomposeProps.Any(ArchProperty::TensorAxis) )
        {
            return QueryResult::Unsupported;
        }
        // TODO MLBEDSW-10668: channel-axis for 32-bit NWHC-transpose is constrained to 14-bits
        static constexpr int TRANSPOSE_32_MAX_CHANNEL = (1 << 14);
        if ( (shapeInfo && typeInfo) && (query->ifm[0].type == DataType::Int32) &&
             (query->transposeMask == TransposeType::NWHC) && (query->ifm[0].shape.Depth() > TRANSPOSE_32_MAX_CHANNEL) )
        {
            return QueryResult::Unsupported;
        }
        // Validate supported transpose-masks
        if ( query->transposeMask != TransposeType::NWHC && query->transposeMask != TransposeType::NHCW && query->transposeMask != TransposeType::NCWH )
        {
            // supported with mask-decomposition requirements
            if ( req )
            {
                req->req.Set(ArchRequirement::Decompose);
                req->decomposeProps.Set(ArchProperty::TransposeMask);
            }
            result.Set(QueryResult::HasRequirements);
        }
        if ( req )
        {
            req->req.Set(ArchRequirement::Tensor);
            Set(req->tensor, TensorUsage::IFM, TensorFormat::NHWC);
            Set(*NextTensor(&req->tensor, usedTensors), TensorUsage::OFM, TensorFormat::NHWC);
        }
        return result;
    }
    else
    {
        if ( !IsNone(query->transposeMask) )
        {
            return QueryResult::Unsupported;
        }
    }

    // reverseType::W and reverseType::H are supported
    if ( Flags<ReverseType>(query->reverseMask).Unset(ReverseType::H, ReverseType::W) != ReverseType::None )
    {
        return QueryResult::Unsupported;
    }

    // Validate zeroPoints
    if ( typeInfo )
    {
        if ( query->ifm[0].quantization )
        {
            for ( auto zp : query->ifm[0].quantization->zeroPoints )
            {
                if ( !SupportedZeroPoint(zp, TensorUsage::IFM0, ifmType, opType) )
                {
                    return QueryResult::Unsupported;
                }
            }
        }
        if ( query->ifm[1].quantization )
        {
            for ( auto zp : query->ifm[1].quantization->zeroPoints )
            {
                if ( !SupportedZeroPoint(zp, TensorUsage::IFM1, ifm2Type, opType) )
                {
                    return QueryResult::Unsupported;
                }
            }
        }
        if ( query->ofm.quantization )
        {
            for ( auto zp : query->ofm.quantization->zeroPoints )
            {
                if ( !SupportedZeroPoint(zp, TensorUsage::OFM, ofmType, opType) )
                {
                    return QueryResult::Unsupported;
                }
            }
        }
    }

    if ( opType == OpType::MatMul )
    {
        if ( req )
        {
            req->req.Set(ArchRequirement::Tensor);
            ArchTensorRequirement *tr = &req->tensor;
            if ( query->ifm[0].shape )
            {
                req->req.Set(ArchRequirement::Tensor);
                Set(*tr, TensorUsage::Scratch, DataType::Int32, TensorFormat::NHWC,
                    query->ifm[0].shape.WithDepth(query->ifm[0].shape.Depth() + 1));
                tr = NextTensor(tr, usedTensors);
            }
            Set(*tr, TensorUsage::IFM1, TensorFormat::NHWC);
            tr = NextTensor(tr, usedTensors);
            Set(*tr, TensorUsage::OFM, TensorFormat::NHWC);
        }
        result.Set(QueryResult::HasRequirements);
    }

    // kernel constraint-checks
    if ( query->kernel )
    {
        auto k = query->kernel;
        if ( k->Stride().x > 3 || k->Stride().y > 3 )
        {
            if ( req )
            {
                req->req.Set(ArchRequirement::Decompose);
                req->decomposeProps.Set(ArchProperty::KernelStride);
            }
            result.Set(QueryResult::HasRequirements);
        }

        if ( k->Dilation().x > 2 || k->Dilation().y > 2 )
        {
            if ( req )
            {
                req->req.Set(ArchRequirement::Decompose);
                req->decomposeProps.Set(ArchProperty::KernelDilation);
            }
            result.Set(QueryResult::HasRequirements);
        }

        if ( k->DepthMultiplier() > 1 )
        {
            if ( req )
            {
                req->req.Set(ArchRequirement::Decompose);
                req->decomposeProps.Set(ArchProperty::DepthMultiplier);
            }
            result.Set(QueryResult::HasRequirements);
        }

        if ( opType == OpType::AvgPool && (k->Size().x > 8 || k->Size().y > 8) && !k->Padding().IsZero() )
        {
            return QueryResult::Unsupported;
        }
    }
    else
    {
        // no kernel provided, more detailed queries might fail
        result.Set(QueryResult::Constrained);
    }

    return result;
}

}  // namespace regor
