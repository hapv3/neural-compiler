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

#include <unordered_map>

namespace regor
{

// TODO: This table is from the EthosU55/U65 Embedded NPU Interface Specification, it's not completely valid for
// Ethos U85 since the allowed data types depend on ifm/ofm as well as selected acc and scaling.
static const std::unordered_map<EthosU85NpuOp, std::unordered_map<DataType, std::vector<DataType>>> s_opDataTypeSupport = {
    {EthosU85NpuOp::Convolution,
        {
            {DataType::UInt8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int16, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
        }},
    {EthosU85NpuOp::Depthwise,
        {
            {DataType::UInt8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int16, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
        }},
    {EthosU85NpuOp::VectorProduct,
        {
            {DataType::UInt8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int16, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
        }},
    {EthosU85NpuOp::Pooling,
        {
            {DataType::Bool8, {DataType::Bool8, DataType::Int32, DataType::Int64}},
            {DataType::UInt8, {DataType::UInt8, DataType::Int32, DataType::Int64}},
            {DataType::Int8, {DataType::Int8, DataType::Int32, DataType::Int64}},
            {DataType::Int16, {DataType::Int16}},
        }},
    {EthosU85NpuOp::ReduceMinMax,
        {
            {DataType::Bool8, {DataType::Bool8, DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::UInt8, {DataType::Bool8, DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int8, {DataType::Bool8, DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int16, {DataType::Bool8, DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int32, {DataType::Bool8, DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
        }},
    {EthosU85NpuOp::ReduceSum,
        {
            {DataType::UInt8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int16, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int32, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
        }},
    {EthosU85NpuOp::ArgMax,
        {
            {DataType::Bool8, {DataType::Int32, DataType::Int64}},
            {DataType::UInt8, {DataType::Int32, DataType::Int64}},
            {DataType::Int8, {DataType::Int32, DataType::Int64}},
            {DataType::Int16, {DataType::Int32, DataType::Int64}},
        }},
    {EthosU85NpuOp::Resize,
        {
            {DataType::UInt8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int16, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
        }},
};

TransposeSupport EthosU85Constraints::SupportsFusedTranspose(OpType opType, TransposeType transposeType)
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

bool EthosU85Constraints::SupportsFusedReverse(OpType opType, ReverseType reverseTypeMask)
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
        if ( npuOp == EthosU85NpuOp::Convolution || npuOp == EthosU85NpuOp::Depthwise || npuOp == EthosU85NpuOp::VectorProduct )
        {
            return true;
        }
        else if ( npuOp == EthosU85NpuOp::Pooling )
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
        else if ( npuOp == EthosU85NpuOp::Resize && globalScale )
        {
            auto &qs = quantization.scales.front();
            // Only shift < 48 supported
            const auto normalized = QuantizedScale::ReduceScale(qs);
            return normalized.scale == 1 && normalized.shift < 48;
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

bool EthosU85Constraints::SupportedDtypes(OpType opType, DataType ifmType, DataType ifm2Type, DataType ofmType)
{
    auto npuOp = _arch->GetHWOp(opType);
    if ( IsFloat(ifmType | ifm2Type | ofmType) )
    {
        return false;
    }

    if ( _arch->UseAvgPoolNop(opType) )
    {
        // TODO MLBEDSW-10667: The rules for UseAvgPoolNop are not the same as for a Pooling operation, skip checks for
        // now
        return true;
    }

    if ( npuOp != EthosU85NpuOp::Elementwise )
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
        auto &ofmTypes = ifmEntry->second;
        if ( 0 == std::count(ofmTypes.begin(), ofmTypes.end(), ofmType) )
        {
            // Unsupported ofm data type
            return false;
        }
    }
    else
    {
        // TODO elementwise
    }
    return true;
}

// Validate that zero-points are supported
bool EthosU85Constraints::SupportedZeroPoint(int64_t zp, TensorUsage usage, DataType dType, OpType opType)
{
    if ( IsIFM(usage) )
    {
        switch ( dType )
        {
            case DataType::Int8:
                return (zp >= -128) && (zp <= 127);
                break;
            case DataType::UInt8:
                return (zp >= 0) && (zp <= 255);
                break;
            case DataType::UInt16:
                return (zp == 0) || (zp == 32768);
                break;
            default:
                return zp == 0;
        }
    }
    else if ( IsOFM(usage) )
    {
        if ( IsSignedInteger(dType) )
        {
            return zp >= -128 && zp <= 127;
        }
        else
        {
            return (zp == 32768) || (zp >= 0 && zp <= 255);
        }
    }
    return true;
}

Flags<QueryResult> EthosU85Constraints::OperatorQuery(OpType opType, const ArchOperatorQuery *query, ArchRequirements *req)
{
    Flags<QueryResult> result = QueryResult::Native;
    static constexpr int32_t MAX_AXIS = (1 << 16);

    // Check hardware-required substitutions first
    if ( (opType == OpType::Sigmoid) || (opType == OpType::Tanh) )
    {
        if ( req )
        {
            req->req.Set(ArchRequirement::OpSubstitution);
            req->substitution = OpType::LUT;
        }
        result.Set(QueryResult::HasRequirements);
    }

    // TransposeConv2D and Conv3D are legalized during decomposition
    if ( opType == OpType::TransposeConv2D || opType == OpType::Conv3D )
    {
        if ( req )
        {
            req->req.Set(ArchRequirement::Decompose);
        }
        return QueryResult::NativeConstrainedHasReq;
    }

    // Check direct native support of the opType
    auto npuOp = _arch->GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None )
    {
        return QueryResult::Unsupported;
    }

    // Short query (no additional detail)
    if ( !query )
    {
        // more detailed query might fail
        return QueryResult::NativeConstrained;
    }

    // Fusing checks
    if ( query->transposeMask != TransposeType::None )
    {
        TransposeSupport tmp = SupportsFusedTranspose(opType, query->transposeMask);
        if ( tmp == TransposeSupport::None )
        {
            if ( opType == OpType::Transpose )
            {
                // unsupported mask for standalone transpose, requires decomposition
                if ( req )
                {
                    req->req.Set(ArchRequirement::Decompose);
                    req->decomposeProps.Set(ArchProperty::TransposeMask);
                }
                result.Set(QueryResult::HasRequirements);
            }
            else
            {
                // unsupported transpose-fusing
                return QueryResult::Unsupported;
            }
        }
    }
    if ( query->reverseMask != ReverseType::None )
    {
        if ( !SupportsFusedReverse(opType, query->reverseMask) )
        {
            return QueryResult::Unsupported;
        }
    }

    if ( npuOp == EthosU85NpuOp::Dma )
    {
        return result;
    }
    else if ( npuOp == EthosU85NpuOp::ReduceMinMax )
    {
        // unsupported reduce axis (only H and W supported)
        if ( query->axis != -3 /* H */ && query->axis != -2 /* W */ )
        {
            if ( req )
            {
                req->req.Set(ArchRequirement::Decompose);
                req->decomposeProps.Set(ArchProperty::ReduceAxis);
            }
            result.Set(QueryResult::HasRequirements);
        }
    }
    else if ( npuOp == EthosU85NpuOp::ReduceSum )
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

    const auto &ifmShape = query->ifm[0].shape;
    const auto &ifm2Shape = query->ifm[1].shape;
    const auto &ofmShape = query->ofm.shape;
    const auto ifmType = query->ifm[0].type;
    const auto ifm2Type = query->ifm[1].type;
    const auto ofmType = query->ofm.type;
    bool typeInfo = (ifmType != DataType::None && ofmType != DataType::None);
    bool shapeInfo = (ifmShape && ofmShape);

    if ( !typeInfo || !shapeInfo || !query->kernel )
    {
        // missing detail, more detailed queries might fail
        result.Set(QueryResult::Constrained);
    }

    // Validate zeroPoints
    if ( typeInfo )
    {
        for ( auto zp : query->ifm[0].quantization.zeroPoints )
        {
            if ( !SupportedZeroPoint(zp, TensorUsage::IFM0, ifmType, opType) )
            {
                return QueryResult::Unsupported;
            }
        }
        for ( auto zp : query->ifm[1].quantization.zeroPoints )
        {
            if ( !SupportedZeroPoint(zp, TensorUsage::IFM1, ifm2Type, opType) )
            {
                return QueryResult::Unsupported;
            }
        }
        for ( auto zp : query->ofm.quantization.zeroPoints )
        {
            if ( !SupportedZeroPoint(zp, TensorUsage::OFM, ofmType, opType) )
            {
                return QueryResult::Unsupported;
            }
        }
    }

    // Validate dataTypes
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
    if ( opType == OpType::MatMul )
    {
        // Constrain Matmul height to 1
        if ( ofmShape.Size() > 2 && ofmShape.Height() > 1 )
        {
            if ( req )
            {
                req->req.Set(ArchRequirement::Decompose);
                req->decomposeProps.Set(ArchProperty::TensorAxis);
            }
            result.Set(QueryResult::HasRequirements);
        }
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
    }

    return result;
}

}  // namespace regor
