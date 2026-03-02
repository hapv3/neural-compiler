//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

// Table of allowed ifm/ofm data type combinations for each HWOp
static const std::array<DataType, 7> s_defaultAllTypes = {DataType::Bool8, DataType::UInt8, DataType::Int8,
    DataType::UInt16, DataType::Int16, DataType::Int32, DataType::Int64};
static const std::array<DataType, 6> s_defaultIntegerTypes = {
    DataType::UInt8, DataType::Int8, DataType::UInt16, DataType::Int16, DataType::Int32, DataType::Int64};
static const std::array<DataType, 5> s_reduceMinMaxTypes = {
    DataType::Bool8, DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32};
static const std::array<DataType, 2> s_defaultInt32_64 = {DataType::Int32, DataType::Int64};

// TODO: This table is from the EthosU55/U65 Embedded NPU Interface Specification, it's not completely valid for
// Ethos U85 since the allowed data types depend on ifm/ofm as well as selected acc and scaling.
static const std::unordered_map<EthosU85NpuOp, std::unordered_map<DataType, readonly_span_t<DataType>>> s_opDataTypeSupport = {
    {EthosU85NpuOp::Convolution,
        {
            {DataType::UInt8, s_defaultIntegerTypes},
            {DataType::Int8, s_defaultIntegerTypes},
            {DataType::Int16, s_defaultIntegerTypes},
        }},
    {EthosU85NpuOp::Depthwise,
        {
            {DataType::UInt8, s_defaultIntegerTypes},
            {DataType::Int8, s_defaultIntegerTypes},
            {DataType::Int16, s_defaultIntegerTypes},
        }},
    {EthosU85NpuOp::VectorProduct,
        {
            {DataType::UInt8, s_defaultIntegerTypes},
            {DataType::Int8, s_defaultIntegerTypes},
            {DataType::Int16, s_defaultIntegerTypes},
        }},
    {EthosU85NpuOp::Pooling,
        {
            {DataType::Bool8, s_defaultAllTypes},
            {DataType::UInt8, s_defaultAllTypes},
            {DataType::Int8, s_defaultAllTypes},
            {DataType::Int16, s_defaultAllTypes},
        }},
    {EthosU85NpuOp::ReduceMinMax,
        {
            {DataType::Bool8, s_reduceMinMaxTypes},
            {DataType::UInt8, s_reduceMinMaxTypes},
            {DataType::Int8, s_reduceMinMaxTypes},
            {DataType::Int16, s_reduceMinMaxTypes},
            {DataType::Int32, s_reduceMinMaxTypes},
        }},
    {EthosU85NpuOp::ReduceSum,
        {
            {DataType::UInt8, s_defaultIntegerTypes},
            {DataType::Int8, s_defaultIntegerTypes},
            {DataType::Int16, s_defaultIntegerTypes},
            {DataType::Int32, s_defaultIntegerTypes},
        }},
    {EthosU85NpuOp::ArgMax,
        {
            {DataType::Bool8, s_defaultInt32_64},
            {DataType::UInt8, s_defaultInt32_64},
            {DataType::Int8, s_defaultInt32_64},
            {DataType::Int16, s_defaultInt32_64},
        }},
    {EthosU85NpuOp::Resize,
        {
            {DataType::UInt8, s_defaultIntegerTypes},
            {DataType::Int8, s_defaultIntegerTypes},
            {DataType::Int16, s_defaultIntegerTypes},
        }},
    {EthosU85NpuOp::Branch,
        {
            {DataType::Bool8, s_defaultAllTypes},
            {DataType::UInt8, s_defaultAllTypes},
            {DataType::Int8, s_defaultAllTypes},
            {DataType::Int16, s_defaultAllTypes},
            {DataType::Int32, s_defaultAllTypes},
        }},
};

TransposeSupport EthosU85Constraints::SupportsFusedTranspose(OpType opType, TransposeType transposeType)
{
    if ( transposeType == TransposeType::None ) return TransposeSupport::Any;

    EthosU85NpuOp npuOp = ArchEthosU85::GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None || npuOp == EthosU85NpuOp::Resize || npuOp == EthosU85NpuOp::Dma || npuOp == EthosU85NpuOp::Branch )
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
    if ( npuOp == EthosU85NpuOp::None || npuOp == EthosU85NpuOp::Elementwise || npuOp == EthosU85NpuOp::Dma || npuOp == EthosU85NpuOp::Branch )
    {
        return false;
    }

    return true;
}

// Helpers for SupportsQuantization
namespace
{
// Check that IFM-scales are supported
bool SupportedIFMQuant(OpType opType, EthosU85NpuOp npuOp, const Quantization &ifmQuant, const Quantization &ifm2Quant, DataType ifmType)
{
    // Per-channel IFM scaling is not supported
    if ( ifmQuant.scales.size() > 1 || ifm2Quant.scales.size() > 1 )
    {
        return false;
    }
    const auto &ifmScale = ifmQuant.Scale();
    const auto &ifm2Scale = ifm2Quant.Scale();
    auto ifmBits = DataTypeSizeBits(ifmType);
    if ( npuOp == EthosU85NpuOp::Elementwise )
    {
        if ( opType == OpType::Div || opType == OpType::Mul )
        {
            // Div and Mul don't support input-scaling
            return ifmScale == QuantizedScale::Unit() && ifm2Scale == QuantizedScale::Unit();
        }
        else if ( ifmBits == 8 || ifmType == DataType::Int16 )
        {
            // non-unit IFM-scaling is supported for int8, uint8, and int16(signed)
            return true;
        }
        else if ( IsBinaryElementwise(opType) )
        {
            return ifmScale == QuantizedScale::Unit() && ifm2Scale == QuantizedScale::Unit();
        }
    }
    return ifmScale == QuantizedScale::Unit();
}
// Check that OFM-scales are supported
bool SupportedOFMQuant(OpType opType, EthosU85NpuOp npuOp, const Quantization &ofmQuant, DataType ifmType)
{
    // Reject per-channel scaling for opTypes that do not support it
    if ( IsElementwise(opType) || opType == OpType::ReduceSum || opType == OpType::AvgPool || opType == OpType::Table || npuOp == EthosU85NpuOp::Resize )
    {
        if ( ofmQuant.scales.size() > 1 )
        {
            return false;
        }
    }
    const auto &ofmScale = ofmQuant.Scale();
    // The hardware natively supports tables with an output shift of 7 as one
    // activation/operation
    if ( opType == OpType::Table )
    {
        return ofmScale == QuantizedScale(1, 7);
    }
    // Activations cannot be fused to
    else if ( IsActivation(opType) )
    {
        return false;
    }
    if ( npuOp == EthosU85NpuOp::Convolution || npuOp == EthosU85NpuOp::Depthwise || npuOp == EthosU85NpuOp::VectorProduct ||
         npuOp == EthosU85NpuOp::ReduceSum || (npuOp == EthosU85NpuOp::Pooling && opType != OpType::AvgPool) )
    {
        return true;
    }
    else if ( npuOp == EthosU85NpuOp::Resize )
    {
        // only shift < 48 is supported
        const auto normalized = ofmScale.ReduceScale(ofmScale);
        return normalized.scale == 1 && normalized.shift < 48;
    }
    else if ( IsElementwise(opType) )
    {
        if ( opType == OpType::Mul && ifmType == DataType::Int32 )
        {
            // Mul with 32-bit IFM supports shift-only
            return ofmScale.scale == 1;
        }
        if ( opType == OpType::SHR || opType == OpType::SHL || opType == OpType::Asr || opType == OpType::Div )
        {
            return ofmScale == QuantizedScale::Unit();
        }
        return true;
    }
    return ofmScale == QuantizedScale::Unit();
}
}  // namespace


// Check if explicit quantization is supported by opType
bool EthosU85Constraints::SupportsQuantization(OpType opType, const Quantization &ifmQuant, DataType ifmType,
    const Quantization &ifm2Quant, DataType ifm2Type, const Quantization &ofmQuant, DataType ofmType)
{
    auto npuOp = ArchEthosU85::GetHWOp(opType);
    // This function assumes EXPLICIT quantization type
    // Other QuantizationTypes must be converted before validation
    assert(ifmQuant.type == QuantizationType::EXPLICIT);
    assert(ifm2Quant.type == QuantizationType::EXPLICIT);
    assert(ofmQuant.type == QuantizationType::EXPLICIT);
    // DMA/Branch operations do not support quantization
    if ( npuOp == EthosU85NpuOp::Dma || npuOp == EthosU85NpuOp::Branch )
    {
        return false;
    }
    // Validate that quantization is valid
    if ( !SupportedIFMQuant(opType, npuOp, ifmQuant, ifm2Quant, ifmType) )
    {
        return false;
    }
    if ( !SupportedOFMQuant(opType, npuOp, ofmQuant, ifmType) )
    {
        return false;
    }
    // Check that dataTypes are supported
    if ( npuOp != EthosU85NpuOp::None )
    {
        if ( !SupportedDtypes(opType, ifmType, ifm2Type, ofmType) )
        {
            return false;
        }
    }
    else
    {
        // Explicit Dtype checks for non-optimised opTypes
        if ( opType == OpType::Table )
        {
            // The hardware natively supports tables with an output shift of 7 as one
            // activation/operation
            if ( ifmType != DataType::Int16 || ofmType != DataType::Int16 )
            {
                return false;
            }
        }
    }
    // Validate that IFM-zeroPoints are valid
    for ( int zp : ifmQuant.zeroPoints )
    {
        if ( !SupportedZeroPoint(zp, TensorUsage::IFM, ifmType, opType) )
        {
            return false;
        }
    }
    // Validate that IFM2-zeroPoints are valid
    if ( IsBinaryElementwise(opType) )
    {
        for ( int zp : ifm2Quant.zeroPoints )
        {
            if ( !SupportedZeroPoint(zp, TensorUsage::IFM, ifm2Type, opType) )
            {
                return false;
            }
        }
    }
    // Validate that OFM-zeroPoints are valid
    for ( int zp : ofmQuant.zeroPoints )
    {
        if ( !SupportedZeroPoint(zp, TensorUsage::OFM, ofmType, opType) )
        {
            return false;
        }
    }
    return true;
}

bool EthosU85Constraints::SupportsRescale(DataType fromType, DataType toType)
{
    UNUSED(toType);
    return fromType != DataType::UInt16;
}

bool EthosU85Constraints::SupportedDtypes(OpType opType, DataType ifmType, DataType ifm2Type, DataType ofmType)
{
    if ( opType == OpType::NullPool )
    {
        return true;
    }

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

    readonly_span_t<DataType> ofmTypes;

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
        ofmTypes = ifmEntry->second;
    }
    else
    {
        readonly_span_t<DataType> ifmTypes = s_defaultAllTypes;
        ofmTypes = s_defaultAllTypes;

        if ( !std::any_of(ifmTypes.begin(), ifmTypes.end(), [&](auto t) { return t == ifmType; }) )
        {
            // Unsupported ifm data type
            return false;
        }

        if ( IsBinaryElementwise(opType) )
        {
            // ifm1 and ifm2 data types must match
            if ( ifmType != ifm2Type )
            {
                // Bool8 can be combined with Int8 as is treated as signed Int8
                return ((ifmType == DataType::Bool8) && (ifm2Type == DataType::Int8)) ||
                       ((ifm2Type == DataType::Bool8) && (ifmType == DataType::Int8));
            }
        }
    }

    if ( !std::any_of(ofmTypes.begin(), ofmTypes.end(), [&](auto t) { return t == ofmType; }) )
    {  // Unsupported ofm data type
        return false;
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

// Validate that tensor dimensions are supported
static bool SupportedTensorDims(OpType opType, Shape ifmShape, Shape ifm2Shape, Shape ofmShape)
{
    for ( const auto &s : {ifmShape, ifm2Shape, ofmShape} )
    {
        if ( !s ) continue;
        auto shape = Shape::PadAxes(s, 4, 1);
        // validate that leading dimensions are unit
        int leadingDims = opType == OpType::Conv3D ? 1 : ofmShape.Size() - 3;
        for ( int i = 0; i < leadingDims; i++ )
        {
            if ( shape[i] > 1 )
            {
                return false;
            }
        }
    }
    return true;
}

// Validate that tensor axes are supported
static bool SupportedTensorAxes(Shape ifmShape, Shape ifm2Shape, Shape ofmShape)
{
    static constexpr int32_t MAX_AXIS = (1 << 16);
    for ( const auto &s : {ifmShape, ifm2Shape, ofmShape} )
    {
        if ( !s ) continue;
        auto shape = Shape::PadAxes(s, 4, 1);
        // validate that HWC are within valid range
        for ( int i = shape.Size() - 3; i < shape.Size(); i++ )
        {
            if ( shape[i] > MAX_AXIS )
            {
                return false;
            }
        }
    }
    return true;
}

Flags<QueryResult> EthosU85Constraints::OperatorQuery(OpType opType, const ArchOperatorQuery *query, ArchRequirements *req)
{
    Flags<QueryResult> result = QueryResult::Native;
    const auto &ifmShape = query ? query->ifm[0].shape : Shape();
    const auto &ifm2Shape = query ? query->ifm[1].shape : Shape();
    const auto &ofmShape = query ? query->ofm.shape : Shape();

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
    if ( opType == OpType::Conv3D )
    {
        if ( req )
        {
            // Validate tensor dimensions
            if ( !SupportedTensorDims(opType, ifmShape, ifm2Shape, ofmShape) )
            {
                req->decomposeProps.Set(ArchProperty::TensorDims);
            }
            req->req.Set(ArchRequirement::Decompose);
        }
        return query ? QueryResult::NativeHasReq : QueryResult::NativeConstrainedHasReq;
    }
    else if ( opType == OpType::TransposeConv2D )
    {
        if ( req )
        {
            // Validate tensor dimensions
            if ( !SupportedTensorDims(opType, ifmShape, ifm2Shape, ofmShape) )
            {
                req->decomposeProps.Set(ArchProperty::TensorDims);
            }
            // TODO: MLBEDSW-11089 - Enable IFM resampling TransposeConv path for Ethos-U85
            // For now, always decompose stride
            req->decomposeProps.Set(ArchProperty::KernelStride);
            req->req.Set(ArchRequirement::Decompose);
        }
        return query ? QueryResult::NativeHasReq : QueryResult::NativeConstrainedHasReq;
    }

    // Check direct native support of the opType
    auto npuOp = _arch->GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None )
    {
        LOG_TRACE1("{} has unsupported type\n", OpTypeToString(opType));
        return QueryResult::Unsupported;
    }

    // Short query (no additional detail)
    if ( !query )
    {
        // more detailed query might fail
        return QueryResult::NativeConstrained;
    }

    // Check for supported weight format for convolution type ops
    if ( opType == OpType::DepthwiseConv2D || opType == OpType::Conv2D || opType == OpType::FullyConnected )
    {
        // Check for non-constant weights or bias
        if ( query->weightFormat == WeightFormat::None && !query->ifm[1].shape )
        {
            if ( req )
            {
                req->req.Set(ArchRequirement::Decompose);
                req->decomposeProps.Set(ArchProperty::NonConstantWeights);
            }
            result.Set(QueryResult::HasRequirements);
        }
        else if ( query->ifm[1].shape && query->accSrc != ArchAccumulatorSource::Ifm2 )
        {
            // If weights are in IFM2 the kernel has to be 1x1
            if ( query->ifm[1].shape.ElementsWH() != 1 )
            {
                LOG_TRACE1("{} has unsupported IFM shape\n", OpTypeToString(opType));
                return QueryResult::Unsupported;
            }
        }
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
                LOG_TRACE1("{} has unsupported transpose type\n", OpTypeToString(opType));
                return QueryResult::Unsupported;
            }
        }
        if ( query->ofm.type == DataType::Int64 )
        {
            // Int64 Transpose requires decomposition
            if ( req )
            {
                req->req.Set(ArchRequirement::Decompose);
                req->decomposeProps.Set(ArchProperty::DataTypeLegalisation);
            }
            result.Set(QueryResult::HasRequirements);
        }
    }
    if ( query->reverseMask != ReverseType::None )
    {
        if ( !SupportsFusedReverse(opType, query->reverseMask) )
        {
            LOG_TRACE1("{} has unsupported reverse type\n", OpTypeToString(opType));
            return QueryResult::Unsupported;
        }
    }

    if ( npuOp == EthosU85NpuOp::Dma || npuOp == EthosU85NpuOp::Branch )
    {
        return result;
    }
    else if ( npuOp == EthosU85NpuOp::ReduceMinMax || npuOp == EthosU85NpuOp::ArgMax )
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
        // maximum supported axis for ArgMax
        static constexpr int32_t MAX_AXIS = (1 << 15);
        if ( npuOp == EthosU85NpuOp::ArgMax && query->ifm->shape && query->ifm->shape[query->axis] > MAX_AXIS )
        {
            if ( req )
            {
                req->req.Set(ArchRequirement::Decompose);
                req->decomposeProps.Set(ArchProperty::TensorAxis);
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
        if ( query->ifm[0].quantization )
        {
            for ( auto zp : query->ifm[0].quantization->zeroPoints )
            {
                if ( !SupportedZeroPoint(zp, TensorUsage::IFM0, ifmType, opType) )
                {
                    LOG_TRACE1("{} IFM0 has unsupported input zero point\n", OpTypeToString(opType), DataTypeToString(ifmType));
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
                    LOG_TRACE1("{} IFM1 has unsupported input zero point\n", OpTypeToString(opType), DataTypeToString(ifmType));
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
                    LOG_TRACE1("{} OFM has unsupported input zero point\n", OpTypeToString(opType), DataTypeToString(ifmType));
                    return QueryResult::Unsupported;
                }
            }
        }
    }

    // Validate dataTypes
    if ( typeInfo && !SupportedDtypes(opType, ifmType, ifm2Type, ofmType) )
    {
        LOG_TRACE1("{} has unsupported data types\n", OpTypeToString(opType));
        return QueryResult::Unsupported;
    }


    // Validate tensor dimensions
    if ( !SupportedTensorDims(opType, ifmShape, ifm2Shape, ofmShape) )
    {
        if ( req )
        {
            req->req.Set(ArchRequirement::Decompose);
            req->decomposeProps.Set(ArchProperty::TensorDims);
        }
        result.Set(QueryResult::HasRequirements);
    }

    // Validate tensor axes
    if ( !SupportedTensorAxes(ifmShape, ifm2Shape, ofmShape) )
    {
        if ( req )
        {
            req->req.Set(ArchRequirement::Decompose);
            req->decomposeProps.Set(ArchProperty::TensorAxis);
        }
        result.Set(QueryResult::HasRequirements);
    }

    // Detailed operator queries
    if ( opType == OpType::MatMul )
    {
        // Constrain MatMul Batch to 1
        // Note that MatMul's OFM are effectively rank 3 with dimensions [N, H, W] with a possible leading 1
        if ( ofmShape.Size() > 2 && ofmShape[ofmShape.Size() - 3] > 1 )
        {
            if ( req )
            {
                req->req.Set(ArchRequirement::Decompose);
                req->decomposeProps.Set(ArchProperty::TensorDims);
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
                if ( opType == OpType::AvgPool )
                {
                    // decomposing AvgPool kernels also requires decomposition of scaling
                    req->decomposeProps.Set(ArchProperty::Scaling);
                }
            }
            result.Set(QueryResult::HasRequirements);
        }

        if ( opType == OpType::AvgPool && (k->Size().x > 8 || k->Size().y > 8) && !k->Padding().IsZero() )
        {
            if ( req )
            {
                req->req.Set(ArchRequirement::Decompose);
                req->decomposeProps.Set(ArchProperty::Scaling);
            }
            result.Set(QueryResult::HasRequirements);
        }

        if ( k->Dilation().x > 1 || k->Dilation().y > 1 )
        {
            if ( req && req->req.Any(ArchRequirement::Decompose) )
            {
                // Dilation > 1 needs to be decomposed if decomposition is needed for any reason.
                // However, dilation > 1 in itself does not require decomposition.
                req->decomposeProps.Set(ArchProperty::KernelDilation);
            }
            if ( k->Dilation().x > 2 || k->Dilation().y > 2 )
            {
                if ( req )
                {
                    // Dilation > 2 requires decomposition.
                    req->req.Set(ArchRequirement::Decompose);
                    req->decomposeProps.Set(ArchProperty::KernelDilation);
                }
                result.Set(QueryResult::HasRequirements);
            }
        }
    }

    if ( opType == OpType::DepthwiseConv2D )
    {
        // Check for depth multiplier
        if ( query->weights.shape && ifmShape.Depth() < query->weights.shape.Depth() )
        {
            req->req.Set(ArchRequirement::Decompose);
            result.Set(QueryResult::HasRequirements);
        }
    }

    return result;
}

}  // namespace regor
