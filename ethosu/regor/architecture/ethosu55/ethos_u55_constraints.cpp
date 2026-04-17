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


// Helpers for SupportsQuantization
namespace
{
// Check that IFM-scales are supported
bool SupportedIFMQuant(OpType opType, const Quantization &ifmQuant, const Quantization &ifm2Quant)
{
    // Per-channel IFM scaling is not supported
    if ( ifmQuant.scales.size() > 1 || ifm2Quant.scales.size() > 1 )
    {
        return false;
    }
    const auto &ifmScale = ifmQuant.Scale();
    const auto &ifm2Scale = ifm2Quant.Scale();
    // Binary IFM-fusing constraints for Add/Sub
    if ( opType == OpType::Add || opType == OpType::Sub )
    {
        // Allow 16-bit (simple) rescale with shift == 0
        if ( ifmScale.shift == 0 && ifm2Scale.shift == 0 && int16_t(ifmScale.scale) == ifmScale.scale &&
             int16_t(ifm2Scale.scale) == ifm2Scale.scale )
        {
            return true;
        }
        // Allow if one of the inputs has power of two scale
        if ( IsPowerOfTwo(ifmScale.scale) || IsPowerOfTwo(ifm2Scale.scale) )
        {
            return true;
        }
        return false;
    }
    // For other operations, IFM scale must be unit
    else if ( IsBinaryElementwise(opType) )
    {
        return ifmScale == QuantizedScale::Unit() && ifm2Scale == QuantizedScale::Unit();
    }
    return ifmScale == QuantizedScale::Unit();
}
// Check that OFM-scales are supported
bool SupportedOFMQuant(OpType opType, EthosU55NpuOp npuOp, const Quantization &ofmQuant, DataType ifmType, DataType ofmType)
{
    // Reject per-channel scaling for opTypes that do not support it
    if ( IsElementwise(opType) || opType == OpType::ReduceSum || opType == OpType::AvgPool || opType == OpType::Table || opType == OpType::MatMul )
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
    if ( npuOp == EthosU55NpuOp::Convolution || npuOp == EthosU55NpuOp::Depthwise || npuOp == EthosU55NpuOp::VectorProduct ||
         npuOp == EthosU55NpuOp::ReduceSum || (npuOp == EthosU55NpuOp::Pooling && opType != OpType::AvgPool) || opType == OpType::MatMul )
    {
        return true;
    }
    else if ( IsElementwise(opType) )
    {
        // The following opTypes require unit ofm-scaling
        if ( opType == OpType::Minimum || opType == OpType::Maximum || opType == OpType::Asr || opType == OpType::SHL ||
             opType == OpType::CLZ || opType == OpType::LeakyRelu )
        {
            return ofmScale == QuantizedScale::Unit();
        }
        // Elementwise supports shift-only if input or output is 32-bit
        return (DataTypeSizeBits(ifmType) < 32 && DataTypeSizeBits(ofmType) < 32) || ofmScale.scale == 1;
    }
    return ofmScale == QuantizedScale::Unit();
}
}  // namespace

// Check if explicit quantization is supported by opType
bool EthosU55Constraints::SupportsQuantization(OpType opType, const Quantization &ifmQuant, DataType ifmType,
    const Quantization &ifm2Quant, DataType ifm2Type, const Quantization &ofmQuant, DataType ofmType)
{
    auto npuOp = ArchEthosU55::GetHWOp(opType);
    // This function assumes EXPLICIT quantization type
    // Other QuantizationTypes must be converted before validation
    assert(ifmQuant.type == QuantizationType::EXPLICIT);
    assert(ifm2Quant.type == QuantizationType::EXPLICIT);
    assert(ofmQuant.type == QuantizationType::EXPLICIT);
    // DMA operations do not support quantization
    if ( npuOp == EthosU55NpuOp::Dma )
    {
        return false;
    }
    // Validate that quantization is valid
    if ( !SupportedIFMQuant(opType, ifmQuant, ifm2Quant) )
    {
        return false;
    }
    if ( !SupportedOFMQuant(opType, npuOp, ofmQuant, ifmType, ofmType) )
    {
        return false;
    }
    // Check that dataTypes are supported
    if ( npuOp != EthosU55NpuOp::None )
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

    if ( IsActivation(opType) && ifmType == DataType::Int16 && ofmType == DataType::Int32 )
    {
        // Int16 activation functions do not support Int32 output.
        return false;
    }

    if ( _arch->UseAvgPoolNop(opType) || opType == OpType::Rescale )
    {
        // TODO MLBEDSW-10667: The rules for UseAvgPoolNop are not the same as for a Pooling operation, so skip checks
        // for now
        return true;
    }

    // Matmul must have 8-bit IFM-precision
    if ( opType == OpType::MatMul && DataTypeSizeBits(ifmType) > 8 )
    {
        return false;
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
            LOG_TRACE1("{} has unsupported IFM shape\n", OpTypeToString(opType));
            return QueryResult::Unsupported;
        }
        if ( req )
        {
            req->req.Set(ArchRequirement::OpSubstitution, ArchRequirement::Decompose);
        }
        return QueryResult::NativeHasReq;
    }
    // TransposeConv2D and Conv3D are legalized during decomposition
    else if ( opType == OpType::Conv3D )
    {
        // Check for supported weight format
        if ( query && query->weightFormat != WeightFormat::Default )
        {
            LOG_TRACE1("{} has unsupported weights format\n", OpTypeToString(opType));
            return QueryResult::Unsupported;
        }
        if ( req )
        {
            // Check for batch size > 1 to set decompose property
            if ( query && query->ifm[0].shape && query->ifm[0].shape[0] > 1 )
            {
                req->decomposeProps.Set(ArchProperty::TensorDims);
            }
            req->req.Set(ArchRequirement::Decompose);
        }
        return query ? QueryResult::NativeHasReq : QueryResult::NativeConstrainedHasReq;
    }
    else if ( opType == OpType::TransposeConv2D )
    {
        assert(query);
        // Check for supported weight format
        if ( query->weightFormat != WeightFormat::Default )
        {
            return QueryResult::Unsupported;
        }

        // TransposeConv2D constrained to only allow strides that work with IFM resampling
        auto k = query->kernel;
        assert(k);
        auto stride = k->Stride();
        if ( stride == Point2i(1, 1) || stride == Point2i(2, 2) ||
             (stride == Point2i(1, 2) && query->ifm[0].shape.Width() == 1 && k->Size().x == 1) ||
             (stride == Point2i(2, 1) && query->ifm[0].shape.Height() == 1 && k->Size().y == 1) )
        {
            if ( req )
            {
                req->req.Set(ArchRequirement::OpSubstitution, ArchRequirement::Decompose);
            }
            return QueryResult::NativeHasReq;
        }
        else
        {
            return QueryResult::Unsupported;
        }
    }
    else if ( opType == OpType::ArgMax )
    {
        assert(query);
        if ( (query->axis == -1 || query->axis == query->ifm[0].shape.Size() - 1) &&
             (query->ifm[0].shape.Size() <= 3 || query->ifm[0].shape.AxisProduct(0, -3) == 1) &&
             (query->ifm[0].type == DataType::Int8 || query->ifm[0].type == DataType::UInt8) && query->ifm[0].shape.Depth() <= 127 )
        {
            if ( req )
            {
                req->req.Set(ArchRequirement::OpSubstitution);
            }
            return QueryResult::NativeHasReq;
        }
        return QueryResult::Unsupported;
    }

    // Check direct native support of the opType
    auto npuOp = _arch->GetHWOp(opType);
    if ( npuOp == EthosU55NpuOp::None )
    {
        LOG_TRACE1("{} has unsupported type\n", OpTypeToString(opType));
        return QueryResult::Unsupported;
    }
    else if ( npuOp == EthosU55NpuOp::Dma )
    {
        return QueryResult::Native;
    }

    Flags<QueryResult> result = QueryResult::Native;


    if ( opType == OpType::Transpose || opType == OpType::MatMul )
    {
        result.Set(QueryResult::Emulated);
    }


    // Short query (no additional detail)
    if ( !query )
    {
        // More detailed query might fail (constrained)
        result.Set(QueryResult::Constrained);
        return result;
    }

    // Check for supported weight format for convolution type ops
    if ( opType == OpType::DepthwiseConv2D || opType == OpType::Conv2D || opType == OpType::FullyConnected )
    {
        if ( query->weightFormat != WeightFormat::Default )
        {
            LOG_TRACE1("{} has unsupported weights format\n", OpTypeToString(opType));
            return QueryResult::Unsupported;
        }
    }


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
    else if ( opType == OpType::ReduceMax )
    {
        // We require axis to be the width, corresponding to (2 - size)
        if ( query->axis != -2 )
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
    else if ( opType == OpType::LUT )
    {
        if ( query->ifm[0].type == DataType::Int16 && query->ofm.type == DataType::Int32 )
        {
            if ( req )
            {
                req->req.Set(ArchRequirement::OpSubstitution);
                req->substitution = OpType::LUT;
            }
            return QueryResult::NativeHasReq;
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
    if ( typeInfo && !SupportedDtypes(opType, ifmType, ifm2Type, ofmType) )
    {
        LOG_TRACE1("{} has unsupported data types\n", OpTypeToString(opType));
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
            LOG_TRACE1("{} has unsupported tensor axis\n", OpTypeToString(opType));
            return QueryResult::Unsupported;
        }
        // TODO MLBEDSW-10668: channel-axis for 32-bit NWHC-transpose is constrained to 14-bits
        static constexpr int TRANSPOSE_32_MAX_CHANNEL = (1 << 14);
        if ( (shapeInfo && typeInfo) && (query->ifm[0].type == DataType::Int32) &&
             (query->transposeMask == TransposeType::NWHC) && (query->ifm[0].shape.Depth() > TRANSPOSE_32_MAX_CHANNEL) )
        {
            LOG_TRACE1("{} has unsupported transpose mask\n", OpTypeToString(opType));
            return QueryResult::Unsupported;
        }
        // Validate supported transpose-masks
        if ( (typeInfo) && (query->ifm[0].type == DataType::Int32) && (query->transposeMask == TransposeType::NCWH) )
        {
            // 32-bit NCWH must be constructed using other permutations since custom striding in channel dimension is
            // not supported
            if ( req )
            {
                req->req.Set(ArchRequirement::Decompose);
                req->decomposeProps.Set(ArchProperty::TransposeMask);
            }
            result.Set(QueryResult::HasRequirements);
        }
        if ( query->transposeMask != TransposeType::NWHC && query->transposeMask != TransposeType::NHCW &&
             query->transposeMask != TransposeType::NCWH && query->transposeMask != TransposeType::None )
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
            LOG_TRACE1("{} has unsupported transpose mask\n", OpTypeToString(opType));
            return QueryResult::Unsupported;
        }
    }

    // reverseType::W and reverseType::H are supported
    if ( Flags<ReverseType>(query->reverseMask).Unset(ReverseType::H, ReverseType::W) != ReverseType::None )
    {
        LOG_TRACE1("{} has unsupported reverse mask\n", OpTypeToString(opType));
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
                    LOG_TRACE1("{} IFM0 has unsupported input zero point\n", OpTypeToString(opType));
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
                    LOG_TRACE1("{} IFM1 has unsupported input zero point\n", OpTypeToString(opType));
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
                    LOG_TRACE1("{} OFM has unsupported input zero point\n", OpTypeToString(opType));
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
    else if ( opType == OpType::MemoryCopy )
    {
        if ( typeInfo && shapeInfo && DataTypeSizeBits(ofmType) > 16 )
        {
            const int ofmBits = DataTypeSizeBits(ofmType);
            assert(ofmBits == 32 || ofmBits == 64);
            result.Set(QueryResult::HasRequirements);
            // Depth has additional constraints for 64-bit and 32-bit copies since they're expanded in RCS generation
            const int depthMultiplier = ofmBits == 64 ? 4 : 2;
            const int maxDepth = MAX_AXIS / depthMultiplier;
            if ( req )
            {
                req->req.Set(ArchRequirement::Tensor);
                // The int16 reinterpretation in Ethos-U55 RCS generation only supports linear format.
                ArchTensorRequirement *tr = &req->tensor;
                Set(*tr, TensorUsage::OFM, DataType::None, TensorFormat::NHWC);
                if ( ofmShape.Depth() > maxDepth )
                {
                    req->req.Set(ArchRequirement::Decompose);
                    req->decomposeProps.Set(ArchProperty::DataTypeLegalisation);
                    tr->shape = ofmShape.WithDepth(maxDepth);
                }
                Set(*NextTensor(tr, usedTensors), TensorUsage::IFM, TensorFormat::NHWC);
            }
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

        if ( opType == OpType::AvgPool && (k->Size().x > 8 || k->Size().y > 8) && !k->Padding().IsZero() )
        {
            LOG_TRACE1("{} has unsupported kernel\n", OpTypeToString(opType));
            return QueryResult::Unsupported;
        }
    }
    else
    {
        // no kernel provided, more detailed queries might fail
        result.Set(QueryResult::Constrained);
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
