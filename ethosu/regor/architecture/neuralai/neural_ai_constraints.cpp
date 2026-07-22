//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_constraints.hpp"

namespace regor
{
namespace
{

bool IsStaticPositiveShape(const Shape &shape)
{
    return shape && shape.Elements64() > 0 && shape.LessMask(shape.WithZeros()) == 0;
}

bool HasBatchOne(const Shape &shape)
{
    for ( int i = 0; i < shape.Size() - 2; ++i )
    {
        if ( shape[i] != 1 ) return false;
    }
    return true;
}

}  // namespace

bool NeuralAIConstraints::IsSupportedOp(OpType opType)
{
    return opType == OpType::FullyConnected || opType == OpType::MatMul || opType == OpType::MemoryCopy;
}

bool NeuralAIConstraints::SupportsQuantization(OpType opType, const Quantization &, DataType ifmType,
    const Quantization &, DataType ifm2Type, const Quantization &, DataType ofmType)
{
    return IsSupportedOp(opType) && ifmType == DataType::Int8 &&
           (ifm2Type == DataType::None || ifm2Type == DataType::Int8) &&
           ofmType == DataType::Int8;
}

bool NeuralAIConstraints::SupportsQuantization(OpType opType, const Quantization &ifmQuant, DataType ifmType,
    const Quantization &ofmQuant, DataType ofmType)
{
    return SupportsQuantization(
        opType, ifmQuant, ifmType, Quantization::Unit(), DataType::Int8, ofmQuant, ofmType);
}

Flags<QueryResult> NeuralAIConstraints::OperatorQuery(
    OpType opType, const ArchOperatorQuery *query, ArchRequirements *req)
{
    UNUSED(req);
    if ( !IsSupportedOp(opType) ) return QueryResult::Unsupported;
    if ( !query ) return QueryResult::NativeConstrained;

    if ( opType == OpType::MemoryCopy )
    {
        return query->ifm[0].type == query->ofm.type && query->ifm[0].shape == query->ofm.shape ?
            QueryResult::Native : QueryResult::Unsupported;
    }

    if ( query->ifm[0].type != DataType::Int8 || query->ofm.type != DataType::Int8 )
    {
        return QueryResult::Unsupported;
    }
    const DataType weightsType = query->weights.type != DataType::None ? query->weights.type : query->ifm[1].type;
    if ( weightsType != DataType::Int8 || query->weightFormat == WeightFormat::None )
    {
        return QueryResult::Unsupported;
    }
    const Shape &weightsShape = query->weights.shape ? query->weights.shape : query->ifm[1].shape;
    if ( !IsStaticPositiveShape(query->ifm[0].shape) || !IsStaticPositiveShape(weightsShape) ||
         !IsStaticPositiveShape(query->ofm.shape) )
    {
        return QueryResult::Unsupported;
    }
    if ( query->transposeMask != TransposeType::None || query->reverseMask != ReverseType::None ||
         query->accSrc != ArchAccumulatorSource::Reset )
    {
        return QueryResult::Unsupported;
    }
    if ( !HasBatchOne(query->ifm[0].shape) || !HasBatchOne(query->ofm.shape) )
    {
        return QueryResult::Unsupported;
    }
    return QueryResult::Native;
}

bool NeuralAIConstraints::SupportedZeroPoint(int64_t zeroPoint, TensorUsage usage, DataType dataType, OpType opType)
{
    if ( !IsSupportedOp(opType) || dataType != DataType::Int8 ) return false;
    if ( IsIFM(usage) || usage == TensorUsage::Weights ) return zeroPoint == 0;
    return IsOFM(usage) && zeroPoint >= -128 && zeroPoint <= 127;
}

}  // namespace regor
