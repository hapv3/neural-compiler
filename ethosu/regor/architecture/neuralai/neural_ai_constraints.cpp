//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_constraints.hpp"

#include "architecture/architecture.hpp"

#include <array>

namespace regor
{
namespace
{

thread_local std::array<ArchTensorRequirement, 2> s_pointwiseTensorRequirements;

bool IsStaticPositiveShape(const Shape &shape)
{
    return shape && shape.Elements64() > 0 && shape.LessMask(shape.WithZeros()) == 0;
}

bool HasBatchOne(const Shape &shape)
{
    // Matrix M may be represented by one or more spatial dimensions after
    // the leading batch axis.  Only the batch dimension is fixed to one.
    return shape.Size() <= 2 || shape[0] == 1;
}

}  // namespace

bool NeuralAIConstraints::IsSupportedOp(OpType opType)
{
    return opType == OpType::FullyConnected || opType == OpType::MatMul || opType == OpType::Conv2D ||
           opType == OpType::MemoryCopy;
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
    if ( opType == OpType::Conv2D )
    {
        if ( query->kernel == nullptr || query->kernel->Size() != Point2i(1, 1) ||
             query->kernel->Stride() != Point2i(1, 1) || query->kernel->Dilation() != Point2i(1, 1) ||
             !query->kernel->Padding().IsZero() )
        {
            return QueryResult::Unsupported;
        }
        if ( weightsShape.Batch() != query->ofm.shape.Depth() ||
             weightsShape.Height() != 1 || weightsShape.Width() != 1 ||
             weightsShape.Depth() != query->ifm[0].shape.Depth() )
        {
            return QueryResult::Unsupported;
        }
        if ( req != nullptr )
        {
            Set(s_pointwiseTensorRequirements[0], TensorUsage::IFM0, TensorFormat::C32Blocked);
            Set(s_pointwiseTensorRequirements[1], TensorUsage::OFM, TensorFormat::C32Blocked);
            s_pointwiseTensorRequirements[0].next = &s_pointwiseTensorRequirements[1];
            req->tensor = s_pointwiseTensorRequirements[0];
            req->req.Set(ArchRequirement::Tensor);
            return QueryResult::NativeHasReq;
        }
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
