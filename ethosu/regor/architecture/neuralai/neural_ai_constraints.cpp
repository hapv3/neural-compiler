//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_constraints.hpp"

#include "architecture/architecture.hpp"

#include <algorithm>
#include <array>

namespace regor
{
namespace
{

thread_local std::array<ArchTensorRequirement, 3> s_tensorRequirements;

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

bool HasSymmetricZeroPoint(const ArchFM &fm)
{
    return fm.quantization == nullptr || fm.quantization->zeroPoints.empty() ||
           std::all_of(fm.quantization->zeroPoints.begin(), fm.quantization->zeroPoints.end(),
               [](int64_t value) { return value == 0; });
}

bool HasFullInt8Clamp(const ArchFM &fm)
{
    if ( fm.quantization == nullptr ) return false;
    const auto &quantization = *fm.quantization;
    return (quantization.quantMin.empty() ||
            std::all_of(quantization.quantMin.begin(), quantization.quantMin.end(),
                [](int64_t value) { return value <= -128; })) &&
           (quantization.quantMax.empty() ||
            std::all_of(quantization.quantMax.begin(), quantization.quantMax.end(),
                [](int64_t value) { return value >= 127; }));
}

bool HasRawAddScales(const ArchFM &lhs, const ArchFM &rhs, const ArchFM &ofm)
{
    if ( lhs.quantization == nullptr || rhs.quantization == nullptr || ofm.quantization == nullptr )
        return false;
    const auto &lhsScales = lhs.quantization->scales;
    const auto &rhsScales = rhs.quantization->scales;
    const auto &ofmScales = ofm.quantization->scales;
    if ( lhsScales.size() != 1 || rhsScales.size() != 1 || ofmScales.size() != 1 )
        return false;

    // TFLite graph optimisation represents equal-scale INT8 Add as two
    // 2^15 input rescales followed by a 2^-15 output rescale.  This exact
    // identity is the only form whose stored values can be added directly.
    const QuantizedScale inputIdentity(32768.0);
    const QuantizedScale outputIdentity(1.0 / 32768.0);
    return lhsScales[0] == inputIdentity && rhsScales[0] == inputIdentity &&
           ofmScales[0] == outputIdentity;
}

int64_t ScalarZeroPoint(const Quantization &quantization)
{
    return quantization.zeroPoints.empty() ? 0 : quantization.zeroPoints[0];
}

bool HasRawAverageQuantization(const ArchFM &ifm, const ArchFM &ofm)
{
    if ( ifm.quantization == nullptr || ofm.quantization == nullptr ||
         ifm.quantization->zeroPoints.size() > 1 || ofm.quantization->zeroPoints.size() > 1 ||
         ScalarZeroPoint(*ifm.quantization) != ScalarZeroPoint(*ofm.quantization) ||
         ofm.quantization->scales.size() != 1 ||
         ofm.quantization->scales[0] != QuantizedScale(1.0) )
        return false;
    return HasFullInt8Clamp(ofm);
}

}  // namespace

bool NeuralAIConstraints::IsSupportedOp(OpType opType)
{
    return opType == OpType::FullyConnected || opType == OpType::MatMul || opType == OpType::Conv2D ||
           opType == OpType::Add || opType == OpType::AvgPool || opType == OpType::LUT ||
           opType == OpType::Sigmoid || IsClipping(opType) ||
           opType == OpType::DepthwiseConv2D || opType == OpType::Resize ||
           opType == OpType::MaxPool || opType == OpType::MemoryCopy;
}

NeuralAIConstraints::Classification NeuralAIConstraints::Classify(OpType opType, const Shape &ifmShape,
    const Shape &weightShape, const Shape &ofmShape, const Kernel *kernel)
{
    Classification result;
    if ( opType != OpType::Conv2D && opType != OpType::DepthwiseConv2D )
    {
        result.diagnostic = "not a Conv operation";
        return result;
    }
    if ( !IsStaticPositiveShape(ifmShape) || !IsStaticPositiveShape(weightShape) ||
         !IsStaticPositiveShape(ofmShape) )
    {
        result.diagnostic = "requires static positive tensor shapes";
        return result;
    }
    if ( !HasBatchOne(ifmShape) || !HasBatchOne(ofmShape) )
    {
        result.diagnostic = "batch must be one";
        return result;
    }
    if ( kernel == nullptr )
    {
        result.diagnostic = "missing kernel";
        return result;
    }
    if ( kernel->Dilation() != Point2i(1, 1) )
    {
        result.diagnostic = "dilation must be 1x1";
        return result;
    }
    if ( kernel->Size() != Point2i(1, 1) && kernel->Size() != Point2i(3, 3) )
    {
        result.diagnostic = "kernel must be 1x1 or 3x3";
        return result;
    }
    if ( kernel->Stride() != Point2i(1, 1) && kernel->Stride() != Point2i(2, 2) )
    {
        result.diagnostic = "stride must be 1x1 or 2x2";
        return result;
    }
    if ( kernel->Padding().Top() < 0 || kernel->Padding().Top() > 1 ||
         kernel->Padding().Left() < 0 || kernel->Padding().Left() > 1 ||
         kernel->Padding().Bottom() < 0 || kernel->Padding().Bottom() > 1 ||
         kernel->Padding().Right() < 0 || kernel->Padding().Right() > 1 )
    {
        result.diagnostic = "padding must be zero or one on every side";
        return result;
    }

    const int ifmC = ifmShape.Depth();
    const int ofmC = ofmShape.Depth();
    const int64_t expectedOfmH = (int64_t(ifmShape.Height()) + kernel->Padding().Top() +
        kernel->Padding().Bottom() - kernel->Size().y) / kernel->Stride().y + 1;
    const int64_t expectedOfmW = (int64_t(ifmShape.Width()) + kernel->Padding().Left() +
        kernel->Padding().Right() - kernel->Size().x) / kernel->Stride().x + 1;
    if ( expectedOfmH <= 0 || expectedOfmW <= 0 || ofmShape.Height() != expectedOfmH ||
         ofmShape.Width() != expectedOfmW )
    {
        result.diagnostic = "output spatial shape does not match kernel, stride, and padding";
        return result;
    }
    result.hasIcTail = (ifmC % 32) != 0;
    result.hasOcTail = (ofmC % 32) != 0;
    const bool sameSpatial =
        ofmShape.Height() == (ifmShape.Height() + kernel->Stride().y - 1) / kernel->Stride().y &&
        ofmShape.Width() == (ifmShape.Width() + kernel->Stride().x - 1) / kernel->Stride().x;

    if ( opType == OpType::Conv2D && kernel->Size() == Point2i(3, 3) &&
         kernel->Stride() == Point2i(2, 2) && sameSpatial && ifmC == 3 && ofmC == 32 &&
         weightShape.Batch() == ofmC && weightShape.Height() == 3 && weightShape.Width() == 3 &&
         weightShape.Depth() == ifmC )
    {
        result.mode = NeuralAIOpMode::Conv2DRgbLinebufRequant;
        result.directNhwcInput = true;
        return result;
    }

    if ( opType == OpType::Conv2D && kernel->Size() == Point2i(1, 1) &&
         kernel->Stride() == Point2i(1, 1) && kernel->Padding().IsZero() &&
         weightShape.Batch() == ofmC && weightShape.Height() == 1 && weightShape.Width() == 1 &&
         weightShape.Depth() == ifmC )
    {
        result.mode = NeuralAIOpMode::Conv2DPointwiseC32Requant;
        return result;
    }

    if ( opType == OpType::Conv2D && kernel->Size() == Point2i(3, 3) &&
         sameSpatial &&
         weightShape.Batch() == ofmC && weightShape.Height() == 3 && weightShape.Width() == 3 &&
         weightShape.Depth() == ifmC )
    {
        if ( result.hasIcTail || result.hasOcTail )
        {
            result.diagnostic = "generic C32 Conv requires input and output channels divisible by 32";
            return result;
        }
        result.mode = kernel->Stride() == Point2i(1, 1) ? NeuralAIOpMode::Conv2DLinebufC32S1Requant :
                                                          NeuralAIOpMode::Conv2DLinebufC32S2Requant;
        result.groupStationary = ifmC > 32;
        return result;
    }

    if ( opType == OpType::DepthwiseConv2D && kernel->Size() == Point2i(3, 3) &&
         sameSpatial &&
         weightShape.Height() == 3 && weightShape.Width() == 3 && ifmC == ofmC &&
         ((weightShape.Batch() == 1 && weightShape.Depth() == ifmC) ||
          (weightShape.Batch() == ifmC && weightShape.Depth() == 1)) )
    {
        result.mode = kernel->Stride() == Point2i(1, 1) ? NeuralAIOpMode::DepthwiseC32S1Requant :
                                                           NeuralAIOpMode::DepthwiseC32S2Requant;
        result.groupStationary = !result.hasIcTail && !result.hasOcTail && ifmC > 32;
        return result;
    }

    result.diagnostic = "shape does not match a Phase 3 Conv datapath";
    return result;
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
    if ( !IsSupportedOp(opType) ) return QueryResult::Unsupported;
    if ( !query ) return QueryResult::NativeConstrained;

    if ( opType == OpType::Sigmoid || IsClipping(opType) )
    {
        if ( query->ifm[0].type != DataType::Int8 || query->ofm.type != DataType::Int8 ||
             !IsStaticPositiveShape(query->ifm[0].shape) ||
             !HasBatchOne(query->ifm[0].shape) ||
             query->ifm[0].shape != query->ofm.shape )
            return QueryResult::Unsupported;
        if ( req != nullptr )
        {
            req->req.Set(ArchRequirement::OpSubstitution);
            req->substitution = OpType::LUT;
        }
        return QueryResult::NativeHasReq;
    }

    if ( opType == OpType::MemoryCopy )
    {
        return query->ifm[0].type == query->ofm.type && query->ifm[0].shape == query->ofm.shape ?
            QueryResult::Native : QueryResult::Unsupported;
    }

    if ( opType == OpType::Resize )
    {
        const Shape &ifmShape = query->ifm[0].shape;
        const Shape &ofmShape = query->ofm.shape;
        if ( query->ifm[0].type != DataType::Int8 || query->ofm.type != DataType::Int8 ||
             !IsStaticPositiveShape(ifmShape) || !IsStaticPositiveShape(ofmShape) ||
             !HasBatchOne(ifmShape) || !HasBatchOne(ofmShape) ||
             ifmShape.Depth() != 32 || ofmShape.Depth() != 32 ||
             ofmShape.Height() % 2 != 0 || ofmShape.Height() / 2 != ifmShape.Height() ||
             ofmShape.Width() % 2 != 0 || ofmShape.Width() / 2 != ifmShape.Width() )
            return QueryResult::Unsupported;
        if ( req != nullptr )
        {
            Set(s_tensorRequirements[0], TensorUsage::IFM0, TensorFormat::C32Blocked);
            Set(s_tensorRequirements[1], TensorUsage::OFM, TensorFormat::C32Blocked);
            s_tensorRequirements[0].next = &s_tensorRequirements[1];
            s_tensorRequirements[1].next = nullptr;
            req->tensor = s_tensorRequirements[0];
            req->req.Set(ArchRequirement::OpSubstitution, ArchRequirement::Decompose);
            req->req.Set(ArchRequirement::Tensor);
        }
        return req != nullptr ? QueryResult::NativeHasReq : QueryResult::Native;
    }

    if ( query->ifm[0].type != DataType::Int8 || query->ofm.type != DataType::Int8 )
    {
        return QueryResult::Unsupported;
    }
    if ( opType == OpType::LUT )
    {
        if ( query->transposeMask != TransposeType::None || query->reverseMask != ReverseType::None ||
             query->accSrc != ArchAccumulatorSource::Reset ||
             !IsStaticPositiveShape(query->ifm[0].shape) ||
             query->ifm[0].shape != query->ofm.shape || !HasBatchOne(query->ifm[0].shape) )
            return QueryResult::Unsupported;
        if ( req != nullptr )
        {
            Set(s_tensorRequirements[0], TensorUsage::IFM0, TensorFormat::C32Blocked);
            Set(s_tensorRequirements[1], TensorUsage::OFM, TensorFormat::C32Blocked);
            s_tensorRequirements[0].next = &s_tensorRequirements[1];
            s_tensorRequirements[1].next = nullptr;
            req->tensor = s_tensorRequirements[0];
            req->req.Set(ArchRequirement::Tensor);
            return QueryResult::NativeHasReq;
        }
        return QueryResult::Native;
    }
    if ( query->transposeMask != TransposeType::None || query->reverseMask != ReverseType::None ||
         query->accSrc != ArchAccumulatorSource::Reset ||
         !IsStaticPositiveShape(query->ifm[0].shape) ||
         !IsStaticPositiveShape(query->ofm.shape) )
    {
        return QueryResult::Unsupported;
    }
    if ( opType == OpType::Add )
    {
        if ( query->ifm[1].type != DataType::Int8 ||
             !IsStaticPositiveShape(query->ifm[1].shape) ||
             !HasBatchOne(query->ifm[0].shape) ||
             !HasBatchOne(query->ifm[1].shape) ||
             !HasBatchOne(query->ofm.shape) ||
             query->ifm[0].shape != query->ifm[1].shape ||
             query->ifm[0].shape != query->ofm.shape ||
             query->ifm[0].quantization == nullptr ||
             query->ifm[1].quantization == nullptr ||
             query->ofm.quantization == nullptr ||
             !HasRawAddScales(query->ifm[0], query->ifm[1], query->ofm) ||
             !HasSymmetricZeroPoint(query->ifm[0]) ||
             !HasSymmetricZeroPoint(query->ifm[1]) ||
             !HasSymmetricZeroPoint(query->ofm) ||
             !HasFullInt8Clamp(query->ofm) )
        {
            return QueryResult::Unsupported;
        }
        if ( req != nullptr )
        {
            Set(s_tensorRequirements[0], TensorUsage::IFM0, TensorFormat::C32Blocked);
            Set(s_tensorRequirements[1], TensorUsage::IFM1, TensorFormat::C32Blocked);
            Set(s_tensorRequirements[2], TensorUsage::OFM, TensorFormat::C32Blocked);
            s_tensorRequirements[0].next = &s_tensorRequirements[1];
            s_tensorRequirements[1].next = &s_tensorRequirements[2];
            s_tensorRequirements[2].next = nullptr;
            req->tensor = s_tensorRequirements[0];
            req->req.Set(ArchRequirement::Tensor);
            return QueryResult::NativeHasReq;
        }
        return QueryResult::Native;
    }
    if ( opType == OpType::AvgPool )
    {
        const Kernel *kernel = query->kernel;
        const Shape &ifmShape = query->ifm[0].shape;
        const Shape &ofmShape = query->ofm.shape;
        const bool nearest2x = kernel != nullptr && kernel->Size() == Point2i(1, 1) &&
                               kernel->Stride() == Point2i(1, 1) && kernel->Padding().IsZero() &&
                               ifmShape.Depth() == 32 && ofmShape.Depth() == 32 &&
                               ofmShape.Height() % 2 == 0 && ofmShape.Height() / 2 == ifmShape.Height() &&
                               ofmShape.Width() % 2 == 0 && ofmShape.Width() / 2 == ifmShape.Width();
        if ( nearest2x )
        {
            if ( req != nullptr )
            {
                Set(s_tensorRequirements[0], TensorUsage::IFM0, TensorFormat::C32Blocked);
                Set(s_tensorRequirements[1], TensorUsage::OFM, TensorFormat::C32Blocked);
                s_tensorRequirements[0].next = &s_tensorRequirements[1];
                s_tensorRequirements[1].next = nullptr;
                req->tensor = s_tensorRequirements[0];
                req->req.Set(ArchRequirement::Tensor);
                return QueryResult::NativeHasReq;
            }
            return QueryResult::Native;
        }
        if ( kernel == nullptr || !HasBatchOne(ifmShape) || !HasBatchOne(ofmShape) ||
             kernel->Size() != Point2i(ifmShape.Width(), ifmShape.Height()) ||
             kernel->Stride() != Point2i(1, 1) ||
             kernel->Dilation() != Point2i(1, 1) || !kernel->Padding().IsZero() ||
             ofmShape.Height() != 1 || ofmShape.Width() != 1 ||
             ofmShape.Depth() != ifmShape.Depth() ||
             !HasRawAverageQuantization(query->ifm[0], query->ofm) )
        {
            return QueryResult::Unsupported;
        }
        if ( req != nullptr )
        {
            Set(s_tensorRequirements[0], TensorUsage::IFM0, TensorFormat::C32Blocked);
            Set(s_tensorRequirements[1], TensorUsage::OFM, TensorFormat::C32Blocked);
            s_tensorRequirements[0].next = &s_tensorRequirements[1];
            s_tensorRequirements[1].next = nullptr;
            req->tensor = s_tensorRequirements[0];
            req->req.Set(ArchRequirement::Tensor);
            return QueryResult::NativeHasReq;
        }
        return QueryResult::Native;
    }
    if ( opType == OpType::MaxPool )
    {
        const Kernel *kernel = query->kernel;
        const Shape &ifmShape = query->ifm[0].shape;
        const Shape &ofmShape = query->ofm.shape;
        if ( kernel == nullptr || !HasBatchOne(ifmShape) || !HasBatchOne(ofmShape) ||
             ifmShape.Depth() != 32 || ofmShape != ifmShape ||
             kernel->Size() != Point2i(5, 5) || kernel->Stride() != Point2i(1, 1) ||
             kernel->Dilation() != Point2i(1, 1) ||
             kernel->Padding().Top() != 2 || kernel->Padding().Bottom() != 2 ||
             kernel->Padding().Left() != 2 || kernel->Padding().Right() != 2 )
            return QueryResult::Unsupported;
        if ( req != nullptr )
        {
            Set(s_tensorRequirements[0], TensorUsage::IFM0, TensorFormat::C32Blocked);
            Set(s_tensorRequirements[1], TensorUsage::OFM, TensorFormat::C32Blocked);
            s_tensorRequirements[0].next = &s_tensorRequirements[1];
            s_tensorRequirements[1].next = nullptr;
            req->tensor = s_tensorRequirements[0];
            req->req.Set(ArchRequirement::Tensor);
            return QueryResult::NativeHasReq;
        }
        return QueryResult::Native;
    }
    const DataType weightsType = query->weights.type != DataType::None ? query->weights.type : query->ifm[1].type;
    if ( weightsType != DataType::Int8 || query->weightFormat == WeightFormat::None )
    {
        return QueryResult::Unsupported;
    }
    const Shape &weightsShape = query->weights.shape ? query->weights.shape : query->ifm[1].shape;
    if ( !IsStaticPositiveShape(weightsShape) )
    {
        return QueryResult::Unsupported;
    }
    if ( !HasBatchOne(query->ifm[0].shape) || !HasBatchOne(query->ofm.shape) )
    {
        return QueryResult::Unsupported;
    }
    if ( opType == OpType::Conv2D || opType == OpType::DepthwiseConv2D )
    {
        const Classification classification = Classify(opType, query->ifm[0].shape, weightsShape,
            query->ofm.shape, query->kernel);
        const ArchFM &weightFm = query->weights.type != DataType::None ? query->weights : query->ifm[1];
        if ( !classification || !HasSymmetricZeroPoint(query->ifm[0]) ||
             !HasSymmetricZeroPoint(weightFm) ) return QueryResult::Unsupported;
        if ( req != nullptr )
        {
            Set(s_tensorRequirements[0], TensorUsage::IFM0,
                classification.directNhwcInput ? TensorFormat::NHWC : TensorFormat::C32Blocked);
            Set(s_tensorRequirements[1], TensorUsage::OFM, TensorFormat::C32Blocked);
            s_tensorRequirements[0].next = &s_tensorRequirements[1];
            s_tensorRequirements[1].next = nullptr;
            req->tensor = s_tensorRequirements[0];
            req->req.Set(ArchRequirement::Tensor);
            return QueryResult::NativeHasReq;
        }
    }
    return QueryResult::Native;
}

bool NeuralAIConstraints::SupportedZeroPoint(int64_t zeroPoint, TensorUsage usage, DataType dataType, OpType opType)
{
    if ( !IsSupportedOp(opType) || dataType != DataType::Int8 ) return false;
    if ( opType == OpType::LUT || opType == OpType::Sigmoid || opType == OpType::Resize ||
         IsClipping(opType) )
        return (IsIFM(usage) || IsOFM(usage)) && zeroPoint >= -128 && zeroPoint <= 127;
    if ( opType == OpType::Add ) return (IsIFM(usage) || IsOFM(usage)) && zeroPoint == 0;
    if ( opType == OpType::AvgPool )
        return (IsIFM(usage) || IsOFM(usage)) && zeroPoint >= -128 && zeroPoint <= 127;
    if ( opType == OpType::MaxPool )
        return (IsIFM(usage) || IsOFM(usage)) && zeroPoint >= -128 && zeroPoint <= 127;
    if ( IsIFM(usage) || usage == TensorUsage::Weights ) return zeroPoint == 0;
    return IsOFM(usage) && zeroPoint >= -128 && zeroPoint <= 127;
}

}  // namespace regor
