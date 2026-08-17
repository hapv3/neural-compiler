//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_graph_optimiser.hpp"

#include "common/buffer_view.hpp"
#include "kernel.hpp"
#include "operation_util.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace regor
{

namespace
{

int64_t ScalarZeroPoint(const Quantization &quantization)
{
    return quantization.zeroPoints.empty() ? 0 : quantization.zeroPoints[0];
}

bool HasScalarQuantization(const Quantization &quantization)
{
    return quantization.scales.size() == 1 && quantization.zeroPoints.size() <= 1;
}

bool IsRawAddScale(const TensorConnection *lhs, const TensorConnection *rhs, const TensorConnection *ofm)
{
    if ( lhs == nullptr || rhs == nullptr || ofm == nullptr ||
         !HasScalarQuantization(lhs->quantization) || !HasScalarQuantization(rhs->quantization) ||
         !HasScalarQuantization(ofm->quantization) ) return false;
    return lhs->quantization.scales[0] == QuantizedScale(32768.0) &&
        rhs->quantization.scales[0] == QuantizedScale(32768.0) &&
        ofm->quantization.scales[0] == QuantizedScale(1.0 / 32768.0) &&
        (ofm->quantization.quantMin.empty() || ofm->quantization.quantMin[0] <= -128) &&
        (ofm->quantization.quantMax.empty() || ofm->quantization.quantMax[0] >= 127);
}

bool ShiftClamp(Quantization &quantization, int64_t delta)
{
    if ( quantization.quantMin.size() != 1 || quantization.quantMax.size() != 1 ) return false;
    const int64_t shiftedMin = quantization.quantMin[0] + delta;
    const int64_t shiftedMax = quantization.quantMax[0] + delta;
    if ( shiftedMin < -128 || shiftedMax > 127 || shiftedMin > shiftedMax ) return false;
    quantization.quantMin[0] = shiftedMin;
    quantization.quantMax[0] = shiftedMax;
    return true;
}

}  // namespace

void NeuralAIGraphOptimiser::CanonicalizeAsymmetricConv(Graph *graph, Operation *operation)
{
    const OpType opType = operation->Type();
    if ( opType != OpType::Conv2D && opType != OpType::DepthwiseConv2D ) return;

    TensorConnection *ifm = operation->Input(TensorUsage::IFM0);
    TensorConnection *weights = operation->Input(TensorUsage::Weights);
    TensorConnection *bias = operation->Input(TensorUsage::Scales);
    TensorConnection *ofm = operation->Output(TensorUsage::OFM);
    const Kernel *kernel = operation->Kernel();
    if ( ifm == nullptr || weights == nullptr || bias == nullptr || ofm == nullptr || kernel == nullptr ||
         ifm->tensor->Type() != DataType::Int8 || weights->tensor->Type() != DataType::Int8 ||
         bias->tensor->Type() != DataType::Int32 || ifm->quantization.zeroPoints.size() != 1 ||
         !weights->tensor->IsConstant() || !bias->tensor->IsConstant() )
        return;

    const int64_t ifmZeroPoint = ifm->quantization.zeroPoints[0];
    if ( ifmZeroPoint == 0 || std::any_of(weights->quantization.zeroPoints.begin(),
                                  weights->quantization.zeroPoints.end(), [](int64_t value) { return value != 0; }) )
        return;

    const Shape weightShape = weights->shape.IsEmpty() ? weights->tensor->StorageShape() : weights->shape;
    const Shape ofmShape = ofm->shape.IsEmpty() ? ofm->tensor->StorageShape() : ofm->shape;
    if ( weightShape.Size() != 4 || ofmShape.Depth() <= 0 || bias->tensor->View().Elements() != ofmShape.Depth() ) return;

    const int outputChannels = ofmShape.Depth();
    const auto weightValues = weights->tensor->View().Values<int>(DataType::Int8);
    const auto biasValues = bias->tensor->View().Values<int32_t>();
    std::vector<int32_t> adjustedBias;
    adjustedBias.reserve(outputChannels);
    for ( int oc = 0; oc < outputChannels; ++oc )
    {
        int64_t weightSum = 0;
        if ( opType == OpType::Conv2D )
        {
            if ( weightShape.Batch() != outputChannels ) return;
            for ( int ky = 0; ky < weightShape.Height(); ++ky )
                for ( int kx = 0; kx < weightShape.Width(); ++kx )
                    for ( int ic = 0; ic < weightShape.Depth(); ++ic )
                        weightSum += weightValues[Shape(oc, ky, kx, ic)];
        }
        else if ( weightShape.Batch() == 1 && weightShape.Depth() == outputChannels )
        {
            for ( int ky = 0; ky < weightShape.Height(); ++ky )
                for ( int kx = 0; kx < weightShape.Width(); ++kx )
                    weightSum += weightValues[Shape(0, ky, kx, oc)];
        }
        else if ( weightShape.Batch() == outputChannels && weightShape.Depth() == 1 )
        {
            for ( int ky = 0; ky < weightShape.Height(); ++ky )
                for ( int kx = 0; kx < weightShape.Width(); ++kx )
                    weightSum += weightValues[Shape(oc, ky, kx, 0)];
        }
        else
        {
            return;
        }

        const int64_t corrected = int64_t(biasValues[oc]) - ifmZeroPoint * weightSum;
        if ( corrected < std::numeric_limits<int32_t>::min() || corrected > std::numeric_limits<int32_t>::max() ) return;
        adjustedBias.push_back(int32_t(corrected));
    }

    Operation *explicitPad = nullptr;
    if ( kernel->Padding().IsZero() && ifm->tensor->Writers().size() == 1 )
    {
        Operation *producer = ifm->tensor->Writers().front().get();
        if ( producer->Type() == OpType::Pad && producer->Output(TensorUsage::OFM)->tensor == ifm->tensor &&
             producer->Attribute<pad_attr_t>()->pad_const == 0 )
            explicitPad = producer;
    }

    Quantization zeroPointQuantization = ifm->quantization;
    zeroPointQuantization.zeroPoints = {0};
    const auto connectPaddingValue = [operation, ifmZeroPoint]()
    {
        const Shape scalarShape(1);
        auto value = CreateConstTensor("neural_ai_padding_value", DataType::Int8,
            std::make_shared<Buffer>(std::vector<int8_t>{int8_t(ifmZeroPoint)}), &scalarShape);
        operation->ConnectInput(TensorUsage::Params1, value).Set(Quantization::Unit());
    };
    if ( !kernel->Padding().IsZero() )
    {
        if ( opType == OpType::DepthwiseConv2D )
        {
            const Margin padding = kernel->Padding();
            const std::vector<int32_t> paddingValues{
                0, 0, padding.Top(), padding.Bottom(), padding.Left(), padding.Right(), 0, 0};
            const Shape paddingShape(int(paddingValues.size()));
            auto paddingTensor = CreateConstTensor("neural_ai_asymmetric_padding", DataType::Int32,
                std::make_shared<Buffer>(std::vector<int32_t>(paddingValues)), &paddingShape);
            const Shape inputShape = ifm->shape.IsEmpty() ? ifm->tensor->StorageShape() : ifm->shape;
            const Shape paddedShape = inputShape.WithHW(inputShape.WH() + padding.TL() + padding.BR());
            auto paddedTensor = std::make_shared<Tensor>(
                ifm->tensor->Name() + "/zero_point_padding", ifm->tensor->Type(), paddedShape);
            auto pad = std::make_shared<Operation>(OpType::Pad);
            pad->CopyInput(TensorUsage::IFM0, *ifm);
            pad->ConnectInput(TensorUsage::Params, paddingTensor);
            pad->ConnectOutput(TensorUsage::OFM, paddedTensor).Set(paddedShape).Set(zeroPointQuantization);
            pad->Attribute<pad_attr_t>()->pad_const = ifmZeroPoint;
            operation->ConnectInput(TensorUsage::IFM0, paddedTensor).Set(paddedShape).Set(zeroPointQuantization);
            operation->SetKernel(std::make_unique<Kernel>(kernel->WithPadding({})));
            RecordOptimisation(*operation, pad.get());
        }
        else
        {
            ifm->Set(zeroPointQuantization);
            connectPaddingValue();
        }
    }
    else if ( graph->IsInput(ifm->tensor.get()) )
    {
        const TensorConnection original = *ifm;
        auto nativeTensor = std::shared_ptr<Tensor>(original.tensor->Clone().release());
        nativeTensor->SetName(original.tensor->Name() + "/zero_point_staging");
        auto copy = std::make_shared<Operation>(OpType::MemoryCopy);
        copy->CopyInput(TensorUsage::IFM, original);
        copy->ConnectOutput(TensorUsage::OFM, nativeTensor)
            .Set(original.shape)
            .Set(original.slice)
            .Set(zeroPointQuantization)
            .Set(original.rounding);
        operation->ConnectInput(TensorUsage::IFM0, nativeTensor)
            .Set(original.shape)
            .Set(original.slice)
            .Set(zeroPointQuantization)
            .Set(original.reverse)
            .Set(original.rounding);
        RecordOptimisation(*operation, copy.get());
    }
    else
    {
        ifm->Set(zeroPointQuantization);
        if ( explicitPad != nullptr )
        {
            const TensorConnection *padIfm = explicitPad->Input(TensorUsage::IFM0);
            const TensorConnection *padParams = explicitPad->Input(TensorUsage::Params);
            const Shape inputShape = padIfm->SliceShape();
            const Shape before = TensorToShape(padParams->tensor.get(), inputShape.Size(), 2, 0);
            const Shape after = TensorToShape(padParams->tensor.get(), inputShape.Size(), 2, 1);
            if ( before.WithHW(0, 0) == before.WithZeros() && after.WithHW(0, 0) == after.WithZeros() )
            {
                int bottom = after.Height();
                int right = after.Width();
                const Point2i dilated = kernel->DilatedWH();
                const auto adjustAfter = [](int input, int stride, int filter, int beforeValue, int afterValue)
                {
                    const int total = NeededTotalPadding(input, stride, filter);
                    const int difference = afterValue % stride - (total - beforeValue) % stride;
                    return std::max(0, afterValue - difference - (difference >= 0 ? 0 : stride));
                };
                bottom = adjustAfter(inputShape.Height(), kernel->Stride().y, dilated.y,
                    before.Height(), bottom);
                right = adjustAfter(inputShape.Width(), kernel->Stride().x, dilated.x,
                    before.Width(), right);
                operation->SetKernel(std::make_unique<Kernel>(kernel->WithPadding(
                    {before.Height(), before.Width(), bottom, right})));
                operation->CopyInput(TensorUsage::IFM0, *padIfm);
                operation->Input(TensorUsage::IFM0)->Set(zeroPointQuantization);
                connectPaddingValue();
                if ( explicitPad->Output(TensorUsage::OFM)->tensor->Readers().empty() ) explicitPad->Disconnect();
            }
            else
            {
                explicitPad->Output(TensorUsage::OFM)->Set(zeroPointQuantization);
                explicitPad->Attribute<pad_attr_t>()->pad_const = ifmZeroPoint;
            }
        }
    }

    auto adjustedBiasTensor = std::shared_ptr<Tensor>(bias->tensor->Clone().release());
    adjustedBiasTensor->SetName(bias->tensor->Name() + "/zero_point_compensated");
    adjustedBiasTensor->SetBuffer(std::make_shared<Buffer>(std::move(adjustedBias)));
    const TensorConnection originalBias = *bias;
    operation->ConnectInput(TensorUsage::Scales, adjustedBiasTensor)
        .Set(originalBias.shape)
        .Set(originalBias.slice)
        .Set(originalBias.quantization)
        .Set(originalBias.reverse)
        .Set(originalBias.rounding);
}

void NeuralAIGraphOptimiser::CanonicalizeConvAdd(Graph *graph, Operation *operation)
{
    if ( operation->Type() != OpType::Add ) return;
    TensorConnection *lhs = operation->Input(TensorUsage::IFM0);
    TensorConnection *rhs = operation->Input(TensorUsage::IFM1);
    TensorConnection *ofm = operation->Output(TensorUsage::OFM);
    if ( !IsRawAddScale(lhs, rhs, ofm) ) return;

    const int64_t delta = ScalarZeroPoint(ofm->quantization) -
        ScalarZeroPoint(lhs->quantization) - ScalarZeroPoint(rhs->quantization);
    if ( delta == 0 ) return;

    const auto isPrivateConvOutput = [graph](const TensorConnection *connection)
    {
        if ( connection == nullptr || !connection->tensor || graph->IsOutput(connection->tensor.get()) ||
             connection->tensor->Writers().size() != 1 || connection->tensor->Readers().size() != 1 ) return false;
        const Operation *writer = connection->tensor->Writers().front().get();
        return writer != nullptr && writer->Type() == OpType::Conv2D;
    };
    const bool lhsCandidate = isPrivateConvOutput(lhs);
    const bool rhsCandidate = isPrivateConvOutput(rhs);
    if ( lhsCandidate == rhsCandidate ) return;

    TensorConnection *candidate = lhsCandidate ? lhs : rhs;
    Operation *producer = candidate->tensor->Writers().front().get();
    TensorConnection *producerOfm = producer->Output(TensorUsage::OFM);
    if ( producerOfm == nullptr || producerOfm->tensor != candidate->tensor ||
         !HasScalarQuantization(producerOfm->quantization) ||
         ScalarZeroPoint(producerOfm->quantization) != ScalarZeroPoint(candidate->quantization) ||
         producerOfm->quantization.quantMin != candidate->quantization.quantMin ||
         producerOfm->quantization.quantMax != candidate->quantization.quantMax ) return;
    const int64_t shiftedZeroPoint = ScalarZeroPoint(producerOfm->quantization) + delta;
    if ( shiftedZeroPoint < -128 || shiftedZeroPoint > 127 ) return;

    Quantization shiftedProducer = producerOfm->quantization;
    Quantization shiftedConsumer = candidate->quantization;
    if ( !ShiftClamp(shiftedProducer, delta) || !ShiftClamp(shiftedConsumer, delta) ) return;
    shiftedProducer.zeroPoints = {shiftedZeroPoint};
    shiftedConsumer.zeroPoints = {ScalarZeroPoint(candidate->quantization) + delta};
    producerOfm->Set(shiftedProducer);
    candidate->Set(shiftedConsumer);
}

void NeuralAIGraphOptimiser::InsertInputConversion(Graph *graph, Operation *operation, TensorUsage usage)
{
    const TensorConnection original = *operation->Input(usage);
    if ( !graph->IsInput(original.tensor.get()) ) return;
    const Kernel *kernel = operation->Kernel();
    TensorConnection *stemOutput = operation->Output(TensorUsage::OFM);
    const bool rgbStemCandidate = operation->Type() == OpType::Conv2D &&
        usage == TensorUsage::IFM0 && original.SliceShape().Depth() == 3 && kernel != nullptr &&
        kernel->Size() == Point2i(3, 3) && kernel->Stride() == Point2i(2, 2) &&
        kernel->Dilation() == Point2i(1, 1) && stemOutput != nullptr &&
        stemOutput->tensor->Readers().size() == 1;
    if ( rgbStemCandidate )
    {
        Operation *activation = stemOutput->tensor->Readers().front().get();
        TensorConnection *activationOutput = activation->Output(TensorUsage::OFM);
        if ( activation->Type() == OpType::LUT && activationOutput != nullptr &&
             activationOutput->tensor->Readers().size() == 1 )
        {
            Operation *consumer = activationOutput->tensor->Readers().front().get();
            const Kernel *consumerKernel = consumer->Kernel();
            if ( consumer->Type() == OpType::Conv2D && consumerKernel != nullptr &&
                 consumerKernel->Size() == Point2i(3, 3) &&
                 consumerKernel->Stride() == Point2i(2, 2) &&
                 consumerKernel->Dilation() == Point2i(1, 1) )
            {
                // This structural stem is scheduled as RGB Conv -> LUT -> Conv.
                // The striped RGB command stages its rows from the public binding.
                return;
            }
        }
    }

    auto nativeTensor = std::shared_ptr<Tensor>(original.tensor->Clone().release());
    nativeTensor->SetName(original.tensor->Name() + "/row32");
    auto copy = std::make_shared<Operation>(OpType::MemoryCopy);
    copy->CopyInput(TensorUsage::IFM, original);
    copy->ConnectOutput(TensorUsage::OFM, nativeTensor)
        .Set(original.shape)
        .Set(original.slice)
        .Set(original.quantization)
        .Set(original.rounding);
    operation->ConnectInput(usage, nativeTensor)
        .Set(original.shape)
        .Set(original.slice)
        .Set(original.quantization)
        .Set(original.reverse)
        .Set(original.rounding);
    RecordOptimisation(*operation, copy.get());
}

void NeuralAIGraphOptimiser::InsertOutputConversion(Graph *graph, Operation *operation)
{
    const TensorConnection original = *operation->Output(TensorUsage::OFM);
    if ( !graph->IsOutput(original.tensor.get()) ) return;

    auto nativeTensor = std::shared_ptr<Tensor>(original.tensor->Clone().release());
    nativeTensor->SetName(original.tensor->Name() + "/row32");
    operation->ConnectOutput(TensorUsage::OFM, nativeTensor)
        .Set(original.shape)
        .Set(original.slice)
        .Set(original.quantization)
        .Set(original.reverse)
        .Set(original.rounding);
    auto copy = std::make_shared<Operation>(OpType::MemoryCopy);
    copy->ConnectInput(TensorUsage::IFM, nativeTensor)
        .Set(original.shape)
        .Set(original.slice)
        .Set(original.quantization)
        .Set(original.reverse)
        .Set(original.rounding);
    copy->CopyOutput(TensorUsage::OFM, original);
    RecordOptimisation(*operation, copy.get());
}

void NeuralAIGraphOptimiser::MaterializeStructuralCspConcatInputs(Operation *operation)
{
    const TensorConnection *ofm = operation->Output(TensorUsage::OFM);
    if ( operation->Type() != OpType::Concat || ofm == nullptr ||
         ofm->shape.Size() != 4 || ofm->shape.Batch() != 1 ||
         ofm->shape.Height() <= 0 || ofm->shape.Width() <= 0 ||
         ofm->shape.Depth() != 48 ||
         operation->Input(TensorUsage::IFM2) == nullptr ) return;

    for ( int index = 0; index < 3; ++index )
    {
        const TensorUsage usage = MakeTensorUsage(TensorUsage::IFM, index);
        const TensorConnection original = *operation->Input(usage);
        const Shape sliceShape = original.SliceShape();
        const Shape sliceOffset = original.slice.offset ?
            original.slice.offset : original.shape.WithZeros();
        if ( sliceShape != ofm->shape.WithDepth(16) ||
             original.tensor->StorageShape().Depth() != 32 ||
             sliceOffset.WithDepth(0) != sliceOffset.WithZeros() ||
             sliceOffset.Depth() != 0 ) continue;

        auto compact = std::make_shared<Tensor>(
            original.tensor->Name() + "/compact_c16", original.tensor->Type(), sliceShape);
        auto copy = std::make_shared<Operation>(OpType::MemoryCopy);
        copy->CopyInput(TensorUsage::IFM, original);
        copy->ConnectOutput(TensorUsage::OFM, compact)
            .Set(sliceShape)
            .Set(original.quantization)
            .Set(original.rounding);
        operation->ConnectInput(usage, compact)
            .Set(sliceShape)
            .Set(original.quantization)
            .Set(original.rounding);
        RecordOptimisation(*operation, copy.get());
    }
}

void NeuralAIGraphOptimiser::FuseStructuralHeadPack(Graph *graph, Operation *operation)
{
    if ( operation->Type() != OpType::Transpose ) return;
    TensorConnection *transposeInput = operation->Input(TensorUsage::IFM0);
    TensorConnection *transposeOutput = operation->Output(TensorUsage::OFM);
    if ( transposeInput == nullptr || transposeOutput == nullptr ||
         transposeInput->tensor->Writers().size() != 1 ) return;

    Operation *concat = transposeInput->tensor->Writers().front().get();
    TensorConnection *lhs = concat->Input(TensorUsage::IFM0);
    TensorConnection *rhs = concat->Input(TensorUsage::IFM1);
    TensorConnection *concatOutput = concat->Output(TensorUsage::OFM);
    if ( concat->Type() != OpType::Concat || lhs == nullptr || rhs == nullptr ||
         concat->Input(TensorUsage::IFM2) != nullptr || concatOutput == nullptr ||
         concatOutput->tensor != transposeInput->tensor ||
         concatOutput->tensor->Readers().size() != 1 || graph->IsOutput(concatOutput->tensor.get()) ) return;

    const Shape lhsShape = Shape::PadAxes(lhs->SliceShape(), 4, 1);
    const Shape rhsShape = Shape::PadAxes(rhs->SliceShape(), 4, 1);
    const Shape mergedShape = Shape::PadAxes(concatOutput->shape, 4, 1);
    const Shape outputShape = Shape::PadAxes(transposeOutput->shape, 4, 1);
    if ( lhsShape.Batch() != 1 || lhsShape.Depth() != 64 || rhsShape != lhsShape.WithDepth(80) ||
         mergedShape != lhsShape.WithDepth(144) ||
         outputShape != Shape(1, 144, lhsShape.Height(), lhsShape.Width()) ||
         !lhs->quantization.EqualScales(rhs->quantization) ||
         !lhs->quantization.EqualScales(concatOutput->quantization) ) return;

    operation->CopyInput(TensorUsage::IFM0, *lhs);
    operation->CopyInput(TensorUsage::IFM1, *rhs);
    concat->Disconnect();
}

void NeuralAIGraphOptimiser::DistributeStructuralHeadMerge(Graph *graph, Operation *operation)
{
    if ( operation->Type() != OpType::Concat ) return;
    TensorConnection *mergeOutput = operation->Output(TensorUsage::OFM);
    TensorConnection *lhs = operation->Input(TensorUsage::IFM0);
    TensorConnection *rhs = operation->Input(TensorUsage::IFM1);
    TensorConnection *tail = operation->Input(TensorUsage::IFM2);
    if ( mergeOutput == nullptr || lhs == nullptr || rhs == nullptr || tail == nullptr ) return;

    const Shape outputShape = Shape::PadAxes(mergeOutput->shape, 4, 1);
    const std::array<TensorConnection *, 3> inputs{lhs, rhs, tail};
    std::array<Shape, 3> inputShapes;
    int locations = 0;
    for ( int index = 0; index < int(inputs.size()); ++index )
    {
        inputShapes[index] = Shape::PadAxes(inputs[index]->SliceShape(), 4, 1);
        if ( inputShapes[index].Batch() != 1 || inputShapes[index].Height() != 1 ||
             inputShapes[index].Width() != 144 || inputShapes[index].Depth() <= 0 ) return;
        locations += inputShapes[index].Depth();
    }
    if ( outputShape != Shape(1, 1, 144, locations) ) return;

    Operation *dfl = nullptr;
    Operation *lut = nullptr;
    auto readers = mergeOutput->tensor->Readers();
    for ( const auto &reader : readers )
    {
        TensorConnection *readerOutput = reader->Output(TensorUsage::OFM);
        if ( reader->Type() == OpType::MemoryCopy && readerOutput != nullptr &&
             !graph->IsOutput(readerOutput->tensor.get()) && readerOutput->tensor->Readers().empty() )
        {
            reader->Disconnect();
        }
    }
    readers = mergeOutput->tensor->Readers();
    if ( readers.size() != 2 ) return;
    for ( const auto &reader : readers )
    {
        if ( reader->Type() == OpType::Dfl ) dfl = reader.get();
        else if ( reader->Type() == OpType::LUT ) lut = reader.get();
    }
    if ( dfl == nullptr || lut == nullptr ) return;

    TensorConnection *dflInput = dfl->Input(TensorUsage::IFM0);
    TensorConnection *dflOutput = dfl->Output(TensorUsage::OFM);
    TensorConnection *lutInput = lut->Input(TensorUsage::IFM0);
    TensorConnection *lutTable = lut->Input(TensorUsage::LUT);
    TensorConnection *lutOutput = lut->Output(TensorUsage::OFM);
    if ( dflInput == nullptr || dflOutput == nullptr || lutInput == nullptr ||
         lutTable == nullptr || lutOutput == nullptr || dflInput->tensor != mergeOutput->tensor ||
         lutInput->tensor != mergeOutput->tensor || !lutTable->tensor->IsConstant() ||
         Shape::PadAxes(dflInput->SliceShape(), 4, 1) != outputShape ||
         Shape::PadAxes(dflOutput->shape, 4, 1) != Shape(1, 1, 4, locations) ||
         Shape::PadAxes(lutInput->SliceShape(), 4, 1) != Shape(1, 1, 80, locations) ||
         Shape::PadAxes(lutOutput->shape, 4, 1) != Shape(1, 1, 80, locations) ) return;
    const Shape lutOffset = lutInput->slice.offset ?
        Shape::PadAxes(lutInput->slice.offset, 4, 0) : outputShape.WithZeros();
    const Shape lutStride = lutInput->slice.stride ?
        Shape::PadAxes(lutInput->slice.stride, 4, 1) : outputShape.WithOnes();
    if ( lutOffset != Shape(0, 0, 64, 0) || lutStride != lutStride.WithOnes() ) return;

    std::array<std::shared_ptr<Tensor>, 3> dflParts;
    std::array<std::shared_ptr<Tensor>, 3> classParts;
    std::array<std::shared_ptr<Operation>, 3> dflOperations;
    for ( int index = 0; index < int(inputs.size()); ++index )
    {
        const int partLocations = inputShapes[index].Depth();
        const Shape dflShape = Shape(1, 1, 4, partLocations);
        dflParts[index] = std::make_shared<Tensor>(
            mergeOutput->tensor->Name() + "/dfl_part" + std::to_string(index),
            dflOutput->tensor->Type(), dflShape);
        dflOperations[index] = std::make_shared<Operation>(OpType::Dfl);
        dflOperations[index]->CopyInput(TensorUsage::IFM0, *inputs[index]);
        dflOperations[index]->ConnectOutput(TensorUsage::OFM, dflParts[index])
            .Set(dflShape)
            .Set(dflOutput->quantization)
            .Set(dflOutput->rounding);

        const Shape classShape = Shape(1, 1, 80, partLocations);
        classParts[index] = std::make_shared<Tensor>(
            mergeOutput->tensor->Name() + "/class_part" + std::to_string(index),
            lutOutput->tensor->Type(), classShape);
        dflOperations[index]->CopyInput(TensorUsage::LUT, *lutTable);
        dflOperations[index]->ConnectOutput(MakeTensorUsage(TensorUsage::OFM, 1), classParts[index])
            .Set(classShape)
            .Set(lutOutput->quantization)
            .Set(lutOutput->rounding);
        RecordOptimisation(*operation, dflOperations[index].get());
    }

    auto dflMerge = std::make_shared<Operation>(OpType::Concat);
    auto classMerge = std::make_shared<Operation>(OpType::Concat);
    for ( int index = 0; index < int(inputs.size()); ++index )
    {
        const TensorUsage usage = MakeTensorUsage(TensorUsage::IFM, index);
        dflMerge->ConnectInput(usage, dflParts[index])
            .Set(Shape(1, 1, 4, inputShapes[index].Depth()))
            .Set(dflOutput->quantization);
        classMerge->ConnectInput(usage, classParts[index])
            .Set(Shape(1, 1, 80, inputShapes[index].Depth()))
            .Set(lutOutput->quantization);
    }
    dflMerge->CopyOutput(TensorUsage::OFM, *dflOutput);
    dflMerge->Attribute<axis_attr_t>()->axis = -1;
    classMerge->CopyOutput(TensorUsage::OFM, *lutOutput);
    classMerge->Attribute<axis_attr_t>()->axis = -1;
    RecordOptimisation(*operation, dflMerge.get());
    RecordOptimisation(*operation, classMerge.get());

    dfl->Disconnect();
    lut->Disconnect();
    operation->Disconnect();
}

void NeuralAIGraphOptimiser::OptimiseGraph(Graph *graph)
{
    std::vector<std::shared_ptr<Operation>> operations;
    graph->GetAllOperations(operations);
    if ( _stage == NeuralAIGraphOptimiserStage::Prepare )
    {
        for ( const auto &operation : operations ) CanonicalizeAsymmetricConv(graph, operation.get());
        return;
    }
    for ( const auto &operation : operations ) CanonicalizeConvAdd(graph, operation.get());
    for ( const auto &operation : operations ) DistributeStructuralHeadMerge(graph, operation.get());
    operations.clear();
    graph->GetAllOperations(operations);
    for ( const auto &operation : operations ) FuseStructuralHeadPack(graph, operation.get());
    operations.clear();
    graph->GetAllOperations(operations);
    for ( const auto &operation : operations ) MaterializeStructuralCspConcatInputs(operation.get());
    for ( const auto &operation : operations )
    {
        if ( operation->Type() != OpType::FullyConnected && operation->Type() != OpType::MatMul &&
             operation->Type() != OpType::Conv2D && operation->Type() != OpType::DepthwiseConv2D &&
             operation->Type() != OpType::LUT && operation->Type() != OpType::Add &&
             operation->Type() != OpType::AvgPool && operation->Type() != OpType::Resize &&
             operation->Type() != OpType::MaxPool && operation->Type() != OpType::Concat &&
             operation->Type() != OpType::Transpose && operation->Type() != OpType::Dfl ) continue;
        InsertInputConversion(graph, operation.get(), TensorUsage::IFM0);
        if ( operation->Type() == OpType::Concat && operation->Input(TensorUsage::IFM2) != nullptr )
            InsertInputConversion(graph, operation.get(), TensorUsage::IFM2);
        if ( operation->Type() == OpType::Add || operation->Type() == OpType::Concat ||
             (operation->Type() == OpType::Transpose && operation->Input(TensorUsage::IFM1) != nullptr) )
            InsertInputConversion(graph, operation.get(), TensorUsage::IFM1);
        InsertOutputConversion(graph, operation.get());
    }
}

}  // namespace regor
