//
// SPDX-FileCopyrightText: Copyright 2021-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "high_level_command_stream_generator.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "common/box.hpp"
#include "common/numeric_util.hpp"
#include "common/vector_span.hpp"
#include "high_level_command_stream.hpp"
#include "scheduler.hpp"

#include <unordered_map>
#include <vector>

namespace regor
{


static void CalcPaddingAndSkirt(const Kernel *kernel, const Shape &inputShape, const Shape &outputShape, HLCPadding &padding, HLCPadding &skirt)
{
    auto dilatedWH = kernel->DilatedWH();
    int ypad = NeededTotalPadding(inputShape.Height(), outputShape.Height(), kernel->Stride().y, dilatedWH.y);
    int xpad = NeededTotalPadding(inputShape.Width(), outputShape.Width(), kernel->Stride().x, dilatedWH.x);
    const auto &pad = kernel->Padding();
    padding.left = pad.Left();
    padding.right = pad.Right();
    padding.top = pad.Top();
    padding.bottom = pad.Bottom();
    skirt.top = padding.top;
    skirt.left = padding.left;
    skirt.bottom = std::max(ypad - padding.top, dilatedWH.y - 1);
    skirt.right = std::max(xpad - padding.left, dilatedWH.x - 1);
}

enum class TransformLimit
{
    None,
    Wrap,
};

static Box TransformWithStridesAndSkirt(const Box &outputArea, const Shape *strides, const HLCPadding *skirt, const Shape &ifmShape,
    OpType opType, const Shape &concatOffsets, const Shape &splitOffset, const Shape &splitShape, int dilatedKernelHeight, int upscalingFactor,
    int &padTop, int &padBottom, TransformLimit limit = TransformLimit::None, TransposeType transposeType = TransposeType::None)
{
    Shape outputAreaStart = outputArea.Start().Untranspose(transposeType);
    Shape outputAreaEnd = outputArea.End().Untranspose(transposeType);
    Shape concatOffsetsUntransposed = concatOffsets.Untranspose(transposeType);
    Shape outputAreaSize = outputAreaEnd - outputAreaStart;
    // Make start/end at least 4 dimensional
    Shape start = Shape::Max(outputAreaStart - concatOffsetsUntransposed, Shape(0, 0, 0, 0));
    Shape end = Shape::Max(start + outputAreaSize, Shape(1, 1, 1, 1));
    start += splitOffset;
    end += splitOffset;
    if ( (IsConvolution(opType) && !IsDepthwise(opType)) )
    {
        if ( splitOffset.Size() == 0 )
        {
            start = start.WithDepth(0);
            end = end.WithDepth(ifmShape.Depth());
        }
        else
        {
            start = start.WithDepth(splitOffset.Depth());
            end = end.WithDepth(start.Depth() + splitShape.Depth());
        }
    }
    else if ( IsVectorProduct(opType) || opType == OpType::ReduceSum )
    {
        // these types of operations do a "dot product" or sum over the entire IFM - full shape needed
        if ( splitOffset.Size() == 0 )
        {
            start = Shape(0, 0, 0, 0);
            end = Shape::PadAxes(ifmShape, 4, 1);
        }
        else
        {
            start = splitOffset;
            end = start + splitShape;
        }
    }
    else if ( opType == OpType::Resize )
    {
        // TODO MLBEDSW-8660: Striping of resize operations
        return (splitOffset.Size() > 0) ? Box(splitOffset, splitOffset + splitShape) : Box(Shape::PadAxes(ifmShape, 4, 1));
    }
    else if ( IsBinaryElementwise(opType) && splitOffset.Size() != 0 )
    {
        // Elementwise with splitShape. IFM might not cover full OFM and can be broadcasted in that case
        start = splitOffset;
        end = start + splitShape;
    }
    end = Shape::Min(end, Shape::Max(ifmShape, Shape(1, 1, 1, 1)).WithHW(ifmShape.Height() * upscalingFactor, ifmShape.Width() * upscalingFactor));
    padTop = 0;
    padBottom = 0;

    assert(strides != nullptr && skirt != nullptr);
    assert(strides->Size() == 4);

    int strideW = strides->Width();
    int validIfmOffset = splitOffset.IsEmpty() ? 0 : splitOffset.Width();
    start = start.WithWidth(std::max(start.Width() * strideW - skirt->left, validIfmOffset));
    int validIfmWidth = ifmShape.Width();
    if ( splitShape.Size() > 1 && splitOffset.Size() > 1 )
        validIfmWidth = std::min(validIfmWidth, splitShape.Width() + splitOffset.Width());
    else if ( splitShape.Size() > 1 ) validIfmWidth = splitShape.Width();
    end = end.WithWidth(std::min(end.Width() * strideW + skirt->right, validIfmWidth));
    int strideH = strides->Height();
    int skirtTopRemainder = skirt->top % upscalingFactor;
    int totalStride = strideH * (outputAreaEnd.Height() - outputAreaStart.Height() - 1);
    int startHeight = start.Height() * strideH - skirt->top + skirtTopRemainder;
    padTop = std::max(0, -startHeight) + skirtTopRemainder;
    start = start.WithHeight(std::max(startHeight, 0));
    if ( end.Height() * strideH + skirt->bottom > ifmShape.Height() * upscalingFactor )
    {
        // padBottom is calculated based the diff between the end position of the weight kernel,
        // after last stride and the ifm height.
        if ( upscalingFactor != 1 && outputAreaEnd.Height() > ifmShape.Height() * upscalingFactor )
        {
            // Special case for Transpose Convolution with VALID padding.
            padBottom = outputAreaEnd.Height() - ifmShape.Height() * upscalingFactor;
        }
        else
        {
            int kernelStart = start.Height() - padTop;
            padBottom = std::max(0, kernelStart + totalStride + dilatedKernelHeight - ifmShape.Height() * upscalingFactor);
        }
    }
    // Adjust for upscaling
    start = start.WithHeight(std::max(start.Height() / upscalingFactor, 0));
    int endHeight = end.Height() * strideH + skirt->bottom + skirt->bottom % upscalingFactor;
    end = end.WithHeight(std::min(std::max(endHeight / upscalingFactor, 1), ifmShape.Height()));

    if ( limit == TransformLimit::Wrap )
    {
        Shape ifmWrap = Shape::PadAxes(ifmShape, 4, 1);
        Shape one(1, 1, 1, 1);
        start = Shape::Wrap(start, ifmWrap);
        end = Shape::Wrap(end - one, ifmWrap) + one;
        assert((end - start).Elements() > 0);
    }

    return Box(start, end);
}

static int StrideAdjustedPadding(int pad, int outOffset, int kernelStride)
{
    return pad - outOffset * (kernelStride - 1);
}

static int MarginForKernel(int paddedInput, int dilatedSize, int stride)
{
    assert(stride > 0);
    int margin = paddedInput % stride;
    if ( margin == 0 )
    {
        margin += stride;
    }
    return std::max(0, dilatedSize - margin);
}

static std::pair<Box, HLCPadding> TransformWithInputOutputSteps(const Box &inputArea, const Point2i &inputStep,
    const Box &outputArea, const Point2i &outputStep, class Kernel *kernel, const HLCPadding &padding)
{
    const auto &stride = kernel->Stride();
    const auto dilatedWH = kernel->DilatedWH();
    HLCPadding newPadding;
    auto adjustedTopPad = StrideAdjustedPadding(padding.top, outputArea.Start().Height(), stride.y);
    auto adjustedLeftPad = StrideAdjustedPadding(padding.left, outputArea.Start().Width(), stride.x);
    newPadding.top = std::max(0, DivRoundUp(adjustedTopPad, inputStep.y));
    newPadding.left = std::max(0, DivRoundUp(adjustedLeftPad, inputStep.x));
    Point2i startAdjustForPadFraction;
    startAdjustForPadFraction.x = newPadding.left * inputStep.x - adjustedLeftPad;
    startAdjustForPadFraction.y = newPadding.top * inputStep.y - adjustedTopPad;
    Point2i neededInput;
    neededInput.x = DivRoundUp(outputArea.End().Width() - outputArea.Start().Width(), outputStep.x) * stride.x;
    neededInput.x += MarginForKernel(neededInput.x, dilatedWH.x, stride.x);
    neededInput.y = DivRoundUp(outputArea.End().Height() - outputArea.Start().Height(), outputStep.y) * stride.y;
    neededInput.y += MarginForKernel(neededInput.y, dilatedWH.y, stride.y);
    Shape newStart =
        inputArea.Start()
            .WithWidth(inputArea.Start().Width() + startAdjustForPadFraction.x)
            .WithHeight(inputArea.Start().Height() + startAdjustForPadFraction.y);
    newPadding.bottom = std::max(0,
        neededInput.y -
            (DivRoundUp((inputArea.End().Height() - inputArea.Start().Height()) - startAdjustForPadFraction.y, inputStep.y) +
                newPadding.top));
    newPadding.right = std::max(0,
        neededInput.x -
            (DivRoundUp(inputArea.End().Width() - inputArea.Start().Width() - startAdjustForPadFraction.x, inputStep.x) +
                newPadding.left));
    return std::make_pair<Box, HLCPadding>(Box(newStart, inputArea.End()), std::move(newPadding));
}

// Calculates STRIDE_C/Y/X
static Shape GetStrides(const HLCFeatureMap &fm)
{
    auto elemSize = DataTypeSizeBits(fm.dataType) / 8;
    if ( fm.format == TensorFormat::NHWC )
    {
        int strideC = elemSize;
        int strideX = fm.shape.Depth() * strideC;
        int strideY = fm.shape.Width() * strideX;
        int strideN = fm.shape.Height() * strideY;
        return Shape(strideN, strideY, strideX, strideC);
    }
    else if ( fm.format == TensorFormat::NHCWB16 )
    {
        int strideX = 16 * elemSize;
        int strideC = strideX * fm.shape.Width();
        int strideY = elemSize * fm.shape.Width() * RoundAway(fm.shape.Depth(), 16);
        int strideN = fm.shape.Height() * strideY;
        return Shape(strideN, strideY, strideX, strideC);
    }
    else
    {
        assert(false && "Unsupported tensor format");
        return Shape(0, 0, 0, 0);
    }
}

static void MakeFeatureMap(const SchedulerConnection *schedConn, HLCFeatureMap &fm)
{
    auto schedTens = schedConn->tensor.get();
    fm.shape = schedConn->shape;
    fm.slice = schedConn->slice;
    fm.dataType = schedTens->dataType;
    fm.memArea = schedTens->memArea;
    fm.format = schedTens->format;
    fm.address = schedTens->allocatedAddress;
    fm.quantization = schedConn->quantization;
    fm.bufferView = schedTens->bufferView;
    fm.strides = GetStrides(fm);
    fm.stepXY = schedConn->stepXY;
    fm.transpose = schedConn->transpose;
    fm.reverse = schedConn->reverse;
    fm.resamplingMode = schedConn->resamplingMode;
    fm.uid = schedConn->tensor->uid;
}

static std::unique_ptr<HLCWeights> MakeWeights(NpuWeightTensor *srcTensor, Buffering buffering, SchedulerTensor *bufTensor = nullptr)
{
    auto weights = std::make_unique<HLCWeights>();
    if ( buffering == Buffering::None )
    {
        assert(!bufTensor);
    }
    if ( bufTensor == nullptr )
    {
        bufTensor = srcTensor;
    }
    weights->address = bufTensor->allocatedAddress;
    weights->memArea = bufTensor->memArea;
    weights->buffering = buffering;
    // Same function is used for generating scales - scales have no config or weight format, so set to default
    weights->format = srcTensor->config ? srcTensor->config->Format() : Flags(WeightFormat::Default);
    weights->doubleBufferOffset = srcTensor->doubleBufferOffset;
    weights->subStreams = srcTensor->subStreams;
    weights->encodedRanges = srcTensor->encodedRanges;
    return weights;
}

static HLCSubOperation MakeSubOperation(const std::unique_ptr<SchedulerOperation> &schedOp)
{
    HLCSubOperation hlcSubOp;
    hlcSubOp.type = schedOp->Type();
    auto lutConn = schedOp->TryInput(TensorUsage::LUT);

    for ( int i = 0; i < MAX_NUM_IFM; ++i )
    {
        auto ifm = schedOp->TryIFM(i);
        if ( ifm != nullptr )
        {
            hlcSubOp.ifm.emplace_back();
            MakeFeatureMap(ifm, hlcSubOp.ifm.back());
        }
    }
    MakeFeatureMap(schedOp->OFM(), hlcSubOp.ofm);
    hlcSubOp.rounding = HLCRoundMode(schedOp->Rounding());
    hlcSubOp._srcKey = schedOp->_srcKey;

    if ( schedOp->Type() == OpType::LeakyRelu )
    {
        const auto *parameters = schedOp->Attribute<leaky_relu_attr_t>();
        hlcSubOp.parameters.leaky_relu.alpha = parameters->alpha;
    }
    else if ( lutConn != nullptr )
    {
        auto lutTensor = lutConn->tensor;
        auto &param = hlcSubOp.parameters.lut;
        param.memArea = lutTensor->memArea;
        param.address = lutTensor->allocatedAddress;
        param.sizeBytes = lutTensor->AllocationSizeBytes();
        param.ifmType = schedOp->IFM(0)->tensor->dataType;
    }
    return hlcSubOp;
}

static std::shared_ptr<HLCOperation> MakeOperation(SchedulerOperation *schedOp, SchedulerOpInfo *opInfo)
{
    assert(opInfo);
    auto op = std::make_shared<HLCOperation>();
    op->type = schedOp->Type();
    op->kernel = *schedOp->Kernel();
    op->config = opInfo->Config();
    op->rounding = HLCRoundMode(schedOp->Rounding());
    op->_srcKey = schedOp->_srcKey;

    for ( int i = 0; i < MAX_NUM_IFM; ++i )
    {
        auto ifm = schedOp->TryIFM(i);
        if ( ifm != nullptr )
        {
            HLCFeatureMap fm;
            MakeFeatureMap(ifm, fm);
            op->ifm.push_back(fm);
        }
    }
    MakeFeatureMap(schedOp->OFM(), op->ofm);
#ifndef NDEBUG
    op->name = schedOp->OFM()->tensor->Name();
#endif
    if ( opInfo->npuWeightsTensor != nullptr )
    {
        assert(schedOp->TryInput(TensorUsage::Weights) != nullptr);
        op->weights = MakeWeights(opInfo->npuWeightsTensor.get(), opInfo->bufferedWeightTensor.buffering,
            opInfo->bufferedWeightTensor.tensor.get());
    }

    if ( opInfo->npuScalesTensor != nullptr )
    {
        // Only scales encoded
        op->scales = MakeWeights(opInfo->npuScalesTensor.get(), Buffering::None);
    }
    else if ( schedOp->TryInput(TensorUsage::Scales) != nullptr )
    {
        // Weights and scales encoded together
        assert(!!opInfo->npuWeightsTensor);
        op->scales = MakeWeights(opInfo->npuWeightsTensor.get(), opInfo->bufferedWeightTensor.buffering,
            opInfo->bufferedWeightTensor.tensor.get());
    }

    // Register command stream generator will allocate the LUT
    // in LUT memory and generate DMA for the LUT; for this
    // it must know the location of the tensor in read-only memory
    auto lutConn = schedOp->TryInput(TensorUsage::LUT);
    if ( lutConn != nullptr )
    {
        auto lutTensor = lutConn->tensor;
        auto &param = op->parameters.lut;
        param.memArea = lutTensor->memArea;
        param.address = lutTensor->allocatedAddress;
        param.sizeBytes = lutTensor->AllocationSizeBytes();
        param.ifmType = schedOp->IFM(0)->tensor->dataType;
    }

    for ( auto &subOp : schedOp->SubOps() )
    {
        HLCSubOperation hlcSubOp = MakeSubOperation(subOp);
        op->subOps.push_back(std::move(hlcSubOp));
    }

    const auto &ifmShape = schedOp->IFM(0)->shape;
    switch ( schedOp->Type() )
    {
        case OpType::LeakyRelu:
        {
            assert(lutConn == nullptr);
            const auto *lrelu = schedOp->Attribute<leaky_relu_attr_t>();
            op->parameters.leaky_relu.alpha = lrelu->alpha;
        }
        break;
        case OpType::Resize:
        {
            assert(lutConn == nullptr);
            const auto *resize = schedOp->Attribute<resize_attr_t>();
            op->parameters.resize.scaleY = resize->scaleY;
            op->parameters.resize.scaleX = resize->scaleX;
            op->parameters.resize.offsetY = resize->offset.y;
            op->parameters.resize.offsetX = resize->offset.x;
            if ( ifmShape.Width() == 1 && ifmShape.Height() == 1 )
            {
                // 1x1 IFMs can be handled with replicate
                op->parameters.resize.mode = ArchResizeMode::Replicate;
            }
            else if ( resize->mode == tosa::ResizeMode::NEAREST )
            {
                op->parameters.resize.mode = ArchResizeMode::Nearest;
            }
            else
            {
                op->parameters.resize.mode = ArchResizeMode::Bilinear;
            }
        }
        break;
        case OpType::ArgMax:
        {
            assert(lutConn == nullptr);
            const auto *axis = schedOp->Attribute<axis_attr_t>();
            op->parameters.argmax.axis = axis->axis;
        }
        break;
        default:
            break;
    }
    return op;
}

// Finds the next stripe command in the stream
static HLCStripe *FindNextStripe(HLCStream &cmds, int fromIndex)
{
    int sz = int(cmds.size());
    for ( int i = fromIndex; i < sz; ++i )
    {
        if ( cmds[i]->IsStripe() )
        {
            return static_cast<HLCStripe *>(cmds[i].get());
        }
    }
    assert(fromIndex != 0);  // Every stream should contain at least one stripe
    return nullptr;
}

// Generates DMA command for Scatter/Gather
void HLCStreamGenerator::GenerateHLCDMACommands(SchedulerOperation *op, const std::shared_ptr<HLCOperation> &hlcOp, HLCStream &cmds)
{
    UNUSED(op);

    auto opType = hlcOp->type;
    assert(opType == OpType::Scatter || opType == OpType::Gather);

    int ifmSrc = 0;

    if ( opType == OpType::Scatter )
    {
        auto &ifm = hlcOp->ifm[0];  // GraphIR Scatter values_in
        auto &ofm = hlcOp->ofm;     // GraphIR Scatter values_out
        assert(ifm.AllocationSizeBytes() == ofm.AllocationSizeBytes());

        // Generate HLCDMA that copies values_in to values_out
        auto dma = std::make_unique<HLCDMA>();
        dma->srcMemArea = ifm.memArea;
        dma->srcAddress = ifm.address;
        dma->srcStrides = GetStrides(ifm);
        dma->destMemArea = ofm.memArea;
        dma->destAddress = ofm.address;
        dma->destStrides = GetStrides(ofm);
        dma->length = ifm.AllocationSizeBytes();

        cmds.push_back(std::move(dma));

        ifmSrc = 2;
    }

    auto &valFm = hlcOp->ifm[0];       // GraphIR Scatter values_in or GraphIR Gather values
    auto &idxFm = hlcOp->ifm[1];       // GraphIR Scatter indicies or GraphIR Gather indices
    auto &srcFm = hlcOp->ifm[ifmSrc];  // GraphIR Scatter input or GraphIR Gather values
    auto &ofm = hlcOp->ofm;            // GraphIR Scatter values_out or GraphIR Gather output
    assert(idxFm.dataType == DataType::Int32 || idxFm.dataType == DataType::Int64);
    assert(srcFm.dataType == ofm.dataType);

    // Generate HLCDMA that scatters or gathers
    auto dma = std::make_unique<HLCDMA>();
    dma->srcMemArea = srcFm.memArea;
    dma->srcAddress = srcFm.address;
    dma->srcIndexed = (opType == OpType::Gather);
    dma->idxMemArea = idxFm.memArea;
    dma->idxAddress = idxFm.address;
    dma->destMemArea = ofm.memArea;
    dma->destAddress = ofm.address;
    dma->destIndexed = (opType == OpType::Scatter);
    dma->length = DataTypeStorageSizeBytes(srcFm.dataType, srcFm.shape[-1]);
    dma->idxMax = valFm.shape[-2] - 1;

    auto srcStrides = GetStrides(srcFm);
    auto destStrides = GetStrides(ofm);

    if ( opType == OpType::Scatter && idxFm.dataType == DataType::Int64 )
    {
        // Do scatter in 3D mode with index skip because HW can only use int32 indicies
        dma->srcStrides = Shape(srcStrides[-2], 0, srcStrides[-1]);
        dma->destStrides = Shape(0, destStrides[-2], destStrides[-1]);
        dma->sizes = idxFm.shape.Extract({-1, -2});
        dma->idxSkip1 = 4;
    }
    else if ( opType == OpType::Gather && idxFm.dataType == DataType::Int64 )
    {
        // Do gather in 3D mode with index skip because HW can only use int32 indicies
        dma->srcStrides = Shape(0, srcStrides[-2], srcStrides[-1]);
        dma->destStrides = Shape(destStrides[-2], 0, destStrides[-1]);
        dma->sizes = idxFm.shape.Extract({-1, -2});
        dma->idxSkip1 = 4;
    }
    else
    {
        // Do scatter or gather in 2D mode
        dma->destStrides = std::move(destStrides);
        dma->srcStrides = std::move(srcStrides);
        dma->sizes = idxFm.shape.Extract({-2, -1});
        dma->idxSkip1 = 0;
    }

    cmds.push_back(std::move(dma));
}

// Generates DMA command for weights
static std::unique_ptr<HLCDMA> GenerateWeightDMA(NpuWeightTensor *weightTens, const SchedulerConnection &bufConn, int depth, int depthIndex)
{
    auto dma = std::make_unique<HLCDMA>();
    dma->srcMemArea = weightTens->memArea;
    dma->srcAddress = weightTens->allocatedAddress;
    dma->length = 0;
    int offset0 = 0;  // offset of the first substream
    for ( int subStream = 0; subStream < weightTens->subStreams; ++subStream )
    {
        auto item = weightTens->encodedRanges.find(WeightKey(subStream, depth));
        if ( item == weightTens->encodedRanges.end() )
        {
            assert(subStream > 0);
        }
        else
        {
            if ( subStream == 0 )
            {
                offset0 = item->second.offset;
                dma->srcAddress += offset0;
            }
            dma->length = RoundAway(item->second.offset + item->second.TotalBytes() - offset0, 16);
        }
    }
    dma->destMemArea = bufConn.tensor->memArea;
    dma->destAddress = bufConn.tensor->allocatedAddress;
    if ( bufConn.buffering == Buffering::Double && depthIndex % 2 == 1 )
    {
        dma->destAddress += weightTens->doubleBufferOffset;
    }
    return dma;
}

void HLCStreamGenerator::GenerateHLCStripeCommands(SchedulerOperation *op, const std::shared_ptr<HLCOperation> &hlcOp, HLCStream &cmds)
{
    auto opInfo = _schedule->Cost(op);
    HLCPadding skirt;
    HLCPadding padding;
    auto kernel = op->Kernel();
    assert(kernel != nullptr && "Operators must have a kernel");
    Shape strides = Shape(1, kernel->Stride().y, kernel->Stride().x, 1);
    auto opType = op->Type();
    auto ofmConn = op->OFM();
    auto ifm0Conn = op->IFM(0);
    auto opGroup = op->OpGroup();
    const auto &ofmShape = ofmConn->SliceShape();
    const auto &ifm0Shape = ifm0Conn->SliceShape();

    auto *ifm1Conn = op->TryIFM(1);
    auto maxIfmShape = ifm0Shape;
    if ( ifm1Conn && IsBinaryElementwise(opType) )
    {
        // Use full ifm shape for broadcast elementwise operators
        maxIfmShape = Shape::Max(ifm0Conn->SliceShape(), ifm1Conn->SliceShape());
    }
    CalcPaddingAndSkirt(kernel, maxIfmShape, ofmShape, padding, skirt);

    int upscaling = 1;
    if ( ifm0Conn->resamplingMode == ArchResampling::Zeros )
    {
        upscaling = 2;
    }
    auto &depthSlices = opInfo->ofmDepthSlices;
    int dilatedKernelHeight = kernel->DilatedWH().y;

    // Define Start and End coordinates for the OFM
    auto ofmStart = Shape(0, 0, 0, depthSlices[0]);
    auto ofmEnd = Shape::PadAxes(ofmShape, 4, 1);
    if ( ofmConn->slice.offset.Size() > 0 )
    {
        ofmStart = ofmConn->slice.offset;
        ofmEnd = ofmConn->slice.offset + ofmConn->slice.shape;
    }
    assert(hlcOp->ifm.size() <= 2);

    // Binary elementwise using broadcast to repeat smaller IFMs over larger IFM volumes need their
    // coordinates to wrap at the limits of the smaller IFM volume.
    TransformLimit ifmLimit = IsBinaryElementwise(op->Type()) ? TransformLimit::Wrap : TransformLimit::None;

    const auto &ofmStep = opInfo->stripe;
    for ( int startHeight = ofmStart.Height(); startHeight < ofmEnd.Height(); startHeight += ofmStep.Height() )
    {
        int endHeight = std::min(startHeight + ofmStep.Height(), ofmEnd.Height());
        for ( int startWidth = ofmStart.Width(); startWidth < ofmEnd.Width(); startWidth += ofmStep.Width() )
        {
            int endWidth = std::min(startWidth + ofmStep.Width(), ofmEnd.Width());
            for ( int depthIndex = 0; depthIndex < int(depthSlices.size()) - 1; ++depthIndex )
            {
                int startChannel = std::max(depthSlices[depthIndex], ofmStart.Depth());
                int endChannel = std::min(depthSlices[depthIndex + 1], ofmEnd.Depth());

                // Construct the output area for the current stripe
                auto outputAreaStart = Shape(ofmStart.Batch(), startHeight, startWidth, startChannel);
                auto outputAreaEnd = Shape(ofmEnd.Batch(), endHeight, endWidth, endChannel);
                auto outputArea = Box(outputAreaStart, outputAreaEnd);
                auto hlcStripe = std::make_unique<HLCStripe>(hlcOp);
                hlcStripe->padding = padding;
                hlcStripe->ofmArea = outputArea;
                hlcStripe->opGroup = opGroup;
                for ( unsigned ifmIndex = 0; ifmIndex < hlcOp->ifm.size(); ++ifmIndex )
                {
                    auto ifmConn = op->IFM(ifmIndex);
                    // Calculate input area based on the output area
                    auto inputArea = TransformWithStridesAndSkirt(outputArea, &strides, &skirt, ifmConn->shape, opType,
                        ofmConn->slice.offset, ifmConn->slice.offset, ifmConn->slice.shape, dilatedKernelHeight,
                        upscaling, hlcStripe->padding.top, hlcStripe->padding.bottom, ifmLimit, ofmConn->transpose);
                    if ( ofmConn->stepXY != Point2i{1, 1} || ifmConn->stepXY != Point2i{1, 1} )
                    {
                        std::tie(inputArea, hlcStripe->padding) = TransformWithInputOutputSteps(
                            inputArea, ifmConn->stepXY, outputArea, ofmConn->stepXY, kernel, hlcStripe->padding);
                    }
                    hlcStripe->ifmAreas.push_back(inputArea);
                }
                if ( opInfo->npuWeightsTensor != nullptr )
                {
                    hlcStripe->weightRangeDepth = startChannel;
                    if ( opInfo->bufferedWeightTensor.tensor != nullptr &&
                         (startHeight == ofmStart.Height() || opInfo->bufferedWeightTensor.buffering == Buffering::Double) )
                    {
                        assert(opInfo->npuWeightsTensor->config->DepthOffsets().size() == depthSlices.size());
                        // Metadata of new weights to put into the weight buffer tensor
                        auto newWeights = std::make_tuple(opInfo->npuWeightsTensor->equivalenceId, startChannel, depthIndex);
                        if ( _filledWeightBuffers.count(opInfo->bufferedWeightTensor.tensor.get()) == 0 )
                        {
                            // There is nothing in the weights buffer tensor yet
                            cmds.push_back(GenerateWeightDMA(opInfo->npuWeightsTensor.get(),
                                opInfo->bufferedWeightTensor, startChannel, depthIndex));
                        }
                        else
                        {
                            auto &currentWeights = _filledWeightBuffers[opInfo->bufferedWeightTensor.tensor.get()];
                            if ( currentWeights != newWeights )
                            {
                                // There is something in the weights buffer tensor, but it's not correct
                                cmds.push_back(GenerateWeightDMA(opInfo->npuWeightsTensor.get(),
                                    opInfo->bufferedWeightTensor, startChannel, depthIndex));
                            }
                        }
                        _filledWeightBuffers[opInfo->bufferedWeightTensor.tensor.get()] = newWeights;
                    }
                }
                else if ( opInfo->npuScalesTensor != nullptr )
                {
                    hlcStripe->weightRangeDepth = startChannel;
                }
                else
                {
                    hlcStripe->weightRangeDepth = -1;
                }
                cmds.push_back(std::move(hlcStripe));
            }
        }
    }
}

void HLCStreamGenerator::GenerateCommands(SchedulerOperation *op, const std::shared_ptr<HLCOperation> &hlcOp, HLCStream &cmds)
{
    auto opType = op->Type();

    if ( opType == OpType::Scatter || opType == OpType::Gather )
    {
        GenerateHLCDMACommands(op, hlcOp, cmds);
    }
    else
    {
        GenerateHLCStripeCommands(op, hlcOp, cmds);
    }
}

void HLCStreamGenerator::GenerateCommandsForCascade(vector_span<std::unique_ptr<SchedulerOperation>> cascadedOps,
    vector_span<std::shared_ptr<HLCOperation>> hlcOps, const CascadeInfo *cascadeInfo, HLCStream &cmds)
{
    // High level command stream for each individual operation
    std::vector<HLCStream> cmdsForOps;
    std::vector<int> currIndex;
    // Performed stripe at each operation
    std::vector<HLCStripe *> availableStripe;
    // Next stripe to be performed at each operation
    std::vector<HLCStripe *> nextStripe;
    int nrOps = cascadedOps.size();
    assert(cascadeInfo != nullptr);
    // Apply intermediate feature map shapes to cascaded operations
    for ( int i = 1; i < nrOps; ++i )
    {
        auto item = cascadeInfo->buffers.find(*cascadedOps[i]);
        if ( item == cascadeInfo->buffers.end() )
        {
            assert(false);
        }
        else
        {
            auto &shape = item->second.shape;
            hlcOps[i - 1]->ofm.shape = shape;
            // TODO MLBEDSW-9142: support fused activations inside chains
            // for now, we assume maximum one subOp (fused activation) on cascades
            if ( hlcOps[i - 1]->subOps.size() )
            {
                assert(hlcOps[i - 1]->subOps.size() == 1);
                assert(hlcOps[i - 1]->subOps[0].ifm.size() == 1);
                hlcOps[i - 1]->subOps[0].ifm[0].shape = shape;
                hlcOps[i - 1]->subOps[0].ofm.shape = shape;
            }
            hlcOps[i]->ifm[cascadedOps[i]->PrimaryIfmIndex()].shape = shape;
        }
    }
    // Generate high level commands for every operation in the cascade;
    // keep the generated streams in separate lists
    for ( int i = 0; i < nrOps; ++i )
    {
        HLCStream stream;
        GenerateCommands(cascadedOps[i].get(), hlcOps[i], stream);
        currIndex.push_back(0);
        availableStripe.push_back(nullptr);
        nextStripe.push_back(FindNextStripe(stream, 0));
        cmdsForOps.push_back(std::move(stream));
    }
    // Combine the generated command streams for the individual operations to a single stream.
    // A command on one level can only performed when its input has been produced at the previous level.
    int opIndex = 0;
    while ( true )
    {
        int &ix = currIndex[opIndex];
        if ( opIndex == 0 ||
             nextStripe[opIndex]->ifmAreas[cascadedOps[opIndex]->PrimaryIfmIndex()].End().IsSubShapeOf(
                 availableStripe[opIndex - 1]->ofmArea.End()) )
        {
            auto &stream = cmdsForOps[opIndex];
            assert(ix < int(stream.size()));
            HighLevelCommand *hlc = stream[ix].get();
            cmds.push_back(std::move(cmdsForOps[opIndex][ix]));
            ++ix;
            if ( hlc->IsStripe() )
            {
                availableStripe[opIndex] = nextStripe[opIndex];
                nextStripe[opIndex] = FindNextStripe(stream, ix);
                if ( opIndex < nrOps - 1 &&
                     nextStripe[opIndex + 1]->ifmAreas[cascadedOps[opIndex + 1]->PrimaryIfmIndex()].End().IsSubShapeOf(
                         availableStripe[opIndex]->ofmArea.End()) )
                {
                    // Enough output has been produced to continue at next level
                    ++opIndex;
                }
                if ( nextStripe[opIndex] == nullptr )
                {
                    // Finished
                    assert(opIndex >= nrOps - 1);
                    break;
                }
            }
        }
        else
        {
            // More input is needed from the previous level
            --opIndex;
        }
    }
}

HLCStream HLCStreamGenerator::GenerateCommandStream(const NPUOperation *npuOp, const Schedule *schedule, bool verbose)
{
    HLCStream cmds;
    _schedule = schedule;
    auto &npuOps = npuOp->Operations();
    // Create HLCOperation for every ScheduledOperation
    std::vector<std::shared_ptr<HLCOperation>> hlcOps;
    for ( auto &schedOp : npuOps )
    {
        auto op = schedOp.get();
        hlcOps.push_back(MakeOperation(op, schedule->Cost(op)));
    }

    // Generate the command stream
    int sz = int(npuOps.size());
    for ( int i = 0; i < sz; ++i )
    {
        auto op = npuOps[i].get();
        auto opInfo = schedule->Cost(op);
        assert(opInfo != nullptr);
        auto &hlcOp = hlcOps[i];
        if ( opInfo->cascade == 0 )
        {
            // Single operation, not in cascade
            GenerateCommands(op, hlcOp, cmds);
        }
        else
        {
            // Cascaded operation: generate commands for all operations in the cascade
            auto cascadeInfo = _schedule->Cascade(opInfo->cascade);
            assert(cascadeInfo != nullptr);
            assert(op->Index() == cascadeInfo->start);
            // Note: below code assumes:
            // - all operations in a cascade are in the same NPU op
            // - operations in a cascade are contiguous
            // - operations in the npuOp appear in same order as in the schedule
            int cascadeSize = cascadeInfo->end - cascadeInfo->start + 1;
            assert(i + cascadeSize <= sz);
            vector_span<std::unique_ptr<SchedulerOperation>> cascadedOps(npuOps, i, i + cascadeSize);
            vector_span<std::shared_ptr<HLCOperation>> cascadedHlcOps(hlcOps, i, i + cascadeSize);
            GenerateCommandsForCascade(cascadedOps, cascadedHlcOps, cascadeInfo, cmds);
            i += cascadeSize - 1;
        }
    }
    if ( verbose )
    {
        PrintCommandStream(npuOp, hlcOps, cmds);
    }
    return cmds;
}

void HLCStreamGenerator::PrintCommandStream(const NPUOperation *npuOp, std::vector<std::shared_ptr<HLCOperation>> &hlcOps, HLCStream &cmds)
{
    LOG_PRINT("High level NPU operations:\n");
    int opIndex = 0;
    for ( auto &schedOp : npuOp->Operations() )
    {
        auto op = schedOp.get();
        const auto hlcOp = hlcOps[opIndex].get();
        LOG_PRINT("{} {}\n", opIndex, hlcOp->ToString());
        LOG_PRINT("  IFM: {}, {}\n", op->IFM(0)->tensor->Name(), hlcOp->ifm[0].ToString());
        if ( hlcOp->ifm.size() > 1 )
        {
            LOG_PRINT("  IFM2: {}, {}\n", op->IFM(1)->tensor->Name(), hlcOp->ifm[1].ToString());
        }
        if ( hlcOp->ifm.size() > 2 )
        {
            LOG_PRINT("  IFM3: {}, {}\n", op->IFM(2)->tensor->Name(), hlcOp->ifm[2].ToString());
        }
        LOG_PRINT("  OFM: {}, {}\n", op->OFM()->tensor->Name(), hlcOp->ofm.ToString());
        if ( hlcOp->weights != nullptr )
        {
            LOG_PRINT("  Weights: {}, {}\n", op->Input(TensorUsage::Weights)->tensor->Name(), hlcOp->weights->ToString());
        }
        ++opIndex;
    }
    LOG_PRINT("High level command stream:\n");
    for ( unsigned i = 0; i < cmds.size(); ++i )
    {
        LOG_PRINT("{} {}\n", i, cmds[i]->ToString());
    }
}

}  // namespace regor
