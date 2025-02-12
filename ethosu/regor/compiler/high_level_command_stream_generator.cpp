//
// SPDX-FileCopyrightText: Copyright 2021-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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


static void CalcPaddingAndSkirt(const Kernel *kernel, const Shape &inputShape, const Shape &outputShape,
    const Point2i &inputDilation, HLCPadding &padding, HLCPadding &skirt)
{
    auto dilatedWH = kernel->DilatedWH() * inputDilation;
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

static Box TransformWithStridesAndSkirt(const Box &outputArea, const Shape *strides, const Point2i &inputStep,
    const HLCPadding *skirt, const Shape &ifmShape, OpType opType, const Shape &concatOffsets, const Shape &splitOffset,
    const Shape &splitShape, int dilatedKernelHeight, int upscalingFactor, int &padTop, int &padBottom,
    TransformLimit limit = TransformLimit::None, TransposeType transposeType = TransposeType::None, bool accIfm = false)
{
    Shape outputAreaStart = outputArea.Start().Unpermute(uint32_t(transposeType));
    Shape outputAreaEnd = outputArea.End().Unpermute(uint32_t(transposeType));
    Shape concatOffsetsUntransposed = concatOffsets.Unpermute(uint32_t(transposeType));
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

    if ( accIfm ) return Box(start, end);

    end = Shape::Min(end, Shape::Max(ifmShape, Shape(1, 1, 1, 1)).WithHW(ifmShape.Height() * upscalingFactor, ifmShape.Width() * upscalingFactor));
    padTop = 0;
    padBottom = 0;

    assert(strides != nullptr && skirt != nullptr);
    assert(strides->Size() == 4);

    int strideW = strides->Width();
    Point2i validIfmOffset = splitOffset.IsEmpty() ? Point2i{0, 0} : splitOffset.WH<int>();
    start = start.WithWidth(std::max(start.Width() * strideW - (skirt->left + validIfmOffset.x), validIfmOffset.x));
    strideW *= inputStep.x;
    int validIfmWidth = ifmShape.Width();
    if ( splitShape.Size() > 1 && splitOffset.Size() > 1 )
        validIfmWidth = std::min(validIfmWidth, splitShape.Width() + splitOffset.Width() + skirt->right);
    else if ( splitShape.Size() > 1 ) validIfmWidth = splitShape.Width();
    end = end.WithWidth(std::min(end.Width() * strideW + skirt->right, validIfmWidth));
    int strideH = strides->Height();
    int skirtTopRemainder = skirt->top % upscalingFactor;
    int startHeight = start.Height() * strideH - (skirt->top + validIfmOffset.y) + skirtTopRemainder;
    strideH *= inputStep.y;
    int totalStride = strideH * (outputAreaEnd.Height() - outputAreaStart.Height() - 1);
    padTop = std::max(0, -startHeight) + skirtTopRemainder;
    start = start.WithHeight(std::max(startHeight, validIfmOffset.y));

    int validIfmHeight = ifmShape.Height();
    if ( splitShape.Size() > 2 && splitOffset.Size() > 2 )
        validIfmHeight = std::min(validIfmHeight, splitShape.Height() + splitOffset.Height() + skirt->bottom);
    else if ( splitShape.Size() > 2 ) validIfmHeight = splitShape.Height();

    if ( end.Height() * strideH + skirt->bottom > validIfmHeight * upscalingFactor )
    {
        // padBottom is calculated based the diff between the end position of the weight kernel,
        // after last stride and the ifm height.
        if ( upscalingFactor != 1 && outputAreaEnd.Height() > validIfmHeight * upscalingFactor )
        {
            // Special case for Transpose Convolution with VALID padding.
            padBottom = outputAreaEnd.Height() - validIfmHeight * upscalingFactor;
        }
        else
        {
            int kernelStart = start.Height() - padTop;
            padBottom = std::max(0, kernelStart + totalStride + dilatedKernelHeight - validIfmHeight * upscalingFactor);
        }
    }
    // Adjust for upscaling
    start = start.WithHeight(std::max(start.Height() / upscalingFactor, 0));
    int endHeight = end.Height() * strideH + skirt->bottom + skirt->bottom % upscalingFactor;
    end = end.WithHeight(std::min(std::max(endHeight / upscalingFactor, 1), validIfmHeight));
    if ( limit == TransformLimit::Wrap )
    {
        Shape ifmWrap = Shape::PadAxes(ifmShape, 4, 1);
        Shape one = start.WithOnes();
        start = Shape::Wrap(start, ifmWrap);
        end = Shape::Wrap(end - one, ifmWrap) + one;
        assert((end - start).Elements() > 0);
    }

    return Box(start, end);
}

static std::pair<Box, HLCPadding> TransformWithInputOutputSteps(const Box &inputArea, const Point2i &inputStep,
    const Box &outputArea, const Point2i &outputStep, const Kernel *kernel, const HLCPadding &padding, const Shape &ifmShape)
{
    const auto &stride = kernel->Stride();
    const auto dilatedWH = kernel->DilatedWH();
    HLCPadding newPadding;
    newPadding.top = std::max(0, DivRoundUp(padding.top, inputStep.y));
    newPadding.left = std::max(0, DivRoundUp(padding.left, inputStep.x));
    Point2i startAdjustForPadFraction;
    if ( padding.left > 0 && inputArea.Start().Width() == 0 )
        startAdjustForPadFraction.x = std::max(0, DivRoundUp(padding.left, inputStep.x) * inputStep.x - padding.left);
    if ( padding.top > 0 && inputArea.Start().Height() == 0 )
        startAdjustForPadFraction.y = std::max(0, DivRoundUp(padding.top, inputStep.y) * inputStep.y - padding.top);
    Point2i neededInput;
    neededInput.x =
        (DivRoundUp(outputArea.End().Width() - outputArea.Start().Width(), outputStep.x) - 1) * stride.x + dilatedWH.x;
    neededInput.y =
        (DivRoundUp(outputArea.End().Height() - outputArea.Start().Height(), outputStep.y) - 1) * stride.y + dilatedWH.y;
    Shape newStart =
        inputArea.Start()
            .WithWidth(inputArea.Start().Width() + startAdjustForPadFraction.x)
            .WithHeight(inputArea.Start().Height() + startAdjustForPadFraction.y);
    Shape newEnd =
        inputArea.End()
            .WithWidth(std::min(inputArea.End().Width() + startAdjustForPadFraction.x, ifmShape.Width()))
            .WithHeight(std::min(inputArea.End().Height() + startAdjustForPadFraction.y, ifmShape.Height()));
    newPadding.bottom = std::max(0,
        neededInput.y -
            (DivRoundUp((ifmShape.Height() - inputArea.Start().Height()) - startAdjustForPadFraction.y, inputStep.y) +
                newPadding.top));
    newPadding.right = std::max(0,
        neededInput.x -
            (DivRoundUp(ifmShape.Width() - inputArea.Start().Width() - startAdjustForPadFraction.x, inputStep.x) +
                newPadding.left));
    return std::make_pair<Box, HLCPadding>(Box(newStart, newEnd), std::move(newPadding));
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

static void MakeFeatureMap(TensorUsage usage, const SchedulerConnection *schedConn, HLCFeatureMap &fm)
{
    auto schedTens = schedConn->tensor.get();
    fm.shape = schedConn->shape;
    fm.slice = schedConn->slice;
    fm.dataType = schedTens->dataType;
    fm.memArea = schedTens->memArea;
    fm.format = schedTens->format;
    fm.usage = usage;
    fm.address = schedTens->AllocatedAddress();
    fm.quantization = schedConn->quantization;
    fm.bufferView = schedTens->bufferView;
    fm.strides = GetStrides(fm);
    fm.stepXY = schedConn->stepXY;
    fm.transpose = schedConn->transpose;
    fm.reverse = schedConn->reverse;
    fm.resamplingMode = schedConn->resamplingMode;
    fm.rounding = HLCRoundMode(schedConn->rounding);
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
    weights->address = bufTensor->AllocatedAddress();
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

    for ( const auto &input : schedOp->inputs.pairs() )
    {
        std::vector<HLCFeatureMap>::iterator at;
        if ( IsIFM(input.first) )
        {
            at = hlcSubOp.ifm.emplace(std::upper_bound(hlcSubOp.ifm.begin(), hlcSubOp.ifm.end(), input.first,
                [](TensorUsage usage, const HLCFeatureMap &fm) { return usage < fm.usage; }));
            MakeFeatureMap(input.first, &input.second, *at);
        }
    }
    MakeFeatureMap(TensorUsage::OFM, schedOp->OFM(), hlcSubOp.ofm);
    hlcSubOp._srcId = schedOp->Uid();

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
        param.address = lutTensor->AllocatedAddress();
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
    op->_srcId = schedOp->Uid();

    for ( const auto &input : schedOp->inputs.pairs() )
    {
        std::vector<HLCFeatureMap>::iterator at;
        if ( IsIFM(input.first) )
        {
            at = op->ifm.emplace(std::upper_bound(op->ifm.begin(), op->ifm.end(), input.first,
                [](TensorUsage usage, const HLCFeatureMap &fm) { return usage < fm.usage; }));
            MakeFeatureMap(input.first, &input.second, *at);
        }
    }
    MakeFeatureMap(TensorUsage::OFM, schedOp->OFM(), op->ofm);
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
        param.address = lutTensor->AllocatedAddress();
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
            // Convert attr->axis to AxisMask
            assert(lutConn == nullptr);
            const auto *attr = schedOp->Attribute<axis_attr_t>();
            int ifmRank = schedOp->Input(TensorUsage::IFM)->SliceShape().Size();
            int axis3D = 3 - ifmRank + attr->axis;
            op->parameters.argmax.axis = axis3D == 0 ? AxisMask::AxisY : AxisMask::AxisX;
        }
        break;
        case OpType::Tile:
        {
            auto *ifmConn = schedOp->Input(TensorUsage::IFM);
            auto *params = schedOp->Input(TensorUsage::Params);
            assert(params);
            assert(params->tensor->dataType == DataType::Int32);
            auto view = params->tensor->srcTensor->View();
            Shape multiples(view.Buffer()->Data<int32_t>(), view.ViewShape().Elements());
            multiples = Shape::PadAxes(multiples, ifmConn->shape.Size(), 1);
            unsigned axisMask = multiples.GreaterMask(multiples.WithOnes());
            assert((axisMask == 0 || IsPowerOfTwo(axisMask)) && "TILE operation should only have one tiled axis");
            // Find tiled axis
            int axis = ifmConn->shape.Size() - 1;
            while ( (axisMask >>= 1) > 0 )
            {
                axis -= 1;
            }
            op->parameters.tile.axis = axis;
            op->parameters.tile.multiplier = multiples[axis];
        }
        break;
        default:
            break;
    }

    if ( schedOp->HasAttribute<sign_attr_t>() )
    {
        const auto *attr = schedOp->Attribute<sign_attr_t>();
        if ( attr->input_unsigned )
        {
            op->ifm[0].dataType = op->ifm[0].dataType & ~unsigned(DataType::Signed);
            if ( op->ifm.size() > 1 )
            {
                op->ifm[1].dataType = op->ifm[1].dataType & ~unsigned(DataType::Signed);
            }
        }
        if ( attr->output_unsigned )
        {
            op->ofm.dataType = op->ofm.dataType & ~unsigned(DataType::Signed);
        }
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
static std::unique_ptr<HLCDMA> GenerateWeightDMA(NpuWeightTensor *weightTens, const SchedulerBufferTensor &bufConn, int depth, int depthIndex)
{
    auto dma = std::make_unique<HLCDMA>();
    dma->srcMemArea = weightTens->memArea;
    dma->srcAddress = weightTens->AllocatedAddress();
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
    dma->destAddress = bufConn.tensor->AllocatedAddress();
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
    CalcPaddingAndSkirt(kernel, maxIfmShape, ofmShape, ofmConn->stepXY, padding, skirt);

    int upscaling = 1;
    if ( ifm0Conn->resamplingMode != ArchResampling::None )
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
        ofmStart = Shape::PadAxes(ofmConn->slice.offset, 4, 0);
        ofmEnd = Shape::PadAxes(ofmConn->slice.offset + ofmConn->slice.shape, 4, 1);
    }
    assert(ofmStart.Size() >= 4);
    assert(ofmEnd.Size() >= 4);

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
                // Depth-slices are computed relative to the offset and sliceShape for a striped or sliced OFM
                int startChannel = ofmStart.Depth() + depthSlices[depthIndex];
                int endChannel = std::min(ofmEnd.Depth(), ofmStart.Depth() + depthSlices[depthIndex + 1]);

                // Construct the output area for the current stripe
                auto outputAreaStart = Shape(ofmStart.Batch(), startHeight, startWidth, startChannel);
                auto outputAreaEnd = Shape(ofmEnd.Batch(), endHeight, endWidth, endChannel);
                auto outputArea = Box(outputAreaStart, outputAreaEnd);
                auto hlcStripe = std::make_unique<HLCStripe>(hlcOp);
                hlcStripe->padding = padding;
                hlcStripe->ofmArea = outputArea;
                hlcStripe->opGroup = opGroup;
                for ( const auto &fm : hlcOp->ifm )
                {
                    if ( !IsIFM(fm.usage) ) continue;
                    auto ifmConn = op->Input(fm.usage);
                    bool accIfm = op->AccumulatorMode().source == AccumulatorSource::Ifm2 && fm.usage == TensorUsage::IFM1;
                    // Calculate input area based on the output area
                    auto inputArea = TransformWithStridesAndSkirt(outputArea, &strides, ifmConn->stepXY, &skirt, ifmConn->shape,
                        opType, ofmConn->slice.offset, ifmConn->slice.offset, ifmConn->slice.shape, dilatedKernelHeight,
                        upscaling, hlcStripe->padding.top, hlcStripe->padding.bottom, ifmLimit, ofmConn->transpose, accIfm);
                    if ( !accIfm && (ofmConn->stepXY != Point2i{1, 1} || ifmConn->stepXY != Point2i{1, 1}) )
                    {
                        std::tie(inputArea, hlcStripe->padding) = TransformWithInputOutputSteps(inputArea,
                            ifmConn->stepXY, outputArea, ofmConn->stepXY, kernel, hlcStripe->padding, ifmConn->shape);
                    }
                    inputArea = Box(inputArea.Start(), Shape::Max(inputArea.End(), inputArea.Start() + inputArea.Start().WithOnes()));
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
        LOG_PRINT("  IFM: {}, {}\n", op->Input(hlcOp->ifm[0].usage)->tensor->Name(), hlcOp->ifm[0].ToString());
        if ( hlcOp->ifm.size() > 1 )
        {
            LOG_PRINT("  IFM2: {}, {}\n", op->Input(hlcOp->ifm[1].usage)->tensor->Name(), hlcOp->ifm[1].ToString());
        }
        if ( hlcOp->ifm.size() > 2 )
        {
            LOG_PRINT("  IFM3: {}, {}\n", op->Input(hlcOp->ifm[2].usage)->tensor->Name(), hlcOp->ifm[2].ToString());
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
