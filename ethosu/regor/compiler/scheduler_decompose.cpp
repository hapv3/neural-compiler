//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "scheduler_decompose.hpp"

#include "common/logging.hpp"

#include "operation_util.hpp"

#include <numeric>
#include <optional>

namespace regor
{

static constexpr int MAX_DIM = 65536;

bool NeedsDecompose(Architecture *arch, const SchedulerOperation *schedOp)
{
    return CanDecompose(arch, schedOp) && !CanRunOnHardware(arch, schedOp);
}

static std::unique_ptr<SchedulerOperation> MakeSubOperation(const SchedulerOperation *schedOp, const Kernel *newKernel = nullptr)
{
    assert(schedOp->SubOps().empty());
    assert(schedOp->Parent() == nullptr);
    auto subOp = std::make_unique<SchedulerOperation>(schedOp->Type());
    subOp->SetKernel(newKernel ? newKernel : schedOp->Kernel());
    subOp->SetHasScaling(schedOp->HasScaling());
    subOp->_srcKey = schedOp->_srcKey;
    subOp->SetPrimaryIfmIndex(schedOp->PrimaryIfmIndex());
    subOp->SetRounding(schedOp->Rounding());
    subOp->SetAttributeRef(schedOp->_attr);
    subOp->SetAccumulatorMode(schedOp->AccumulatorMode());
    for ( const auto *list : {&schedOp->inputs, &schedOp->outputs} )
    {
        for ( const auto &item : list->pairs() )
        {
            auto usage = item.first;
            const auto &connection = item.second;
            if ( IsOFM(usage) )
            {
                connection.tensor->producers.push_back(subOp.get());
                *subOp->AddOutput(usage) = connection;
            }
            else
            {
                connection.tensor->consumers.push_back(subOp.get());
                *subOp->AddInput(usage) = connection;
            }
        }
    }
    return subOp;
}

static auto GetArchAccumulatorSource(const AccumulatorControl &ac)
{
    switch ( ac.source )
    {
        case AccumulatorSource::Reset:
            return ArchAccumulatorSource::Reset;
        case AccumulatorSource::Acc:
            return ArchAccumulatorSource::Acc;
        case AccumulatorSource::Ifm2:
            return ArchAccumulatorSource::Ifm2;
        default:
            assert(false);
            return ArchAccumulatorSource::Reset;
    }
}

bool CanRunOnHardware(Architecture *arch, const SchedulerOperation *schedOp)
{
    regor::ArchitectureOpGroupQuery qOpGroup{};
    if ( DecomposeAsElementwise(schedOp->Type()) || schedOp->Type() == OpType::MemoryCopy )
    {
        auto &ofmShape = schedOp->OFM()->SliceShape();
        if ( ofmShape.Size() > 3 && ofmShape.Elements() > ofmShape.Width() * ofmShape.Height() * ofmShape.Depth() )
            return false;
    }
    if ( schedOp->Type() == OpType::MatMul )
    {
        auto &ofmShape = schedOp->OFM()->SliceShape();
        if ( ofmShape.Size() > 2 && ofmShape.Elements() > ofmShape.Width() * ofmShape.Depth() ) return false;
    }
    if ( IsConvolution(schedOp->Type()) )
    {
        auto &ofmShape = schedOp->OFM()->SliceShape();
        if ( ofmShape.Size() > 3 && ofmShape.Batch() > 1 ) return false;
    }
    if ( schedOp->Type() == OpType::Transpose )
    {
        auto &ifmShape = schedOp->IFM(0)->SliceShape();
        if ( ifmShape.Size() > 3 && ifmShape.Elements() > ifmShape.Width() * ifmShape.Height() * ifmShape.Depth() )
            return false;
        auto &ofmShape = schedOp->OFM()->SliceShape();
        if ( ofmShape.Size() > 3 && ofmShape.Elements() > ofmShape.Width() * ofmShape.Height() * ofmShape.Depth() )
            return false;
    }
    auto *ifm = schedOp->TryIFM(0);
    auto *ifm2 = schedOp->TryIFM(1);
    auto *ofm = schedOp->TryOFM();

    if ( !ifm || !ofm ) return false;
    qOpGroup.type = schedOp->Type();
    qOpGroup.kernel = schedOp->Kernel();
    qOpGroup.ifm[0].key = ifm->tensor->uid;
    qOpGroup.ifm[0].type = ifm->tensor->dataType;
    qOpGroup.ifm[0].shape = ifm->SliceShape();
    if ( ifm2 )
    {
        qOpGroup.ifm[1].key = ifm2->tensor->uid;
        qOpGroup.ifm[1].type = ifm2->tensor->dataType;
        qOpGroup.ifm[1].shape = ifm2->SliceShape();
    }
    qOpGroup.ofm.key = ofm->tensor->uid;
    qOpGroup.ofm.type = ofm->tensor->dataType;
    qOpGroup.ofm.shape = ofm->SliceShape();
    qOpGroup.ofm.transpose = ofm->transpose;
    if ( arch->CreateOpGroup(qOpGroup) == nullptr ) return false;
    regor::ArchitectureConfigQuery qConfig;
    qConfig.ofmShape = Shape::PadAxes(ofm->SliceShape(), 3, 1);
    qConfig.ifmShape[0] = ifm->SliceShape();
    if ( ifm2 )
    {
        qConfig.ifmShape[1] = ifm2->SliceShape();
    }
    qConfig.ifmBits = DataTypeSizeBits(ifm->tensor->dataType);
    qConfig.kernel = schedOp->Kernel();
    qConfig.lutBytes = schedOp->TryInput(TensorUsage::LUT) ? 2048 : 0;
    qConfig.scaled = schedOp->HasScaling();
    qConfig.ifmResampling = ifm->resamplingMode;
    qConfig.ofmShape = qConfig.ofmShape.Unpermute(uint32_t(ofm->transpose));
    qConfig.transpose = ofm->transpose;
    qConfig.ofmFormat = ofm->tensor->format;
    const auto &accMode = schedOp->AccumulatorMode();
    qConfig.accSource = GetArchAccumulatorSource(accMode);
    qConfig.accOutputEnabled = accMode.outputEnabled;
    return arch->GetOpConfig(schedOp->Type(), qConfig) != nullptr;
}

bool CanDecompose(Architecture *, const SchedulerOperation *schedOp)
{
    if ( schedOp->Type() == OpType::Conv2D ) return true;
    if ( schedOp->Type() == OpType::DepthwiseConv2DBias ) return true;
    if ( schedOp->Type() == OpType::TransposeConv2D ) return true;
    if ( DecomposeAsElementwise(schedOp->Type()) || schedOp->Type() == OpType::MemoryCopy ) return true;
    if ( schedOp->Type() == OpType::MatMul ) return true;
    if ( schedOp->Type() == OpType::ReduceSum ) return true;
    if ( schedOp->Type() == OpType::ReduceMin ) return true;
    if ( schedOp->Type() == OpType::ReduceMax ) return true;
    if ( schedOp->Type() == OpType::ReduceAny ) return true;
    if ( schedOp->Type() == OpType::ReduceAll ) return true;
    if ( schedOp->Type() == OpType::ArgMax ) return true;
    if ( schedOp->Type() == OpType::Reverse ) return true;
    if ( schedOp->Type() == OpType::Transpose ) return true;
    return false;
}

typedef std::function<std::vector<std::unique_ptr<SchedulerOperation>>(Architecture *, std::unique_ptr<SchedulerOperation>)> DecomposeFunc;

static std::vector<std::unique_ptr<SchedulerOperation>> DecomposeBlocksElementwise(
    Architecture *arch, std::unique_ptr<SchedulerOperation> op, Shape &blockShape, const DecomposeFunc &doDecompose)
{
    std::vector<std::unique_ptr<SchedulerOperation>> result;
    const auto BH = blockShape.Height();
    const auto BW = blockShape.Width();
    const auto BC = blockShape.Depth();
    auto *ofmConn = op->OFM();
    auto *ifmConn = op->IFM(0);
    auto *ifm2Conn = op->TryIFM(1);
    auto &ofmShape = ofmConn->SliceShape();
    const auto N = Shape::DivRoundUp(ofmShape, blockShape);  // Block count per dimension
    auto NewIfmSlice = [&](SchedulerConnection *ifmC, int x, int y, int c)
    {
        auto newIfmSlice = ifmC->slice;

        newIfmSlice.offset += Shape(y * BH, x * BW, c * BC);
        auto &newIfmShape = newIfmSlice.shape;
        newIfmSlice.shape = Shape::Max(
            Shape::Min(newIfmShape - Shape(y * BH, x * BW, c * BC), Shape(BH, BW, BC)), newIfmShape.WithOnes());
        return newIfmSlice;
    };
    for ( auto by = 0; by < N.Height(); by++ )
    {
        for ( auto bx = 0; bx < N.Width(); bx++ )
        {
            for ( auto bc = 0; bc < N.Depth(); bc++ )
            {
                auto newIfmSlice = NewIfmSlice(ifmConn, bx, by, bc);
                TensorSlice newIfm2Slice;
                if ( ifm2Conn )
                {
                    newIfm2Slice = NewIfmSlice(ifm2Conn, bx, by, bc);
                }
                auto newOfmSlice = ofmConn->slice;
                newOfmSlice.offset += Shape(by * BH, bx * BW, bc * BC);
                auto &newOfmShape = newOfmSlice.shape;
                newOfmSlice.shape = Shape::Max(
                    Shape::Min(newOfmShape - Shape(by * BH, bx * BW, bc * BC), Shape(BH, BW, BC)), newOfmShape.WithOnes());
                std::unique_ptr<SchedulerOperation> subOp = MakeSubOperation(op.get());
                auto *subIfmConn = subOp->IFM(0);
                subIfmConn->slice = std::move(newIfmSlice);
                if ( ifm2Conn )
                {
                    auto *subIfm2Conn = subOp->IFM(1);
                    subIfm2Conn->slice = std::move(newIfm2Slice);
                }
                auto *subOfmConn = subOp->OFM();
                subOfmConn->slice = std::move(newOfmSlice);
                auto subOps = doDecompose(arch, std::move(subOp));
                result.insert(result.end(), std::make_move_iterator(subOps.begin()), std::make_move_iterator(subOps.end()));
            }
        }
    }
    return result;
}

// Decompose to sub-operations in slices along the specified axis.
static std::vector<std::unique_ptr<SchedulerOperation>> DecomposeLargeAxis(int axis, int sliceSize, Architecture *arch,
    std::unique_ptr<SchedulerOperation> op, const DecomposeFunc &doDecompose)
{
    std::vector<std::unique_ptr<SchedulerOperation>> result;
    auto *ofmConn = op->Output(TensorUsage::OFM);
    auto *ifmConn = op->Input(TensorUsage::IFM0);
    auto *ifm2Conn = op->TryInput(TensorUsage::IFM1);

    static auto SliceFunc = [](SchedulerConnection *conn, int ax, int maxSlice, int offset) -> TensorSlice
    {
        Shape newOffset = conn->slice.offset;
        Shape newShape = conn->SliceShape();
        // maxIndex is the largest index of the undecomposed slice
        int maxIndex = newOffset[ax] + newShape[ax];
        // Don't offset broadcasted axis
        if ( newShape[ax] != 1 )
        {
            // clamp slices if they exceed maxIndex
            newOffset[ax] = std::min(newOffset[ax] + offset, maxIndex - 1);
            newShape[ax] = std::min(maxSlice, maxIndex - newOffset[ax]);
        }
        assert(newOffset[ax] + newShape[ax] <= maxIndex);
        return {newOffset, newShape};
    };

    auto axisSize = ofmConn->SliceShape()[axis];
    for ( int i = 0; i < axisSize; i += sliceSize )
    {
        std::unique_ptr<SchedulerOperation> subOp = MakeSubOperation(op.get());
        subOp->Input(TensorUsage::IFM0)->slice = SliceFunc(ifmConn, axis, sliceSize, i);
        subOp->Output(TensorUsage::OFM)->slice = SliceFunc(ofmConn, axis, sliceSize, i);
        if ( ifm2Conn )
        {
            subOp->Input(TensorUsage::IFM1)->slice = SliceFunc(ifm2Conn, axis, sliceSize, i);
        }
        auto subOps = doDecompose(arch, std::move(subOp));
        result.insert(result.end(), std::make_move_iterator(subOps.begin()), std::make_move_iterator(subOps.end()));
    }
    return result;
}

// Decompose to sub-operations with size 1 along the leading <dimension> axes.
// Used for the batch dimension, and for the leading N-3 dimensions for elementwise operations.
static std::vector<std::unique_ptr<SchedulerOperation>> DecomposeLeadingDimensions(
    int dimensions, Architecture *arch, std::unique_ptr<SchedulerOperation> op, const DecomposeFunc &doDecompose)
{
    int axis = dimensions - 1;
    // Use a callback-mechanism to recurse over all leading dimensions
    DecomposeFunc cb = [axis, &doDecompose](Architecture *_arch, std::unique_ptr<SchedulerOperation> _op)
    {
        if ( axis > 0 )
        {
            return DecomposeLeadingDimensions(axis, _arch, std::move(_op), doDecompose);
        }
        return doDecompose(_arch, std::move(_op));
    };

    return DecomposeLargeAxis(axis, 1, arch, std::move(op), cb);
}

// Handle dilation by decomposing to suboperations with input stride = dilation and dilation 1
static std::vector<std::unique_ptr<SchedulerOperation>>
HandleDilation(Architecture *arch, std::unique_ptr<SchedulerOperation> op, const DecomposeFunc &doDecompose)
{
    std::vector<std::unique_ptr<SchedulerOperation>> result;
    auto *ofmConn = op->Output(TensorUsage::OFM);
    auto *ifmConn = op->Input(TensorUsage::IFM);
    auto *kernel = op->Kernel();
    auto &dilation = kernel->Dilation();
    auto &stride = kernel->Stride();
    auto GY = std::gcd(dilation.y, stride.y);
    auto GX = std::gcd(dilation.x, stride.x);
    auto DY = dilation.y / GY;
    auto DX = dilation.x / GX;
    for ( auto dy = 0; dy < DY; ++dy )
    {
        for ( auto dx = 0; dx < DX; ++dx )
        {
            auto newIfmSlice = ifmConn->slice;
            auto newOfmSlice = ofmConn->slice;
            auto ifmStrides = ifmConn->stepXY;
            auto ofmStrides = ofmConn->stepXY;
            newIfmSlice.offset[1] += dy * GY;
            newIfmSlice.offset[2] += dx * GX;
            ifmStrides.y *= DY * GY;
            ifmStrides.x *= DX * GX;
            newOfmSlice.offset[1] += dy;
            newOfmSlice.offset[2] += dx;
            newOfmSlice.shape[1] -= dy;
            newOfmSlice.shape[2] -= dx;
            if ( newOfmSlice.shape.Elements() > 0 )
            {
                ofmStrides.y *= DY;
                ofmStrides.x *= DX;
                auto newKernel = kernel->WithDilation({1, 1}).WithStride(stride / Point2i{GX, GY});
                std::unique_ptr<SchedulerOperation> subOp = MakeSubOperation(op.get(), &newKernel);
                auto *subIfmConn = subOp->Input(TensorUsage::IFM);
                subIfmConn->slice = std::move(newIfmSlice);
                subIfmConn->stepXY = ifmStrides;
                auto *subOfmConn = subOp->Output(TensorUsage::OFM);
                subOfmConn->slice = std::move(newOfmSlice);
                subOfmConn->stepXY = ofmStrides;
                auto subOps = doDecompose(arch, std::move(subOp));
                result.insert(result.end(), std::make_move_iterator(subOps.begin()), std::make_move_iterator(subOps.end()));
            }
        }
    }
    return result;
}

static void UpdatePaddingAndIfmOffset(SchedulerOperation *op)
{
    auto *kernel = op->Kernel();
    auto &padding = kernel->Padding();
    auto &ifmSlice = op->Input(TensorUsage::IFM)->slice;
    // Negative ifm offsets indicate new padding values with ifm offset 0
    auto topPad = std::max(0, -ifmSlice.offset.Height());
    auto leftPad = std::max(0, -ifmSlice.offset.Width());
    ifmSlice.offset[1] = std::max(0, ifmSlice.offset.Height());
    ifmSlice.offset[2] = std::max(0, ifmSlice.offset.Width());
    auto newPadding = Margin(topPad, leftPad, padding.Bottom(), padding.Right());
    auto newKernel = kernel->WithPadding(newPadding);
    op->SetKernel(&newKernel);
}

static void InitializeSlice(TensorSlice &slice, const Shape &offset, const Shape &shape)
{
    if ( !slice.offset.IsValid() )
    {
        slice.offset = offset;
    }
    if ( !slice.shape.IsValid() )
    {
        slice.shape = shape;
    }
}

std::vector<std::unique_ptr<SchedulerOperation>> DecomposeConv2D(Architecture *arch, std::unique_ptr<SchedulerOperation> op)
{
    std::vector<std::unique_ptr<SchedulerOperation>> result;
    auto *ofmConn = op->Output(TensorUsage::OFM);
    auto *ifmConn = op->Input(TensorUsage::IFM);
    const auto &ofmShape = ofmConn->SliceShape();
    const auto &ifmShape = ifmConn->SliceShape();
    auto &ofmSlice = ofmConn->slice;
    auto &ifmSlice = ifmConn->slice;
    auto *kernel = op->Kernel();
    auto &padding = kernel->Padding();
    InitializeSlice(ofmSlice, ofmShape.WithZeros(), ofmShape);
    InitializeSlice(ifmSlice, ifmShape.WithZeros().WithHW(-padding.Top(), -padding.Left()), ifmShape);

    if ( ofmShape.Batch() > 1 )
    {
        return DecomposeLeadingDimensions(1, arch, std::move(op), DecomposeConv2D);
    }
    if ( CanRunOnHardware(arch, op.get()) )
    {
        UpdatePaddingAndIfmOffset(op.get());
        result.emplace_back(std::move(op));
        return result;
    }
    auto &dilation = kernel->Dilation();
    if ( dilation.x > 1 || dilation.y > 1 )
    {
        return HandleDilation(arch, std::move(op), DecomposeConv2D);
    }
    // TODO: MLBEDSW-8783 Decompose convolutions with large stride
    // If we get here, decomposition has failed, the resulting operations will be executed on CPU
    result.emplace_back(std::move(op));
    return result;
}

std::vector<std::unique_ptr<SchedulerOperation>> DecomposeDepthwiseConv2D(Architecture *arch, std::unique_ptr<SchedulerOperation> op)
{
    std::vector<std::unique_ptr<SchedulerOperation>> result;
    auto *ofmConn = op->Output(TensorUsage::OFM);
    auto *ifmConn = op->Input(TensorUsage::IFM);
    auto *weightsConn = op->Input(TensorUsage::Weights);
    const auto &ofmShape = ofmConn->SliceShape();
    const auto &ifmShape = ifmConn->SliceShape();
    const auto &weightsShape = weightsConn->shape;
    auto &ofmSlice = ofmConn->slice;
    auto &ifmSlice = ifmConn->slice;
    auto *kernel = op->Kernel();
    auto &padding = kernel->Padding();
    InitializeSlice(ofmSlice, ofmShape.WithZeros(), ofmShape);
    InitializeSlice(ifmSlice, ifmShape.WithZeros().WithHW(-padding.Top(), -padding.Left()), ifmShape);

    if ( ofmShape.Batch() > 1 )
    {
        return DecomposeLeadingDimensions(1, arch, std::move(op), DecomposeDepthwiseConv2D);
    }
    if ( weightsShape.Depth() > 1 )
    {
        // TODO: MLBEDSW-8789 Handle depthwise convolution with depth multiplier > 1
        // If we get here, decomposition has failed, the resulting operations will be executed on CPU
        result.emplace_back(std::move(op));
        return result;
    }
    if ( CanRunOnHardware(arch, op.get()) )
    {
        UpdatePaddingAndIfmOffset(op.get());
        result.emplace_back(std::move(op));
        return result;
    }
    auto &dilation = kernel->Dilation();
    if ( dilation.x > 1 || dilation.y > 1 )
    {
        return HandleDilation(arch, std::move(op), DecomposeDepthwiseConv2D);
    }
    // TODO: MLBEDSW-8783 Decompose convolutions with large stride
    // If we get here, decomposition has failed, the resulting operations will be executed on CPU
    result.emplace_back(std::move(op));
    return result;
}

// Reverse elements along H and W axes
template<typename TYPE>
static std::shared_ptr<SchedulerTensor> ReverseHW2(SchedulerTensor *tensor)
{
    const auto &inBufferView = tensor->bufferView;
    const auto inBufferValues = inBufferView.Values<TYPE>();

    // Create output buffer that will contain reversed weights
    const auto size = inBufferView.Elements();
    auto outBuffer = std::make_shared<Buffer>(std::make_unique<TYPE[]>(size), size);
    BufferView outBufferView(std::move(outBuffer), tensor->bufferView);
    auto outBufferValues = outBufferView.WritableValues<TYPE>();

    // Reverse height and width into the output buffer
    int batch = outBufferView.ViewShape().Batch();
    int height = outBufferView.ViewShape().Height();
    int width = outBufferView.ViewShape().Width();
    int depth = outBufferView.ViewShape().Depth();
    for ( int n = 0; n < batch; n++ )
    {
        for ( int h = 0; h < height; h++ )
        {
            for ( int w = 0; w < width; w++ )
            {
                for ( int c = 0; c < depth; c++ )
                {
                    outBufferValues[{n, height - h - 1, width - w - 1, c}] = inBufferValues[{n, h, w, c}];
                }
            }
        }
    }

    // Clone tensor with new buffer with new unique ID because now the tensor is different
    auto clonedTensor = std::make_shared<SchedulerTensor>(*tensor);
    clonedTensor->bufferView = std::move(outBufferView);
    clonedTensor->equivalenceId = GenerateUniqueId();

    return clonedTensor;
}

// Reverse elements along H and W axes
static std::shared_ptr<SchedulerTensor> ReverseHW(SchedulerTensor *tensor)
{
    assert(tensor->IsConstant());
    assert(tensor->producers.size() == 0);
    assert(tensor->consumers.size() == 1);

    switch ( tensor->dataType )
    {
        case DataType::Int8:
            return ReverseHW2<int8_t>(tensor);
        case DataType::UInt8:
            return ReverseHW2<uint8_t>(tensor);
        default:
            assert(false && "Unknown data type");
            return nullptr;
    }
}

// Decompose Transpose Conv2D into Conv2D
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeTransposeConv2D(Architecture *arch, std::unique_ptr<SchedulerOperation> op)
{
    UNUSED(arch);
    std::vector<std::unique_ptr<SchedulerOperation>> result;

    auto ifmConn = op->Input(TensorUsage::IFM);
    auto weightsConn = op->Input(TensorUsage::Weights);
    auto ofmConn = op->Output(TensorUsage::OFM);

    auto kernel = op->Kernel();
    const int32_t kernel_h = kernel->Size().y;
    assert(kernel_h > 0);
    const int32_t kernel_w = kernel->Size().x;
    assert(kernel_w > 0);
    const int32_t stride_h = kernel->Stride().y;
    assert(stride_h > 0);
    const int32_t stride_w = kernel->Stride().x;
    assert(stride_w > 0);

    int actualIfmHeight = ifmConn->shape.Height() * stride_h;
    int actualIfmWidth = ifmConn->shape.Width() * stride_w;
    int actualOfmHeight = ofmConn->shape.Height();
    int actualOfmWidth = ofmConn->shape.Width();
    int heightPadding = NeededTotalPadding(actualIfmHeight, actualOfmHeight, 1, kernel_h);
    int widthPadding = NeededTotalPadding(actualIfmWidth, actualOfmWidth, 1, kernel_w);

    if ( (stride_h == 1 && stride_w == 1) || (stride_h == 2 && stride_w == 2) ||
         (stride_h == 1 && stride_w == 2 && ifmConn->shape.Height() == 1 && kernel_h == 1) )
    {
        // IFM pad for 1x1 stride
        int bottom = heightPadding / 2;
        int top = heightPadding - bottom;
        int right = widthPadding / 2;
        int left = widthPadding - right;

        // Reverse H and W weights
        weightsConn->tensor = ReverseHW(weightsConn->tensor.get());

        if ( (stride_h == 2 || stride_w == 2) )
        {
            ifmConn->resamplingMode = ArchResampling::Zeros;

            // IFM pad for 2x2 stride
            if ( kernel->Padding().IsZero() )
            {
                // TFLite VALID padding
                bottom = std::max(kernel_h - 2, 0);
                top = kernel_h - 1;
                right = std::max(kernel_w - 2, 0);
                left = kernel_w - 1;
            }
            else
            {
                // TFLite SAME padding
                bottom = std::max(((heightPadding + 1) / stride_h) - 1, 0);
                top = std::max(kernel_h - 1 - bottom, 0);
                right = std::max(((widthPadding + 1) / stride_w) - 1, 0);
                left = std::max(kernel_w - 1 - right, 0);
            }
        }

        Kernel newKernel = kernel->WithStride({1, 1}).WithPadding({top, left, bottom, right});

        // Switch to Conv2D
        op->_type = OpType::Conv2D;
        op->SetKernel(&newKernel);
        result.emplace_back(std::move(op));
    }
    else
    {
        result.emplace_back(std::move(op));
    }

    return result;
}

std::vector<std::unique_ptr<SchedulerOperation>> DecomposeElementwise(Architecture *arch, std::unique_ptr<SchedulerOperation> op)
{
    std::vector<std::unique_ptr<SchedulerOperation>> result;
    auto ofmConn = op->Output(TensorUsage::OFM);
    auto &ofmShape = ofmConn->SliceShape();
    auto &ofmSlice = ofmConn->slice;
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto &ifmShape = ifmConn->SliceShape();
    auto &ifmSlice = ifmConn->slice;

    InitializeSlice(ofmSlice, ofmShape.WithZeros(), ofmShape);
    InitializeSlice(ifmSlice, ifmShape.WithZeros(), ifmShape);
    if ( auto ifm2Conn = op->TryInput(TensorUsage::IFM1) )
    {
        auto &ifm2Shape = ifm2Conn->shape;
        auto &ifm2Slice = ifm2Conn->slice;

        InitializeSlice(ifm2Slice, ifm2Shape.WithZeros(), ifm2Shape);
    }
    auto ofmRank = ofmShape.Size();
    if ( ofmRank > 3 && ofmShape.Elements() > ofmShape.Width() * ofmShape.Height() * ofmShape.Depth() )
    {
        return DecomposeLeadingDimensions(ofmRank - 3, arch, std::move(op), DecomposeElementwise);
    }
    if ( auto maxShape = Shape::Min(Shape(nullptr, ofmShape.Size(), MAX_DIM), ofmShape); maxShape != ofmShape )
    {
        return DecomposeBlocksElementwise(arch, std::move(op), maxShape, DecomposeElementwise);
    }
    result.emplace_back(std::move(op));
    return result;
}

std::vector<std::unique_ptr<SchedulerOperation>> DecomposeMatmul(Architecture *arch, std::unique_ptr<SchedulerOperation> op)
{
    std::vector<std::unique_ptr<SchedulerOperation>> result;
    auto ofmConn = op->Output(TensorUsage::OFM);
    auto &ofmShape = ofmConn->SliceShape();
    auto &ofmSlice = ofmConn->slice;
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto &ifmShape = ifmConn->SliceShape();
    auto &ifmSlice = ifmConn->slice;

    InitializeSlice(ofmSlice, ofmShape.WithZeros(), ofmShape);
    InitializeSlice(ifmSlice, ifmShape.WithZeros(), ifmShape);

    // TODO MLBEDSW-9535: large tensor decomposition
    if ( auto ifm2Conn = op->TryInput(TensorUsage::IFM1) )
    {
        auto &ifm2Shape = ifm2Conn->shape;
        auto &ifm2Slice = ifm2Conn->slice;

        InitializeSlice(ifm2Slice, ifm2Shape.WithZeros(), ifm2Shape);
    }

    auto ofmRank = ofmShape.Size();
    if ( ofmRank > 2 && (ofmShape.Elements() > ofmShape.Width() * ofmShape.Depth()) )
    {
        return DecomposeLeadingDimensions(ofmRank - 2, arch, std::move(op), DecomposeMatmul);
    }

    result.emplace_back(std::move(op));
    return result;
}

std::vector<std::unique_ptr<SchedulerOperation>> DecomposeReduce(Architecture *arch, std::unique_ptr<SchedulerOperation> op)
{
    std::vector<std::unique_ptr<SchedulerOperation>> result;
    auto ofmConn = op->Output(TensorUsage::OFM);
    auto ofmShape = ofmConn->SliceShape();
    auto &ofmSlice = ofmConn->slice;
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto ifmShape = ifmConn->SliceShape();
    auto &ifmSlice = ifmConn->slice;

    InitializeSlice(ofmSlice, ofmShape.WithZeros(), ofmShape);
    InitializeSlice(ifmSlice, ifmShape.WithZeros(), ifmShape);

    if ( auto ifm2Conn = op->TryInput(TensorUsage::IFM1) )
    {
        auto ifm2Shape = ifm2Conn->shape;
        auto &ifm2Slice = ifm2Conn->slice;

        InitializeSlice(ifm2Slice, ifm2Shape.WithZeros(), ifm2Shape);
    }

    auto ofmRank = ofmShape.Size();
    auto attr = op->Attribute<axis_attr_t>();
    int reducedAxis = attr->axis;

    for ( int axis = 0; axis < ofmRank; axis++ )
    {
        if ( ofmShape[axis] > MAX_DIM )
        {
            if ( axis == reducedAxis )
            {
                // TODO: MLBEDSW-9408 reduced axis requires specific decomposition
                continue;
            }
            return DecomposeLargeAxis(axis, MAX_DIM, arch, std::move(op), DecomposeReduce);
        }
    }

    result.emplace_back(std::move(op));
    return result;
}

std::vector<std::unique_ptr<SchedulerOperation>> DecomposeReverse(Architecture *arch, std::unique_ptr<SchedulerOperation> op)
{
    std::vector<std::unique_ptr<SchedulerOperation>> result;
    auto ofmConn = op->Output(TensorUsage::OFM);
    auto ofmShape = ofmConn->SliceShape();
    auto &ofmSlice = ofmConn->slice;
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto ifmShape = ifmConn->SliceShape();
    auto &ifmSlice = ifmConn->slice;

    InitializeSlice(ofmSlice, ofmShape.WithZeros(), ofmShape);
    InitializeSlice(ifmSlice, ifmShape.WithZeros(), ifmShape);

    if ( auto ifm2Conn = op->TryInput(TensorUsage::IFM1) )
    {
        auto ifm2Shape = ifm2Conn->shape;
        auto &ifm2Slice = ifm2Conn->slice;

        InitializeSlice(ifm2Slice, ifm2Shape.WithZeros(), ifm2Shape);
    }

    auto ofmRank = ofmShape.Size();
    auto attr = op->Attribute<axis_attr_t>();
    int reversedAxis = attr->axis;

    for ( int axis = 0; axis < ofmRank; axis++ )
    {
        if ( ofmShape[axis] > MAX_DIM )
        {
            auto subOps = DecomposeLargeAxis(axis, MAX_DIM, arch, std::move(op), DecomposeReverse);
            if ( axis == reversedAxis )
            {
                // For the reversed axis, we need to invert
                // the slice-offsets for the OFM.
                for ( auto &subOp : subOps )
                {
                    auto *subOfmConn = subOp->Output(TensorUsage::OFM);
                    subOfmConn->slice.offset[axis] = ofmShape[axis] - subOfmConn->slice.offset[axis] - subOfmConn->SliceShape()[axis];
                }
            }
            return subOps;
        }
    }

    result.emplace_back(std::move(op));
    return result;
}

// Swap two axes of a shape by adding one or more transpose ops to a scheduler connection
static std::vector<std::unique_ptr<SchedulerOperation>> SwapAxes(Architecture *arch, Shape &shape, SchedulerConnection *tail, int a, int b)
{
    std::vector<std::unique_ptr<SchedulerOperation>> result;

    assert(shape.IsValid());
    assert(tail);
    assert(a < shape.Size());
    assert(b < shape.Size());
    assert(a < b);

    LOG_TRACE2("SwapAxes: Swap ({}), {} <-> {}\n", shape.ToString(), a, b);

    // The hardware can perform all permutations of a 3D tensor. This means we can swap two arbitrary axes of a shape
    // with N axes like this:
    //
    // 1. Reshape to a 3D shape (0*1 ..., A, ... N-2*N-1) and swap axis 0 and 1 in the 3D shape. Now A is leftmost in
    //    the original shape.
    // 2. Reshape to a 3D shape (0*1 ..., B, ... N-2*N-1) and swap axis 1 and 2 in the 3D shape. Now B is rightmost in
    //    the original shape.
    // 3. Swap 0 and N-1.
    // 4. Swap back axis N-1 to position B, like in step 2.
    // 5. Swap back axis 0 to position A, like in step 1.

    // We can handle all swaps in a 3D shape
    if ( shape.Size() < 4 )
    {
        // Build transpose type for this swap
        Shape perm(nullptr, shape.Size());
        for ( int i = 0; i < shape.Size(); i++ )
            perm[i] = i;
        std::swap(perm[a], perm[b]);
        const auto transposeType = TransposeTypeFromShape(perm);

        const Shape &ifmShape = shape;
        const Shape ofmShape = ifmShape.Permute(uint32_t(transposeType));
        LOG_TRACE2("SwapAxes: Transpose ({}) -> ({}), 0x{:08x}\n", ifmShape.ToString(), ofmShape.ToString(), transposeType);

        // Create SchedulerOperation
        auto op = std::make_unique<SchedulerOperation>(OpType::Transpose);
        Kernel kernel({1, 1} /* size */, {1, 1} /* stride */, {1, 1} /* dilation */);
        op->SetKernel(&kernel);
        auto ifmConn = op->AddInput(TensorUsage::IFM);
        auto ofmConn = op->AddOutput(TensorUsage::OFM);

        // Create SchedulerTensor
        auto newTensor = std::make_shared<SchedulerTensor>();
        newTensor->format = tail->tensor->format;
        newTensor->memArea = tail->tensor->memArea;
        newTensor->storageShape = ofmShape;
        newTensor->bufferView = BufferView(nullptr, 0, DataTypeSizeBits(tail->tensor->dataType), ofmShape, Shape());
        newTensor->dataType = tail->tensor->dataType;
        newTensor->uid = GenerateUniqueId();

        // Connect input/output
        Quantization unitQuantZp = Quantization::Unit();
        unitQuantZp.zeroPoints = tail->quantization.zeroPoints;
        unitQuantZp.forceZeroPoint = tail->quantization.forceZeroPoint;
        ifmConn->tensor = tail->tensor;
        ifmConn->tensor->consumers.push_back(op.get());
        ifmConn->shape = ifmShape;
        ifmConn->quantization = unitQuantZp;
        ofmConn->tensor = std::move(newTensor);
        ofmConn->tensor->producers.push_back(op.get());
        ofmConn->shape = ofmShape;
        ofmConn->quantization = std::move(unitQuantZp);
        ofmConn->transpose = transposeType;

        result.push_back(std::move(op));
        std::swap(shape[a], shape[b]);

        return result;
    }

    if ( a == 0 && b == 1 )
    {
        // Swap of index 0 and 1 can always be done with 1 op
        LOG_TRACE2("SwapAxes: Left-anchored swap\n");
        auto tmp = ReshapeTo3D(shape, {1, 1, shape.Size() - 2});
        auto ops = SwapAxes(arch, tmp, tail, 0, 1);
        result.insert(result.end(), std::make_move_iterator(ops.begin()), std::make_move_iterator(ops.end()));
        std::swap(shape[a], shape[b]);
        LOG_TRACE2("SwapAxes: Current shape ({})\n", shape.ToString());

        return result;
    }

    if ( a == shape.Size() - 2 && b == shape.Size() - 1 )
    {
        // Swap of index N-2 and N-1 can always be done with 1 op
        LOG_TRACE2("SwapAxes: Right-anchored swap\n");
        auto tmp = ReshapeTo3D(shape, {shape.Size() - 2, 1, 1});
        auto ops = SwapAxes(arch, tmp, tail, 1, 2);
        result.insert(result.end(), std::make_move_iterator(ops.begin()), std::make_move_iterator(ops.end()));
        std::swap(shape[a], shape[b]);
        LOG_TRACE2("SwapAxes: Current shape ({})\n", shape.ToString());

        return result;
    }

    if ( a != 0 )
    {
        // Move A left
        LOG_TRACE2("SwapAxes: Move index {} ({}) leftmost\n", a, shape[a]);
        auto tmp1 = ReshapeTo3DAroundAxis(shape, a);
        auto ops1 = SwapAxes(arch, tmp1, tail, 0, 1);
        result.insert(result.end(), std::make_move_iterator(ops1.begin()), std::make_move_iterator(ops1.end()));
        shape = shape.Erase(a).Insert(0, shape[a]);
        LOG_TRACE2("SwapAxes: Current shape ({})\n", shape.ToString());

        // Swap (A is now leftmost)
        auto ops = SwapAxes(arch, shape, result.back()->OFM(), 0, b);
        result.insert(result.end(), std::make_move_iterator(ops.begin()), std::make_move_iterator(ops.end()));

        // Move A back to where it was
        LOG_TRACE2("SwapAxes: Move leftmost ({}) back to index {}\n", shape[0], a);
        auto tmp2 = ReshapeTo3D(shape, {1, a, shape.Size() - a - 1});
        auto ops2 = SwapAxes(arch, tmp2, result.back()->OFM(), 0, 1);
        result.insert(result.end(), std::make_move_iterator(ops2.begin()), std::make_move_iterator(ops2.end()));
        shape = shape.Erase(0).Insert(a, shape[0]);
        LOG_TRACE2("SwapAxes: Current shape ({})\n", shape.ToString());

        return result;
    }

    if ( b != shape.Size() - 1 )
    {
        // Move B right
        LOG_TRACE2("SwapAxes: Move index {} ({}) rightmost\n", b, shape[b]);
        auto tmp1 = ReshapeTo3DAroundAxis(shape, b);
        auto ops1 = SwapAxes(arch, tmp1, tail, 1, 2);
        result.insert(result.end(), std::make_move_iterator(ops1.begin()), std::make_move_iterator(ops1.end()));
        shape = shape.Erase(b).Insert(-1, shape[b]);
        LOG_TRACE2("SwapAxes: Current shape ({})\n", shape.ToString());

        // Swap (A is leftmost and B is rightmost)
        auto ops = SwapAxes(arch, shape, result.back()->OFM(), 0, shape.Size() - 1);
        result.insert(result.end(), std::make_move_iterator(ops.begin()), std::make_move_iterator(ops.end()));

        // Move B back to where it was
        LOG_TRACE2("SwapAxes: Move rightmost ({}) back to index {}\n", shape[-1], b);
        auto tmp2 = ReshapeTo3D(shape, {b, shape.Size() - b - 1, 1});
        auto ops2 = SwapAxes(arch, tmp2, result.back()->OFM(), 1, 2);
        result.insert(result.end(), std::make_move_iterator(ops2.begin()), std::make_move_iterator(ops2.end()));
        shape = shape.Erase(-1).Insert(b, shape[-1]);
        LOG_TRACE2("SwapAxes: Current shape ({})\n", shape.ToString());

        return result;
    }

    // Swap (A is leftmost and B is rightmost)
    assert(a == 0);
    assert(b == shape.Size() - 1);
    LOG_TRACE2("SwapAxes: Swap leftmost and rightmost\n");
    auto tmp = ReshapeTo3DAroundEdges(shape);
    auto ops = SwapAxes(arch, tmp, tail, 0, 2);
    result.insert(result.end(), std::make_move_iterator(ops.begin()), std::make_move_iterator(ops.end()));
    std::swap(shape[a], shape[b]);
    LOG_TRACE2("SwapAxes: Current shape ({})\n", shape.ToString());

    return result;
}

std::vector<std::unique_ptr<SchedulerOperation>> DecomposeTranspose(Architecture *arch, std::unique_ptr<SchedulerOperation> op)
{
    std::vector<std::unique_ptr<SchedulerOperation>> result;
    auto ifmConn = op->Input(TensorUsage::IFM);
    auto ofmConn = op->Output(TensorUsage::OFM);
    const auto attr = op->Attribute<transpose_attr_t>();
    const auto &perm = attr->perm;
    const auto axes = attr->perm.Size();

    // Calculate sort order
    Shape order(nullptr, axes);
    for ( int i = 0; i < axes; i++ )
        order[perm[i]] = i;

    auto shape = ifmConn->shape;

    LOG_TRACE1("DecomposeTranspose: Initial shape ({})\n", shape.ToString());

    // Decompose a transpose peforming a selection sort of the axes. Each swap in the selection sort algorithm expands
    // to one or more transpose ops.
    //
    // Example:
    //
    // Input shape:        [ 3,  7, 11, 13]
    // Permutation vector: [ 1,  3,  0,  2]
    // Sort order:         [ 2,  0,  3,  1]
    // Output shape:       [ 7, 13,  3, 11]
    //
    // Selection sort swaps:
    //
    // Swap 1: Pos 0 <-> Pos 1: [7, 3,  11, 13]
    // Swap 2: Pos 1 <-> Pos 3: [7, 13, 11,  3]
    // Swap 3: Pos 2 <-> Pos 3: [7, 13,  3, 11]

    for ( int axis = 0; axis < axes; axis++ )
    {
        // Check if axis is already in the right place
        if ( order[axis] == axis ) continue;

        // Find where the axis is
        int i;
        for ( i = axis + 1; i < axes; i++ )
            if ( order[i] == axis ) break;
        assert(i < axes);

        // Move axis to right place
        LOG_TRACE1("DecomposeTranspose: Swap {} <-> {}\n", axis, i);
        auto tail = !result.empty() ? result.back()->OFM() : ifmConn;
        auto subOps = SwapAxes(arch, shape, tail, axis, i);
        result.insert(result.end(), std::make_move_iterator(subOps.begin()), std::make_move_iterator(subOps.end()));
        std::swap(order[axis], order[i]);
        LOG_TRACE1("DecomposeTranspose: Shape is now ({})\n", shape.ToString());
    }

    LOG_TRACE1("DecomposeTranspose: Final shape ({})\n", shape.ToString());

    if ( result.empty() )
    {
        // Didn't decompose the op
        result.push_back(std::move(op));
    }
    else
    {
        // Adjust to that first is read from the original IFM
        auto ifm = result.front()->IFM(0);
        ifm->tensor = ifmConn->tensor;
        ifm->quantization = ifmConn->quantization;

        // Adjust to that last output is written to the original OFM
        auto ofm = result.back()->OFM();
        ofm->tensor = ofmConn->tensor;
        ofm->quantization = ofmConn->quantization;
    }

    return result;
}

}  // namespace regor
