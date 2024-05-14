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

#include "operation_util.hpp"

#include <numeric>
#include <optional>

namespace regor
{

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
    subOp->SetParameters(schedOp->Parameters());
    subOp->SetRounding(schedOp->Rounding());
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
    auto *ifm = schedOp->TryIFM(0);
    auto *ifm2 = schedOp->TryIFM(1);
    auto *ofm = schedOp->TryOFM();

    if ( !ifm || !ofm ) return false;
    qOpGroup.type = schedOp->Type();
    qOpGroup.kernel = schedOp->Kernel();
    qOpGroup.ifm.key = ifm->tensor->uid;
    qOpGroup.ifm.type = ifm->tensor->dataType;
    if ( ifm2 )
    {
        qOpGroup.ifm2.key = ifm2->tensor->uid;
        qOpGroup.ifm2.type = ifm2->tensor->dataType;
    }
    qOpGroup.ofm.key = ofm->tensor->uid;
    qOpGroup.ofm.type = ofm->tensor->dataType;
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
    qConfig.ofmShape = qConfig.ofmShape.Untranspose(ofm->transpose);
    qConfig.transpose = ofm->transpose;
    qConfig.ofmFormat = ofm->tensor->format;
    const auto &accMode = schedOp->AccumulatorMode();
    qConfig.accSource = GetArchAccumulatorSource(accMode);
    qConfig.accOutputEnabled = accMode.outputEnabled;
    return arch->GetOpConfig(schedOp->Type(), qConfig) != nullptr;
}

bool CanDecompose(Architecture *, const SchedulerOperation *schedOp)
{
    // TODO: MLBEDSW-8868 Restore fused transpose/reverse
    if ( auto ofm = schedOp->TryOFM(); ofm && (ofm->transpose != TransposeType::None || ofm->reverse != ReverseType::None) )
        return false;
    if ( schedOp->Type() == OpType::Conv2D ) return true;
    if ( schedOp->Type() == OpType::DepthwiseConv2DBias ) return true;
    return false;
}

using DecomposeFunc = std::vector<std::unique_ptr<SchedulerOperation>> (*)(Architecture *, std::unique_ptr<SchedulerOperation>);

// Decompose to sub-operations with size 1 along the leading <dimension> axes.
// Used for the batch dimension, and for the leading N-3 dimensions for elementwise operations.
static std::vector<std::unique_ptr<SchedulerOperation>> DecomposeLeadingDimensions(
    int dimensions, Architecture *arch, std::unique_ptr<SchedulerOperation> op, DecomposeFunc doDecompose)
{
    std::vector<std::unique_ptr<SchedulerOperation>> result;
    int axis = --dimensions;
    auto *ofmConn = op->Output(TensorUsage::OFM);
    auto *ifmConn = op->Input(TensorUsage::IFM0);
    auto *ifm2Conn = op->TryInput(TensorUsage::IFM1);
    auto newIfmSlice = ifmConn->slice;
    auto newOfmSlice = ofmConn->slice;
    newIfmSlice.shape[axis] = 1;
    newOfmSlice.shape[axis] = 1;
    TensorSlice newIfm2Slice;
    if ( ifm2Conn != nullptr )
    {
        newIfm2Slice = ifm2Conn->slice;
        newIfm2Slice.shape[axis] = 1;
    }
    auto dimSize = ofmConn->shape[axis];
    for ( int i = 0; i < dimSize; i++ )
    {
        std::unique_ptr<SchedulerOperation> subOp = MakeSubOperation(op.get());
        newIfmSlice.offset[axis] = i;
        newOfmSlice.offset[axis] = i;
        if ( ifm2Conn != nullptr )
        {
            newIfm2Slice.offset[axis] = i;
        }
        subOp->Input(TensorUsage::IFM)->slice = newIfmSlice;
        subOp->Output(TensorUsage::OFM)->slice = newOfmSlice;
        if ( ifm2Conn != nullptr )
        {
            subOp->Input(TensorUsage::IFM1)->slice = newIfm2Slice;
        }
        auto subOps = (dimensions > 0) ? DecomposeLeadingDimensions(dimensions, arch, std::move(subOp), doDecompose) : doDecompose(arch, std::move(subOp));
        result.insert(result.end(), std::make_move_iterator(subOps.begin()), std::make_move_iterator(subOps.end()));
    }
    return result;
}

// Handle dilation by decomposing to suboperations with input stride = dilation and dilation 1
static std::vector<std::unique_ptr<SchedulerOperation>>
HandleDilation(Architecture *arch, std::unique_ptr<SchedulerOperation> op, DecomposeFunc doDecompose)
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
    return result;
}

// Negative ifm offsets indicate new padding values with ifm offset 0
static void UpdatePaddingIfOffsetNegative(SchedulerOperation *op)
{
    auto &ifmSlice = op->Input(TensorUsage::IFM)->slice;
    if ( ifmSlice.offset.Height() < 0 || ifmSlice.offset.Width() < 0 )
    {
        auto *kernel = op->Kernel();
        auto &padding = kernel->Padding();
        auto topPad = std::max(0, -ifmSlice.offset.Height());
        auto leftPad = std::max(0, -ifmSlice.offset.Width());
        auto newPadding = Margin(topPad, leftPad, padding.Bottom(), padding.Right());
        ifmSlice.offset[1] = std::max(0, ifmSlice.offset.Height());
        ifmSlice.offset[2] = std::max(0, ifmSlice.offset.Width());
        auto newKernel = kernel->WithPadding(newPadding);
        op->SetKernel(&newKernel);
    }
}

std::vector<std::unique_ptr<SchedulerOperation>> DecomposeConv2D(Architecture *arch, std::unique_ptr<SchedulerOperation> op)
{
    std::vector<std::unique_ptr<SchedulerOperation>> result;
    auto *ofmConn = op->Output(TensorUsage::OFM);
    auto *ifmConn = op->Input(TensorUsage::IFM);
    const auto &ofmShape = ofmConn->shape;
    const auto &ifmShape = ifmConn->shape;
    auto &ofmSlice = ofmConn->slice;
    auto &ifmSlice = ifmConn->slice;
    auto *kernel = op->Kernel();
    auto &padding = kernel->Padding();
    if ( !ofmSlice.offset.IsValid() )
    {
        ofmSlice.offset = ofmShape.WithZeros();
        ifmSlice.offset = ifmShape.WithZeros().WithHW(-padding.Top(), -padding.Left());
    }
    if ( ofmShape.Batch() > 1 )
    {
        return DecomposeLeadingDimensions(1, arch, std::move(op), DecomposeConv2D);
    }
    if ( CanRunOnHardware(arch, op.get()) )
    {
        UpdatePaddingIfOffsetNegative(op.get());
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
    const auto &ofmShape = ofmConn->shape;
    const auto &ifmShape = ifmConn->shape;
    const auto &weightsShape = weightsConn->shape;
    auto &ofmSlice = ofmConn->slice;
    auto &ifmSlice = ifmConn->slice;
    auto *kernel = op->Kernel();
    auto &padding = kernel->Padding();
    if ( !ofmSlice.offset.IsValid() )
    {
        ofmSlice.offset = ofmShape.WithZeros();
        ifmSlice.offset = ifmShape.WithZeros().WithHW(-padding.Top(), -padding.Left());
    }
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
        UpdatePaddingIfOffsetNegative(op.get());
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

}  // namespace regor
