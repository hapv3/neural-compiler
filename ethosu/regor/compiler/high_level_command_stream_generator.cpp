//
// SPDX-FileCopyrightText: Copyright 2021-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "compiler/compiler.hpp"
#include "high_level_command_stream.hpp"
#include "scheduler.hpp"

#include <unordered_map>
#include <vector>

namespace regor
{

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

static void MakeFeatureMap(TensorUsage usage, const SchedulerConnection *schedConn, HLCFeatureMap &fm, Address base,
    std::unordered_map<UniqueId, Address> &addresses)
{
    auto schedTens = schedConn->tensor.get();
    fm.shape = schedConn->shape;
    fm.slice = schedConn->slice;
    fm.dataType = schedConn->Type();
    fm.memArea = schedTens->memArea;
    fm.format = schedTens->format;
    fm.usage = usage;
    // Calculate address of this FeatureMap
    auto address = schedTens->AllocatedAddress();
    if ( address != -1 )
    {
        address += (schedTens->memArea.usage.Any(MemUsage::FeatureMap) ? base : 0);
    }
    // Look up (or store) address
    auto item = addresses.emplace(schedTens->equivalenceId, address);
    fm.address = (*item.first).second;
    fm.quantization = schedConn->quantization;
    if ( schedTens->bufferView.HasBuffer() )
    {
        fm.constBuffer = schedTens->bufferView.Buffer()->shared_from_this();
    }
    fm.strides = GetStrides(fm);
    fm.stepXY = schedConn->stepXY;
    fm.transpose = schedConn->transpose;
    fm.reverse = schedConn->reverse;
    fm.resamplingMode = schedConn->resamplingMode;
    fm.rounding = HLCRoundMode(schedConn->rounding);
    fm.uid = schedTens->uid;
}

static std::unique_ptr<HLCWeights>
MakeWeights(NpuWeightTensor *srcTensor, Buffering buffering, SchedulerTensor *bufTensor, SchedulerTensor *buf2Tensor)
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
    weights->address[0] = bufTensor->AllocatedAddress();
    weights->address[1] = buf2Tensor ? buf2Tensor->AllocatedAddress() : -1;
    weights->memArea = bufTensor->memArea;
    weights->buffering = buffering;
    // Same function is used for generating scales - scales have no config or weight format, so set to default
    weights->format = srcTensor->config ? srcTensor->config->Format() : Flags(WeightFormat::Default);
    weights->subStreams = srcTensor->subStreams;
    weights->encodedRanges = srcTensor->encodedRanges;
    return weights;
}

static HLCSubOperation MakeSubOperation(
    const std::unique_ptr<SchedulerOperation> &schedOp, Address base, std::unordered_map<UniqueId, Address> &addresses)
{
    HLCSubOperation hlcSubOp;
    hlcSubOp.type = schedOp->Type();
    auto lutConn = schedOp->TryInput(TensorUsage::LUT);
    size_t ifms = 0;
    for ( const auto &input : schedOp->inputs.pairs() )
    {
        if ( IsIFM(input.first) || GetUsageType(input.first) == TensorUsage::Scratch )
        {
            std::vector<HLCFeatureMap>::iterator at;
            if ( IsIFM(input.first) )
            {
                // Insert IFMs, into the IFM section [0..ifms) sorted into order.
                at = hlcSubOp.ifm.emplace(std::upper_bound(hlcSubOp.ifm.begin(),
                    hlcSubOp.ifm.begin() + std::min(ifms, hlcSubOp.ifm.size()), input.first,
                    [](TensorUsage usage, const HLCFeatureMap &fm) { return usage < fm.usage; }));
                ifms++;  // Increase size of IFM section
            }
            else
            {
                // Non-IFM tensors get appended
                at = hlcSubOp.ifm.emplace(hlcSubOp.ifm.end());
            }
            MakeFeatureMap(input.first, &input.second, *at, base, addresses);
        }
    }
    MakeFeatureMap(TensorUsage::OFM, schedOp->OFM(), hlcSubOp.ofm, base, addresses);

    hlcSubOp.srcId = schedOp->Uid();

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
        param.ifmType = schedOp->IFM(0)->Type();
    }
    else if ( schedOp->Type() == OpType::Add || schedOp->Type() == OpType::Sub )
    {
        if ( schedOp->HasAttribute<double_round_shift_attr_t>() )
        {
            const auto *double_round_shift = schedOp->Attribute<double_round_shift_attr_t>();
            hlcSubOp.parameters.double_round.shift = double_round_shift->shift;
        }
        else
        {
            // Default to zero
            hlcSubOp.parameters.double_round.shift = 0;
        }
    }
    return hlcSubOp;
}

static std::shared_ptr<HLCOperation> MakeOperation(SchedulerOperation *schedOp, SchedulerOpInfo *opInfo, Address base,
    std::unordered_map<UniqueId, Address> &addresses)
{
    assert(opInfo);
    auto op = std::make_shared<HLCOperation>();
    op->type = schedOp->Type();
    op->kernel = *schedOp->Kernel();
    op->config = opInfo->Config();
    op->srcId = schedOp->Uid();
    size_t ifms = 0;
    for ( const auto &input : schedOp->inputs.pairs() )
    {
        if ( IsIFM(input.first) || GetUsageType(input.first) == TensorUsage::Scratch )
        {
            std::vector<HLCFeatureMap>::iterator at;
            if ( IsIFM(input.first) )
            {
                // Insert IFMs, into the IFM section [0..ifms) sorted into order.
                at = op->ifm.emplace(std::upper_bound(op->ifm.begin(), op->ifm.begin() + std::min(ifms, op->ifm.size()),
                    input.first, [](TensorUsage usage, const HLCFeatureMap &fm) { return usage < fm.usage; }));
                ifms++;  // Increase size of IFM section
            }
            else
            {
                // Non-IFM tensors get appended
                at = op->ifm.emplace(op->ifm.end());
            }
            MakeFeatureMap(input.first, &input.second, *at, base, addresses);
        }
    }
    MakeFeatureMap(TensorUsage::OFM, schedOp->OFM(), op->ofm, base, addresses);

#ifndef NDEBUG
    op->name = schedOp->OFM()->tensor->Name();
#endif
    if ( opInfo->npuWeightsTensor != nullptr )
    {
        assert(schedOp->TryInput(TensorUsage::Weights) != nullptr);
        op->weights = MakeWeights(opInfo->npuWeightsTensor.get(), opInfo->bufferedWeightTensor.buffering,
            opInfo->bufferedWeightTensor.tensor[0].get(), opInfo->bufferedWeightTensor.tensor[1].get());
    }

    if ( opInfo->npuScalesTensor != nullptr )
    {
        // Only scales encoded
        op->scales = MakeWeights(opInfo->npuScalesTensor.get(), Buffering::None, nullptr, nullptr);
    }
    else if ( schedOp->TryInput(TensorUsage::Scales) != nullptr )
    {
        // Weights and scales encoded together
        assert(!!opInfo->npuWeightsTensor);
        op->scales = MakeWeights(opInfo->npuWeightsTensor.get(), opInfo->bufferedWeightTensor.buffering,
            opInfo->bufferedWeightTensor.tensor[0].get(), opInfo->bufferedWeightTensor.tensor[1].get());
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
        param.ifmType = schedOp->IFM(0)->Type();
    }

    for ( auto &subOp : schedOp->SubOps() )
    {
        HLCSubOperation hlcSubOp = MakeSubOperation(subOp, base, addresses);
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
            if ( resize->mode == tosa::ResizeMode::NEAREST )
            {
                op->parameters.resize.mode = ArchResizeMode::Nearest;
            }
            else
            {
                assert(resize->mode == tosa::ResizeMode::BILINEAR);
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
        case OpType::Add:
        case OpType::Sub:
        {
            if ( schedOp->HasAttribute<double_round_shift_attr_t>() )
            {
                const auto *double_round_shift = schedOp->Attribute<double_round_shift_attr_t>();
                op->parameters.double_round.shift = double_round_shift->shift;
            }
            else
            {
                // Default to zero
                op->parameters.double_round.shift = 0;
            }
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
        if ( cmds[i]->CommandType() == HighLevelCommandType::STRIPE )
        {
            return static_cast<HLCStripe *>(cmds[i].get());
        }
    }
    assert(fromIndex != 0);  // Every stream should contain at least one stripe
    return nullptr;
}

// Returns true when more output can be produced without overflowing the rolling buffer.
static bool CanFitRollingBuffer(const CascadeInfo *cascadeInfo, vector_span<std::unique_ptr<SchedulerOperation>> cascadedOps,
    const std::vector<HLCStripe *> &availableStripe, const std::vector<HLCStripe *> &nextStripe, int opIndex)
{
    int nextOpIndex = opIndex + 1;
    assert(opIndex >= 0 && opIndex < int(nextStripe.size()));
    assert(nextOpIndex > 0 && nextOpIndex < int(nextStripe.size()));
    if ( nextOpIndex >= int(cascadedOps.size()) )
    {
        return false;
    }
    if ( nextStripe[opIndex] == nullptr || nextStripe[nextOpIndex] == nullptr )
    {
        return false;
    }
    auto bufferPos = cascadeInfo->buffers.find(*cascadedOps[nextOpIndex]);
    if ( bufferPos == cascadeInfo->buffers.end() )
    {
        return false;
    }
    int bufferHeight = bufferPos->second.shape.Height();
    int primaryIfmIndex = cascadedOps[nextOpIndex]->PrimaryIfmIndex();
    // Use the next producer stripe to check if emitting it would overflow the rolling buffer.
    const Shape &producerEnd = nextStripe[opIndex]->stripeAreas[0].ofmArea.End();
    const Shape &consumerStart = nextStripe[nextOpIndex]->stripeAreas[0].ifmAreas.at(primaryIfmIndex).Start();
    int bufferedHeight = producerEnd.Height() - consumerStart.Height();
    return bufferedHeight <= bufferHeight;
}

// Generates Branch command for If/While
void HLCStreamGenerator::GenerateHLCBranchCommands(
    SchedulerOperation *op, const std::shared_ptr<HLCOperation> &hlcOp, SubGraphs &subgraphs, HLCStream &cmds)
{
    const auto opType = hlcOp->type;
    assert(IsControlFlow(opType));

    // Carveout that all FeatureMap tensors in the subgraphs (except the graph input/outputs) will use
    auto *carveOut = op->Input(TensorUsage::Scratch);
    assert(carveOut);
    auto carveOutBase = carveOut->tensor->AllocatedAddress();

    // Store the addresses of the inputs/outputs so they can be reused by the subgraphs
    for ( const auto &[ifmUsage, ifmConn] : op->inputs.pairs() )
    {
        if ( IsIFM(ifmUsage) && _addresses.count(ifmConn.tensor->equivalenceId) == 0 )
            _addresses[ifmConn.tensor->equivalenceId] = ifmConn.tensor->AllocatedAddress() + _base;
    }
    for ( const auto &[ofmUsage, ofmConn] : op->outputs.pairs() )
    {
        if ( IsOFM(ofmUsage) && _addresses.count(ofmConn.tensor->equivalenceId) == 0 )
            _addresses[ofmConn.tensor->equivalenceId] = ofmConn.tensor->AllocatedAddress() + _base;
    }

    if ( opType == OpType::If )
    {
        // Generate a HLC stream for this IF operation. Basic strategy:
        //
        //  BRANCH (to "THEN" or "ELSE")
        // THEN:
        //  INLINED "THEN" STREAM
        //  BRANCH ALWAYS (to "EXIT")
        // ELSE:
        //  INLINED "ELSE" STREAM
        //  BRANCH ALWAYS (to "EXIT")
        // EXIT:

        const auto attr = op->Attribute<internal_if_attr_t>();

        // Deal with "then" branch
        assert(attr->then_graph);
        CompiledGraph *thenGraph = reinterpret_cast<CompiledGraph *>(attr->then_graph);
        assert(thenGraph->npuOps.size() == 1);  // Subgraph fully mapped to NPU
        assert(thenGraph->npuOps[0].first->Type() == OpType::CustomNpuOp);
        HLCStreamGenerator thenGen(_base + carveOutBase, _verbose, _addresses);
        HLCStream thenStream = thenGen.GenerateCommandStream(
            thenGraph->npuOps[0].second.get(), thenGraph->schedule.get(), thenGraph->compiledSubGraphs);

        // Deal with "else" branch
        assert(attr->else_graph);
        CompiledGraph *elseGraph = reinterpret_cast<CompiledGraph *>(attr->else_graph);
        assert(elseGraph->npuOps.size() == 1);  // Subgraph fully mapped to NPU
        assert(elseGraph->npuOps[0].first->Type() == OpType::CustomNpuOp);
        HLCStreamGenerator elseGen(_base + carveOutBase, _verbose, _addresses);
        HLCStream elseStream = elseGen.GenerateCommandStream(
            elseGraph->npuOps[0].second.get(), elseGraph->schedule.get(), elseGraph->compiledSubGraphs);

        // Create branch targets
        auto exit = std::make_unique<HLCBranchTarget>();
        auto thenEntry = std::make_unique<HLCBranchTarget>();
        auto elseEntry = std::make_unique<HLCBranchTarget>();

        // Create HLCBranch for the condition
        auto condBranch = std::make_unique<HLCBranch>();
        condBranch->true_target = thenEntry.get();
        condBranch->false_target = elseEntry.get();

        // Add a branch back to the call site
        auto branchAfterThen = std::make_unique<HLCBranch>();
        branchAfterThen->true_target = exit.get();

        // Add a branch back to the call site
        auto branchAfterElse = std::make_unique<HLCBranch>();
        branchAfterElse->true_target = exit.get();

        // Assemble the HLC stream for this IF operation
        cmds.push_back(std::move(condBranch));
        cmds.push_back(std::move(thenEntry));
        cmds.insert(cmds.end(), std::make_move_iterator(thenStream.begin()), std::make_move_iterator(thenStream.end()));
        cmds.push_back(std::move(branchAfterThen));
        cmds.push_back(std::move(elseEntry));
        cmds.insert(cmds.end(), std::make_move_iterator(elseStream.begin()), std::make_move_iterator(elseStream.end()));
        cmds.push_back(std::move(branchAfterElse));
        cmds.push_back(std::move(exit));
    }
    else if ( opType == OpType::While )
    {
        // Generate a HLC stream for this WHILE operation. Basic strategy:
        //
        //  COPY WHILE IFMs TO WHILE OFMs
        // COND:
        //  INLINED "COND" STREAM (this consumes WHILE IFMs)
        //  BRANCH (to "BODY" or "EXIT")
        // BODY:
        //  INLINED "BODY" STREAM (this produces WHILE OFMs)
        //  COPY WHILE OFMs TO WHILE IFMs
        //  BRANCH ALWAYS (to "COND")
        // EXIT:

        const auto attr = op->Attribute<internal_while_attr_t>();

        // Generate HLCDMA to copy IFM -> OFM
        HLCStream copyAtoB;
        for ( const auto &[ifmUsage, ifmConn] : op->inputs.pairs() )
        {
            if ( !IsIFM(ifmUsage) ) continue;

            // Find IFM and matching OFM
            const auto &ifm = ifmConn.tensor;
            assert(!ifm->IsConstant());
            const auto ofmConn = op->Output(MakeTensorUsage(TensorUsage::OFM, GetUsageIndex(ifmUsage)));
            assert(ofmConn);
            const auto &ofm = ofmConn->tensor;
            assert(!ofm->IsConstant());

            // Create a HLCDMA copy op that copies data in the input tensor to the output tensor
            assert(ifm->AllocationSizeBytes() == ofm->AllocationSizeBytes());
            assert(ifm->format == ofm->format);
            auto dma = std::make_unique<HLCDMA>();
            dma->length = ifm->AllocationSizeBytes();
            dma->srcAddress = _addresses.at(ifm->equivalenceId);
            dma->srcMemArea = ifm->memArea;
            dma->destAddress = _addresses.at(ofm->equivalenceId);
            dma->destMemArea = ofm->memArea;
            copyAtoB.push_back(std::move(dma));
        }

        // Generate HLCDMA to copy OFM -> IFM
        HLCStream copyBtoA;
        for ( const auto &[ifmUsage, ifmConn] : op->inputs.pairs() )
        {
            if ( !IsIFM(ifmUsage) ) continue;

            // Find IFM and matching OFM
            const auto &ifm = ifmConn.tensor;
            assert(!ifm->IsConstant());
            const auto ofmConn = op->Output(MakeTensorUsage(TensorUsage::OFM, GetUsageIndex(ifmUsage)));
            assert(ofmConn);
            const auto &ofm = ofmConn->tensor;
            assert(!ofm->IsConstant());

            // Create a HLCDMA copy op that copies data in the output tensor to the input tensor
            assert(ifm->AllocationSizeBytes() == ofm->AllocationSizeBytes());
            assert(ifm->format == ofm->format);
            auto dma = std::make_unique<HLCDMA>();
            dma->length = ifm->AllocationSizeBytes();
            dma->srcAddress = _addresses.at(ofm->equivalenceId);
            dma->srcMemArea = ofm->memArea;
            dma->destAddress = _addresses.at(ifm->equivalenceId);
            dma->destMemArea = ifm->memArea;
            copyBtoA.push_back(std::move(dma));
        }

        // Deal with "cond" branch
        assert(attr->cond_graph);
        CompiledGraph *condGraph = reinterpret_cast<CompiledGraph *>(attr->cond_graph);
        assert(condGraph->npuOps.size() == 1);  // Subgraph fully mapped to NPU
        assert(condGraph->npuOps[0].first->Type() == OpType::CustomNpuOp);
        HLCStreamGenerator condGen(_base + carveOutBase, _verbose, _addresses);
        HLCStream condStream = condGen.GenerateCommandStream(
            condGraph->npuOps[0].second.get(), condGraph->schedule.get(), condGraph->compiledSubGraphs);

        // Deal with "body" branch
        assert(attr->body_graph);
        CompiledGraph *bodyGraph = reinterpret_cast<CompiledGraph *>(attr->body_graph);
        assert(bodyGraph->npuOps.size() == 1);  // Subgraph fully mapped to NPU
        assert(bodyGraph->npuOps[0].first->Type() == OpType::CustomNpuOp);
        HLCStreamGenerator bodyGen(_base + carveOutBase, _verbose, _addresses);
        HLCStream bodyStream = bodyGen.GenerateCommandStream(
            bodyGraph->npuOps[0].second.get(), bodyGraph->schedule.get(), bodyGraph->compiledSubGraphs);

        // Create branch targets
        auto exit = std::make_unique<HLCBranchTarget>();
        auto bodyEntry = std::make_unique<HLCBranchTarget>();
        auto condEntry = std::make_unique<HLCBranchTarget>();

        // Create the branch after COND
        auto branchAfterCond = std::make_unique<HLCBranch>();
        branchAfterCond->true_target = bodyEntry.get();
        branchAfterCond->false_target = exit.get();

        // Create the branch after BODY
        auto branchAfterBody = std::make_unique<HLCBranch>();
        branchAfterBody->true_target = condEntry.get();
        branchAfterBody->false_target = nullptr;

        // Assemble the HLC stream for this WHILE operation
        cmds.insert(cmds.end(), std::make_move_iterator(copyAtoB.begin()), std::make_move_iterator(copyAtoB.end()));
        cmds.push_back(std::move(condEntry));
        cmds.insert(cmds.end(), std::make_move_iterator(condStream.begin()), std::make_move_iterator(condStream.end()));
        cmds.push_back(std::move(branchAfterCond));
        cmds.push_back(std::move(bodyEntry));
        cmds.insert(cmds.end(), std::make_move_iterator(bodyStream.begin()), std::make_move_iterator(bodyStream.end()));
        cmds.insert(cmds.end(), std::make_move_iterator(copyBtoA.begin()), std::make_move_iterator(copyBtoA.end()));
        cmds.push_back(std::move(branchAfterBody));
        cmds.push_back(std::move(exit));
    }
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
    assert(bufConn.parts);
    unsigned index = unsigned(depthIndex) % bufConn.parts;
    assert(index < std::size(bufConn.tensor));
    dma->destMemArea = bufConn.tensor[index]->memArea;
    dma->destAddress = bufConn.tensor[index]->AllocatedAddress();
    return dma;
}

static std::tuple<Shape, Shape, HLCPadding> CalculateStripePadding(const Shape &stripeOffset, const Shape &stripeShape,
    const Shape &ifmShape, Point2i ifmStep, const Shape &ofmShape, Point2i ofmStep, const HLCPadding &stripePadding, OpType opType)
{
    const Shape stripeEnd = stripeOffset + stripeShape;
    // Set padding if ifmStripe is outside the IFM shape
    Point2i startPad = Point2i::Max({0, 0}, Point2i(0, 0) - stripeOffset.WH());
    Point2i endPad = Point2i::Max({0, 0}, stripeEnd.WH() - ifmShape.WH());
    // Adjust offset and shape based on padded values
    Shape newStart = stripeOffset.WithHW(stripeOffset.WH() + startPad);
    Shape newEnd = stripeEnd.WithHW(stripeEnd.WH() - endPad);
    // When operations use stepping, we need to adjust padding as if we were also stepping
    // in the padded area.
    bool stepped = (ofmStep != Point2i{1, 1} || ifmStep != Point2i{1, 1});
    if ( opType != OpType::MatMul && stepped )
    {
        // Adjust stripe start-coordinate as if we were performing input steps in the padded-area
        Point2i startAdjustForPadFraction = DivRoundUp(startPad, ifmStep) * ifmStep - startPad;
        // Divide top/left padding by input step to simulate stepping
        startPad = Point2i::Max({0, 0}, DivRoundUp(startPad, ifmStep));
        newStart = newStart.WithHW(newStart.WH() + startAdjustForPadFraction);
        newEnd = newEnd.WithHW(Point2i::Min(newEnd.WH() + startAdjustForPadFraction, ifmShape.WH()));
        // Adjust start-coordinate based on stepped padding
        Point2i neededInput = DivRoundUp(stripeShape.WH(), ifmStep);
        // Calculate bottom/right padding based on new start/end coordinates
        endPad = Point2i::Max({0, 0}, neededInput - (DivRoundUp(ifmShape.WH() - newStart.WH(), ifmStep) + startPad));
    }
    Shape newShape = newEnd - newStart;
    assert(newShape.Elements() > 0);
    return std::make_tuple<Shape, Shape, HLCPadding>(
        std::move(newStart), std::move(newShape), {startPad.y, startPad.x, endPad.y, endPad.x});
}

// Calculate the IFM-shape required to produce the OFM-shape for the stripe.
// If upscaling is used, this function returns the upscaled shape requirement.
// (shape needs to be adjusted for padding before it can be downscaled)
static Shape CalculateUpscaledIfmStripeShape(const Shape &ofmStripeShape, const Point2i &ofmStep,
    const Point2i &ifmStep, const Kernel *kernel, const TensorSlice &ifmSlice, OpType opType, int upscaling)
{
    auto dilatedSize = kernel->DilatedWH();
    auto effectiveKernelStride = kernel->Stride() * ifmStep;
    // Amount of written elements in the OfmStripeShape when accounting for ofmStep
    Point2i writtenElements = DivRoundUp(Point2i(ofmStripeShape.Width(), ofmStripeShape.Height()), ofmStep);
    // size of IFM covered by the kernel when accounting for ifmStep
    Point2i kernelBorder = (dilatedSize - Point2i(1, 1)) * ifmStep + Point2i(1, 1);
    // calculated required width/height in upscaled space
    int requiredWidth = RequiredInputSize(writtenElements.x, effectiveKernelStride.x, kernelBorder.x, 1, 0);
    int requiredHeight = RequiredInputSize(writtenElements.y, effectiveKernelStride.y, kernelBorder.y, 1, 0);
    // Truncate (or promote) to IFM-rank (but at least 4D)
    int ifmRank = std::max(4, ifmSlice.shape.Size());
    Shape stripeShape = Shape(ofmStripeShape, ifmRank, 1).WithHW(requiredHeight, requiredWidth);
    // Handle broadcast for elementwise operations
    if ( IsElementwise(opType) )
    {
        // The shape is converted to the pre-broadcasted shape by truncating against the ifmSlice shape.
        Shape ifmSliceShapeUpscaled = ifmSlice.shape.WithHW(ifmSlice.shape.Height() * upscaling, ifmSlice.shape.Width() * upscaling);
        stripeShape = Shape::Min(stripeShape, ifmSliceShapeUpscaled);
    }
    // Special-treatments for operators where depth cannot be inferred
    // use full IFM-depth.
    else if ( (IsConvolution(opType) && !IsDepthwise(opType)) || opType == OpType::ReduceSum )
    {
        stripeShape = stripeShape.WithDepth(ifmSlice.shape.Depth());
    }
    return stripeShape;
}

// Calculate the read-offset for the IFM-stripe relative to the IFM slice.
// If upscaling is used, this function returns the offset in the upscaled IFM.
// (offset needs to be adjusted for padding before it can be downscaled)
static Shape CalculateUpscaledIfmStripeOffset(const Shape &ofmStripeOffset, const Point2i &ofmStep,
    const Point2i &ifmStep, const Kernel *kernel, const TensorSlice &ifmSlice, OpType opType, int upscaling)
{
    const auto &padding = kernel->Padding();
    assert(ofmStripeOffset.Width() % ofmStep.x == 0);
    assert(ofmStripeOffset.Height() % ofmStep.y == 0);
    // Calculate the step-normalised offset in the OFM stripe
    // This represents how many rows/cols have already been written
    // (when accounting for ofmStep)
    int ofmX = ofmStripeOffset.Width() / ofmStep.x;
    int ofmY = ofmStripeOffset.Height() / ofmStep.y;
    // Calculate how much the kernel moves between each IFM-element
    // (accounting for IFM-step)
    Point2i effectiveKernelStride = kernel->Stride() * ifmStep;
    // Calculate how far we must have traveled in the IFM to produce ofmX/ofmY
    // Adjust coordinate based on left/top padding.
    int ifmStripeOffsetWidth = ofmX * effectiveKernelStride.x - padding.Left();
    int ifmStripeOffsetHeight = ofmY * effectiveKernelStride.y - padding.Top();
    // Truncate (or promote) to IFM-rank (but at least 4D)
    int ifmRank = std::max(4, ifmSlice.shape.Size());
    Shape stripeOffset = Shape(ofmStripeOffset, ifmRank, 0).WithHW(ifmStripeOffsetHeight, ifmStripeOffsetWidth);
    // Handle broadcast for elementwise operations
    if ( IsElementwise(opType) )
    {
        // The offset is wrapped (modulo) against the ifm slice shape.
        Shape ifmShapeUpscaled = ifmSlice.shape.WithHW(ifmSlice.shape.Height() * upscaling, ifmSlice.shape.Width() * upscaling);
        stripeOffset = Shape::Wrap(stripeOffset, ifmShapeUpscaled);
    }
    // Special-treatments for operators where depth cannot be inferred
    // use depth offset 0 to cover the whole IFM-slice
    else if ( (IsConvolution(opType) && !IsDepthwise(opType)) || opType == OpType::ReduceSum )
    {
        stripeOffset = stripeOffset.WithDepth(0);
    }
    return stripeOffset;
}

// CalculateIfmStripeAndPadding finds the IFM-area and
// padding required to produce the OFM-stripe.
//
// This is performed in 5 steps:
//     1. Offset (upscaled) calculations for the IFM stripe
//     2. Shape (upscaled) calculations for the IFM stripe
//     3. Padding calculations and offset/shape adjustment
//     4. Downscaling shape & offset (based on upscaling-factor)
//     5. Offset adjustments (based on IFM slice offset)
//
// "Offsets" in this function are defined relative to the tensor-slice.
// "Coordinates" are defined relative to the whole tensor.
// i.e. "offset" (0,0,0,0) represents the start of the tensor-slice
// and "coordinate" (0,0,0,0) represents the start of the whole tensor.
//
// The "offset" definition is useful when calculating IFM stripe and padding
// as the IFM offset can be inferred from the OFM offset (step 1), and padding can
// be inferred from IFM offset relative to the slice shape (step 3).
//
// The IFM offset is later converted into a true tensor coordinate before returning.
static Box CalculateIfmStripeAndPadding(const Shape &ofmStripeStartOffset, const Shape &ofmStripeEndOffset,
    const Point2i &ofmStep, const TensorSlice &ifmSlice, const Point2i &ifmStep, const Kernel *kernel, OpType opType,
    int upscaling, TransposeType transposeType, bool accIfm, HLCPadding &stripePadding)
{
    if ( accIfm || IsVectorProduct(opType) || opType == OpType::Resize )
    {
        // IFM shape cannot be inferred from the OFM shape for these operations.
        // use full ifm-slice.
        return Box(ifmSlice.offset, Box::Size(ifmSlice.shape));
    }
    // We infer the IFM stripe from the untransposed OFM stripe
    Shape untransposedOfmStart = ofmStripeStartOffset.Unpermute(uint32_t(transposeType));
    Shape untransposedOfmEnd = ofmStripeEndOffset.Unpermute(uint32_t(transposeType));
    Shape untransposedOfmShape = untransposedOfmEnd - untransposedOfmStart;
    assert(untransposedOfmShape.Elements() > 0 && "Unexpected OFM volume");

    // 1. Calculate IFM offset
    Shape stripeOffsetUpscaled = CalculateUpscaledIfmStripeOffset(untransposedOfmStart, ofmStep, ifmStep, kernel, ifmSlice, opType, upscaling);

    // 2. Calculate IFM shape
    Shape stripeShapeUpscaled = CalculateUpscaledIfmStripeShape(untransposedOfmShape, ofmStep, ifmStep, kernel, ifmSlice, opType, upscaling);

    // 3. Calculate stripe-padding
    // Upscale IFM shape to calculate padding
    // Upscaled space is required to perform padding calculations as padding is applied after upscaling in HW
    Shape ifmShapeUpscaled = Shape(ifmSlice.shape, 4, 1).WithHW(ifmSlice.shape.Height() * upscaling, ifmSlice.shape.Width() * upscaling);
    std::tie(stripeOffsetUpscaled, stripeShapeUpscaled, stripePadding) = CalculateStripePadding(stripeOffsetUpscaled,
        stripeShapeUpscaled, ifmShapeUpscaled, ifmStep, untransposedOfmShape, ofmStep, stripePadding, opType);

    // 4. Adjust to IFM coordinate-system by dividing with upscaling
    // TODO MLBEDSW-7003: Handle stripe-offsets that start on an upscaled coordinate
    assert(stripeOffsetUpscaled.Height() % upscaling == 0 && "stripe starts on an upscaled coordinate");
    assert(stripeOffsetUpscaled.Width() % upscaling == 0 && "stripe starts on an upscaled coordinate");
    Shape stripeOffset = stripeOffsetUpscaled.WithHW(stripeOffsetUpscaled.Height() / upscaling, stripeOffsetUpscaled.Width() / upscaling);
    // The downscaled ifm-shape is rounded up so that it produces (at least) stripeShapeUpscaled H/W
    // note: This cannot be done in step 2, as shape needs to be adjusted for padding before rounding is applied
    // (padding is not upscaled).
    Shape stripeShape = stripeShapeUpscaled.WithHW(
        DivRoundUp(stripeShapeUpscaled.Height(), upscaling), DivRoundUp(stripeShapeUpscaled.Width(), upscaling));
    assert(stripeShape.Elements() > 0);

    // 5. Convert to true IFM coordinate by adding the IFM slice-offset
    Shape stripeStart = stripeOffset + ifmSlice.offset;
    return Box(stripeStart, Box::Size(stripeShape));
}

// Generate a sequence of HLCStripe commands for a single scheduled operation.
// This walks the OFM slice in stripes (H, W, C), computes the corresponding IFM
// area for each input FM, derives padding, and optionally emits weight DMA
// commands when buffered weights need (re)loading.
// The end result is a list of HLCStripe commands that fully cover the OFM slice for this op.
void HLCStreamGenerator::GenerateHLCStripeCommands(SchedulerOperation *op, const std::shared_ptr<HLCOperation> &hlcOp, HLCStream &cmds)
{
    auto opInfo = _schedule->Cost(op);
    auto kernel = op->Kernel();
    assert(kernel != nullptr && "Operators must have a kernel");

    auto opType = op->Type();
    auto ofmConn = op->OFM();
    auto ifm0Conn = op->IFM(0);
    auto *ifm1Conn = op->TryIFM(1);
    auto opGroup = op->OpGroup();
    int upscaling = ifm0Conn->resamplingMode == ArchResampling::None ? 1 : 2;
    auto &depthSlices = opInfo->ofmDepthSlices;
    // Define Start offset and shape for the OFM
    auto ofmOffset = Shape::PadAxes(ofmConn->shape, 4, 0).WithZeros();
    auto ofmShape = Shape::PadAxes(ofmConn->shape, 4, 1);
    if ( ofmConn->slice )
    {
        ofmOffset = Shape::PadAxes(ofmConn->slice.offset, 4, 0);
        ofmShape = Shape::PadAxes(ofmConn->slice.shape, 4, 1);
    }
    assert(ofmOffset.Size() >= 4);
    assert(ofmShape.Size() >= 4);
    int ofmRank = ofmShape.Size();
    auto ofmStep = ofmConn->stepXY;
    const auto &stripeShape = opInfo->stripe;
    // round H/W steps to multiples of the ofm write-step (stepXY)
    int stripeHeightIncrement = RoundAway(stripeShape.Height(), ofmStep.y);
    int stripeWidthIncrement = RoundAway(stripeShape.Width(), ofmStep.x);
    // Slice OFM shape in stripes based on stripeShape.
    for ( int height = 0; height < ofmShape.Height(); height += stripeHeightIncrement )
    {
        int endHeight = std::min(height + stripeHeightIncrement, ofmShape.Height());
        for ( int width = 0; width < ofmShape.Width(); width += stripeWidthIncrement )
        {
            int endWidth = std::min(width + stripeWidthIncrement, ofmShape.Width());
            for ( int depthIndex = 0; depthIndex < int(depthSlices.size()) - 1; ++depthIndex )
            {
                // "Offsets" in this function are defined relative to the tensor-slice.
                // "Coordinates" are defined relative to the whole tensor.
                // i.e. "offset" (0,0,0,0) represents the start of the tensor-slice
                // and "coordinate" (0,0,0,0) represents the start of the whole tensor.
                //
                // The offset definition is useful when calculating IFM stripes (See CalculateIfmStripeAndPadding)
                // Offsets are later converted into true tensor coordinates before creating the stripe-areas.
                int depth = depthSlices[depthIndex];
                int nextDepth = depthSlices[depthIndex + 1];
                Shape ofmStripeStartOffset = Shape(0, height, width, depth);
                Shape ofmStripeEndOffset = Shape(1, endHeight, endWidth, nextDepth);
                // assert that stripe-offsets are inside the OFM slice shape.
                assert((Shape::Min(ofmStripeStartOffset, ofmShape) == ofmStripeStartOffset) && "OFM-start offset not inside OFM Shape");
                assert((Shape::Min(ofmStripeEndOffset, ofmShape) == ofmStripeEndOffset) && "OFM-end offset not inside OFM Shape");
                auto hlcStripe = std::make_unique<HLCStripe>(hlcOp);
                auto &primaryStripeArea = hlcStripe->stripeAreas.emplace_back();
                hlcStripe->opGroup = op->OpGroup();
                // Convert from offsets to true OFM coordinates
                // by converting to ofmRank and adding the OFM slice offset.
                Shape ofmStripeStart = Shape(ofmStripeStartOffset, ofmRank, 0) + ofmOffset;
                Shape ofmStripeEnd = Shape(ofmStripeEndOffset, ofmRank, 1) + ofmOffset;
                primaryStripeArea.ofmArea = Box(ofmStripeStart, ofmStripeEnd);
                assert(primaryStripeArea.ofmArea.Intersection(Box(ofmOffset, Box::Size(ofmShape))) == primaryStripeArea.ofmArea && "ofmBox not inside the OFM slice");
                // compute IFM stripes for every input
                for ( const auto &fm : hlcOp->ifm )
                {
                    if ( !IsIFM(fm.usage) ) continue;
                    auto ifmConn = op->Input(fm.usage);
                    // Create at least 4D ifm-slice
                    Shape sliceShape = Shape::PadAxes(ifmConn->SliceShape(), 4, 1);
                    Shape sliceOffset = Shape::PadAxes(ifmConn->slice.offset ? ifmConn->slice.offset : sliceShape.WithZeros(), 4, 0);
                    TensorSlice ifmSlice(sliceOffset, sliceShape);
                    bool accIfm = op->AccumulatorMode().source == AccumulatorSource::Ifm2 && fm.usage == TensorUsage::IFM1;
                    // Calculate the IFM stripe area and padding
                    Box ifmBox = CalculateIfmStripeAndPadding(ofmStripeStartOffset, ofmStripeEndOffset, ofmConn->stepXY,
                        ifmSlice, ifmConn->stepXY, kernel, opType, upscaling, ofmConn->transpose, accIfm, hlcStripe->padding);
                    assert(ifmBox.Intersection(Box(ifmSlice.offset, Box::Size(ifmSlice.shape))) == ifmBox && "ifmBox not inside the IFM slice");
                    primaryStripeArea.AddIfm(std::move(ifmBox));
                }
                if ( opInfo->npuWeightsTensor != nullptr )
                {
                    hlcStripe->weightRangeDepth = depth;  // Now a relative zero-offset OFM depth
                    if ( opInfo->bufferedWeightTensor.parts > 0 &&
                         (primaryStripeArea.ofmArea.Start().Height() == ofmOffset.Height() ||
                             opInfo->bufferedWeightTensor.buffering == Buffering::Double) )
                    {
                        unsigned bufferIndex = unsigned(depthIndex) % opInfo->bufferedWeightTensor.parts;
                        assert(bufferIndex < std::size(opInfo->bufferedWeightTensor.tensor));
                        // Metadata of new weights to put into the weight buffer tensor
                        auto newWeights = std::make_tuple(opInfo->npuWeightsTensor->equivalenceId, depth, depthIndex);
                        if ( _filledWeightBuffers.count(opInfo->bufferedWeightTensor.tensor[bufferIndex].get()) == 0 )
                        {
                            cmds.push_back(GenerateWeightDMA(opInfo->npuWeightsTensor.get(), opInfo->bufferedWeightTensor, depth, depthIndex));
                        }
                        else
                        {
                            auto &currentWeights = _filledWeightBuffers[opInfo->bufferedWeightTensor.tensor[bufferIndex].get()];
                            if ( currentWeights != newWeights )
                            {
                                cmds.push_back(GenerateWeightDMA(
                                    opInfo->npuWeightsTensor.get(), opInfo->bufferedWeightTensor, depth, depthIndex));
                            }
                        }
                        _filledWeightBuffers[opInfo->bufferedWeightTensor.tensor[bufferIndex].get()] = newWeights;
                    }
                }
                else if ( opInfo->npuScalesTensor != nullptr )
                {
                    hlcStripe->weightRangeDepth = depth;
                }
                else
                {
                    hlcStripe->weightRangeDepth = -1;
                }

                // Generate FM areas for sub-operations
                assert(hlcOp->subOps.size() == op->SubOps().size());
                for ( size_t i = 0; i < hlcOp->subOps.size(); ++i )
                {
                    const auto &subOp = op->SubOps()[i];
                    HLCSubOperation &hlcSubOp = hlcOp->subOps[i];

                    StripeArea hlcSubOpArea;
                    assert(!hlcStripe->stripeAreas.empty());
                    hlcSubOpArea.ofmArea = hlcStripe->stripeAreas[0].ofmArea;
                    if ( hlcSubOp.ofm.slice.IsValid() )
                    {
                        // If the sub-op produces a slice of the ofm, move the area accordingly
                        hlcSubOpArea.ofmArea.Move(Shape::PadAxes(hlcSubOp.ofm.slice.offset, hlcSubOpArea.ofmArea.Start().Size(), 0));
                    }
                    for ( const auto &subOpIfm : hlcSubOp.ifm )
                    {
                        if ( !IsIFM(subOpIfm.usage) ) continue;
                        if ( subOpIfm.address == -1 )
                        {
                            // Address -1 signifies that the FM is in an internal buffer
                            hlcSubOpArea.AddIfm(hlcStripe->stripeAreas[0].ofmArea);
                        }
                        else
                        {
                            // Calculate the external input area based on the output area
                            auto subOpIfmConn = subOp->Input(subOpIfm.usage);
                            auto subOpOfmConn = subOp->Output(TensorUsage::OFM);
                            auto subOpKernel = subOp->Kernel();
                            HLCPadding stripePadding = {0, 0, 0, 0};
                            Shape sliceShape = Shape::PadAxes(subOpIfmConn->SliceShape(), 4, 1);
                            Shape sliceOffset = Shape::PadAxes(
                                subOpIfmConn->slice.offset ? subOpIfmConn->slice.offset : sliceShape.WithZeros(), 4, 0);
                            TensorSlice ifmSlice(sliceOffset, sliceShape);
                            Box subOpIfmBox = CalculateIfmStripeAndPadding(ofmStripeStartOffset, ofmStripeEndOffset,
                                ofmConn->stepXY, ifmSlice, subOpIfmConn->stepXY, subOpKernel, subOp->Type(), 1,
                                subOpOfmConn->transpose, false, stripePadding);
                            assert(subOpIfmBox.Intersection(Box(ifmSlice.offset, Box::Size(ifmSlice.shape))) == subOpIfmBox && "subOpIfmBox not inside the IFM slice");
                            assert(stripePadding.top == 0 && stripePadding.bottom == 0 && stripePadding.left == 0 &&
                                   stripePadding.right == 0 && "subOp with non-zero stripe padding.");
                            hlcSubOpArea.AddIfm(subOpIfmBox);
                        }
                    }
                    hlcStripe->stripeAreas.push_back(std::move(hlcSubOpArea));
                }

                cmds.push_back(std::move(hlcStripe));
            }
        }
    }
}

void HLCStreamGenerator::GenerateCommands(
    SchedulerOperation *op, const std::shared_ptr<HLCOperation> &hlcOp, SubGraphs &subgraphs, HLCStream &cmds)
{
    auto opType = op->Type();

    if ( IsControlFlow(opType) )
    {
        GenerateHLCBranchCommands(op, hlcOp, subgraphs, cmds);
    }
    else if ( opType == OpType::Scatter || opType == OpType::Gather )
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
            // TODO MLBEDSW-9143: support cascading of chains
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
    SubGraphs subgraps;
    for ( int i = 0; i < nrOps; ++i )
    {
        HLCStream stream;
        GenerateCommands(cascadedOps[i].get(), hlcOps[i], subgraps, stream);
        assert(subgraps.empty());
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
        assert(!nextStripe[opIndex]->stripeAreas.empty());
        if ( opIndex == 0 ||
             nextStripe[opIndex]
                 ->stripeAreas[0]
                 .ifmAreas.at(cascadedOps[opIndex]->PrimaryIfmIndex())
                 .End()
                 .IsSubShapeOf(availableStripe[opIndex - 1]->stripeAreas[0].ofmArea.End()) )
        {
            auto &stream = cmdsForOps[opIndex];
            assert(ix < int(stream.size()));
            HighLevelCommand *hlc = stream[ix].get();
            cmds.push_back(std::move(cmdsForOps[opIndex][ix]));
            ++ix;
            if ( hlc->CommandType() == HighLevelCommandType::STRIPE )
            {
                availableStripe[opIndex] = nextStripe[opIndex];
                nextStripe[opIndex] = FindNextStripe(stream, ix);
                if ( nextStripe[opIndex] == nullptr )
                {
                    if ( opIndex >= nrOps - 1 )
                    {
                        // Finished
                        break;
                    }
                    ++opIndex;
                    continue;
                }
                // Move to next operation once the rolling buffer is full
                if ( opIndex < nrOps - 1 && nextStripe[opIndex + 1] != nullptr &&
                     !CanFitRollingBuffer(cascadeInfo, cascadedOps, availableStripe, nextStripe, opIndex) )
                {
                    // Validate that the available data in the rolling-buffer fulfills
                    // The next stripes IFM area
                    assert(
                        nextStripe[opIndex + 1]
                            ->stripeAreas[0]
                            .ifmAreas.at(cascadedOps[opIndex + 1]->PrimaryIfmIndex())
                            .End()
                            .IsSubShapeOf(availableStripe[opIndex]->stripeAreas[0].ofmArea.End()) &&
                        "Full rolling buffer is insufficient to cover next stripe");
                    ++opIndex;
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

HLCStream HLCStreamGenerator::GenerateCommandStream(const NPUOperation *npuOp, const Schedule *schedule, SubGraphs &subgraphs)
{
    HLCStream cmds;
    _schedule = schedule;
    const auto &npuOps = npuOp->Operations();
    // Create HLCOperation for every ScheduledOperation
    std::vector<std::shared_ptr<HLCOperation>> hlcOps;
    for ( const auto &schedOp : npuOps )
    {
        auto op = schedOp.get();
        hlcOps.push_back(MakeOperation(op, schedule->Cost(op), _base, _addresses));
    }

    // Compiled subgraphs
    std::vector<HLCStream> substreams;

    // Generate the command stream
    const int sz = int(npuOps.size());
    for ( int i = 0; i < sz; ++i )
    {
        auto op = npuOps[i].get();
        auto opInfo = schedule->Cost(op);
        assert(opInfo != nullptr);
        auto &hlcOp = hlcOps[i];
        if ( opInfo->cascade == 0 )
        {
            // Single operation, not in cascade
            GenerateCommands(op, hlcOp, subgraphs, cmds);
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
    if ( _verbose )
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
        auto ofm = op->OFM();
        auto hlcOfm = hlcOp->ofm;
        if ( !op->SubOps().empty() )
        {
            // Print OFM of final sub operation in a group
            ofm = op->SubOps().back()->OFM();
            hlcOfm = hlcOp->subOps.back().ofm;
        }
        LOG_PRINT("  OFM: {}, {}\n", ofm->tensor->Name(), hlcOfm.ToString());
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
