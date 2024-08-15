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

#include "scheduler_packing.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "common/shape.hpp"
#include "graph.hpp"
#include "operation.hpp"
#include "scheduler_decompose.hpp"
#include "scheduler_operation.hpp"
#include "tensor.hpp"

#include <vector>

namespace regor
{

namespace
{

// Returns true if no IFMs, or if all IFMs are graph inputs
bool AllInputsAreGraphInputs(const SchedulerOperation &op)
{
    for ( const auto &schedConn : op.inputs )
    {
        if ( !schedConn.tensor->IsConstant() && !schedConn.tensor->isGraphInput ) return false;
    }
    return true;
}

// Returns true if no OFMs, or if all OFMs are graph outputs
bool AllOutputsAreGraphOutputs(const SchedulerOperation &op)
{
    for ( const auto &outputSchedConn : op.outputs )
    {
        if ( !outputSchedConn.tensor->isGraphOutput ) return false;
    }
    return true;
}

// Returns true if any of first's OFMs are same as second's IFMs
bool IsConnected(const SchedulerOperation &first, const SchedulerOperation &second)
{
    for ( const auto &firstOutputSchedConn : first.outputs )
    {
        for ( const auto &secondInputSchedConn : second.inputs )
        {
            if ( firstOutputSchedConn.tensor == secondInputSchedConn.tensor ) return true;
        }
    }
    return false;
}

}  // namespace

SchedulerPacking::SchedulerPacking(Architecture *arch, bool disableChaining) :
        _arch(arch), _disableChaining(disableChaining)
{
}

std::vector<std::unique_ptr<SchedulerOperation>> SchedulerPacking::Process(const Graph *graph)
{
    // Get operation list in execution order
    std::vector<Operation *> executionList;
    Graph::TraverseGraphFromEnd(graph->Outputs(),
        [&](Operation *op) -> bool
        {
            executionList.push_back(op);
            return true;
        });

    FilterOperations(executionList, graph);

    PackOperations();

    ReorderOperations();

    return std::move(_schedList);
}

void SchedulerPacking::FilterOperations(const std::vector<Operation *> &executionList, const Graph *graph)
{
    // Convert linear Graph Operations to a list of Scheduler Operations
    for ( Operation *op : executionList )
    {
        auto schedOp = MakeSchedulerOperation(op, graph);
        if ( NeedsDecompose(_arch, schedOp.get()) )
        {
            auto schedOps = DecomposeSchedulerOperation(std::move(schedOp));
            _schedList.insert(
                _schedList.end(), std::make_move_iterator(schedOps.begin()), std::make_move_iterator(schedOps.end()));
        }
        else
        {
            _schedList.push_back(std::move(schedOp));
        }
    }
}

ArchitectureOpGroupQuery SchedulerPacking::CreateOpGroupQuery(const SchedulerOperation *schedOp) const
{
    ArchitectureOpGroupQuery query{};
    query.type = schedOp->Type();
    query.inputs = schedOp->TryIFM(1) ? 2 : 1;
    query.kernel = schedOp->Kernel();

    auto ifm0 = schedOp->IFM(0);
    auto ifm1 = schedOp->TryIFM(1);
    auto ofm = schedOp->OFM();
    query.ifm[0].key = ifm0->tensor->uid;
    query.ifm[0].type = ifm0->tensor->dataType;
    query.ifm[0].shape = ifm0->shape;
    query.ifm[0].isReordered = (!IsNone(ifm0->transpose)) || (ifm0->reverse != ReverseType::None);
    query.ifm[0].isConst = ifm0->tensor->IsConstant();
    if ( ifm1 )
    {
        query.ifm[1].key = ifm1->tensor->uid;
        query.ifm[1].type = ifm1->tensor->dataType;
        query.ifm[1].shape = ifm1->shape;
        query.ifm[1].isReordered = (!IsNone(ifm1->transpose)) || (ifm1->reverse != ReverseType::None);
        query.ifm[1].isConst = ifm1->tensor->IsConstant();
    }
    query.ofm.key = ofm->tensor->uid;
    query.ofm.type = ofm->tensor->dataType;
    query.ofm.shape = ofm->shape;
    query.ofm.isReordered = (!IsNone(ofm->transpose)) || (ofm->reverse != ReverseType::None);
    query.ofm.isConst = false;
    return query;
}

void SchedulerPacking::SchedulerPacking::PackOperations()
{
    LOG_TRACE1("Scheduler Packing (of {0} Ops)\n", _schedList.size());

    auto cur = _schedList.begin();
    auto write = cur;

    while ( cur != _schedList.end() )
    {
        SchedulerOperation *primaryOp = cur->get();

        // Compact the list as we go
        if ( std::distance(write, cur) > 0 )
        {
            *write = std::move(*cur);
        }
        primaryOp->_index = int(std::distance(_schedList.begin(), write));

        cur++;

        LOG_TRACE1("Creating new group with {}\n", OpTypeToString(primaryOp->Type()));

        auto op0 = CreateOpGroupQuery(primaryOp);

        // Try to create OpGroup
        auto group = _arch->CreateOpGroup(op0);

        // OpGroup is nullptr if op can't run on NPU
        if ( group )
        {
            primaryOp->SetNpuOp(true);

            // First op in group has key 0
            int prevOpKey = 0;
            primaryOp->SetOpGroupKey(prevOpKey);
            LOG_TRACE1("Created new group with {} (key {})\n", OpTypeToString(primaryOp->Type()), prevOpKey);

            // Root SchedulerOperation takes ownership of the ArchitectureOpGroup here
            primaryOp->SetOpGroup(std::move(group));

            // Pack any future ops that will fit
            auto prevOp = primaryOp;

            // Try chaining subsequent ops into the primary
            while ( cur != _schedList.end() )
            {
                SchedulerOperation *nextOp = cur->get();
                assert(nextOp);  // Empty op may be possible if we seek ahead

                int key = CanPack(primaryOp, prevOp, nextOp, prevOpKey);
                if ( !key )
                {
                    LOG_TRACE1("Can't add next op\n");
                    break;
                }
                nextOp->SetNpuOp(true);
                nextOp->SetParent(primaryOp);
                nextOp->SetOpGroupKey(key);

                LOG_TRACE1("Added {} (key {}) to {} (key {})\n", OpTypeToString(nextOp->Type()), key,
                    OpTypeToString(prevOp->Type()), prevOpKey);

                if ( IsActivation(nextOp->Type()) )
                {
                    primaryOp->AddSubOp(std::move(*cur));
                    // Replace primary op's OFM by nextOp's OFM
                    auto *ofmConn = prevOp->OFM();
                    ofmConn->tensor = nextOp->OFM()->tensor;
                    ofmConn->quantization = prevOp->Output(TensorUsage::OFM)->quantization;
                    ofmConn->quantization.quantMin = nextOp->Output(TensorUsage::OFM)->quantization.quantMin;
                    ofmConn->quantization.quantMax = nextOp->Output(TensorUsage::OFM)->quantization.quantMax;
                }
                else
                {
                    primaryOp->AddSubOp(std::move(*cur));
                }
                prevOpKey = key;
                prevOp = nextOp;
                cur++;
            }
            LOG_TRACE1("\t{0}: {1} - OFM [{2}] <- (IFM0 [{3}], IFM1 [{4}], Primary={5})\n", primaryOp->Index(),
                OpTypeToString(primaryOp->Type()), primaryOp->OFM()->shape.ToString(), primaryOp->IFM(0)->shape.ToString(),
                primaryOp->TryIFM(1) ? primaryOp->IFM(1)->shape.ToString() : "", primaryOp->PrimaryIfmIndex());
        }
        write++;
    }

    // Shorten list to contain only those operators written
    _schedList.erase(write, cur);
}

// Reorder CPU ops so that there are fewer groups of consecutive CPU ops in the list of ops
void SchedulerPacking::ReorderOperations()
{
    // Graphs with both CPU and NPU ops might not have an optimal order in the ops list due to how the graph is
    // traversed (depth first search). This can result in more context switching between CPU and NPU. Try to optimise
    // this by moving/grouping CPU ops where that is possible. Criteria for CPU pass to be moved:
    //
    // 1) CPU passes that only consumes graph input tensors can be moved to the top of the list.
    //
    // 2) CPU passes that only produces graph output tensors can be moved to the bottom of the list.
    //
    // 3) A CPU pass X is allowed to be grouped together with CPU pass Y if there is no NPU pass between pass X and pass
    //    Y that depends on output from pass X. Criteria 3 will try to move as many CPU passes towards the bottom of the
    //    list.

    // Ops with only graph input IFMs
    std::vector<std::unique_ptr<SchedulerOperation>> earlyOps;

    // Ops with only graph output OFMs
    std::vector<std::unique_ptr<SchedulerOperation>> lateOps;

    // Ops not in the above two lists
    std::vector<std::unique_ptr<SchedulerOperation>> otherOps;

    // Reserving space since most ops are likely to end up here
    otherOps.reserve(_schedList.size());

    // Iterate in execution order to find CPU ops with only graph input IFMs or only graph output OFMs
    for ( auto i = _schedList.begin(); i != _schedList.end(); ++i )
    {
        std::unique_ptr<SchedulerOperation> &op = *i;

        if ( !op->IsNpuOp() && AllInputsAreGraphInputs(*op) )
        {
            earlyOps.push_back(std::move(*i));
        }
        else if ( !op->IsNpuOp() && AllOutputsAreGraphOutputs(*op) )
        {
            lateOps.push_back(std::move(*i));
        }
        else
        {
            otherOps.push_back(std::move(*i));
        }
    }

    // Iterate in reverse execution order to find CPU ops
    for ( auto i = otherOps.rbegin(); i != otherOps.rend(); ++i )
    {
        std::unique_ptr<SchedulerOperation> &op = *i;

        // We're looking for CPU ops
        if ( op->IsNpuOp() ) continue;

        // Iterate in execution order from the CPU op's position
        for ( auto j = i; j != otherOps.rbegin(); --j )
        {
            std::unique_ptr<SchedulerOperation> &op0 = *(j - 0);  // Earlier in execution order
            std::unique_ptr<SchedulerOperation> &op1 = *(j - 1);  // Later in execution order
            assert(!op0->IsNpuOp());

            // Don't move past another CPU op
            if ( !op1->IsNpuOp() ) break;

            // If our CPU op and the op after are connected, we can't move it down
            if ( IsConnected(*op0, *op1) ) break;

            // Move our CPU op one step later in execution order
            std::iter_swap(j, j - 1);
        }
    }

    // Reassemble the list
    _schedList.clear();
    _schedList.reserve(earlyOps.size() + otherOps.size() + lateOps.size());
    _schedList.insert(_schedList.end(), std::make_move_iterator(earlyOps.begin()), std::make_move_iterator(earlyOps.end()));
    _schedList.insert(_schedList.end(), std::make_move_iterator(otherOps.begin()), std::make_move_iterator(otherOps.end()));
    _schedList.insert(_schedList.end(), std::make_move_iterator(lateOps.begin()), std::make_move_iterator(lateOps.end()));

    // Recalculate the op index now when the list may have a different order
    for ( auto i = _schedList.begin(); i != _schedList.end(); ++i )
    {
        (*i)->_index = int(std::distance(_schedList.begin(), i));
    }
}

int SchedulerPacking::CanPack(const SchedulerOperation *schedOp, const SchedulerOperation *prevOp,
    const SchedulerOperation *nextOp, const int prevOpKey) const
{
    const auto prevConnOfm = prevOp->OFM();
    const auto nextConnIfm = nextOp->IFM(0);
    const auto nextConnIfm2 = nextOp->TryIFM(1);
    const auto nextConnOfm = nextOp->OFM();

    SchedulerTensor *prevOFM = prevConnOfm->tensor.get();
    SchedulerTensor *ifmTensor = nextConnIfm->tensor.get();
    SchedulerTensor *ifm2Tensor = nextConnIfm2 ? nextConnIfm2->tensor.get() : nullptr;
    assert(prevOFM && "primary/prev op must have OFM");
    assert(ifmTensor && "next op must have IFM");

    // Previous op in execution order doesn't connect to this one
    if ( prevOFM != ifmTensor && prevOFM != ifm2Tensor )
    {
        return 0;
    }

    if ( _disableChaining && !IsActivation(nextOp->Type()) )
    {
        return 0;
    }

    // Highly unlikely constant tensor between ops
    assert(!prevOFM->srcTensor->IsConstant() && "Unexpected constant tensor between ops");

    // Only pack tensors on single-reader/writer paths (i.e. can't pack across concat/split)
    if ( prevOFM->producers.size() != 1 || prevOFM->consumers.size() != 1 )
    {
        return 0;
    }

    if ( schedOp->OFM()->tensor->isGraphOutput )
    {
        return 0;
    }

    if ( IsActivation(nextOp->Type()) && !nextConnIfm->quantization.EqualScales(nextConnOfm->quantization) )
    {
        // Can not fuse activation with different scales
        return 0;
    }

    auto op1 = CreateOpGroupQuery(nextOp);
    return schedOp->_opGroup->Add(op1, {prevOpKey});
}

void SchedulerPacking::InitSchedulerConnection(
    SchedulerConnection *schedConn, const std::shared_ptr<SchedulerTensor> &tensor, const TensorConnection &conn)
{
    schedConn->tensor = tensor;
    schedConn->slice = {Shape::PadAxes(conn.slice.offset, 4, 0), Shape::PadAxes(conn.slice.shape, 4, 1)};
    schedConn->shape = Shape::PadAxes(conn.shape, 3, 1);  // Scheduler needs minimum HWC axes to stripe
    schedConn->quantization = conn.quantization;
    schedConn->transpose = conn.transpose;
    schedConn->reverse = conn.reverse;
}

void SchedulerPacking::InitSchedulerTensor(SchedulerTensor *schedTensor, Tensor *tensor, const Graph *graph)
{
    // Take scheduler-local copies of graph tensor parameters.
    schedTensor->format = TensorFormat::NHWC;
    schedTensor->memArea = tensor->IsConstant() ? _arch->ReadonlyMemory() : _arch->FeatureMapMemory();
    schedTensor->storageShape = Shape::PadAxes(tensor->StorageShape(), 4, 1);
    schedTensor->dataType = tensor->Type();
    schedTensor->bufferView = tensor->View();
    schedTensor->isGraphInput = graph->IsInput(tensor);
    schedTensor->isGraphOutput = graph->IsOutput(tensor);
    schedTensor->uid = tensor->Uid();
}

std::unique_ptr<SchedulerOperation> SchedulerPacking::MakeSchedulerOperation(Operation *op, const Graph *graph)
{
    assert(op->Type() != OpType::None);

    std::unique_ptr<SchedulerOperation> schedOp = std::make_unique<SchedulerOperation>(op->Type());

    schedOp->SetKernel(op->Kernel());
    schedOp->SetHasScaling(op->HasScaling());
    schedOp->SetRounding(op->Rounding());
    schedOp->SetAttributeRef(op->AttributeRef());
    schedOp->_srcKey = op;

    // Get the inputs from the source op and connect with scheduler specific tensor
    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &item : list->pairs() )
        {
            Tensor *tensor = item.second.tensor.get();

            // Get/update scheduler's metadata for the graph tensor.
            auto pos = _tensorMap.find(tensor);
            if ( pos == _tensorMap.end() )
            {
                // Create new scheduler tensor if metadata is missing.
                auto tmp = std::make_shared<SchedulerTensor>();
                pos = _tensorMap.emplace(tensor, tmp).first;
                tmp->srcTensor = item.second.tensor;
                InitSchedulerTensor(tmp.get(), tensor, graph);
            }

            // Update consumers and manage connectivity
            const std::shared_ptr<SchedulerTensor> &schedTensor = pos->second;

            if ( IsOFM(item.first) )
            {
                schedTensor->producers.push_back(schedOp.get());
            }
            else
            {
                schedTensor->consumers.push_back(schedOp.get());
            }
            SchedulerConnection *schedConn = IsOFM(item.first) ? schedOp->AddOutput(item.first) : schedOp->AddInput(item.first);
            InitSchedulerConnection(schedConn, schedTensor, item.second);
            schedConn->resamplingMode = ResamplingMode(item.first, schedOp->Type());
        }
    }

    // Examine elementwise and set a primary path for cascading.
    if ( IsBinaryElementwise(op->Type()) )
    {
        auto ifm0 = op->Input(TensorUsage::IFM0);
        auto ifm1 = op->Input(TensorUsage::IFM1);
        auto ofm = op->Output(TensorUsage::OFM);
        assert(ifm0 && "Binary elementwise op must have IFM0");
        assert(ifm1 && "Binary elementwise op must have IFM1");
        assert(ofm && "Binary elementwise op must have OFM");
        assert(ifm0->shape.Size() > 0 && "IFM0 must have dimension");
        assert(ifm1->shape.Size() > 0 && "IFM1 must have dimension");
        // Choose the non-const IFM path for binary operations that have
        // a constant input on the first IFM
        if ( ifm0->tensor->IsConstant() && !ifm1->tensor->IsConstant() )
        {
            schedOp->SetPrimaryIfmIndex(1);
        }
        // Favour the non-broadcast shape for cascading.
        else if ( (ifm0->shape != ofm->shape) && (ifm1->shape == ofm->shape) )
        {
            schedOp->SetPrimaryIfmIndex(1);
        }
    }
    return schedOp;
}

std::vector<std::unique_ptr<SchedulerOperation>> SchedulerPacking::DecomposeSchedulerOperation(std::unique_ptr<SchedulerOperation> op)
{
    std::vector<std::unique_ptr<SchedulerOperation>> result;
    switch ( op->Type() )
    {
        case OpType::Conv2D:
            result = DecomposeConv2D(_arch, std::move(op));
            break;
        case OpType::DepthwiseConv2DBias:
            result = DecomposeDepthwiseConv2D(_arch, std::move(op));
            break;
        case OpType::TransposeConv2D:
            result = DecomposeTransposeConv2D(_arch, std::move(op));
            break;
        default:
            assert(false);
            break;
    }
    return result;
}

ArchResampling SchedulerPacking::ResamplingMode(TensorUsage usage, OpType opType) const
{
    UNUSED(usage);
    UNUSED(opType);
    return ArchResampling::None;
}

}  // namespace regor
