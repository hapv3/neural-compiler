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

#include "scheduler_packing.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "common/shape.hpp"
#include "graph.hpp"
#include "operation.hpp"
#include "scheduler_decompose.hpp"
#include "scheduler_operation.hpp"
#include "shape_util.hpp"
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

// Returns true if no IFMs, or if all producers of all IFMs are in the list
bool AllInputsProducersAreEarlyOps(const SchedulerOperation &op, const std::vector<std::unique_ptr<SchedulerOperation>> &earlyOps)
{
    for ( const auto &schedConn : op.inputs )
    {
        for ( const auto &producer : schedConn.tensor->producers )
        {
            bool producerIsEarlyOp = false;
            for ( const auto &earlyOp : earlyOps )
            {
                if ( producer == earlyOp.get() )
                {
                    producerIsEarlyOp = true;
                    break;
                }
            }
            if ( !producerIsEarlyOp ) return false;
        }
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
    // This function assumes that first is a CPU op and second is an NPU op
    assert(!first.IsNpuOp());
    assert(second.IsNpuOp());

    for ( const auto &firstOutputSchedConn : first.outputs )
    {
        for ( const auto &secondInputSchedConn : second.inputs )
        {
            if ( firstOutputSchedConn.tensor == secondInputSchedConn.tensor ) return true;
        }
    }

    for ( const auto &secondSubOp : second.SubOps() )
    {
        if ( IsConnected(first, *secondSubOp) ) return true;
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
    _graph = graph;

    // Get operation list in execution order
    std::vector<Operation *> executionList;
    Graph::TraverseGraphFromEnd(graph->Outputs(), !graph->Persistent().empty(),
        [&](Operation *op) -> bool
        {
            executionList.push_back(op);
            return true;
        });

    ConvertOperations(executionList);

    PrePackOperations();

    PackOperations();

    ReorderOperations();

    _graph = nullptr;
    return std::move(_schedList);
}

void SchedulerPacking::ConvertOperation(const Operation *op, std::vector<std::unique_ptr<SchedulerOperation>> &result)
{
    result.clear();

    std::unique_ptr<SchedulerOperation> schedOp = MakeSchedulerOperation(op);
    if ( !schedOp )
    {
        return;
    }

    // Apply architecture related requirements
    ArchRequirements req;
    if ( OperatorQuery(_arch, schedOp.get(), &req).Any(QueryResult::HasRequirements) )
    {
        // Operator has a list of tensor requirements
        if ( req.req.Any(ArchRequirement::Tensor) )
        {
            const ArchTensorRequirement *tr = &req.tensor;
            do
            {
                SchedulerConnection *conn = schedOp->TryInput(tr->usage);
                conn = conn ? conn : schedOp->TryOutput(tr->usage);
                if ( conn )
                {
                    if ( tr->shape )
                    {
                        conn->tensor->storageShape = conn->shape = conn->slice.shape = tr->shape;
                    }
                    if ( tr->format != TensorFormat::Unknown )
                    {
                        conn->tensor->format = tr->format;
                    }
                    if ( tr->type != DataType::None )
                    {
                        conn->tensor->dataType = tr->type;
                    }
                }
                else
                {
                    if ( (tr->usage == TensorUsage::Scratch) && tr->shape )
                    {
                        auto scratchTensor = std::make_shared<SchedulerTensor>(tr->type, tr->shape, tr->format);
                        SchedulerConnection *scratchConn = schedOp->ConnectInput(TensorUsage::Scratch0, scratchTensor);
                        scratchConn->shape = tr->shape;
                        scratchTensor->memArea = _arch->FeatureMapMemory();
                    }
                }
                tr = tr->next;
            } while ( tr );
        }
    }

    result.push_back(std::move(schedOp));
}

void SchedulerPacking::ConvertOperations(const std::vector<Operation *> &executionList)
{
    std::vector<std::unique_ptr<SchedulerOperation>> converted(4);

    // Convert linear Graph Operations to a list of Scheduler Operations
    for ( Operation *op : executionList )
    {
        ConvertOperation(op, converted);

        for ( auto &schedOp : converted )
        {
            if ( ShouldDecompose(_arch, schedOp.get()) )
            {
                auto schedOps = DecomposeSchedulerOperation(std::move(schedOp));
                _schedList.insert(_schedList.end(), std::make_move_iterator(schedOps.begin()),
                    std::make_move_iterator(schedOps.end()));
            }
            else
            {
                _schedList.push_back(std::move(schedOp));
            }
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
    query.ifm[0].type = ifm0->Type();
    query.ifm[0].shape = ifm0->SliceShape();
    query.ifm[0].transpose = ifm0->transpose;
    query.ifm[0].reverse = ifm0->reverse;
    query.ifm[0].isConst = ifm0->tensor->IsConstant();
    query.ifm[0].isSliced = !Shape::IsReducedEqual(ifm0->SliceShape(), ifm0->shape);
    if ( ifm1 )
    {
        query.ifm[1].key = ifm1->tensor->uid;
        query.ifm[1].type = ifm1->Type();
        query.ifm[1].shape = ifm1->SliceShape();
        query.ifm[1].transpose = ifm1->transpose;
        query.ifm[1].reverse = ifm1->reverse;
        query.ifm[1].isConst = ifm1->tensor->IsConstant();
        query.ifm[1].isSliced = !Shape::IsReducedEqual(ifm1->SliceShape(), ifm1->shape);
    }
    query.ofm.key = ofm->tensor->uid;
    query.ofm.type = ofm->Type();
    query.ofm.shape = ofm->SliceShape();
    query.ofm.transpose = ofm->transpose;
    query.ofm.reverse = ofm->reverse;
    query.ofm.isConst = false;
    query.ofm.isSliced = !Shape::IsReducedEqual(ofm->SliceShape(), ofm->shape);
    return query;
}

// We handle reinterpret by catching it before we create a SchedulerOperation.
// Mapping is modified so that the OFM GraphIR tensor of the preceding OP and
// the GraphIR IFM tensor of the succeeding OP map to the same SchedulerTensor.
void SchedulerPacking::HandleReinterpretCast(const Operation *op)
{
    assert(op->Type() == OpType::ReinterpretCast && "Op Type is not ReinterpretCast.");

    const auto ifmConn = op->Input(TensorUsage::IFM);
    const auto ofmConn = op->Output(TensorUsage::OFM);

    // Try finding the SchedulerTensor mapped to the ReinterpretCast OP's IFM tensor.
    // If no preceding OP has created it, and it can't be found, create it.
    auto pos = _tensorMap.find(ifmConn->tensor.get());
    std::shared_ptr<SchedulerTensor> schedTensor;
    if ( pos == _tensorMap.end() )
    {
        schedTensor = std::make_shared<SchedulerTensor>();
        schedTensor->srcTensor = ifmConn->tensor;
        InitSchedulerTensor(schedTensor.get(), ifmConn->tensor.get());
        _tensorMap.emplace(ifmConn->tensor.get(), schedTensor);
    }
    else
    {
        schedTensor = pos->second;
    }
    // Ensure that both the ReinterpretCast IFM and OFM tensor maps to the same SchedulerTensor.
    _tensorMap.emplace(ofmConn->tensor.get(), schedTensor);

    // If reinterpret cast is the last OP, that means that it's output tensor is the output tensor of the network.
    // We therefore set isGraphOutput to true and make sure the srcTensor maps to the graph output tensor.
    if ( _graph->IsOutput(ofmConn->tensor.get()) )
    {
        InitSchedulerTensor(schedTensor.get(), ofmConn->tensor.get());
        schedTensor->dataType = ifmConn->tensor->Type();
        schedTensor->srcTensor = ofmConn->tensor;
    }
}

void SchedulerPacking::PrePackOperations()
{
    for ( auto &schedOp : _schedList )
    {
        ArchRequirements oReq{};
        Flags<QueryResult> result = OperatorQuery(_arch, schedOp.get(), &oReq);
        assert(result.Any(QueryResult::Constrained) == false && "Constrained result from complete OperatorQuery");
        if ( schedOp->Type() == OpType::Passthrough )
        {
            schedOp->SetNpuOp(false);
        }
        else if ( result.Any(QueryResult::Native) )
        {
            // TODO MLBEDSW-10643: This should be a direct-check against QueryResult::Native
            // HasRequirements at this point should result in failure
            if ( result.Any(QueryResult::HasRequirements) && oReq.req.Any(ArchRequirement::Decompose) )
            {
                throw std::runtime_error("Non-passthrough operation could not run on NPU.");
            }
            else
            {
                schedOp->SetNpuOp(true);
            }
        }
        else
        {
            throw std::runtime_error("Non-passthrough operation could not run on NPU.");
        }

        // Examine elementwise and set a primary path for cascading.
        if ( IsBinaryElementwise(schedOp->Type()) )
        {
            auto ifm0 = schedOp->Input(TensorUsage::IFM0);
            auto ifm1 = schedOp->Input(TensorUsage::IFM1);
            auto ofm = schedOp->Output(TensorUsage::OFM);
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
    }
}

void SchedulerPacking::PackOperations()
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

        // OpGroup is nullptr if op can't run on NPU
        if ( primaryOp->IsNpuOp() )
        {
            LOG_TRACE1("Creating new group with {}\n", OpTypeToString(primaryOp->Type()));
            auto op0 = CreateOpGroupQuery(primaryOp);
            // Try to create OpGroup
            auto group = _arch->CreateOpGroup(op0);
            assert(group);

            // First op in group has key 0
            int prevOpKey = 0;
            primaryOp->SetOpGroupKey(prevOpKey);
            LOG_TRACE1("Created new group with {} (key {})\n", OpTypeToString(primaryOp->Type()), prevOpKey);

            // Root SchedulerOperation takes ownership of the ArchitectureOpGroup here
            primaryOp->SetOpGroup(std::move(group));

            // Pack any future ops that will fit
            auto prevOp = primaryOp;
            auto lastNonFusedOp = primaryOp;

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
                nextOp->SetParent(primaryOp);
                nextOp->SetOpGroupKey(key);

                LOG_TRACE1("Added {} (key {}) to {} (key {})\n", OpTypeToString(nextOp->Type()), key,
                    OpTypeToString(prevOp->Type()), prevOpKey);

                // Replace the last non-fused op's OFM by nextOp's OFM if nextOp is fused. Also adjust producers and
                // consumers of the tensors involved like below.
                //
                // Before: [T1] -> PRIMARY -> [T2] -> ACTIVATION -> [T3]
                // T1 producers = ?
                // T1 consumers = {PRIMARY}
                // T2 producers = {PRIMARY}
                // T2 consumers = {ACTIVATION}
                // T3 producers = {ACTIVATION}
                // T3 consumers = ?
                //
                // After: [T1] -> PRIMARY (with ACTIVATION as subop) -> [T3]
                // T1 producers = ?
                // T1 consumers = {PRIMARY}
                // T2 producers = {}
                // T2 consumers = {ACTIVATION}
                // T3 producers = {PRIMARY}
                // T3 consumers = ?
                //
                // After fuse T2 still exists, but only as an unused input to ACTIVATION.

                if ( IsActivation(nextOp->Type()) )
                {
                    auto *ofmConn = lastNonFusedOp->OFM();
                    ofmConn->tensor->RemoveWriter(lastNonFusedOp);  // Remove PRIMARY from T2 producers
                    ofmConn->tensor = nextOp->OFM()->tensor;
                    ofmConn->tensor->producers.push_back(lastNonFusedOp);  // Add PRIMARY to T3 producers
                    ofmConn->tensor->RemoveWriter(nextOp);                 // Remove ACTIVATION from T3 producers
                    ofmConn->SetType(nextOp->OFM()->Type());

                    auto quantMin = ofmConn->quantization.quantMin;
                    auto quantMax = ofmConn->quantization.quantMax;
                    // Clamping is defined as  disabled when quantMin[0] and quantmax[0] are at the int64 limits
                    bool clampingDisabled =
                        !quantMin.empty() && quantMin[0] == std::numeric_limits<int64_t>::min() && !quantMax.empty() &&
                        quantMax[0] == std::numeric_limits<int64_t>::max();
                    // If clamping has been explicitly unset this if-statement makes sure we don't re-enable it.
                    if ( !clampingDisabled )
                    {
                        ofmConn->quantization.quantMin = nextOp->Output(TensorUsage::OFM)->quantization.quantMin;
                        ofmConn->quantization.quantMax = nextOp->Output(TensorUsage::OFM)->quantization.quantMax;
                    }
                }
                else if ( nextOp->Type() == OpType::Transpose )
                {
                    auto *ofmConn = lastNonFusedOp->OFM();
                    ofmConn->tensor->RemoveWriter(lastNonFusedOp);
                    ofmConn->tensor = nextOp->OFM()->tensor;
                    ofmConn->tensor->producers.push_back(lastNonFusedOp);
                    ofmConn->tensor->RemoveWriter(nextOp);
                    ofmConn->SetType(nextOp->OFM()->Type());
                    ofmConn->shape = nextOp->OFM()->shape;
                    ofmConn->slice = nextOp->OFM()->slice;
                    ofmConn->transpose = nextOp->OFM()->transpose;
                }
                else if ( nextOp->Type() == OpType::Reverse )
                {
                    auto *ofmConn = lastNonFusedOp->OFM();
                    ofmConn->tensor->RemoveWriter(lastNonFusedOp);
                    ofmConn->tensor = nextOp->OFM()->tensor;
                    ofmConn->tensor->producers.push_back(lastNonFusedOp);
                    ofmConn->tensor->RemoveWriter(nextOp);
                    ofmConn->SetType(nextOp->OFM()->Type());
                    ofmConn->shape = nextOp->OFM()->shape;
                    ofmConn->slice = nextOp->OFM()->slice;
                    ofmConn->reverse = nextOp->OFM()->reverse;
                }
                else
                {
                    // This is a non-fused op - update the tracking variable
                    lastNonFusedOp = nextOp;
                }

                primaryOp->AddSubOp(std::move(*cur));
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

        if ( !op->IsNpuOp() && (AllInputsAreGraphInputs(*op) || AllInputsProducersAreEarlyOps(*op, earlyOps)) )
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

    // can't pack CPU operations
    if ( !nextOp->IsNpuOp() )
    {
        return 0;
    }

    // Do not pack persistent tensors with non persistent tensors
    if ( prevOFM->isPersistent != nextOp->OFM()->tensor->isPersistent )
    {
        return 0;
    }

    // Previous op in execution order doesn't connect to this one
    if ( prevOFM != ifmTensor && prevOFM != ifm2Tensor )
    {
        return 0;
    }

    // Do not pack elementwise operations if chaining is disabled
    if ( _disableChaining && IsElementwise(nextOp->Type()) )
    {
        return 0;
    }

    // Highly unlikely constant tensor between ops
    assert(!prevOFM->IsConstant() && "Unexpected constant tensor between ops");

    // Only pack tensors on single-reader/writer paths (i.e. can't pack across concat/split)
    if ( prevOFM->producers.size() != 1 || prevOFM->consumers.size() != 1 )
    {
        return 0;
    }

    // Previous op has OFM slice
    if ( !Shape::IsReducedEqual(prevConnOfm->shape, prevConnOfm->SliceShape()) )
    {
        return 0;
    }

    if ( schedOp->OFM()->tensor->isGraphOutput || prevOp->OFM()->tensor->isGraphOutput )
    {
        return 0;
    }

    // Can not fuse operations with different scales
    if ( IsElementwise(nextOp->Type()) == false && !nextConnIfm->quantization.EqualScales(nextConnOfm->quantization) )
    {
        return 0;
    }

    auto op1 = CreateOpGroupQuery(nextOp);
    return schedOp->_opGroup->Add(op1, {prevOpKey});
}

void SchedulerPacking::InitSchedulerConnection(
    SchedulerConnection *schedConn, const std::shared_ptr<SchedulerTensor> &tensor, const TensorConnection &conn)
{
    schedConn->tensor = tensor;
    // Convert to (minimum) 4D-shapes in scheduler-IR
    schedConn->slice = {Shape::PadAxes(conn.slice.offset, 4, 0), Shape::PadAxes(conn.slice.shape, 4, 1),
        Shape::PadAxes(conn.slice.stride, 4, 1)};
    schedConn->shape = Shape::PadAxes(conn.shape, 4, 1);
    schedConn->quantization = conn.quantization;
    schedConn->reverse = conn.reverse;
    schedConn->resamplingMode = ArchResampling::None;
    schedConn->rounding = conn.rounding;
    if ( schedConn->slice.stride )
    {
        schedConn->stepXY = schedConn->slice.stride.WH<int>();
    }
}

void SchedulerPacking::InitSchedulerTensor(SchedulerTensor *schedTensor, Tensor *tensor)
{
    const auto type = tensor->Type();
    // Take scheduler-local copies of graph tensor parameters.
    schedTensor->format = TensorFormat::NHWC;
    schedTensor->memArea = tensor->IsConstant() ? _arch->ReadonlyMemory() : _arch->FeatureMapMemory();
    schedTensor->storageShape = Shape::PadAxes(tensor->StorageShape(), 4, 1);
    schedTensor->dataType = type;
    schedTensor->bufferView = (IsVariablySized(type) || type == DataType::None) ? BufferView() : tensor->View();
    schedTensor->isGraphInput = _graph->IsInput(tensor);
    schedTensor->isGraphOutput = _graph->IsOutput(tensor);
    schedTensor->isPersistent = _graph->IsPersistent(tensor);
    schedTensor->uid = tensor->Uid();
    if ( tensor->View().HasBuffer() )
    {
        // Assign equivalenceIds based on the underlying buffer of the GraphIR tensor (if present).
        // This ensures that all tensors sharing a buffer will be allocated to the same memory.
        auto buffer = tensor->View().Buffer();
        auto eqId = _bufferEquivalenceIdMap.find(buffer->Hash());
        if ( eqId == _bufferEquivalenceIdMap.end() )
        {
            _bufferEquivalenceIdMap.emplace(buffer->Hash(), schedTensor->equivalenceId);
        }
        else
        {
            schedTensor->equivalenceId = eqId->second;
        }
    }
}

std::unique_ptr<SchedulerOperation> SchedulerPacking::MakeSchedulerOperation(const Operation *op)
{
    assert(op->Type() != OpType::None);

    if ( op->Type() == OpType::ReinterpretCast )
    {
        HandleReinterpretCast(op);
        return nullptr;
    }

    std::unique_ptr<SchedulerOperation> schedOp = std::make_unique<SchedulerOperation>(op->Type());

    schedOp->SetKernel(*op->Kernel());
    schedOp->SetHasScaling(op->HasScaling());
    schedOp->SetAttributes(op->AttributeRef());
    schedOp->_srcKey = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(op));

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
                InitSchedulerTensor(tmp.get(), tensor);
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
            if ( IsIFM(item.first) && tensor->Type() != schedTensor->dataType )
            {
                schedConn->SetType(tensor->Type());
            }
            schedConn->transpose = TransposeType::None;
        }
    }

    // Adjust axis attribute if tensors have been reshaped
    if ( schedOp->HasAttribute<axis_attr_t>() )
    {
        auto attr = schedOp->Attribute<axis_attr_t>();
        // The axis attribute refers to the IFM rank
        int paddedAxes = schedOp->Input(TensorUsage::IFM)->shape.Size() - op->Input(TensorUsage::IFM)->shape.Size();
        assert(paddedAxes >= 0);
        attr->axis += paddedAxes;
        assert(attr->axis < schedOp->Input(TensorUsage::IFM)->shape.Size());
    }
    // Update OFM transpose mask if operator has the attribute
    else if ( schedOp->HasAttribute<transpose_attr_t>() )
    {
        auto attr = schedOp->Attribute<transpose_attr_t>();
        schedOp->OFM()->transpose = TransposeTypeFromShape(attr->perm);
    }

    return schedOp;
}

std::vector<std::unique_ptr<SchedulerOperation>> SchedulerPacking::DecomposeSchedulerOperation(std::unique_ptr<SchedulerOperation> op)
{
    auto srcKey = op->_srcKey;

    std::vector<std::unique_ptr<SchedulerOperation>> result;
    ArchRequirements req{};

    switch ( op->Type() )
    {
        case OpType::FullyConnected:
            [[fallthrough]];
        case OpType::Conv2D:
            result = DecomposeConv2D(_arch, std::move(op));
            break;
        case OpType::Conv3D:
            result = DecomposeConv3D(_arch, std::move(op));
            break;
        case OpType::DepthwiseConv2D:
            result = DecomposeDepthwiseConv2D(_arch, std::move(op));
            break;
        case OpType::TransposeConv2D:
            result = DecomposeTransposeConv2D(_arch, std::move(op));
            break;
        case OpType::MatMul:
            result = DecomposeMatmul(_arch, std::move(op));
            break;
        case OpType::ArgMax:
            [[fallthrough]];
        case OpType::ReduceSum:
            [[fallthrough]];
        case OpType::ReduceMin:
            [[fallthrough]];
        case OpType::ReduceMax:
            [[fallthrough]];
        case OpType::ReduceAny:
            [[fallthrough]];
        case OpType::ReduceAll:
            result = DecomposeReduce(_arch, std::move(op));
            break;
        case OpType::Reverse:
            result = DecomposeReverse(_arch, std::move(op));
            break;
        case OpType::Resize:
            OperatorQuery(_arch, op.get(), &req);
            if ( req.req.Any(ArchRequirement::OpSubstitution) )
            {
                result = LegaliseResize(_arch, std::move(op));
            }
            else
            {
                result = DecomposeResize(_arch, std::move(op));
            }
            break;
        case OpType::Transpose:
            result = DecomposeTranspose(_arch, std::move(op));
            break;
        case OpType::AvgPool:
            result = DecomposeAvgPool(_arch, std::move(op));
            break;
        case OpType::MaxPool:
            result = DecomposeMaxPool(_arch, std::move(op));
            break;
        default:
            if ( DecomposeAsElementwise(op->Type()) || op->Type() == OpType::MemoryCopy )
            {
                result = DecomposeElementwise(_arch, std::move(op));
            }
            else
            {
                LOG_PRINT("!!!! Can't decompose op:{}", OpTypeToString(op->Type()));
                assert(false);
            }
            break;
    }

    for ( std::unique_ptr<SchedulerOperation> &so : result )
        so->_srcKey = srcKey;

    return result;
}


}  // namespace regor
