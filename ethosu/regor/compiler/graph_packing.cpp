//
// SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "graph_packing.hpp"

#include "common/logging.hpp"

#include "graph.hpp"
#include "scheduler_operation.hpp"
#include "tensor.hpp"

#include <cassert>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace regor
{

GraphPacking::GraphPacking()
{
}

std::unique_ptr<Graph> GraphPacking::Process(std::vector<std::pair<Operation *, std::unique_ptr<NPUOperation>>> &npuOps,
    std::vector<std::unique_ptr<SchedulerOperation>> &ops, std::unordered_map<const Tensor *, Address> &tensorAddressMap, const Graph *srcGraph)
{
    // Build a new graph where consecutive operations running on NPU are collapsed into a Ethos-U op
    // (OpType::CustomNpuOp). CPU operations are left unchanged. The algorithm makes two passes over the scheduled
    // operations, where the first pass creates new operations (Ethos-U and CPU) and connects the CPU operations, and
    // the second pass connects the Ethos-U operations.

    // List of all ops in new graph in scheduled order
    std::vector<Operation *> newOpsInScheduledOrder;

    std::shared_ptr<Operation> currentOp = nullptr;
    NPUOperation *currentNpuOp = nullptr;

    bool allowCPUOps = (srcGraph->Notation() != GraphNotation::GraphAPI);

    // Pack consecutive NPU ops into a NPUOperation
    for ( auto &schedOp : ops )
    {
        if ( schedOp->IsNpuOp() )
        {
            if ( !currentNpuOp )
            {
                // Create new Ethos-U operation for the new graph
                currentOp = std::make_shared<Operation>(OpType::CustomNpuOp);

                // Create new NPUOperation that collects consecutive NPU operations
                auto newNpuOp = std::make_unique<NPUOperation>();
                currentNpuOp = newNpuOp.get();

                newOpsInScheduledOrder.push_back(currentOp.get());

                npuOps.emplace_back(currentOp.get(), std::move(newNpuOp));
            }

            // Map old scheduler operation and its sub-ops to new operation
            _oldOpToNewOp[schedOp.get()] = currentOp;
            for ( const auto &subOp : schedOp->SubOps() )
            {
                assert(subOp->IsNpuOp());
                _oldOpToNewOp[subOp.get()] = currentOp;
            }
            currentNpuOp->AddOperation(std::move(schedOp));
        }
        else if ( allowCPUOps )
        {
            // Create new CPU operation for the new graph
            assert(schedOp->_srcKey != nullptr);
            currentOp = std::make_shared<Operation>(*static_cast<Operation *>(schedOp->_srcKey));
            currentNpuOp = nullptr;

            newOpsInScheduledOrder.push_back(currentOp.get());

            // Map old scheduler operation to new CPU operation
            _oldOpToNewOp[schedOp.get()] = currentOp;

            for ( const auto &[usage, schedConn] : schedOp->inputs.pairs() )
            {
                const auto &schedTensor = schedConn.tensor;

                // Connect input tensor to new CPU operation
                const auto oldTensor = schedTensor->srcTensor;
                assert(oldTensor && "Missing source graph tensor");
                const auto newTensor = LookupNewTensor(oldTensor.get(), tensorAddressMap, schedTensor->AllocatedAddress());
                currentOp->ConnectInput(usage, newTensor).Set(schedConn.shape).Set(schedConn.quantization);
            }

            for ( const auto &[usage, schedConn] : schedOp->outputs.pairs() )
            {
                const auto &schedTensor = schedConn.tensor;

                // Connect output tensor to new CPU operation
                const auto oldTensor = schedTensor->srcTensor;
                assert(oldTensor && "Missing source graph tensor");
                const auto newTensor = LookupNewTensor(oldTensor.get(), tensorAddressMap, schedTensor->AllocatedAddress());
                currentOp->ConnectOutput(usage, newTensor).Set(schedConn.shape).Set(schedConn.quantization);
            }
        }
        else
        {
            throw std::runtime_error("CPU operations are not supported for GraphAPI input");
        }
    }

    currentNpuOp = nullptr;
    currentOp = nullptr;

    // Connect Input/Output tensors
    for ( auto &item : npuOps )
    {
        Operation *op = item.first;
        for ( const auto &schedOp : item.second->Operations() )
        {
            ConnectTensors(op, schedOp, tensorAddressMap);
            for ( const auto &subOp : schedOp->SubOps() )
            {
                ConnectTensors(op, subOp, tensorAddressMap);
            }
        }
    }

    // Clear ops since they have been moved into relevant NPUOperation object
    ops.clear();
    _oldOpToNewOp.clear();

    auto graph = std::make_unique<Graph>(srcGraph->Notation());
    graph->SetName(srcGraph->Name());
    graph->SetPassthrough(srcGraph->Passthrough());

    // Transfer graph input tensors from old graph
    for ( const auto &graphInput : srcGraph->Inputs() )
    {
        graph->AddInput(LookupNewTensor(graphInput.get()));
    }

    // Transfer graph output tensors from old graph
    for ( const auto &graphOutput : srcGraph->Outputs() )
    {
        graph->AddOutput(LookupNewTensor(graphOutput.get()));
    }

    // Transfer persistent tensors from old graph
    for ( const auto &tensor : srcGraph->Persistent() )
    {
        graph->AddPersistent(LookupNewTensor(tensor.get()));
    }

    // Transfer placeholder tensors from old graph
    for ( const auto &tensor : srcGraph->Placeholder() )
    {
        graph->AddPlaceholder(LookupNewTensor(tensor.get()));
    }

    _oldTensorToNewTensor.clear();

    // Save the execution order of all ops in the new graph
    graph->SetScheduledOrder(std::move(newOpsInScheduledOrder));

    return graph;
}

void GraphPacking::ConnectTensors(Operation *op, const std::unique_ptr<SchedulerOperation> &schedOp,
    std::unordered_map<const Tensor *, Address> &tensorAddressMap)
{
    auto isCurrentOp = [&map = _oldOpToNewOp, op](SchedulerOperation *sop) { return map[sop].get() == op; };
    for ( const auto &schedConn : schedOp->inputs )
    {
        const auto &schedTensor = schedConn.tensor;
        if ( schedTensor->IsConstant() )
        {
            // Don't connect constant tensors - they are handled at scheduler level
            continue;
        }

        const bool isConsumedByUs = std::any_of(schedTensor->consumers.begin(), schedTensor->consumers.end(), isCurrentOp);
        const bool isProducedByUsOnly = std::all_of(schedTensor->producers.begin(), schedTensor->producers.end(), isCurrentOp);
        if ( isConsumedByUs && isProducedByUsOnly && !schedTensor->isGraphInput && !schedTensor->isPersistent )
        {
            // Don't connect NPU internal tensors
            continue;
        }

        // Connect input tensor to new Ethos-U operation, but only once
        const auto &oldTensor = schedTensor->srcTensor;
        if ( oldTensor )
        {
            const auto newTensor = LookupNewTensor(oldTensor.get(), tensorAddressMap, schedTensor->AllocatedAddress());
            if ( op->UsageOfTensor(newTensor.get()) == TensorUsage::None )
            {
                const auto usage = MakeTensorUsage(TensorUsage::IFM, op->Inputs().size());
                op->ConnectInput(usage, newTensor).Set(schedConn.quantization);
            }
        }
    }

    for ( const auto &schedConn : schedOp->outputs )
    {
        const auto &schedTensor = schedConn.tensor;

        const bool isProducedByUs = std::any_of(schedTensor->producers.begin(), schedTensor->producers.end(), isCurrentOp);
        const bool isConsumedByUsOnly = std::all_of(schedTensor->consumers.begin(), schedTensor->consumers.end(), isCurrentOp);
        if ( isProducedByUs && isConsumedByUsOnly && !schedTensor->isGraphOutput && !schedTensor->isPersistent )
        {
            // Don't connect NPU internal tensors
            continue;
        }

        // Connect output tensor to new Ethos-U operation, but only once
        const auto &oldTensor = schedTensor->srcTensor;
        if ( oldTensor )
        {
            const auto newTensor = LookupNewTensor(oldTensor.get(), tensorAddressMap, schedTensor->AllocatedAddress());
            if ( op->UsageOfTensor(newTensor.get()) == TensorUsage::None )
            {
                const auto usage = MakeTensorUsage(TensorUsage::OFM, op->Outputs().size());
                op->ConnectOutput(usage, newTensor).Set(schedConn.quantization);
            }
        }
    }
}

std::shared_ptr<Tensor> GraphPacking::LookupNewTensor(Tensor *oldTensor)
{
    const auto it = _oldTensorToNewTensor.find(oldTensor);
    if ( it == _oldTensorToNewTensor.end() )
    {
        // This cloned tensor will be used in the new graph
        std::shared_ptr<Tensor> newTensor = oldTensor->Clone();

        _oldTensorToNewTensor[oldTensor] = newTensor;

        return newTensor;
    }
    else
    {
        return it->second;
    }
}

std::shared_ptr<Tensor> GraphPacking::LookupNewTensor(
    Tensor *oldTensor, std::unordered_map<const Tensor *, Address> &tensorAddressMap, Address allocatedAddress)
{
    const auto newTensor = LookupNewTensor(oldTensor);

    // This cloned tensor will use same address as the original tensor
    tensorAddressMap[newTensor.get()] = allocatedAddress;

    return newTensor;
}

std::unique_ptr<Graph> PackScheduleToGraph(std::vector<std::pair<Operation *, std::unique_ptr<NPUOperation>>> &npuOps,
    std::vector<std::unique_ptr<SchedulerOperation>> &ops, std::unordered_map<const Tensor *, Address> &tensorAddressMap, const Graph *srcGraph)
{
    GraphPacking p;

    return p.Process(npuOps, ops, tensorAddressMap, srcGraph);
}

}  // namespace regor
