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

#pragma once

#include "graph.hpp"
#include "operation.hpp"
#include "scheduler_operation.hpp"
#include "tensor.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace regor
{

/// <summary>
/// Graph packing
/// </summary>
class GraphPacking
{
public:
    GraphPacking();

public:
    std::unique_ptr<Graph> Process(std::vector<std::pair<Operation *, std::unique_ptr<NPUOperation>>> &npuOps,
        std::vector<std::unique_ptr<SchedulerOperation>> &ops,
        std::unordered_map<const Tensor *, Address> &tensorAddressMap, const Graph *srcGraph);

private:
    std::unordered_map<SchedulerOperation *, std::shared_ptr<Operation>> _oldOpToNewOp;
    std::unordered_map<Tensor *, std::shared_ptr<Tensor>> _oldTensorToNewTensor;

    void ConnectTensors(Operation *op, const std::unique_ptr<SchedulerOperation> &schedOp,
        std::unordered_map<const Tensor *, Address> &tensorAddressMap);

    std::shared_ptr<Tensor> LookupNewTensor(Tensor *oldTensor);
    std::shared_ptr<Tensor> LookupNewTensor(
        Tensor *oldTensor, std::unordered_map<const Tensor *, Address> &tensorAddressMap, Address allocatedAddress);
};

// Pack list of scheduler operations into one or more graphs
std::unique_ptr<Graph> PackScheduleToGraph(std::vector<std::pair<Operation *, std::unique_ptr<NPUOperation>>> &npuOps,
    std::vector<std::unique_ptr<SchedulerOperation>> &ops,
    std::unordered_map<const Tensor *, Address> &tensorAddressMap, const Graph *srcGraph);

}  // namespace regor
