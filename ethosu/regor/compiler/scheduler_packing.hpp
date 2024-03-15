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

#pragma once

#include "common/common.hpp"
#include "common/logging.hpp"

#include "common/shape.hpp"
#include "graph.hpp"
#include "operation.hpp"
#include "scheduler_operation.hpp"
#include "tensor.hpp"

#include <deque>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace regor
{

/// <summary>
/// Pack graph-level operations into flat/linear SchedulerOperation objects
/// (NOTE: Was pass_packing, but we skip the Pass to pack directly into SchedulerOperation)
/// </summary>
class SchedulerPacking
{
protected:
    Architecture *_arch = nullptr;
    bool _disableChaining = false;
    std::vector<std::unique_ptr<SchedulerOperation>> _schedList;
    std::unordered_map<Tensor *, std::shared_ptr<SchedulerTensor>> _tensorMap;

public:
    SchedulerPacking(Architecture *arch, bool disableChaining);

public:
    std::vector<std::unique_ptr<SchedulerOperation>> Process(const Graph *graph);

private:
    void FilterOperations(const std::vector<Operation *> &executionList, const Graph *graph);
    void PackOperations();
    void ReorderOperations();

    int CanPack(const SchedulerOperation *schedOp, const SchedulerOperation *prevOp, const SchedulerOperation *op, const int prevOpKey) const;
    void InitSchedulerConnection(SchedulerConnection *schedConn, const std::shared_ptr<SchedulerTensor> &tensor, const TensorConnection &conn);
    void InitSchedulerTensor(SchedulerTensor *schedTensor, Tensor *tensor, const Graph *graph);
    std::unique_ptr<SchedulerOperation> MakeSchedulerOperation(Operation *op, const Graph *graph);
    std::vector<std::unique_ptr<SchedulerOperation>> DecomposeSchedulerOperation(std::unique_ptr<SchedulerOperation> op);
    ArchResampling ResamplingMode(TensorUsage usage, OpType opType) const;
    ArchitectureOpGroupQuery CreateOpGroupQuery(const SchedulerOperation *schedOp) const;
};

}  // namespace regor
