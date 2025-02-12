//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "architecture/architecture_constraints.hpp"
#include "graph.hpp"
#include "scheduler_operation.hpp"

#include <vector>

namespace regor
{
class DecompositionFailure : public std::runtime_error
{
public:
    DecompositionFailure(const std::string &what = "") : std::runtime_error(what) {}
};

bool NeedsDecompose(Architecture *arch, const SchedulerOperation *schedOp);
bool CanRunOnHardware(Architecture *arch, const SchedulerOperation *schedOp);
bool CanDecompose(Architecture *arch, const SchedulerOperation *schedOp);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeConv2D(Architecture *arch, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeConv3D(Architecture *arch, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeDepthwiseConv2D(Architecture *arch, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeTransposeConv2D(Architecture *arch, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeElementwise(Architecture *arch, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeMatmul(Architecture *arch, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeReduce(Architecture *arch, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeReverse(Architecture *arch, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeTranspose(Architecture *arch, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeMaxPool(Architecture *arch, std::unique_ptr<SchedulerOperation> op);


// Operator query helpers
inline ArchFM &Set(ArchFM &fm, const SchedulerConnection *conn)
{
    if ( conn )
    {
        fm.type = conn->tensor->dataType;
        fm.shape = conn->slice.shape ? conn->slice.shape : conn->shape;
        fm.format = conn->tensor->format;
    }
    return fm;
}

}  // namespace regor
