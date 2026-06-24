//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include <unordered_map>
#include <vector>

namespace regor
{
class DecompositionFailure : public std::runtime_error
{
public:
    DecompositionFailure(const std::string &what = "") : std::runtime_error(what) {}
};

struct DecompositionContext
{
    DecompositionContext(Architecture *a) : arch(a) {}
    Architecture *arch = nullptr;
    std::unordered_map<UniqueId, std::shared_ptr<ArchitectureOpConfig>> opConfigCompatablility;
};

Flags<QueryResult> OperatorQuery(Architecture *arch, const SchedulerOperation *schedOp, ArchRequirements *req);
bool ShouldDecompose(Architecture *arch, const SchedulerOperation *schedOp);
bool NeedsDecompose(Architecture *arch, const SchedulerOperation *schedOp);
bool CanDecompose(Architecture *arch, const SchedulerOperation *schedOp);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeConv2D(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeConv3D(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeDepthwiseConv2D(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeTransposeConv2D(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>>
DecomposeTransposeConv2DLargeStride(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeElementwise(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeMemoryCopy(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeMatmul(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeReduce(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeReverse(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeTranspose(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeAvgPool(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeMaxPool(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeResize(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> LegaliseResize(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> LegaliseTransposeConv2D(DecompositionContext &ctx, std::unique_ptr<SchedulerOperation> op);


// Operator query helpers
inline ArchFM &Set(ArchFM &fm, const SchedulerConnection *conn)
{
    if ( conn )
    {
        fm.type = conn->Type();
        fm.shape = conn->SliceShape();
        fm.format = conn->tensor->format;
        fm.quantization = &conn->quantization;
    }
    return fm;
}

inline ArchPerfFM &Set(ArchPerfFM &fm, const SchedulerConnection *conn)
{
    if ( conn )
    {
        Set(static_cast<ArchFM &>(fm), conn);
        fm.memory = conn->tensor->memArea.memory;
    }
    return fm;
}

}  // namespace regor
