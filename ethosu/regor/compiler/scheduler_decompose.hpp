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

#pragma once

#include "graph.hpp"
#include "operation.hpp"
#include "scheduler_operation.hpp"

#include <vector>

namespace regor
{
bool NeedsDecompose(Architecture *arch, const SchedulerOperation *schedOp);
bool CanRunOnHardware(Architecture *arch, const SchedulerOperation *schedOp);
bool CanDecompose(Architecture *arch, const SchedulerOperation *schedOp);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeConv2D(Architecture *arch, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeDepthwiseConv2D(Architecture *arch, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeTransposeConv2D(Architecture *arch, std::unique_ptr<SchedulerOperation> op);
std::vector<std::unique_ptr<SchedulerOperation>> DecomposeElementwise(Architecture *arch, std::unique_ptr<SchedulerOperation> op);
}  // namespace regor
