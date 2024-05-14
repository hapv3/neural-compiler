//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "cascade_builder.hpp"
#include "common/vector_span.hpp"
#include "high_level_command_stream.hpp"
#include "scheduler.hpp"
#include "scheduler_operation.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace regor
{

using HLCStream = std::vector<std::unique_ptr<HighLevelCommand>>;

/// <summary>
/// High level command stream generator
/// </summary>
class HLCStreamGenerator
{
public:
    // Generates high level command stream for the scheduled operations in the given NPU op
    HLCStream GenerateCommandStream(const NPUOperation *npuOp, const Schedule *schedule, bool verbose);

private:
    // Generates one or more HLCStripe commands from a given operation and adds them to the stream
    void GenerateHLCStripeCommands(SchedulerOperation *op, const std::shared_ptr<HLCOperation> &hlcOp, HLCStream &cmds);
    // Generates one or more HLCDMA commands from a given operation and adds them to the stream
    void GenerateHLCDMACommands(SchedulerOperation *op, const std::shared_ptr<HLCOperation> &hlcOp, HLCStream &cmds);
    // Generates high level commands for the given operation and adds them to the command stream
    void GenerateCommands(SchedulerOperation *op, const std::shared_ptr<HLCOperation> &hlcOp, HLCStream &cmds);
    // Generates high level commands for all operations in the cascade and adds them to the command stream
    void GenerateCommandsForCascade(vector_span<std::unique_ptr<SchedulerOperation>> cascadedOps,
        vector_span<std::shared_ptr<HLCOperation>> hlcOps, const CascadeInfo *cascadeInfo, HLCStream &cmds);
    void PrintCommandStream(const NPUOperation *npuOp, std::vector<std::shared_ptr<HLCOperation>> &hlcOps, HLCStream &cmds);

    // Tracking what has been put in the weight buffers
    std::unordered_map<SchedulerTensor *, std::tuple<UniqueId, int /* start channel */, int /* depth index */>> _filledWeightBuffers;

    const Schedule *_schedule = nullptr;
};

}  // namespace regor
