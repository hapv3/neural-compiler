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

#include "architecture/architecture.hpp"
#include "scheduler.hpp"
#include "scheduler_operation.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace regor
{

/// <summary>
/// Live range
/// </summary>
struct LiveRange
{
    // Tensors with equivalence ids that are assigned to the same LiveRange will be allocated to the same address
    std::unordered_set<SchedulerTensor *> tensors;
    // Time at which the live range's tensors start being used.
    int startTime = std::numeric_limits<int>::max();
    // Note: the end time is inclusive
    int endTime = -1;
    int size = 0;
    MemArea memArea;
    std::string name = "";

    LiveRange(SchedulerTensor *tensor)
    {
        size = tensor->AllocationSizeBytes();
        memArea = tensor->memArea;
        name = tensor->Name();
        AddTensor(tensor);
    }

    void AddTensor(SchedulerTensor *tensor) { tensors.insert(tensor); }

    void MarkUsage(int opTime, int opDuration = 1)
    {
        assert(opDuration >= 0);
        int opTimeStart = std::max(opTime, 0);
        int opTimeEnd = opTime + opDuration;
        if ( opTimeEnd > opTimeStart )
        {
            startTime = std::min(startTime, opTimeStart);
            endTime = std::max(endTime, opTimeEnd);
        }
    }

    void SetAddress(Address address)
    {
        for ( auto &tensor : tensors )
        {
            tensor->SetAddress(address);
        }
    }

    std::string ToString() const
    {
        return fmt::format("<LiveRange {}, time: {}-{}, size: {}>", name, startTime, endTime, size);
    }
};

class LiveRangeGraph
{
private:
    /** All allocated live ranges */
    std::vector<std::shared_ptr<LiveRange>> _lrs;
    /** Map from equivalence id -> live range */
    std::unordered_map<UniqueId, LiveRange *> _equivalenceIdToLr;
    int _currentTime = 0;

public:
    virtual ~LiveRangeGraph() = default;
    int EndTime() const { return _currentTime + 1; }

    std::vector<std::shared_ptr<LiveRange>> LiveRanges() const { return _lrs; };

    /** usage[t] will be set to the memory usage at time t, for each timestamp t in the live graph */
    std::vector<int> GetTemporalMemoryUsage(int &maxUsage, int granularity = 16);
    void ExtractLiveRangesFromCascades(const std::vector<std::unique_ptr<SchedulerOperation>> &schedOps,
        Schedule *schedule, const MemArea &targetMemory, bool addRollingBuffers);
    LiveRange *GetOrCreateRange(SchedulerTensor *tens);
    LiveRange *FuseRanges(SchedulerTensor *inTens, SchedulerTensor *outTens);
    SchedulerTensor *ReusableIFM(const std::unique_ptr<SchedulerOperation> &schedOp, const MemArea &targetMemory);
    virtual bool ShouldBeIgnored(SchedulerTensor *tens, const MemArea &targetMemory);
};

}  // namespace regor
