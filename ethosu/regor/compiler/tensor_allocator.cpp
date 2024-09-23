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

#include "tensor_allocator.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "architecture/architecture.hpp"
#include "common/bit_flags.hpp"
#include "common/numeric_util.hpp"
#include "hillclimb_allocator.hpp"
#include "live_range.hpp"

namespace regor
{

namespace
{

// Implementation of the linear allocator
Address LinearAllocateLiveRanges(LiveRangeGraph &lrGraph, int alignment)
{
    Address address = 0;
    for ( const auto &lr : lrGraph.LiveRanges() )
    {
        lr->SetAddress(address);
        address += RoundAway<int>(lr->size, alignment);
    }
    return address;
}

void PrintAllocation(LiveRangeGraph &lrGraph, Address totalSize)
{
    LOG_PRINT("{0:10} - {1:10}: {2:>10} - {3:>10}: {4:11}: {5:12} : {6}\n", "Start Time", "End Time", "Start Addr",
        "End Addr", "Tensor Size", "Memory Usage", "Name");
    auto lrs = lrGraph.LiveRanges();
    // Sort LiveRanges
    std::sort(lrs.begin(), lrs.end(),
        [](const std::shared_ptr<LiveRange> &a, const std::shared_ptr<LiveRange> &b)
        {
            if ( a->startTime != b->startTime ) return a->startTime < b->startTime;
            return a->endTime < b->endTime;
        });
    // Create memory histogram to track usage over time
    std::vector<int> memHist(lrGraph.EndTime(), 0);
    for ( const auto &lr : lrs )
    {
        for ( int t = lr->startTime; t <= lr->endTime; ++t )
        {
            memHist[t] += lr->size;
        }
    }

    for ( const auto &lr : lrs )
    {
        if ( lr->tensors.empty() )
        {
            continue;
        }
        auto address = (*lr->tensors.begin())->AllocatedAddress();
        auto peakUsageDuringLiveRange = *std::max_element(memHist.begin() + lr->startTime, memHist.begin() + lr->endTime + 1);
        for ( const auto &tens : lr->tensors )
        {
            LOG_PRINT("{0:10} - {1:10}: {2:#10x} - {3:#10x}: {4:11}: {5:12} : {6}\n", lr->startTime, lr->endTime,
                address, address + lr->size, lr->size, peakUsageDuringLiveRange, tens->Name());
        }
    }
    LOG_PRINT("Allocation Peak Tensor Size: {} bytes == {} KiB\n", totalSize, double(totalSize) / 1024.0);
}

Address Allocate(LiveRangeGraph &lrGraph, const std::vector<std::unique_ptr<SchedulerOperation>> &schedOps,
    Schedule *schedule, const MemArea &targetMemory, TensorAllocator allocator, int alignment, Address sizeLimit)
{
    lrGraph.ExtractLiveRangesFromCascades(schedOps, schedule, targetMemory, false);
    Address totalSize = 0;
    if ( allocator == TensorAllocator::LinearAlloc )
    {
        totalSize = LinearAllocateLiveRanges(lrGraph, alignment);
    }
    else if ( allocator == TensorAllocator::HillClimb )
    {
        totalSize = HillClimbAllocateLiveRanges(lrGraph, alignment, sizeLimit);
    }
    return totalSize;
}

}  // namespace

Address IncrementalLinearAllocator::Allocate(LiveRangeGraph *lrGraph, int alignment, bool verboseAllocation)
{
    for ( const auto &lr : lrGraph->LiveRanges() )
    {
        if ( !lr->tensors.empty() )
        {
            auto tensor = *(lr->tensors.begin());
            auto it = _allocatedAddresses.find(tensor->equivalenceId);
            if ( it == _allocatedAddresses.end() )
            {
                lr->SetAddress(_highestAddress);
                _allocatedAddresses[tensor->equivalenceId] = _highestAddress;
                _highestAddress += RoundAway<int>(lr->size, alignment);
            }
            else
            {
                // An equivalent tensor has previously been allocated, reuse its address
                lr->SetAddress(it->second);
            }
        }
    }
    if ( verboseAllocation )
    {
        LOG_PRINT("{0:#^{1}}\n", "", 80);
        LOG_PRINT("Tensor Allocation for {}:\n", _name);
        PrintAllocation(*lrGraph, _highestAddress);
    }
    return _highestAddress;
}

void AllocateTensors(const std::vector<std::unique_ptr<SchedulerOperation>> &schedOps, Schedule *schedule,
    const MemArea &memArea, TensorAllocator allocator, int alignment, bool verboseAllocation, Address sizeLimit)
{
    LiveRangeGraph lrGraph;
    auto totalSize = Allocate(lrGraph, schedOps, schedule, memArea, allocator, alignment, sizeLimit);
    if ( verboseAllocation )
    {
        LOG_PRINT("{0:#^{1}}\n", "", 80);
        LOG_PRINT("Allocation, memory {}, usage mask: {}\n", memArea.memory->Name(), memArea.usage.ToString());
        PrintAllocation(lrGraph, totalSize);
    }
    schedule->memoryUsage[memArea] = int(totalSize);
}

}  // namespace regor
