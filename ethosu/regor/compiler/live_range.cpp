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


#include "live_range.hpp"

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

std::vector<int> LiveRangeGraph::GetTemporalMemoryUsage(int &maxUsage, int granularity)
{
    assert(granularity > 0);
    std::vector<int> usage(_currentTime + 1);
    maxUsage = 0;
    for ( const auto &lr : _lrs )
    {
        assert(lr->endTime <= _currentTime);
        for ( int i = lr->startTime; i <= lr->endTime; ++i )
        {
            assert((i >= 0) && (i < int(usage.size())));
            usage[i] += RoundAway(lr->size, granularity);
            maxUsage = std::max(maxUsage, usage[i]);
        }
    }
    return usage;
}

void LiveRangeGraph::ExtractLiveRangesFromCascades(const std::vector<std::unique_ptr<SchedulerOperation>> &schedOps,
    Schedule *schedule, const MemArea &targetMemory, bool addRollingBuffers)
{
    std::unordered_map<int, int> timeForCascade;
    auto startTime = _currentTime;
    // Live ranges containing graph output
    std::vector<LiveRange *> graphOutputRanges;
    // Live ranges containing persistent tensors
    std::vector<LiveRange *> persistentRanges;
    for ( const auto &schedOp : schedOps )
    {
        SchedulerOpInfo *opInfo = schedule->Cost(schedOp.get());
        int cascade = opInfo->cascade;

        CascadeInfo *cascadeInfo = cascade == 0 ? nullptr : &schedule->cascades[cascade];
        CascadeBuffer *cascadeBuffer = nullptr;

        int timeToSet = _currentTime;
        if ( schedOp->IsNpuOp() )
        {
            if ( cascadeInfo == nullptr )
            {
                auto opGroup = schedOp->OpGroup();
                assert(opGroup != nullptr);
                if ( opGroup->NeedsAllocation(schedOp->OFM()->tensor->uid) )
                {
                    // Check if op have an ifm tensor that can be reused for the ofm
                    auto ifmTens = ReusableIFM(schedOp, targetMemory);
                    if ( ifmTens != nullptr )
                    {
                        // ifm can be reused
                        FuseRanges(ifmTens, schedOp->OFM()->tensor.get());
                    }
                }
            }
            else
            {
                auto entry = cascadeInfo->buffers.find(*schedOp);
                if ( entry != cascadeInfo->buffers.end() )
                {
                    cascadeBuffer = &entry->second;
                }
                auto tfcEntry = timeForCascade.find(cascade);
                if ( tfcEntry != timeForCascade.end() )
                {
                    timeToSet = tfcEntry->second;
                }
                timeForCascade[cascade] = timeToSet;
            }
            // Buffered weight tensor
            auto weightTens = opInfo->bufferedWeightTensor.tensor.get();
            if ( !ShouldBeIgnored(weightTens, targetMemory) )
            {
                auto lr = GetOrCreateRange(weightTens);
                if ( opInfo->bufferedWeightTensor.preBuffer )
                {
                    lr->MarkUsage(timeToSet - 1, 2);
                }
                else
                {
                    lr->MarkUsage(timeToSet);
                }
            }
            // Read-only weight/scale tensors
            for ( auto tens : {opInfo->npuWeightsTensor, opInfo->npuScalesTensor} )
            {
                if ( !ShouldBeIgnored(tens.get(), targetMemory) )
                {
                    auto lr = GetOrCreateRange(tens.get());
                    lr->MarkUsage(timeToSet);
                }
            }
        }

        // Set time index for the op and its subops
        opInfo->timeIndex = timeToSet;
        for ( auto &subOp : schedOp->SubOps() )
        {
            SchedulerOpInfo *subOpInfo = schedule->Cost(subOp.get());
            subOpInfo->timeIndex = timeToSet;
        }

        // Mark usage for all relevant tensors related to this operation
        for ( auto &liveTensor : schedOp->LiveRangeTensors() )
        {
            auto usage = liveTensor.first;
            auto tens = liveTensor.second;
            bool isRollingBuffer = cascadeBuffer != nullptr && usage == MakeTensorUsage(TensorUsage::IFM, schedOp->PrimaryIfmIndex());
            if ( ShouldBeIgnored(tens, targetMemory) && !(addRollingBuffers && isRollingBuffer) )
            {
                continue;
            }
            auto lr = GetOrCreateRange(tens);
            if ( tens->isGraphInput )
            {
                // Graph input must not be overwritten by preceding schedOps
                lr->MarkUsage(startTime);
            }
            if ( tens->isGraphOutput )
            {
                // Graph output must not be overwritten by following schedOps
                graphOutputRanges.push_back(lr);
            }
            if ( tens->isPersistent )
            {
                // Persistent tensors must be alive for the entire inference
                persistentRanges.push_back(lr);
            }
            lr->MarkUsage(timeToSet);
            if ( isRollingBuffer )
            {
                // This tensor is a rolling buffer in a cascade and the size of the LiveRange needs to be modified
                // for enabling temporal memory snapshots without modifying the original Tensor
                lr->size = cascadeBuffer->sizeBytes;
            }
        }
        if ( timeToSet == _currentTime )
        {
            _currentTime += 2;
        }
    }
    for ( auto lr : graphOutputRanges )
    {
        lr->MarkUsage(_currentTime, 1);
    }

    // Persistent tensor live-range is for entire inference
    for ( auto lr : persistentRanges )
    {
        lr->MarkUsage(0, EndTime());
    }
    ++_currentTime;
}

LiveRange *LiveRangeGraph::GetOrCreateRange(SchedulerTensor *tens)
{
    // Return the live range of the tensor (or any of its clones)
    const auto entry = _equivalenceIdToLr.find(tens->equivalenceId);
    if ( entry != _equivalenceIdToLr.end() )
    {
        entry->second->AddTensor(tens);
        return entry->second;
    }
    // No live range found for the tensor, create a new one
    auto lr = std::make_shared<LiveRange>(tens);
    _lrs.push_back(lr);
    _equivalenceIdToLr[tens->equivalenceId] = lr.get();
    return lr.get();
}

LiveRange *LiveRangeGraph::FuseRanges(SchedulerTensor *inTens, SchedulerTensor *outTens)
{
    assert(outTens->AllocationSizeBytes() <= inTens->AllocationSizeBytes());
    auto lr = GetOrCreateRange(inTens);
    lr->AddTensor(outTens);
    const auto entry = _equivalenceIdToLr.find(outTens->equivalenceId);
    if ( entry != _equivalenceIdToLr.end() )
    {
        // Live range already existed for outTens, move over tensors
        auto &lr2 = entry->second;
        lr->tensors.insert(lr2->tensors.begin(), lr2->tensors.end());
        lr2->tensors.clear();
        lr2->size = 0;
    }
    _equivalenceIdToLr[outTens->equivalenceId] = lr;
    return lr;
}

SchedulerTensor *LiveRangeGraph::ReusableIFM(const std::unique_ptr<SchedulerOperation> &schedOp, const MemArea &targetMemory)
{
    SchedulerTensor *reusableIfm = nullptr;
    if ( IsElementwise(schedOp->Type()) )
    {
        // Check if possible to merge ifm/ofm live ranges of elementwise op
        const auto ofmConn = schedOp->OFM();
        const auto ofmTens = ofmConn->tensor.get();

        if ( !ShouldBeIgnored(ofmTens, targetMemory) )
        {
            for ( const auto &[usage, ifmConn] : schedOp->inputs.pairs() )
            {
                const auto ifmTens = ifmConn.tensor.get();

                if ( IsIFM(usage) && !ifmTens->isGraphOutput && !ifmTens->isPersistent && !ofmTens->isPersistent &&
                     ifmTens->storageShape == ofmTens->storageShape && ifmTens->format == ofmTens->format &&
                     ifmTens->dataType == ofmTens->dataType && !ShouldBeIgnored(ifmTens, targetMemory) &&
                     ifmTens->consumers.size() == 1 && ofmTens->producers.size() == 1 )
                {
                    reusableIfm = ifmTens;
                    break;
                }
            }
        }
    }
    return reusableIfm;
}

bool LiveRangeGraph::ShouldBeIgnored(SchedulerTensor *tens, const MemArea &targetMemory)
{
    if ( tens == nullptr )
    {
        return true;
    }
    return tens->memArea != targetMemory;
}

}  // namespace regor
