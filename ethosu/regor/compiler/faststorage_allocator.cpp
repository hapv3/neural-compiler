//
// SPDX-FileCopyrightText: Copyright 2021-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "compiler/faststorage_allocator.hpp"

#include "common/logging.hpp"

#include "architecture/architecture.hpp"
#include "common/vector_span.hpp"
#include "live_range.hpp"
#include "scheduler.hpp"

#include <cstdint>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

namespace regor
{

// FastStorageComponentAllocator
FastStorageComponentAllocator::FastStorageComponentAllocator(std::vector<int> *baseMemUsage,
    std::vector<int> *maxMemUsage, int stagingLimit, std::unordered_map<LiveRange *, int64_t> *elementAccessLrs) :
        _baseMemUsage(baseMemUsage),
        _maxMemUsage(maxMemUsage), _stagingLimit(stagingLimit), _elementAccessLrs(elementAccessLrs)
{
}

// Allocates live ranges. Outputs a vector that gives for each live range if it should be evicted or kept
void FastStorageComponentAllocator::Allocate(vector_span<LiveRange *> &lrs, std::vector<bool> &evicted)
{
    int sz = lrs.size();
    evicted.resize(sz);
    _lrs = lrs;
    _evicted = &evicted;
    _currEvicted.resize(sz);
    _bestScore = 0;
    AllocateExhaustive(0, 0);
    _evicted = nullptr;
}

// Exhaustive, recursive search, starting at the given index
void FastStorageComponentAllocator::AllocateExhaustive(int ix, int score)
{
    if ( ix >= _lrs.size() )
    {
        // Check if score is better (more access is better)
        if ( score > _bestScore || _bestScore == 0 )
        {
            // Best so far, remember this solution
            _bestScore = score;
            *_evicted = _currEvicted;
        }
        return;
    }

    auto lr = _lrs[ix];
    for ( int t = lr->startTime; t <= lr->endTime; ++t )
    {
        assert((*_baseMemUsage)[t] <= (*_maxMemUsage)[t]);
    }
    // Current peak usage during this live range
    int baseUsage = *std::max_element(&(*_baseMemUsage)[lr->startTime], &(*_baseMemUsage)[lr->endTime + 1]);
    bool canFit = baseUsage + lr->size <= _stagingLimit;
    bool alwaysFits = canFit;
    if ( canFit )
    {
        // Keep current lr
        int maxUsage = *std::max_element(&(*_maxMemUsage)[lr->startTime], &(*_maxMemUsage)[lr->endTime + 1]);
        // If alwaysFits is true, lr can be kept regardless of the allocation of the other lrs
        alwaysFits = maxUsage <= _stagingLimit;
        _currEvicted[ix] = false;
        int lrScore = 0;
        auto entry = _elementAccessLrs->find(lr);
        if ( entry != _elementAccessLrs->end() )
        {
            lrScore = int(entry->second);
        }
        UpdateMemUsage(_baseMemUsage, lr, true);

        AllocateExhaustive(ix + 1, score + lrScore);
        UpdateMemUsage(_baseMemUsage, lr, false);
    }
    if ( !alwaysFits )
    {
        // Evict current lr
        _currEvicted[ix] = true;
        UpdateMemUsage(_maxMemUsage, lr, false);
        AllocateExhaustive(ix + 1, score);
        UpdateMemUsage(_maxMemUsage, lr, true);
    }
}

void FastStorageComponentAllocator::UpdateMemUsage(std::vector<int> *memUsage, LiveRange *lr, bool increase)
{
    for ( int t = lr->startTime; t <= lr->endTime; ++t )
    {
        (*memUsage)[t] += increase ? lr->size : -lr->size;
        assert((*memUsage)[t] >= 0);
    }
}


// FastStorageAllocator

void FastStorageAllocator::AllocateFeatureMaps(const std::vector<std::unique_ptr<SchedulerOperation>> &schedOps,
    Schedule *schedule, const MemArea &fastStorage, Address stagingLimit)
{
    _stagingLimit = int(std::min(INT64_C(1) << 30, stagingLimit));
    // Force all OFMs to fast-storage (except final outputs)
    // _scratchedFms contains the original tensor MemArea (not fast storage) that tensors will be evicted to
    _scratchedFms.clear();
    for ( auto &schedOp : schedOps )
    {
        if ( !schedOp->IsNpuOp() )
        {
            continue;
        }
        auto opGroup = schedOp->OpGroup();
        assert(opGroup != nullptr);
        auto cost = schedule->Cost(schedOp.get());
        if ( cost->cascade == 0 )
        {
            SchedulerConnection *ofm = schedOp->OFM();
            if ( !ofm->tensor->consumers.empty() && !ofm->tensor->hasCPUReaders && !ofm->tensor->isGraphOutput &&
                 _scratchedFms.count(ofm->tensor.get()) == 0 && opGroup->NeedsAllocation(ofm->tensor->uid) )
            {
                LOG_TRACE1("Candidate fast storage tensor: {}\n", ofm->tensor->Name());
                _scratchedFms[ofm->tensor.get()] = ofm->tensor->memArea;
                ofm->tensor->memArea = fastStorage;
            }
            for ( auto &subOp : schedOp->SubOps() )
            {
                ofm = subOp->OFM();
                if ( !ofm->tensor->consumers.empty() && !ofm->tensor->hasCPUReaders && !ofm->tensor->isGraphOutput &&
                     _scratchedFms.count(ofm->tensor.get()) == 0 && opGroup->NeedsAllocation(ofm->tensor->uid) )
                {
                    LOG_TRACE1("Candidate fast storage tensor: {}\n", ofm->tensor->Name());
                    _scratchedFms[ofm->tensor.get()] = ofm->tensor->memArea;
                    ofm->tensor->memArea = fastStorage;
                }
            }
        }
    }

    auto lrGraph = LiveRangeGraph();
    lrGraph.ExtractLiveRangesFromCascades(schedOps, schedule, fastStorage, true);
    // Populate time-array with memory used by live ranges
    int maxUsage;
    _maxMemUsage = lrGraph.GetTemporalMemoryUsage(maxUsage);

    // Collect all live ranges that can potentially be in fast storage
    _baseMemUsage = _maxMemUsage;
    std::vector<LiveRange *> lrs;
    for ( auto lr : lrGraph.LiveRanges() )
    {
        for ( auto &tens : lr->tensors )
        {
            if ( _scratchedFms.count(tens) )
            {
                lrs.push_back(lr.get());
                for ( int t = lr->startTime; t <= lr->endTime; ++t )
                {
                    _baseMemUsage[t] -= lr->size;
                }
                break;
            }
        }
    }

    // Collect time indices of all CPU operators
    std::vector<int> cpuTimeIndices;
    for ( auto &schedOp : schedOps )
    {
        if ( !schedOp->IsNpuOp() )
        {
            auto cost = schedule->Cost(schedOp.get());
            cpuTimeIndices.push_back(cost->timeIndex);
        }
    }
    assert(std::is_sorted(cpuTimeIndices.cbegin(), cpuTimeIndices.cend()));

    // Evict live ranges that cross a CPU operator
    std::vector<LiveRange *> npuOnlyLrs;
    for ( auto lr : lrs )
    {
        auto cpuTimeIndex = std::lower_bound(cpuTimeIndices.begin(), cpuTimeIndices.end(), lr->startTime);
        if ( cpuTimeIndex != cpuTimeIndices.end() && *cpuTimeIndex <= lr->endTime )
        {
            // Live range crosses CPU operator
            LOG_TRACE1("Evicting cross-CPU live range {}-{}\n", lr->startTime, lr->endTime);
            Evict(lr);
        }
        else
        {
            npuOnlyLrs.push_back(lr);
        }
    }

    if ( maxUsage <= _stagingLimit )
    {
        // All feature maps fit in fast storage
        ElementwiseSanitizer(schedOps, schedule, fastStorage, lrGraph);
        return;
    }

    // Perform a first sweep to keep/evict live ranges that are obviously too big
    std::vector<LiveRange *> canFitLrs;
    for ( auto lr : npuOnlyLrs )
    {
        // Highest memory usage in this live range
        int baseUsage = *std::max_element(&_baseMemUsage[lr->startTime], &_baseMemUsage[lr->endTime + 1]);

        if ( baseUsage + lr->size > _stagingLimit )
        {
            // Cannot possibly fit
            Evict(lr);
        }
        else
        {
            canFitLrs.push_back(lr);
        }
    }
    std::vector<LiveRange *> competingLrs;
    for ( auto lr : canFitLrs )
    {
        maxUsage = *std::max_element(&_maxMemUsage[lr->startTime], &_maxMemUsage[lr->endTime + 1]);
        if ( maxUsage <= _stagingLimit )
        {
            // Definitively fits without impacting other feature maps
            Keep(lr);
        }
        else
        {
            competingLrs.push_back(lr);
        }
    }
    // For the remaining live ranges a choice must be made which to keep and which to evict.
    // Divide the live ranges in connected components and do a search for each component
    int sz = int(competingLrs.size());
    if ( sz == 0 )
    {
        ElementwiseSanitizer(schedOps, schedule, fastStorage, lrGraph);
        return;
    }

    // For every competing live range accumulate the total element access for the ranges.
    // Include all tensors access for a range - both read and write access.
    // A live range that is used within a cascade is given the highest score possible.
    // The reason is for cascaded elementwise operators where the other ifm (the cascade buffer)
    // is already in fast storage.
    // A live range with higher element access is considered more important to keep in
    // in fast storage.
    std::unordered_map<LiveRange *, int64_t> elementAccessLrs;
    for ( auto lr : competingLrs )
    {
        bool lrUsedWithinCascade = false;
        int64_t access = 0;
        for ( auto tens : lr->tensors )
        {
            // Look at readers
            for ( auto cons : tens->consumers )
            {
                auto *ifm = cons->IFM(0);
                auto *ifm2 = cons->TryIFM(1);
                auto consCost = schedule->Cost(cons);

                CascadeInfo *cascadeInfo =
                    consCost == nullptr || consCost->cascade == 0 ? nullptr : &schedule->cascades[consCost->cascade];

                if ( cascadeInfo && cons->Index() > cascadeInfo->start )
                {
                    lrUsedWithinCascade = true;
                    break;
                }

                if ( ifm->tensor->srcTensor == tens->srcTensor && consCost )
                {
                    access += consCost->elementAccess.ifmRead[0];
                }
                else if ( ifm2 && ifm2->tensor->srcTensor == tens->srcTensor && consCost )
                {
                    access += consCost->elementAccess.ifmRead[1];
                }
            }
            if ( !lrUsedWithinCascade )
            {
                // Look at writers
                for ( auto prod : tens->producers )
                {
                    auto cost = schedule->Cost(prod);
                    if ( cost == nullptr && prod->Parent() )
                    {
                        // Most likely a fused LUT, use cost from primary op
                        cost = schedule->Cost(prod->Parent());
                    }
                    if ( cost )
                    {
                        access += cost->elementAccess.ofmWrite;
                    }
                }
            }
            else
            {
                access = FastStorageComponentAllocator::MAX_ACCESS_SIZE;
            }
        }
        elementAccessLrs[lr] = access;
    }

    int start = 0;
    int startTime = competingLrs[0]->startTime;
    int endTime = competingLrs[0]->endTime;
    FastStorageComponentAllocator componentAllocator(&_baseMemUsage, &_maxMemUsage, _stagingLimit, &elementAccessLrs);

    // Calculate and allocate connected components
    for ( int i = 1; i < sz; ++i )
    {
        auto lr = competingLrs[i];
        if ( lr->startTime <= endTime && i - start <= MAX_COMPONENT_SIZE )
        {
            // Add to existing component
            startTime = std::min(startTime, lr->startTime);
            endTime = std::max(endTime, lr->endTime);
        }
        else
        {
            // lr is start of a new component; allocate the current component
            vector_span<LiveRange *> span(competingLrs, start, i);
            AllocateComponent(componentAllocator, span);
            // Start a new component
            start = i;
            startTime = lr->startTime;
            endTime = lr->endTime;
        }
    }
    vector_span<LiveRange *> span(competingLrs, start, sz);
    AllocateComponent(componentAllocator, span);
    ElementwiseSanitizer(schedOps, schedule, fastStorage, lrGraph);
}

// Allocates a connected range of live ranges
void FastStorageAllocator::AllocateComponent(FastStorageComponentAllocator &allocator, vector_span<LiveRange *> &lrs)
{
    std::vector<bool> evicted;
    int sz = lrs.size();
    allocator.Allocate(lrs, evicted);
    assert(sz == int(evicted.size()));
    for ( int i = 0; i < sz; ++i )
    {
        if ( evicted[i] )
        {
            Evict(lrs[i]);
        }
        else
        {
            Keep(lrs[i]);
        }
    }
}

void FastStorageAllocator::ElementwiseSanitizer(const std::vector<std::unique_ptr<SchedulerOperation>> &schedOps,
    Schedule *schedule, const MemArea &fastStorage, LiveRangeGraph &lrGraph)

{
    // For now - enforce that both ifm's should be in the same memory for elementwise
    for ( auto &schedOp : schedOps )
    {
        if ( !schedOp->IsNpuOp() )
        {
            continue;
        }

        if ( IsBinaryElementwise(schedOp->_type) )
        {
            auto *ifm = schedOp->IFM(0);
            auto *ifm2 = schedOp->TryIFM(1);
            auto consCost = schedule->Cost(schedOp.get());

            CascadeInfo *cascadeInfo =
                consCost == nullptr || consCost->cascade == 0 ? nullptr : &schedule->cascades[consCost->cascade];

            if ( cascadeInfo && schedOp->Index() > cascadeInfo->start )
                // Within cascade there is nothing to do, since cascade buffer is in fast storage
                continue;

            if ( ifm2 && ifm2->tensor->memArea != ifm->tensor->memArea )
            {
                // One ifm not in fast storage
                if ( ifm->tensor->memArea == fastStorage && !ifm2->tensor->IsConstant() )
                {
                    // Ifm in fast storage and ifm2 is not a constant
                    auto lr = lrGraph.GetOrCreateRange(ifm->tensor.get());
                    Evict(lr);
                }

                if ( ifm2->tensor->memArea == fastStorage && !ifm->tensor->IsConstant() )
                {
                    // Ifm2 in fast storage and ifm is not a constant
                    auto lr = lrGraph.GetOrCreateRange(ifm2->tensor.get());
                    Evict(lr);
                }
            }
        }
    }
}

void FastStorageAllocator::Evict(LiveRange *lr)
{
    for ( int t = lr->startTime; t <= lr->endTime; ++t )
    {
        _maxMemUsage[t] -= lr->size;
    }
    for ( auto &tens : lr->tensors )
    {
        auto entry = _scratchedFms.find(tens);
        if ( entry != _scratchedFms.end() )
        {
            tens->memArea = entry->second;
        }
    }
}

void FastStorageAllocator::Keep(LiveRange *lr)
{
    for ( int t = lr->startTime; t <= lr->endTime; ++t )
    {
        _baseMemUsage[t] += lr->size;
        assert(_baseMemUsage[t] <= _stagingLimit);
    }
}

}  // namespace regor
