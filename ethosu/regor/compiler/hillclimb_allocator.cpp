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

#include "hillclimb_allocator.hpp"

#include "common/numeric_util.hpp"

#include <algorithm>
#include <cstdint>
#include <queue>
#include <set>
#include <utility>
#include <vector>

namespace regor
{

constexpr Address MAX_ADDRESS = std::numeric_limits<Address>::max();
constexpr Address NOT_ALLOCATED = -1;

void HillClimbAllocator::SetLiveRanges(const std::vector<std::shared_ptr<LiveRange>> &liveRanges, int alignment)
{
    int maxEndTime = 0;
    int id = 0;
    for ( const auto &lr : liveRanges )
    {
        HillClimbLiveRange hlr = {};

        hlr.startTime = lr->startTime;
        hlr.endTime = lr->endTime;
        hlr.size = RoundAway(lr->size, alignment);
        hlr.id = id;
        maxEndTime = std::max(maxEndTime, lr->endTime);
        lrs.push_back(hlr);
        ++id;
    }
    lrsAtTime.resize(maxEndTime + 1);
    sizeAtTime.resize(maxEndTime + 1);
    neighbours.resize(lrs.size());
    // Calculate which live ranges are active at every timestamp
    for ( int t = 0; t <= maxEndTime; ++t )
    {
        lrsAtTime[t].clear();
    }
    for ( auto &lr : lrs )
    {
        for ( auto t = lr.startTime; t <= lr.endTime; ++t )
        {
            lrsAtTime[t].push_back(&lr);
        }
    }
    minRequiredSize = 0;
    for ( int t = 0; t <= maxEndTime; ++t )
    {
        // Calculate minimum needed size at each timestamp
        Address neededSize = 0;
        for ( auto &lr : lrsAtTime[t] )
        {
            neededSize += lr->size;
        }
        sizeAtTime[t] = neededSize;
        minRequiredSize = std::max(neededSize, minRequiredSize);
        // Calculate all neighbours
        for ( unsigned i = 0; i < lrsAtTime[t].size(); ++i )
        {
            auto lr1 = lrsAtTime[t][i];
            auto &nb1 = neighbours[lr1->id];
            for ( auto j = i + 1; j < lrsAtTime[t].size(); ++j )
            {
                auto lr2 = lrsAtTime[t][j];
                if ( find(nb1.begin(), nb1.end(), lr2) == nb1.end() )
                {
                    nb1.push_back(lr2);
                    neighbours[lr2->id].push_back(lr1);
                }
            }
        }
    }
    targetSize = minRequiredSize;
    // Calculate the urgency of each live range
    lrUrgency.resize(lrs.size());
    for ( unsigned i = 0; i < lrs.size(); ++i )
    {
        auto &lr = lrs[i];
        Address urgency = 0;
        for ( auto t = lr.startTime; t <= lr.endTime; ++t )
        {
            urgency = std::max(sizeAtTime[t], urgency);
        }
        lrUrgency[i] = urgency;
    }
}

Address HillClimbAllocator::Allocate(const std::vector<std::shared_ptr<LiveRange>> &liveRanges, int alignment, Address sizeLimit)
{
    SetLiveRanges(liveRanges, alignment);
    maxAllowedSize = sizeLimit;
    iterations = 0;
    std::vector<int> indices;
    int sz = int(liveRanges.size());
    // Initial solution, using a heuristic allocator
    for ( int i = 0; i < sz; ++i )
    {
        indices.push_back(i);
    }
    SortIndicesOnPrio(indices);
    // Allocate the initial solution
    bestSize = MAX_ADDRESS;
    bestSize = AllocateIndices(indices);
    if ( bestSize <= targetSize )
    {
        // The heuristic allocation returned an optimal solution.
        // No need to search.
    }
    else
    {
        // Try to improve the heuristic allocation
        Search(indices, MAX_ITERATIONS);
    }
    // Allocate addresses
    for ( int i = 0; i < sz; ++i )
    {
        liveRanges[i]->SetAddress(lrs[i].address);
    }
    return bestSize;
}

void HillClimbAllocator::AllocateLr(HillClimbLiveRange &lr) const
{
    Address address = 0;
    int predecessor = NO_PREDECESSOR;
    bool fits = false;
    while ( !fits )
    {
        fits = true;
        // Find neighbours that overlap with address
        for ( auto lr2_p : neighbours[lr.id] )
        {
            if ( lr2_p->address == NOT_ALLOCATED || lr2_p->endAddress <= address )
            {
                continue;
            }
            if ( lr2_p->Overlaps(address, lr.size) )
            {
                // Overlap found; increase address
                fits = false;
                address = lr2_p->endAddress;
                predecessor = lr2_p->id;
            }
        }
    }
    lr.address = address;
    lr.endAddress = address + lr.size;
    lr.predecessor = predecessor;
}

Address HillClimbAllocator::AllocateIndices(const std::vector<int> &indices)
{
    ++iterations;
    int sz = int(indices.size());
    std::vector<int> count(sz);
    for ( auto &lr : lrs )
    {
        lr.address = NOT_ALLOCATED;
    }
    Address size = 0;
    for ( int turn = 0; size <= bestSize && turn < sz; ++turn )
    {
        auto &lr = lrs[indices[turn]];
        AllocateLr(lr);
        lr.turn = turn;
        size = std::max(size, lr.endAddress);
    }
    return size;
}

void HillClimbAllocator::SortIndicesOnPrio(std::vector<int> &indices) const
{
    std::sort(indices.begin(), indices.end(),
        [&lrUrgency_ = std::as_const(lrUrgency), &lrs_ = std::as_const(lrs)](int const &a, int const &b)
        {
            // urgent first
            if ( lrUrgency_[a] != lrUrgency_[b] )
            {
                return lrUrgency_[a] > lrUrgency_[b];
            }
            auto &lr1 = lrs_[a];
            auto &lr2 = lrs_[b];
            // long duration before short duration
            auto duration1 = lr1.endTime - lr1.startTime;
            auto duration2 = lr2.endTime - lr2.startTime;
            if ( duration1 != duration2 )
            {
                return duration1 > duration2;
            }
            if ( lr1.startTime != lr2.startTime )
            {
                return lr1.startTime < lr2.startTime;
            }
            if ( lr1.size != lr2.size )
            {
                return lr1.size > lr2.size;
            }
            return lr1.id < lr2.id;
        });
}

void HillClimbAllocator::AddPredecessorTurns(std::set<int> &turns, const HillClimbLiveRange &lr) const
{
    turns.insert(lr.turn);
    int id = lr.id;
    while ( lrs[id].predecessor != NO_PREDECESSOR )
    {
        id = lrs[id].predecessor;
        turns.insert(lrs[id].turn);
    }
}

void HillClimbAllocator::AttemptBottleneckFix(std::vector<int> &indices, int iterationsStuck)
{
    // Find the bottleneck
    HillClimbLiveRange *maxLr = &lrs[0];
    for ( auto &lr : lrs )
    {
        if ( lr.endAddress > maxLr->endAddress )
        {
            maxLr = &lr;
        }
    }
    // Find all live ranges that affected the placement of the bottleneck live range.
    // This consists of two types of live ranges:
    // - direct neighbours of the bottleneck live range
    // - direct and indirect predecessors of these neighbours + bottleneck
    // The turns at which these live ranges were allocated are put in the turns vector.
    std::set<int> turns;
    AddPredecessorTurns(turns, *maxLr);
    for ( auto lr_p : neighbours[maxLr->id] )
    {
        AddPredecessorTurns(turns, *lr_p);
    }
    // Non-direct neighbours that interfere with the allocation of the bottleneck are the
    // immediate cause for gaps in the allocation, and are selected with higher probability.
    std::vector<int> turnList;
    std::vector<int> nonNbTurnList;
    for ( auto turn : turns )
    {
        turnList.push_back(turn);
        auto &lr = lrs[indices[turn]];
        if ( !maxLr->IsNeighbour(lr) )
        {
            nonNbTurnList.push_back(turn);
        }
    }
    // Pick from non-neighbour list with 30% probability (magic number based on tuning)
    int ix1;
    using dist = std::uniform_int_distribution<size_t>;
    if ( dist(0, 100)(rng) < 30 && !nonNbTurnList.empty() )
    {
        // Pick a live range from the "non-neighbour list"
        ix1 = nonNbTurnList[dist(0, nonNbTurnList.size() - 1u)(rng)];
    }
    else
    {
        // Pick any affecting live range.
        ix1 = turnList[dist(0, turnList.size() - 1u)(rng)];
    }
    // Note: turnList has always at least 2 elements for bottlenecks
    int ix2 = turnList[dist(0, turnList.size() - 1u)(rng)];
    if ( ix1 == ix2 )
    {
        ix2 = turnList[turnList.size() - 1u];
    }
    // Swap indices
    std::swap(indices[ix1], indices[ix2]);
    if ( iterationsStuck > MAX_ITERATIONS_STUCK )
    {
        // The best allocation has not improved for a while, maybe improvement is not possible
        // by single-swapping indices; add more neighbour live ranges and swap 2 more indices.
        // Adding more neighbours can sometimes resolve the situation where the current bottleneck
        // is resolved, but always results in a higher bottleneck at a nearby live range.
        // Magic number is based on tuning
        std::unordered_set<int> visited;  // Set contains LR IDs
        std::queue<int> pending;          // Queue contains LR IDs
        pending.push(maxLr->id);
        while ( pending.size() > 0 )
        {
            int id = pending.front();
            pending.pop();
            if ( visited.count(id) ) continue;
            visited.insert(id);
            assert(id >= 0 && id < int(neighbours.size()));
            for ( auto lr_p : neighbours[id] )
            {
                if ( turns.count(lr_p->turn) == 0 )
                {
                    turns.insert(lr_p->turn);
                    turnList.push_back(lr_p->turn);
                }
                pending.push(lr_p->id);
            }
        }
        ix1 = turnList[dist(0, turnList.size() - 1)(rng)];
        ix2 = turnList[dist(0, turnList.size() - 1)(rng)];
        if ( ix1 == ix2 )
        {
            ix2 = turnList[turnList.size() - 1];
        }
        std::swap(indices[ix1], indices[ix2]);
    }
}

void HillClimbAllocator::Search(std::vector<int> &indices, int iters)
{
    std::vector<int> bestIndices = indices;
    std::vector<HillClimbLiveRange> bestLrs = lrs;
    int lastImprovementIteration = 0;

    for ( int i = 0; i < iters; ++i )
    {
        // Reorder the indices
        AttemptBottleneckFix(indices, i - lastImprovementIteration);
        // Allocate the reordered indices and check if it gave an improvement
        auto newSize = AllocateIndices(indices);
        if ( newSize <= bestSize )
        {
            // The new allocation produced a new best result; remember it
            if ( newSize < bestSize )
            {
                lastImprovementIteration = i;
            }
            bestSize = newSize;
            bestIndices = indices;
            bestLrs = lrs;
            if ( bestSize <= targetSize )
            {
                // Target reached; stop
                return;
            }
        }
        else
        {
            // The new allocation produced worse result; undo the change
            indices = bestIndices;
            lrs = bestLrs;
        }
        if ( (bestSize <= maxAllowedSize) && (i - lastImprovementIteration > MIN_ITERATIONS_IMPROVE) )
        {
            // A Solution has been found and solution hasn't improved in the last MIN_ITERATIONS_IMPROVE iterations;
            // stop
            return;
        }
    }
    lrs = std::move(bestLrs);
}

Address HillClimbAllocateLiveRanges(LiveRangeGraph &lrGraph, int alignment, Address sizeLimit)
{
    HillClimbAllocator allocator;
    return allocator.Allocate(lrGraph.LiveRanges(), alignment, sizeLimit);
}

}  // namespace regor
