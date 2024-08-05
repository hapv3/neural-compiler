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
#include "live_range.hpp"

#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <vector>

namespace regor
{

struct HillClimbLiveRange
{
    // Start time, input to the allocator
    int startTime;
    // End time (inclusive), input to the allocator
    int endTime;
    // Size, input to the allocator
    int size;
    // Allocated address, output of the allocator
    Address address;
    // End address, exclusive
    Address endAddress;
    // Index of this live range
    int id;
    // id of predecessor live range (predecessor's end address == this lr's address)
    int predecessor;
    // Turn at which the live range was allocated
    int turn;

    bool Overlaps(Address addr2, Address size2) const { return address < addr2 + size2 && addr2 < endAddress; }
    bool IsNeighbour(const HillClimbLiveRange &lr) const { return startTime <= lr.endTime && lr.startTime <= endTime; }
};

// Implementation of a tensor allocator using state space exploration.
//
// The basic algorithm is:
//
// - Use a heuristic allocator to find an initial allocation
// - while allocation is not optimal and iterations < MAX_ITERATIONS:
//     - find the "bottleneck": the live range with highest end address
//     - find all live ranges that affected the allocation of the bottleneck
//     - swap the order of any two affecting live ranges
//     - reallocate tensors using the reordered live ranges
//     - if the new allocation is better: keep it, else set allocation to previous allocation
class HillClimbAllocator
{
private:
    static constexpr int MAX_ITERATIONS = 99999;
    // Special handling if best solution has not improved during this many iterations
    static constexpr int MAX_ITERATIONS_STUCK = 25;
    // Minimum number of iterations since the last improvement (unless an optimal solution is found)
    static constexpr int MIN_ITERATIONS_IMPROVE = 5000;
    // Used for live ranges allocated at address 0
    static constexpr int NO_PREDECESSOR = -1;
    // Contains the live ranges
    std::vector<HillClimbLiveRange> lrs;
    // Contains active live ranges at each timestamp
    std::vector<std::vector<HillClimbLiveRange *>> lrsAtTime;
    //
    // Contains neighbours of each live range (indexed by lr.id), i.e.
    // live ranges with overlapping start/end time.
    std::vector<std::vector<HillClimbLiveRange *>> neighbours;
    //
    // At each timestamp: accumulated size of active live ranges
    std::vector<Address> sizeAtTime;
    //
    // For each live range: max value of sizeAtTime (only used in the heuristic allocation)
    std::vector<Address> lrUrgency;
    //
    // The maximum allowed size (the size of the physical available memory)
    Address maxAllowedSize = 0;
    // The minimum possible size, assuming all live ranges can be perfectly allocated
    Address minRequiredSize = 0;
    // The algorithm stops once the target size has been achieved
    Address targetSize = 0;
    // The highest end address of the best found allocation
    Address bestSize = 0;
    // Number of performed iterations
    int iterations = 0;
    // Random number generator; use default seed (which is well-defined)
    std::mt19937 rng;

public:
    // Runs the allocation algorithm and updates the address field of lrs.
    // Finishes when the target size has been reached or when maximum iterations have been run.
    //
    // Implementation note: the algorithm produces reproducible results by using
    // a well-defined random number generator with well-defined default seed,
    // and using a fixed number of iterations.
    Address Allocate(const std::vector<std::shared_ptr<LiveRange>> &lrs, int alignment, Address sizeLimit);

    Address MinimumRequiredSize() const { return minRequiredSize; }
    int Iterations() const { return iterations; }

private:
    void SetLiveRanges(const std::vector<std::shared_ptr<LiveRange>> &liveRanges, int alignment);

    // Allocates the given live range at the smallest possible address
    void AllocateLr(HillClimbLiveRange &lr) const;
    //
    // Allocates the live ranges in the order indicated by the indices;
    // allocates each live range at the lowest possible address.

    Address AllocateIndices(const std::vector<int> &indices);

    // Sorts live ranges based on heuristics, used for the initial allocation
    void SortIndicesOnPrio(std::vector<int> &indices) const;

    // Adds the given live range + predecessors to the turns vector
    void AddPredecessorTurns(std::set<int> &turns, const HillClimbLiveRange &lr) const;

    // Finds the "bottleneck", the live range with highest end address, and reorders the indices
    // such that a next allocation might lower the memory usage.
    //
    //                          ---------
    //                          |       |
    //                          |   D   |
    //                          |       |
    // ----------------------------------
    // |           B                 |
    // -------------------------------
    // | |
    // |A|                      ---
    // | |                      |C|
    // | |                      | |
    // ---------------------------------------
    //
    // In the above example, the allocation order was [A, B, C, D] and D is the resulting bottle-neck.
    // The live ranges that affected the allocation of D are the direct neighbours of D (i.e. B and C),
    // and all direct and indirect predecessors of D and its neighbours
    // (i.e. A, which is the predecessor of B, and indirect predecessor of D).
    //
    // By permuting the order in which the affecting live ranges are allocated, the bottleneck might
    // be lowered. In the above example, almost any permutation would lower the bottleneck.
    //
    // Note that there is room to improve the efficiency of the algorithm.
    // One way could be to first allocate all direct neighbours of the bottleneck
    // (i.e. B, C, D) and then the other affecting live ranges (i.e. A). The algorithm currently does
    // not actively try this, as it may lead to allocation loops (A could become the new bottle-neck);
    // it just uses a higher probability of selecting A.
    void AttemptBottleneckFix(std::vector<int> &indices, int iterationsStuck);

    // Search for a solution, using the given indices as initial solution.
    void Search(std::vector<int> &indices, int iterations);
};

// Wrapper function to perform live range allocation
Address HillClimbAllocateLiveRanges(LiveRangeGraph &lrGraph, int alignment, Address sizeLimit);

}  // namespace regor
