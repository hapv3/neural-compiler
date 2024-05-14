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

// Allocates a connected set of live ranges to fast storage.
//
// The allocator attempts to maximize the sum of the sizes of all feature maps
// that are placed in fast storage.
class FastStorageComponentAllocator
{
private:
    // Memory usage per timestamp when no lrs are kept
    std::vector<int> *_baseMemUsage = nullptr;
    // Memory usage per timestamp when all lrs are kept
    std::vector<int> *_maxMemUsage = nullptr;

    Address _stagingLimit;
    vector_span<LiveRange *> _lrs;
    // Indices of evicted lrs in best solution
    std::vector<bool> *_evicted = nullptr;
    // Indices of evicted lrs in current solution
    std::vector<bool> _currEvicted;
    int _bestScore = 0;
    std::unordered_map<LiveRange *, int64_t> *_elementAccessLrs = nullptr;
    // Use default seed (which is well-defined) to guarantee reproducible results
    std::mt19937 _rng;

public:
    static constexpr int MAX_ACCESS_SIZE = std::numeric_limits<int>::max();

    FastStorageComponentAllocator(std::vector<int> *baseMemUsage, std::vector<int> *maxMemUsage, int stagingLimit,
        std::unordered_map<LiveRange *, int64_t> *elementAccessLrs);
    // Allocates live ranges. Outputs a vector that gives for each live range if it should be evicted or kept
    void Allocate(vector_span<LiveRange *> &lrs, std::vector<bool> &evicted);

private:
    // Exhaustive, recursive search, starting at the given index
    void AllocateExhaustive(int ix, int score);
    void UpdateMemUsage(std::vector<int> *memUsage, LiveRange *lr, bool increase);
};

// Allocates feature maps to fast storage
class FastStorageAllocator
{
private:
    static constexpr int64_t MAX_COMPONENT_SIZE = 20;
    // Remembers feature map's memory before it was allocated to fast storage
    std::unordered_map<SchedulerTensor *, MemArea> _scratchedFms;
    // Memory usage with all feature maps in fast storage
    std::vector<int> _maxMemUsage;
    // Memory usage without feature maps that still need to be allocated
    std::vector<int> _baseMemUsage;
    int _stagingLimit = 0;

public:
    void AllocateFeatureMaps(const std::vector<std::unique_ptr<SchedulerOperation>> &schedOps, Schedule *schedule,
        const MemArea &fastStorage, Address stagingLimit);

private:
    // Allocates a connected range of live ranges
    void AllocateComponent(FastStorageComponentAllocator &allocator, vector_span<LiveRange *> &lrs);
    void ElementwiseSanitizer(const std::vector<std::unique_ptr<SchedulerOperation>> &schedOps, Schedule *schedule,
        const MemArea &fastStorage, LiveRangeGraph &lrGraph);
    void Evict(LiveRange *lr);
    void Keep(LiveRange *lr);
};

}  // namespace regor
