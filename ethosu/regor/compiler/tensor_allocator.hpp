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

#include "architecture/architecture.hpp"
#include "scheduler.hpp"
#include "scheduler_operation.hpp"

#include <cstdint>
#include <vector>

namespace regor
{

class LiveRangeGraph;

// Tensor allocation algorithms
enum class TensorAllocator : uint16_t
{
    // Allocator that does not reuse memory
    LinearAlloc = 0,
    // Search based allocator
    HillClimb = 1,
    Last,
};

/// <summary>
/// Linear allocator that can be used to allocate addresses across multiple subgraphs
/// </summary>
class IncrementalLinearAllocator
{
public:
    IncrementalLinearAllocator(const std::string &name) : _name(name) {}
    Address Allocate(LiveRangeGraph *lrGraph, int alignment, bool verboseAllocation);

private:
    std::string _name;
    // Map from tensor's equivalence id to allocated address
    std::unordered_map<UniqueId, Address> _allocatedAddresses;
    Address _highestAddress = 0;
};

// Allocates addresses to the tensors involved in the given operations/mem area(s)
// using the given tensor allocation algorithm.
void AllocateTensors(const std::vector<std::unique_ptr<SchedulerOperation>> &schedOps, Schedule *schedule, const MemArea &memArea,
    TensorAllocator allocator, int alignment, bool verboseAllocation, Address sizeLimit = std::numeric_limits<Address>::max());

}  // namespace regor
