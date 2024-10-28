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

#include "common/common.hpp"

#include "compiler/database.hpp"
#include "compiler/graph_optimiser.hpp"
#include "compiler/scheduler.hpp"

#include <unordered_set>

namespace regor
{

/// <summary>
/// Performance information for a whole schedule
/// </summary>
enum class AccessType
{
    Lut = 0,
    FeatureMap = 1,
    Weights = 2,
    Scales = 3,
    Last,
};

struct PerformanceResult
{

    struct MemoryAccess
    {
        int64_t bytesRead = 0;
        int64_t bytesWritten = 0;
        int64_t accessCycles = 0;

        MemoryAccess &operator+=(const MemoryAccess &other)
        {
            this->bytesRead += other.bytesRead;
            this->bytesWritten += other.bytesWritten;
            this->accessCycles += other.accessCycles;
            return *this;
        }
    };

    struct MemoryAccesses
    {
        std::unordered_map<AccessType, MemoryAccess> access;
        int64_t peakUsage = 0;
        int64_t readTransferOverhead = 0;
        int64_t writeTransferOverhead = 0;
        float readTransferEff = 1;
        float writeTransferEff = 1;
        int64_t AccessCycles() const
        {
            int64_t cycles = 0;
            for ( const auto &[type, acc] : access )
            {
                cycles += acc.accessCycles;
            }
            return cycles;
        }

        MemoryAccesses &operator+=(const MemoryAccesses &other)
        {
            for ( const auto &[type, acc] : other.access )
            {
                access[type] += acc;
            }
            peakUsage = std::max(peakUsage, other.peakUsage);
            return *this;
        }
    };

    std::unordered_map<ArchitectureMemory *, MemoryAccesses> memory;
    int64_t npuCycles = 0;
    int64_t cpuCycles = 0;
    int64_t totalCycles = 0;
    int64_t macCount = 0;
    int64_t cpuOps = 0;
    int64_t npuOps = 0;
    int64_t cascadedOps = 0;
    int64_t cascades = 0;
    int64_t originalWeights = 0;
    int64_t encodedWeights = 0;

    int Accesses() const
    {
        int accesses = 0;
        for ( const auto &[archMem, stats] : memory )
        {
            accesses += int(stats.access.size());
        }
        return accesses;
    }

    PerformanceResult &operator+=(const PerformanceResult &other)
    {
        // Not ideal for performance
        for ( const auto &[arch, memoryStat] : other.memory )
        {
            memory[arch] += memoryStat;
        }
        this->npuCycles += other.npuCycles;
        this->cpuCycles += other.cpuCycles;
        this->totalCycles += other.totalCycles;
        this->macCount += other.macCount;
        this->cpuOps += other.cpuOps;
        this->npuOps += other.npuOps;
        this->cascadedOps += other.cascadedOps;
        this->cascades += other.cascades;
        this->originalWeights += other.originalWeights;
        this->encodedWeights += other.encodedWeights;
        return *this;
    }
};

/// <summary>
/// Whole-schedule performance calculation module
/// </summary>
class NetworkPerformance
{
private:
    Architecture *_arch;
    const std::vector<std::unique_ptr<SchedulerOperation>> &_ops;

public:
    NetworkPerformance(Architecture *arch, const std::vector<std::unique_ptr<SchedulerOperation>> &ops);

public:
    PerformanceResult Measure(Schedule *schedule, OptimiserDatabase *optDb);

private:
    PerformanceResult ProcessOpPerformance(SchedulerOperation *schedOp, SchedulerOpInfo *cost, Schedule *schedule,
        SchedulerOperation *prevOp, SchedulerOpInfo *prevCost, const std::unordered_set<ArchitectureMemory *> &memories);
    PerformanceResult EstimateFullOpPerformance(
        SchedulerOperation *schedOp, SchedulerOpInfo *cost, SchedulerOperation *prevOp, SchedulerOpInfo *prevCost);
    void AddToDatabase(const PerformanceResult &perf, SchedulerOperation *schedOp, SchedulerOpInfo *cost, int opTable,
        int perfDebugTable, const std::unordered_set<ArchitectureMemory *> &memories, OptimiserDatabase *optDb);
};



}  // namespace regor
