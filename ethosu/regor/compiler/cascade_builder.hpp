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

#include "common/common.hpp"

#include "common/shape.hpp"
#include "common/vector_span.hpp"
#include "op_type.hpp"
#include "scheduler_operation.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace regor
{

class Schedule;

/// <summary>
/// Information about a cascade buffer
/// </summary>
struct CascadeBuffer
{
    Shape shape;
    int sizeBytes = 0;
    CascadeBuffer() = default;
    CascadeBuffer(const CascadeBuffer &) = default;
    CascadeBuffer(const Shape &s, int size) : shape(s), sizeBytes(size) {}
    CascadeBuffer &operator=(const CascadeBuffer &) = default;
};


/// <summary>
/// Information about a cascade within a schedule
/// </summary>
struct CascadeInfo
{
    int start = 0;
    int end = 0;
    int memUsage = 0;
    std::unordered_map<UniqueId, CascadeBuffer> buffers;

    CascadeInfo() = default;
    CascadeInfo(const CascadeInfo &) = default;
    CascadeInfo(int start_, int end_, int memUsage_, std::unordered_map<UniqueId, CascadeBuffer> buffers_)
    {
        this->start = start_;
        this->end = end_;
        this->memUsage = memUsage_;
        this->buffers = std::move(buffers_);
    }
    CascadeInfo &operator=(const CascadeInfo &) = default;
};


class SchedulerOpInfo;

/// <summary>
/// Cascade builder for lists of scheduler operations
/// </summary>
class CascadeBuilder
{
private:
    vector_span<std::unique_ptr<SchedulerOperation>> _ops;
    const std::unordered_map<UniqueId, int> &_nonLocalMemUsage;
    bool _spilling = false;

public:
    CascadeBuilder(vector_span<std::unique_ptr<SchedulerOperation>> ops,
        const std::unordered_map<UniqueId, int> &nonLocalMemUsage, bool spilling);

public:
    void BuildCascades(Schedule *refSchedule, Schedule *fallbackSchedule, Address guidingStagingLimit);

private:
    bool IsCascadable(const SchedulerOperation *op, SchedulerConnection *ifmConn, SchedulerOpInfo *cost) const;
    int EstimateBufferUsage(SchedulerOperation *op, SchedulerOpInfo *cost) const;
    int NonLocalUsage(UniqueId uid) const;
};

}  // namespace regor
