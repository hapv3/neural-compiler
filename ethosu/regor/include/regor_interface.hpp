//
// SPDX-FileCopyrightText: Copyright 2022-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "regor_database.hpp"

#include <cstdint>

namespace regor
{

// Regor Binary Data interface
struct IRegorBlob
{
    virtual ~IRegorBlob() = default;
    virtual void AddRef() = 0;
    virtual void Release() = 0;
    virtual void *Map(int64_t &size) = 0;
    virtual void Unmap(void *p) = 0;
};

// Regor Reporting interface
struct IRegorReporting
{
    virtual ~IRegorReporting() = default;
    virtual IDatabase *OptimiserDatabase() = 0;
};

}  // namespace regor
