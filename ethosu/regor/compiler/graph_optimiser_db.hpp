//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "database.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace regor
{

class Operation;

/// <summary>
/// Graph optimiser database implementation
/// </summary>
class OptimiserDatabase
{
private:
    Database *_db = nullptr;
    int _sourceId = 0;
    int _optId = 0;
    int _streamId = 0;
    int _sourceTable = 0;
    int _optTable = 0;
    int _groupTable = 0;
    int _cmdTable = 0;
    int _streamTable = 0;
    std::unordered_map<const void *, int> _source;
    std::unordered_map<const void *, std::tuple<int, int>> _optimised;

public:
    OptimiserDatabase(Database *db);
    Database *Get();
    int SourceId(const void *op);
    int OptimisedId(const void *op);
    int SourceOp(const Operation *op, int ext_key = -1);
    void AddOptimised(const void *from, const Operation *to);
    void AddSubOp(UniqueId primaryUid, UniqueId subOpUid);
    void AddCommand(void *key, int stream, int cmdIndex, UniqueId id);
    int AddStream();
};

}  // namespace regor
