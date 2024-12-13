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

#pragma once

#include "compiler/high_level_command_stream.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace regor
{

using CmdRanges = std::vector<std::tuple<UniqueId, int, int>>;

class IRegisterCommandStreamGenerator
{
public:
    virtual ~IRegisterCommandStreamGenerator() = default;
    virtual std::vector<uint32_t> GenerateCommandStream(
        std::vector<std::unique_ptr<HighLevelCommand>> &highLevelCommandStream, CmdRanges *cmdRanges, bool verbose) = 0;
    virtual void PrintCommandStream(const std::vector<uint32_t> &stream, std::vector<std::pair<unsigned, std::string>> &debugInfo) = 0;
};

}  // namespace regor
