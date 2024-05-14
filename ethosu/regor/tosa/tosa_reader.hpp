//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "tosa_schema_generated.hpp"

#include <list>

namespace regor
{

class GraphBuilder;

// Parses a Tosa flatbuffer to create a Builder.
class TosaReader
{
public:
    TosaReader() {}

    static void LoadGraphs(const tosaFb::TosaGraph *model, std::list<GraphBuilder> &builders);
    static void LoadGraphs(const void *input, size_t size, std::list<GraphBuilder> &builders);  // From buffer
};

}  // namespace regor
