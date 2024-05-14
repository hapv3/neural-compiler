//
// SPDX-FileCopyrightText: Copyright 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "graph_validator.hpp"
#include "tosa/tosa_validator.hpp"

#include <functional>

namespace regor
{

class TosaGraphValidator : public GraphValidator
{
    tosa::validator::Context _context;

public:
    TosaGraphValidator(GraphNotation notation, uint32_t syntaxVersion, Compiler *compiler);
    static bool HandlesSyntax(uint32_t syntaxVersion);
    bool Validate(Graph *graph) override;
};

}  // namespace regor
