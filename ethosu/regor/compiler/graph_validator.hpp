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

#include "graph.hpp"

#include <string>
#include <vector>

#include "include/regor.h"

namespace regor
{

class Compiler;

class GraphValidator
{
public:
    struct Error
    {
        OpType operation;
        std::string errorMessage;
    };

protected:
    GraphNotation _notation;
    uint32_t _syntaxVersion;
    std::vector<Error> _validationErrors;

public:
    GraphValidator(GraphNotation notation, uint32_t syntaxVersion) : _notation(notation), _syntaxVersion(syntaxVersion)
    {
    }
    virtual ~GraphValidator() = default;

    static std::unique_ptr<GraphValidator> MakeGraphValidator(GraphNotation notation, uint32_t syntaxVersion, Compiler *compiler);
    virtual bool Validate(Graph *graph);
    std::vector<Error> &GetErrors() { return _validationErrors; }
    std::string GetErrorMsg();
};

}  // namespace regor
