//
// SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "tosa_graph_validator.hpp"

#include "compiler.hpp"

#include <optional>

namespace
{

std::optional<uint32_t> MaybeGetTosaVersion(uint32_t syntaxVersion)
{
    if ( syntaxVersion == 0 ) syntaxVersion = (GraphApi::VERSION_TOSA_0_80 | GraphApi::PROFILE_BASELINE);
    if ( (syntaxVersion & GraphApi::VERSION_TOSA_0_60) == GraphApi::VERSION_TOSA_0_60 )
    {
        return GraphApi::VERSION_TOSA_0_60;
    }
    else if ( (syntaxVersion & GraphApi::VERSION_TOSA_0_80) == GraphApi::VERSION_TOSA_0_80 )
    {
        return GraphApi::VERSION_TOSA_0_80;
    }
    else if ( (syntaxVersion & GraphApi::VERSION_TOSA_1_00) == GraphApi::VERSION_TOSA_1_00 )
    {
        return GraphApi::VERSION_TOSA_1_00;
    }
    return std::nullopt;
}

}  // namespace

namespace regor
{

bool TosaGraphValidator::HandlesSyntax(uint32_t syntaxVersion)
{
    return MaybeGetTosaVersion(syntaxVersion).has_value();
}

TosaGraphValidator::TosaGraphValidator(GraphNotation notation, uint32_t syntaxVersion, Compiler *compiler) :
        GraphValidator(notation, syntaxVersion)
{
    _context.version = MaybeGetTosaVersion(syntaxVersion).value_or(GraphApi::VERSION_TOSA_0_60);

    if ( (syntaxVersion & GraphApi::PROFILE_MAIN) == GraphApi::PROFILE_MAIN )
    {
        _context.profile = GraphApi::PROFILE_MAIN;
    }
    else
    {
        _context.profile = GraphApi::PROFILE_BASELINE;
    }
    _context.GetGraph = [compiler](const char *name) { return compiler->GetGraph(name); };
}

bool TosaGraphValidator::Validate(Graph *graph)
{
    bool graphValid = true;
    Graph::TraverseGraphFromEnd(graph->Outputs(),
        [&graphValid, &graph, this](Operation *op) -> bool
        {
            try
            {
                tosa::validator::ValidateOperator(op, _context);
            }
            catch ( const std::invalid_argument &e )
            {
                graphValid = false;
                _validationErrors.emplace_back(Error{op->Type(), e.what()});
            }
            return true;
        });
    return graphValid;
}

}  // namespace regor
