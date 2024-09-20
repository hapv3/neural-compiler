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

#include "graph_validator.hpp"

#include "tosa_graph_validator.hpp"

namespace regor
{

std::unique_ptr<GraphValidator> GraphValidator::MakeGraphValidator(GraphNotation notation, uint32_t syntaxVersion, Compiler *compiler)
{
    if ( notation == GraphNotation::GraphAPI )
    {
        if ( TosaGraphValidator::HandlesSyntax(syntaxVersion) )
        {
            return std::make_unique<TosaGraphValidator>(notation, syntaxVersion, compiler);
        }
    }
    return std::make_unique<GraphValidator>(notation, syntaxVersion);
}

bool GraphValidator::Validate(Graph *)
{
    _validationErrors.emplace_back(Error{OpType::None, "Unsupported graph Notation/SyntaxVersion"});
    return false;
}

std::string GraphValidator::GetErrorMsg()
{
    std::string errorMsg = "Validation error:\n";
    for ( auto &error : _validationErrors )
    {
        errorMsg += OpTypeToString(error.operation) + " " + error.errorMessage + "\n";
    }
    return errorMsg;
}

}  // namespace regor
