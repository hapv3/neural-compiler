//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/logging.hpp"

#include "graph.hpp"
#include "graph_optimiser.hpp"
#include "operation.hpp"
#include "tensor.hpp"

#include <vector>

namespace regor
{

/// <summary>
/// TOSA Graph optimiser
/// </summary>
class TosaGraphOptimiser : public GraphOptimiser
{
    using OpRewriteFunction = Operation *(TosaGraphOptimiser::*)(Graph *, Operation *);
    using TensorRewriteFunction = Tensor *(TosaGraphOptimiser::*)(Graph *, Tensor *);
    using GraphOptStepArray = std::vector<RewriteFunctions<TosaGraphOptimiser>>;

private:
    Operation *ConvertZeroPointTensors(Graph *const graph, Operation *const operation);

public:
    // The graph optimisation steps.
    // Order matters, array of rewrites processed in order.
    // clang-format off
    const GraphOptStepArray _graphOptimisationSteps =
    {{
        {
            {
#if LOG_TRACE1_ON
               &GraphOptimiser::VisitTensorLog
#endif
            },
            {
#if LOG_TRACE1_ON
                &TosaGraphOptimiser::VisitOperatorLog,
#endif
            }
        },
        {
            {},
            {
                &TosaGraphOptimiser::ConvertZeroPointTensors,
            }
        },
        {
            {
#if LOG_TRACE1_ON
                &GraphOptimiser::VisitTensorLog
#endif
            },
            {
#if LOG_TRACE1_ON
                &GraphOptimiser::VisitOperatorLog,
#endif
                &GraphOptimiser::RecordOptimisation
            }
        }
    }};
    // clang-format on

public:
    explicit TosaGraphOptimiser(IArchitectureConstraints *constraints, const GraphOptimiserOptions &options, OptimiserDatabase *db);

    const GraphOptStepArray &GraphOptimisationSteps() const { return _graphOptimisationSteps; }

    void OptimiseGraph(Graph *graph);
};

}  // namespace regor
