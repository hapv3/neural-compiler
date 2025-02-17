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

#include "common/logging.hpp"

#include "graph.hpp"
#include "graph_optimiser.hpp"
#include "operation.hpp"
#include "tensor.hpp"

#include <vector>

namespace regor
{

/// <summary>
/// GraphIR Graph optimiser
/// </summary>
class GraphIrOptimiser : public GraphOptimiser
{
    using OpRewriteFunction = Operation *(GraphIrOptimiser::*)(Graph *, Operation *);
    using TensorRewriteFunction = Tensor *(GraphIrOptimiser::*)(Graph *, Tensor *);
    using GraphOptStepArray = std::vector<RewriteFunctions<GraphIrOptimiser>>;

private:
    Operation *ConstPropagation(Graph *const graph, Operation *const operation);
    Operation *RemoveReshape(Graph *const graph, Operation *const operation);
    Operation *ConvertAttributes(Graph *const graph, Operation *const operation);
    Operation *ConvertResizeOffsets(Graph *const graph, Operation *const operation);
    Tensor *ConvertInt48Tensors(Graph *graph, Tensor *tensor);
    Tensor *ConvertBool8Tensors(Graph *graph, Tensor *tensor);
    Tensor *ConvertInt4Tensors(Graph *graph, Tensor *tensor);
    Operation *RewriteFullyConnected(Graph *const graph, Operation *const operation);
    Operation *FixupPoolStrides(Graph *const, Operation *const operation);
    Operation *RewriteRescaleInputs(Graph *const graph, Operation *const operation);
    Operation *RewriteRescale(Graph *const graph, Operation *const operation);
    Operation *RewritePad(Graph *const graph, Operation *const operation);
    Operation *FuseRescale(Graph *const graph, Operation *const operation);
    Operation *RewriteTable(Graph *const graph, Operation *const operation);
    Operation *RewriteCast(Graph *const graph, Operation *const operation);
    Operation *RewriteConcat(Graph *const graph, Operation *const operation);
    Operation *RewriteSlice(Graph *const graph, Operation *const operation);
    Operation *RewriteNegate(Graph *const graph, Operation *const operation);
    Operation *RewriteSelect(Graph *const graph, Operation *const operation);
    Operation *RewriteReduceMinMaxAnyAll(Graph *const graph, Operation *const operation);
    Operation *RewriteReduceSum(Graph *const graph, Operation *const operation);
    Operation *RewriteTile(Graph *const graph, Operation *const operation);
    Operation *RewriteMatmul(Graph *const graph, Operation *const operation);
    Operation *RewriteArgmax(Graph *const graph, Operation *const operation);
    Operation *RewriteDepthwise(Graph *const graph, Operation *const operation);
    Operation *RewriteTransposeConvOFMPadding(Graph *const graph, Operation *const operation);
    Operation *OptimiseElementwise(Graph *const graph, Operation *const operation);
    Operation *MergeTransposes(Graph *const graph, Operation *const operation);
    Operation *RearrangeTranspose(Graph *const graph, Operation *const operation);
    Operation *ReshapeReverse(Graph *const graph, Operation *const operation);
    void MoveToConsumer(const Operation *const operation, Operation *const cons);
    Operation *MoveSplitSliceToConsumer(Graph *const, Operation *const operation);
    Operation *UnrollConv(Graph *const, Operation *const operation);
    // Utility/Helper methods
    Operation *MakeFillOperation(TensorConnection *const ofmConn, const Shape &ofmShape, const TensorSlice &ofmSlice,
        std::shared_ptr<Tensor> padTensor);

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
                &GraphOptimiser::VisitOperatorLog,
#endif
                &GraphOptimiser::RecordOperation
            }
        },
        {
            {
                &GraphIrOptimiser::ConvertInt48Tensors,
                &GraphIrOptimiser::ConvertBool8Tensors,
                &GraphIrOptimiser::ConvertInt4Tensors,
            },
            {
                &GraphIrOptimiser::RemoveReshape,
            }
        },
        {
            {},
            {
                &GraphIrOptimiser::ConstPropagation,
            },
            true,
        },
        {
            {},
            {
                &GraphIrOptimiser::ConvertAttributes,
                &GraphIrOptimiser::RewriteRescaleInputs,
                &GraphIrOptimiser::FuseRescale,  // First pass fuse all possible ifm and ofm rescales
            }
        },
        {
            {},
            {
                &GraphIrOptimiser::FuseRescale,  // Second pass, fuse any remaining ofm rescales after ifm fusing in first pass
            }
        },
        {
            {},
            {
                &GraphIrOptimiser::ConvertAttributes,
                &GraphIrOptimiser::RewriteRescale,
                &GraphIrOptimiser::ConvertResizeOffsets,
                &GraphIrOptimiser::RewriteFullyConnected,
                &GraphIrOptimiser::FixupPoolStrides,
                &GraphIrOptimiser::RewritePad,
                &GraphIrOptimiser::RewriteTable,
                &GraphIrOptimiser::RewriteCast,
                &GraphIrOptimiser::RewriteConcat,
                &GraphIrOptimiser::RewriteSlice,
                &GraphIrOptimiser::RewriteNegate,
                &GraphIrOptimiser::RewriteReduceMinMaxAnyAll,
                &GraphIrOptimiser::RewriteReduceSum,
                &GraphIrOptimiser::RewriteTile,
                &GraphIrOptimiser::RewriteMatmul,
                &GraphIrOptimiser::RewriteSelect,
                &GraphIrOptimiser::RewriteArgmax,
                &GraphIrOptimiser::RewriteDepthwise,
                &GraphIrOptimiser::RewriteTransposeConvOFMPadding,
                &GraphIrOptimiser::OptimiseElementwise,
                &GraphIrOptimiser::MergeTransposes,
                &GraphIrOptimiser::RearrangeTranspose,
                &GraphIrOptimiser::ReshapeReverse,
                &GraphIrOptimiser::UnrollConv
            }
        },
        // MoveSplitSliceToConsumer need to be done after any other optimisation that can affect the ifm/ofm shapes
        // has been performed, since the ifm/ofm shapes are of importance to this function.
        {
            {},
            {
                &GraphIrOptimiser::MoveSplitSliceToConsumer
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
    explicit GraphIrOptimiser(IArchitectureConstraints *constraints, const GraphOptimiserOptions &options, OptimiserDatabase *db);

    const GraphOptStepArray &GraphOptimisationSteps() const { return _graphOptimisationSteps; }

    void OptimiseGraph(Graph *graph);
};

}  // namespace regor
