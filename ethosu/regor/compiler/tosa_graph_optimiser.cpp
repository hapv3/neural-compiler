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

#include "compiler/tosa_graph_optimiser.hpp"

#include "optimiser_utils.hpp"


namespace regor
{

using namespace GraphOptimisation;

// Convert compile time constant zero point tensors to quantization zero points
Operation *TosaGraphOptimiser::ConvertZeroPointTensors(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    auto SetZeroPoint = [&](TensorUsage target, TensorUsage param, bool asUnsigned = false)
    {
        if ( const auto zpConn = operation->Input(param) )
        {
            assert(zpConn->tensor->IsConstant());
            const auto targetConn = IsOFM(target) ? operation->Output(target) : operation->Input(target);
            assert(targetConn);
            auto dataType = asUnsigned ? zpConn->tensor->Type() & ~unsigned(DataType::Signed) : zpConn->tensor->Type();
            auto values = zpConn->tensor->View().Values<int64_t>(dataType);
            targetConn->quantization.zeroPoints = {values.begin(), values.end()};
        }
    };
    switch ( operation->Type() )
    {
        case OpType::AvgPool:
        case OpType::Neg:
            SetZeroPoint(TensorUsage::IFM, TensorUsage::Params0);
            SetZeroPoint(TensorUsage::OFM, TensorUsage::Params1);
            break;
        case OpType::Conv2D:
        case OpType::Conv3D:
        case OpType::DepthwiseConv2D:
        case OpType::TransposeConv2D:
            SetZeroPoint(TensorUsage::IFM, TensorUsage::Params0);
            SetZeroPoint(TensorUsage::Weights, TensorUsage::Params1);
            break;
        case OpType::MatMul:
            SetZeroPoint(TensorUsage::IFM0, TensorUsage::Params0);
            SetZeroPoint(TensorUsage::IFM1, TensorUsage::Params1);
            break;
        case OpType::Rescale:
        {
            const auto signAttr = operation->Attribute<sign_attr_t>();
            SetZeroPoint(TensorUsage::IFM, TensorUsage::Params2, signAttr->input_unsigned);
            SetZeroPoint(TensorUsage::OFM, TensorUsage::Params3, signAttr->output_unsigned);
            break;
        }
        default:
            break;
    }
    return operation;
}

TosaGraphOptimiser::TosaGraphOptimiser(IArchitectureConstraints *constraints, const GraphOptimiserOptions &options, OptimiserDatabase *db) :
        GraphOptimiser(constraints, options, db)
{
}

void TosaGraphOptimiser::OptimiseGraph(Graph *graph)
{
    for ( auto iOpt = GraphOptimisationSteps().begin(); iOpt != GraphOptimisationSteps().end(); ++iOpt )
    {
        LOG_TRACE1("GraphOptimiser {0}/{1}\n", std::distance(GraphOptimisationSteps().begin(), iOpt) + 1,
            GraphOptimisationSteps().size());
        // Check if function lists are empty. Do not call for step that only contain disabled debug functions.
        if ( !iOpt->opFunction.empty() || !iOpt->tensorFunction.empty() )
        {
            RewriteGraph<TosaGraphOptimiser>(graph, *iOpt);
        }
    }
}

}  // namespace regor
