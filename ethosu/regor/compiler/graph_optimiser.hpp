//
// SPDX-FileCopyrightText: Copyright 2021-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "architecture/architecture.hpp"
#include "common/ini_reader.hpp"
#include "graph.hpp"
#include "graph_optimiser_db.hpp"
#include "operation.hpp"
#include "tensor.hpp"

#include <memory>
#include <string>
#include <vector>

#include "include/regor.h"

namespace regor
{

/// <summary>
/// Tensor and Operation rewrite functions for the graph optimisation.
/// </summary>
template<typename T>
struct RewriteFunctions
{
    const std::vector<Tensor *(T::*)(Graph *, Tensor *)> tensorFunction;
    const std::vector<Operation *(T::*)(Graph *, Operation *)> opFunction;
    const bool fromStart = false;
};

/// <summary>
/// Graph optimiser options
/// </summary>
struct GraphOptimiserOptions
{
    bool verboseGraph = false;
    bool verboseQuantization = false;
};

/// <summary>
/// Graph optimiser
/// </summary>
class GraphOptimiser
{
protected:
    IArchitectureConstraints *_constraints = nullptr;
    const GraphOptimiserOptions _options;
    OptimiserDatabase *_db;

public:
    GraphOptimiser(IArchitectureConstraints *constraints, const GraphOptimiserOptions &options, OptimiserDatabase *db) :
            _constraints(constraints), _options(options), _db(db)
    {
        assert(_constraints != nullptr);
    }
    const GraphOptimiserOptions &Options() const { return _options; }



    static std::unique_ptr<GraphOptimiser> MakeGraphOptimiser(GraphNotation notation,
        IArchitectureConstraints *constraints, const GraphOptimiserOptions &options, OptimiserDatabase *db);
    static void ParseGraphOptimiserOptions(GraphOptimiserOptions &opt, IniReader &reader);

    void Process(Graph *graph);
    virtual void OptimiseGraph(Graph *graph) = 0;

    // Note no check for if NPU operator, or "rewrite_unsupported".
    // Such checks are delegated to each specific rewrite function.
    template<typename T>
    void RewriteGraph(Graph *const graph, const RewriteFunctions<T> &rewriteFuncs)
    {
        using OpFunction = Operation *(T::*)(Graph *, Operation *);
        using TensFunction = Tensor *(T::*)(Graph *, Tensor *);

        const std::vector<OpFunction> *opFunctions = &rewriteFuncs.opFunction;
        const std::vector<TensFunction> *tensFunctions = &rewriteFuncs.tensorFunction;

        // TODO: MLBEDSW-9057: Check when specific rewrite functions are added
        struct Entry
        {
            bool done;
            std::shared_ptr<Operation> op;
            Entry(bool done_, const std::shared_ptr<Operation> &op_) : done(done_), op(op_) {}
        };
        std::unordered_set<Operation *> opVisited;
        std::unordered_set<Tensor *> tensVisited;
        std::stack<Entry> stack;

        if ( rewriteFuncs.fromStart )
        {
            // Traverse from End and collect operators that are at the start of the graph. Their inputs are only either
            // - Constant
            // - Graph inputs
            Graph::TraverseGraphFromEnd(graph->Outputs(),
                [&](Operation *op) -> bool
                {
                    for ( auto [usage, ifmConn] : op->Inputs().pairs() )
                    {
                        if (
                            !ifmConn.tensor->IsConstant() &&
                            std::find(graph->Inputs().begin(), graph->Inputs().end(), ifmConn.tensor) == graph->Inputs().end() )
                            return true;
                    }
                    stack.emplace(false, op->shared_from_this());
                    return true;
                });
        }
        else
        {
            for ( const auto &tensor : graph->Outputs() )
            {
                for ( const auto &op : tensor->Writers() )
                {
                    stack.emplace(false, op);
                }
            }
        }

        while ( !stack.empty() )
        {
            Entry entry = stack.top();
            stack.pop();

            // Currently we do not do anything with entry.done elements.
            if ( !entry.done && opVisited.count(entry.op.get()) == 0 && !entry.op->IsDisconnected() )
            {
                Operation *updatedOp = entry.op.get();
                Operation *prevOp = nullptr;

                // Process op
                if ( !opFunctions->empty() )
                {
                    // Op have been updated, parse it again.
                    while ( prevOp != updatedOp )
                    {
                        if ( prevOp != nullptr )
                        {
                            // prevOp was removed and replaced by updatedOp; remove the stale pointer from opVisited
                            opVisited.erase(prevOp);
                        }
                        prevOp = updatedOp;
                        // Execute operator functions
                        for ( const auto &func : *opFunctions )
                        {
                            updatedOp = (static_cast<T *>(this)->*(func))(graph, updatedOp);
                            assert(updatedOp && "Operator rewrite function returned NULL");
                        }
                    }
                }
                opVisited.insert(updatedOp);
                stack.emplace(true, updatedOp->shared_from_this());

                std::array<std::vector<Tensor *>, 2> tensors;
                for ( const auto &pair : updatedOp->Outputs().pairs() )
                {
                    tensors[0].push_back(pair.second.tensor.get());
                }
                for ( const auto &pair : updatedOp->Inputs().pairs() )
                {
                    tensors[1].push_back(pair.second.tensor.get());
                }
                for ( auto idx = 0; idx < 2; ++idx )
                {
                    for ( const auto &tens : tensors[idx] )
                    {
                        Tensor *updatedTensor = tens;

                        // Process Tensor if not already visited
                        if ( tensVisited.count(tens) == 0 )
                        {
                            Tensor *prevTensor = nullptr;

                            if ( !tensFunctions->empty() )
                            {
                                // Tensor have been updated, parse it again.
                                while ( prevTensor != updatedTensor )
                                {
                                    if ( prevTensor != nullptr )
                                    {
                                        tensVisited.erase(prevTensor);
                                    }
                                    prevTensor = updatedTensor;
                                    // Execute tensor functions
                                    for ( const auto &func : *tensFunctions )
                                    {
                                        updatedTensor = (static_cast<T *>(this)->*(func))(graph, updatedTensor);
                                        assert(updatedTensor && "Tensor rewrite function returned NULL");
                                    }
                                }
                            }
                            tensVisited.insert(updatedTensor);
                        }
                        // Check Reader/Writers for tensor, even if visited as we can bypass op updating the tensors.
                        const std::vector<std::shared_ptr<Operation>>
                            *ops = rewriteFuncs.fromStart ? &updatedTensor->Readers() : &updatedTensor->Writers();
                        for ( const auto &op : *ops )
                        {
                            if ( opVisited.count(op.get()) == 0 )
                            {
                                stack.emplace(false, op);
                            }
                        }
                    }
                }
            }
        }
    }

    void ReplaceOperation(Operation *const operationToReplace, Operation *const newOperation)
    {
        auto oldOperation = operationToReplace->shared_from_this();

        for ( const auto &input : oldOperation->Inputs().pairs() )
        {
            newOperation->CopyInput(input.first, input.second);
        }
        for ( const auto &output : oldOperation->Outputs().pairs() )
        {
            newOperation->CopyOutput(output.first, output.second);
        }
        oldOperation->Disconnect();
    }

#if LOG_TRACE1_ON
    Operation *VisitOperatorLog(Graph *const graph, Operation *const operation);
    Tensor *VisitTensorLog(Graph *const graph, Tensor *const tensor);
#endif
    Operation *RecordOperation(Graph *const graph, Operation *const operation);
    Operation *RecordOptimisation(Graph *const graph, Operation *const operation);
    void RecordOptimisation(const Operation *operation, const Operation *op);
    void PrintGraph(const Graph *graph, const std::string &label) const;
    void PrintQuantization(const Graph *graph, const std::string &label) const;
    virtual ~GraphOptimiser() = default;
};

}  // namespace regor
