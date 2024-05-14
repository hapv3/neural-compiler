//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "operation.hpp"
#include "tensor.hpp"

#include <list>
#include <memory>
#include <stack>
#include <unordered_set>
#include <vector>

namespace regor
{

enum class GraphNotation
{
    Invalid = 0,
    GraphAPI = 1,
    TFLite = 2,
};

/// <summary>
/// Top level Neural Network Graph (NNG)
/// </summary>
class Graph
{
private:
    std::string _name;
    std::vector<std::shared_ptr<Tensor>> _inputs;
    std::vector<std::shared_ptr<Tensor>> _outputs;
    GraphNotation _notation = GraphNotation::Invalid;
    uint32_t _syntaxVersion = 0;
    const void *_passthrough = nullptr;  // Original flatbuffer description of this model (if it was loaded from one)
    std::vector<Operation *> _opsInScheduledOrder;

public:
    Graph() = delete;

    Graph(GraphNotation nt) : _notation(nt) {}

    Graph(const std::string &name, std::vector<std::shared_ptr<Tensor>> inputs, std::vector<std::shared_ptr<Tensor>> outputs, GraphNotation nt, uint32_t version) :
            _name(name), _inputs(std::move(inputs)), _outputs(std::move(outputs)), _notation(nt), _syntaxVersion(version)
    {
    }

    ~Graph()
    {
        _notation = GraphNotation::Invalid;
        std::vector<Operation *> operations;
        GetAllOperations(operations);
        for ( auto operation : operations )
        {
            operation->Disconnect();
        }
    }

public:
    const std::string &Name() const { return _name; }
    uint32_t SyntaxVersion() const { return _syntaxVersion; }

    const std::vector<std::shared_ptr<Tensor>> &Inputs() const { return _inputs; }
    const std::vector<std::shared_ptr<Tensor>> &Outputs() const { return _outputs; }

    void AddInput(const std::shared_ptr<Tensor> &input) { _inputs.push_back(input); }
    void AddOutput(const std::shared_ptr<Tensor> &output) { _outputs.push_back(output); }

    bool IsInput(const Tensor *tensor) const
    {
        return std::find_if(_inputs.begin(), _inputs.end(),
                   [&](const std::shared_ptr<Tensor> &input) { return input.get() == tensor; }) != _inputs.end();
    }
    bool IsOutput(const Tensor *tensor) const
    {
        return std::find_if(_outputs.begin(), _outputs.end(),
                   [&](const std::shared_ptr<Tensor> &output) { return output.get() == tensor; }) != _outputs.end();
    }

    GraphNotation Notation() const
    {
        assert(_notation != GraphNotation::Invalid);
        return _notation;
    }

    uint32_t Version() const { return _syntaxVersion; }

    const void *Passthrough() const { return _passthrough; }
    void SetPassthrough(const void *passthrough) { _passthrough = passthrough; }

    // Finds all operations which precede a graph output and adds them to the vector in execution order
    void GetAllOperations(std::vector<Operation *> &operations) const
    {
        TraverseGraphFromEnd(Outputs(),
            [&](Operation *op) -> bool
            {
                operations.push_back(op);
                return true;
            });
    }

    // Get all operations in the graph, in scheduled order
    const std::vector<Operation *> &ScheduledOrder() const { return _opsInScheduledOrder; };

    void SetScheduledOrder(std::vector<Operation *> operations) { _opsInScheduledOrder = std::move(operations); }

    template<typename OPFUNC>
    static void TraverseGraphFromEnd(const std::vector<std::shared_ptr<Tensor>> &from, OPFUNC opFunc)
    {
        struct Entry
        {
            bool done;
            Operation *op;
        };
        std::unordered_set<Operation *> visited;
        std::stack<Entry> stack;

        for ( const auto &tensor : from )
        {
            for ( const auto &op : tensor->Writers() )
            {
                stack.push(Entry{false, op.get()});
            }
        }

        while ( !stack.empty() )
        {
            Entry entry = stack.top();
            stack.pop();
            if ( entry.done )
            {
                if ( !opFunc(entry.op) )
                {
                    return;
                }
            }
            else if ( visited.count(entry.op) == 0 )
            {
                visited.insert(entry.op);
                stack.push(Entry{true, entry.op});
                for ( const auto &pair : entry.op->Inputs().pairs() )
                {
                    for ( const auto &op : pair.second.tensor->Writers() )
                    {
                        if ( visited.count(op.get()) == 0 )
                        {
                            stack.push(Entry{false, op.get()});
                        }
                    }
                }
            }
        }
    }
};

}  // namespace regor
