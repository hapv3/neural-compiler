//
// SPDX-FileCopyrightText: Copyright 2022-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/buffer_view.hpp"
#include "include/graphapi.hpp"
#include "include/graphapi_tosa_types.hpp"
#include "tensor_properties.hpp"

#include <memory>
#include <string>
#include <vector>

namespace regor
{

class Compiler;
class Operation;
class Tensor;

/// <summary>
/// Graph Builder implementation
/// </summary>
class GraphBuilder : public GraphApi::IGraphBuilder
{
    friend class Compiler;
    using GraphTensor = GraphApi::GraphTensor;
    using GraphShape = GraphApi::GraphShape;
    using GraphKernel = GraphApi::GraphKernel;
    using GraphOperation = GraphApi::GraphOperation;
    using GraphTensorUsage = GraphApi::GraphTensorUsage;
    using GraphDataType = GraphApi::GraphDataType;
    using GraphBuffer = GraphApi::GraphBuffer;
    using BufferMapping = GraphApi::BufferMapping;
    using GraphTensorLayout = GraphApi::GraphTensorLayout;

protected:
    std::string _graphName;
    uint32_t _syntaxVersion = 0;
    std::vector<std::shared_ptr<Operation>> _operations;
    std::vector<std::shared_ptr<Tensor>> _tensors;
    std::vector<std::shared_ptr<Tensor>> _inputs;
    std::vector<std::shared_ptr<Tensor>> _outputs;
    std::vector<std::shared_ptr<Tensor>> _persistent;
    std::vector<std::shared_ptr<Buffer>> _buffers;

public:
    GraphBuilder(const std::string &name);
    ~GraphBuilder();

public:
    // Inherited via IGraphBuilder
    bool RequireSyntaxVersion(uint32_t version, int32_t level) override;
    GraphOperation *CreateOp(tosa::Op opType, const GraphKernel *kernel) override;
    GraphBuffer *CreateBuffer(size_t sizeBytes, BufferMapping mapping, const void *initialData) override;
    GraphTensor *CreateTensor(const char *name, const GraphShape &shape, GraphTensorLayout layout,
        GraphDataType dataType, GraphBuffer *buffer) override;
    // Set graph inputs/outputs
    void AddInput(GraphTensor *graphTensor) override;
    void AddOutput(GraphTensor *graphTensor) override;
    void AddPersistent(GraphTensor *graphTensor) override;
    // Connect operator inputs/outputs
    void AddInput(GraphOperation *graphOp, GraphTensorUsage usage, GraphTensor *graphTensor) override;
    void AddOutput(GraphOperation *graphOp, GraphTensorUsage usage, GraphTensor *graphTensor) override;
    // Object attribute and properties
    bool Set(GraphOperation *graphOp, GraphApi::OpAttr attr, bool value) override;
    bool Set(GraphOperation *graphOp, GraphApi::OpAttr attr, int32_t value) override;
    bool Set(GraphOperation *graphOp, GraphApi::OpAttr attr, double value) override;
    bool Set(GraphOperation *graphOp, GraphApi::OpAttr attr, const GraphApi::GraphShape &value) override;
    bool Set(GraphOperation *graphOp, GraphApi::OpAttr attr, const GraphApi::FractionND &value) override;
    bool Set(GraphOperation *graphOp, GraphApi::OpAttr attr, const GraphApi::Point2 &value) override;
    bool Set(GraphOperation *graphOp, GraphApi::OpAttr attr, const char *value) override;
    void SetZeroPoint(GraphOperation *op, GraphTensorUsage usage, double zeroPoint) override;
    void SetAxisOrder(GraphTensor *graphTensor, GraphApi::AxisOrder order) override;
    void SetAxisStrides(GraphTensor *graphTensor, const GraphApi::GraphShape *axisStrides) override;
    // Utility
    const std::string &Name() const { return _graphName; }
    uint32_t SyntaxVersion() const { return _syntaxVersion; }

private:
    void FreeUnconnected();
};

}  // namespace regor
