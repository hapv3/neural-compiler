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
#include "architecture/architecture_constraints.hpp"
#include "common/scaling.hpp"
#include "compiler/graph.hpp"
#include "compiler/graph_optimiser.hpp"
#include "compiler/operation.hpp"
#include "compiler/tensor.hpp"
#include "tflite_schema_generated.hpp"

#include <memory>
#include <unordered_map>

namespace regor
{

// Parses a TensorFlow Lite flatbuffer to create a Graph.
// The flatbuffer is expected to remain in place and unmodified for the entire lifetime of the created Graph.
class TfLiteReader
{
public:
    TfLiteReader() {}

    static void LoadGraphs(const void *input, size_t size, std::vector<std::unique_ptr<Graph>> &graphs,
        OptimiserDatabase *optDb, IArchitectureConstraints *constraints);  // From buffer

private:
    static void LoadGraphs(const uint8_t *input, const tflite::Model *model, std::vector<std::unique_ptr<Graph>> &graphs,
        OptimiserDatabase *optDb, IArchitectureConstraints *constraints);  // From model
    static const tflite::Model *LoadModel(const void *input, size_t size);
    static std::shared_ptr<Tensor> ParseTensor(const tflite::Tensor *tflite_tensor,
        const std::shared_ptr<Buffer> &buffer, std::unordered_map<UniqueId, Quantization> &tensorQuantization);
    static void ParseOperatorOptions(const std::shared_ptr<Operation> &operation,
        const tflite::Operator *tflite_operator, OptimiserDatabase *optDb, IArchitectureConstraints *constraints);
    static void SetOFMRounding(const std::shared_ptr<Operation> &operation);
    static void UnFuseActivation(const std::shared_ptr<Operation> &operation, tflite::ActivationFunctionType type, OptimiserDatabase *optDb);
    static void DefaultOperatorOptions(const std::shared_ptr<Operation> &operation);
    static ExecutionQuery OperationToExecQuery(const Operation &operation);
};

}  // namespace regor
