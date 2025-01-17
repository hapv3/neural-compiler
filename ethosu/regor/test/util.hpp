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

#include "architecture/architecture.hpp"
#include "compiler/graph.hpp"
#include "compiler/scheduler.hpp"
#include "compiler/scheduler_operation.hpp"

using namespace regor;

// Helpers for Architecture
// -----------------------------
// Creates a default test-config
// macs can be specified
std::string TestConfig(int macs);
// Parse-helper to create architecture
void ParseConfig(const std::string &config, size_t size, std::unique_ptr<Architecture> &arch);
// Creates an architecture-object based on default config
// macs can be specified
template<typename T>
std::unique_ptr<Architecture> CreateArchDefault(int macs = 128)
{
    std::unique_ptr<Architecture> arch = std::make_unique<T>();
    std::string config = TestConfig(macs);
    ParseConfig(config, sizeof(config), arch);
    return arch;
}
// Creates an architecture-object
template<typename T>
std::unique_ptr<Architecture> CreateArchDefault(const std::string &config)
{
    std::unique_ptr<Architecture> arch = std::make_unique<T>();
    ParseConfig(config, sizeof(config), arch);
    return arch;
}
// Create a Graph from a list of operations
// Populates inputs/outputs based on tensorUsage
std::unique_ptr<Graph> CreateGraph(std::vector<std::shared_ptr<Operation>> &ops);

// Helpers for Graph IR
// -----------------------------
// Create a Tensor with name, storageshape and datatype
std::shared_ptr<Tensor> CreateTensor(const std::string &name, const Shape &storageShape, DataType dtype);
// Create a Const Tensor
template<typename T>
std::shared_ptr<Tensor> CreateTensor(const std::string &name, const Shape &storageShape, DataType dtype, std::vector<T> &&values);
std::shared_ptr<Tensor> CreateTensor(const std::string &name, const Shape &storageShape, DataType dtype, unsigned value);
// Create a Operation with unary input
std::shared_ptr<Operation> CreateOperation(OpType opType, TensorUsage ifmUsage, std::shared_ptr<Tensor> &ifm,
    TensorUsage ofmUsage, std::shared_ptr<Tensor> &ofm);
// Create a Operation with binary input
std::shared_ptr<Operation> CreateOperation(OpType opType, TensorUsage ifmUsage, std::shared_ptr<Tensor> &ifm,
    TensorUsage ifm2Usage, std::shared_ptr<Tensor> &ifm2, TensorUsage ofmUsage, std::shared_ptr<Tensor> &ofm);

// Helpers for Scheduler IR
// -----------------------------
// Create a SchedulerTensor with name, storageshape and datatype
// also creates a srcTensor (in GraphIR)
std::shared_ptr<SchedulerTensor> CreateSchedulerTensor(const std::string &name, const Shape &storageShape, DataType dtype);

// Create a SchedulerOperation with unary input
std::unique_ptr<SchedulerOperation> CreateSchedulerOperation(OpType opType, TensorUsage ifmUsage,
    std::shared_ptr<SchedulerTensor> &ifm, TensorUsage ofmUsage, std::shared_ptr<SchedulerTensor> &ofm);

// Create a SchedulerOperation with binary input
std::unique_ptr<SchedulerOperation> CreateSchedulerOperation(OpType opType, TensorUsage ifmUsage, std::shared_ptr<SchedulerTensor> &ifm,
    TensorUsage ifm2Usage, std::shared_ptr<SchedulerTensor> &ifm2, TensorUsage ofmUsage, std::shared_ptr<SchedulerTensor> &ofm);
