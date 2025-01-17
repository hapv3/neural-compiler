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

#include "util.hpp"

#include "common/data_type.hpp"
#include "common/ini_reader.hpp"

#include <memory>
#include <mutex>
#include <thread>

using namespace regor;

// Helpers for Architecture
// -----------------------------
// Creates a default test-config
// macs can be specified
std::string TestConfig(int macs)
{
    std::string config = "[architecture]\n";
    config += fmt::format("macs={}\n", macs);
    // flash
    config += "[memory.flash]\n";
    config += "name=flash\n";
    config += "size=128mb\n";
    config += "read_latency=32\n";
    config += "write_latency=32\n";
    config += "bandwidth=8\n";
    config += "burst_length=128\n";
    // sram
    config += "[memory.sram]\n";
    config += "name=sram\n";
    config += "size=8192kb\n";
    config += "read_latency=32\n";
    config += "write_latency=32\n";
    config += "bandwidth=8\n";
    config += "burst_length=32\n";
    // dram
    config += "[memory.dram]\n";
    config += "name=dram\n";
    config += "size=16mb\n";
    config += "bandwidth=8\n";
    config += "burst_length=32\n";
    // System configuration
    config += "[system]\n";
    config += "const=flash\n";
    config += "feature_maps=dram\n";
    config += "staging=sram\n";
    return config;
}

// Parse-helper to create architecture
void ParseConfig(const std::string &config, size_t size, std::unique_ptr<Architecture> &arch)
{
    IniReader reader(config.c_str(), config.size());
    std::string section;
    while ( reader.Begin(section) )
    {
        auto result = arch->ParseSection(section, &reader);
        if ( result == IniParseResult::Error )
        {
            return;
        }
        reader.End();
    }
}

// Create a Graph from a list of operations
// Populates inputs/outputs based on tensorUsage
std::unique_ptr<Graph> CreateGraph(std::vector<std::shared_ptr<Operation>> &ops)
{
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::vector<std::shared_ptr<Tensor>> outputs;
    std::vector<std::shared_ptr<Tensor>> persistent;
    std::vector<std::shared_ptr<Tensor>> placeholder;
    for ( const auto &op : ops )
    {
        for ( auto &conn : op->Inputs() )
        {
            if ( conn.tensor->Writers().empty() && !conn.tensor->IsConstant() )
            {
                inputs.push_back(conn.tensor);
            }
        }
        for ( auto &conn : op->Outputs() )
        {
            if ( conn.tensor->Readers().empty() && !conn.tensor->IsConstant() )
            {
                outputs.push_back(conn.tensor);
            }
        }
    }
    auto graph = std::make_unique<Graph>("testGraph", inputs, outputs, persistent, placeholder, GraphNotation::GraphAPI, 1);
    return graph;
}

// Helpers for Graph IR
// -----------------------------
// Create a Tensor with name, storageshape and datatype
std::shared_ptr<Tensor> CreateTensor(const std::string &name, const Shape &storageShape, DataType dtype)
{
    auto tensor = std::make_shared<Tensor>(name, dtype, storageShape);
    return tensor;
}

// Create a Const Tensor
template<typename T>
std::shared_ptr<Tensor> CreateTensor(const std::string &name, const Shape &storageShape, DataType dtype, std::vector<T> &&values)
{
    assert(int(values.size()) == storageShape.Elements());
    assert(DataTypeSizeBits(dtype) == sizeof(T) * 8);
    auto buf = std::make_shared<Buffer>(std::move(values));
    auto tensor = std::make_shared<Tensor>(name, dtype, storageShape, std::move(buf));
    return tensor;
}

// Create a Const Tensor
std::shared_ptr<Tensor> CreateTensor(const std::string &name, const Shape &storageShape, DataType dtype, unsigned value)
{
    switch ( dtype )
    {
        case DataType::Int8:
            return CreateTensor(name, storageShape, dtype, std::vector<int8_t>(storageShape.Elements(), int8_t(value)));
            break;
        case DataType::UInt8:
            return CreateTensor(name, storageShape, dtype, std::vector<uint8_t>(storageShape.Elements(), uint8_t(value)));
            break;
        case DataType::Int16:
            return CreateTensor(name, storageShape, dtype, std::vector<int16_t>(storageShape.Elements(), int16_t(value)));
            break;
        case DataType::Int32:
            return CreateTensor(name, storageShape, dtype, std::vector<int32_t>(storageShape.Elements(), int32_t(value)));
            break;
        default:
            assert(false);
            return CreateTensor(name, storageShape, dtype, std::vector<int8_t>(storageShape.Elements(), int8_t(value)));
    }
}

// Create a Operation with unary input
std::shared_ptr<Operation> CreateOperation(OpType opType, TensorUsage ifmUsage, std::shared_ptr<Tensor> &ifm,
    TensorUsage ofmUsage, std::shared_ptr<Tensor> &ofm)
{
    auto op = std::make_shared<Operation>(opType);
    op->SetKernel(std::make_unique<Kernel>(Kernel::UnitKernel()));
    op->ConnectInput(ifmUsage, ifm);
    op->ConnectOutput(ofmUsage, ofm);
    return op;
}

// Create a Operation with binary input
std::shared_ptr<Operation> CreateOperation(OpType opType, TensorUsage ifmUsage, std::shared_ptr<Tensor> &ifm,
    TensorUsage ifm2Usage, std::shared_ptr<Tensor> &ifm2, TensorUsage ofmUsage, std::shared_ptr<Tensor> &ofm)
{
    auto op = CreateOperation(opType, ifmUsage, ifm, ofmUsage, ofm);
    op->ConnectInput(ifm2Usage, ifm2);
    return op;
}

// Helpers for Scheduler IR
// -----------------------------
// Create a SchedulerTensor with name, storageshape and datatype
// also creates a srcTensor (in GraphIR)
std::shared_ptr<SchedulerTensor> CreateSchedulerTensor(const std::string &name, const Shape &storageShape, DataType dtype)
{
    auto tensor = CreateTensor(name, storageShape, dtype);
    auto schedTensor = std::make_shared<SchedulerTensor>();
    schedTensor->srcTensor = tensor;
    schedTensor->storageShape = storageShape;
    schedTensor->dataType = dtype;
    return schedTensor;
}

static struct temporary_op_scope_t
{
    std::vector<std::shared_ptr<Operation>> _ops;
    std::mutex _lock;

    void add_op(const std::shared_ptr<Operation> &op)
    {
        std::lock_guard lock(_lock);
        _ops.push_back(op);
    }

    ~temporary_op_scope_t()
    {
        try
        {
            for ( const auto &op : _ops )
                op->Disconnect();
        }
        catch ( std::bad_weak_ptr & )
        {
        }
    }
} s_ops;


// Create a SchedulerOperation with unary input
std::unique_ptr<SchedulerOperation> CreateSchedulerOperation(OpType opType, TensorUsage ifmUsage,
    std::shared_ptr<SchedulerTensor> &ifm, TensorUsage ofmUsage, std::shared_ptr<SchedulerTensor> &ofm)
{
    // use static vector to keep operation reference alive
    auto op = CreateOperation(opType, ifmUsage, ifm->srcTensor, ofmUsage, ofm->srcTensor);
    s_ops.add_op(op);

    auto schedOp = std::make_unique<SchedulerOperation>(opType);
    schedOp->SetKernel(op->Kernel());
    schedOp->_srcKey = static_cast<void *>(op.get());
    // ifm
    auto *ifmConn = schedOp->AddInput(ifmUsage);
    ifmConn->tensor = ifm;
    ifmConn->shape = ifm->storageShape;
    ifm->consumers.push_back(schedOp.get());
    // ofm
    auto *ofmConn = schedOp->AddOutput(ofmUsage);
    ofmConn->tensor = ofm;
    ofmConn->shape = ofm->storageShape;
    ofm->producers.push_back(schedOp.get());
    return schedOp;
}

// Create a SchedulerOperation with binary input
std::unique_ptr<SchedulerOperation> CreateSchedulerOperation(OpType opType, TensorUsage ifmUsage, std::shared_ptr<SchedulerTensor> &ifm,
    TensorUsage ifm2Usage, std::shared_ptr<SchedulerTensor> &ifm2, TensorUsage ofmUsage, std::shared_ptr<SchedulerTensor> &ofm)
{
    auto schedOp = CreateSchedulerOperation(opType, ifmUsage, ifm, ofmUsage, ofm);
    auto *ifm2Conn = schedOp->AddInput(ifm2Usage);
    ifm2Conn->tensor = ifm2;
    ifm2Conn->shape = ifm2->storageShape;
    ifm2->consumers.push_back(schedOp.get());
    return schedOp;
}
