//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/common.hpp"

#include "compiler/scheduler_packing.hpp"

#include <fmt/format.h>
#include <catch_all.hpp>
#include <iostream>

#include "regor.h"


using namespace regor;

#include "architecture/architecture.hpp"
#include "architecture/ethosu85/ethos_u85.hpp"
#include "common/ini_reader.hpp"

namespace
{

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
    config += "feature_maps=sram\n";
    config += "staging=sram\n";
    return config;
}

void ParseConfig(std::string config, size_t size, std::unique_ptr<Architecture> &arch)
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

template<typename T>
std::unique_ptr<Architecture> CreateArchDefault(int macs = 128)
{
    std::unique_ptr<Architecture> arch = std::make_unique<T>();
    std::string config = TestConfig(macs);
    ParseConfig(config, sizeof(config), arch);
    return arch;
}

std::unique_ptr<Graph> CreateGraph(std::vector<std::shared_ptr<Operation>> &ops)
{
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::vector<std::shared_ptr<Tensor>> outputs;
    std::vector<std::shared_ptr<Tensor>> persistent;
    for ( const auto &op : ops )
    {
        for ( auto &conn : op->Inputs() )
        {
            if ( conn.tensor->Writers().empty() )
            {
                inputs.push_back(conn.tensor);
            }
        }
        for ( auto &conn : op->Outputs() )
        {
            if ( conn.tensor->Readers().empty() )
            {
                outputs.push_back(conn.tensor);
            }
        }
    }
    auto graph = std::make_unique<Graph>("testGraph", inputs, outputs, persistent, GraphNotation::GraphAPI, 1);
    return graph;
}

std::shared_ptr<Tensor> CreateTensor(std::string name, Shape storageShape, DataType dtype)
{
    auto tensor = std::make_shared<Tensor>(name, dtype, storageShape);
    return tensor;
}

std::shared_ptr<Operation> CreateOperation(OpType opType, std::shared_ptr<Tensor> &ifm0, std::shared_ptr<Tensor> &ofm)
{
    auto op = std::make_shared<Operation>(opType);
    op->ConnectInput(TensorUsage::IFM0, ifm0);
    op->ConnectOutput(TensorUsage::OFM, ofm);
    return op;
}

std::shared_ptr<Operation> CreateOperation(
    OpType opType, std::shared_ptr<Tensor> &ifm0, std::shared_ptr<Tensor> &ifm1, std::shared_ptr<Tensor> &ofm)
{
    auto op = CreateOperation(opType, ifm0, ofm);
    op->ConnectInput(TensorUsage::IFM1, ifm1);
    return op;
}

}  // namespace

TEST_CASE("test_scheduler_packing")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    // Create packing
    auto packing = SchedulerPacking(arch.get(), false);
    SECTION("Pack operation (with axis)")
    {
        // Perform packing on an ArgMax operation
        // Validate that attr_axis still represents the reduced axis.
        std::vector<std::shared_ptr<Operation>> ops;
        auto ifm = CreateTensor("IFM", Shape(10, 10), DataType::Int8);
        auto ofm = CreateTensor("OFM", Shape(1, 10), DataType::Int8);
        auto op = CreateOperation(OpType::ArgMax, ifm, ofm);
        auto attr = op->Attribute<axis_attr_t>();
        attr->axis = 0;
        ops.push_back(op);

        // Create graph with ops
        auto graph = CreateGraph(ops);

        // Perform scheduler_packing
        auto schedOps = packing.Process(graph.get());
        REQUIRE(schedOps.size() == ops.size());

        // Validate that the reduced axis is still Width after packing
        for ( const auto &schedOp : schedOps )
        {
            auto *ifmConn = schedOp->Input(TensorUsage::IFM);
            auto *ofmConn = schedOp->Output(TensorUsage::OFM);
            const auto &ifmShape = ifmConn->SliceShape();
            const auto &ofmShape = ofmConn->SliceShape();
            int axis = schedOp->Attribute<axis_attr_t>()->axis;
            REQUIRE(ifmShape[axis] == 10);
            REQUIRE(ofmShape[axis] == 1);
            REQUIRE(axis == ofmShape.Size() - 2);
        }
    }
    SECTION("Pack sliced operation (with axis)")
    {
        // Perform packing on a sliced ArgMax operation
        // Validate that attr_axis still represents the reduced axis.
        std::vector<std::shared_ptr<Operation>> ops;
        auto ifm = CreateTensor("IFM", Shape(10, 10, 10), DataType::Int8);
        auto ofm = CreateTensor("OFM", Shape(10, 2, 10), DataType::Int8);

        // first op
        //  reads  0,0,0 - shape 10,5,10
        //  writes 0,0,0 - shape 10,1,10
        // second op
        //  reads  0,5,0 - shape 10,5,10
        //  writes 0,1,0 - shape 10,1,10
        for ( int i = 0; i < 2; i++ )
        {
            auto op = CreateOperation(OpType::ArgMax, ifm, ofm);
            auto attr = op->Attribute<axis_attr_t>();
            attr->axis = 1;
            TensorSlice ifmSlice{Shape(0, 5 * i, 0), Shape(10, 5, 10)};
            TensorSlice ofmSlice{Shape(0, i, 0), Shape(10, 1, 10)};
            op->Input(TensorUsage::IFM)->Set(ifmSlice);
            op->Output(TensorUsage::OFM)->Set(ofmSlice);
            ops.push_back(op);
        }

        // Create graph with ops
        auto graph = CreateGraph(ops);

        // Perform scheduler_packing
        auto schedOps = packing.Process(graph.get());
        REQUIRE(schedOps.size() == ops.size());

        // Validate that the reduced axis is still Width after packing
        for ( const auto &schedOp : schedOps )
        {
            auto *ifmConn = schedOp->Input(TensorUsage::IFM);
            auto *ofmConn = schedOp->Output(TensorUsage::OFM);
            const auto &ifmShape = ifmConn->SliceShape();
            const auto &ofmShape = ofmConn->SliceShape();
            int axis = schedOp->Attribute<axis_attr_t>()->axis;
            REQUIRE(ifmShape[axis] == 5);
            REQUIRE(ofmShape[axis] == 1);
            REQUIRE(axis == ofmShape.Size() - 2);
        }
    }
}
