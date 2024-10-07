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

#include "compiler/graph_packing.hpp"
#include "util.hpp"

#include <catch_all.hpp>
#include <memory>

#include "regor.h"

using AddressMap = std::unordered_map<const Tensor *, Address>;

static std::shared_ptr<SchedulerTensor> CreateTensor(std::string name, Address tensorAddress, AddressMap &tensorAddressMap)
{
    auto schedTensor = CreateSchedulerTensor(name, Shape(10, 10, 10), DataType::Int8);
    schedTensor->SetAddress(tensorAddress);
    tensorAddressMap[schedTensor->srcTensor.get()] = tensorAddress;
    return schedTensor;
}

static std::unique_ptr<SchedulerOperation> CreateSchedulerOperation(bool npu, TensorUsage ifmUsage,
    std::shared_ptr<SchedulerTensor> &ifm, TensorUsage ofmUsage, std::shared_ptr<SchedulerTensor> &ofm)
{
    auto schedOp = CreateSchedulerOperation(OpType::AvgPool, ifmUsage, ifm, ofmUsage, ofm);
    schedOp->SetNpuOp(npu);
    return schedOp;
}

static std::unique_ptr<SchedulerOperation> CreateSchedulerOperation(bool npu, TensorUsage ifmUsage, std::shared_ptr<SchedulerTensor> &ifm,
    TensorUsage ifm2Usage, std::shared_ptr<SchedulerTensor> &ifm2, TensorUsage ofmUsage, std::shared_ptr<SchedulerTensor> &ofm)
{
    auto schedOp = CreateSchedulerOperation(OpType::AvgPool, ifmUsage, ifm, ifm2Usage, ifm2, ofmUsage, ofm);
    schedOp->SetNpuOp(npu);
    return schedOp;
}

TEST_CASE("test_graph_packing")
{
    AddressMap tensorAddressMap;

    // Create some tensors
    auto tens1 = CreateTensor("t1", 0x04, tensorAddressMap);
    auto tens2 = CreateTensor("t2", 0x08, tensorAddressMap);
    auto tens3 = CreateTensor("t3", 0x12, tensorAddressMap);
    auto tens4 = CreateTensor("t4", 0x16, tensorAddressMap);
    auto tens5 = CreateTensor("t5", 0x1A, tensorAddressMap);
    auto var1 = CreateTensor("v1", 0x01, tensorAddressMap);

    SECTION("All NPU")
    {
        std::vector<std::unique_ptr<SchedulerOperation>> ops;
        ops.push_back(CreateSchedulerOperation(true, TensorUsage::IFM, tens1, TensorUsage::OFM, tens2));
        ops.push_back(CreateSchedulerOperation(true, TensorUsage::IFM, tens2, TensorUsage::OFM, tens3));
        ops.push_back(CreateSchedulerOperation(true, TensorUsage::IFM, tens3, TensorUsage::OFM, tens4));

        auto oldGraph = std::make_unique<Graph>(GraphNotation::TFLite);
        oldGraph->AddInput(tens1->srcTensor);
        tens1->isGraphInput = true;
        oldGraph->AddOutput(tens4->srcTensor);
        tens4->isGraphOutput = true;

        std::vector<std::pair<Operation *, std::unique_ptr<NPUOperation>>> npuOps;
        auto newGraph = PackScheduleToGraph(npuOps, ops, tensorAddressMap, oldGraph.get());

        REQUIRE(npuOps.size() == 1);
        REQUIRE(ops.size() == 0);
        REQUIRE(newGraph->ScheduledOrder().size() == 1);
        REQUIRE(newGraph->Inputs().size() == 1);
        REQUIRE(newGraph->Outputs().size() == 1);

        // Check packing
        REQUIRE(npuOps[0].second->Operations().size() == 3);

        // Check new graph operations
        REQUIRE(newGraph->ScheduledOrder()[0]->Type() == OpType::CustomNpuOp);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().contains(TensorUsage::OFM));

        // Check new graph connections
        REQUIRE(newGraph->ScheduledOrder()[0]->IFM(0) == newGraph->Inputs()[0].get());
        REQUIRE(newGraph->ScheduledOrder()[0]->OFM() == newGraph->Outputs()[0].get());

        // Check new graph I/O
        REQUIRE(newGraph->IsInput(newGraph->ScheduledOrder()[0]->IFM(0)));
        REQUIRE(newGraph->IsOutput(newGraph->ScheduledOrder()[0]->OFM()));
    }

    SECTION("All NPU with variable")
    {
        std::vector<std::unique_ptr<SchedulerOperation>> ops;
        ops.push_back(CreateSchedulerOperation(true, TensorUsage::IFM, tens1, TensorUsage::OFM, tens2));
        ops.push_back(CreateSchedulerOperation(true, TensorUsage::IFM, tens2, TensorUsage::IFM1, var1, TensorUsage::OFM, tens3));
        ops.push_back(CreateSchedulerOperation(true, TensorUsage::IFM, tens3, TensorUsage::OFM, tens4));

        auto oldGraph = std::make_unique<Graph>(GraphNotation::TFLite);
        oldGraph->AddInput(tens1->srcTensor);
        tens1->isGraphInput = true;
        oldGraph->AddOutput(tens4->srcTensor);
        tens4->isGraphOutput = true;
        oldGraph->AddPersistent(var1->srcTensor);
        var1->isPersistent = true;

        std::vector<std::pair<Operation *, std::unique_ptr<NPUOperation>>> npuOps;
        auto newGraph = PackScheduleToGraph(npuOps, ops, tensorAddressMap, oldGraph.get());

        REQUIRE(npuOps.size() == 1);
        REQUIRE(ops.size() == 0);
        REQUIRE(newGraph->ScheduledOrder().size() == 1);
        REQUIRE(newGraph->Inputs().size() == 1);
        REQUIRE(newGraph->Outputs().size() == 1);
        REQUIRE(newGraph->Persistent().size() == 1);

        // Check packing
        REQUIRE(npuOps[0].second->Operations().size() == 3);

        // Check new graph operations
        REQUIRE(newGraph->ScheduledOrder()[0]->Type() == OpType::CustomNpuOp);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().size() == 2);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().contains(TensorUsage::IFM1));
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().contains(TensorUsage::OFM));

        // Check new graph connections
        REQUIRE(newGraph->ScheduledOrder()[0]->IFM(0) == newGraph->Inputs()[0].get());
        REQUIRE(newGraph->ScheduledOrder()[0]->IFM(1) == newGraph->Persistent()[0].get());
        REQUIRE(newGraph->ScheduledOrder()[0]->OFM() == newGraph->Outputs()[0].get());

        // Check new graph I/O
        REQUIRE(newGraph->IsInput(newGraph->ScheduledOrder()[0]->IFM(0)));
        REQUIRE(newGraph->IsPersistent(newGraph->ScheduledOrder()[0]->IFM(1)));
        REQUIRE(newGraph->IsOutput(newGraph->ScheduledOrder()[0]->OFM()));
    }

    SECTION("All CPU")
    {
        std::vector<std::unique_ptr<SchedulerOperation>> ops;
        ops.push_back(CreateSchedulerOperation(false, TensorUsage::IFM, tens1, TensorUsage::Params, tens2, TensorUsage::OFM, tens3));
        ops.push_back(CreateSchedulerOperation(false, TensorUsage::IFM, tens3, TensorUsage::OFM, tens4));
        ops.push_back(CreateSchedulerOperation(false, TensorUsage::IFM, tens4, TensorUsage::OFM, tens5));

        auto oldGraph = std::make_unique<Graph>(GraphNotation::TFLite);
        oldGraph->AddInput(tens1->srcTensor);
        tens1->isGraphInput = true;
        oldGraph->AddInput(tens2->srcTensor);
        tens2->isGraphInput = true;
        oldGraph->AddOutput(tens5->srcTensor);
        tens5->isGraphOutput = true;

        std::vector<std::pair<Operation *, std::unique_ptr<NPUOperation>>> npuOps;
        auto newGraph = PackScheduleToGraph(npuOps, ops, tensorAddressMap, oldGraph.get());

        REQUIRE(npuOps.size() == 0);
        REQUIRE(ops.size() == 0);
        REQUIRE(newGraph->ScheduledOrder().size() == 3);
        REQUIRE(newGraph->Inputs().size() == 2);
        REQUIRE(newGraph->Outputs().size() == 1);

        // Check new graph operations
        REQUIRE(newGraph->ScheduledOrder()[0]->Type() == OpType::AvgPool);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().size() == 2);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().contains(TensorUsage::Params));
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().contains(TensorUsage::OFM));
        REQUIRE(newGraph->ScheduledOrder()[1]->Type() == OpType::AvgPool);
        REQUIRE(newGraph->ScheduledOrder()[1]->Inputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[1]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[1]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[1]->Outputs().contains(TensorUsage::OFM));
        REQUIRE(newGraph->ScheduledOrder()[2]->Type() == OpType::AvgPool);
        REQUIRE(newGraph->ScheduledOrder()[2]->Inputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[2]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[2]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[2]->Outputs().contains(TensorUsage::OFM));

        // Check new graph connections
        REQUIRE(newGraph->ScheduledOrder()[0]->IFM(0) == newGraph->Inputs()[0].get());
        REQUIRE(newGraph->ScheduledOrder()[0]->OFM() == newGraph->ScheduledOrder()[1]->IFM(0));
        REQUIRE(newGraph->ScheduledOrder()[1]->OFM() == newGraph->ScheduledOrder()[2]->IFM(0));
        REQUIRE(newGraph->ScheduledOrder()[2]->OFM() == newGraph->Outputs()[0].get());

        // Check new graph I/O
        REQUIRE(newGraph->IsInput(newGraph->ScheduledOrder()[0]->IFM(0)));
        REQUIRE(newGraph->IsOutput(newGraph->ScheduledOrder()[2]->OFM()));
    }

    SECTION("All CPU with variable")
    {
        std::vector<std::unique_ptr<SchedulerOperation>> ops;
        ops.push_back(CreateSchedulerOperation(false, TensorUsage::IFM, tens1, TensorUsage::Params, tens2, TensorUsage::OFM, tens3));
        ops.push_back(CreateSchedulerOperation(false, TensorUsage::IFM, tens3, TensorUsage::Params, var1, TensorUsage::OFM, tens4));
        ops.push_back(CreateSchedulerOperation(false, TensorUsage::IFM, tens4, TensorUsage::OFM, tens5));

        auto oldGraph = std::make_unique<Graph>(GraphNotation::TFLite);
        oldGraph->AddInput(tens1->srcTensor);
        tens1->isGraphInput = true;
        oldGraph->AddInput(tens2->srcTensor);
        tens2->isGraphInput = true;
        oldGraph->AddOutput(tens5->srcTensor);
        tens5->isGraphOutput = true;
        oldGraph->AddPersistent(var1->srcTensor);
        var1->isPersistent = true;

        std::vector<std::pair<Operation *, std::unique_ptr<NPUOperation>>> npuOps;
        auto newGraph = PackScheduleToGraph(npuOps, ops, tensorAddressMap, oldGraph.get());

        REQUIRE(npuOps.size() == 0);
        REQUIRE(ops.size() == 0);
        REQUIRE(newGraph->ScheduledOrder().size() == 3);
        REQUIRE(newGraph->Inputs().size() == 2);
        REQUIRE(newGraph->Outputs().size() == 1);
        REQUIRE(newGraph->Persistent().size() == 1);

        // Check new graph operations
        REQUIRE(newGraph->ScheduledOrder()[0]->Type() == OpType::AvgPool);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().size() == 2);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().contains(TensorUsage::Params));
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().contains(TensorUsage::OFM));
        REQUIRE(newGraph->ScheduledOrder()[1]->Type() == OpType::AvgPool);
        REQUIRE(newGraph->ScheduledOrder()[1]->Inputs().size() == 2);
        REQUIRE(newGraph->ScheduledOrder()[1]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[1]->Inputs().contains(TensorUsage::Params));
        REQUIRE(newGraph->ScheduledOrder()[1]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[1]->Outputs().contains(TensorUsage::OFM));
        REQUIRE(newGraph->ScheduledOrder()[2]->Type() == OpType::AvgPool);
        REQUIRE(newGraph->ScheduledOrder()[2]->Inputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[2]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[2]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[2]->Outputs().contains(TensorUsage::OFM));

        // Check new graph connections
        REQUIRE(newGraph->ScheduledOrder()[0]->IFM(0) == newGraph->Inputs()[0].get());
        REQUIRE(newGraph->ScheduledOrder()[0]->OFM() == newGraph->ScheduledOrder()[1]->IFM(0));
        REQUIRE(newGraph->ScheduledOrder()[1]->OFM() == newGraph->ScheduledOrder()[2]->IFM(0));
        REQUIRE(newGraph->ScheduledOrder()[2]->OFM() == newGraph->Outputs()[0].get());

        // Check new graph I/O
        REQUIRE(newGraph->IsInput(newGraph->ScheduledOrder()[0]->IFM(0)));
        REQUIRE(newGraph->IsPersistent(newGraph->ScheduledOrder()[1]->Input(TensorUsage::Params)->tensor.get()));
        REQUIRE(newGraph->IsOutput(newGraph->ScheduledOrder()[2]->OFM()));
    }

    SECTION("Mixed NPU/CPU with subop")
    {
        std::vector<std::unique_ptr<SchedulerOperation>> ops;
        ops.push_back(CreateSchedulerOperation(true, TensorUsage::IFM, tens1, TensorUsage::OFM, tens3));
        ops.back()->AddSubOp(CreateSchedulerOperation(true, TensorUsage::IFM, tens2, TensorUsage::OFM, tens3));
        ops.push_back(CreateSchedulerOperation(false, TensorUsage::IFM, tens3, TensorUsage::OFM, tens4));

        auto oldGraph = std::make_unique<Graph>(GraphNotation::TFLite);
        oldGraph->AddInput(tens1->srcTensor);
        tens1->isGraphInput = true;
        oldGraph->AddOutput(tens4->srcTensor);
        tens4->isGraphOutput = true;

        std::vector<std::pair<Operation *, std::unique_ptr<NPUOperation>>> npuOps;
        auto newGraph = PackScheduleToGraph(npuOps, ops, tensorAddressMap, oldGraph.get());

        REQUIRE(npuOps.size() == 1);
        REQUIRE(ops.size() == 0);
        REQUIRE(newGraph->ScheduledOrder().size() == 2);
        REQUIRE(newGraph->Inputs().size() == 1);
        REQUIRE(newGraph->Outputs().size() == 1);

        // Check new graph I/O
        REQUIRE(newGraph->IsInput(newGraph->ScheduledOrder()[0]->IFM(0)));
        REQUIRE(newGraph->IsOutput(newGraph->ScheduledOrder()[1]->OFM()));

        // Check new graph operations
        REQUIRE(newGraph->ScheduledOrder()[0]->Type() == OpType::CustomNpuOp);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().contains(TensorUsage::OFM));
        REQUIRE(newGraph->ScheduledOrder()[1]->Type() == OpType::AvgPool);
        REQUIRE(newGraph->ScheduledOrder()[1]->Inputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[1]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[1]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[1]->Outputs().contains(TensorUsage::OFM));

        // Check new graph connections
        REQUIRE(newGraph->ScheduledOrder()[0]->IFM(0) == newGraph->Inputs()[0].get());
        REQUIRE(newGraph->ScheduledOrder()[0]->OFM() == newGraph->ScheduledOrder()[1]->IFM(0));
        REQUIRE(newGraph->ScheduledOrder()[1]->OFM() == newGraph->Outputs()[0].get());
    }

    SECTION("Mixed NPU/CPU with subop and variable")
    {
        std::vector<std::unique_ptr<SchedulerOperation>> ops;
        ops.push_back(CreateSchedulerOperation(true, TensorUsage::IFM, tens1, TensorUsage::OFM, tens3));
        ops.back()->AddSubOp(CreateSchedulerOperation(true, TensorUsage::IFM, tens2, TensorUsage::OFM, tens3));
        ops.push_back(CreateSchedulerOperation(false, TensorUsage::IFM, tens3, TensorUsage::IFM1, var1, TensorUsage::OFM, tens4));

        auto oldGraph = std::make_unique<Graph>(GraphNotation::TFLite);
        oldGraph->AddInput(tens1->srcTensor);
        tens1->isGraphInput = true;
        oldGraph->AddOutput(tens4->srcTensor);
        tens4->isGraphOutput = true;
        oldGraph->AddPersistent(var1->srcTensor);
        var1->isPersistent = true;

        std::vector<std::pair<Operation *, std::unique_ptr<NPUOperation>>> npuOps;
        auto newGraph = PackScheduleToGraph(npuOps, ops, tensorAddressMap, oldGraph.get());

        REQUIRE(npuOps.size() == 1);
        REQUIRE(ops.size() == 0);
        REQUIRE(newGraph->ScheduledOrder().size() == 2);
        REQUIRE(newGraph->Inputs().size() == 1);
        REQUIRE(newGraph->Outputs().size() == 1);
        REQUIRE(newGraph->Persistent().size() == 1);


        // Check new graph I/O
        REQUIRE(newGraph->IsInput(newGraph->ScheduledOrder()[0]->IFM(0)));
        REQUIRE(newGraph->IsOutput(newGraph->ScheduledOrder()[1]->OFM()));
        REQUIRE(newGraph->IsPersistent(newGraph->ScheduledOrder()[1]->IFM(1)));

        // Check new graph operations
        REQUIRE(newGraph->ScheduledOrder()[0]->Type() == OpType::CustomNpuOp);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().contains(TensorUsage::OFM));
        REQUIRE(newGraph->ScheduledOrder()[1]->Type() == OpType::AvgPool);
        REQUIRE(newGraph->ScheduledOrder()[1]->Inputs().size() == 2);
        REQUIRE(newGraph->ScheduledOrder()[1]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[1]->Inputs().contains(TensorUsage::IFM1));
        REQUIRE(newGraph->ScheduledOrder()[1]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[1]->Outputs().contains(TensorUsage::OFM));

        // Check new graph connections
        REQUIRE(newGraph->ScheduledOrder()[0]->IFM(0) == newGraph->Inputs()[0].get());
        REQUIRE(newGraph->ScheduledOrder()[0]->OFM() == newGraph->ScheduledOrder()[1]->IFM(0));
        REQUIRE(newGraph->ScheduledOrder()[1]->OFM() == newGraph->Outputs()[0].get());
    }

    SECTION("NPU to NPU connection")
    {
        std::vector<std::unique_ptr<SchedulerOperation>> ops;
        ops.push_back(CreateSchedulerOperation(true, TensorUsage::IFM, tens1, TensorUsage::OFM, tens2));
        ops.push_back(CreateSchedulerOperation(false, TensorUsage::IFM, tens3, TensorUsage::OFM, tens4));
        ops.push_back(CreateSchedulerOperation(true, TensorUsage::IFM0, tens2, TensorUsage::IFM1, tens4, TensorUsage::OFM, tens5));

        auto oldGraph = std::make_unique<Graph>(GraphNotation::TFLite);
        oldGraph->AddInput(tens1->srcTensor);
        tens1->isGraphInput = true;
        oldGraph->AddInput(tens3->srcTensor);
        tens3->isGraphInput = true;
        oldGraph->AddOutput(tens5->srcTensor);
        tens5->isGraphOutput = true;

        std::vector<std::pair<Operation *, std::unique_ptr<NPUOperation>>> npuOps;
        auto newGraph = PackScheduleToGraph(npuOps, ops, tensorAddressMap, oldGraph.get());

        REQUIRE(npuOps.size() == 2);
        REQUIRE(ops.size() == 0);
        REQUIRE(newGraph->ScheduledOrder().size() == 3);
        REQUIRE(newGraph->Inputs().size() == 2);
        REQUIRE(newGraph->Outputs().size() == 1);

        // Check new graph I/O
        REQUIRE(newGraph->IsInput(newGraph->ScheduledOrder()[0]->IFM(0)));
        REQUIRE(newGraph->IsInput(newGraph->ScheduledOrder()[1]->IFM(0)));
        REQUIRE(newGraph->IsOutput(newGraph->ScheduledOrder()[2]->OFM()));

        // Check new graph operations
        REQUIRE(newGraph->ScheduledOrder()[0]->Type() == OpType::CustomNpuOp);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().contains(TensorUsage::OFM));
        REQUIRE(newGraph->ScheduledOrder()[1]->Type() == OpType::AvgPool);
        REQUIRE(newGraph->ScheduledOrder()[1]->Inputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[1]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[1]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[1]->Outputs().contains(TensorUsage::OFM));
        REQUIRE(newGraph->ScheduledOrder()[2]->Type() == OpType::CustomNpuOp);
        REQUIRE(newGraph->ScheduledOrder()[2]->Inputs().size() == 2);
        REQUIRE(newGraph->ScheduledOrder()[2]->Inputs().contains(TensorUsage::IFM0));
        REQUIRE(newGraph->ScheduledOrder()[2]->Inputs().contains(TensorUsage::IFM1));
        REQUIRE(newGraph->ScheduledOrder()[2]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[2]->Outputs().contains(TensorUsage::OFM));

        // Check new graph connections
        REQUIRE(newGraph->ScheduledOrder()[0]->IFM(0) == newGraph->Inputs()[0].get());
        REQUIRE(newGraph->ScheduledOrder()[0]->OFM() == newGraph->ScheduledOrder()[2]->IFM(0));
        REQUIRE(newGraph->ScheduledOrder()[1]->IFM(0) == newGraph->Inputs()[1].get());
        REQUIRE(newGraph->ScheduledOrder()[1]->OFM() == newGraph->ScheduledOrder()[2]->IFM(1));
        REQUIRE(newGraph->ScheduledOrder()[2]->OFM() == newGraph->Outputs()[0].get());
    }

    SECTION("NPU to NPU connection with variable")
    {
        std::vector<std::unique_ptr<SchedulerOperation>> ops;
        ops.push_back(CreateSchedulerOperation(true, TensorUsage::IFM, tens1, TensorUsage::Params, var1, TensorUsage::OFM, tens2));
        ops.push_back(CreateSchedulerOperation(false, TensorUsage::IFM, tens3, TensorUsage::OFM, tens4));
        ops.push_back(CreateSchedulerOperation(true, TensorUsage::IFM0, tens2, TensorUsage::IFM1, tens4, TensorUsage::OFM, tens5));

        auto oldGraph = std::make_unique<Graph>(GraphNotation::TFLite);
        oldGraph->AddInput(tens1->srcTensor);
        tens1->isGraphInput = true;
        oldGraph->AddInput(tens3->srcTensor);
        tens3->isGraphInput = true;
        oldGraph->AddOutput(tens5->srcTensor);
        tens5->isGraphOutput = true;
        oldGraph->AddPersistent(var1->srcTensor);
        var1->isPersistent = true;

        std::vector<std::pair<Operation *, std::unique_ptr<NPUOperation>>> npuOps;
        auto newGraph = PackScheduleToGraph(npuOps, ops, tensorAddressMap, oldGraph.get());

        REQUIRE(npuOps.size() == 2);
        REQUIRE(ops.size() == 0);
        REQUIRE(newGraph->ScheduledOrder().size() == 3);
        REQUIRE(newGraph->Inputs().size() == 2);
        REQUIRE(newGraph->Outputs().size() == 1);
        REQUIRE(newGraph->Persistent().size() == 1);

        // Check new graph I/O
        REQUIRE(newGraph->IsInput(newGraph->ScheduledOrder()[0]->IFM(0)));
        REQUIRE(newGraph->IsInput(newGraph->ScheduledOrder()[1]->IFM(0)));
        REQUIRE(newGraph->IsOutput(newGraph->ScheduledOrder()[2]->OFM()));
        REQUIRE(newGraph->IsPersistent(newGraph->ScheduledOrder()[0]->Input(TensorUsage::IFM1)->tensor.get()));

        // Check new graph operations
        REQUIRE(newGraph->ScheduledOrder()[0]->Type() == OpType::CustomNpuOp);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().size() == 2);
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[0]->Inputs().contains(TensorUsage::IFM1));
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[0]->Outputs().contains(TensorUsage::OFM));
        REQUIRE(newGraph->ScheduledOrder()[1]->Type() == OpType::AvgPool);
        REQUIRE(newGraph->ScheduledOrder()[1]->Inputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[1]->Inputs().contains(TensorUsage::IFM));
        REQUIRE(newGraph->ScheduledOrder()[1]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[1]->Outputs().contains(TensorUsage::OFM));
        REQUIRE(newGraph->ScheduledOrder()[2]->Type() == OpType::CustomNpuOp);
        REQUIRE(newGraph->ScheduledOrder()[2]->Inputs().size() == 2);
        REQUIRE(newGraph->ScheduledOrder()[2]->Inputs().contains(TensorUsage::IFM0));
        REQUIRE(newGraph->ScheduledOrder()[2]->Inputs().contains(TensorUsage::IFM1));
        REQUIRE(newGraph->ScheduledOrder()[2]->Outputs().size() == 1);
        REQUIRE(newGraph->ScheduledOrder()[2]->Outputs().contains(TensorUsage::OFM));

        // Check new graph connections
        REQUIRE(newGraph->ScheduledOrder()[0]->IFM(0) == newGraph->Inputs()[0].get());
        REQUIRE(newGraph->ScheduledOrder()[0]->IFM(1) == newGraph->Persistent()[0].get());
        REQUIRE(newGraph->ScheduledOrder()[0]->OFM() == newGraph->ScheduledOrder()[2]->IFM(0));
        REQUIRE(newGraph->ScheduledOrder()[1]->IFM(0) == newGraph->Inputs()[1].get());
        REQUIRE(newGraph->ScheduledOrder()[1]->OFM() == newGraph->ScheduledOrder()[2]->IFM(1));
        REQUIRE(newGraph->ScheduledOrder()[2]->OFM() == newGraph->Outputs()[0].get());
    }
}
