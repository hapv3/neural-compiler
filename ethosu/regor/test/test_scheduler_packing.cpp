//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "architecture/ethosu85/ethos_u85.hpp"
#include "compiler/scheduler_packing.hpp"
#include "util.hpp"

#include <fmt/format.h>
#include <catch_all.hpp>

#include "regor.h"

using namespace regor;


TEST_CASE("Pack operation with axis attribute")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    // Create packing
    const std::unordered_map<UniqueId, UniqueId> emptyTensorEquivalenceIdMap;
    auto packing = SchedulerPacking(arch.get(), false, emptyTensorEquivalenceIdMap);
    SECTION("Validate axis after packing")
    {
        // Perform packing on an ArgMax operation
        // Validate that attr_axis still represents the reduced axis.
        std::vector<std::shared_ptr<Operation>> ops;
        auto ifm = CreateTensor("IFM", Shape(10, 10), DataType::Int8);
        auto ofm = CreateTensor("OFM", Shape(1, 10), DataType::Int32);
        auto op = CreateOperation(OpType::ArgMax, TensorUsage::IFM, ifm, TensorUsage::OFM, ofm);
        auto attr = op->Attribute<axis_attr_t>();
        attr->axis = 0;
        ops.push_back(std::move(op));

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

    SECTION("Pack sliced operation with axis")
    {
        // Perform packing on two sliced ArgMax operations
        // Validate that attr_axis still represent the reduced axes.
        std::vector<std::shared_ptr<Operation>> ops;
        auto ifm = CreateTensor("IFM", Shape(10, 10, 10), DataType::Int8);
        auto ofm = CreateTensor("OFM", Shape(10, 2, 10), DataType::Int32);

        // first op
        //  reads  0,0,0 - shape 10,5,10
        //  writes 0,0,0 - shape 10,1,10
        // second op
        //  reads  0,5,0 - shape 10,5,10
        //  writes 0,1,0 - shape 10,1,10
        for ( int i = 0; i < 2; i++ )
        {
            auto op = CreateOperation(OpType::ArgMax, TensorUsage::IFM, ifm, TensorUsage::OFM, ofm);
            auto attr = op->Attribute<axis_attr_t>();
            attr->axis = 1;
            TensorSlice ifmSlice{Shape(0, 5 * i, 0), Shape(10, 5, 10)};
            TensorSlice ofmSlice{Shape(0, i, 0), Shape(10, 1, 10)};
            op->Input(TensorUsage::IFM)->Set(ifmSlice);
            op->Output(TensorUsage::OFM)->Set(ofmSlice);
            ops.push_back(std::move(op));
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

TEST_CASE("Test activation fusing")
{
    // Activations can be fused to previous operation
    // as long as they don't perform any rescaling.
    // This is represented by ifm and ofm quantization
    // both being unit

    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    // Create packing
    const std::unordered_map<UniqueId, UniqueId> emptyTensorEquivalenceIdMap;
    auto packing = SchedulerPacking(arch.get(), false, emptyTensorEquivalenceIdMap);

    // create ops
    // ABS: primary operation
    // Relu: Activation considered for fusing
    std::vector<std::shared_ptr<Operation>> ops;
    auto ifm = CreateTensor("IFM", Shape(10, 10, 10), DataType::Int8);
    auto ofm = CreateTensor("OFM", Shape(10, 10, 10), DataType::Int8);
    auto actofm = CreateTensor("ACTOFM", Shape(10, 10, 10), DataType::Int8);
    auto op1 = CreateOperation(OpType::Abs, TensorUsage::IFM, ifm, TensorUsage::OFM, ofm);
    ops.push_back(std::move(op1));
    auto op2 = CreateOperation(OpType::Relu, TensorUsage::IFM, ofm, TensorUsage::OFM, actofm);
    ops.push_back(op2);
    SECTION("Don't fuse non-unit quant")
    {
        // Fusing requires unit IFM and OFM quant
        // validate that we don't fuse symmetric
        auto *reluIfmConn = op2->Input(TensorUsage::IFM);
        auto *reluOfmConn = op2->Output(TensorUsage::OFM);
        Quantization ifmQuant;
        ifmQuant.scales.push_back({1, 5});
        ifmQuant.type = QuantizationType::EXPLICIT;
        reluIfmConn->quantization = std::move(ifmQuant);

        Quantization ofmQuant;
        ofmQuant.scales.push_back({1, 5});
        ofmQuant.type = QuantizationType::EXPLICIT;
        reluOfmConn->quantization = std::move(ofmQuant);

        // Create graph with ops
        auto graph = CreateGraph(ops);
        // Perform scheduler_packing
        auto schedOps = packing.Process(graph.get());
        // no activation fusing
        REQUIRE(schedOps.size() == 2);
    }
    SECTION("Fuse unit quant")
    {
        // Fusing requires unit IFM and OFM quant
        // validate that we fuse unit even in the asymmetric case (one can be empty)
        auto *reluIfmConn = op2->Input(TensorUsage::IFM);
        auto *reluOfmConn = op2->Output(TensorUsage::OFM);
        Quantization ifmQuant;
        ifmQuant.scales.push_back({1, 0});
        ifmQuant.type = QuantizationType::EXPLICIT;
        reluIfmConn->quantization = std::move(ifmQuant);

        Quantization ofmQuant;
        ofmQuant.type = QuantizationType::EXPLICIT;
        reluOfmConn->quantization = std::move(ofmQuant);

        // Create graph with ops
        auto graph = CreateGraph(ops);
        // Perform scheduler_packing
        auto schedOps = packing.Process(graph.get());
        // Activation fusing
        REQUIRE(schedOps.size() == 1);

        // Validate that the second op is packed as subop of first
        auto &abs = schedOps[0];
        REQUIRE(abs->Type() == OpType::Abs);
        REQUIRE(abs->SubOps().size() == 1);
        auto &relu = abs->SubOps()[0];
        REQUIRE(relu->Type() == OpType::Relu);

        // Validate the primary op
        auto *absIfmConn = abs->Input(TensorUsage::IFM);
        auto *absOfmConn = abs->Output(TensorUsage::OFM);
        REQUIRE(absIfmConn->tensor->Name() == "IFM");
        REQUIRE(absIfmConn->tensor->producers.empty());
        REQUIRE(absIfmConn->tensor->consumers.size() == 1);
        REQUIRE(absIfmConn->tensor->consumers[0] == abs.get());
        REQUIRE(absOfmConn->tensor->Name() == "ACTOFM");
        REQUIRE(absOfmConn->tensor->producers.size() == 1);
        REQUIRE(absOfmConn->tensor->producers[0] == abs.get());
        REQUIRE(absOfmConn->tensor->consumers.empty());

        // Validate the activation op
        auto *actIfmConn = relu->Input(TensorUsage::IFM);
        auto *actOfmConn = relu->Output(TensorUsage::OFM);
        REQUIRE(actIfmConn->tensor->Name() == "OFM");
        REQUIRE(actIfmConn->tensor->producers.empty());
        REQUIRE(actIfmConn->tensor->consumers.size() == 1);
        REQUIRE(actIfmConn->tensor->consumers[0] == relu.get());
        REQUIRE(actOfmConn->tensor->Name() == "ACTOFM");
        REQUIRE(actOfmConn->tensor->producers.size() == 1);
        REQUIRE(actOfmConn->tensor->producers[0] == abs.get());
        REQUIRE(actOfmConn->tensor->consumers.empty());
    }
    SECTION("Fuse reshaped activation")
    {
        auto *reluIfmConn = op2->Input(TensorUsage::IFM);
        auto *reluOfmConn = op2->Output(TensorUsage::OFM);
        reluIfmConn->shape = {1, 100, 10};
        reluOfmConn->shape = {1, 100, 10};

        // Create graph with ops
        auto graph = CreateGraph(ops);
        // Perform scheduler_packing
        auto schedOps = packing.Process(graph.get());
        // Activation fusing
        REQUIRE(schedOps.size() == 1);

        // Validate that the second op is packed as subop of first
        auto &abs = schedOps[0];
        REQUIRE(abs->Type() == OpType::Abs);
        REQUIRE(abs->SubOps().size() == 1);
        auto &relu = abs->SubOps()[0];
        REQUIRE(relu->Type() == OpType::Relu);

        // Validate the primary op
        auto *absIfmConn = abs->Input(TensorUsage::IFM);
        auto *absOfmConn = abs->Output(TensorUsage::OFM);
        REQUIRE(absIfmConn->tensor->Name() == "IFM");
        REQUIRE(absIfmConn->tensor->producers.empty());
        REQUIRE(absIfmConn->tensor->consumers.size() == 1);
        REQUIRE(absIfmConn->tensor->consumers[0] == abs.get());
        REQUIRE(absIfmConn->shape == Shape(1, 10, 10, 10));  // Original shape
        REQUIRE(absOfmConn->tensor->Name() == "ACTOFM");
        REQUIRE(absOfmConn->tensor->producers.size() == 1);
        REQUIRE(absOfmConn->tensor->producers[0] == abs.get());
        REQUIRE(absOfmConn->tensor->consumers.empty());
        REQUIRE(absOfmConn->shape == Shape(1, 10, 10, 10));  // Original shape

        // Validate the activation op
        auto *actIfmConn = relu->Input(TensorUsage::IFM);
        auto *actOfmConn = relu->Output(TensorUsage::OFM);
        REQUIRE(actIfmConn->tensor->Name() == "OFM");
        REQUIRE(actIfmConn->tensor->producers.empty());
        REQUIRE(actIfmConn->tensor->consumers.size() == 1);
        REQUIRE(actIfmConn->tensor->consumers[0] == relu.get());
        REQUIRE(actIfmConn->shape == Shape(1, 10, 10, 10));  // Inherited shape from ABS
        REQUIRE(actOfmConn->tensor->Name() == "ACTOFM");
        REQUIRE(actOfmConn->tensor->producers.size() == 1);
        REQUIRE(actOfmConn->tensor->producers[0] == abs.get());
        REQUIRE(actOfmConn->tensor->consumers.empty());
        REQUIRE(actOfmConn->shape == Shape(1, 10, 10, 10));  // Inherited shape from ABS
    }
}

TEST_CASE("Pack operation with resource data type")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    // Create packing
    const std::unordered_map<UniqueId, UniqueId> emptyTensorEquivalenceIdMap;
    auto packing = SchedulerPacking(arch.get(), false, emptyTensorEquivalenceIdMap);

    std::vector<std::shared_ptr<Operation>> ops;
    auto ifm = CreateTensor("IFM", Shape(), DataType::Resource);
    auto ofm = CreateTensor("OFM", Shape(10, 10, 10), DataType::Int8);
    auto op1 = CreateOperation(OpType::Passthrough, TensorUsage::IFM, ifm, TensorUsage::OFM, ofm);
    ops.push_back(std::move(op1));

    // Create graph with ops
    auto graph = CreateGraph(ops);

    // Perform scheduler_packing
    auto schedOps = packing.Process(graph.get());
    REQUIRE(schedOps.size() == 1);

    // Validate the op and tensors
    auto &schedOp1 = schedOps[0];
    REQUIRE(schedOp1->Type() == OpType::Passthrough);
    auto *ifmConn = schedOp1->Input(TensorUsage::IFM);
    auto *ofmConn = schedOp1->Output(TensorUsage::OFM);
    REQUIRE(ifmConn->tensor->Name() == "IFM");
    REQUIRE(ifmConn->tensor->dataType == DataType::Resource);
    REQUIRE(ifmConn->tensor->producers.empty());
    REQUIRE(ifmConn->tensor->consumers.size() == 1);
    REQUIRE(ifmConn->tensor->consumers[0] == schedOp1.get());
    REQUIRE(ofmConn->tensor->Name() == "OFM");
    REQUIRE(ofmConn->tensor->dataType == DataType::Int8);
    REQUIRE(ofmConn->tensor->producers.size() == 1);
    REQUIRE(ofmConn->tensor->producers[0] == schedOp1.get());
    REQUIRE(ofmConn->tensor->consumers.empty());
}

TEST_CASE("Pack CPU operation with output that is graph output, but also consumed by NPU")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    // Create packing
    const std::unordered_map<UniqueId, UniqueId> emptyTensorEquivalenceIdMap;
    auto packing = SchedulerPacking(arch.get(), false, emptyTensorEquivalenceIdMap);

    std::vector<std::shared_ptr<Operation>> ops;
    auto ifm = CreateTensor("IFM", Shape(10, 10, 10), DataType::Int8);
    auto mid = CreateTensor("MID", Shape(10, 10, 10), DataType::Int8);
    auto ofm = CreateTensor("OFM", Shape(10, 10, 10), DataType::Int8);
    auto ofm2 = CreateTensor("OFM2", Shape(10, 10, 10), DataType::Int8);

    auto op1 = CreateOperation(OpType::Abs, TensorUsage::IFM, ifm, TensorUsage::OFM, mid);
    ops.push_back(std::move(op1));
    auto op2 = CreateOperation(OpType::Passthrough, TensorUsage::IFM, mid, TensorUsage::OFM, ofm);
    ops.push_back(std::move(op2));
    auto op3 = CreateOperation(OpType::Abs, TensorUsage::IFM, ofm, TensorUsage::OFM, ofm2);
    ops.push_back(std::move(op3));

    // Create graph with ops
    auto graph = CreateGraph(ops);

    // Mark CPU op OFM as graph output
    graph->AddOutput(ofm);

    // Perform scheduler_packing
    auto schedOps = packing.Process(graph.get());
    REQUIRE(schedOps.size() == 3);

    // Ensure operation order is unchanged
    REQUIRE(schedOps[0]->Type() == OpType::Abs);
    REQUIRE(schedOps[0]->SubOps().empty());
    REQUIRE(schedOps[1]->Type() == OpType::Passthrough);
    REQUIRE(schedOps[1]->SubOps().empty());
    REQUIRE(schedOps[2]->Type() == OpType::Abs);
    REQUIRE(schedOps[2]->SubOps().empty());
}
