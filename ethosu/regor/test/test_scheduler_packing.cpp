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

#include "common/common.hpp"

#include "architecture/ethosu85/ethos_u85.hpp"
#include "compiler/scheduler_packing.hpp"
#include "util.hpp"

#include <fmt/format.h>
#include <catch_all.hpp>

#include "regor.h"

using namespace regor;


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

    SECTION("Pack sliced operation (with axis)")
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

    SECTION("Pack operation and activation")
    {
        std::vector<std::shared_ptr<Operation>> ops;
        auto ifm = CreateTensor("IFM", Shape(10, 10, 10), DataType::Int8);
        auto ofm = CreateTensor("OFM", Shape(10, 10, 10), DataType::Int8);
        auto actofm = CreateTensor("ACTOFM", Shape(10, 10, 10), DataType::Int8);
        auto op1 = CreateOperation(OpType::Abs, TensorUsage::IFM, ifm, TensorUsage::OFM, ofm);
        ops.push_back(std::move(op1));
        auto op2 = CreateOperation(OpType::Relu, TensorUsage::IFM, ofm, TensorUsage::OFM, actofm);
        ops.push_back(std::move(op2));

        // Create graph with ops
        auto graph = CreateGraph(ops);

        // Perform scheduler_packing
        auto schedOps = packing.Process(graph.get());
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
        auto *reluIfmConn = relu->Input(TensorUsage::IFM);
        auto *reluOfmConn = relu->Output(TensorUsage::OFM);
        REQUIRE(reluIfmConn->tensor->Name() == "OFM");
        REQUIRE(reluIfmConn->tensor->producers.empty());
        REQUIRE(reluIfmConn->tensor->consumers.size() == 1);
        REQUIRE(reluIfmConn->tensor->consumers[0] == relu.get());
        REQUIRE(reluOfmConn->tensor->Name() == "ACTOFM");
        REQUIRE(reluOfmConn->tensor->producers.size() == 1);
        REQUIRE(reluOfmConn->tensor->producers[0] == abs.get());
        REQUIRE(reluOfmConn->tensor->consumers.empty());
    }
}
