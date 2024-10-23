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

#include "architecture/ethosu85/ethos_u85.hpp"
#include "compiler/graphir_optimiser.hpp"
#include "compiler/scheduler_packing.hpp"
#include "compiler/tensor_properties.hpp"
#include "util.hpp"

#include <fmt/format.h>
#include <catch_all.hpp>

#include "regor.h"

using namespace regor;


TEST_CASE("test_graphir_optimiser - constant propagation")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    SECTION("SHL operation")
    {
        auto graph = [&]()
        {
            std::vector<std::shared_ptr<Operation>> ops;
            auto cifm = CreateTensor("CIFM", Shape(1, 1, 1, 10), DataType::Int8, 1);
            auto cifm1 = CreateTensor("CIFM1", Shape(1, 1, 10, 1), DataType::Int8, 2);
            auto cofm = CreateTensor("COFM", Shape(1, 1, 10, 10), DataType::Int8);
            auto ifm = CreateTensor("IFM", Shape(1, 1, 10, 10), DataType::Int8);
            auto ofm = CreateTensor("OFM", Shape(1, 1, 10, 10), DataType::Int8);
            auto cop = CreateOperation(OpType::SHL, TensorUsage::IFM, cifm, TensorUsage::IFM1, cifm1, TensorUsage::OFM, cofm);
            auto op = CreateOperation(OpType::Add, TensorUsage::IFM, ifm, TensorUsage::IFM1, cofm, TensorUsage::OFM, ofm);
            ops.push_back(std::move(cop));
            ops.push_back(std::move(op));

            // Create graph with ops
            return CreateGraph(ops);
        }();

        GraphOptimiserOptions options;
        auto optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch->Constraints(), options, nullptr);

        std::vector<Operation *> allOps;

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 2);

        optimiser->Process(graph.get());
        allOps.clear();

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 1);
        REQUIRE(allOps[0]->Inputs()[TensorUsage::IFM1].tensor->IsConstant());
        auto iview = allOps[0]->Inputs()[TensorUsage::IFM1].tensor->View();
        auto idata = iview.RawData<int8_t>();
        for ( int i = 0; i < allOps[0]->Inputs()[TensorUsage::IFM1].tensor->StorageShape().Elements(); i++ )
        {
            REQUIRE(idata[i] == 1 << 2);
        }
    }

    SECTION("Traversal order")
    {
        auto graph = [&]()
        {
            std::vector<std::shared_ptr<Operation>> ops;
            auto cifm = CreateTensor("CIFM", Shape(1, 1, 1, 10), DataType::Int8, 1);
            auto cifm1 = CreateTensor("CIFM1", Shape(1, 1, 10, 1), DataType::Int8, 2);
            auto cifm2 = CreateTensor("CIFM2", Shape(1, 1, 10, 1), DataType::Int8, 3);
            auto cofm = CreateTensor("COFM", Shape(1, 1, 10, 10), DataType::Int8);
            auto cofm2 = CreateTensor("COFM2", Shape(1, 1, 10, 10), DataType::Int8);
            auto ifm = CreateTensor("IFM", Shape(1, 1, 10, 10), DataType::Int8);
            auto ofm = CreateTensor("OFM", Shape(1, 1, 10, 10), DataType::Int8);
            auto cop = CreateOperation(OpType::SHL, TensorUsage::IFM, cifm, TensorUsage::IFM1, cifm1, TensorUsage::OFM, cofm);
            auto cop2 = CreateOperation(OpType::SHL, TensorUsage::IFM, cofm, TensorUsage::IFM1, cifm2, TensorUsage::OFM, cofm2);
            auto op = CreateOperation(OpType::Add, TensorUsage::IFM, ifm, TensorUsage::IFM1, cofm2, TensorUsage::OFM, ofm);
            ops.push_back(std::move(cop));
            ops.push_back(std::move(cop2));
            ops.push_back(std::move(op));

            // Create graph with ops
            return CreateGraph(ops);
        }();

        GraphOptimiserOptions options;
        auto optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch->Constraints(), options, nullptr);

        std::vector<Operation *> allOps;

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 3);

        optimiser->Process(graph.get());
        allOps.clear();

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 1);
        REQUIRE(allOps[0]->Inputs()[TensorUsage::IFM1].tensor->IsConstant());
        auto iview = allOps[0]->Inputs()[TensorUsage::IFM1].tensor->View();
        auto idata = iview.RawData<int8_t>();
        for ( int i = 0; i < allOps[0]->Inputs()[TensorUsage::IFM1].tensor->StorageShape().Elements(); i++ )
        {
            REQUIRE(idata[i] == (1 << 2) << 3);
        }
    }
}
