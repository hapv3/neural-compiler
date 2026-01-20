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

#include "architecture/ethosu55/ethos_u55.hpp"
#include "architecture/ethosu85/ethos_u85.hpp"
#include "compiler/graphir_optimiser.hpp"
#include "compiler/optimiser_utils.hpp"
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
        const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);

        std::vector<Operation *> allOps;

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 2);

        REQUIRE(!optimiser.empty());
        optimiser.back()->Process(graph.get());
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

    SECTION("MemoryCopy operation")
    {
        auto graph = [&]()
        {
            std::vector<std::shared_ptr<Operation>> ops;
            auto cifm = CreateTensor("CIFM", Shape(1, 1, 1, 10), DataType::Int8, 1);
            auto cofm = CreateTensor("COFM", Shape(1, 1, 10, 10), DataType::Int8);
            auto ifm = CreateTensor("IFM", Shape(1, 1, 10, 10), DataType::Int8);
            auto ofm = CreateTensor("OFM", Shape(1, 1, 10, 10), DataType::Int8);
            auto cop = CreateOperation(OpType::MemoryCopy, TensorUsage::IFM, cifm, TensorUsage::OFM, cofm);
            auto op = CreateOperation(OpType::Add, TensorUsage::IFM, ifm, TensorUsage::IFM1, cofm, TensorUsage::OFM, ofm);
            ops.push_back(std::move(cop));
            ops.push_back(std::move(op));

            // Create graph with ops
            return CreateGraph(ops);
        }();

        GraphOptimiserOptions options;
        const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);

        std::vector<Operation *> allOps;

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 2);

        REQUIRE(!optimiser.empty());
        optimiser.back()->Process(graph.get());
        allOps.clear();

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 1);
        REQUIRE(allOps[0]->Inputs()[TensorUsage::IFM1].tensor->IsConstant());
        auto iview = allOps[0]->Inputs()[TensorUsage::IFM1].tensor->View();
        auto idata = iview.RawData<int8_t>();
        for ( int i = 0; i < allOps[0]->Inputs()[TensorUsage::IFM1].tensor->StorageShape().Elements(); i++ )
        {
            REQUIRE(idata[i] == 1);
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
        const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);

        std::vector<Operation *> allOps;

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 3);

        REQUIRE(!optimiser.empty());
        optimiser.back()->Process(graph.get());
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

    SECTION("Constant graph")
    {
        auto graph = [&]()
        {
            std::vector<std::shared_ptr<Operation>> ops;
            auto cifm = CreateTensor("CIFM1", Shape(1, 1, 10, 1), DataType::Int8, 2);
            auto cifm1 = CreateTensor("CIFM2", Shape(1, 1, 10, 1), DataType::Int8, 3);
            auto ofm = CreateTensor("OFM", Shape(1, 1, 10, 10), DataType::Int8);
            auto cop = CreateOperation(OpType::SHL, TensorUsage::IFM, cifm, TensorUsage::IFM1, cifm1, TensorUsage::OFM, ofm);
            ops.push_back(std::move(cop));

            // Create graph with ops
            return CreateGraph(ops);
        }();

        GraphOptimiserOptions options;
        const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);

        std::vector<Operation *> allOps;

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 1);

        REQUIRE(!optimiser.empty());
        optimiser.back()->Process(graph.get());
        allOps.clear();

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 1);
        REQUIRE(allOps[0]->Type() == OpType::MemoryCopy);
        REQUIRE(allOps[0]->Inputs()[TensorUsage::IFM].tensor->IsConstant());
        auto iview = allOps[0]->Inputs()[TensorUsage::IFM].tensor->View();
        auto idata = iview.RawData<int8_t>();
        for ( int i = 0; i < allOps[0]->Inputs()[TensorUsage::IFM].tensor->StorageShape().Elements(); i++ )
        {
            REQUIRE(idata[i] == 2 << 3);
        }
    }

    SECTION("Quantize operation")
    {
        auto graph = [&]()
        {
            std::vector<std::shared_ptr<Operation>> ops;
            std::vector<int8_t> cifmData{3, -24, -30, -13, 7, -5, 0, -6, 9, 20};
            auto cifm = CreateTensor("CIFM", Shape(1, 1, 1, 10), DataType::Int8, std::move(cifmData));
            auto cofm = CreateTensor("COFM", Shape(1, 1, 10, 10), DataType::Int8);
            auto ifm = CreateTensor("IFM", Shape(1, 1, 10, 10), DataType::Int8);
            auto ofm = CreateTensor("OFM", Shape(1, 1, 10, 10), DataType::Int8);
            auto qop = CreateOperation(OpType::Quantize, TensorUsage::IFM, cifm, TensorUsage::OFM, cofm);
            auto &outQuant = qop->Output(TensorUsage::OFM)->quantization;
            outQuant.scales.clear();
            outQuant.scales.push_back(QuantizedScale(static_cast<double>(0.0016390486853197217) / static_cast<double>(0.010346830822527409)));
            outQuant.zeroPoints.clear();
            outQuant.zeroPoints.push_back(5);
            outQuant.type = QuantizationType::EXPLICIT;
            qop->Output(TensorUsage::OFM)->rounding = RoundMode::DBL;

            auto &inQuant = qop->Input(TensorUsage::IFM)->quantization;
            inQuant.scales.clear();
            inQuant.scales.emplace_back(QuantizedScale{1, 0});
            inQuant.zeroPoints.clear();
            inQuant.zeroPoints.emplace_back(0);
            inQuant.type = QuantizationType::EXPLICIT;

            auto op = CreateOperation(OpType::Add, TensorUsage::IFM, ifm, TensorUsage::IFM1, cofm, TensorUsage::OFM, ofm);
            ops.push_back(std::move(qop));
            ops.push_back(std::move(op));

            return CreateGraph(ops);
        }();

        GraphOptimiserOptions options;
        const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);

        std::vector<Operation *> allOps;

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 2);

        REQUIRE(!optimiser.empty());
        optimiser.back()->Process(graph.get());
        allOps.clear();

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 1);
        REQUIRE(allOps[0]->Inputs()[TensorUsage::IFM1].tensor->IsConstant());
        auto iview = allOps[0]->Inputs()[TensorUsage::IFM1].tensor->View();
        auto idata = iview.RawData<int8_t>();
        std::vector<int8_t> actualData(idata, idata + 10);
        std::vector<int8_t> expectedData{6, 1, 0, 3, 6, 4, 5, 4, 7, 8};
        REQUIRE_THAT(actualData, Catch::Matchers::Equals(expectedData));
    }
}

TEST_CASE("test_graphir_optimiser - ReduceSum")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    SECTION("Zero point")
    {
        constexpr int ZP = 10;

        auto graph = [&]()
        {
            std::vector<std::shared_ptr<Operation>> ops;
            auto ifm = CreateTensor("IFM", Shape(1, 4, 4, 25), DataType::Int8);
            auto ofm = CreateTensor("OFM", ifm->StorageShape().WithDepth(1), DataType::Int8);
            auto op = CreateOperation(OpType::ReduceSum, TensorUsage::IFM, ifm, TensorUsage::OFM, ofm);
            op->Input(TensorUsage::IFM)->quantization.zeroPoints.clear();
            op->Input(TensorUsage::IFM)->quantization.zeroPoints.push_back(ZP);
            op->Attribute<axis_attr_t>()->axis = ifm->StorageShape().Size() - 1;
            ops.push_back(std::move(op));

            // Create graph with ops
            return CreateGraph(ops);
        }();

        GraphOptimiserOptions options;
        const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);

        REQUIRE(!optimiser.empty());
        optimiser.back()->Process(graph.get());

        const std::unordered_map<UniqueId, UniqueId> emptyTensorEquivalenceIdMap;
        SchedulerPacking packing(arch.get(), false, emptyTensorEquivalenceIdMap);
        auto scheduleOps = packing.Process(graph.get());

        REQUIRE(scheduleOps.size() == 1);
        REQUIRE(scheduleOps[0]->SubOps().size() == 1);
        REQUIRE(scheduleOps[0]->SubOps()[0]->IFM(1)->tensor->IsConstant());
        REQUIRE(scheduleOps[0]->SubOps()[0]->IFM(1)->tensor->bufferView.Elements() == 1);
        REQUIRE(scheduleOps[0]->SubOps()[0]->IFM(1)->tensor->bufferView.StrideBytes().Elements() == sizeof(int32_t));
        auto view = scheduleOps[0]->SubOps()[0]->IFM(1)->tensor->bufferView.Values<int32_t>();
        REQUIRE(view[0] == scheduleOps[0]->IFM(0)->shape.Depth() * ZP);
        if ( scheduleOps[0]->IFM(0)->quantization.zeroPoints.size() > 0 )
            REQUIRE(scheduleOps[0]->IFM(0)->quantization.zeroPoints[0] == 0);
    }
}

TEST_CASE("test_graphir_optimiser - transpose removal")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    std::vector<std::shared_ptr<Operation>> ops;
    auto cadd = CreateTensor("CADD", Shape(1, 1, 1, 1), DataType::Int8, 1);
    auto input = CreateTensor("INPUT", Shape(1, 10, 5, 4), DataType::Int8);
    auto ofm1 = CreateTensor("OFM", Shape(1, 10, 5, 4), DataType::Int8);
    auto ofm2 = CreateTensor("OFM", Shape(1, 10, 5, 4), DataType::Int8);
    auto output = CreateTensor("OUTPUT", Shape(1, 10, 5, 4), DataType::Int8);

    // Add->Transpose(none)->Add
    ops.push_back(CreateOperation(OpType::Add, TensorUsage::IFM, input, TensorUsage::IFM1, cadd, TensorUsage::OFM, ofm1));

    ops.push_back(CreateOperation(OpType::Transpose, TensorUsage::IFM, ofm1, TensorUsage::OFM, ofm2));
    transpose_attr_t *attr = ops.back()->Attribute<transpose_attr_t>();
    attr->perm = Shape(0, 1, 2, 3);

    ops.push_back(CreateOperation(OpType::Add, TensorUsage::IFM, ofm2, TensorUsage::IFM1, cadd, TensorUsage::OFM, output));

    auto graph = CreateGraph(ops);

    GraphOptimiserOptions options;
    const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);

    optimiser.back()->Process(graph.get());

    std::vector<Operation *> allOps;
    graph->GetAllOperations(allOps);
    REQUIRE(allOps.size() == 2);
    REQUIRE(allOps.front()->Type() == OpType::Add);
    REQUIRE(allOps.back()->Type() == OpType::Add);
    REQUIRE(allOps.front()->Output(TensorUsage::OFM)->tensor == allOps.back()->Input(TensorUsage::IFM)->tensor);
}

TEST_CASE("test_graphir_optimiser - transpose removal for unit tensors")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    std::vector<std::shared_ptr<Operation>> ops;
    auto input = CreateTensor("INPUT", Shape(1, 1, 1, 1), DataType::Int8);
    auto transposeOfm = CreateTensor("TRANSPOSE_OFM", Shape(1, 1, 1, 1), DataType::Int8);
    auto output = CreateTensor("OUTPUT", Shape(1, 1, 1, 1), DataType::Int8);

    ops.push_back(CreateOperation(OpType::Transpose, TensorUsage::IFM, input, TensorUsage::OFM, transposeOfm));
    transpose_attr_t *attr = ops.back()->Attribute<transpose_attr_t>();
    attr->perm = Shape(0, 1, 3, 2);
    ops.push_back(CreateOperation(OpType::Abs, TensorUsage::IFM, transposeOfm, TensorUsage::OFM, output));

    auto graph = CreateGraph(ops);

    GraphOptimiserOptions options;
    const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);

    optimiser.back()->Process(graph.get());

    std::vector<Operation *> allOps;
    graph->GetAllOperations(allOps);
    REQUIRE(allOps.size() == 1);
    REQUIRE(allOps[0]->Type() == OpType::Abs);
}

TEST_CASE("test_graphir_optimiser - transpose merge")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    std::vector<std::shared_ptr<Operation>> ops;
    auto cadd = CreateTensor("CADD", Shape(1, 1, 1, 1), DataType::Int8, 1);
    auto input = CreateTensor("INPUT", Shape(1, 10, 4, 5), DataType::Int8);
    auto ofm1 = CreateTensor("OFM", Shape(1, 10, 4, 5), DataType::Int8);
    auto ofm2 = CreateTensor("OFM", Shape(1, 10, 5, 4), DataType::Int8);
    auto ofm3 = CreateTensor("OFM", Shape(1, 10, 4, 5), DataType::Int8);
    auto output = CreateTensor("OUTPUT", Shape(1, 10, 4, 5), DataType::Int8);

    // Add->Transpose(there)->Transpose(back)->Add
    ops.push_back(CreateOperation(OpType::Add, TensorUsage::IFM, input, TensorUsage::IFM1, cadd, TensorUsage::OFM, ofm1));

    ops.push_back(CreateOperation(OpType::Transpose, TensorUsage::IFM, ofm1, TensorUsage::OFM, ofm2));
    transpose_attr_t *attr = ops.back()->Attribute<transpose_attr_t>();
    attr->perm = Shape(0, 1, 3, 2);

    ops.push_back(CreateOperation(OpType::Transpose, TensorUsage::IFM, ofm2, TensorUsage::OFM, ofm3));
    attr = ops.back()->Attribute<transpose_attr_t>();
    attr->perm = Shape(0, 1, 3, 2);

    ops.push_back(CreateOperation(OpType::Add, TensorUsage::IFM, ofm3, TensorUsage::IFM1, cadd, TensorUsage::OFM, output));

    auto graph = CreateGraph(ops);

    GraphOptimiserOptions options;
    const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);

    optimiser.back()->Process(graph.get());

    // Result Add->Add
    std::vector<Operation *> allOps;
    graph->GetAllOperations(allOps);
    REQUIRE(allOps.size() == 2);
    REQUIRE(allOps.front()->Type() == OpType::Add);
    REQUIRE(allOps.back()->Type() == OpType::Add);
    REQUIRE(allOps.front()->Output(TensorUsage::OFM)->tensor == allOps.back()->Input(TensorUsage::IFM)->tensor);
}

TEST_CASE("test_graphir_optimiser - transpose merge with metadata reshape resulting in memory copy")
{
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    std::vector<std::shared_ptr<Operation>> ops;
    auto input = CreateTensor("INPUT", Shape(1, 374, 144), DataType::Int8);
    auto transposeOfm = CreateTensor("T1_OFM", Shape(1, 144, 374), DataType::Int8);
    auto reshapeOfm = CreateTensor("RESHAPE_OFM", Shape(1, 144, 374, 1), DataType::Int8);
    auto output = CreateTensor("OUTPUT", Shape(1, 374, 1, 144), DataType::Int8);

    ops.push_back(CreateOperation(OpType::Transpose, TensorUsage::IFM, input, TensorUsage::OFM, transposeOfm));
    auto *transposeAttr = ops.back()->Attribute<transpose_attr_t>();
    transposeAttr->perm = Shape(0, 2, 1);

    ops.push_back(CreateOperation(OpType::Reshape, TensorUsage::IFM, transposeOfm, TensorUsage::OFM, reshapeOfm));
    auto *reshapeAttr = ops.back()->Attribute<reshape_attr_t>();
    reshapeAttr->shape = reshapeOfm->StorageShape();

    ops.push_back(CreateOperation(OpType::Transpose, TensorUsage::IFM, reshapeOfm, TensorUsage::OFM, output));
    transposeAttr = ops.back()->Attribute<transpose_attr_t>();
    transposeAttr->perm = Shape(0, 2, 3, 1);

    auto graph = CreateGraph(ops);

    GraphOptimiserOptions options;
    const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);

    optimiser.back()->Process(graph.get());

    std::vector<Operation *> allOps;
    graph->GetAllOperations(allOps);
    REQUIRE(allOps.size() == 1);
    REQUIRE(allOps[0]->Type() == OpType::MemoryCopy);

    auto *ifmConn = allOps[0]->Input(TensorUsage::IFM);
    REQUIRE(ifmConn->shape == Shape(1, 374, 144));

    auto *ofmConn = allOps[0]->Output(TensorUsage::OFM);
    REQUIRE(ofmConn->shape == Shape(1, 374, 144));
}

TEST_CASE("test_graphir_optimiser - transpose merge with metadata reshape resulting in transpose")
{
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    std::vector<std::shared_ptr<Operation>> ops;
    auto input = CreateTensor("INPUT", Shape(4, 374, 144), DataType::Int8);
    auto transposeOfm = CreateTensor("T1_OFM", Shape(4, 144, 374), DataType::Int8);
    auto reshapeOfm = CreateTensor("RESHAPE_OFM", Shape(1, 4, 144, 374), DataType::Int8);
    auto output = CreateTensor("OUTPUT", Shape(1, 374, 144, 4), DataType::Int8);

    ops.push_back(CreateOperation(OpType::Transpose, TensorUsage::IFM, input, TensorUsage::OFM, transposeOfm));
    auto *transposeAttr = ops.back()->Attribute<transpose_attr_t>();
    transposeAttr->perm = Shape(0, 2, 1);

    ops.push_back(CreateOperation(OpType::Reshape, TensorUsage::IFM, transposeOfm, TensorUsage::OFM, reshapeOfm));
    auto *reshapeAttr = ops.back()->Attribute<reshape_attr_t>();
    reshapeAttr->shape = reshapeOfm->StorageShape();

    ops.push_back(CreateOperation(OpType::Transpose, TensorUsage::IFM, reshapeOfm, TensorUsage::OFM, output));
    transposeAttr = ops.back()->Attribute<transpose_attr_t>();
    transposeAttr->perm = Shape(0, 3, 2, 1);

    auto graph = CreateGraph(ops);

    GraphOptimiserOptions options;
    const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);

    optimiser.back()->Process(graph.get());

    std::vector<Operation *> allOps;
    graph->GetAllOperations(allOps);
    REQUIRE(allOps.size() == 1);
    REQUIRE(allOps[0]->Type() == OpType::Transpose);
    transposeAttr = allOps[0]->Attribute<transpose_attr_t>();
    REQUIRE(transposeAttr->perm == Shape(1, 2, 0));

    auto *ifmConn = allOps[0]->Input(TensorUsage::IFM);
    REQUIRE(ifmConn->shape == Shape(4, 374, 144));

    auto *ofmConn = allOps[0]->Output(TensorUsage::OFM);
    REQUIRE(ofmConn->shape == Shape(374, 144, 4));
}

TEST_CASE("test_graphir_optimiser - replace pad by explicit padding")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    // Constant data for the Pad op's paddings tensor
    std::vector<int8_t> paddings = {{
        0,
        0,
        1 /* top */,
        4 /* bottom*/,
        3 /* left */,
        2 /* right */,
        0,
        0,
    }};

    std::vector<std::shared_ptr<Operation>> ops;
    auto padIfm = CreateTensor("INPUT", Shape(1, 7, 7, 3), DataType::Int8, 1);
    auto padParam = CreateTensor("PADPARAM", Shape(8), DataType::Int8, std::move(paddings));
    auto padOfm = CreateTensor("PADOFM", Shape(1, 12, 12, 3), DataType::Int8);
    auto convWeights = CreateTensor("WEIGHTS", Shape(1, 6, 6, 9), DataType::Int8, 42);
    auto convBias = CreateTensor("BIAS", Shape(1, 1, 1, 9), DataType::Int8, 0);
    auto convOfm = CreateTensor("OUTPUT", Shape(1, 7, 7, 9), DataType::Int8);

    // Create Pad op
    ops.push_back(CreateOperation(OpType::Pad, TensorUsage::IFM, padIfm, TensorUsage::Params, padParam, TensorUsage::OFM, padOfm));
    pad_attr_t *attr = ops.back()->Attribute<pad_attr_t>();
    attr->pad_const = 0;

    // Create Conv2D op
    ops.push_back(CreateOperation(OpType::Conv2D, TensorUsage::IFM, padOfm, TensorUsage::Weights, convWeights,
        TensorUsage::Scales, convBias, TensorUsage::OFM, convOfm));
    Kernel kernel = Kernel::UnitKernel().WithSize({6, 6});
    ops.back()->SetKernel(std::make_unique<Kernel>(std::move(kernel)));

    auto graph = CreateGraph(ops);

    GraphOptimiserOptions options;
    const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);

    optimiser.back()->Process(graph.get());

    std::vector<Operation *> allOps;
    graph->GetAllOperations(allOps);
    REQUIRE(allOps.size() == 1);
    REQUIRE(allOps[0]->Type() == OpType::Conv2D);
    auto &padding = allOps[0]->Kernel()->Padding();
    REQUIRE(padding.Top() == 1);
    REQUIRE(padding.Left() == 3);
    REQUIRE(padding.Bottom() == 4);
    REQUIRE(padding.Right() == 2);
    REQUIRE(padding.Near() == 0);
    REQUIRE(padding.Far() == 0);
}

TEST_CASE("test_graphir_optimiser - fuse rescale with reshape, before")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    std::vector<std::shared_ptr<Operation>> ops;
    auto input = CreateTensor("INPUT", Shape(1, 8, 2, 1), DataType::Int8);
    auto rescaleOfm = CreateTensor("RESCALE_OFM", Shape(1, 8, 2, 1), DataType::Int8);
    auto reshapeOfm = CreateTensor("RESHAPE_OFM", Shape(1, 4, 4, 1), DataType::Int8);
    auto absOfm = CreateTensor("ABS_OFM", Shape(1, 4, 4, 1), DataType::Int8);

    // Create a RESCALE-RESHAPE-ABS graph
    ops.push_back(CreateOperation(OpType::Rescale, TensorUsage::IFM, input, TensorUsage::OFM, rescaleOfm));
    auto *rescaleAttr = ops.back()->Attribute<rescale_attr_t>();
    rescaleAttr->double_round = false;
    rescaleAttr->per_channel = false;
    rescaleAttr->scale32 = true;
    auto *signAttr = ops.back()->Attribute<sign_attr_t>();
    signAttr->input_unsigned = false;
    signAttr->output_unsigned = false;
    auto &rescaleQuant = ops.back()->Output(TensorUsage::OFM)->quantization;
    rescaleQuant.scales.clear();
    rescaleQuant.scales.push_back(QuantizedScale(1073741824, 31));
    ops.push_back(CreateOperation(OpType::Reshape, TensorUsage::IFM, rescaleOfm, TensorUsage::OFM, reshapeOfm));
    ops.push_back(CreateOperation(OpType::Abs, TensorUsage::IFM, reshapeOfm, TensorUsage::OFM, absOfm));

    auto graph = CreateGraph(ops);

    GraphOptimiserOptions options;
    const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);

    optimiser.back()->Process(graph.get());

    std::vector<Operation *> allOps;
    graph->GetAllOperations(allOps);
    REQUIRE(allOps.size() == 1);
    REQUIRE(allOps[0]->Type() == OpType::Abs);
    REQUIRE(allOps[0]->Input(TensorUsage::IFM)->SliceShape() == Shape(1, 4, 4, 1));
    REQUIRE(allOps[0]->Input(TensorUsage::IFM)->quantization.zeroPoints[0] == 0);
    // Reduced scale/shift
    REQUIRE(allOps[0]->Input(TensorUsage::IFM)->quantization.scales[0].scale == 1);
    REQUIRE(allOps[0]->Input(TensorUsage::IFM)->quantization.scales[0].shift == 1);
    REQUIRE(allOps[0]->Output(TensorUsage::OFM)->SliceShape() == Shape(1, 4, 4, 1));
}

TEST_CASE("test_graphir_optimiser - fuse rescale with reshape, after")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    std::vector<std::shared_ptr<Operation>> ops;
    auto input = CreateTensor("INPUT", Shape(1, 4, 4, 1), DataType::Int8);
    auto absOfm = CreateTensor("ABS_OFM", Shape(1, 4, 4, 1), DataType::Int8);
    auto reshapeOfm = CreateTensor("RESHAPE_OFM", Shape(1, 8, 2, 1), DataType::Int8);
    auto rescaleOfm = CreateTensor("RESCALE_OFM", Shape(1, 8, 2, 1), DataType::Int8);

    // Create a ABS-RESHAPE-RESCALE graph
    ops.push_back(CreateOperation(OpType::Abs, TensorUsage::IFM, input, TensorUsage::OFM, absOfm));
    ops.push_back(CreateOperation(OpType::Reshape, TensorUsage::IFM, absOfm, TensorUsage::OFM, reshapeOfm));
    ops.push_back(CreateOperation(OpType::Rescale, TensorUsage::IFM, reshapeOfm, TensorUsage::OFM, rescaleOfm));
    auto *rescaleAttr = ops.back()->Attribute<rescale_attr_t>();
    rescaleAttr->double_round = false;
    rescaleAttr->per_channel = false;
    rescaleAttr->scale32 = true;
    auto &rescaleQuant = ops.back()->Output(TensorUsage::OFM)->quantization;
    rescaleQuant.scales.clear();
    rescaleQuant.scales.push_back(QuantizedScale(1073741824, 31));
    auto *signAttr = ops.back()->Attribute<sign_attr_t>();
    signAttr->input_unsigned = false;
    signAttr->output_unsigned = false;

    auto graph = CreateGraph(ops);

    GraphOptimiserOptions options;
    const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);

    optimiser.back()->Process(graph.get());

    std::vector<Operation *> allOps;
    graph->GetAllOperations(allOps);
    REQUIRE(allOps.size() == 1);
    REQUIRE(allOps[0]->Type() == OpType::Abs);
    REQUIRE(allOps[0]->Input(TensorUsage::IFM)->SliceShape() == Shape(1, 4, 4, 1));
    REQUIRE(allOps[0]->Output(TensorUsage::OFM)->SliceShape() == Shape(1, 4, 4, 1));
    REQUIRE(allOps[0]->Output(TensorUsage::OFM)->quantization.zeroPoints[0] == 0);
    // Reduced scale/shift
    REQUIRE(allOps[0]->Output(TensorUsage::OFM)->quantization.scales[0].scale == 1);
    REQUIRE(allOps[0]->Output(TensorUsage::OFM)->quantization.scales[0].shift == 1);
}

TEST_CASE("test_graphir_optimiser - resource data type")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    std::vector<std::shared_ptr<Operation>> ops;
    auto input = CreateTensor("IFM", Shape(), DataType::Resource);
    auto output = CreateTensor("OFM", Shape(1, 4, 4, 1), DataType::Int8);

    // Create a graph with something that looks like a READ_VARIABLE op
    ops.push_back(CreateOperation(OpType::Passthrough, TensorUsage::IFM, input, TensorUsage::OFM, output));

    auto graph = CreateGraph(ops);

    GraphOptimiserOptions options;
    const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);

    optimiser.back()->Process(graph.get());

    std::vector<Operation *> allOps;
    graph->GetAllOperations(allOps);
    REQUIRE(allOps.size() == 1);
    REQUIRE(allOps[0]->Type() == OpType::Passthrough);
    REQUIRE(allOps[0]->Input(TensorUsage::IFM)->SliceShape() == Shape());
    REQUIRE(allOps[0]->Output(TensorUsage::OFM)->SliceShape() == Shape(1, 4, 4, 1));
}

TEST_CASE("test_graphir_optimiser - duplicated tensor readers")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU55>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    std::vector<std::shared_ptr<Operation>> ops;
    auto input = CreateTensor("INPUT", Shape(1, 4, 4, 1), DataType::Int8);
    auto absOfm = CreateTensor("ABS_OFM", Shape(1, 4, 4, 1), DataType::Int8);
    auto reshapeOfm = CreateTensor("RESHAPE_OFM", Shape(1, 8, 2, 1), DataType::Int8);
    auto matmulOfm = CreateTensor("MATMUL_OFM", Shape(1, 8, 2, 1), DataType::Int8);
    auto rescaleOfm = CreateTensor("RESCALE_OFM", Shape(1, 8, 2, 1), DataType::Int8);

    // Create a ABS-RESHAPE-MATMUL-RESCALE graph
    ops.push_back(CreateOperation(OpType::Abs, TensorUsage::IFM, input, TensorUsage::OFM, absOfm));
    ops.push_back(CreateOperation(OpType::Reshape, TensorUsage::IFM, absOfm, TensorUsage::OFM, reshapeOfm));
    // Use reshapeOfm as both inputs to matmul to test multiple readers
    ops.push_back(CreateOperation(OpType::MatMul, TensorUsage::IFM0, reshapeOfm, TensorUsage::IFM1, reshapeOfm, TensorUsage::OFM, matmulOfm));
    auto matmulOp = ops.back();

    ops.push_back(CreateOperation(OpType::Rescale, TensorUsage::IFM, matmulOfm, TensorUsage::OFM, rescaleOfm));

    auto &reshapeOp = ops[1];
    REQUIRE(reshapeOp->Type() == OpType::Reshape);

    auto graph = CreateGraph(ops);
    std::vector<Operation *> allOps;

    SECTION("Insert copy before reshape")
    {
        auto copyOp = regor::GraphOptimisation::InsertCopyOpAfterTensor(
            reshapeOp->Input(TensorUsage::IFM)->tensor, reshapeOp->Input(TensorUsage::IFM)->quantization);

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 5);
        REQUIRE(copyOp->OFM()->Readers().size() == 1);
        REQUIRE(copyOp->OFM()->Readers()[0] == reshapeOp);
    }

    SECTION("Insert copy before matmul ifm0")
    {
        auto copyOp = regor::GraphOptimisation::InsertCopyOpAfterTensor(
            matmulOp->Input(TensorUsage::IFM0)->tensor, matmulOp->Input(TensorUsage::IFM0)->quantization);

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 5);

        REQUIRE(copyOp->OFM()->Readers().size() == 2);
        REQUIRE(copyOp->OFM()->Readers()[0] == matmulOp);
        REQUIRE(copyOp->OFM()->Readers()[1] == matmulOp);
        REQUIRE(reshapeOp->OFM()->Readers().size() == 1);
        REQUIRE(reshapeOp->OFM()->Readers()[0] == copyOp);

        REQUIRE(matmulOp->Input(TensorUsage::IFM0)->tensor.get()->Writers().size() == 1);
        REQUIRE(matmulOp->Input(TensorUsage::IFM0)->tensor.get()->Name() == "RESHAPE_OFM_copy");
        REQUIRE(matmulOp->Input(TensorUsage::IFM1)->tensor.get()->Writers().size() == 1);
        REQUIRE(matmulOp->Input(TensorUsage::IFM1)->tensor.get()->Name() == "RESHAPE_OFM_copy");
    }

    SECTION("Insert copy before matmul using pointer variable")
    {
        auto copyOp = regor::GraphOptimisation::InsertCopyOpAfterTensor(reshapeOfm, matmulOp->Input(TensorUsage::IFM0)->quantization);

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 5);

        REQUIRE(copyOp->OFM()->Readers().size() == 2);
        REQUIRE(copyOp->OFM()->Readers()[0] == matmulOp);
        REQUIRE(copyOp->OFM()->Readers()[1] == matmulOp);
        REQUIRE(reshapeOp->OFM()->Readers().size() == 1);
        REQUIRE(reshapeOp->OFM()->Readers()[0] == copyOp);

        REQUIRE(matmulOp->Input(TensorUsage::IFM0)->tensor.get()->Writers().size() == 1);
        REQUIRE(matmulOp->Input(TensorUsage::IFM0)->tensor.get()->Name() == "RESHAPE_OFM_copy");
        REQUIRE(matmulOp->Input(TensorUsage::IFM1)->tensor.get()->Writers().size() == 1);
        REQUIRE(matmulOp->Input(TensorUsage::IFM1)->tensor.get()->Name() == "RESHAPE_OFM_copy");
    }

    SECTION("Replace consumer input)")
    {
        auto newTensor = CreateTensor("NEW_TENSOR", Shape(1, 8, 2, 1), DataType::Int8);
        regor::GraphOptimisation::ReplaceConsumerInput(nullptr, reshapeOfm->Readers(), reshapeOfm.get(), newTensor);

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 2);
        REQUIRE(allOps[0]->Type() == OpType::MatMul);  //  Abs and Reshape are disconnected from the graph now
        REQUIRE(newTensor->Readers().size() == 2);
        REQUIRE(reshapeOfm->Readers().empty());
        REQUIRE(matmulOp->Input(TensorUsage::IFM0)->tensor.get()->Name() == "NEW_TENSOR");
        REQUIRE(matmulOp->Input(TensorUsage::IFM1)->tensor.get()->Name() == "NEW_TENSOR");

        reshapeOp->Disconnect();  // Clean up to avoid memory leaks
        ops[0]->Disconnect();
    }

    SECTION("Remove Reshape")
    {
        GraphOptimiserOptions options;
        const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);
        REQUIRE(!optimiser.empty());
        optimiser.back()->Process(graph.get());

        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 3);
        for ( const auto *op : allOps )
        {
            REQUIRE(op->Type() != OpType::Reshape);
        }
        REQUIRE(absOfm->Readers().size() == 2);
        // Check that tensor name starts with ABS_OFM
        REQUIRE(matmulOp->Input(TensorUsage::IFM0)->tensor.get()->Name().rfind("ABS_OFM", 0) == 0);
        REQUIRE(matmulOp->Input(TensorUsage::IFM1)->tensor.get()->Name().rfind("ABS_OFM", 0) == 0);
    }
}

TEST_CASE("test_graphir_optimiser - convert TFLite Quantization to Explicit Quantization")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU55>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    SECTION("Quantize operation with Data type int16")
    {

        std::vector<std::shared_ptr<Operation>> ops;
        auto ifm = CreateTensor("QIFM", Shape(1, 1, 1, 10), DataType::Int16);
        auto ofm = CreateTensor("QOFM", Shape(1, 1, 10, 10), DataType::Int16);
        auto quantizeOp = CreateOperation(OpType::Quantize, TensorUsage::IFM0, ifm, TensorUsage::OFM, ofm);

        auto &ifmQuant = quantizeOp->Input(TensorUsage::IFM0)->quantization;
        ifmQuant.scales.clear();
        ifmQuant.scales.push_back(QuantizedScale(int32_t(1387686912), 42));
        ifmQuant.type = QuantizationType::TFLITE;

        auto &ofmQuant = quantizeOp->Output(TensorUsage::OFM)->quantization;
        ofmQuant.scales.clear();
        ofmQuant.scales.push_back(QuantizedScale(int32_t(1899507328), 45));
        ofmQuant.type = QuantizationType::TFLITE;

        ops.push_back(std::move(quantizeOp));
        auto graph = CreateGraph(ops);

        GraphOptimiserOptions options;
        const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(GraphNotation::TFLite, arch.get(), options, nullptr);
        REQUIRE(!optimiser.empty());
        optimiser.front()->Process(graph.get());

        std::vector<Operation *> allOps;
        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 1);

        REQUIRE(ofmQuant.type == QuantizationType::EXPLICIT);
        REQUIRE(ifmQuant.type == QuantizationType::EXPLICIT);
        REQUIRE(ifmQuant.scales[0] == QuantizedScale::Unit());

        REQUIRE(ofmQuant.scales.size() == 1);
        auto quantScale = ofmQuant.scales[0];
        REQUIRE(quantScale.scale == 1568846252);
        REQUIRE(quantScale.shift == 28);
    }
    SECTION("Mul operation with Data type int8")
    {

        std::vector<std::shared_ptr<Operation>> ops;
        auto ifm0 = CreateTensor("IFM0", Shape(1, 1, 1, 10), DataType::Int8);
        auto ifm1 = CreateTensor("IFM1", Shape(1, 1, 1, 10), DataType::Int8);
        auto ofm = CreateTensor("OFM", Shape(1, 1, 10, 10), DataType::Int8);
        auto mulOp = CreateOperation(OpType::Mul, TensorUsage::IFM0, ifm0, TensorUsage::IFM1, ifm1, TensorUsage::OFM, ofm);

        auto &ifmQuant0 = mulOp->Input(TensorUsage::IFM0)->quantization;
        ifmQuant0.scales.clear();
        ifmQuant0.scales.push_back(QuantizedScale(int32_t(1888360448), 37));
        ifmQuant0.type = QuantizationType::TFLITE;

        auto &ifmQuant1 = mulOp->Input(TensorUsage::IFM1)->quantization;
        ifmQuant1.scales.clear();
        ifmQuant1.scales.push_back(QuantizedScale(int32_t(1888360448), 37));
        ifmQuant1.type = QuantizationType::TFLITE;

        auto &ofmQuant = mulOp->Output(TensorUsage::OFM)->quantization;
        ofmQuant.scales.clear();
        ofmQuant.scales.push_back(QuantizedScale(int32_t(1578641920), 37));
        ofmQuant.type = QuantizationType::TFLITE;

        ops.push_back(std::move(mulOp));
        auto graph = CreateGraph(ops);

        GraphOptimiserOptions options;
        const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(GraphNotation::TFLite, arch.get(), options, nullptr);
        REQUIRE(!optimiser.empty());
        optimiser.front()->Process(graph.get());

        std::vector<Operation *> allOps;
        graph->GetAllOperations(allOps);
        REQUIRE(allOps.size() == 1);

        REQUIRE(ofmQuant.type == QuantizationType::EXPLICIT);
        REQUIRE(ifmQuant0.type == QuantizationType::EXPLICIT);
        REQUIRE(ifmQuant0.scales[0] == QuantizedScale::Unit());
        REQUIRE(ifmQuant1.type == QuantizationType::EXPLICIT);
        REQUIRE(ifmQuant1.scales[0] == QuantizedScale::Unit());

        REQUIRE(ofmQuant.scales.size() == 1);
        auto quantScale = ofmQuant.scales[0];
        REQUIRE(quantScale.scale == 1129421696);
        REQUIRE(quantScale.shift == 36);
    }
}

TEST_CASE("test_graphir_optimiser - MoveConcatSliceToProducer")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    // Create a graph like (1) and check that it turns into a graph like (2).
    //
    //    (1)          (2)
    //
    //  T1   T2      T1   T2
    //  |    |       |    |
    //  O1   O2      O1   O2
    //  |    |   =>   \  / (slice offset on OFM conn.)
    //  T3   T4        T5
    //   \  /
    //   PACK
    //    |
    //    T5

    std::vector<std::shared_ptr<Operation>> ops;
    auto ifm0 = CreateTensor("IFM0", Shape(1, 2, 2, 4), DataType::Int8);
    auto ifm1 = CreateTensor("IFM1", Shape(1, 2, 2, 4), DataType::Int8);
    auto abs0 = CreateTensor("ABS0", Shape(1, 2, 2, 4), DataType::Int8);
    auto abs1 = CreateTensor("ABS1", Shape(1, 2, 2, 4), DataType::Int8);
    auto packed = CreateTensor("PACKED", Shape(1, 2, 2, 8), DataType::Int8);

    ops.push_back(CreateOperation(OpType::Abs, TensorUsage::IFM, ifm0, TensorUsage::OFM, abs0));
    ops.push_back(CreateOperation(OpType::Abs, TensorUsage::IFM, ifm1, TensorUsage::OFM, abs1));
    ops.push_back(CreateOperation(OpType::Concat, TensorUsage::IFM0, abs0, TensorUsage::IFM1, abs1, TensorUsage::OFM, packed));
    auto *packAttr = ops.back()->Attribute<axis_attr_t>();
    packAttr->axis = 3;  // C

    auto graph = CreateGraph(ops);
    GraphOptimiserOptions options;
    const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);
    REQUIRE(!optimiser.empty());

    // Run the last optimiser step (where MoveConcatSliceToProducer lives)
    optimiser.back()->Process(graph.get());

    std::vector<Operation *> allOps;
    graph->GetAllOperations(allOps);
    REQUIRE(allOps.size() == 2);

    // Check first ABS
    Operation *absOp0 = allOps[0];
    REQUIRE(absOp0);
    REQUIRE(absOp0->Type() == OpType::Abs);
    auto *abs0Ifm = absOp0->Input(TensorUsage::IFM);
    REQUIRE(abs0Ifm->tensor == ifm0);
    REQUIRE(abs0Ifm->tensor->Readers().size() == 1);
    REQUIRE(abs0Ifm->tensor->Writers().empty());
    REQUIRE(abs0Ifm->shape == Shape(1, 2, 2, 4));
    REQUIRE_FALSE(!!abs0Ifm->slice);  // No slice
    auto *abs0Ofm = absOp0->Output(TensorUsage::OFM);
    REQUIRE(abs0Ofm->tensor == packed);
    REQUIRE(abs0Ofm->tensor->Readers().empty());
    REQUIRE(abs0Ofm->tensor->Writers().size() == 2);
    REQUIRE(abs0Ofm->shape == Shape(1, 2, 2, 8));
    REQUIRE(abs0Ofm->slice.offset == Shape(0, 0, 0, 0));
    REQUIRE(abs0Ofm->slice.shape == Shape(1, 2, 2, 4));

    // Check second ABS
    Operation *absOp1 = allOps[1];
    REQUIRE(absOp1);
    REQUIRE(absOp1->Type() == OpType::Abs);
    auto *abs1Ifm = absOp1->Input(TensorUsage::IFM);
    REQUIRE(abs1Ifm->tensor == ifm1);
    REQUIRE(abs1Ifm->tensor->Readers().size() == 1);
    REQUIRE(abs1Ifm->tensor->Writers().empty());
    REQUIRE(abs1Ifm->shape == Shape(1, 2, 2, 4));
    REQUIRE_FALSE(!!abs1Ifm->slice);
    auto *abs1Ofm = absOp1->Output(TensorUsage::OFM);
    REQUIRE(abs1Ofm->tensor == packed);
    REQUIRE(abs1Ofm->tensor->Readers().empty());
    REQUIRE(abs1Ofm->tensor->Writers().size() == 2);
    REQUIRE(abs1Ofm->shape == Shape(1, 2, 2, 8));
    REQUIRE(abs1Ofm->slice.offset == Shape(0, 0, 0, 4));
    REQUIRE(abs1Ofm->slice.shape == Shape(1, 2, 2, 4));
}
