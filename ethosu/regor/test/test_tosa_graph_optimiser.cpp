//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

// Validate that rescale parameter tensors are converted to
// ofmConn quantization
TEST_CASE("test_tosa_optimiser - rewriterescales")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    std::vector<std::shared_ptr<Operation>> ops;
    auto input = CreateTensor("INPUT", Shape(1, 4, 4, 1), DataType::Int8);
    auto rescaleOfm = CreateTensor("RESCALE_OFM", Shape(1, 4, 4, 1), DataType::Int8);
    auto mulParam = CreateTensor("MUL_PARAM", Shape(1, 1), DataType::Int32, 1073741824);
    auto shiftParam = CreateTensor("SHIFT_PARAM", Shape(1, 1), DataType::Int8, 31);

    // Create a graph with a RESCALE
    ops.push_back(CreateOperation(OpType::Rescale, TensorUsage::IFM, input, TensorUsage::Params0, mulParam,
        TensorUsage::Params1, shiftParam, TensorUsage::OFM, rescaleOfm));

    auto *rescaleAttr = ops.back()->Attribute<rescale_attr_t>();
    rescaleAttr->double_round = false;
    rescaleAttr->per_channel = false;
    rescaleAttr->scale32 = true;
    auto *signAttr = ops.back()->Attribute<sign_attr_t>();
    signAttr->input_unsigned = false;
    signAttr->output_unsigned = false;

    auto graph = CreateGraph(ops);

    GraphOptimiserOptions options;

    const auto &optimiser = GraphOptimiser::MakeGraphOptimiser(graph->Notation(), arch.get(), options, nullptr);
    optimiser.front()->Process(graph.get());

    std::vector<Operation *> allOps;
    graph->GetAllOperations(allOps);
    REQUIRE(allOps.size() == 1);
    REQUIRE(allOps[0]->Type() == OpType::Rescale);
    REQUIRE(allOps[0]->Input(TensorUsage::IFM)->SliceShape() == Shape(1, 4, 4, 1));
    REQUIRE(allOps[0]->Output(TensorUsage::OFM)->SliceShape() == Shape(1, 4, 4, 1));
    REQUIRE(allOps[0]->Output(TensorUsage::OFM)->quantization.zeroPoints[0] == 0);
    REQUIRE(allOps[0]->Output(TensorUsage::OFM)->quantization.scales[0].scale == 1073741824);
    REQUIRE(allOps[0]->Output(TensorUsage::OFM)->quantization.scales[0].shift == 31);
    REQUIRE(allOps[0]->Input(TensorUsage::IFM)->quantization == Quantization::Unit());
}
