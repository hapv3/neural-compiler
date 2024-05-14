//
// SPDX-FileCopyrightText: Copyright 2021-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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


#include "include/graphapi.hpp"
#include "randomize.hpp"
#include "tosa/tosa_validator.hpp"

#include <catch_all.hpp>
#include <memory>
#include <string>
#include <vector>

#include "include/regor.h"

namespace
{

struct TestOperation
{
    tosa::Op tosaOp;
    GraphApi::GraphOperation *op;
    std::string description;
};

std::vector<TestOperation> expectedToPass{};
std::vector<TestOperation> expectedToFail{};

void CreateTestVectors(GraphApi::IGraphBuilder *builder);

TEST_CASE("tosa_validator")
{
    regor_context_t context;
    REQUIRE(regor_create(&context, REGOR_ARCH_ETHOSU85) == 1);

    GraphApi::IGraphBuilder *builder = regor_get_graph_builder(context, "Tosa validator UnitTest");
    REQUIRE(builder);

    CreateTestVectors(builder);
    for ( auto &test : expectedToPass )
    {
        CAPTURE(test.tosaOp);
        auto description = test.description + " should pass.";
        CAPTURE(description);
        REQUIRE_NOTHROW(tosa::validator::ValidateOperator(test.op));
    }
    for ( auto &test : expectedToFail )
    {
        CAPTURE(test.tosaOp);
        auto description = test.description + " should fail.";
        CAPTURE(description);
        REQUIRE_THROWS_AS(tosa::validator::ValidateOperator(test.op), std::logic_error);
    }
    regor_destroy(context);
}

void CreateTestVectors(GraphApi::IGraphBuilder *builder)
{
    expectedToFail.emplace_back(TestOperation{tosa::Op::ABS, nullptr, "Null GraphOperation"});
    expectedToFail.emplace_back(TestOperation{tosa::Op::DIM, builder->CreateOp(tosa::Op::DIM, nullptr), "DIM not supported in 0.60"});
    expectedToFail.emplace_back(TestOperation{tosa::Op::SIGMOID, builder->CreateOp(tosa::Op::SIGMOID, nullptr), "SIGMOID unsupported in BI"});
    {
        auto op{builder->CreateOp(tosa::Op::ABS, nullptr)};
        auto inTensor{builder->CreateTensor("input-0", GraphApi::GraphShape{4, {55, 53, 18, 40}},
            GraphApi::GraphTensorLayout::Linear, GraphApi::GraphDataType::Int32, nullptr)};
        auto outTensor{builder->CreateTensor("result-0", GraphApi::GraphShape{4, {55, 53, 18, 40}},
            GraphApi::GraphTensorLayout::Linear, GraphApi::GraphDataType::Int32, nullptr)};
        builder->AddInput(op, GraphApi::GraphTensorUsage::IFM, inTensor);
        builder->AddOutput(op, GraphApi::GraphTensorUsage::OFM, outTensor);
        expectedToPass.emplace_back(TestOperation{tosa::Op::ABS, op, "ABS supported"});
    }
}

}  // namespace
