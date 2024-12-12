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

#include "compiler/raw_writer.hpp"

#include <catch_all.hpp>

#include "regor.h"

using namespace regor;

TEST_CASE("raw_writer")
{
    // Build command stream tensor
    std::vector<uint8_t> commandStreamData = {'C', 'O', 'P', '1'};
    const auto commandStreamBuffer = std::make_shared<Buffer>(std::move(commandStreamData));
    const auto commandStreamTensor = std::make_shared<Tensor>("command_stream", DataType::Int8, Shape(4), commandStreamBuffer);
    REQUIRE(commandStreamTensor->IsConstant());

    // Build read only tensor
    std::vector<uint8_t> readOnlyData = {21, 22, 23, 24, 25};
    const auto readOnlyBuffer = std::make_shared<Buffer>(std::move(readOnlyData));
    const auto readOnlyTensor = std::make_shared<Tensor>("read_only", DataType::Int8, Shape(5), readOnlyBuffer);
    REQUIRE(readOnlyTensor->IsConstant());

    // Build scratch tensor
    const auto scratch = std::make_shared<Tensor>("scratch", DataType::Int8, Shape({1, 6}));
    REQUIRE_FALSE(scratch->IsConstant());

    // Build scratch fast tensor
    const auto scratchFast = std::make_shared<Tensor>("scratch_fast", DataType::Int8, Shape({1, 7}));
    REQUIRE_FALSE(scratchFast->IsConstant());

    // Build input tensor
    const auto input = std::make_shared<Tensor>("input_1", DataType::Int8, Shape({1, 1, 8}));
    REQUIRE_FALSE(input->IsConstant());

    // Build output tensor
    const auto output = std::make_shared<Tensor>("output_1", DataType::Int8, Shape({1, 1, 1, 10}));
    REQUIRE_FALSE(output->IsConstant());

    // Build variable tensor
    const auto variable1 = std::make_shared<Tensor>("variable_1", DataType::Int8, Shape({1, 1, 1, 9}));
    REQUIRE_FALSE(variable1->IsConstant());

    // Build another variable tensor
    const auto variable2 = std::make_shared<Tensor>("variable_2", DataType::Int8, Shape({1, 1, 1, 11}));
    REQUIRE_FALSE(variable2->IsConstant());

    // Create custom op
    auto op = std::make_shared<Operation>(OpType::CustomNpuOp);
    op->ConnectInput(MakeTensorUsage(TensorUsage::Params, 0), commandStreamTensor);
    op->ConnectInput(MakeTensorUsage(TensorUsage::Params, 1), readOnlyTensor);
    op->ConnectInput(MakeTensorUsage(TensorUsage::State, 0), scratch);
    op->ConnectInput(MakeTensorUsage(TensorUsage::State, 1), scratchFast);
    op->ConnectInput(TensorUsage::IFM0, input);
    op->ConnectInput(TensorUsage::IFM1, variable1);
    op->ConnectOutput(TensorUsage::OFM, output);
    op->ConnectOutput(MakeTensorUsage(TensorUsage::OFM, 1), variable2);

    // Create graph
    std::vector<std::unique_ptr<Graph>> graphs;
    graphs.push_back(std::make_unique<Graph>(GraphNotation::TFLite));
    graphs[0]->AddInput(input);
    graphs[0]->AddPersistent(variable1);
    graphs[0]->AddPersistent(variable2);
    graphs[0]->AddOutput(output);

    // Create tensor address map
    std::vector<std::unordered_map<const Tensor *, Address>> addresses;
    addresses.push_back({});
    addresses[0][commandStreamTensor.get()] = 44;
    addresses[0][readOnlyTensor.get()] = 55;
    addresses[0][scratch.get()] = 66;
    addresses[0][scratchFast.get()] = 77;
    addresses[0][input.get()] = 88;
    addresses[0][variable1.get()] = 11;
    addresses[0][output.get()] = 99;
    addresses[0][variable2.get()] = 22;

    // Create the raw output blobs
    RawWriter writer;
    auto blobs = writer.Serialise(graphs, addresses);

    // Check number of blobs
    REQUIRE(blobs.size() == 8);

    // Check command stream
    {
        // Check blob size
        size_t dataSize = sizeof(regor_raw_tensor_header_t) + 4;
        REQUIRE(blobs[0].second == dataSize);

        // Check header
        auto &data = blobs[0].first;
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_COMMAND_STREAM);
        REQUIRE(header.tensor.command_stream.size == 4);

        // Check tensor data
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 0] == 'C');
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 1] == 'O');
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 2] == 'P');
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 3] == '1');
    }

    // Check read only
    {
        // Check blob size
        size_t dataSize = sizeof(regor_raw_tensor_header_t) + 5;
        REQUIRE(blobs[1].second == dataSize);

        // Check header
        auto &data = blobs[1].first;
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_READ_ONLY);
        REQUIRE(header.tensor.read_only.size == 5);
        REQUIRE(header.tensor.read_only.region == 0);

        // Check tensor data
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 0] == 21);
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 1] == 22);
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 2] == 23);
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 3] == 24);
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 4] == 25);
    }

    // Check scratch
    {
        // Check blob size
        size_t dataSize = sizeof(regor_raw_tensor_header_t);
        REQUIRE(blobs[2].second == dataSize);

        // Check header
        auto &data = blobs[2].first;
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_SCRATCH);
        REQUIRE(header.tensor.scratch.size == 6);
        REQUIRE(header.tensor.scratch.region == 1);
        REQUIRE(header.tensor.scratch.address == 66);
    }

    // Check scratch fast
    {
        // Check blob size
        size_t dataSize = sizeof(regor_raw_tensor_header_t);
        REQUIRE(blobs[3].second == dataSize);

        // Check header
        auto &data = blobs[3].first;
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_SCRATCH_FAST);
        REQUIRE(header.tensor.scratch_fast.size == 7);
        REQUIRE(header.tensor.scratch_fast.region == 2);
        REQUIRE(header.tensor.scratch_fast.address == 77);
    }
    // Check input
    {
        // Check blob size
        size_t dataSize = sizeof(regor_raw_tensor_header_t);
        REQUIRE(blobs[4].second == dataSize);

        // Check header
        auto &data = blobs[4].first;
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_INPUT);
        REQUIRE(header.tensor.input.size == 8);
        REQUIRE(header.tensor.input.region == 1);
        REQUIRE(header.tensor.input.address == 88);
        REQUIRE(header.tensor.input.element_size == 1);
        REQUIRE(header.tensor.input.shape[0] == 1);
        REQUIRE(header.tensor.input.shape[1] == 1);
        REQUIRE(header.tensor.input.shape[2] == 1);
        REQUIRE(header.tensor.input.shape[3] == 8);
    }

    // Check output
    {
        // Check blob size
        size_t dataSize = sizeof(regor_raw_tensor_header_t);
        REQUIRE(blobs[5].second == dataSize);

        // Check header
        auto &data = blobs[5].first;
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_OUTPUT);
        REQUIRE(header.tensor.output.size == 10);
        REQUIRE(header.tensor.output.region == 1);
        REQUIRE(header.tensor.output.address == 99);
        REQUIRE(header.tensor.output.element_size == 1);
        REQUIRE(header.tensor.output.shape[0] == 1);
        REQUIRE(header.tensor.output.shape[1] == 1);
        REQUIRE(header.tensor.output.shape[2] == 1);
        REQUIRE(header.tensor.output.shape[3] == 10);
    }

    // Check (input) variable
    {
        // Check blob size
        size_t dataSize = sizeof(regor_raw_tensor_header_t);
        REQUIRE(blobs[6].second == dataSize);

        // Check header
        auto &data = blobs[6].first;
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_VARIABLE);
        REQUIRE(header.tensor.input.size == 9);
        REQUIRE(header.tensor.input.region == 1);
        REQUIRE(header.tensor.input.address == 11);
        REQUIRE(header.tensor.input.element_size == 1);
        REQUIRE(header.tensor.input.shape[0] == 1);
        REQUIRE(header.tensor.input.shape[1] == 1);
        REQUIRE(header.tensor.input.shape[2] == 1);
        REQUIRE(header.tensor.input.shape[3] == 9);
    }

    // Check (output) variable
    {
        // Check blob size
        size_t dataSize = sizeof(regor_raw_tensor_header_t);
        REQUIRE(blobs[7].second == dataSize);

        // Check header
        auto &data = blobs[7].first;
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_VARIABLE);
        REQUIRE(header.tensor.input.size == 11);
        REQUIRE(header.tensor.input.region == 1);
        REQUIRE(header.tensor.input.address == 22);
        REQUIRE(header.tensor.input.element_size == 1);
        REQUIRE(header.tensor.input.shape[0] == 1);
        REQUIRE(header.tensor.input.shape[1] == 1);
        REQUIRE(header.tensor.input.shape[2] == 1);
        REQUIRE(header.tensor.input.shape[3] == 11);
    }
}
