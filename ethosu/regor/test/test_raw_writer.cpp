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

#include "compiler/quantization.hpp"
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

    // Build scratch tensors
    const auto scratch = std::make_shared<Tensor>("scratch", DataType::Int8, Shape(1, 6));
    const auto scratchFast = std::make_shared<Tensor>("scratch_fast", DataType::Int8, Shape(1, 7));
    REQUIRE_FALSE(scratch->IsConstant());
    REQUIRE_FALSE(scratchFast->IsConstant());

    // Build tensors with quantization
    const auto input = std::make_shared<Tensor>("input_1", DataType::Int8, Shape({2, 3, 4, 5, 6, 7}));
    const auto inputNoQuant = std::make_shared<Tensor>("input_2", DataType::Int8, Shape({2, 3, 4, 5, 6, 7}));
    const auto output = std::make_shared<Tensor>("output_1", DataType::Int8, Shape({3, 4, 5, 6, 7, 8}));
    const auto variable = std::make_shared<Tensor>("variable_1", DataType::Int8, Shape({4, 5, 6, 7, 8, 9}));
    REQUIRE_FALSE(input->IsConstant());
    REQUIRE_FALSE(inputNoQuant->IsConstant());
    REQUIRE_FALSE(output->IsConstant());
    REQUIRE_FALSE(variable->IsConstant());

    Quantization perTensorQuant;
    perTensorQuant.scales.push_back(QuantizedScale(2, 0));
    perTensorQuant.zeroPoints.push_back(3);

    Quantization perChannelQuant;
    perChannelQuant.scales.push_back(QuantizedScale(1, 0));
    perChannelQuant.scales.push_back(QuantizedScale(2, 0));
    perChannelQuant.zeroPoints.push_back(0);
    perChannelQuant.zeroPoints.push_back(10);
    perChannelQuant.dimension = 3;

    Quantization variableQuant;
    variableQuant.scales.push_back(QuantizedScale(5, 0));
    variableQuant.zeroPoints.push_back(0);

    // Create custom op
    auto op = std::make_shared<Operation>(OpType::CustomNpuOp);
    op->ConnectInput(MakeTensorUsage(TensorUsage::Params, 0), commandStreamTensor);
    op->ConnectInput(MakeTensorUsage(TensorUsage::Params, 1), readOnlyTensor);
    op->ConnectInput(MakeTensorUsage(TensorUsage::State, 0), scratch);
    op->ConnectInput(MakeTensorUsage(TensorUsage::State, 1), scratchFast);
    op->ConnectInput(TensorUsage::IFM0, input).Set(perTensorQuant);
    op->ConnectInput(TensorUsage::IFM1, inputNoQuant);
    op->ConnectOutput(TensorUsage::OFM, output).Set(perChannelQuant);
    op->ConnectOutput(MakeTensorUsage(TensorUsage::OFM, 1), variable).Set(variableQuant);

    // Create graph
    std::vector<std::unique_ptr<Graph>> graphs;
    graphs.push_back(std::make_unique<Graph>(GraphNotation::TFLite));
    graphs[0]->AddInput(input);
    graphs[0]->AddInput(inputNoQuant);
    graphs[0]->AddOutput(output);
    graphs[0]->AddPersistent(variable);

    // Create tensor address map
    std::vector<std::unordered_map<const Tensor *, Address>> addresses;
    addresses.push_back({});
    addresses[0][commandStreamTensor.get()] = 44;
    addresses[0][readOnlyTensor.get()] = 55;
    addresses[0][scratch.get()] = 66;
    addresses[0][scratchFast.get()] = 77;
    addresses[0][input.get()] = 88;
    addresses[0][inputNoQuant.get()] = 89;
    addresses[0][output.get()] = 99;
    addresses[0][variable.get()] = 111;

    // Create the raw output blobs
    RawWriter writer;
    auto blobs = writer.Serialise(graphs, addresses);

    // Expect one blob per tensor
    REQUIRE(blobs.size() == 8);

    // Check command stream
    {
        size_t dataSize = sizeof(regor_raw_tensor_header_t) + 4;
        REQUIRE(blobs[0].second == dataSize);

        auto &data = blobs[0].first;
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_COMMAND_STREAM);
        REQUIRE(header.flags == 0);
        REQUIRE(header.tensor.command_stream.size == 4);

        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 0] == 'C');
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 1] == 'O');
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 2] == 'P');
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 3] == '1');
    }

    // Check read only
    {
        size_t dataSize = sizeof(regor_raw_tensor_header_t) + 5;
        REQUIRE(blobs[1].second == dataSize);

        auto &data = blobs[1].first;
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_READ_ONLY);
        REQUIRE(header.flags == 0);
        REQUIRE(header.tensor.read_only.size == 5);
        REQUIRE(header.tensor.read_only.region == 0);

        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 0] == 21);
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 1] == 22);
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 2] == 23);
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 3] == 24);
        REQUIRE(data[sizeof(regor_raw_tensor_header_t) + 4] == 25);
    }

    // Check scratch
    {
        size_t dataSize = sizeof(regor_raw_tensor_header_t);
        REQUIRE(blobs[2].second == dataSize);

        auto &data = blobs[2].first;
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_SCRATCH);
        REQUIRE(header.flags == 0);
        REQUIRE(header.tensor.scratch.size == 6);
        REQUIRE(header.tensor.scratch.region == 1);
        REQUIRE(header.tensor.scratch.address == 66);
    }

    // Check scratch fast
    {
        size_t dataSize = sizeof(regor_raw_tensor_header_t);
        REQUIRE(blobs[3].second == dataSize);

        auto &data = blobs[3].first;
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_SCRATCH_FAST);
        REQUIRE(header.flags == 0);
        REQUIRE(header.tensor.scratch_fast.size == 7);
        REQUIRE(header.tensor.scratch_fast.region == 2);
        REQUIRE(header.tensor.scratch_fast.address == 77);
    }

    auto readQuant = [](const std::unique_ptr<const uint8_t[]> &buffer, size_t offset) -> const regor_raw_quantization_t *
    { return reinterpret_cast<const regor_raw_quantization_t *>(buffer.get() + offset); };
    auto expectedQuantBytes = [](uint32_t count)
    { return sizeof(regor_raw_quantization_t) + (size_t(count) * sizeof(float)) + (size_t(count) * sizeof(int32_t)); };

    // Input quantization (per-tensor)
    {
        auto &data = blobs[4].first;
        REQUIRE(blobs[4].second >= sizeof(regor_raw_tensor_header_t));
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_INPUT);
        REQUIRE((header.flags & REGOR_RAW_TENSOR_FLAG_HAS_QUANTIZATION) != 0);
        REQUIRE(header.tensor.input.size == 2 * 3 * 4 * 5 * 6 * 7);
        REQUIRE(header.tensor.input.region == 1);
        REQUIRE(header.tensor.input.address == 88);
        REQUIRE(header.tensor.input.element_size == 1);
        REQUIRE(header.tensor.input.shape[0] == 2);
        REQUIRE(header.tensor.input.shape[1] == 3);
        REQUIRE(header.tensor.input.shape[2] == 4);
        REQUIRE(header.tensor.input.shape[3] == 5);
        REQUIRE(header.tensor.input.shape[4] == 6);
        REQUIRE(header.tensor.input.shape[5] == 7);

        const auto *quantHeader = readQuant(data, sizeof(regor_raw_tensor_header_t));
        auto quantSize = blobs[4].second - sizeof(regor_raw_tensor_header_t);
        const uint32_t count = quantHeader->count;
        REQUIRE(quantSize >= expectedQuantBytes(count));
        REQUIRE(count == 1);
        const auto *scales = reinterpret_cast<const float *>(quantHeader + 1);
        const auto *zeroPoints = reinterpret_cast<const int32_t *>(scales + count);
        REQUIRE(scales[0] == Catch::Approx(2.0f));
        REQUIRE(zeroPoints[0] == 3);
    }

    // Unquantized input
    {
        auto &data = blobs[5].first;
        REQUIRE(blobs[5].second == sizeof(regor_raw_tensor_header_t));
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_INPUT);
        REQUIRE((header.flags & REGOR_RAW_TENSOR_FLAG_HAS_QUANTIZATION) == 0);
        REQUIRE(header.tensor.input.size == 2 * 3 * 4 * 5 * 6 * 7);
        REQUIRE(header.tensor.input.region == 1);
        REQUIRE(header.tensor.input.address == 89);
        REQUIRE(header.tensor.input.element_size == 1);
    }

    // Output quantization (per-channel)
    {
        auto &data = blobs[6].first;
        REQUIRE(blobs[6].second >= sizeof(regor_raw_tensor_header_t));
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_OUTPUT);
        REQUIRE((header.flags & REGOR_RAW_TENSOR_FLAG_HAS_QUANTIZATION) != 0);
        REQUIRE(header.tensor.output.size == 3 * 4 * 5 * 6 * 7 * 8);
        REQUIRE(header.tensor.output.region == 1);
        REQUIRE(header.tensor.output.address == 99);
        REQUIRE(header.tensor.output.element_size == 1);
        REQUIRE(header.tensor.output.shape[0] == 3);
        REQUIRE(header.tensor.output.shape[1] == 4);
        REQUIRE(header.tensor.output.shape[2] == 5);
        REQUIRE(header.tensor.output.shape[3] == 6);
        REQUIRE(header.tensor.output.shape[4] == 7);
        REQUIRE(header.tensor.output.shape[5] == 8);

        const auto *quantHeader = readQuant(data, sizeof(regor_raw_tensor_header_t));
        auto quantSize = blobs[6].second - sizeof(regor_raw_tensor_header_t);
        const uint32_t count = quantHeader->count;
        REQUIRE(quantSize >= expectedQuantBytes(count));
        REQUIRE(count == 2);
        const auto *scales = reinterpret_cast<const float *>(quantHeader + 1);
        const auto *zeroPoints = reinterpret_cast<const int32_t *>(scales + count);
        REQUIRE(scales[0] == Catch::Approx(1.0f));
        REQUIRE(scales[1] == Catch::Approx(2.0f));
        REQUIRE(zeroPoints[0] == 0);
        REQUIRE(zeroPoints[1] == 10);
    }

    // Variable quantization (per-tensor)
    {
        auto &data = blobs[7].first;
        REQUIRE(blobs[7].second >= sizeof(regor_raw_tensor_header_t));
        regor_raw_tensor_header_t header;
        std::copy_n(data.get(), sizeof(header), reinterpret_cast<uint8_t *>(&header));
        REQUIRE(header.type == regor_raw_tensor_header_t::RAW_TENSOR_TYPE_VARIABLE);
        REQUIRE((header.flags & REGOR_RAW_TENSOR_FLAG_HAS_QUANTIZATION) != 0);
        REQUIRE(header.tensor.variable.size == 4 * 5 * 6 * 7 * 8 * 9);
        REQUIRE(header.tensor.variable.region == 1);
        REQUIRE(header.tensor.variable.address == 111);
        REQUIRE(header.tensor.variable.element_size == 1);
        REQUIRE(header.tensor.variable.shape[0] == 4);
        REQUIRE(header.tensor.variable.shape[1] == 5);
        REQUIRE(header.tensor.variable.shape[2] == 6);
        REQUIRE(header.tensor.variable.shape[3] == 7);
        REQUIRE(header.tensor.variable.shape[4] == 8);
        REQUIRE(header.tensor.variable.shape[5] == 9);

        const auto *quantHeader = readQuant(data, sizeof(regor_raw_tensor_header_t));
        auto quantSize = blobs[7].second - sizeof(regor_raw_tensor_header_t);
        const uint32_t count = quantHeader->count;
        REQUIRE(quantSize >= expectedQuantBytes(count));
        REQUIRE(count == 1);
        const auto *scales = reinterpret_cast<const float *>(quantHeader + 1);
        const auto *zeroPoints = reinterpret_cast<const int32_t *>(scales + count);
        REQUIRE(scales[0] == Catch::Approx(5.0f));
        REQUIRE(zeroPoints[0] == 0);
    }
}
