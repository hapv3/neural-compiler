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
#include "tflite/tflite_schema_generated.hpp"

#include <flatbuffers/flatbuffers.h>
#include <catch_all.hpp>

#include "regor.h"

using namespace regor;

TEST_CASE("raw_writer - metadata")
{
    const bool separateIORegions = GENERATE(false, true);

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

    // Build tensors with quantization metadata
    const auto input = std::make_shared<Tensor>("input_1", DataType::Int8, Shape({2, 3, 4, 5, 6, 7}));
    const auto inputNoQuant = std::make_shared<Tensor>("input_2", DataType::Int8, Shape({2, 3, 4, 5, 6, 7}));
    const auto output = std::make_shared<Tensor>("output_1", DataType::Int8, Shape({3, 4, 5, 6, 7, 8}));
    const auto variable = std::make_shared<Tensor>("variable_1", DataType::Int8, Shape({4, 5, 6, 7, 8, 9}));
    REQUIRE_FALSE(input->IsConstant());
    REQUIRE_FALSE(inputNoQuant->IsConstant());
    REQUIRE_FALSE(output->IsConstant());
    REQUIRE_FALSE(variable->IsConstant());

    auto createPassthroughTensor =
        [](const char *name, const std::vector<int32_t> &shape, const std::vector<float> *scale,
            const std::vector<int64_t> *zeroPoint, int32_t quantizedDimension = 0, bool isVariable = false)
    {
        flatbuffers::FlatBufferBuilder builder;
        flatbuffers::Offset<tflite::QuantizationParameters> quantization = 0;
        if ( scale && zeroPoint )
        {
            quantization = tflite::CreateQuantizationParametersDirect(
                builder, nullptr, nullptr, scale, zeroPoint, tflite::QuantizationDetails::NONE, 0, quantizedDimension);
        }

        const auto tensor = tflite::CreateTensorDirect(builder, &shape, tflite::TensorType::INT8, 0, name, quantization, isVariable);
        builder.Finish(tensor);

        size_t rawSize = 0;
        size_t rawOffset = 0;
        const uint8_t *raw = builder.ReleaseRaw(rawSize, rawOffset);
        UNUSED(rawSize);
        return std::make_pair(std::unique_ptr<const uint8_t[]>(raw), flatbuffers::GetRoot<tflite::Tensor>(&raw[rawOffset]));
    };

    const std::vector<float> inputScale = {2.0f};
    const std::vector<int64_t> inputZeroPoint = {3};
    const auto inputPassthrough = createPassthroughTensor("input_1", {2, 3, 4, 5, 6, 7}, &inputScale, &inputZeroPoint);

    const auto inputNoQuantPassthrough = createPassthroughTensor("input_2", {2, 3, 4, 5, 6, 7}, nullptr, nullptr);

    const std::vector<float> outputScale = {1.0f, 2.0f};
    const std::vector<int64_t> outputZeroPoint = {0, 10};
    const auto outputPassthrough = createPassthroughTensor("output_1", {3, 4, 5, 6, 7, 8}, &outputScale, &outputZeroPoint, 3);

    const std::vector<float> variableScale = {5.0f};
    const std::vector<int64_t> variableZeroPoint = {0};
    const auto variablePassthrough = createPassthroughTensor("variable_1", {4, 5, 6, 7, 8, 9}, &variableScale, &variableZeroPoint, 0, true);

    input->SetPassthrough(inputPassthrough.second);
    inputNoQuant->SetPassthrough(inputNoQuantPassthrough.second);
    output->SetPassthrough(outputPassthrough.second);
    variable->SetPassthrough(variablePassthrough.second);

    Quantization perTensorConnQuant;
    perTensorConnQuant.scales.push_back(QuantizedScale(8, 0));
    perTensorConnQuant.zeroPoints.push_back(9);

    Quantization noQuantConnQuant;
    noQuantConnQuant.scales.push_back(QuantizedScale(10, 0));
    noQuantConnQuant.zeroPoints.push_back(11);

    Quantization perChannelConnQuant;
    perChannelConnQuant.scales.push_back(QuantizedScale(12, 0));
    perChannelConnQuant.scales.push_back(QuantizedScale(13, 0));
    perChannelConnQuant.zeroPoints.push_back(14);
    perChannelConnQuant.zeroPoints.push_back(15);

    Quantization variableConnQuant;
    variableConnQuant.scales.push_back(QuantizedScale(16, 0));
    variableConnQuant.zeroPoints.push_back(17);

    // Create custom op
    auto op = std::make_shared<Operation>(OpType::CustomNpuOp);
    op->ConnectInput(MakeTensorUsage(TensorUsage::Params, 0), commandStreamTensor);
    op->ConnectInput(MakeTensorUsage(TensorUsage::Params, 1), readOnlyTensor);
    op->ConnectInput(MakeTensorUsage(TensorUsage::State, 0), scratch);
    op->ConnectInput(MakeTensorUsage(TensorUsage::State, 1), scratchFast);
    op->ConnectInput(TensorUsage::IFM0, input).Set(perTensorConnQuant);
    op->ConnectInput(TensorUsage::IFM1, inputNoQuant).Set(noQuantConnQuant);
    op->ConnectOutput(TensorUsage::OFM, output).Set(perChannelConnQuant);
    op->ConnectOutput(MakeTensorUsage(TensorUsage::OFM, 1), variable).Set(variableConnQuant);

    // Create graph
    std::vector<std::unique_ptr<Graph>> graphs;
    graphs.push_back(std::make_unique<Graph>(GraphNotation::TFLite));
    graphs[0]->AddInput(input);
    graphs[0]->AddInput(inputNoQuant);
    graphs[0]->AddOutput(output);
    graphs[0]->AddPersistent(variable);

    // Create tensor address map
    std::vector<TensorAddressMap> addresses;
    addresses.push_back({});
    addresses[0][commandStreamTensor->Uid()] = {{MemUsage::None}, 44};
    addresses[0][readOnlyTensor->Uid()] = {{MemUsage::ReadOnly}, 55};
    addresses[0][scratch->Uid()] = {{MemUsage::FeatureMap}, 66};
    addresses[0][scratchFast->Uid()] = {{MemUsage::Staging}, 77};
    if ( separateIORegions )
    {
        addresses[0][input->Uid()] = {{MemUsage::FeatureMap, MemUsage::Input}, 88};
        addresses[0][inputNoQuant->Uid()] = {{MemUsage::FeatureMap, MemUsage::Input}, 89};
        addresses[0][output->Uid()] = {{MemUsage::FeatureMap, MemUsage::Output}, 99};
    }
    else
    {
        addresses[0][input->Uid()] = {{MemUsage::FeatureMap}, 88};
        addresses[0][inputNoQuant->Uid()] = {{MemUsage::FeatureMap}, 89};
        addresses[0][output->Uid()] = {{MemUsage::FeatureMap}, 99};
    }
    addresses[0][variable->Uid()] = {{MemUsage::FeatureMap}, 111};

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
        REQUIRE(header.tensor.input.region == (separateIORegions ? 3 : 1));
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
        REQUIRE(header.tensor.input.region == (separateIORegions ? 3 : 1));
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
        REQUIRE(header.tensor.output.region == (separateIORegions ? 4 : 1));
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

// Tests that we fail with a meaningful error message when raw writer encounters a passthrough op
TEST_CASE("raw_writer - passthrough op")
{
    const tflite::Model *tfliteModel = nullptr;
    const tflite::Operator *tfliteOp = nullptr;
    std::unique_ptr<const uint8_t[]> base;

    {
        flatbuffers::FlatBufferBuilder builder;
        std::vector<flatbuffers::Offset<tflite::OperatorCode>> codes;
        std::vector<flatbuffers::Offset<tflite::Operator>> operators;
        std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;

        // Create a subgraph with a single ABS operator
        codes.push_back(tflite::CreateOperatorCodeDirect(builder, 0, nullptr, 1, tflite::BuiltinOperator::ABS));
        operators.push_back(tflite::CreateOperator(
            builder, 0, 0, 0, tflite::BuiltinOptions::AbsOptions, tflite::CreateAbsOptions(builder).Union()));
        subgraphs.push_back(tflite::CreateSubGraphDirect(builder, nullptr, nullptr, nullptr, &operators));
        const auto model = tflite::CreateModelDirect(builder, 3 /* version */, &codes, &subgraphs);
        tflite::FinishModelBuffer(builder, model);

        // Get raw flatbuffer buffer
        size_t passthroughSize = 0;
        size_t passthroughOffset = 0;
        const uint8_t *raw = builder.ReleaseRaw(passthroughSize, passthroughOffset);

        // Extract pointers to the model and the ABS operator
        tfliteModel = tflite::GetModel(&raw[passthroughOffset]);
        assert(tfliteModel->operator_codes());
        auto tfliteSubgraphs = tfliteModel->subgraphs();
        assert(tfliteSubgraphs->size() == 1);
        auto tfliteOperators = (*tfliteSubgraphs)[0]->operators();
        assert(tfliteOperators->size() == 1);
        tfliteOp = (*tfliteOperators)[0];

        // Store the raw flatbuffer buffer so we don't leak it
        base = std::unique_ptr<const uint8_t[]>(raw);
    }

    // Create a passthrough op
    auto op = std::make_shared<Operation>(OpType::Passthrough);
    auto ifm = std::make_shared<Tensor>("ifm", DataType::Int8, Shape(1));
    auto ofm = std::make_shared<Tensor>("ofm", DataType::Int8, Shape(1));
    op->ConnectInput(TensorUsage::IFM, ifm);
    op->ConnectOutput(TensorUsage::OFM, ofm);
    op->SetPassthrough(tfliteOp);

    // Create a graph with the above passthrough op
    std::vector<std::unique_ptr<Graph>> graphs;
    graphs.push_back(std::make_unique<Graph>(GraphNotation::TFLite));
    graphs[0]->AddInput(ifm);
    graphs[0]->AddOutput(ofm);
    graphs[0]->SetPassthrough(tfliteModel);

    // Create an empty tensor address map - we don't need a complete tensor address map for this test
    std::vector<TensorAddressMap> addresses;
    addresses.push_back({});

    RawWriter writer;
    try
    {
        writer.Serialise(graphs, addresses);
        FAIL("Expected RawWriter to reject passthrough op");
    }
    catch ( const std::invalid_argument &ex )
    {
        const std::string message = ex.what();
        REQUIRE(message == "RawWriter expects a graph without passthrough/CPU operations (found ABS)");
    }
}
