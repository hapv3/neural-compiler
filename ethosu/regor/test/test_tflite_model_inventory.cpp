//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "tflite/tflite_model_inventory.hpp"
#include "tflite/tflite_schema_generated.hpp"

#include <catch_all.hpp>

#include <string>
#include <vector>

using namespace regor;

namespace
{

std::vector<uint8_t> CreateConvModel()
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<float> inputScale{0.125F};
    const std::vector<int64_t> inputZeroPoint{-3};
    const std::vector<float> weightScales{0.25F, 0.5F};
    const std::vector<int64_t> weightZeroPoints{0, 0};
    const auto inputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &inputScale, &inputZeroPoint, tflite::QuantizationDetails::NONE, 0, 0);
    const auto weightQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &weightScales, &weightZeroPoints, tflite::QuantizationDetails::NONE, 0, 0);
    const std::vector<int32_t> inputShape{1, 8, 8, 3};
    const std::vector<int32_t> weightShape{32, 3, 3, 3};
    const std::vector<int32_t> outputShape{1, 4, 4, 32};
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors{
        tflite::CreateTensorDirect(builder, &inputShape, tflite::TensorType::INT8, 0, "input\n0", inputQuant),
        tflite::CreateTensorDirect(builder, &weightShape, tflite::TensorType::INT8, 1, "weights", weightQuant),
        tflite::CreateTensorDirect(builder, &outputShape, tflite::TensorType::INT8, 0, "output", inputQuant),
    };
    const auto options = tflite::CreateConv2DOptions(
        builder, tflite::Padding::SAME, 2, 2, tflite::ActivationFunctionType::RELU6, 1, 1);
    const std::vector<int32_t> opInputs{0, 1, -1};
    const std::vector<int32_t> opOutputs{2};
    const auto serialisedInputs = builder.CreateVector(opInputs);
    const auto serialisedOutputs = builder.CreateVector(opOutputs);
    std::vector<flatbuffers::Offset<tflite::Operator>> operators{
        tflite::CreateOperator(builder, 0, serialisedInputs, serialisedOutputs,
            tflite::BuiltinOptions::Conv2DOptions, options.Union()),
    };
    const std::vector<int32_t> graphInputs{0};
    const std::vector<int32_t> graphOutputs{2};
    std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs{
        tflite::CreateSubGraphDirect(builder, &tensors, &graphInputs, &graphOutputs, &operators, "main"),
    };
    std::vector<flatbuffers::Offset<tflite::OperatorCode>> codes{
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::CONV_2D), nullptr, 3,
            tflite::BuiltinOperator::CONV_2D),
    };
    const auto model = tflite::CreateModelDirect(builder, 3, &codes, &subgraphs, "inventory test");
    tflite::FinishModelBuffer(builder, model);
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

std::vector<uint8_t> CreateAddChainModel()
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int32_t> shape{1, 2};
    const std::vector<uint8_t> constantValues{4, uint8_t(int8_t(-7))};
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers{
        tflite::CreateBufferDirect(builder),
        tflite::CreateBufferDirect(builder, &constantValues),
        tflite::CreateBufferDirect(builder),
    };
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
    for ( int index = 0; index < 5; ++index )
    {
        const uint32_t buffer = index == 0 ? 2 : (index == 1 ? 1 : 0);
        tensors.push_back(tflite::CreateTensorDirect(
            builder, &shape, tflite::TensorType::INT8, buffer, ("tensor" + std::to_string(index)).c_str()));
    }
    std::vector<flatbuffers::Offset<tflite::Operator>> operators;
    for ( int index = 0; index < 3; ++index )
    {
        const std::vector<int32_t> inputs{index == 0 ? 0 : index + 1, 1};
        const std::vector<int32_t> outputs{index + 2};
        operators.push_back(tflite::CreateOperatorDirect(builder, 0, &inputs, &outputs,
            tflite::BuiltinOptions::AddOptions, tflite::CreateAddOptions(builder).Union()));
    }
    const std::vector<int32_t> graphInputs{0};
    const std::vector<int32_t> graphOutputs{4};
    std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs{
        tflite::CreateSubGraphDirect(builder, &tensors, &graphInputs, &graphOutputs, &operators, "main"),
    };
    std::vector<flatbuffers::Offset<tflite::OperatorCode>> codes{
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::ADD), nullptr, 2,
            tflite::BuiltinOperator::ADD),
    };
    const auto model = tflite::CreateModelDirect(builder, 3, &codes, &subgraphs, "topology test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

}  // namespace

TEST_CASE("TFLite model inventory is deterministic and describes model contracts")
{
    const auto model = CreateConvModel();
    const auto first = BuildTfLiteModelInventory(model.data(), model.size(), "model\"name.tflite");
    const auto second = BuildTfLiteModelInventory(model.data(), model.size(), "model\"name.tflite");

    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(first.json == second.json);
    REQUIRE(first.json.find("\"name\":\"model\\\"name.tflite\"") != std::string::npos);
    REQUIRE(first.json.find("\"hash_algorithm\":\"md5\"") != std::string::npos);
    REQUIRE(first.json.find("\"name\":\"CONV_2D\",\"count\":1") != std::string::npos);
    REQUIRE(first.json.find("\"name\":\"input\\n0\",\"shape\":[1,8,8,3]") != std::string::npos);
    REQUIRE(first.json.find("\"scales\":[0.125],\"zero_points\":[-3]") != std::string::npos);
    REQUIRE(first.json.find("\"scales\":[0.25,0.5],\"zero_points\":[0,0],\"quantized_dimension\":0") !=
            std::string::npos);
    REQUIRE(first.json.find("\"padding\":\"SAME\",\"stride_h\":2,\"stride_w\":2") != std::string::npos);
    REQUIRE(first.json.find("\"activation\":\"RELU6\"") != std::string::npos);
    REQUIRE(first.json.find("\"inputs\":[0,1,-1],\"outputs\":[2]") != std::string::npos);
}

TEST_CASE("TFLite model inventory rejects malformed input")
{
    const std::vector<uint8_t> invalid{'T', 'F', 'L', '3', 0, 1, 2, 3};
    const auto inventory = BuildTfLiteModelInventory(invalid.data(), invalid.size(), "invalid.tflite");
    REQUIRE_FALSE(inventory);
    REQUIRE(inventory.json.empty());
    REQUIRE(inventory.error == "Input is not a valid TFLite FlatBuffer");
}

TEST_CASE("TFLite topology micrograph preserves a selected source chain")
{
    const auto source = CreateAddChainModel();
    const auto extracted = BuildTfLiteTopologyMicrograph(source.data(), source.size(), 0, {1, 2}, "chain.tflite");
    REQUIRE(extracted);
    REQUIRE(extracted.provenanceJson.find("\"source_operator_indices\":[1,2]") != std::string::npos);
    REQUIRE(extracted.provenanceJson.find("\"source_input_tensor_indices\":[2]") != std::string::npos);
    REQUIRE(extracted.provenanceJson.find("\"source_output_tensor_indices\":[4]") != std::string::npos);

    flatbuffers::Verifier verifier(extracted.model.data(), extracted.model.size());
    REQUIRE(tflite::VerifyModelBuffer(verifier));
    const auto *model = tflite::GetModel(extracted.model.data());
    REQUIRE(model->subgraphs()->size() == 1);
    const auto *graph = model->subgraphs()->Get(0);
    REQUIRE(graph->operators()->size() == 2);
    REQUIRE(graph->tensors()->size() == 4);
    REQUIRE(graph->inputs()->size() == 1);
    REQUIRE(graph->inputs()->Get(0) == 1);
    REQUIRE(graph->outputs()->size() == 1);
    REQUIRE(graph->outputs()->Get(0) == 3);
    REQUIRE(graph->operators()->Get(0)->inputs()->Get(0) == 1);
    REQUIRE(graph->operators()->Get(0)->inputs()->Get(1) == 0);
    REQUIRE(graph->operators()->Get(0)->outputs()->Get(0) == 2);
    REQUIRE(graph->operators()->Get(1)->inputs()->Get(0) == 2);
    REQUIRE(graph->operators()->Get(1)->outputs()->Get(0) == 3);
    REQUIRE(graph->tensors()->Get(0)->buffer() == 1);
    REQUIRE(model->buffers()->size() == 2);
    REQUIRE(model->buffers()->Get(1)->data()->size() == 2);
    REQUIRE(int8_t(model->buffers()->Get(1)->data()->Get(0)) == 4);
    REQUIRE(int8_t(model->buffers()->Get(1)->data()->Get(1)) == -7);
}

TEST_CASE("TFLite topology micrograph rejects invalid selections")
{
    const auto source = CreateAddChainModel();
    REQUIRE_FALSE(BuildTfLiteTopologyMicrograph(source.data(), source.size(), 0, {}, "chain.tflite"));
    REQUIRE_FALSE(BuildTfLiteTopologyMicrograph(source.data(), source.size(), 0, {1, 1}, "chain.tflite"));
    REQUIRE_FALSE(BuildTfLiteTopologyMicrograph(source.data(), source.size(), 0, {3}, "chain.tflite"));
    REQUIRE_FALSE(BuildTfLiteTopologyMicrograph(source.data(), source.size(), 1, {0}, "chain.tflite"));
}

TEST_CASE("TFLite topology micrograph treats an empty nonzero buffer as a graph input")
{
    const auto source = CreateAddChainModel();
    const auto extracted = BuildTfLiteTopologyMicrograph(source.data(), source.size(), 0, {0}, "chain.tflite");
    REQUIRE(extracted);
    REQUIRE(extracted.provenanceJson.find("\"source_input_tensor_indices\":[0]") != std::string::npos);
    const auto *model = tflite::GetModel(extracted.model.data());
    REQUIRE(model->subgraphs()->Get(0)->inputs()->size() == 1);
}
