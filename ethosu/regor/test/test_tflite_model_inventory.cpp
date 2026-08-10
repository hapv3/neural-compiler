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
