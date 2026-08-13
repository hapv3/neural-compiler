//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "architecture/neuralai/neural_ai.hpp"
#include "architecture/neuralai/neural_ai_constraints.hpp"
#include "architecture/neuralai/neural_ai_op_config.hpp"
#include "architecture/neuralai/neural_ai_performance.hpp"
#include "architecture/neuralai/neural_ai_weight_encoder.hpp"
#include "compiler/compiler.hpp"
#include "compiler/graphir_optimiser.hpp"
#include "compiler/high_level_command_stream_generator.hpp"
#include "compiler/neural_ai_command_generator.hpp"
#include "compiler/neural_ai_graph_optimiser.hpp"
#include "compiler/neural_ai_writer.hpp"
#include "compiler/scheduler.hpp"
#include "compiler/scheduler_packing.hpp"
#include "tflite/tflite_schema_generated.hpp"
#include "tflite/tflite_supported_operators.hpp"
#include "util.hpp"

#include <catch_all.hpp>

#include <cmath>
#include <cstring>
#include <limits>

#include "regor.h"

using namespace regor;

namespace
{

uint32_t Read32(const uint8_t *data)
{
    return uint32_t(data[0]) | (uint32_t(data[1]) << 8) |
           (uint32_t(data[2]) << 16) | (uint32_t(data[3]) << 24);
}

uint16_t Read16(const uint8_t *data)
{
    return uint16_t(data[0]) | (uint16_t(data[1]) << 8);
}

uint32_t Read32(const std::vector<uint8_t> &data, size_t offset)
{
    return Read32(data.data() + offset);
}

uint16_t Read16(const std::vector<uint8_t> &data, size_t offset)
{
    return uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
}

flatbuffers::DetachedBuffer BuildFullyConnectedModel(int rows, int depthK, int depthN)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<float> scale = {1.0f};
    const std::vector<int64_t> zeroPoint = {0};
    const auto inputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto weightQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto biasQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);

    std::vector<uint8_t> weightData(depthK * depthN, 1);
    std::vector<int32_t> bias(depthN, 0);
    std::vector<uint8_t> biasData(bias.size() * sizeof(int32_t));
    std::memcpy(biasData.data(), bias.data(), biasData.size());
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
        tflite::CreateBufferDirect(builder, &weightData),
        tflite::CreateBufferDirect(builder, &biasData),
    };
    const std::vector<int32_t> inputShape = rows == 1 ?
        std::vector<int32_t>{1, depthK} : std::vector<int32_t>{1, rows, depthK};
    const std::vector<int32_t> weightShape = {depthN, depthK};
    const std::vector<int32_t> biasShape = {depthN};
    const std::vector<int32_t> outputShape = rows == 1 ?
        std::vector<int32_t>{1, depthN} : std::vector<int32_t>{1, rows, depthN};
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(builder, &inputShape, tflite::TensorType::INT8, 0, "input", inputQuant),
        tflite::CreateTensorDirect(builder, &weightShape, tflite::TensorType::INT8, 1, "weights", weightQuant),
        tflite::CreateTensorDirect(builder, &biasShape, tflite::TensorType::INT32, 2, "bias", biasQuant),
        tflite::CreateTensorDirect(builder, &outputShape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };
    const auto options = tflite::CreateFullyConnectedOptions(builder);
    const std::vector<int32_t> opInputs = {0, 1, 2};
    const std::vector<int32_t> opOutputs = {3};
    const std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 0, &opInputs, &opOutputs,
            tflite::BuiltinOptions::FullyConnectedOptions, options.Union()),
    };
    const std::vector<int32_t> graphInputs = {0};
    const std::vector<int32_t> graphOutputs = {3};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(
            builder, &tensors, &graphInputs, &graphOutputs, &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::FULLY_CONNECTED),
            nullptr, 1, tflite::BuiltinOperator::FULLY_CONNECTED),
    };
    const auto model = tflite::CreateModelDirect(builder, 3, &operatorCodes, &subgraphs,
        "Neural-AI FullyConnected test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

flatbuffers::DetachedBuffer BuildFullyConnectedModel(int depthK, int depthN)
{
    return BuildFullyConnectedModel(1, depthK, depthN);
}

flatbuffers::DetachedBuffer BuildPointwiseConvModel(int height, int width, int depthK, int depthN,
    tflite::ActivationFunctionType activation = tflite::ActivationFunctionType::NONE)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<float> scale = {1.0f};
    const std::vector<int64_t> zeroPoint = {0};
    const auto inputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto weightQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto biasQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);

    std::vector<uint8_t> weightData(size_t(depthK) * depthN, 1);
    std::vector<int32_t> bias(depthN, 0);
    std::vector<uint8_t> biasData(bias.size() * sizeof(int32_t));
    std::memcpy(biasData.data(), bias.data(), biasData.size());
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
        tflite::CreateBufferDirect(builder, &weightData),
        tflite::CreateBufferDirect(builder, &biasData),
    };
    const std::vector<int32_t> inputShape = {1, height, width, depthK};
    const std::vector<int32_t> weightShape = {depthN, 1, 1, depthK};
    const std::vector<int32_t> biasShape = {depthN};
    const std::vector<int32_t> outputShape = {1, height, width, depthN};
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(builder, &inputShape, tflite::TensorType::INT8, 0, "input", inputQuant),
        tflite::CreateTensorDirect(builder, &weightShape, tflite::TensorType::INT8, 1, "weights", weightQuant),
        tflite::CreateTensorDirect(builder, &biasShape, tflite::TensorType::INT32, 2, "bias", biasQuant),
        tflite::CreateTensorDirect(builder, &outputShape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };
    const auto options = tflite::CreateConv2DOptions(
        builder, tflite::Padding::VALID, 1, 1, activation, 1, 1,
        tflite::TensorType::INT32);
    const std::vector<int32_t> opInputs = {0, 1, 2};
    const std::vector<int32_t> opOutputs = {3};
    const std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 0, &opInputs, &opOutputs,
            tflite::BuiltinOptions::Conv2DOptions, options.Union()),
    };
    const std::vector<int32_t> graphInputs = {0};
    const std::vector<int32_t> graphOutputs = {3};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(
            builder, &tensors, &graphInputs, &graphOutputs, &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::CONV_2D),
            nullptr, 1, tflite::BuiltinOperator::CONV_2D),
    };
    const auto model = tflite::CreateModelDirect(builder, 3, &operatorCodes, &subgraphs,
        "Neural-AI pointwise Conv test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

flatbuffers::DetachedBuffer BuildAddModel(float lhsScale = 1.0f, float rhsScale = 1.0f,
    float outputScale = 1.0f, int64_t zeroPoint = 0,
    tflite::ActivationFunctionType activation = tflite::ActivationFunctionType::NONE)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> zeroPoints = {zeroPoint};
    const std::vector<float> lhsScales = {lhsScale};
    const std::vector<float> rhsScales = {rhsScale};
    const std::vector<float> outputScales = {outputScale};
    const auto lhsQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &lhsScales, &zeroPoints);
    const auto rhsQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &rhsScales, &zeroPoints);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &outputScales, &zeroPoints);
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
    };
    const std::vector<int32_t> shape = {1, 2, 2, 32};
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(builder, &shape, tflite::TensorType::INT8, 0, "lhs", lhsQuant),
        tflite::CreateTensorDirect(builder, &shape, tflite::TensorType::INT8, 0, "rhs", rhsQuant),
        tflite::CreateTensorDirect(builder, &shape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };
    const auto options = tflite::CreateAddOptions(builder, activation, false);
    const std::vector<int32_t> opInputs = {0, 1};
    const std::vector<int32_t> opOutputs = {2};
    const std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 0, &opInputs, &opOutputs,
            tflite::BuiltinOptions::AddOptions, options.Union()),
    };
    const std::vector<int32_t> graphInputs = {0, 1};
    const std::vector<int32_t> graphOutputs = {2};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(
            builder, &tensors, &graphInputs, &graphOutputs, &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::ADD),
            nullptr, 1, tflite::BuiltinOperator::ADD),
    };
    const auto model = tflite::CreateModelDirect(
        builder, 3, &operatorCodes, &subgraphs, "Neural-AI Add test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

flatbuffers::DetachedBuffer BuildSigmoidModel()
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<float> inputScales = {0.125f};
    const std::vector<float> outputScales = {1.0f / 256.0f};
    const std::vector<int64_t> inputZeroPoints = {0};
    const std::vector<int64_t> outputZeroPoints = {-128};
    const auto inputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &inputScales, &inputZeroPoints);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &outputScales, &outputZeroPoints);
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
    };
    const std::vector<int32_t> shape = {1, 2, 3, 33};
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(builder, &shape, tflite::TensorType::INT8, 0, "input", inputQuant),
        tflite::CreateTensorDirect(builder, &shape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };
    const std::vector<int32_t> opInputs = {0};
    const std::vector<int32_t> opOutputs = {1};
    const std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 0, &opInputs, &opOutputs),
    };
    const std::vector<int32_t> graphInputs = {0};
    const std::vector<int32_t> graphOutputs = {1};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(
            builder, &tensors, &graphInputs, &graphOutputs, &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::LOGISTIC),
            nullptr, 1, tflite::BuiltinOperator::LOGISTIC),
    };
    const auto model = tflite::CreateModelDirect(
        builder, 3, &operatorCodes, &subgraphs, "Neural-AI Sigmoid test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

flatbuffers::DetachedBuffer BuildSiluModel(bool reverseMulInputs = false)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<float> inputScales = {0.125f};
    const std::vector<float> sigmoidScales = {1.0f / 256.0f};
    const std::vector<float> outputScales = {1.0f / 16.0f};
    const std::vector<int64_t> inputZeroPoints = {0};
    const std::vector<int64_t> sigmoidZeroPoints = {-128};
    const std::vector<int64_t> outputZeroPoints = {0};
    const auto inputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &inputScales, &inputZeroPoints);
    const auto sigmoidQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &sigmoidScales, &sigmoidZeroPoints);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &outputScales, &outputZeroPoints);
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
    };
    const std::vector<int32_t> shape = {1, 2, 3, 33};
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(builder, &shape, tflite::TensorType::INT8, 0, "input", inputQuant),
        tflite::CreateTensorDirect(builder, &shape, tflite::TensorType::INT8, 0, "sigmoid", sigmoidQuant),
        tflite::CreateTensorDirect(builder, &shape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };
    const std::vector<int32_t> sigmoidInputs = {0};
    const std::vector<int32_t> sigmoidOutputs = {1};
    const std::vector<int32_t> mulInputs = reverseMulInputs ?
        std::vector<int32_t>{1, 0} : std::vector<int32_t>{0, 1};
    const std::vector<int32_t> mulOutputs = {2};
    const auto mulOptions = tflite::CreateMulOptions(builder);
    const std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 0, &sigmoidInputs, &sigmoidOutputs),
        tflite::CreateOperatorDirect(builder, 1, &mulInputs, &mulOutputs,
            tflite::BuiltinOptions::MulOptions, mulOptions.Union()),
    };
    const std::vector<int32_t> graphInputs = {0};
    const std::vector<int32_t> graphOutputs = {2};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(
            builder, &tensors, &graphInputs, &graphOutputs, &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::LOGISTIC),
            nullptr, 1, tflite::BuiltinOperator::LOGISTIC),
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::MUL),
            nullptr, 1, tflite::BuiltinOperator::MUL),
    };
    const auto model = tflite::CreateModelDirect(
        builder, 3, &operatorCodes, &subgraphs, "Neural-AI SiLU test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

flatbuffers::DetachedBuffer BuildClippingModel(tflite::BuiltinOperator activation)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<float> scales = {0.25f};
    const std::vector<int64_t> zeroPoints = {-3};
    const auto inputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scales, &zeroPoints);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scales, &zeroPoints);
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
    };
    const std::vector<int32_t> shape = {1, 2, 3, 33};
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(builder, &shape, tflite::TensorType::INT8, 0, "input", inputQuant),
        tflite::CreateTensorDirect(builder, &shape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };
    const std::vector<int32_t> opInputs = {0};
    const std::vector<int32_t> opOutputs = {1};
    const std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 0, &opInputs, &opOutputs),
    };
    const std::vector<int32_t> graphInputs = {0};
    const std::vector<int32_t> graphOutputs = {1};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(
            builder, &tensors, &graphInputs, &graphOutputs, &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(builder, int8_t(activation), nullptr, 1, activation),
    };
    const auto model = tflite::CreateModelDirect(
        builder, 3, &operatorCodes, &subgraphs, "Neural-AI clipping test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

flatbuffers::DetachedBuffer BuildGlobalAvgPoolModel(
    int height = 2, int width = 3, int channels = 33,
    float inputScale = 0.25f, float outputScale = 0.25f, int64_t zeroPoint = -3,
    tflite::ActivationFunctionType activation = tflite::ActivationFunctionType::NONE,
    int filterHeight = 0, int filterWidth = 0)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> zeroPoints = {zeroPoint};
    const std::vector<float> inputScales = {inputScale};
    const std::vector<float> outputScales = {outputScale};
    const auto inputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &inputScales, &zeroPoints);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &outputScales, &zeroPoints);
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
    };
    const std::vector<int32_t> inputShape = {1, height, width, channels};
    const std::vector<int32_t> outputShape = {1, 1, 1, channels};
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(
            builder, &inputShape, tflite::TensorType::INT8, 0, "input", inputQuant),
        tflite::CreateTensorDirect(
            builder, &outputShape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };
    if ( filterHeight == 0 ) filterHeight = height;
    if ( filterWidth == 0 ) filterWidth = width;
    const auto options = tflite::CreatePool2DOptions(
        builder, tflite::Padding::VALID, 1, 1, filterWidth, filterHeight, activation);
    const std::vector<int32_t> opInputs = {0};
    const std::vector<int32_t> opOutputs = {1};
    const std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 0, &opInputs, &opOutputs,
            tflite::BuiltinOptions::Pool2DOptions, options.Union()),
    };
    const std::vector<int32_t> graphInputs = {0};
    const std::vector<int32_t> graphOutputs = {1};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(
            builder, &tensors, &graphInputs, &graphOutputs, &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(
            builder, int8_t(tflite::BuiltinOperator::AVERAGE_POOL_2D),
            nullptr, 1, tflite::BuiltinOperator::AVERAGE_POOL_2D),
    };
    const auto model = tflite::CreateModelDirect(
        builder, 3, &operatorCodes, &subgraphs, "Neural-AI global AvgPool test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

flatbuffers::DetachedBuffer BuildResizeNearestModel(
    int height = 2, int width = 3, int channels = 32,
    int outputHeight = 4, int outputWidth = 6, float outputScale = 0.25f)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<float> inputScales = {0.25f};
    const std::vector<float> outputScales = {outputScale};
    const std::vector<int64_t> zeroPoints = {-3};
    const auto inputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &inputScales, &zeroPoints);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &outputScales, &zeroPoints);
    const std::vector<int32_t> outputSize = {outputHeight, outputWidth};
    std::vector<uint8_t> outputSizeData(sizeof(int32_t) * outputSize.size());
    std::memcpy(outputSizeData.data(), outputSize.data(), outputSizeData.size());
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
        tflite::CreateBufferDirect(builder, &outputSizeData),
    };
    const std::vector<int32_t> inputShape = {1, height, width, channels};
    const std::vector<int32_t> sizeShape = {2};
    const std::vector<int32_t> outputShape = {1, outputHeight, outputWidth, channels};
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(
            builder, &inputShape, tflite::TensorType::INT8, 0, "input", inputQuant),
        tflite::CreateTensorDirect(
            builder, &sizeShape, tflite::TensorType::INT32, 1, "output_size"),
        tflite::CreateTensorDirect(
            builder, &outputShape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };
    const auto options = tflite::CreateResizeNearestNeighborOptions(builder, false, false);
    const std::vector<int32_t> opInputs = {0, 1};
    const std::vector<int32_t> opOutputs = {2};
    const std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 0, &opInputs, &opOutputs,
            tflite::BuiltinOptions::ResizeNearestNeighborOptions, options.Union()),
    };
    const std::vector<int32_t> graphInputs = {0};
    const std::vector<int32_t> graphOutputs = {2};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(
            builder, &tensors, &graphInputs, &graphOutputs, &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(
            builder, int8_t(tflite::BuiltinOperator::RESIZE_NEAREST_NEIGHBOR),
            nullptr, 1, tflite::BuiltinOperator::RESIZE_NEAREST_NEIGHBOR),
    };
    const auto model = tflite::CreateModelDirect(
        builder, 3, &operatorCodes, &subgraphs, "Neural-AI nearest resize test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

flatbuffers::DetachedBuffer BuildMaxPoolModel(
    int height = 4, int width = 4, int channels = 32,
    int filterHeight = 5, int filterWidth = 5,
    float outputScale = 0.25f,
    tflite::ActivationFunctionType activation = tflite::ActivationFunctionType::NONE)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<float> inputScales = {0.25f};
    const std::vector<float> outputScales = {outputScale};
    const std::vector<int64_t> zeroPoints = {-3};
    const auto inputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &inputScales, &zeroPoints);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &outputScales, &zeroPoints);
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
    };
    const std::vector<int32_t> shape = {1, height, width, channels};
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(
            builder, &shape, tflite::TensorType::INT8, 0, "input", inputQuant),
        tflite::CreateTensorDirect(
            builder, &shape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };
    const auto options = tflite::CreatePool2DOptions(
        builder, tflite::Padding::SAME, 1, 1, filterWidth, filterHeight, activation);
    const std::vector<int32_t> opInputs = {0};
    const std::vector<int32_t> opOutputs = {1};
    const std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 0, &opInputs, &opOutputs,
            tflite::BuiltinOptions::Pool2DOptions, options.Union()),
    };
    const std::vector<int32_t> graphInputs = {0};
    const std::vector<int32_t> graphOutputs = {1};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(
            builder, &tensors, &graphInputs, &graphOutputs, &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::MAX_POOL_2D),
            nullptr, 1, tflite::BuiltinOperator::MAX_POOL_2D),
    };
    const auto model = tflite::CreateModelDirect(
        builder, 3, &operatorCodes, &subgraphs, "Neural-AI MaxPool test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

flatbuffers::DetachedBuffer BuildConcatModel(
    int lhsChannels = 32, int rhsChannels = 32, int axis = 3,
    float outputScale = 0.25f, int height = 2, int width = 3, int tailChannels = 0)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<float> inputScales = {0.25f};
    const std::vector<float> outputScales = {outputScale};
    const std::vector<int64_t> zeroPoints = {-3};
    const auto lhsQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &inputScales, &zeroPoints);
    const auto rhsQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &inputScales, &zeroPoints);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &outputScales, &zeroPoints);
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
    };
    const std::vector<int32_t> lhsShape = {1, height, width, lhsChannels};
    const std::vector<int32_t> rhsShape = {1, height, width, rhsChannels};
    const std::vector<int32_t> tailShape = {1, height, width, tailChannels};
    const std::vector<int32_t> outputShape = axis == 2 ?
        std::vector<int32_t>{1, height, width * 2, lhsChannels} :
        std::vector<int32_t>{1, height, width, lhsChannels + rhsChannels + tailChannels};
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(
            builder, &lhsShape, tflite::TensorType::INT8, 0, "lhs", lhsQuant),
        tflite::CreateTensorDirect(
            builder, &rhsShape, tflite::TensorType::INT8, 0, "rhs", rhsQuant),
        tflite::CreateTensorDirect(
            builder, &outputShape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };
    if ( tailChannels > 0 )
        tensors.insert(tensors.begin() + 2, tflite::CreateTensorDirect(
            builder, &tailShape, tflite::TensorType::INT8, 0, "tail", rhsQuant));
    const auto options = tflite::CreateConcatenationOptions(
        builder, axis, tflite::ActivationFunctionType::NONE);
    const std::vector<int32_t> opInputs = tailChannels > 0 ?
        std::vector<int32_t>{0, 1, 2} : std::vector<int32_t>{0, 1};
    const std::vector<int32_t> opOutputs = {tailChannels > 0 ? 3 : 2};
    const std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 0, &opInputs, &opOutputs,
            tflite::BuiltinOptions::ConcatenationOptions, options.Union()),
    };
    const std::vector<int32_t> graphInputs = tailChannels > 0 ?
        std::vector<int32_t>{0, 1, 2} : std::vector<int32_t>{0, 1};
    const std::vector<int32_t> graphOutputs = {tailChannels > 0 ? 3 : 2};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(
            builder, &tensors, &graphInputs, &graphOutputs, &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::CONCATENATION),
            nullptr, 1, tflite::BuiltinOperator::CONCATENATION),
    };
    const auto model = tflite::CreateModelDirect(
        builder, 3, &operatorCodes, &subgraphs, "Neural-AI Concat test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

flatbuffers::DetachedBuffer BuildHeadTransposeModel(
    int height = 10, int width = 10, int channels = 144,
    std::vector<int32_t> permutation = {0, 3, 1, 2})
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<float> scales = {0.265464634f};
    const std::vector<int64_t> zeroPoints = {63};
    const auto inputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scales, &zeroPoints);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scales, &zeroPoints);
    std::vector<uint8_t> permutationData(permutation.size() * sizeof(int32_t));
    std::memcpy(permutationData.data(), permutation.data(), permutationData.size());
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
        tflite::CreateBufferDirect(builder, &permutationData),
    };
    const std::vector<int32_t> inputShape = {1, height, width, channels};
    const std::vector<int32_t> permutationShape = {int(permutation.size())};
    std::vector<int32_t> outputShape(inputShape.size());
    for ( int i = 0; i < int(permutation.size()); ++i )
        outputShape[i] = inputShape[permutation[i]];
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(
            builder, &inputShape, tflite::TensorType::INT8, 0, "input", inputQuant),
        tflite::CreateTensorDirect(
            builder, &permutationShape, tflite::TensorType::INT32, 1, "permutation"),
        tflite::CreateTensorDirect(
            builder, &outputShape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };
    const std::vector<int32_t> opInputs = {0, 1};
    const std::vector<int32_t> opOutputs = {2};
    const auto options = tflite::CreateTransposeOptions(builder);
    const std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 0, &opInputs, &opOutputs,
            tflite::BuiltinOptions::TransposeOptions, options.Union()),
    };
    const std::vector<int32_t> graphInputs = {0};
    const std::vector<int32_t> graphOutputs = {2};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(
            builder, &tensors, &graphInputs, &graphOutputs, &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::TRANSPOSE),
            nullptr, 1, tflite::BuiltinOperator::TRANSPOSE),
    };
    const auto model = tflite::CreateModelDirect(
        builder, 3, &operatorCodes, &subgraphs, "Neural-AI head Transpose test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

flatbuffers::DetachedBuffer BuildViewModel(
    tflite::BuiltinOperator viewOperator, bool followedByAdd)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> zeroPoints = {0};
    const std::vector<float> scales = {1.0f};
    const auto lhsQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scales, &zeroPoints);
    const auto rhsQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scales, &zeroPoints);
    const auto reshapedQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scales, &zeroPoints);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scales, &zeroPoints);
    const std::vector<int32_t> addShape = {1, 2, 2, 32};
    std::vector<int32_t> parameterData;
    std::vector<int32_t> parameterShape;
    std::vector<int32_t> lhsShape;
    if ( viewOperator == tflite::BuiltinOperator::RESHAPE )
    {
        lhsShape = {1, 4, 32};
        parameterData = addShape;
        parameterShape = {4};
    }
    else if ( viewOperator == tflite::BuiltinOperator::SQUEEZE )
    {
        lhsShape = {1, 1, 2, 2, 32};
    }
    else
    {
        REQUIRE(viewOperator == tflite::BuiltinOperator::EXPAND_DIMS);
        lhsShape = {2, 2, 32};
        parameterData = {0};
    }
    std::vector<uint8_t> parameterBytes(parameterData.size() * sizeof(int32_t));
    if ( !parameterBytes.empty() )
        std::memcpy(parameterBytes.data(), parameterData.data(), parameterBytes.size());
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
    };
    if ( !parameterBytes.empty() )
        buffers.push_back(tflite::CreateBufferDirect(builder, &parameterBytes));
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(builder, &lhsShape, tflite::TensorType::INT8, 0, "lhs", lhsQuant),
        tflite::CreateTensorDirect(builder, &addShape, tflite::TensorType::INT8, 0, "rhs", rhsQuant),
        tflite::CreateTensorDirect(
            builder, &addShape, tflite::TensorType::INT8, 0, "view", reshapedQuant),
        tflite::CreateTensorDirect(
            builder, &addShape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };
    std::vector<int32_t> viewInputs = {0};
    tflite::BuiltinOptions viewOptionsType = tflite::BuiltinOptions::NONE;
    flatbuffers::Offset<void> viewOptions;
    if ( viewOperator == tflite::BuiltinOperator::RESHAPE )
    {
        tensors.push_back(tflite::CreateTensorDirect(
            builder, &parameterShape, tflite::TensorType::INT32, 1, "shape"));
        viewInputs.push_back(4);
        viewOptionsType = tflite::BuiltinOptions::ReshapeOptions;
        viewOptions = tflite::CreateReshapeOptionsDirect(builder, &parameterData).Union();
    }
    else if ( viewOperator == tflite::BuiltinOperator::SQUEEZE )
    {
        const std::vector<int32_t> squeezeDimensions = {1};
        viewOptionsType = tflite::BuiltinOptions::SqueezeOptions;
        viewOptions = tflite::CreateSqueezeOptionsDirect(builder, &squeezeDimensions).Union();
    }
    else
    {
        tensors.push_back(tflite::CreateTensorDirect(
            builder, &parameterShape, tflite::TensorType::INT32, 1, "axis"));
        viewInputs.push_back(4);
        viewOptionsType = tflite::BuiltinOptions::ExpandDimsOptions;
        viewOptions = tflite::CreateExpandDimsOptions(builder).Union();
    }
    const auto addOptions = tflite::CreateAddOptions(
        builder, tflite::ActivationFunctionType::NONE, false);
    const std::vector<int32_t> viewOutputs = {2};
    const std::vector<int32_t> addInputs = {2, 1};
    const std::vector<int32_t> addOutputs = {3};
    std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 0, &viewInputs, &viewOutputs,
            viewOptionsType, viewOptions),
    };
    if ( followedByAdd )
    {
        operations.push_back(tflite::CreateOperatorDirect(builder, 1, &addInputs, &addOutputs,
            tflite::BuiltinOptions::AddOptions, addOptions.Union()));
    }
    const std::vector<int32_t> graphInputs = followedByAdd ?
        std::vector<int32_t>{0, 1} : std::vector<int32_t>{0};
    const std::vector<int32_t> graphOutputs = followedByAdd ?
        std::vector<int32_t>{3} : std::vector<int32_t>{2};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(
            builder, &tensors, &graphInputs, &graphOutputs, &operations, "main"),
    };
    std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(
            builder, int8_t(viewOperator), nullptr, 1, viewOperator),
    };
    if ( followedByAdd )
    {
        operatorCodes.push_back(tflite::CreateOperatorCodeDirect(builder,
            int8_t(tflite::BuiltinOperator::ADD), nullptr, 1, tflite::BuiltinOperator::ADD));
    }
    const auto model = tflite::CreateModelDirect(
        builder, 3, &operatorCodes, &subgraphs, "Neural-AI view Add test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

flatbuffers::DetachedBuffer BuildK3ConvModel(int height, int width, int depthK, int depthN, int stride,
    int64_t inputZeroPoint = 0, tflite::Padding padding = tflite::Padding::SAME,
    float inputScaleValue = 1.0f, float outputScaleValue = 1.0f)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<float> scale = {1.0f};
    const std::vector<float> inputScale = {inputScaleValue};
    const std::vector<float> biasScale = {inputScaleValue};
    const std::vector<float> outputScale = {outputScaleValue};
    const std::vector<int64_t> zeroPoint = {0};
    const std::vector<int64_t> inputZeroPoints = {inputZeroPoint};
    const auto inputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &inputScale, &inputZeroPoints);
    const auto weightQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto biasQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &biasScale, &zeroPoint);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &outputScale, &zeroPoint);
    std::vector<uint8_t> weightData(size_t(depthK) * depthN * 9, 1);
    std::vector<int32_t> bias(depthN, 0);
    std::vector<uint8_t> biasData(bias.size() * sizeof(int32_t));
    std::memcpy(biasData.data(), bias.data(), biasData.size());
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
        tflite::CreateBufferDirect(builder, &weightData),
        tflite::CreateBufferDirect(builder, &biasData),
    };
    const int outputHeight = padding == tflite::Padding::SAME ?
        (height + stride - 1) / stride : (height - 3) / stride + 1;
    const int outputWidth = padding == tflite::Padding::SAME ?
        (width + stride - 1) / stride : (width - 3) / stride + 1;
    const std::vector<int32_t> inputShape = {1, height, width, depthK};
    const std::vector<int32_t> weightShape = {depthN, 3, 3, depthK};
    const std::vector<int32_t> biasShape = {depthN};
    const std::vector<int32_t> outputShape = {1, outputHeight, outputWidth, depthN};
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(builder, &inputShape, tflite::TensorType::INT8, 0, "input", inputQuant),
        tflite::CreateTensorDirect(builder, &weightShape, tflite::TensorType::INT8, 1, "weights", weightQuant),
        tflite::CreateTensorDirect(builder, &biasShape, tflite::TensorType::INT32, 2, "bias", biasQuant),
        tflite::CreateTensorDirect(builder, &outputShape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };
    const auto options = tflite::CreateConv2DOptions(
        builder, padding, stride, stride, tflite::ActivationFunctionType::NONE,
        1, 1, tflite::TensorType::INT32);
    const std::vector<int32_t> opInputs = {0, 1, 2};
    const std::vector<int32_t> opOutputs = {3};
    const std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 0, &opInputs, &opOutputs,
            tflite::BuiltinOptions::Conv2DOptions, options.Union()),
    };
    const std::vector<int32_t> graphInputs = {0};
    const std::vector<int32_t> graphOutputs = {3};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(builder, &tensors, &graphInputs, &graphOutputs,
            &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::CONV_2D),
            nullptr, 1, tflite::BuiltinOperator::CONV_2D),
    };
    const auto model = tflite::CreateModelDirect(builder, 3, &operatorCodes, &subgraphs,
        "Neural-AI K3 Conv test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

flatbuffers::DetachedBuffer BuildC32SliceConvModel(
    int height, int width, int inputDepth, int sliceBegin, int sliceDepth, int outputDepth,
    int consumerKernel = 1)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<float> scale = {1.0f};
    const std::vector<int64_t> zeroPoint = {0};
    const auto inputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto weightQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto biasQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);

    auto Int32Bytes = [](const std::vector<int32_t> &values)
    {
        std::vector<uint8_t> bytes(values.size() * sizeof(int32_t));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        return bytes;
    };
    const std::vector<int32_t> begin = {0, 0, 0, sliceBegin};
    const std::vector<int32_t> end = {0, 0, 0, sliceBegin + sliceDepth};
    const std::vector<int32_t> strides = {1, 1, 1, 1};
    const auto beginData = Int32Bytes(begin);
    const auto endData = Int32Bytes(end);
    const auto stridesData = Int32Bytes(strides);
    std::vector<uint8_t> producerWeightData(size_t(inputDepth) * 32, 1);
    std::vector<int32_t> producerBias(inputDepth, 0);
    const auto producerBiasData = Int32Bytes(producerBias);
    std::vector<uint8_t> consumerWeightData(
        size_t(outputDepth) * sliceDepth * consumerKernel * consumerKernel, 1);
    std::vector<int32_t> consumerBias(outputDepth, 0);
    const auto consumerBiasData = Int32Bytes(consumerBias);
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
        tflite::CreateBufferDirect(builder, &beginData),
        tflite::CreateBufferDirect(builder, &endData),
        tflite::CreateBufferDirect(builder, &stridesData),
        tflite::CreateBufferDirect(builder, &producerWeightData),
        tflite::CreateBufferDirect(builder, &producerBiasData),
        tflite::CreateBufferDirect(builder, &consumerWeightData),
        tflite::CreateBufferDirect(builder, &consumerBiasData),
    };

    const std::vector<int32_t> inputShape = {1, height, width, 32};
    const std::vector<int32_t> paramsShape = {4};
    const std::vector<int32_t> producerShape = {1, height, width, inputDepth};
    const std::vector<int32_t> producerWeightShape = {inputDepth, 1, 1, 32};
    const std::vector<int32_t> producerBiasShape = {inputDepth};
    const std::vector<int32_t> sliceShape = {1, height, width, sliceDepth};
    const std::vector<int32_t> consumerWeightShape = {
        outputDepth, consumerKernel, consumerKernel, sliceDepth};
    const std::vector<int32_t> consumerBiasShape = {outputDepth};
    const std::vector<int32_t> outputShape = {1, height, width, outputDepth};
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(builder, &inputShape, tflite::TensorType::INT8, 0, "input", inputQuant),
        tflite::CreateTensorDirect(builder, &paramsShape, tflite::TensorType::INT32, 1, "begin"),
        tflite::CreateTensorDirect(builder, &paramsShape, tflite::TensorType::INT32, 2, "end"),
        tflite::CreateTensorDirect(builder, &paramsShape, tflite::TensorType::INT32, 3, "strides"),
        tflite::CreateTensorDirect(builder, &producerShape, tflite::TensorType::INT8, 0, "producer", inputQuant),
        tflite::CreateTensorDirect(builder, &producerWeightShape, tflite::TensorType::INT8, 4,
            "producer_weights", weightQuant),
        tflite::CreateTensorDirect(builder, &producerBiasShape, tflite::TensorType::INT32, 5,
            "producer_bias", biasQuant),
        tflite::CreateTensorDirect(builder, &sliceShape, tflite::TensorType::INT8, 0, "slice", inputQuant),
        tflite::CreateTensorDirect(builder, &consumerWeightShape, tflite::TensorType::INT8, 6,
            "consumer_weights", weightQuant),
        tflite::CreateTensorDirect(builder, &consumerBiasShape, tflite::TensorType::INT32, 7,
            "consumer_bias", biasQuant),
        tflite::CreateTensorDirect(builder, &outputShape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };

    const auto sliceOptions = tflite::CreateStridedSliceOptions(builder, 7, 7, 0, 0, 0);
    const auto producerOptions = tflite::CreateConv2DOptions(builder, tflite::Padding::VALID, 1, 1,
        tflite::ActivationFunctionType::NONE, 1, 1, tflite::TensorType::INT32);
    const auto consumerOptions = tflite::CreateConv2DOptions(builder,
        consumerKernel == 1 ? tflite::Padding::VALID : tflite::Padding::SAME, 1, 1,
        tflite::ActivationFunctionType::NONE, 1, 1, tflite::TensorType::INT32);
    const std::vector<int32_t> producerInputs = {0, 5, 6};
    const std::vector<int32_t> producerOutputs = {4};
    const std::vector<int32_t> sliceInputs = {4, 1, 2, 3};
    const std::vector<int32_t> sliceOutputs = {7};
    const std::vector<int32_t> consumerInputs = {7, 8, 9};
    const std::vector<int32_t> consumerOutputs = {10};
    const std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 1, &producerInputs, &producerOutputs,
            tflite::BuiltinOptions::Conv2DOptions, producerOptions.Union()),
        tflite::CreateOperatorDirect(builder, 0, &sliceInputs, &sliceOutputs,
            tflite::BuiltinOptions::StridedSliceOptions, sliceOptions.Union()),
        tflite::CreateOperatorDirect(builder, 1, &consumerInputs, &consumerOutputs,
            tflite::BuiltinOptions::Conv2DOptions, consumerOptions.Union()),
    };
    const std::vector<int32_t> graphInputs = {0};
    const std::vector<int32_t> graphOutputs = {10};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(
            builder, &tensors, &graphInputs, &graphOutputs, &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::STRIDED_SLICE),
            nullptr, 2, tflite::BuiltinOperator::STRIDED_SLICE),
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::CONV_2D),
            nullptr, 3, tflite::BuiltinOperator::CONV_2D),
    };
    const auto model = tflite::CreateModelDirect(builder, 3, &operatorCodes, &subgraphs,
        "Neural-AI C32 slice pointwise test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

flatbuffers::DetachedBuffer BuildDepthwiseConvModel(int height, int width, int channels, int stride,
    int64_t inputZeroPoint = 0)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<float> scale = {1.0f};
    const std::vector<int64_t> zeroPoint = {0};
    const std::vector<int64_t> inputZeroPoints = {inputZeroPoint};
    const auto inputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &inputZeroPoints);
    const auto weightQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto biasQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    std::vector<uint8_t> weightData(size_t(channels) * 9, 1);
    std::vector<int32_t> bias(channels, 0);
    std::vector<uint8_t> biasData(bias.size() * sizeof(int32_t));
    std::memcpy(biasData.data(), bias.data(), biasData.size());
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
        tflite::CreateBufferDirect(builder, &weightData),
        tflite::CreateBufferDirect(builder, &biasData),
    };
    const int outputHeight = (height + stride - 1) / stride;
    const int outputWidth = (width + stride - 1) / stride;
    const std::vector<int32_t> inputShape = {1, height, width, channels};
    const std::vector<int32_t> weightShape = {1, 3, 3, channels};
    const std::vector<int32_t> biasShape = {channels};
    const std::vector<int32_t> outputShape = {1, outputHeight, outputWidth, channels};
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(builder, &inputShape, tflite::TensorType::INT8, 0, "input", inputQuant),
        tflite::CreateTensorDirect(builder, &weightShape, tflite::TensorType::INT8, 1, "weights", weightQuant),
        tflite::CreateTensorDirect(builder, &biasShape, tflite::TensorType::INT32, 2, "bias", biasQuant),
        tflite::CreateTensorDirect(builder, &outputShape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };
    const auto options = tflite::CreateDepthwiseConv2DOptions(
        builder, tflite::Padding::SAME, stride, stride, 1,
        tflite::ActivationFunctionType::NONE, 1, 1);
    const std::vector<int32_t> opInputs = {0, 1, 2};
    const std::vector<int32_t> opOutputs = {3};
    const std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 0, &opInputs, &opOutputs,
            tflite::BuiltinOptions::DepthwiseConv2DOptions, options.Union()),
    };
    const std::vector<int32_t> graphInputs = {0};
    const std::vector<int32_t> graphOutputs = {3};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(builder, &tensors, &graphInputs, &graphOutputs,
            &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::DEPTHWISE_CONV_2D),
            nullptr, 1, tflite::BuiltinOperator::DEPTHWISE_CONV_2D),
    };
    const auto model = tflite::CreateModelDirect(builder, 3, &operatorCodes, &subgraphs,
        "Neural-AI depthwise Conv test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

flatbuffers::DetachedBuffer BuildMicroMobileNetModel(int stageCount = 13)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<float> scale = {1.0f};
    const std::vector<int64_t> zeroPoint = {0};
    const auto int8Quant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto int32Quant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
    };
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
    std::vector<flatbuffers::Offset<tflite::Operator>> operations;

    const auto addActivation = [&](const std::vector<int32_t> &shape)
    {
        const int index = int(tensors.size());
        tensors.push_back(tflite::CreateTensorDirect(
            builder, &shape, tflite::TensorType::INT8, 0, "activation", int8Quant));
        return index;
    };
    const auto addWeights = [&](const std::vector<int32_t> &shape, const std::vector<uint8_t> &data)
    {
        const int buffer = int(buffers.size());
        buffers.push_back(tflite::CreateBufferDirect(builder, &data));
        const int index = int(tensors.size());
        tensors.push_back(tflite::CreateTensorDirect(
            builder, &shape, tflite::TensorType::INT8, buffer, "weights", int8Quant));
        return index;
    };
    const auto weightByte = [](int value) { return uint8_t(value & 0xFF); };
    const auto stemWeight = [&](int kh, int kw, int ic, int oc)
    {
        return weightByte(((kh * 2 + kw * 3 + ic * 5 + oc * 7 + 2) % 3) - 1);
    };
    const auto pointwiseWeight = [&](int tag, int ic, int oc)
    {
        if ( tag == 5 ) return weightByte(ic == oc ? 1 : 0);
        return weightByte(((tag * 5 + ic * 3 + oc * 7 + 4) % 3) - 1);
    };
    const auto convWeight = [&](int tag, int kh, int kw, int ic, int oc)
    {
        return weightByte(((tag * 11 + kh * 3 + kw * 5 + ic * 7 + oc * 2 + 6) % 3) - 1);
    };
    const auto depthwiseWeight = [&](int tag, int kh, int kw, int channel)
    {
        return weightByte(((tag * 3 + kh * 5 + kw * 7 + channel * 2 + 1) % 3) - 1);
    };
    const auto addBias = [&](int channels)
    {
        std::vector<int32_t> values(channels, 0);
        std::vector<uint8_t> data(values.size() * sizeof(int32_t));
        std::memcpy(data.data(), values.data(), data.size());
        const int buffer = int(buffers.size());
        buffers.push_back(tflite::CreateBufferDirect(builder, &data));
        const int index = int(tensors.size());
        const std::vector<int32_t> shape = {channels};
        tensors.push_back(tflite::CreateTensorDirect(
            builder, &shape, tflite::TensorType::INT32, buffer, "bias", int32Quant));
        return index;
    };
    const auto addConv = [&](int input, int height, int width, int depthK, int depthN,
                             int kernel, int stride, int tag,
                             tflite::ActivationFunctionType activation)
    {
        const int outputHeight = (height + stride - 1) / stride;
        const int outputWidth = (width + stride - 1) / stride;
        const std::vector<int32_t> weightShape = {depthN, kernel, kernel, depthK};
        const std::vector<int32_t> outputShape = {1, outputHeight, outputWidth, depthN};
        std::vector<uint8_t> weightData;
        weightData.reserve(kernel * kernel * depthK * depthN);
        for ( int oc = 0; oc < depthN; ++oc )
        {
            for ( int kh = 0; kh < kernel; ++kh )
            {
                for ( int kw = 0; kw < kernel; ++kw )
                {
                    for ( int ic = 0; ic < depthK; ++ic )
                    {
                        weightData.push_back(
                            depthK == 3 && kernel == 3 ? stemWeight(kh, kw, ic, oc) :
                            kernel == 1 ? pointwiseWeight(tag, ic, oc) :
                                          convWeight(tag, kh, kw, ic, oc));
                    }
                }
            }
        }
        const int weights = addWeights(weightShape, weightData);
        const int bias = addBias(depthN);
        const int output = addActivation(outputShape);
        const auto options = tflite::CreateConv2DOptions(
            builder, tflite::Padding::SAME, stride, stride, activation, 1, 1,
            tflite::TensorType::INT32);
        const std::vector<int32_t> inputs = {input, weights, bias};
        const std::vector<int32_t> outputs = {output};
        operations.push_back(tflite::CreateOperatorDirect(
            builder, 0, &inputs, &outputs, tflite::BuiltinOptions::Conv2DOptions,
            options.Union()));
        return output;
    };
    const auto addDepthwise = [&](int input, int height, int width, int channels,
                                  int stride, int tag,
                                  tflite::ActivationFunctionType activation)
    {
        const int outputHeight = (height + stride - 1) / stride;
        const int outputWidth = (width + stride - 1) / stride;
        const std::vector<int32_t> weightShape = {1, 3, 3, channels};
        const std::vector<int32_t> outputShape = {1, outputHeight, outputWidth, channels};
        std::vector<uint8_t> weightData;
        weightData.reserve(9 * channels);
        for ( int kh = 0; kh < 3; ++kh )
        {
            for ( int kw = 0; kw < 3; ++kw )
            {
                for ( int channel = 0; channel < channels; ++channel )
                    weightData.push_back(depthwiseWeight(tag, kh, kw, channel));
            }
        }
        const int weights = addWeights(weightShape, weightData);
        const int bias = addBias(channels);
        const int output = addActivation(outputShape);
        const auto options = tflite::CreateDepthwiseConv2DOptions(
            builder, tflite::Padding::SAME, stride, stride, 1, activation, 1, 1);
        const std::vector<int32_t> inputs = {input, weights, bias};
        const std::vector<int32_t> outputs = {output};
        operations.push_back(tflite::CreateOperatorDirect(
            builder, 1, &inputs, &outputs,
            tflite::BuiltinOptions::DepthwiseConv2DOptions, options.Union()));
        return output;
    };
    const auto addResidual = [&](int lhs, int rhs, int height, int width, int channels)
    {
        const std::vector<int32_t> outputShape = {1, height, width, channels};
        const int output = addActivation(outputShape);
        const auto options = tflite::CreateAddOptions(
            builder, tflite::ActivationFunctionType::NONE, false);
        const std::vector<int32_t> inputs = {lhs, rhs};
        const std::vector<int32_t> outputs = {output};
        operations.push_back(tflite::CreateOperatorDirect(
            builder, 2, &inputs, &outputs, tflite::BuiltinOptions::AddOptions,
            options.Union()));
        return output;
    };

    const int input = addActivation({1, 96, 96, 3});
    int output = input;
    int stem = -1;
    int pw1 = -1;
    if ( stageCount >= 1 )
        output = stem = addConv(output, 96, 96, 3, 32, 3, 2, 0,
            tflite::ActivationFunctionType::RELU6);
    if ( stageCount >= 2 )
        output = addDepthwise(output, 48, 48, 32, 1, 0,
            tflite::ActivationFunctionType::RELU6);
    if ( stageCount >= 3 )
        output = addConv(output, 48, 48, 32, 32, 1, 1, 0,
            tflite::ActivationFunctionType::NONE);
    if ( stageCount >= 4 ) output = addResidual(output, stem, 48, 48, 32);
    if ( stageCount >= 5 )
        output = addDepthwise(output, 48, 48, 32, 2, 1,
            tflite::ActivationFunctionType::NONE);
    if ( stageCount >= 6 )
        output = pw1 = addConv(output, 24, 24, 32, 64, 1, 1, 1,
            tflite::ActivationFunctionType::RELU6);
    if ( stageCount >= 7 )
        output = addConv(output, 24, 24, 64, 128, 1, 1, 2,
            tflite::ActivationFunctionType::NONE);
    if ( stageCount >= 8 )
        output = addDepthwise(output, 24, 24, 128, 1, 2,
            tflite::ActivationFunctionType::NONE);
    if ( stageCount >= 9 )
        output = addConv(output, 24, 24, 128, 64, 1, 1, 3,
            tflite::ActivationFunctionType::NONE);
    if ( stageCount >= 10 ) output = addResidual(output, pw1, 24, 24, 64);
    if ( stageCount >= 11 )
        output = addConv(output, 24, 24, 64, 64, 3, 1, 4,
            tflite::ActivationFunctionType::NONE);
    if ( stageCount >= 12 )
    {
        const int pooled = addActivation({1, 1, 1, 64});
        const auto poolOptions = tflite::CreatePool2DOptions(
            builder, tflite::Padding::VALID, 1, 1, 24, 24,
            tflite::ActivationFunctionType::NONE);
        const std::vector<int32_t> poolInputs = {output};
        const std::vector<int32_t> poolOutputs = {pooled};
        operations.push_back(tflite::CreateOperatorDirect(
            builder, 3, &poolInputs, &poolOutputs, tflite::BuiltinOptions::Pool2DOptions,
            poolOptions.Union()));
        output = pooled;
    }
    if ( stageCount >= 13 )
        output = addConv(output, 1, 1, 64, 32, 1, 1, 5,
            tflite::ActivationFunctionType::NONE);

    const std::vector<int32_t> graphInputs = {input};
    const std::vector<int32_t> graphOutputs = {output};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(
            builder, &tensors, &graphInputs, &graphOutputs, &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::CONV_2D),
            nullptr, 1, tflite::BuiltinOperator::CONV_2D),
        tflite::CreateOperatorCodeDirect(
            builder, int8_t(tflite::BuiltinOperator::DEPTHWISE_CONV_2D),
            nullptr, 1, tflite::BuiltinOperator::DEPTHWISE_CONV_2D),
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::ADD),
            nullptr, 1, tflite::BuiltinOperator::ADD),
        tflite::CreateOperatorCodeDirect(
            builder, int8_t(tflite::BuiltinOperator::AVERAGE_POOL_2D),
            nullptr, 1, tflite::BuiltinOperator::AVERAGE_POOL_2D),
    };
    const auto model = tflite::CreateModelDirect(builder, 3, &operatorCodes, &subgraphs,
        "Neural-AI Micro-MobileNet compiler test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
}

}  // namespace

TEST_CASE("Neural-AI architecture factory")
{
    regor_context_t context = 0;
    REQUIRE(regor_create(&context, REGOR_ARCH_NEURALAI) == 1);
    REQUIRE(context != 0);
    regor_destroy(context);
}

TEST_CASE("Neural-AI architecture exposes fixed hardware configuration")
{
    ArchNeuralAI arch;
    std::string error;

    REQUIRE(arch.CheckConfiguration(error));
    REQUIRE(error.empty());
    REQUIRE(arch.AllocationQuantum() == 32);
    REQUIRE(arch.TensorAlignment(TensorUsage::IFM, TensorFormat::Row32) == 32);
    REQUIRE(arch.TensorAlignment(TensorUsage::IFM, TensorFormat::C32Blocked) == 32);
    REQUIRE(arch.TensorAlignment(TensorUsage::IFM, TensorFormat::WeightsEncoded) == 32);
    REQUIRE(arch.TensorAlignment(TensorUsage::IFM, TensorFormat::NHWC) == 1);
    REQUIRE(arch.TensorAlignment(TensorUsage::IFM, TensorFormat::CompactNHWC) == 1);
    REQUIRE(arch.FeatureMapMemory().memory->Name() == "tcdm");
    REQUIRE(arch.FeatureMapMemory().memory->SizeBytes() == ArchNeuralAI::AllocatableTCDMBytes);
    REQUIRE(arch.ReadonlyMemory().memory->Name() == "model");
    REQUIRE(arch.ModelBindingFormat(TensorUsage::IFM) == TensorFormat::NHWC);
    REQUIRE(arch.DefaultInternalTensorFormat(TensorUsage::IFM, false) == TensorFormat::Row32);
    REQUIRE(arch.DefaultInternalTensorFormat(TensorUsage::OFM, true) == TensorFormat::NHWC);
    REQUIRE(arch.IdealBufferingFormat() == TensorFormat::C32Blocked);
    REQUIRE(arch.CanSubdivide(OpType::Conv2D, TransposeType::None, ReverseType::None) == AxisMask::AxisY);
    REQUIRE(arch.CanSubdivide(OpType::DepthwiseConv2D, TransposeType::None, ReverseType::None) == AxisMask::AxisY);
    REQUIRE(arch.CanSubdivide(OpType::LUT, TransposeType::None, ReverseType::None) == AxisMask::AxisY);
    REQUIRE(arch.CanSubdivide(OpType::Concat, TransposeType::None, ReverseType::None) == AxisMask::AxisY);
    REQUIRE(arch.CanSubdivide(OpType::LUT, TransposeType::NCHW, ReverseType::None) == AxisMask::None);

    std::string target;
    arch.Call([&target](const std::string &name) { target = name; });
    REQUIRE(target == REGOR_ARCH_NEURALAI);
}

TEST_CASE("Neural-AI ROW32 storage and alignment")
{
    ArchNeuralAI arch;
    const Shape logical(1, 2, 3, 33);

    REQUIRE(arch.StorageShape(logical, TensorFormat::Row32) == Shape(1, 2, 3, 64));
    REQUIRE(arch.StorageBytes(logical, TensorFormat::Row32, DataType::Int8) == 384);
    REQUIRE(arch.StorageBytes(Shape(31), TensorFormat::Row32, DataType::Int8) == 32);
    REQUIRE(arch.StorageBytes(Shape(33), TensorFormat::Row32, DataType::Int32) == 256);
    REQUIRE(arch.TensorStrides(logical, TensorFormat::Row32, DataType::Int8) == Shape(384, 192, 64, 1));
    REQUIRE(arch.CanAliasDepthOffset(TensorFormat::Row32, 32));
    REQUIRE_FALSE(arch.CanAliasDepthOffset(TensorFormat::Row32, 1));
}

TEST_CASE("Neural-AI compact NHWC preserves unpadded channel storage")
{
    ArchNeuralAI arch;
    const Shape logical(1, 2, 3, 16);

    REQUIRE(arch.StorageShape(logical, TensorFormat::CompactNHWC) == logical);
    REQUIRE(arch.StorageBytes(logical, TensorFormat::CompactNHWC, DataType::Int8) == 96);
    REQUIRE(arch.TensorStrides(logical, TensorFormat::CompactNHWC, DataType::Int8) ==
        Shape(96, 48, 16, 1));
}

TEST_CASE("Neural-AI selected CSP gather reuses one rolling C48 row")
{
    ArchNeuralAI arch;
    for ( const int width : {11, 80, 96} )
    {
        const Shape row(1, 1, width, 48);
        REQUIRE(arch.RollingBufferShape(row, row, TensorFormat::C32Blocked) ==
            Shape(1, 1, width, 64));
        REQUIRE(arch.StorageBytes(row, TensorFormat::C32Blocked, DataType::Int8) ==
            width * 64);
    }
}

TEST_CASE("Neural-AI architecture rejects conflicting RTL limits")
{
    ArchNeuralAI arch;
    const std::string config = "[architecture]\nclusters=2\n";
    IniReader reader(config.c_str(), config.size());
    std::string section;

    REQUIRE(reader.Begin(section));
    REQUIRE(section == "architecture");
    REQUIRE(arch.ParseSection(section, &reader) == IniParseResult::Error);
}

TEST_CASE("Neural-AI architecture accepts the fixed RTL limits")
{
    ArchNeuralAI arch;
    const std::string config =
        "[architecture]\n"
        "clusters=1\n"
        "array_dimension=32\n"
        "dma_alignment=32\n"
        "tcdm_size=524288\n"
        "command_staging_size=4096\n"
        "linebuffer_max_kernel=5\n"
        "linebuffer_max_stride=2\n"
        "linebuffer_max_input_width=640\n"
        "requant_shift_max=31\n";
    IniReader reader(config.c_str(), config.size());
    std::string section;

    REQUIRE(reader.Begin(section));
    REQUIRE(section == "architecture");
    REQUIRE(arch.ParseSection(section, &reader) == IniParseResult::Done);
}

TEST_CASE("Neural-AI constraints accept INT8 matrix and pointwise operations")
{
    ArchNeuralAI arch;
    auto *constraints = arch.Constraints();

    REQUIRE(constraints->OperatorQuery(OpType::FullyConnected) == QueryResult::NativeConstrained);
    REQUIRE(constraints->OperatorQuery(OpType::MatMul) == QueryResult::NativeConstrained);
    REQUIRE(constraints->OperatorQuery(OpType::Conv2D) == QueryResult::NativeConstrained);

    ArchOperatorQuery query;
    query.ifm[0].type = DataType::Int8;
    query.ifm[0].shape = Shape(1, 31);
    query.weights.type = DataType::Int8;
    query.weights.shape = Shape(31, 33);
    query.ofm.type = DataType::Int8;
    query.ofm.shape = Shape(1, 33);
    query.weightFormat = WeightFormat::Default;
    REQUIRE(constraints->OperatorQuery(OpType::MatMul, &query) == QueryResult::Native);

    query.ofm.type = DataType::UInt8;
    REQUIRE(constraints->OperatorQuery(OpType::MatMul, &query) == QueryResult::Unsupported);
    query.ofm.type = DataType::Int8;
    query.weightFormat = WeightFormat::None;
    REQUIRE(constraints->OperatorQuery(OpType::MatMul, &query) == QueryResult::Unsupported);
    query.weightFormat = WeightFormat::Default;
    query.ifm[0].shape = Shape(2, 1, 31);
    REQUIRE(constraints->OperatorQuery(OpType::MatMul, &query) == QueryResult::Unsupported);

    ArchOperatorQuery pointwise;
    pointwise.ifm[0].type = DataType::Int8;
    pointwise.ifm[0].shape = Shape(1, 2, 3, 33);
    pointwise.weights.type = DataType::Int8;
    pointwise.weights.shape = Shape(34, 1, 1, 33);
    pointwise.ofm.type = DataType::Int8;
    pointwise.ofm.shape = Shape(1, 2, 3, 34);
    pointwise.weightFormat = WeightFormat::Default;
    pointwise.kernel = &Kernel::UnitKernel();
    REQUIRE(constraints->OperatorQuery(OpType::Conv2D, &pointwise) == QueryResult::Native);
    const Kernel nonPointwiseKernel({3, 3}, {1, 1}, {1, 1});
    pointwise.kernel = &nonPointwiseKernel;
    REQUIRE(constraints->OperatorQuery(OpType::Conv2D, &pointwise) == QueryResult::Unsupported);
}

TEST_CASE("Neural-AI constraints admit only 16-lane generic K3 channel tails")
{
    const Kernel kernel({3, 3}, {2, 2}, {1, 1});
    const auto ic16 = NeuralAIConstraints::Classify(OpType::Conv2D,
        Shape(1, 18, 18, 16), Shape(32, 3, 3, 16), Shape(1, 8, 8, 32), &kernel);
    REQUIRE(ic16.mode == NeuralAIOpMode::Conv2DLinebufC32S2Requant);
    REQUIRE(ic16.hasIcTail);
    REQUIRE_FALSE(ic16.hasOcTail);
    REQUIRE_FALSE(ic16.groupStationary);

    const auto ic48 = NeuralAIConstraints::Classify(OpType::Conv2D,
        Shape(1, 18, 18, 48), Shape(32, 3, 3, 48), Shape(1, 8, 8, 32), &kernel);
    REQUIRE(ic48.mode == NeuralAIOpMode::Conv2DLinebufC32S2Requant);
    REQUIRE(ic48.hasIcTail);
    REQUIRE_FALSE(ic48.hasOcTail);
    REQUIRE_FALSE(ic48.groupStationary);

    const auto ic80 = NeuralAIConstraints::Classify(OpType::Conv2D,
        Shape(1, 18, 18, 80), Shape(32, 3, 3, 80), Shape(1, 8, 8, 32), &kernel);
    REQUIRE(ic80.mode == NeuralAIOpMode::Conv2DLinebufC32S2Requant);
    REQUIRE(ic80.hasIcTail);
    REQUIRE_FALSE(ic80.hasOcTail);
    REQUIRE_FALSE(ic80.groupStationary);

    const auto oc48 = NeuralAIConstraints::Classify(OpType::Conv2D,
        Shape(1, 18, 18, 32), Shape(48, 3, 3, 32), Shape(1, 8, 8, 48), &kernel);
    REQUIRE(oc48.mode == NeuralAIOpMode::Conv2DLinebufC32S2Requant);
    REQUIRE_FALSE(oc48.hasIcTail);
    REQUIRE(oc48.hasOcTail);

    const auto unsupported = NeuralAIConstraints::Classify(OpType::Conv2D,
        Shape(1, 18, 18, 24), Shape(32, 3, 3, 24), Shape(1, 8, 8, 32), &kernel);
    REQUIRE_FALSE(unsupported);
    REQUIRE(unsupported.diagnostic ==
        "generic C32 Conv supports full input groups or 16-lane tails only");
}

TEST_CASE("Neural-AI constraints accept shape-preserving memory copies")
{
    ArchNeuralAI arch;
    auto *constraints = arch.Constraints();

    REQUIRE(constraints->OperatorQuery(OpType::MemoryCopy) == QueryResult::NativeConstrained);

    ArchOperatorQuery query;
    query.ifm[0].type = DataType::Int8;
    query.ifm[0].shape = Shape(1, 1, 1, 33);
    query.ofm.type = DataType::Int8;
    query.ofm.shape = query.ifm[0].shape;
    REQUIRE(constraints->OperatorQuery(OpType::MemoryCopy, &query) == QueryResult::Native);

    query.ofm.shape = Shape(1, 1, 1, 34);
    REQUIRE(constraints->OperatorQuery(OpType::MemoryCopy, &query) == QueryResult::Unsupported);
}

TEST_CASE("Neural-AI constraints keep selected CSP Concat tails compact")
{
    ArchNeuralAI arch;
    auto *constraints = arch.Constraints();
    ArchOperatorQuery query;
    query.ifm[0].type = DataType::Int8;
    for ( const Shape &inputShape : {Shape(1, 7, 11, 16), Shape(1, 80, 80, 16),
             Shape(1, 160, 96, 16)} )
    {
        query.ifm[0].shape = inputShape;
        query.ifm[1] = query.ifm[0];
        query.ofm.type = DataType::Int8;
        query.ofm.shape = inputShape.WithDepth(48);
        query.axis = -1;

        ArchRequirements requirements;
        REQUIRE(constraints->OperatorQuery(OpType::Concat, &query, &requirements) ==
            QueryResult::NativeHasReq);
        const std::array<std::pair<TensorUsage, TensorFormat>, 4> expected{{
            {TensorUsage::IFM0, TensorFormat::CompactNHWC},
            {TensorUsage::IFM1, TensorFormat::CompactNHWC},
            {TensorUsage::IFM2, TensorFormat::CompactNHWC},
            {TensorUsage::OFM, TensorFormat::C32Blocked},
        }};
        const ArchTensorRequirement *tensor = &requirements.tensor;
        for ( const auto &[usage, format] : expected )
        {
            REQUIRE(tensor != nullptr);
            REQUIRE(tensor->usage == usage);
            REQUIRE(tensor->format == format);
            tensor = tensor->next;
        }
        REQUIRE(tensor == nullptr);
    }

    query.ofm.shape = query.ifm[0].shape.WithDepth(64);
    REQUIRE(constraints->OperatorQuery(OpType::Concat, &query) == QueryResult::Unsupported);
}

TEST_CASE("Neural-AI constraints admit only selected YOLO C144 head transposes")
{
    ArchNeuralAI arch;
    auto *constraints = arch.Constraints();
    Quantization quantization;
    quantization.scales = {QuantizedScale(0.265464634)};
    quantization.zeroPoints = {63};
    ArchOperatorQuery query;
    query.ifm[0].type = DataType::Int8;
    query.ifm[0].shape = Shape(1, 10, 10, 144);
    query.ifm[0].quantization = &quantization;
    query.ofm.type = DataType::Int8;
    query.ofm.shape = Shape(1, 144, 10, 10);
    query.ofm.quantization = &quantization;
    query.transposeMask = TransposeType::NCHW;

    ArchRequirements requirements;
    REQUIRE(constraints->OperatorQuery(OpType::Transpose, &query, &requirements) ==
        QueryResult::NativeHasReq);
    REQUIRE(requirements.req.Any(ArchRequirement::Tensor));
    REQUIRE(requirements.tensor.usage == TensorUsage::IFM0);
    REQUIRE(requirements.tensor.format == TensorFormat::C32Blocked);
    REQUIRE(requirements.tensor.next != nullptr);
    REQUIRE(requirements.tensor.next->usage == TensorUsage::OFM);
    REQUIRE(requirements.tensor.next->format == TensorFormat::NHWC);

    for ( int spatial : {20, 40} )
    {
        query.ifm[0].shape = Shape(1, spatial, spatial, 144);
        query.ofm.shape = Shape(1, 144, spatial, spatial);
        REQUIRE(constraints->OperatorQuery(OpType::Transpose, &query) == QueryResult::Native);
    }
    query.ifm[0].shape = Shape(1, 10, 10, 128);
    query.ofm.shape = Shape(1, 128, 10, 10);
    REQUIRE(constraints->OperatorQuery(OpType::Transpose, &query) == QueryResult::Unsupported);
    query.ifm[0].shape = Shape(1, 10, 10, 144);
    query.ofm.shape = Shape(1, 10, 144, 10);
    query.transposeMask = TransposeType::NHCW;
    REQUIRE(constraints->OperatorQuery(OpType::Transpose, &query) == QueryResult::Unsupported);
    query.ifm[0].shape = Shape(1, 4, 16, 2100);
    query.ofm.shape = Shape(1, 4, 2100, 16);
    REQUIRE(constraints->OperatorQuery(OpType::Transpose, &query) == QueryResult::Unsupported);
}

TEST_CASE("Neural-AI constraints reserve tiled scratch for fused YOLO DFL16")
{
    ArchNeuralAI arch;
    auto *constraints = arch.Constraints();
    ArchOperatorQuery query;
    query.ifm[0].type = DataType::Int8;
    query.ifm[0].shape = Shape(1, 1, 144, 2100);
    query.ofm.type = DataType::Int8;
    query.ofm.shape = Shape(1, 1, 4, 2100);

    ArchRequirements requirements;
    REQUIRE(constraints->OperatorQuery(OpType::Dfl, &query, &requirements) == QueryResult::NativeHasReq);
    REQUIRE(requirements.req.Any(ArchRequirement::Tensor));
    REQUIRE(requirements.tensor.usage == TensorUsage::IFM0);
    REQUIRE(requirements.tensor.format == TensorFormat::NHWC);
    REQUIRE(requirements.tensor.next != nullptr);
    REQUIRE(requirements.tensor.next->usage == TensorUsage::OFM);
    REQUIRE(requirements.tensor.next->format == TensorFormat::NHWC);
    REQUIRE(requirements.tensor.next->next != nullptr);
    REQUIRE(requirements.tensor.next->next->usage == TensorUsage::Scratch);
    REQUIRE(requirements.tensor.next->next->format == TensorFormat::Row32);
    REQUIRE(requirements.tensor.next->next->shape == Shape(1, 1, 34, 32));

    ArchitectureConfigQuery configQuery{};
    configQuery.ifmBits = 8;
    configQuery.ofmBits = 8;
    configQuery.transpose = TransposeType::None;
    configQuery.reverse = ReverseType::None;
    auto config = arch.GetOpConfig(OpType::Dfl, configQuery);
    REQUIRE(config != nullptr);
    REQUIRE(static_cast<NeuralAIOpConfig *>(config.get())->Mode() == NeuralAIOpMode::Dfl16);

    query.ifm[0].shape = Shape(1, 1, 64, 2100);
    REQUIRE(constraints->OperatorQuery(OpType::Dfl, &query) == QueryResult::Unsupported);
}

TEST_CASE("Neural-AI constraints substitute INT8 Sigmoid with an AFU LUT")
{
    ArchNeuralAI arch;
    auto *constraints = arch.Constraints();
    REQUIRE(constraints->SupportsSiluLUTFusion());
    ArchOperatorQuery query;
    query.ifm[0].type = DataType::Int8;
    query.ifm[0].shape = Shape(1, 2, 3, 33);
    query.ofm.type = DataType::Int8;
    query.ofm.shape = query.ifm[0].shape;
    ArchRequirements sigmoidRequirements;
    REQUIRE(constraints->OperatorQuery(OpType::Sigmoid, &query, &sigmoidRequirements) ==
        QueryResult::NativeHasReq);
    REQUIRE(sigmoidRequirements.req.Any(ArchRequirement::OpSubstitution));
    REQUIRE(sigmoidRequirements.substitution == OpType::LUT);
    ArchRequirements clippingRequirements;
    REQUIRE(constraints->OperatorQuery(OpType::Relu6, &query, &clippingRequirements) ==
        QueryResult::NativeHasReq);
    REQUIRE(clippingRequirements.req.Any(ArchRequirement::OpSubstitution));
    REQUIRE(clippingRequirements.substitution == OpType::LUT);

    ArchRequirements lutRequirements;
    REQUIRE(constraints->OperatorQuery(OpType::LUT, &query, &lutRequirements) ==
        QueryResult::NativeHasReq);
    REQUIRE(lutRequirements.req.Any(ArchRequirement::Tensor));
    REQUIRE(lutRequirements.tensor.usage == TensorUsage::IFM0);
    REQUIRE(lutRequirements.tensor.format == TensorFormat::C32Blocked);
    REQUIRE(lutRequirements.tensor.next != nullptr);
    REQUIRE(lutRequirements.tensor.next->usage == TensorUsage::OFM);
    REQUIRE(lutRequirements.tensor.next->format == TensorFormat::C32Blocked);

    query.ifm[0].shape = Shape(1, 80, 2100);
    query.ofm.shape = query.ifm[0].shape;
    ArchRequirements classHeadRequirements;
    REQUIRE(constraints->OperatorQuery(OpType::LUT, &query, &classHeadRequirements) ==
        QueryResult::NativeHasReq);
    REQUIRE(classHeadRequirements.tensor.format == TensorFormat::NHWC);
    REQUIRE(classHeadRequirements.tensor.next != nullptr);
    REQUIRE(classHeadRequirements.tensor.next->format == TensorFormat::NHWC);

    query.transposeMask = TransposeType::NCHW;
    REQUIRE(constraints->OperatorQuery(OpType::LUT, &query) == QueryResult::Unsupported);
    query.transposeMask = TransposeType::None;
    query.ofm.shape = Shape(1, 2, 3, 32);
    REQUIRE(constraints->OperatorQuery(OpType::Sigmoid, &query) == QueryResult::Unsupported);
    REQUIRE(constraints->OperatorQuery(OpType::LUT, &query) == QueryResult::Unsupported);
    query.ifm[0].shape = Shape(2, 2, 3, 33);
    query.ofm.shape = query.ifm[0].shape;
    REQUIRE(constraints->OperatorQuery(OpType::Sigmoid, &query) == QueryResult::Unsupported);
    REQUIRE(constraints->OperatorQuery(OpType::LUT, &query) == QueryResult::Unsupported);
}

TEST_CASE("Neural-AI constraints accept scalar INT8 Add quantization")
{
    ArchNeuralAI arch;
    auto *constraints = arch.Constraints();
    Quantization inputQuant;
    inputQuant.scales = {QuantizedScale(32768.0)};
    inputQuant.zeroPoints = {0};
    Quantization outputQuant;
    outputQuant.scales = {QuantizedScale(1.0 / 32768.0)};
    outputQuant.zeroPoints = {0};
    outputQuant.quantMin = {-128};
    outputQuant.quantMax = {127};

    ArchOperatorQuery query;
    for ( ArchFM &ifm : query.ifm )
    {
        ifm.type = DataType::Int8;
        ifm.shape = Shape(1, 2, 2, 32);
        ifm.quantization = &inputQuant;
    }
    query.ofm.type = DataType::Int8;
    query.ofm.shape = Shape(1, 2, 2, 32);
    query.ofm.quantization = &outputQuant;
    REQUIRE(constraints->OperatorQuery(OpType::Add, &query) == QueryResult::Native);

    inputQuant.zeroPoints = {1};
    REQUIRE(constraints->OperatorQuery(OpType::Add, &query) == QueryResult::Native);
    inputQuant.zeroPoints = {0};
    outputQuant.quantMin = {0};
    REQUIRE(constraints->OperatorQuery(OpType::Add, &query) == QueryResult::Native);

    inputQuant.scales = {QuantizedScale(1.0), QuantizedScale(0.5)};
    REQUIRE(constraints->OperatorQuery(OpType::Add, &query) == QueryResult::Unsupported);
}

TEST_CASE("Neural-AI constraints keep selected CSP Add compact")
{
    ArchNeuralAI arch;
    auto *constraints = arch.Constraints();
    Quantization quantization = Quantization::Unit();
    ArchOperatorQuery query;
    for ( const Shape &shape : {Shape(1, 7, 11, 16), Shape(1, 80, 80, 16),
             Shape(1, 160, 96, 16)} )
    {
        for ( ArchFM &ifm : query.ifm )
        {
            ifm.type = DataType::Int8;
            ifm.shape = shape;
            ifm.quantization = &quantization;
        }
        query.ofm.type = DataType::Int8;
        query.ofm.shape = shape;
        query.ofm.quantization = &quantization;

        ArchRequirements requirements;
        REQUIRE(constraints->OperatorQuery(OpType::Add, &query, &requirements) ==
            QueryResult::NativeHasReq);
        for ( const ArchTensorRequirement *tensor = &requirements.tensor; tensor != nullptr;
              tensor = tensor->next )
            REQUIRE(tensor->format == TensorFormat::CompactNHWC);
    }
}

TEST_CASE("Neural-AI admits only native-depth-preserving reshape-like views")
{
    auto checker = MakeSupportedOpsChecker(REGOR_ARCH_NEURALAI);
    for ( const OpType opType : {OpType::Reshape, OpType::Squeeze, OpType::ExpandDims} )
    {
        INFO("opType=" << OpTypeToString(opType));
        auto input = CreateTensor("input", Shape(1, 2, 2, 32), DataType::Int8);
        auto output = CreateTensor("output", Shape(1, 1, 4, 32), DataType::Int8);
        auto view = CreateOperation(
            opType, TensorUsage::IFM0, input, TensorUsage::OFM, output);
        view->Input(TensorUsage::IFM0)->Set(Quantization::Unit());
        view->Output(TensorUsage::OFM)->Set(Quantization::Unit());
        REQUIRE(checker->Check(view.get()));

        view->Output(TensorUsage::OFM)->Set(Shape(1, 2, 32, 2));
        REQUIRE_FALSE(checker->Check(view.get()));
        view->Disconnect();
    }
}

TEST_CASE("Neural-AI keeps generic TFLite Mul outside the supported operator set")
{
    auto checker = MakeSupportedOpsChecker(REGOR_ARCH_NEURALAI);
    auto lhs = CreateTensor("lhs", Shape(1, 2, 2, 32), DataType::Int8);
    auto rhs = CreateTensor("rhs", Shape(1, 2, 2, 32), DataType::Int8);
    auto output = CreateTensor("output", Shape(1, 2, 2, 32), DataType::Int8);
    auto mul = CreateOperation(
        OpType::Mul, TensorUsage::IFM0, lhs, TensorUsage::IFM1, rhs, TensorUsage::OFM, output);
    mul->Input(TensorUsage::IFM0)->Set(Quantization::Unit());
    mul->Input(TensorUsage::IFM1)->Set(Quantization::Unit());
    mul->Output(TensorUsage::OFM)->Set(Quantization::Unit());
    REQUIRE_FALSE(checker->Check(mul.get()));
    mul->Disconnect();
}

TEST_CASE("Neural-AI Graph IR keeps depth-changing views for explicit rejection")
{
    ArchNeuralAI arch;
    GraphOptimiserOptions options;
    GraphIrOptimiser optimiser(arch.Constraints(), options, nullptr);

    const auto optimise = [&](const Shape &outputShape)
    {
        auto input = CreateTensor("input", Shape(1, 2, 2, 32), DataType::Int8);
        auto output = CreateTensor("output", outputShape, DataType::Int8);
        auto reshape = CreateOperation(
            OpType::Reshape, TensorUsage::IFM0, input, TensorUsage::OFM, output);
        std::vector<std::shared_ptr<Operation>> sourceOps = {reshape};
        auto graph = CreateGraph(sourceOps);
        optimiser.Process(graph.get());
        std::vector<std::shared_ptr<Operation>> operations;
        graph->GetAllOperations(operations);
        REQUIRE(operations.size() == 1);
        return operations[0]->Type();
    };

    REQUIRE(optimise(Shape(1, 1, 4, 32)) == OpType::MemoryCopy);
    REQUIRE(optimise(Shape(1, 2, 32, 2)) == OpType::Reshape);
}

TEST_CASE("Neural-AI graph optimiser inserts ROW32 boundary conversions")
{
    ArchNeuralAI arch;
    auto input = CreateTensor("input", Shape(1, 1, 1, 33), DataType::Int8);
    auto output = CreateTensor("output", Shape(1, 1, 1, 34), DataType::Int8);
    auto matrix = CreateOperation(
        OpType::FullyConnected, TensorUsage::IFM0, input, TensorUsage::OFM, output);
    std::vector<std::shared_ptr<Operation>> sourceOps = {matrix};
    auto graph = CreateGraph(sourceOps);

    GraphOptimiserOptions options;
    NeuralAIGraphOptimiser optimiser(arch.Constraints(), options, nullptr);
    optimiser.Process(graph.get());

    REQUIRE(graph->IsInput(input.get()));
    REQUIRE(graph->IsOutput(output.get()));
    REQUIRE(matrix->IFM(0) != input.get());
    REQUIRE(matrix->OFM() != output.get());
    REQUIRE(matrix->IFM(0)->Name() == "input/row32");
    REQUIRE(matrix->OFM()->Name() == "output/row32");

    REQUIRE(input->Readers().size() == 1);
    REQUIRE(input->Readers().front()->Type() == OpType::MemoryCopy);
    REQUIRE(input->Readers().front()->OFM() == matrix->IFM(0));
    REQUIRE(output->Writers().size() == 1);
    REQUIRE(output->Writers().front()->Type() == OpType::MemoryCopy);
    REQUIRE(output->Writers().front()->IFM(0) == matrix->OFM());

    std::vector<Operation *> operations;
    graph->GetAllOperations(operations);
    REQUIRE(operations.size() == 3);
}

TEST_CASE("Neural-AI prepare optimiser compensates asymmetric Conv padding byte-exactly")
{
    const Shape inputShape(1, 3, 3, 1);
    const Shape weightShape(1, 3, 3, 1);
    const Shape outputShape(1, 3, 3, 1);
    const std::vector<int8_t> inputValues{-8, -2, 4, 7, -3, 1, 5, 9, -6};
    const std::vector<int8_t> weightValues{1, -2, 3, -4, 5, -6, 7, -8, 9};
    constexpr int32_t originalBias = 17;
    constexpr int64_t inputZeroPoint = -3;

    auto input = CreateTensor("input", inputShape, DataType::Int8);
    auto weights = CreateTensor("weights", weightShape, DataType::Int8, std::vector<int8_t>(weightValues));
    auto bias = CreateTensor("bias", Shape(1), DataType::Int32, std::vector<int32_t>{originalBias});
    auto output = CreateTensor("output", outputShape, DataType::Int8);
    auto conv = CreateOperation(OpType::Conv2D, TensorUsage::IFM0, input, TensorUsage::OFM, output);
    conv->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
    conv->ConnectInput(TensorUsage::Scales, bias).Set(Quantization::Unit());
    conv->SetKernel(std::make_unique<Kernel>(
        Point2i(3, 3), Point2i(1, 1), Point2i(1, 1), Margin(1, 1, 1, 1)));
    Quantization inputQuantization = Quantization::Unit();
    inputQuantization.zeroPoints = {inputZeroPoint};
    conv->Input(TensorUsage::IFM0)->Set(inputQuantization);
    std::vector<std::shared_ptr<Operation>> operations{conv};
    auto graph = CreateGraph(operations);

    ArchNeuralAI architecture;
    GraphOptimiserOptions options;
    NeuralAIGraphOptimiser optimiser(
        architecture.Constraints(), options, nullptr, NeuralAIGraphOptimiserStage::Prepare);
    optimiser.OptimiseGraph(graph.get());

    REQUIRE(conv->Input(TensorUsage::IFM0)->quantization.zeroPoints == std::vector<int64_t>{0});
    REQUIRE(conv->Input(TensorUsage::IFM0)->shape == inputShape);
    REQUIRE(conv->Kernel()->Padding().Top() == 1);
    REQUIRE(conv->Kernel()->Padding().Left() == 1);
    REQUIRE(conv->Kernel()->Padding().Bottom() == 1);
    REQUIRE(conv->Kernel()->Padding().Right() == 1);
    REQUIRE(conv->IFM(0)->Writers().empty());
    const auto paddingValue = conv->Input(TensorUsage::Params1);
    REQUIRE(paddingValue != nullptr);
    REQUIRE(paddingValue->tensor->View().Values<int8_t>()[0] == int8_t(inputZeroPoint));

    int32_t weightSum = 0;
    for ( const int8_t value : weightValues ) weightSum += value;
    const int32_t expectedBias = originalBias - int32_t(inputZeroPoint) * weightSum;
    const auto compensatedBias = conv->Input(TensorUsage::Scales)->tensor->View().Values<int32_t>();
    REQUIRE(compensatedBias[0] == expectedBias);
    REQUIRE(conv->Input(TensorUsage::Scales)->tensor != bias);

    for ( int oy = 0; oy < 3; ++oy )
    {
        for ( int ox = 0; ox < 3; ++ox )
        {
            int32_t reference = originalBias;
            int32_t transformed = expectedBias;
            for ( int ky = 0; ky < 3; ++ky )
            {
                for ( int kx = 0; kx < 3; ++kx )
                {
                    const int iy = oy + ky - 1;
                    const int ix = ox + kx - 1;
                    const int8_t raw = iy < 0 || iy >= 3 || ix < 0 || ix >= 3 ?
                        int8_t(inputZeroPoint) : inputValues[iy * 3 + ix];
                    const int8_t weight = weightValues[ky * 3 + kx];
                    reference += (int32_t(raw) - int32_t(inputZeroPoint)) * int32_t(weight);
                    transformed += int32_t(raw) * int32_t(weight);
                }
            }
            REQUIRE(transformed == reference);
        }
    }
}

TEST_CASE("Neural-AI prepare optimiser rejects overflowing asymmetric Conv compensation")
{
    auto input = CreateTensor("input", Shape(1, 1, 1, 1), DataType::Int8);
    auto weights = CreateTensor("weights", Shape(1, 1, 1, 1), DataType::Int8, std::vector<int8_t>{1});
    auto bias = CreateTensor(
        "bias", Shape(1), DataType::Int32, std::vector<int32_t>{std::numeric_limits<int32_t>::max()});
    auto output = CreateTensor("output", Shape(1, 1, 1, 1), DataType::Int8);
    auto conv = CreateOperation(OpType::Conv2D, TensorUsage::IFM0, input, TensorUsage::OFM, output);
    conv->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
    conv->ConnectInput(TensorUsage::Scales, bias).Set(Quantization::Unit());
    conv->SetKernel(std::make_unique<Kernel>(Point2i(1, 1), Point2i(1, 1), Point2i(1, 1)));
    Quantization inputQuantization = Quantization::Unit();
    inputQuantization.zeroPoints = {-1};
    conv->Input(TensorUsage::IFM0)->Set(inputQuantization);
    std::vector<std::shared_ptr<Operation>> operations{conv};
    auto graph = CreateGraph(operations);

    ArchNeuralAI architecture;
    GraphOptimiserOptions options;
    NeuralAIGraphOptimiser optimiser(
        architecture.Constraints(), options, nullptr, NeuralAIGraphOptimiserStage::Prepare);
    optimiser.OptimiseGraph(graph.get());

    REQUIRE(conv->Input(TensorUsage::IFM0)->quantization.zeroPoints == std::vector<int64_t>{-1});
    REQUIRE(conv->Input(TensorUsage::Scales)->tensor == bias);
}

TEST_CASE("Neural-AI constraints enforce signed matrix zero points")
{
    ArchNeuralAI arch;
    auto *constraints = arch.Constraints();

    REQUIRE(constraints->SupportedZeroPoint(0, TensorUsage::IFM, DataType::Int8, OpType::MatMul));
    REQUIRE_FALSE(constraints->SupportedZeroPoint(1, TensorUsage::IFM, DataType::Int8, OpType::MatMul));
    REQUIRE(constraints->SupportedZeroPoint(0, TensorUsage::Weights, DataType::Int8, OpType::MatMul));
    REQUIRE_FALSE(constraints->SupportedZeroPoint(-1, TensorUsage::Weights, DataType::Int8, OpType::MatMul));
    REQUIRE(constraints->SupportedZeroPoint(-128, TensorUsage::OFM, DataType::Int8, OpType::MatMul));
    REQUIRE(constraints->SupportedZeroPoint(127, TensorUsage::OFM, DataType::Int8, OpType::MatMul));
    REQUIRE_FALSE(constraints->SupportedZeroPoint(128, TensorUsage::OFM, DataType::Int8, OpType::MatMul));
}

TEST_CASE("Neural-AI matrix op configuration exposes GEMM granules")
{
    ArchNeuralAI arch;
    ArchitectureConfigQuery query{};
    query.ifmBits = 8;
    query.ofmBits = 8;
    query.transpose = TransposeType::None;
    query.reverse = ReverseType::None;

    auto config = arch.GetOpConfig(OpType::MatMul, query);
    REQUIRE(config);
    REQUIRE(config->OptimalStripeGranule() == Point2i(32, 1));
    REQUIRE(config->MinimalStripeGranule() == Point2i(1, 1));
    REQUIRE(config->OptimalDepthGranule() == 32);
    REQUIRE(config->MinimumDepthGranule() == 32);
    REQUIRE(arch.GetOpConfig(OpType::Conv2D, query));
}

TEST_CASE("Neural-AI performance model estimates GEMM and DMA costs")
{
    ArchNeuralAI arch;
    PerformanceQuery query{};
    query.type = OpType::FullyConnected;
    query.ifm[0].shape = Shape(1, 1, 1, 33);
    query.ifm[0].type = DataType::Int8;
    query.ifm[0].memory = arch.FeatureMapMemory().memory;
    query.ofm.shape = Shape(1, 1, 1, 34);
    query.ofm.type = DataType::Int8;
    query.ofm.memory = arch.FeatureMapMemory().memory;
    query.constMemory = arch.ReadonlyMemory().memory;
    query.encodedWeightSize = 4096;
    query.encodedScaleSize = 1024;

    auto *performance = arch.Performance();
    REQUIRE(performance != nullptr);
    const CycleCost gemm = performance->MeasureCycleCost(query);
    REQUIRE(gemm.macs == 33 * 34);
    REQUIRE(gemm.opCycles == 2);

    query.type = OpType::AvgPool;
    query.ifm[0].shape = Shape(1, 2, 3, 33);
    query.ofm.shape = Shape(1, 1, 1, 33);
    const CycleCost globalAverage = performance->MeasureCycleCost(query);
    REQUIRE(globalAverage.macs == 0);
    REQUIRE(globalAverage.opCycles == 12);

    query.type = OpType::FullyConnected;
    query.ifm[0].shape = Shape(1, 1, 1, 33);
    query.ofm.shape = Shape(1, 1, 1, 34);
    const ElementAccess elements = performance->MeasureElementAccess(query);
    const ElementAccess bytes = performance->ElementTransferToBytes(query, elements);
    REQUIRE(bytes.ifmRead[0] == 33);
    REQUIRE(bytes.ofmWrite == 34);
    REQUIRE(bytes.constRead[0] == 4096);
    REQUIRE(bytes.constRead[1] == 1024);

    query.type = OpType::MemoryCopy;
    REQUIRE(performance->MeasureCycleCost(query).opCycles == 0);
    REQUIRE(performance->MemToMemCycles(
        arch.FeatureMapMemory().memory, arch.ReadonlyMemory().memory, 64) == 3);
}

TEST_CASE("Neural-AI scheduler emits interleaved Conv LUT stripes")
{
    constexpr int height = 162;
    constexpr int width = 162;
    constexpr int outputHeight = 80;
    constexpr int outputWidth = 80;
    constexpr int inputChannels = 3;
    constexpr int outputChannels = 16;
    ArchNeuralAI arch;
    auto input = CreateTensor("input", Shape(1, height, width, inputChannels), DataType::Int8);
    auto convOutput = CreateTensor("conv_output", Shape(1, outputHeight, outputWidth, outputChannels), DataType::Int8);
    auto output = CreateTensor("output", Shape(1, outputHeight, outputWidth, outputChannels), DataType::Int8);
    auto weights = CreateTensor("weights", Shape(outputChannels, 3, 3, inputChannels), DataType::Int8,
        std::vector<int8_t>(outputChannels * 3 * 3 * inputChannels, 1));
    auto scales = CreateTensor("scales", Shape(outputChannels), DataType::Int32,
        std::vector<int32_t>(outputChannels, 0));
    std::vector<int8_t> lutValues(256);
    for ( int index = 0; index < 256; ++index ) lutValues[index] = int8_t(index - 128);
    auto lut = CreateTensor("lut", Shape(256), DataType::Int8, std::move(lutValues));

    auto conv = CreateOperation(
        OpType::Conv2D, TensorUsage::IFM0, input, TensorUsage::OFM, convOutput);
    conv->Input(TensorUsage::IFM0)->Set(Quantization::Unit());
    conv->Output(TensorUsage::OFM)->Set(Quantization::Unit());
    conv->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
    conv->ConnectInput(TensorUsage::Scales, scales).Set(Quantization::Unit());
    conv->SetKernel(std::make_unique<Kernel>(
        Point2i(3, 3), Point2i(2, 2), Point2i(1, 1)));
    auto activation = CreateOperation(
        OpType::LUT, TensorUsage::IFM0, convOutput, TensorUsage::OFM, output);
    activation->Input(TensorUsage::IFM0)->Set(Quantization::Unit());
    activation->Output(TensorUsage::OFM)->Set(Quantization::Unit());
    activation->ConnectInput(TensorUsage::LUT, lut);
    std::vector<std::shared_ptr<Operation>> sourceOps = {conv, activation};
    auto graph = CreateGraph(sourceOps);

    GraphOptimiserOptions graphOptions;
    NeuralAIGraphOptimiser optimiser(arch.Constraints(), graphOptions, nullptr);
    optimiser.Process(graph.get());
    const std::unordered_map<UniqueId, UniqueId> equivalenceIds;
    SchedulerPacking packing(&arch, false, equivalenceIds);
    auto scheduleOps = packing.Process(graph.get());
    SchedulerOptions schedulerOptions;
    schedulerOptions.optimizationStrategy = OptimizationStrategy::Performance;
    schedulerOptions.optimizationStagingLimit = 256 * 1024;
    schedulerOptions.disabled.Set(SchedulerFeature::WeightBuffering);
    Scheduler scheduler(
        &arch, schedulerOptions, "neural-ai-cascade-hlc", scheduleOps,
        packing.OpConfigCompatablility());
    auto schedule = scheduler.Process();

    const SchedulerOperation *scheduledConv = nullptr;
    const SchedulerOperation *scheduledLut = nullptr;
    for ( const auto &operation : scheduleOps )
    {
        if ( operation->Type() == OpType::Conv2D ) scheduledConv = operation.get();
        if ( operation->Type() == OpType::LUT ) scheduledLut = operation.get();
    }
    REQUIRE(scheduledConv != nullptr);
    REQUIRE(scheduledLut != nullptr);
    const int cascade = schedule->Cost(scheduledConv)->cascade;
    REQUIRE(cascade > 0);
    REQUIRE(schedule->Cost(scheduledLut)->cascade == cascade);
    REQUIRE(scheduledConv->OFM()->tensor->storageShape.Height() < outputHeight);

    HLCStreamGenerator generator(0, false);
    HLCStream commands = generator.GenerateCommandStream(
        vector_span<std::unique_ptr<SchedulerOperation>>(scheduleOps), schedule.get());
    std::vector<OpType> stripeTypes;
    for ( const auto &command : commands )
    {
        if ( command->CommandType() != HighLevelCommandType::STRIPE ) continue;
        stripeTypes.push_back(static_cast<const HLCStripe *>(command.get())->operation->type);
    }
    REQUIRE(stripeTypes.size() > 2);
    const auto convPosition = std::find(stripeTypes.begin(), stripeTypes.end(), OpType::Conv2D);
    const auto lutPosition = std::find(stripeTypes.begin(), stripeTypes.end(), OpType::LUT);
    REQUIRE(convPosition != stripeTypes.end());
    REQUIRE(lutPosition != stripeTypes.end());
    REQUIRE(convPosition < lutPosition);
    REQUIRE(std::find(lutPosition + 1, stripeTypes.end(), OpType::Conv2D) != stripeTypes.end());

    CompiledNeuralAIArtifact artifact;
    std::string error;
    NeuralAICommandGenerator commandGenerator;
    const bool generated = commandGenerator.Generate(graph.get(), scheduleOps, schedule.get(), artifact, error);
    INFO(error);
    REQUIRE(generated);
    REQUIRE(error.empty());
    uint32_t linebufferJobs = 0;
    uint32_t lutCommands = 0;
    uint32_t linebufferRows = 0;
    uint32_t lutBytes = 0;
    size_t commandOffset = 0;
    while ( commandOffset < artifact.commands.size() )
    {
        const uint16_t type = Read16(artifact.commands, commandOffset);
        const uint16_t bytes = Read16(artifact.commands, commandOffset + 2);
        REQUIRE(bytes >= 32);
        if ( type == uint16_t(neuralai::CommandType::LineBufferJob) )
        {
            ++linebufferJobs;
            linebufferRows += Read32(artifact.commands, commandOffset + 132);
        }
        if ( type == uint16_t(neuralai::CommandType::AFULut) )
        {
            ++lutCommands;
            lutBytes += Read32(artifact.commands, commandOffset + 40);
        }
        commandOffset += bytes;
    }
    REQUIRE(commandOffset == artifact.commands.size());
    REQUIRE(linebufferJobs > 1);
    REQUIRE(lutCommands > 1);
    REQUIRE(linebufferRows == outputHeight * outputWidth);
    REQUIRE(lutBytes == outputHeight * outputWidth * 32);
    REQUIRE(artifact.requiredTCDMBytes <= ArchNeuralAI::AllocatableTCDMBytes);
}

TEST_CASE("Neural-AI scheduler gathers native Concat stripes with DMA3D")
{
    constexpr int height = 40;
    constexpr int width = 40;
    constexpr int inputChannels = 64;
    constexpr int concatChannels = 2 * inputChannels;
    constexpr int outputChannels = 64;
    ArchNeuralAI arch;
    auto lhs = CreateTensor("lhs", Shape(1, height, width, inputChannels), DataType::Int8);
    auto rhs = CreateTensor("rhs", Shape(1, height, width, inputChannels), DataType::Int8);
    auto concatOutput = CreateTensor(
        "concat_output", Shape(1, height, width, concatChannels), DataType::Int8);
    auto output = CreateTensor("output", Shape(1, height, width, outputChannels), DataType::Int8);
    auto weights = CreateTensor("weights", Shape(outputChannels, 1, 1, concatChannels), DataType::Int8,
        std::vector<int8_t>(outputChannels * concatChannels, 1));
    auto scales = CreateTensor("scales", Shape(outputChannels), DataType::Int32,
        std::vector<int32_t>(outputChannels, 0));

    auto concat = CreateOperation(
        OpType::Concat, TensorUsage::IFM0, lhs, TensorUsage::OFM, concatOutput);
    concat->ConnectInput(TensorUsage::IFM1, rhs).Set(Quantization::Unit());
    concat->Input(TensorUsage::IFM0)->Set(Quantization::Unit());
    concat->Output(TensorUsage::OFM)->Set(Quantization::Unit());
    concat->Attribute<axis_attr_t>()->axis = 3;
    auto pointwise = CreateOperation(
        OpType::Conv2D, TensorUsage::IFM0, concatOutput, TensorUsage::OFM, output);
    pointwise->Input(TensorUsage::IFM0)->Set(Quantization::Unit());
    pointwise->Output(TensorUsage::OFM)->Set(Quantization::Unit());
    pointwise->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
    pointwise->ConnectInput(TensorUsage::Scales, scales).Set(Quantization::Unit());
    pointwise->SetKernel(std::make_unique<Kernel>(
        Point2i(1, 1), Point2i(1, 1), Point2i(1, 1)));
    std::vector<std::shared_ptr<Operation>> sourceOps = {concat, pointwise};
    auto graph = CreateGraph(sourceOps);

    GraphOptimiserOptions graphOptions;
    NeuralAIGraphOptimiser optimiser(arch.Constraints(), graphOptions, nullptr);
    optimiser.Process(graph.get());
    const std::unordered_map<UniqueId, UniqueId> equivalenceIds;
    SchedulerPacking packing(&arch, false, equivalenceIds);
    auto scheduleOps = packing.Process(graph.get());
    SchedulerOptions schedulerOptions;
    schedulerOptions.optimizationStrategy = OptimizationStrategy::Performance;
    schedulerOptions.optimizationStagingLimit = 128 * 1024;
    schedulerOptions.disabled.Set(SchedulerFeature::WeightBuffering);
    Scheduler scheduler(
        &arch, schedulerOptions, "neural-ai-concat-cascade", scheduleOps,
        packing.OpConfigCompatablility());
    auto schedule = scheduler.Process();

    const SchedulerOperation *scheduledConcat = nullptr;
    const SchedulerOperation *scheduledConv = nullptr;
    for ( const auto &operation : scheduleOps )
    {
        if ( operation->Type() == OpType::Concat ) scheduledConcat = operation.get();
        if ( operation->Type() == OpType::Conv2D ) scheduledConv = operation.get();
    }
    REQUIRE(scheduledConcat != nullptr);
    REQUIRE(scheduledConv != nullptr);
    const int cascade = schedule->Cost(scheduledConcat)->cascade;
    REQUIRE(cascade > 0);
    REQUIRE(schedule->Cost(scheduledConv)->cascade == cascade);
    REQUIRE(scheduledConcat->OFM()->tensor->storageShape.Height() < height);

    CompiledNeuralAIArtifact artifact;
    std::string error;
    NeuralAICommandGenerator commandGenerator;
    const bool generated = commandGenerator.Generate(graph.get(), scheduleOps, schedule.get(), artifact, error);
    INFO(error);
    REQUIRE(generated);
    REQUIRE(error.empty());
    uint32_t gatherCommands = 0;
    size_t commandOffset = 0;
    while ( commandOffset < artifact.commands.size() )
    {
        const uint16_t type = Read16(artifact.commands, commandOffset);
        const uint16_t bytes = Read16(artifact.commands, commandOffset + 2);
        REQUIRE(bytes >= 32);
        if ( type == uint16_t(neuralai::CommandType::DMA3D) &&
             Read32(artifact.commands, commandOffset + 60) ==
                 uint32_t(neuralai::DMADirection::LocalToLocal) )
        {
            REQUIRE(bytes == sizeof(neuralai::CommandDMA3DV2));
            REQUIRE(Read32(artifact.commands, commandOffset + 32) == 32);
            ++gatherCommands;
        }
        commandOffset += bytes;
    }
    REQUIRE(commandOffset == artifact.commands.size());
    REQUIRE(gatherCommands > 2);
    REQUIRE(artifact.requiredTCDMBytes <= ArchNeuralAI::AllocatableTCDMBytes);
}

TEST_CASE("Neural-AI scheduler lowers a complete FullyConnected graph")
{
    constexpr int depthK = 33;
    constexpr int depthN = 34;
    ArchNeuralAI arch;
    auto input = CreateTensor("input", Shape(1, 1, 1, depthK), DataType::Int8);
    auto output = CreateTensor("output", Shape(1, 1, 1, depthN), DataType::Int8);
    auto weights = CreateTensor(
        "weights", Shape(depthN, 1, 1, depthK), DataType::Int8,
        std::vector<int8_t>(depthK * depthN, 1));
    auto scales = CreateTensor(
        "scales", Shape(1, 1, 1, depthN), DataType::Int32,
        std::vector<int32_t>(depthN, 0));
    auto matrix = CreateOperation(
        OpType::FullyConnected, TensorUsage::IFM0, input, TensorUsage::OFM, output);
    matrix->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
    matrix->ConnectInput(TensorUsage::Scales, scales).Set(Quantization::Unit());
    std::vector<std::shared_ptr<Operation>> sourceOps = {matrix};
    auto graph = CreateGraph(sourceOps);

    GraphOptimiserOptions graphOptions;
    NeuralAIGraphOptimiser optimiser(arch.Constraints(), graphOptions, nullptr);
    optimiser.Process(graph.get());

    const std::unordered_map<UniqueId, UniqueId> equivalenceIds;
    SchedulerPacking packing(&arch, false, equivalenceIds);
    auto scheduleOps = packing.Process(graph.get());
    REQUIRE(scheduleOps.size() == 3);

    SchedulerOptions schedulerOptions;
    schedulerOptions.disabled.Set(SchedulerFeature::Cascading);
    schedulerOptions.disabled.Set(SchedulerFeature::WeightBuffering);
    Scheduler scheduler(
        &arch, schedulerOptions, "neural-ai", scheduleOps, packing.OpConfigCompatablility());
    auto schedule = scheduler.Process();

    REQUIRE(schedule != nullptr);
    REQUIRE(schedule->Costs().size() == 3);
    REQUIRE(schedule->memoryUsage.at(arch.FeatureMapMemory()) % ArchNeuralAI::DMAAlignment == 0);
    REQUIRE(scheduleOps[0]->OFM()->tensor->format == TensorFormat::Row32);
    REQUIRE(scheduleOps[1]->OFM()->tensor->format == TensorFormat::Row32);
    REQUIRE(scheduleOps[2]->OFM()->tensor->format == TensorFormat::NHWC);
    const SchedulerOpInfo *matrixCost = schedule->Cost(scheduleOps[1].get());
    REQUIRE(matrixCost->npuWeightsTensor != nullptr);
    REQUIRE(matrixCost->npuWeightsTensor->totalWeightBytes == 4 * 32 * 32);
    REQUIRE(matrixCost->npuWeightsTensor->AllocationSizeBytes() == 6 * 32 * 32);
    REQUIRE(matrixCost->npuWeightsTensor->encodedRanges.size() == 1);
    const WeightRange &range = matrixCost->npuWeightsTensor->encodedRanges.begin()->second;
    REQUIRE(range.scaleBytes == 2 * 32 * 32);
    REQUIRE(range.weightOffset == 2 * 32 * 32);
    REQUIRE(range.weightBytes == 4 * 32 * 32);
    REQUIRE(matrixCost->npuScalesTensor == nullptr);

    CompiledNeuralAIArtifact artifact;
    std::string error;
    NeuralAICommandGenerator commandGenerator;
    REQUIRE(commandGenerator.Generate(graph.get(), scheduleOps, schedule.get(), artifact, error));
    REQUIRE(error.empty());
    REQUIRE(artifact.commandCount == 12);
    REQUIRE(artifact.commands.size() == 928);
    REQUIRE(artifact.constants.size() == 4 * 32 * 32);
    REQUIRE(artifact.qparams.size() == 2 * 32);
    REQUIRE(artifact.bindings.size() == 2);
    REQUIRE(artifact.requiredTCDMBytes % ArchNeuralAI::DMAAlignment == 0);

    const std::vector<uint16_t> expectedTypes = {
        uint16_t(neuralai::CommandType::CopyLayout),
        uint16_t(neuralai::CommandType::RQLoad),
        uint16_t(neuralai::CommandType::DMA2D),
        uint16_t(neuralai::CommandType::Gemm32),
        uint16_t(neuralai::CommandType::DMA2D),
        uint16_t(neuralai::CommandType::Gemm32Requant),
        uint16_t(neuralai::CommandType::RQLoad),
        uint16_t(neuralai::CommandType::DMA2D),
        uint16_t(neuralai::CommandType::Gemm32),
        uint16_t(neuralai::CommandType::DMA2D),
        uint16_t(neuralai::CommandType::Gemm32Requant),
        uint16_t(neuralai::CommandType::CopyLayout),
        uint16_t(neuralai::CommandType::End),
    };
    size_t commandOffset = 0;
    uint32_t externalToLocalDMAs = 0;
    uint32_t localToLocalDMAs = 0;
    for ( uint16_t expected : expectedTypes )
    {
        REQUIRE(Read16(artifact.commands, commandOffset) == expected);
        if ( expected == uint16_t(neuralai::CommandType::DMA2D) )
        {
            const uint16_t sourceRegion = Read16(artifact.commands, commandOffset + 16);
            const uint16_t destinationRegion = Read16(artifact.commands, commandOffset + 24);
            const uint32_t direction = Read32(artifact.commands, commandOffset + 48);
            if ( sourceRegion == uint16_t(neuralai::Region::ModelConstants) )
            {
                REQUIRE(destinationRegion == uint16_t(neuralai::Region::TCDMScratch));
                REQUIRE(direction == uint32_t(neuralai::DMADirection::ExternalToLocal));
                ++externalToLocalDMAs;
            }
            else
            {
                REQUIRE(sourceRegion == uint16_t(neuralai::Region::TCDMScratch));
                REQUIRE(destinationRegion == uint16_t(neuralai::Region::TCDMScratch));
                REQUIRE(direction == uint32_t(neuralai::DMADirection::LocalToLocal));
                ++localToLocalDMAs;
            }
        }
        commandOffset += Read16(artifact.commands, commandOffset + 2);
    }
    REQUIRE(commandOffset == artifact.commands.size());
    REQUIRE(externalToLocalDMAs == 0);
    REQUIRE(localToLocalDMAs == 4);
    REQUIRE(Read32(artifact.commands, 60) == depthK);
    REQUIRE(Read32(artifact.commands, 64) == 64);
    REQUIRE(Read32(artifact.commands, 96 + 16) == 0);
    REQUIRE(Read32(artifact.commands, 352 + 20) == 32 * 32);
    REQUIRE(Read32(artifact.commands, 448 + 16) == 32);
    REQUIRE(Read32(artifact.commands, 704 + 20) == 3 * 32 * 32);
    REQUIRE(Read32(artifact.commands, 800 + 60) == 64);
    REQUIRE(Read32(artifact.commands, 800 + 64) == depthN);

    std::vector<uint8_t> package;
    REQUIRE(WriteNeuralAIModel(artifact, package, error));
    REQUIRE(Read32(package, 0) == neuralai::ModelMagic);
}

TEST_CASE("Neural-AI command generator emits direct single-tile requant")
{
    constexpr int depthK = 31;
    constexpr int depthN = 17;
    ArchNeuralAI arch;
    auto input = CreateTensor("input", Shape(1, 1, 1, depthK), DataType::Int8);
    auto output = CreateTensor("output", Shape(1, 1, 1, depthN), DataType::Int8);
    auto weights = CreateTensor("weights", Shape(depthN, 1, 1, depthK), DataType::Int8,
        std::vector<int8_t>(depthK * depthN, 1));
    auto scales = CreateTensor("scales", Shape(1, 1, 1, depthN), DataType::Int32,
        std::vector<int32_t>(depthN, 0));
    auto matrix = CreateOperation(
        OpType::FullyConnected, TensorUsage::IFM0, input, TensorUsage::OFM, output);
    matrix->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
    matrix->ConnectInput(TensorUsage::Scales, scales).Set(Quantization::Unit());
    std::vector<std::shared_ptr<Operation>> sourceOps = {matrix};
    auto graph = CreateGraph(sourceOps);

    GraphOptimiserOptions graphOptions;
    NeuralAIGraphOptimiser optimiser(arch.Constraints(), graphOptions, nullptr);
    optimiser.Process(graph.get());
    const std::unordered_map<UniqueId, UniqueId> equivalenceIds;
    SchedulerPacking packing(&arch, false, equivalenceIds);
    auto scheduleOps = packing.Process(graph.get());
    SchedulerOptions schedulerOptions;
    schedulerOptions.disabled.Set(SchedulerFeature::Cascading);
    schedulerOptions.disabled.Set(SchedulerFeature::WeightBuffering);
    Scheduler scheduler(
        &arch, schedulerOptions, "neural-ai-direct", scheduleOps, packing.OpConfigCompatablility());
    auto schedule = scheduler.Process();

    CompiledNeuralAIArtifact artifact;
    std::string error;
    NeuralAICommandGenerator commandGenerator;
    REQUIRE(commandGenerator.Generate(graph.get(), scheduleOps, schedule.get(), artifact, error));
    REQUIRE(artifact.commandCount == 4);
    REQUIRE(artifact.commands.size() == 352);
    REQUIRE(artifact.constants.size() == 32 * 32);
    REQUIRE(artifact.qparams.size() == 32);
    REQUIRE(Read16(artifact.commands, 0) == uint16_t(neuralai::CommandType::CopyLayout));
    REQUIRE(Read16(artifact.commands, 96) == uint16_t(neuralai::CommandType::RQLoad));
    REQUIRE(Read16(artifact.commands, 128) == uint16_t(neuralai::CommandType::Gemm32Requant));
    REQUIRE(Read16(artifact.commands, 128 + 32) == 0);
    REQUIRE(Read32(artifact.commands, 128 + 52) == 32);
    REQUIRE(Read16(artifact.commands, 224) == uint16_t(neuralai::CommandType::CopyLayout));
    REQUIRE(Read16(artifact.commands, 320) == uint16_t(neuralai::CommandType::End));
}

TEST_CASE("Neural-AI command generator rejects sliced external-to-external MemoryCopy")
{
    ArchNeuralAI arch;
    const Shape shape(1, 1, 2, 32);
    auto input = CreateTensor("input", shape, DataType::Int8);
    auto output = CreateTensor("output", shape, DataType::Int8);
    auto copy = CreateOperation(
        OpType::MemoryCopy, TensorUsage::IFM0, input, TensorUsage::OFM, output);
    const TensorSlice slice(Shape(0, 0, 1, 0), Shape(1, 1, 1, 32));
    copy->Input(TensorUsage::IFM0)->Set(slice);
    copy->Output(TensorUsage::OFM)->Set(slice);
    std::vector<std::shared_ptr<Operation>> sourceOps = {copy};
    auto graph = CreateGraph(sourceOps);

    const std::unordered_map<UniqueId, UniqueId> equivalenceIds;
    SchedulerPacking packing(&arch, false, equivalenceIds);
    auto scheduleOps = packing.Process(graph.get());
    SchedulerOptions schedulerOptions;
    schedulerOptions.disabled.Set(SchedulerFeature::Cascading);
    schedulerOptions.disabled.Set(SchedulerFeature::WeightBuffering);
    Scheduler scheduler(
        &arch, schedulerOptions, "neural-ai-sliced-copy", scheduleOps,
        packing.OpConfigCompatablility());
    auto schedule = scheduler.Process();

    CompiledNeuralAIArtifact artifact;
    std::string error;
    NeuralAICommandGenerator commandGenerator;
    REQUIRE_FALSE(commandGenerator.Generate(
        graph.get(), scheduleOps, schedule.get(), artifact, error));
    REQUIRE(error == "Neural-AI DMA1D requires a local source or destination");
}

TEST_CASE("Neural-AI command generator reads compact class LUT slice directly")
{
    constexpr int locations = 37;
    ArchNeuralAI arch;
    const Shape fullShape(1, 1, 144, locations);
    const Shape classShape(1, 1, 80, locations);
    auto input = CreateTensor("input", fullShape, DataType::Int8);
    auto staged = CreateTensor("staged", fullShape, DataType::Int8);
    auto lutOutput = CreateTensor("lut_output", classShape, DataType::Int8);
    auto output = CreateTensor("output", classShape, DataType::Int8);
    std::vector<int8_t> lutValues(256);
    for ( int index = 0; index < 256; ++index ) lutValues[index] = int8_t(index - 128);
    auto lut = CreateTensor("lut", Shape(256), DataType::Int8, std::move(lutValues));

    auto inputCopy = CreateOperation(
        OpType::MemoryCopy, TensorUsage::IFM0, input, TensorUsage::OFM, staged);
    inputCopy->Input(TensorUsage::IFM0)->Set(Quantization::Unit());
    inputCopy->Output(TensorUsage::OFM)->Set(Quantization::Unit());
    auto activation = CreateOperation(
        OpType::LUT, TensorUsage::IFM0, staged, TensorUsage::OFM, lutOutput);
    activation->Input(TensorUsage::IFM0)->Set(
        TensorSlice(Shape(0, 0, 64, 0), classShape, fullShape.WithOnes()));
    activation->Input(TensorUsage::IFM0)->Set(Quantization::Unit());
    activation->Output(TensorUsage::OFM)->Set(Quantization::Unit());
    activation->ConnectInput(TensorUsage::LUT, lut);
    auto outputCopy = CreateOperation(
        OpType::MemoryCopy, TensorUsage::IFM0, lutOutput, TensorUsage::OFM, output);
    outputCopy->Input(TensorUsage::IFM0)->Set(Quantization::Unit());
    outputCopy->Output(TensorUsage::OFM)->Set(Quantization::Unit());
    std::vector<std::shared_ptr<Operation>> sourceOps = {inputCopy, activation, outputCopy};
    auto graph = CreateGraph(sourceOps);

    const std::unordered_map<UniqueId, UniqueId> equivalenceIds;
    SchedulerPacking packing(&arch, false, equivalenceIds);
    auto scheduleOps = packing.Process(graph.get());
    SchedulerOptions schedulerOptions;
    schedulerOptions.disabled.Set(SchedulerFeature::Cascading);
    schedulerOptions.disabled.Set(SchedulerFeature::WeightBuffering);
    Scheduler scheduler(
        &arch, schedulerOptions, "neural-ai-compact-lut-slice", scheduleOps,
        packing.OpConfigCompatablility());
    auto schedule = scheduler.Process();
    const SchedulerOperation *scheduledLut = nullptr;
    for ( const auto &operation : scheduleOps )
        if ( operation->Type() == OpType::LUT ) scheduledLut = operation.get();
    REQUIRE(scheduledLut != nullptr);

    CompiledNeuralAIArtifact artifact;
    std::string error;
    NeuralAICommandGenerator commandGenerator;
    const bool generated = commandGenerator.Generate(
        graph.get(), scheduleOps, schedule.get(), artifact, error);
    INFO(error);
    REQUIRE(generated);
    uint32_t lutCommands = 0;
    size_t offset = 0;
    while ( offset < artifact.commands.size() )
    {
        const uint16_t type = Read16(artifact.commands, offset);
        const uint16_t bytes = Read16(artifact.commands, offset + 2);
        REQUIRE(bytes >= 32);
        if ( type == uint16_t(neuralai::CommandType::AFULut) )
        {
            REQUIRE(Read32(artifact.commands, offset + 20) ==
                uint32_t(scheduledLut->IFM(0)->tensor->AllocatedAddress()) + 64u * locations);
            REQUIRE(Read32(artifact.commands, offset + 40) == 80u * locations);
            ++lutCommands;
        }
        offset += bytes;
    }
    REQUIRE(offset == artifact.commands.size());
    REQUIRE(lutCommands == 1);
}

TEST_CASE("Neural-AI compiler emits a native model package")
{
    constexpr int depthK = 33;
    constexpr int depthN = 34;
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildFullyConnectedModel(depthK, depthN);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));

    REQUIRE(compiler.Compile());
    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    REQUIRE(size > int64_t(sizeof(neuralai::ModelHeaderV1)));
    REQUIRE(Read32(data) == neuralai::ModelMagic);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler emits deterministic model packages")
{
    const auto model = BuildFullyConnectedModel(33, 34);
    std::vector<uint8_t> packages[2];

    for ( int run = 0; run < 2; ++run )
    {
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        REQUIRE(compiler.Compile());

        IRegorBlob *blob = compiler.Output();
        REQUIRE(blob != nullptr);
        int64_t size = 0;
        const auto *data = static_cast<const uint8_t *>(blob->Map(size));
        REQUIRE(size > 0);
        packages[run].assign(data, data + size);
        blob->Unmap(const_cast<uint8_t *>(data));
        blob->Release();
    }

    REQUIRE(packages[0] == packages[1]);
}

TEST_CASE("Neural-AI compiler tiles oversized M dimensions")
{
    for ( const int rows : {257, 511} )
    {
        INFO("rows=" << rows);
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        const auto model = BuildFullyConnectedModel(rows, 32, 32);
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        const bool compiled = compiler.Compile();
        INFO(compiler.LastError());
        REQUIRE(compiled);

        IRegorBlob *blob = compiler.Output();
        REQUIRE(blob != nullptr);
        int64_t size = 0;
        const auto *data = static_cast<const uint8_t *>(blob->Map(size));
        REQUIRE(size > 224 + 32);
        const uint32_t commandBytes = Read32(data + 64 + 12);
        uint32_t offset = 224;
        uint32_t gemmCommands = 0;
        while ( offset < 224 + commandBytes )
        {
            const uint16_t type = uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
            const uint16_t commandSize = uint16_t(data[offset + 2]) |
                (uint16_t(data[offset + 3]) << 8);
            REQUIRE(commandSize >= 32);
            if ( type == uint16_t(neuralai::CommandType::Gemm32) ||
                 type == uint16_t(neuralai::CommandType::Gemm32Accum) ||
                 type == uint16_t(neuralai::CommandType::Gemm32Requant) )
            {
                REQUIRE(Read32(data + offset + 48) > 0);
                REQUIRE(Read32(data + offset + 48) <= 256);
                ++gemmCommands;
            }
            offset += commandSize;
        }
        REQUIRE(offset == 224 + commandBytes);
        REQUIRE(gemmCommands == uint32_t((rows + 255) / 256));
        blob->Unmap(const_cast<uint8_t *>(data));
        blob->Release();
    }
}

TEST_CASE("Neural-AI compiler lowers pointwise Conv2D through C32 GEMM32")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildPointwiseConvModel(2, 3, 33, 34);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));

    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);
    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    REQUIRE(size > int64_t(sizeof(neuralai::ModelHeaderV1)));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    bool sawPointwiseC32 = false;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t type = uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
        const uint16_t commandSize = uint16_t(data[offset + 2]) | (uint16_t(data[offset + 3]) << 8);
        if ( type == uint16_t(neuralai::CommandType::PointwiseC32) )
        {
            sawPointwiseC32 = true;
            REQUIRE(commandSize == 96);
            REQUIRE(Read32(data + offset + 48) <= 256);
            REQUIRE(Read32(data + offset + 52) > 0);
            REQUIRE(Read32(data + offset + 56) == 1);
            REQUIRE(Read32(data + offset + 64) % 32 == 0);
            REQUIRE(Read32(data + offset + 68) % 32 == 0);
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(sawPointwiseC32);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler lowers quantization-safe Add through AFU binary")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildAddModel();
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));

    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);
    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    uint32_t addCommands = 0;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t type = uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
        const uint16_t commandSize = uint16_t(data[offset + 2]) |
            (uint16_t(data[offset + 3]) << 8);
        REQUIRE(commandSize >= 32);
        if ( type == uint16_t(neuralai::CommandType::AFUBinary) )
        {
            REQUIRE(commandSize == sizeof(neuralai::CommandAFUBinaryV2));
            REQUIRE(Read16(data + offset + 16) == uint16_t(neuralai::Region::TCDMScratch));
            REQUIRE(Read16(data + offset + 24) == uint16_t(neuralai::Region::TCDMScratch));
            REQUIRE(Read16(data + offset + 32) == uint16_t(neuralai::Region::TCDMScratch));
            const uint32_t lhsOffset = Read32(data + offset + 20);
            const uint32_t rhsOffset = Read32(data + offset + 28);
            const uint32_t ofmOffset = Read32(data + offset + 36);
            REQUIRE((ofmOffset + 128 <= lhsOffset || lhsOffset + 128 <= ofmOffset));
            REQUIRE((ofmOffset + 128 <= rhsOffset || rhsOffset + 128 <= ofmOffset));
            REQUIRE(Read32(data + offset + 40) == 128);
            REQUIRE(Read32(data + offset + 44) == uint32_t(neuralai::AFUBinaryMode::AddI8));
            ++addCommands;
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(addCommands == 1);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler lowers quantized Add through Spatz")
{
    const auto lowersToSpatz = [](flatbuffers::DetachedBuffer model)
    {
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        const bool compiled = compiler.Compile();
        INFO(compiler.LastError());
        REQUIRE(compiled);
        IRegorBlob *blob = compiler.Output();
        REQUIRE(blob != nullptr);
        int64_t size = 0;
        const auto *data = static_cast<const uint8_t *>(blob->Map(size));
        const uint32_t commandBytes = Read32(data + 64 + 12);
        uint32_t offset = 224;
        uint32_t spatzCommands = 0;
        while ( offset < 224 + commandBytes )
        {
            const uint16_t type = Read16(data + offset);
            const uint16_t commandSize = Read16(data + offset + 2);
            REQUIRE(commandSize >= 32);
            if ( type == uint16_t(neuralai::CommandType::SpatzAdd) )
            {
                REQUIRE(commandSize == sizeof(neuralai::CommandSpatzAddV2));
                REQUIRE(Read32(data + offset + 40) == 128);
                REQUIRE(int32_t(Read32(data + offset + 44)) > 0);
                REQUIRE(int32_t(Read32(data + offset + 52)) > 0);
                REQUIRE(int32_t(Read32(data + offset + 60)) > 0);
                REQUIRE(Read32(data + offset + 88) == 20);
                ++spatzCommands;
            }
            offset += commandSize;
        }
        REQUIRE(offset == 224 + commandBytes);
        REQUIRE(spatzCommands == 1);
        blob->Unmap(const_cast<uint8_t *>(data));
        blob->Release();
    };

    lowersToSpatz(BuildAddModel(1.0f, 1.0f, 1.0f, 1));
    lowersToSpatz(BuildAddModel(1.0f, 0.5f, 1.0f));
}

TEST_CASE("Neural-AI canonicalizes a private clamped Conv producer for raw AFU Add")
{
    auto input = CreateTensor("input", Shape(1, 2, 2, 32), DataType::Int8);
    auto convOfm = CreateTensor("conv", Shape(1, 2, 2, 32), DataType::Int8);
    auto skip = CreateTensor("skip", Shape(1, 2, 2, 32), DataType::Int8);
    auto output = CreateTensor("output", Shape(1, 2, 2, 32), DataType::Int8);
    auto conv = CreateOperation(
        OpType::Conv2D, TensorUsage::IFM0, input, TensorUsage::OFM, convOfm);
    auto add = CreateOperation(OpType::Add, TensorUsage::IFM0, convOfm,
        TensorUsage::IFM1, skip, TensorUsage::OFM, output);

    Quantization producerQuant;
    producerQuant.scales = {QuantizedScale::Unit()};
    producerQuant.zeroPoints = {10};
    producerQuant.quantMin = {10};
    producerQuant.quantMax = {100};
    conv->Output(TensorUsage::OFM)->Set(producerQuant);
    Quantization convAddQuant = producerQuant;
    convAddQuant.scales = {QuantizedScale(32768.0)};
    add->Input(TensorUsage::IFM0)->Set(convAddQuant);
    Quantization skipQuant;
    skipQuant.scales = {QuantizedScale(32768.0)};
    skipQuant.zeroPoints = {5};
    skipQuant.quantMin = {-128};
    skipQuant.quantMax = {127};
    add->Input(TensorUsage::IFM1)->Set(skipQuant);
    Quantization outputQuant;
    outputQuant.scales = {QuantizedScale(1.0 / 32768.0)};
    outputQuant.zeroPoints = {20};
    outputQuant.quantMin = {-128};
    outputQuant.quantMax = {127};
    add->Output(TensorUsage::OFM)->Set(outputQuant);

    std::vector<std::shared_ptr<Operation>> operations = {conv, add};
    auto graph = CreateGraph(operations);
    GraphOptimiserOptions options;
    ArchNeuralAI architecture;
    NeuralAIGraphOptimiser optimiser(architecture.Constraints(), options, nullptr);
    optimiser.OptimiseGraph(graph.get());

    REQUIRE(conv->Output(TensorUsage::OFM)->quantization.zeroPoints[0] == 15);
    REQUIRE(conv->Output(TensorUsage::OFM)->quantization.quantMin[0] == 15);
    REQUIRE(conv->Output(TensorUsage::OFM)->quantization.quantMax[0] == 105);
    REQUIRE(add->Input(TensorUsage::IFM0)->quantization.zeroPoints[0] == 15);
    REQUIRE(add->Input(TensorUsage::IFM0)->quantization.zeroPoints[0] +
        add->Input(TensorUsage::IFM1)->quantization.zeroPoints[0] ==
        add->Output(TensorUsage::OFM)->quantization.zeroPoints[0]);
}

TEST_CASE("Neural-AI does not canonicalize Conv Add when AFU would lose the output clamp")
{
    auto input = CreateTensor("input", Shape(1, 2, 2, 32), DataType::Int8);
    auto convOfm = CreateTensor("conv", Shape(1, 2, 2, 32), DataType::Int8);
    auto skip = CreateTensor("skip", Shape(1, 2, 2, 32), DataType::Int8);
    auto output = CreateTensor("output", Shape(1, 2, 2, 32), DataType::Int8);
    auto conv = CreateOperation(
        OpType::Conv2D, TensorUsage::IFM0, input, TensorUsage::OFM, convOfm);
    auto add = CreateOperation(OpType::Add, TensorUsage::IFM0, convOfm,
        TensorUsage::IFM1, skip, TensorUsage::OFM, output);

    Quantization producerQuant;
    producerQuant.scales = {QuantizedScale::Unit()};
    producerQuant.zeroPoints = {10};
    producerQuant.quantMin = {10};
    producerQuant.quantMax = {100};
    conv->Output(TensorUsage::OFM)->Set(producerQuant);
    Quantization convAddQuant = producerQuant;
    convAddQuant.scales = {QuantizedScale(32768.0)};
    add->Input(TensorUsage::IFM0)->Set(convAddQuant);
    Quantization skipQuant;
    skipQuant.scales = {QuantizedScale(32768.0)};
    skipQuant.zeroPoints = {5};
    add->Input(TensorUsage::IFM1)->Set(skipQuant);
    Quantization outputQuant;
    outputQuant.scales = {QuantizedScale(1.0 / 32768.0)};
    outputQuant.zeroPoints = {20};
    outputQuant.quantMin = {0};
    outputQuant.quantMax = {127};
    add->Output(TensorUsage::OFM)->Set(outputQuant);

    std::vector<std::shared_ptr<Operation>> operations = {conv, add};
    auto graph = CreateGraph(operations);
    GraphOptimiserOptions options;
    ArchNeuralAI architecture;
    NeuralAIGraphOptimiser optimiser(architecture.Constraints(), options, nullptr);
    optimiser.OptimiseGraph(graph.get());

    REQUIRE(conv->Output(TensorUsage::OFM)->quantization.zeroPoints[0] == 10);
    REQUIRE(add->Input(TensorUsage::IFM0)->quantization.zeroPoints[0] == 10);
}

TEST_CASE("Neural-AI compiler lowers INT8 Sigmoid through a raw-byte AFU LUT")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildSigmoidModel();
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);

    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    const uint32_t constantsOffset = Read32(data + 96 + 8);
    const uint32_t constantsBytes = Read32(data + 96 + 12);
    uint32_t offset = 224;
    uint32_t lutCommands = 0;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t type = Read16(data + offset);
        const uint16_t commandSize = Read16(data + offset + 2);
        REQUIRE(commandSize >= 32);
        if ( type == uint16_t(neuralai::CommandType::AFULut) )
        {
            REQUIRE(commandSize == sizeof(neuralai::CommandAFULutV2));
            REQUIRE(Read16(data + offset + 16) == uint16_t(neuralai::Region::TCDMScratch));
            REQUIRE(Read16(data + offset + 24) == uint16_t(neuralai::Region::TCDMScratch));
            REQUIRE(Read16(data + offset + 32) == uint16_t(neuralai::Region::ModelConstants));
            const uint32_t ifmOffset = Read32(data + offset + 20);
            const uint32_t ofmOffset = Read32(data + offset + 28);
            const uint32_t lutOffset = Read32(data + offset + 36);
            REQUIRE((ifmOffset + 384 <= ofmOffset || ofmOffset + 384 <= ifmOffset));
            REQUIRE(Read32(data + offset + 40) == 384);
            REQUIRE(lutOffset <= constantsBytes);
            REQUIRE(256 <= constantsBytes - lutOffset);
            for ( uint32_t raw = 0; raw < 256; ++raw )
            {
                const int32_t input = raw < 128 ? int32_t(raw) : int32_t(raw) - 256;
                const double real = std::max(-8.0, std::min(8.0, 0.125 * double(input)));
                const double sigmoid = 1.0 / (1.0 + std::exp(-real));
                const int32_t quantized = std::max(-128,
                    std::min(127, int32_t(std::round(-128.0 + 256.0 * sigmoid))));
                REQUIRE(data[constantsOffset + lutOffset + raw] == uint8_t(quantized));
            }
            ++lutCommands;
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(lutCommands == 1);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler fuses quantized YOLO SiLU into one AFU LUT")
{
    for ( const bool reverseMulInputs : {false, true} )
    {
        INFO("reverseMulInputs=" << reverseMulInputs);
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        const auto model = BuildSiluModel(reverseMulInputs);
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        const bool compiled = compiler.Compile();
        INFO(compiler.LastError());
        REQUIRE(compiled);

        IRegorBlob *blob = compiler.Output();
        REQUIRE(blob != nullptr);
        int64_t size = 0;
        const auto *data = static_cast<const uint8_t *>(blob->Map(size));
        const uint32_t commandBytes = Read32(data + 64 + 12);
        const uint32_t constantsOffset = Read32(data + 96 + 8);
        const uint32_t constantsBytes = Read32(data + 96 + 12);
        uint32_t offset = 224;
        uint32_t lutCommands = 0;
        while ( offset < 224 + commandBytes )
        {
            const uint16_t type = Read16(data + offset);
            const uint16_t commandSize = Read16(data + offset + 2);
            REQUIRE(commandSize >= 32);
            REQUIRE(type != uint16_t(neuralai::CommandType::AFUBinary));
            REQUIRE(type != uint16_t(neuralai::CommandType::SpatzMul));
            if ( type == uint16_t(neuralai::CommandType::AFULut) )
            {
                REQUIRE(commandSize == sizeof(neuralai::CommandAFULutV2));
                REQUIRE(Read16(data + offset + 16) == uint16_t(neuralai::Region::TCDMScratch));
                REQUIRE(Read16(data + offset + 24) == uint16_t(neuralai::Region::TCDMScratch));
                REQUIRE(Read16(data + offset + 32) == uint16_t(neuralai::Region::ModelConstants));
                const uint32_t ifmOffset = Read32(data + offset + 20);
                const uint32_t ofmOffset = Read32(data + offset + 28);
                const uint32_t lutOffset = Read32(data + offset + 36);
                REQUIRE((ifmOffset + 384 <= ofmOffset || ofmOffset + 384 <= ifmOffset));
                REQUIRE(Read32(data + offset + 40) == 384);
                REQUIRE(lutOffset <= constantsBytes);
                REQUIRE(256 <= constantsBytes - lutOffset);
                for ( uint32_t raw = 0; raw < 256; ++raw )
                {
                    const int input = raw < 128 ? int(raw) : int(raw) - 256;
                    const double real = std::clamp(0.125 * double(input), -8.0, 8.0);
                    const double sigmoid = 1.0 / (1.0 + std::exp(-real));
                    const int quantizedSigmoid = std::clamp(
                        int(std::round(-128.0 + 256.0 * sigmoid)), -128, 127);
                    const int product = input * (quantizedSigmoid + 128);
                    const int rounded = product >= 0 ?
                        (product + 64) / 128 : -((-product + 64) / 128);
                    const int expected = std::clamp(rounded, -128, 127);
                    REQUIRE(data[constantsOffset + lutOffset + raw] == uint8_t(expected));
                }
                ++lutCommands;
            }
            offset += commandSize;
        }
        REQUIRE(offset == 224 + commandBytes);
        REQUIRE(lutCommands == 1);
        blob->Unmap(const_cast<uint8_t *>(data));
        blob->Release();
    }
}

TEST_CASE("Neural-AI compiler lowers standalone INT8 ReLU6 through an AFU LUT")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildClippingModel(tflite::BuiltinOperator::RELU6);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);

    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    const uint32_t constantsOffset = Read32(data + 96 + 8);
    const uint32_t constantsBytes = Read32(data + 96 + 12);
    uint32_t offset = 224;
    uint32_t lutCommands = 0;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t type = Read16(data + offset);
        const uint16_t commandSize = Read16(data + offset + 2);
        REQUIRE(commandSize >= 32);
        if ( type == uint16_t(neuralai::CommandType::AFULut) )
        {
            REQUIRE(commandSize == sizeof(neuralai::CommandAFULutV2));
            const uint32_t lutOffset = Read32(data + offset + 36);
            REQUIRE(Read32(data + offset + 40) == 384);
            REQUIRE(lutOffset <= constantsBytes);
            REQUIRE(256 <= constantsBytes - lutOffset);
            for ( uint32_t raw = 0; raw < 256; ++raw )
            {
                const int32_t input = raw < 128 ? int32_t(raw) : int32_t(raw) - 256;
                const double real = 0.25 * double(input + 3);
                const double clamped = std::max(0.0, std::min(6.0, real));
                const int32_t quantized = std::max(-128,
                    std::min(127, int32_t(std::round(-3.0 + clamped / 0.25))));
                REQUIRE(data[constantsOffset + lutOffset + raw] == uint8_t(quantized));
            }
            ++lutCommands;
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(lutCommands == 1);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler lowers full-spatial AvgPool through AFU C32 reduction")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildGlobalAvgPoolModel();
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);

    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    uint32_t avgPoolCommands = 0;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t type = Read16(data + offset);
        const uint16_t commandSize = Read16(data + offset + 2);
        REQUIRE(commandSize >= 32);
        if ( type == uint16_t(neuralai::CommandType::AFUGlobalAvgPool) )
        {
            REQUIRE(commandSize == sizeof(neuralai::CommandAFUGlobalAvgPoolV2));
            REQUIRE(Read16(data + offset + 16) == uint16_t(neuralai::Region::TCDMScratch));
            REQUIRE(Read16(data + offset + 24) == uint16_t(neuralai::Region::TCDMScratch));
            REQUIRE(Read32(data + offset + 32) == 2);
            REQUIRE(Read32(data + offset + 36) == 3);
            REQUIRE(Read32(data + offset + 40) == 33);
            ++avgPoolCommands;
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(avgPoolCommands == 1);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler rejects non-global or requantized AvgPool")
{
    const auto rejects = [](flatbuffers::DetachedBuffer model)
    {
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        bool rejected = false;
        try
        {
            rejected = !compiler.Compile();
        }
        catch ( const std::runtime_error & )
        {
            rejected = true;
        }
        REQUIRE(rejected);
    };

    rejects(BuildGlobalAvgPoolModel(2, 3, 33, 0.25f, 0.5f));
    rejects(BuildGlobalAvgPoolModel(2, 3, 33, 0.25f, 0.25f, -3,
        tflite::ActivationFunctionType::RELU));
    rejects(BuildGlobalAvgPoolModel(2, 3, 33, 0.25f, 0.25f, -3,
        tflite::ActivationFunctionType::NONE, 1, 3));
}

TEST_CASE("Neural-AI compiler admits MobileNet-scale global AvgPool")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildGlobalAvgPoolModel(24, 24, 64);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);

    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    bool sawGlobalAverage = false;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t type = Read16(data + offset);
        const uint16_t commandSize = Read16(data + offset + 2);
        if ( type == uint16_t(neuralai::CommandType::AFUGlobalAvgPool) )
        {
            REQUIRE(Read32(data + offset + 32) == 24);
            REQUIRE(Read32(data + offset + 36) == 24);
            REQUIRE(Read32(data + offset + 40) == 64);
            sawGlobalAverage = true;
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(sawGlobalAverage);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler lowers selected nearest 2x C32-grouped resize depths")
{
    for ( const int channels : {32, 128, 256} )
    {
        DYNAMIC_SECTION("channels=" << channels)
        {
            std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
            Compiler compiler(architecture);
            const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
            REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
            const auto model = BuildResizeNearestModel(2, 3, channels, 4, 6);
            REQUIRE(compiler.LoadTflite(model.data(), model.size()));
            const bool compiled = compiler.Compile();
            INFO(compiler.LastError());
            REQUIRE(compiled);

            IRegorBlob *blob = compiler.Output();
            REQUIRE(blob != nullptr);
            int64_t size = 0;
            const auto *data = static_cast<const uint8_t *>(blob->Map(size));
            const uint32_t commandBytes = Read32(data + 64 + 12);
            uint32_t offset = 224;
            uint32_t upsampleCommands = 0;
            while ( offset < 224 + commandBytes )
            {
                const uint16_t type = Read16(data + offset);
                const uint16_t commandSize = Read16(data + offset + 2);
                REQUIRE(commandSize >= 32);
                if ( type == uint16_t(neuralai::CommandType::UpsampleNearest) )
                {
                    REQUIRE(commandSize == sizeof(neuralai::CommandUpsampleNearestV2));
                    REQUIRE(Read16(data + offset + 16) == uint16_t(neuralai::Region::TCDMScratch));
                    REQUIRE(Read16(data + offset + 24) == uint16_t(neuralai::Region::TCDMScratch));
                    REQUIRE(Read32(data + offset + 16 + 4) != Read32(data + offset + 24 + 4));
                    REQUIRE(Read32(data + offset + 32) == 2);
                    REQUIRE(Read32(data + offset + 36) == 3);
                    REQUIRE(Read32(data + offset + 40) == uint32_t(channels));
                    REQUIRE(Read32(data + offset + 44) == 2);
                    REQUIRE(Read32(data + offset + 48) == 2);
                    ++upsampleCommands;
                }
                offset += commandSize;
            }
            REQUIRE(offset == 224 + commandBytes);
            REQUIRE(upsampleCommands == 1);
            blob->Unmap(const_cast<uint8_t *>(data));
            blob->Release();
        }
    }
}

TEST_CASE("Neural-AI compiler rejects resize outside nearest 2x C32 contract")
{
    const auto rejects = [](flatbuffers::DetachedBuffer model)
    {
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        bool rejected = false;
        try
        {
            rejected = !compiler.Compile();
        }
        catch ( const std::runtime_error & )
        {
            rejected = true;
        }
        REQUIRE(rejected);
    };

    rejects(BuildResizeNearestModel(2, 3, 33, 4, 6));
    rejects(BuildResizeNearestModel(2, 3, 64, 4, 6));
    rejects(BuildResizeNearestModel(2, 3, 32, 6, 9));
    rejects(BuildResizeNearestModel(2, 3, 32, 4, 6, 0.5f));
}

TEST_CASE("Neural-AI compiler lowers selected K5 S1 P2 C32-grouped MaxPool depths")
{
    for ( const int channels : {32, 128} )
    {
        DYNAMIC_SECTION("channels=" << channels)
        {
            std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
            Compiler compiler(architecture);
            const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
            REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
            const auto model = BuildMaxPoolModel(10, 10, channels);
            REQUIRE(compiler.LoadTflite(model.data(), model.size()));
            const bool compiled = compiler.Compile();
            INFO(compiler.LastError());
            REQUIRE(compiled);

            IRegorBlob *blob = compiler.Output();
            REQUIRE(blob != nullptr);
            int64_t size = 0;
            const auto *data = static_cast<const uint8_t *>(blob->Map(size));
            const uint32_t commandBytes = Read32(data + 64 + 12);
            uint32_t offset = 224;
            uint32_t maxPoolCommands = 0;
            while ( offset < 224 + commandBytes )
            {
                const uint16_t type = Read16(data + offset);
                const uint16_t commandSize = Read16(data + offset + 2);
                REQUIRE(commandSize >= 32);
                if ( type == uint16_t(neuralai::CommandType::MaxPool) )
                {
                    REQUIRE(commandSize == sizeof(neuralai::CommandMaxPoolV2));
                    REQUIRE(Read16(data + offset + 16) == uint16_t(neuralai::Region::TCDMScratch));
                    REQUIRE(Read16(data + offset + 24) == uint16_t(neuralai::Region::TCDMScratch));
                    REQUIRE(Read32(data + offset + 20) != Read32(data + offset + 28));
                    REQUIRE(Read32(data + offset + 32) == 10);
                    REQUIRE(Read32(data + offset + 36) == 10);
                    REQUIRE(Read32(data + offset + 40) == uint32_t(channels));
                    REQUIRE(Read32(data + offset + 44) == 5);
                    REQUIRE(Read32(data + offset + 48) == 5);
                    REQUIRE(Read32(data + offset + 52) == 1);
                    REQUIRE(Read32(data + offset + 56) == 1);
                    REQUIRE(Read32(data + offset + 60) == 2);
                    REQUIRE(Read32(data + offset + 64) == 2);
                    ++maxPoolCommands;
                }
                offset += commandSize;
            }
            REQUIRE(offset == 224 + commandBytes);
            REQUIRE(maxPoolCommands == 1);
            blob->Unmap(const_cast<uint8_t *>(data));
            blob->Release();
        }
    }
}

TEST_CASE("Neural-AI compiler rejects MaxPool outside K5 S1 P2 C32 contract")
{
    const auto rejects = [](flatbuffers::DetachedBuffer model)
    {
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        bool rejected = false;
        try
        {
            rejected = !compiler.Compile();
        }
        catch ( const std::runtime_error & )
        {
            rejected = true;
        }
        REQUIRE(rejected);
    };

    rejects(BuildMaxPoolModel(4, 4, 33));
    rejects(BuildMaxPoolModel(4, 4, 64));
    rejects(BuildMaxPoolModel(4, 4, 32, 3, 3));
    rejects(BuildMaxPoolModel(4, 4, 32, 5, 5, 0.5f));
    rejects(BuildMaxPoolModel(4, 4, 32, 5, 5, 0.25f,
        tflite::ActivationFunctionType::RELU));
}

TEST_CASE("Neural-AI compiler materializes two-input C32 channel Concat with DMA")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildConcatModel();
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);

    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    uint32_t concatCopies = 0;
    std::vector<uint32_t> destinationOffsets;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t type = Read16(data + offset);
        const uint16_t commandSize = Read16(data + offset + 2);
        REQUIRE(commandSize >= 32);
        if ( type == uint16_t(neuralai::CommandType::DMA1D) &&
             Read16(data + offset + 16) == uint16_t(neuralai::Region::TCDMScratch) &&
             Read16(data + offset + 24) == uint16_t(neuralai::Region::TCDMScratch) )
        {
            REQUIRE(commandSize == sizeof(neuralai::CommandDMA1DV2));
            REQUIRE(Read32(data + offset + 32) == 2 * 3 * 32);
            REQUIRE(Read32(data + offset + 36) == uint32_t(neuralai::DMADirection::LocalToLocal));
            destinationOffsets.push_back(Read32(data + offset + 28));
            ++concatCopies;
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(concatCopies == 2);
    std::sort(destinationOffsets.begin(), destinationOffsets.end());
    REQUIRE(destinationOffsets[1] - destinationOffsets[0] == 2 * 3 * 32);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler gathers structural three-way C16 Concat with DMA3D")
{
    for ( const Point2i spatial : {Point2i(11, 7), Point2i(13, 9)} )
    {
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        const auto model = BuildConcatModel(
            16, 16, 3, 0.25f, spatial.y, spatial.x, 16);
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        const bool compiled = compiler.Compile();
        INFO(compiler.LastError());
        REQUIRE(compiled);

        IRegorBlob *blob = compiler.Output();
        REQUIRE(blob != nullptr);
        int64_t size = 0;
        const auto *data = static_cast<const uint8_t *>(blob->Map(size));
        const uint32_t commandBytes = Read32(data + 64 + 12);
        uint32_t offset = 224;
        uint32_t gatherCommands = 0;
        while ( offset < 224 + commandBytes )
        {
            const uint16_t commandSize = Read16(data + offset + 2);
            REQUIRE(commandSize >= 32);
            if ( Read16(data + offset) == uint16_t(neuralai::CommandType::DMA3D) &&
                 Read32(data + offset + 60) == uint32_t(neuralai::DMADirection::LocalToLocal) )
            {
                REQUIRE(commandSize == sizeof(neuralai::CommandDMA3DV2));
                REQUIRE(Read32(data + offset + 32) == 16);
                REQUIRE(Read32(data + offset + 44) == uint32_t(spatial.x));
                REQUIRE(Read32(data + offset + 56) == uint32_t(spatial.y));
                ++gatherCommands;
            }
            offset += commandSize;
        }
        REQUIRE(offset == 224 + commandBytes);
        REQUIRE(gatherCommands == 3);
        blob->Unmap(const_cast<uint8_t *>(data));
        blob->Release();
    }
}

TEST_CASE("Neural-AI compiler rejects Concat outside two-input C32 channel contract")
{
    const auto rejects = [](flatbuffers::DetachedBuffer model)
    {
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        bool rejected = false;
        try
        {
            rejected = !compiler.Compile();
        }
        catch ( const std::runtime_error & )
        {
            rejected = true;
        }
        REQUIRE(rejected);
    };

    rejects(BuildConcatModel(31, 32));
    rejects(BuildConcatModel(32, 32, 2));
    rejects(BuildConcatModel(32, 32, 3, 0.5f));
}

TEST_CASE("Neural-AI compiler materializes selected YOLO C64+C80 head Concat")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildConcatModel(64, 80, 3, 0.25f, 10, 10);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);

    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    std::vector<uint32_t> localCopyBytes;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t commandSize = Read16(data + offset + 2);
        if ( Read16(data + offset) == uint16_t(neuralai::CommandType::DMA1D) &&
             Read32(data + offset + 36) == uint32_t(neuralai::DMADirection::LocalToLocal) )
            localCopyBytes.push_back(Read32(data + offset + 32));
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    std::sort(localCopyBytes.begin(), localCopyBytes.end());
    REQUIRE(localCopyBytes == std::vector<uint32_t>{6400, 9600});
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler emits vector head pack only for selected C144 NCHW transpose")
{
    const auto compile = [](flatbuffers::DetachedBuffer model, bool expected)
    {
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        if ( expected )
        {
            Graph *graph = compiler.GetGraph("main");
            REQUIRE(graph != nullptr);
            std::vector<std::shared_ptr<Operation>> operations;
            graph->GetAllOperations(operations);
            REQUIRE(operations.size() == 1);
            INFO(OpTypeToString(operations[0]->Type()));
            REQUIRE(operations[0]->Type() == OpType::Transpose);
            auto supported = MakeSupportedOpsChecker(REGOR_ARCH_NEURALAI);
            REQUIRE(supported->Check(operations[0].get()));
        }
        bool compiled = false;
        std::string exceptionMessage;
        try
        {
            compiled = compiler.Compile();
        }
        catch ( const std::runtime_error &exception )
        {
            exceptionMessage = exception.what();
        }
        INFO(exceptionMessage);
        INFO(compiler.LastError());
        REQUIRE(compiled == expected);
        if ( !compiled ) return;

        IRegorBlob *blob = compiler.Output();
        REQUIRE(blob != nullptr);
        int64_t size = 0;
        const auto *data = static_cast<const uint8_t *>(blob->Map(size));
        const uint32_t commandBytes = Read32(data + 64 + 12);
        uint32_t offset = 224;
        uint32_t headPacks = 0;
        while ( offset < 224 + commandBytes )
        {
            const uint16_t commandSize = Read16(data + offset + 2);
            if ( Read16(data + offset) == uint16_t(neuralai::CommandType::CopyLayout) &&
                 Read16(data + offset + 32) == uint16_t(neuralai::CopyLayoutMode::C32ToCHW) )
            {
                REQUIRE(Read16(data + offset + 34) == uint16_t(neuralai::TensorLayout::C32Blocked));
                REQUIRE(Read16(data + offset + 36) == uint16_t(neuralai::TensorLayout::NHWC));
                REQUIRE(Read32(data + offset + 40) == 1);
                REQUIRE(Read32(data + offset + 44) == 10);
                REQUIRE(Read32(data + offset + 48) == 10);
                REQUIRE(Read32(data + offset + 52) == 144);
                REQUIRE(Read32(data + offset + 56) == 144);
                REQUIRE(Read32(data + offset + 60) == 10 * 10 * 32);
                REQUIRE(Read32(data + offset + 64) == 10 * 10);
                ++headPacks;
            }
            offset += commandSize;
        }
        REQUIRE(offset == 224 + commandBytes);
        REQUIRE(headPacks == 1);
        blob->Unmap(const_cast<uint8_t *>(data));
        blob->Release();
    };

    compile(BuildHeadTransposeModel(), true);
    compile(BuildHeadTransposeModel(10, 10, 128), false);
    compile(BuildHeadTransposeModel(10, 10, 144, {0, 2, 3, 1}), false);
    compile(BuildHeadTransposeModel(4, 16, 2100, {0, 1, 3, 2}), false);
}

TEST_CASE("Neural-AI compiler lowers the complete Micro-MobileNet topology")
{
    for ( int stage = 1; stage <= 13; ++stage )
    {
        std::unique_ptr<Architecture> prefixArchitecture = std::make_unique<ArchNeuralAI>();
        Compiler prefixCompiler(prefixArchitecture);
        const std::string prefixOptions = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(prefixCompiler.ParseOptions(prefixOptions.c_str(), prefixOptions.size()));
        const auto prefixModel = BuildMicroMobileNetModel(stage);
        REQUIRE(prefixCompiler.LoadTflite(prefixModel.data(), prefixModel.size()));
        INFO("Micro-MobileNet stage " << stage);
        REQUIRE(prefixCompiler.Compile());
    }

    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildMicroMobileNetModel();
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);

    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    uint32_t addCommands = 0;
    uint32_t averageCommands = 0;
    uint32_t convCommands = 0;
    while ( offset < 224 + commandBytes )
    {
        const auto type = neuralai::CommandType(Read16(data + offset));
        const uint16_t commandSize = Read16(data + offset + 2);
        REQUIRE(commandSize >= sizeof(neuralai::CommandHeaderV2));
        REQUIRE(type != neuralai::CommandType::AFULut);
        if ( type == neuralai::CommandType::AFUBinary ) ++addCommands;
        if ( type == neuralai::CommandType::AFUGlobalAvgPool ) ++averageCommands;
        if ( type == neuralai::CommandType::PointwiseC32 ||
             type == neuralai::CommandType::DepthwiseC32 ||
             type == neuralai::CommandType::LineBufferJob )
            ++convCommands;
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(addCommands == 2);
    REQUIRE(averageCommands == 1);
    REQUIRE(convCommands > 10);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler removes internal reshape-like views without adding commands")
{
    for ( const tflite::BuiltinOperator viewOperator : {
              tflite::BuiltinOperator::RESHAPE,
              tflite::BuiltinOperator::SQUEEZE,
              tflite::BuiltinOperator::EXPAND_DIMS,
          } )
    {
        INFO("viewOperator=" << int(viewOperator));
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        const auto model = BuildViewModel(viewOperator, true);
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        const bool compiled = compiler.Compile();
        INFO(compiler.LastError());
        REQUIRE(compiled);

        IRegorBlob *blob = compiler.Output();
        REQUIRE(blob != nullptr);
        int64_t size = 0;
        const auto *data = static_cast<const uint8_t *>(blob->Map(size));
        REQUIRE(Read32(data + 32) == 4);
        const uint32_t commandBytes = Read32(data + 64 + 12);
        uint32_t offset = 224;
        uint32_t copyCommands = 0;
        uint32_t addCommands = 0;
        while ( offset < 224 + commandBytes )
        {
            const uint16_t type = Read16(data + offset);
            const uint16_t commandSize = Read16(data + offset + 2);
            REQUIRE(commandSize >= 32);
            if ( type == uint16_t(neuralai::CommandType::CopyLayout) ) ++copyCommands;
            if ( type == uint16_t(neuralai::CommandType::AFUBinary) ) ++addCommands;
            offset += commandSize;
        }
        REQUIRE(offset == 224 + commandBytes);
        REQUIRE(copyCommands == 3);
        REQUIRE(addCommands == 1);
        blob->Unmap(const_cast<uint8_t *>(data));
        blob->Release();
    }
}

TEST_CASE("Neural-AI compiler aliases C32-aligned YOLO depth slices into pointwise Conv")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildC32SliceConvModel(2, 3, 64, 32, 32, 32);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);

    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    uint32_t producerOutputOffset = 0;
    uint32_t consumerInputOffset = 0;
    uint32_t copyCommands = 0;
    uint32_t pointwiseCommands = 0;
    while ( offset < 224 + commandBytes )
    {
        const auto type = neuralai::CommandType(Read16(data + offset));
        const uint16_t commandSize = Read16(data + offset + 2);
        REQUIRE(commandSize >= sizeof(neuralai::CommandHeaderV2));
        if ( type == neuralai::CommandType::CopyLayout )
        {
            ++copyCommands;
        }
        if ( type == neuralai::CommandType::PointwiseC32 )
        {
            if ( pointwiseCommands == 1 ) producerOutputOffset = Read32(data + offset + 44);
            else if ( pointwiseCommands == 2 ) consumerInputOffset = Read32(data + offset + 28);
            ++pointwiseCommands;
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(copyCommands == 2);
    REQUIRE(pointwiseCommands == 3);
    REQUIRE(consumerInputOffset == producerOutputOffset);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler aliases C32-aligned YOLO depth slices into linebuffer Conv")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildC32SliceConvModel(2, 3, 64, 32, 32, 32, 3);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);

    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    uint32_t producerOutputOffset = 0;
    uint32_t linebufferInputOffset = 0;
    uint32_t pointwiseCommands = 0;
    uint32_t linebufferCommands = 0;
    while ( offset < 224 + commandBytes )
    {
        const auto type = neuralai::CommandType(Read16(data + offset));
        const uint16_t commandSize = Read16(data + offset + 2);
        REQUIRE(commandSize >= sizeof(neuralai::CommandHeaderV2));
        if ( type == neuralai::CommandType::PointwiseC32 )
        {
            if ( pointwiseCommands == 1 ) producerOutputOffset = Read32(data + offset + 44);
            ++pointwiseCommands;
        }
        if ( type == neuralai::CommandType::LineBufferJob )
        {
            if ( linebufferCommands == 0 ) linebufferInputOffset = Read32(data + offset + 16);
            ++linebufferCommands;
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(pointwiseCommands == 2);
    REQUIRE(linebufferCommands > 0);
    REQUIRE(linebufferInputOffset == producerOutputOffset);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler aliases the low C16 half of a C32 YOLO tensor")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildC32SliceConvModel(2, 3, 32, 0, 16, 32);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);
    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    std::vector<uint32_t> pointwiseInputs;
    std::vector<uint32_t> pointwiseOutputs;
    uint32_t localStridedCopies = 0;
    while ( offset < 224 + commandBytes )
    {
        const auto type = neuralai::CommandType(Read16(data + offset));
        const uint16_t commandSize = Read16(data + offset + 2);
        REQUIRE(commandSize >= sizeof(neuralai::CommandHeaderV2));
        if ( type == neuralai::CommandType::PointwiseC32 )
        {
            pointwiseInputs.push_back(Read32(data + offset + 28));
            pointwiseOutputs.push_back(Read32(data + offset + 44));
        }
        if ( type == neuralai::CommandType::DMA3D &&
             Read32(data + offset + 60) == uint32_t(neuralai::DMADirection::LocalToLocal) )
            ++localStridedCopies;
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(pointwiseInputs.size() == 2);
    REQUIRE(pointwiseOutputs.size() == 2);
    REQUIRE(pointwiseInputs[1] == pointwiseOutputs[0]);
    REQUIRE(localStridedCopies == 0);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler vector-materializes the high C16 half of a C32 YOLO tensor")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildC32SliceConvModel(2, 3, 32, 16, 16, 32);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);
    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    std::vector<uint32_t> pointwiseInputs;
    std::vector<uint32_t> pointwiseOutputs;
    uint32_t copySource = 0;
    uint32_t copyDestination = 0;
    uint32_t localStridedCopies = 0;
    while ( offset < 224 + commandBytes )
    {
        const auto type = neuralai::CommandType(Read16(data + offset));
        const uint16_t commandSize = Read16(data + offset + 2);
        REQUIRE(commandSize >= sizeof(neuralai::CommandHeaderV2));
        if ( type == neuralai::CommandType::PointwiseC32 )
        {
            pointwiseInputs.push_back(Read32(data + offset + 28));
            pointwiseOutputs.push_back(Read32(data + offset + 44));
        }
        if ( type == neuralai::CommandType::DMA3D &&
             Read32(data + offset + 60) == uint32_t(neuralai::DMADirection::LocalToLocal) )
        {
            REQUIRE(Read32(data + offset + 32) == 16);
            REQUIRE(Read32(data + offset + 36) == 32);
            REQUIRE(Read32(data + offset + 40) == 32);
            REQUIRE(Read32(data + offset + 44) == 3);
            REQUIRE(Read32(data + offset + 56) == 2);
            copySource = Read32(data + offset + 20);
            copyDestination = Read32(data + offset + 28);
            ++localStridedCopies;
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(pointwiseInputs.size() == 2);
    REQUIRE(pointwiseOutputs.size() == 2);
    REQUIRE(localStridedCopies == 1);
    REQUIRE(copySource == pointwiseOutputs[0] + 16);
    REQUIRE(pointwiseInputs[1] == copyDestination);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler rejects C16 slices outside the YOLO half-depth contract")
{
    const auto rejects = [](flatbuffers::DetachedBuffer model)
    {
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        REQUIRE_FALSE(compiler.Compile());
    };
    rejects(BuildC32SliceConvModel(2, 3, 32, 8, 16, 32));
    rejects(BuildC32SliceConvModel(2, 3, 64, 16, 16, 32));
}

TEST_CASE("Neural-AI compiler materializes reshape-like graph outputs")
{
    for ( const tflite::BuiltinOperator viewOperator : {
              tflite::BuiltinOperator::RESHAPE,
              tflite::BuiltinOperator::SQUEEZE,
              tflite::BuiltinOperator::EXPAND_DIMS,
          } )
    {
        INFO("viewOperator=" << int(viewOperator));
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        const auto model = BuildViewModel(viewOperator, false);
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        const bool compiled = compiler.Compile();
        INFO(compiler.LastError());
        REQUIRE(compiled);

        IRegorBlob *blob = compiler.Output();
        REQUIRE(blob != nullptr);
        int64_t size = 0;
        const auto *data = static_cast<const uint8_t *>(blob->Map(size));
        REQUIRE(Read32(data + 32) == 2);
        const uint32_t commandBytes = Read32(data + 64 + 12);
        REQUIRE(commandBytes == 160);
        REQUIRE(Read16(data + 224) == uint16_t(neuralai::CommandType::DMA1D));
        REQUIRE(Read32(data + 224 + 32) == 128);
        REQUIRE(Read32(data + 224 + 36) == 0);
        REQUIRE(Read16(data + 224 + 64) == uint16_t(neuralai::CommandType::DMA1D));
        REQUIRE(Read32(data + 224 + 64 + 32) == 128);
        REQUIRE(Read32(data + 224 + 64 + 36) == 1);
        REQUIRE(Read16(data + 224 + 128) == uint16_t(neuralai::CommandType::End));
        blob->Unmap(const_cast<uint8_t *>(data));
        blob->Release();
    }
}

TEST_CASE("Neural-AI compiler coalesces pointwise M stripes into one command")
{
    for ( const int width : {257, 511} )
    {
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        const auto model = BuildPointwiseConvModel(1, width, 32, 32);
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        const bool compiled = compiler.Compile();
        INFO(compiler.LastError());
        REQUIRE(compiled);
        IRegorBlob *blob = compiler.Output();
        REQUIRE(blob != nullptr);
        int64_t size = 0;
        const auto *data = static_cast<const uint8_t *>(blob->Map(size));
        const uint32_t commandBytes = Read32(data + 64 + 12);
        uint32_t offset = 224;
        uint32_t pointwiseCommands = 0;
        uint32_t rqLoads = 0;
        while ( offset < 224 + commandBytes )
        {
            const uint16_t type = uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
            const uint16_t commandSize = uint16_t(data[offset + 2]) |
                (uint16_t(data[offset + 3]) << 8);
            if ( type == uint16_t(neuralai::CommandType::PointwiseC32) )
            {
                ++pointwiseCommands;
                REQUIRE(Read32(data + offset + 48) > 0);
                REQUIRE(Read32(data + offset + 48) == uint32_t(width));
            }
            if ( type == uint16_t(neuralai::CommandType::RQLoad) ) ++rqLoads;
            offset += commandSize;
        }
        REQUIRE(offset == 224 + commandBytes);
        REQUIRE(pointwiseCommands == 1);
        REQUIRE(rqLoads == 1);
        blob->Unmap(const_cast<uint8_t *>(data));
        blob->Release();
    }
}

TEST_CASE("Neural-AI compiler fuses ReLU and ReLU6 into qparam clamps")
{
    const auto checkClamp = [](tflite::ActivationFunctionType activation, int32_t expectedMin,
                               int32_t expectedMax)
    {
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        const auto model = BuildPointwiseConvModel(1, 1, 32, 32, activation);
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        const bool compiled = compiler.Compile();
        INFO(compiler.LastError());
        REQUIRE(compiled);
        IRegorBlob *blob = compiler.Output();
        REQUIRE(blob != nullptr);
        int64_t size = 0;
        const auto *data = static_cast<const uint8_t *>(blob->Map(size));
        const uint32_t qparamOffset = Read32(data + 64 + 4 * 32 + 8);
        REQUIRE(int32_t(Read32(data + qparamOffset + 16)) == expectedMin);
        REQUIRE(int32_t(Read32(data + qparamOffset + 20)) == expectedMax);
        const uint32_t commandBytes = Read32(data + 64 + 12);
        uint32_t commandOffset = 224;
        while ( commandOffset < 224 + commandBytes )
        {
            REQUIRE(Read16(data + commandOffset) != uint16_t(neuralai::CommandType::AFULut));
            commandOffset += Read16(data + commandOffset + 2);
        }
        REQUIRE(commandOffset == 224 + commandBytes);
        blob->Unmap(const_cast<uint8_t *>(data));
        blob->Release();
    };

    checkClamp(tflite::ActivationFunctionType::RELU, 0, 127);
    checkClamp(tflite::ActivationFunctionType::RELU6, 0, 6);
}

TEST_CASE("Neural-AI compiler emits grouped linebuffer jobs for generic K3 Conv2D")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildK3ConvModel(24, 24, 96, 64, 1);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);
    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    REQUIRE(size > int64_t(sizeof(neuralai::ModelHeaderV1)));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    uint32_t linebufferJobs = 0;
    uint32_t firstAccumMode = 0xffffffffu;
    uint32_t finalAccumMode = 0xffffffffu;
    std::vector<uint32_t> accumModes;
    std::vector<uint32_t> weightOffsets;
    std::vector<uint32_t> psumOffsets;
    std::vector<uint32_t> ofmOffsets;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t type = uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
        const uint16_t commandSize = uint16_t(data[offset + 2]) |
            (uint16_t(data[offset + 3]) << 8);
        if ( type == uint16_t(neuralai::CommandType::LineBufferJob) )
        {
            REQUIRE(commandSize == 160);
            ++linebufferJobs;
            const uint32_t mode = Read32(data + offset + 16 + 80 + 20);
            if ( firstAccumMode == 0xffffffffu ) firstAccumMode = mode;
            finalAccumMode = mode;
            REQUIRE(Read32(data + offset + 16 + 80 + 16) <= 256);
            accumModes.push_back(mode);
            weightOffsets.push_back(Read32(data + offset + 16 + 80));
            psumOffsets.push_back(Read32(data + offset + 16 + 80 + 4));
            ofmOffsets.push_back(Read32(data + offset + 16 + 80 + 12));
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(linebufferJobs == 18);
    REQUIRE(firstAccumMode == 1);
    REQUIRE(finalAccumMode == 2);
    REQUIRE(weightOffsets.size() == 18);
    const uint32_t inputGroupWeightBytes = 9u * 32u * 32u;
    for ( int job = 0; job < int(weightOffsets.size()); job += 3 )
    {
        const int outputGroup = job / 9;
        REQUIRE(accumModes[job] == 1);
        REQUIRE(accumModes[job + 1] == 3);
        REQUIRE(accumModes[job + 2] == 2);
        REQUIRE(weightOffsets[job] ==
            weightOffsets.front() + uint32_t(outputGroup * 3) * inputGroupWeightBytes);
        REQUIRE(weightOffsets[job + 1] == weightOffsets[job] + inputGroupWeightBytes);
        REQUIRE(weightOffsets[job + 2] == weightOffsets[job + 1] + inputGroupWeightBytes);
        REQUIRE(psumOffsets[job] == psumOffsets[job + 1]);
        REQUIRE(psumOffsets[job] == psumOffsets[job + 2]);
        REQUIRE(ofmOffsets[job] == ofmOffsets[job + 1]);
        REQUIRE(ofmOffsets[job] == ofmOffsets[job + 2]);
    }
    REQUIRE(Read32(data + 36) >= weightOffsets.front() + 6u * 9u * 32u * 32u);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler stages a zero-padded C16 linebuffer group for YOLO K3 Conv")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildK3ConvModel(
        16, 16, 16, 32, 2, -126, tflite::Padding::SAME, 0.166790381f, 0.854145825f);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);
    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    uint32_t linebufferJobs = 0;
    uint32_t tailCopies = 0;
    uint32_t stagingCopies = 0;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t type = Read16(data + offset);
        const uint16_t commandSize = Read16(data + offset + 2);
        REQUIRE(commandSize >= 32);
        if ( type == uint16_t(neuralai::CommandType::LineBufferJob) )
        {
            REQUIRE(commandSize == sizeof(neuralai::CommandLineBufferJobV2));
            REQUIRE(Read16(data + offset + 24) == 16);
            REQUIRE(Read16(data + offset + 72) == 16);
            REQUIRE(Read16(data + offset + 52) == 1);
            REQUIRE(Read16(data + offset + 54) == 1);
            REQUIRE(Read32(data + offset + 80) == 1);
            const uint32_t expectedAccum = linebufferJobs == 0 ? 1u :
                (linebufferJobs == 8 ? 2u : 3u);
            REQUIRE(Read32(data + offset + 116) == expectedAccum);
            ++linebufferJobs;
        }
        if ( type == uint16_t(neuralai::CommandType::DMA3D) )
        {
            REQUIRE(commandSize == sizeof(neuralai::CommandDMA3DV2));
            REQUIRE(Read32(data + offset + 32) == 16);
            REQUIRE(Read32(data + offset + 40) == 32);
            ++tailCopies;
        }
        if ( type == uint16_t(neuralai::CommandType::DMA2D) &&
             Read16(data + offset + 16) == uint16_t(neuralai::Region::TCDMScratch) &&
             Read16(data + offset + 24) == uint16_t(neuralai::Region::TCDMScratch) )
            ++stagingCopies;
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(linebufferJobs == 9);
    REQUIRE(tailCopies == 0);
    REQUIRE(stagingCopies > 0);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler emits full and masked C32 groups for IC48 and IC80 K3 Conv")
{
    for ( const int inputChannels : {48, 80} )
    {
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        const auto model = BuildK3ConvModel(8, 8, inputChannels, 32, 1, 0, tflite::Padding::VALID);
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        REQUIRE(compiler.Compile());
        IRegorBlob *blob = compiler.Output();
        REQUIRE(blob != nullptr);
        int64_t size = 0;
        const auto *data = static_cast<const uint8_t *>(blob->Map(size));
        const uint32_t commandBytes = Read32(data + 64 + 12);
        uint32_t offset = 224;
        uint32_t linebufferJobs = 0;
        std::vector<uint32_t> accumModes;
        std::vector<uint16_t> validBytes;
        std::vector<uint16_t> groupStationary;
        std::vector<uint32_t> inputBases;
        std::vector<uint32_t> channelOffsets;
        while ( offset < 224 + commandBytes )
        {
            const uint16_t type = Read16(data + offset);
            const uint16_t commandSize = Read16(data + offset + 2);
            if ( type == uint16_t(neuralai::CommandType::LineBufferJob) )
            {
                REQUIRE(commandSize == sizeof(neuralai::CommandLineBufferJobV2));
                validBytes.push_back(Read16(data + offset + 16 + 56));
                groupStationary.push_back(Read16(data + offset + 16 + 54));
                accumModes.push_back(Read32(data + offset + 16 + 80 + 20));
                inputBases.push_back(Read32(data + offset + 16));
                channelOffsets.push_back(Read32(data + offset + 16 + 72));
                ++linebufferJobs;
            }
            offset += commandSize;
        }
        REQUIRE(offset == 224 + commandBytes);
        const uint32_t groups = uint32_t((inputChannels + 31) / 32);
        const uint32_t fullGroups = groups - 1;
        const uint32_t tailJobs = 9;
        REQUIRE(linebufferJobs == fullGroups + tailJobs);
        REQUIRE(validBytes.front() == 32);
        REQUIRE(validBytes.back() == 16);
        REQUIRE(groupStationary.front() == 1);
        REQUIRE(groupStationary.back() == 0);
        REQUIRE(accumModes.front() == 1);
        REQUIRE(accumModes.back() == 2);
        const uint32_t groupPlaneBytes = 8u * 8u * 32u;
        REQUIRE(channelOffsets.front() == groupPlaneBytes);
        for ( uint32_t group = 1; group < fullGroups; ++group )
        {
            REQUIRE(inputBases[group] == inputBases.front() + group * groupPlaneBytes);
            REQUIRE(channelOffsets[group] == groupPlaneBytes);
            REQUIRE(validBytes[group] == 32);
            REQUIRE(groupStationary[group] == 1);
            REQUIRE(accumModes[group] == 3);
        }
        const uint32_t tailStart = fullGroups;
        for ( uint32_t tap = 0; tap < tailJobs; ++tap )
        {
            const uint32_t tapH = tap / 3;
            const uint32_t tapW = tap % 3;
            REQUIRE(inputBases[tailStart + tap] == inputBases.front() + fullGroups * groupPlaneBytes +
                tapH * 8u * 32u + tapW * 32u);
            REQUIRE(channelOffsets[tailStart + tap] == 0u);
            REQUIRE(validBytes[tailStart + tap] == 16);
            REQUIRE(groupStationary[tailStart + tap] == 0);
            REQUIRE(accumModes[tailStart + tap] == (tap == 0 ? 3u :
                (tap == tailJobs - 1 ? 2u : 3u)));
        }
        blob->Unmap(const_cast<uint8_t *>(data));
        blob->Release();
    }
}

TEST_CASE("Neural-AI compiler splits a width-641 Conv into legal linebuffer jobs")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildK3ConvModel(1, 641, 32, 32, 1);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    REQUIRE(compiler.Compile());
    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    uint32_t jobs = 0;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t type = uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
        const uint16_t commandSize = uint16_t(data[offset + 2]) |
            (uint16_t(data[offset + 3]) << 8);
        if ( type == uint16_t(neuralai::CommandType::LineBufferJob) )
        {
            ++jobs;
            REQUIRE(Read32(data + offset + 16 + 80 + 16) <= 1024u);
            const uint16_t inputW = uint16_t(data[offset + 16 + 8]) |
                (uint16_t(data[offset + 16 + 9]) << 8);
            REQUIRE(inputW <= 640u);
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(jobs == 2);
    REQUIRE(Read32(data + 36) <= ArchNeuralAI::AllocatableTCDMBytes);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler reports full-size schedules that require spatial tiling")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildK3ConvModel(
        224, 224, 3, 32, 2, -1, tflite::Padding::SAME, 1.0f / 127.5f, 1.0f);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    REQUIRE_FALSE(compiler.Compile());
    REQUIRE(compiler.LastError().find("Neural-AI schedule requires") != std::string::npos);
    REQUIRE(compiler.LastError().find("spatial tiling or rolling-buffer scheduling is required") !=
        std::string::npos);
}

TEST_CASE("Neural-AI compiler consumes compact TCDM RGB input directly")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildK3ConvModel(96, 96, 3, 32, 2);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    REQUIRE(compiler.Compile());
    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    uint32_t linebufferJobs = 0;
    uint32_t copyLayouts = 0;
    uint32_t dma2dCommands = 0;
    uint32_t localCopies = 0;
    uint32_t inputCopies = 0;
    bool sawShortDma = false;
    bool sawWideM = false;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t type = uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
        const uint16_t commandSize = uint16_t(data[offset + 2]) |
            (uint16_t(data[offset + 3]) << 8);
        if ( type == uint16_t(neuralai::CommandType::LineBufferJob) )
        {
            ++linebufferJobs;
            const uint32_t rows = Read32(data + offset + 16 + 80 + 16);
            REQUIRE(rows <= 1024u);
            sawWideM |= rows > 256u;
        }
        if ( type == uint16_t(neuralai::CommandType::CopyLayout) ) ++copyLayouts;
        if ( type == uint16_t(neuralai::CommandType::DMA1D) )
        {
            localCopies += Read32(data + offset + 36) ==
                uint32_t(neuralai::DMADirection::LocalToLocal);
            inputCopies += Read16(data + offset + 16) ==
                uint16_t(neuralai::Region::InputBinding);
        }
        if ( type == uint16_t(neuralai::CommandType::DMA2D) )
        {
            ++dma2dCommands;
            const uint32_t length = Read32(data + offset + 32);
            sawShortDma |= length == 3;
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(linebufferJobs == 3);
    REQUIRE(copyLayouts == 1);
    REQUIRE(localCopies == 0);
    REQUIRE(inputCopies == 1);
    REQUIRE(dma2dCommands == 1);
    REQUIRE_FALSE(sawShortDma);
    REQUIRE(sawWideM);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler supports a partial output group for an RGB stem")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildK3ConvModel(
        18, 18, 3, 16, 2, -128, tflite::Padding::VALID, 1.0f / 255.0f, 0.345353007f);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    REQUIRE(compiler.Compile());
    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    uint32_t bindingOffset = 0;
    const uint32_t sectionCount = Read32(data + 20);
    const uint32_t sectionTable = Read32(data + 24);
    for ( uint32_t section = 0; section < sectionCount; ++section )
    {
        const uint32_t offset = sectionTable + section * sizeof(neuralai::SectionV1);
        if ( Read32(data + offset) == uint32_t(neuralai::SectionType::Bindings) )
            bindingOffset = Read32(data + offset + 8);
    }
    REQUIRE(bindingOffset != 0);
    REQUIRE(Read32(data + bindingOffset + 16) == 1);
    REQUIRE(Read32(data + bindingOffset + 20) == 18);
    REQUIRE(Read32(data + bindingOffset + 24) == 18);
    REQUIRE(Read32(data + bindingOffset + 28) == 3);
    REQUIRE(Read32(data + bindingOffset + 32) == 18 * 18 * 3);
    uint32_t inputScaleBits = 0;
    const float inputScale = 1.0f / 255.0f;
    std::memcpy(&inputScaleBits, &inputScale, sizeof(inputScaleBits));
    REQUIRE(Read32(data + bindingOffset + 36) == inputScaleBits);
    REQUIRE(int32_t(Read32(data + bindingOffset + 40)) == -128);
    uint32_t outputScaleBits = 0;
    const float outputScale = 0.345353007f;
    std::memcpy(&outputScaleBits, &outputScale, sizeof(outputScaleBits));
    REQUIRE(Read32(data + bindingOffset + sizeof(neuralai::BindingV1) + 36) == outputScaleBits);

    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    uint32_t linebufferJobs = 0;
    uint32_t inputCopies = 0;
    uint32_t stagedInput = 0;
    uint32_t linebufferInput = 0;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t type = uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
        const uint16_t commandSize = uint16_t(data[offset + 2]) |
            (uint16_t(data[offset + 3]) << 8);
        if ( type == uint16_t(neuralai::CommandType::DMA1D) &&
             Read16(data + offset + 16) == uint16_t(neuralai::Region::InputBinding) )
        {
            ++inputCopies;
            stagedInput = Read32(data + offset + 28);
            REQUIRE(Read32(data + offset + 32) == 18 * 18 * 3);
        }
        if ( type == uint16_t(neuralai::CommandType::LineBufferJob) )
        {
            ++linebufferJobs;
            linebufferInput = Read32(data + offset + 16);
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(linebufferJobs > 0);
    REQUIRE(inputCopies == 1);
    REQUIRE(linebufferInput == stagedInput);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler serializes asymmetric Conv padding at the raw zero point")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    constexpr int8_t inputZeroPoint = -3;
    const auto model = BuildK3ConvModel(16, 16, 3, 32, 2, inputZeroPoint);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    INFO(compiler.LastError());
    REQUIRE(compiler.Compile());

    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t sectionCount = Read32(data + 20);
    const uint32_t sectionTable = Read32(data + 24);
    uint32_t commandOffset = 0;
    uint32_t commandBytes = 0;
    uint32_t constantOffset = 0;
    uint32_t constantBytes = 0;
    for ( uint32_t section = 0; section < sectionCount; ++section )
    {
        const uint32_t offset = sectionTable + section * sizeof(neuralai::SectionV1);
        const uint32_t type = Read32(data + offset);
        if ( type == uint32_t(neuralai::SectionType::Commands) )
        {
            commandOffset = Read32(data + offset + 8);
            commandBytes = Read32(data + offset + 12);
        }
        else if ( type == uint32_t(neuralai::SectionType::Constants) )
        {
            constantOffset = Read32(data + offset + 8);
            constantBytes = Read32(data + offset + 12);
        }
    }
    REQUIRE(commandOffset != 0);
    REQUIRE(commandBytes != 0);
    REQUIRE(constantOffset != 0);
    REQUIRE(constantBytes != 0);

    uint32_t paddingFills = 0;
    uint32_t offset = commandOffset;
    while ( offset < commandOffset + commandBytes )
    {
        const uint16_t type = Read16(data + offset);
        const uint16_t commandSize = Read16(data + offset + 2);
        if ( type == uint16_t(neuralai::CommandType::DMA1D) ||
             type == uint16_t(neuralai::CommandType::DMA2D) )
        {
            const uint16_t sourceRegion = Read16(data + offset + 16);
            const uint32_t sourceOffset = Read32(data + offset + 20);
            const uint32_t length = Read32(data + offset + 32);
            if ( type == uint16_t(neuralai::CommandType::DMA2D) &&
                 sourceRegion == uint16_t(neuralai::Region::ModelConstants) &&
                 length == 32 && Read32(data + offset + 36) == 0 )
            {
                REQUIRE(sourceOffset + length <= constantBytes);
                for ( uint32_t index = 0; index < length; ++index )
                    REQUIRE(int8_t(data[constantOffset + sourceOffset + index]) == inputZeroPoint);
                REQUIRE(Read32(data + offset + 44) > 1);
                ++paddingFills;
            }
        }
        offset += commandSize;
    }
    REQUIRE(offset == commandOffset + commandBytes);
    REQUIRE(paddingFills == 1);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler emits group-scoped depthwise C32 commands")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildDepthwiseConvModel(8, 8, 33, 2);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);
    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    uint32_t depthwiseCommands = 0;
    uint32_t rqLoads = 0;
    bool sawTail = false;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t type = uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
        const uint16_t commandSize = uint16_t(data[offset + 2]) |
            (uint16_t(data[offset + 3]) << 8);
        if ( type == uint16_t(neuralai::CommandType::RQLoad) ) ++rqLoads;
        if ( type == uint16_t(neuralai::CommandType::DepthwiseC32) )
        {
            ++depthwiseCommands;
            REQUIRE(commandSize == 96);
            sawTail |= Read32(data + offset + 56) == 1;
            REQUIRE(Read32(data + offset + 56) <= 32);
            REQUIRE(Read32(data + offset + 68) == 0);
            REQUIRE(Read32(data + offset + 72) == 0);
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(depthwiseCommands == 2);
    REQUIRE(rqLoads == 2);
    REQUIRE(sawTail);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler copies an asymmetric public C32 group into padded depthwise storage")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildDepthwiseConvModel(8, 8, 32, 1, -128);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));
    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);

    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    const uint32_t commandBytes = Read32(data + 64 + 12);
    uint32_t offset = 224;
    bool sawInputRectangle = false;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t type = Read16(data + offset);
        const uint16_t commandSize = Read16(data + offset + 2);
        if ( type == uint16_t(neuralai::CommandType::DMA2D) &&
             Read16(data + offset + 16) == uint16_t(neuralai::Region::InputBinding) )
        {
            REQUIRE(Read32(data + offset + 32) == 8 * 32);
            REQUIRE(Read32(data + offset + 36) == 8 * 32);
            REQUIRE(Read32(data + offset + 40) == 10 * 32);
            REQUIRE(Read32(data + offset + 44) == 8);
            sawInputRectangle = true;
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(sawInputRectangle);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler copies multi-group compact NHWC into depthwise C32 storage")
{
    for ( const int channels : {64, 960} )
    {
        INFO("channels=" << channels);
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        const auto model = BuildDepthwiseConvModel(7, 7, channels, 1, -128);
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        REQUIRE(compiler.Compile());

        IRegorBlob *blob = compiler.Output();
        REQUIRE(blob != nullptr);
        int64_t size = 0;
        const auto *data = static_cast<const uint8_t *>(blob->Map(size));
        const uint32_t commandBytes = Read32(data + 64 + 12);
        uint32_t offset = 224;
        uint32_t inputDma2d = 0;
        uint32_t inputDma3d = 0;
        uint32_t previousGroupDestination = 0;
        while ( offset < 224 + commandBytes )
        {
            const uint16_t type = Read16(data + offset);
            const uint16_t commandSize = Read16(data + offset + 2);
            if ( type == uint16_t(neuralai::CommandType::DMA3D) &&
                 Read16(data + offset + 16) == uint16_t(neuralai::Region::InputBinding) )
            {
                REQUIRE(commandSize == sizeof(neuralai::CommandDMA3DV2));
                REQUIRE(Read32(data + offset + 32) == 32);
                REQUIRE(Read32(data + offset + 36) == uint32_t(channels));
                REQUIRE(Read32(data + offset + 40) == 32);
                REQUIRE(Read32(data + offset + 44) == 7);
                REQUIRE(Read32(data + offset + 48) == 7u * uint32_t(channels));
                REQUIRE(Read32(data + offset + 52) == 9u * 32u);
                REQUIRE(Read32(data + offset + 56) == 7);
                const uint32_t group = inputDma3d;
                const uint32_t sourceOffset = Read32(data + offset + 20);
                const uint32_t destinationOffset = Read32(data + offset + 28);
                REQUIRE(sourceOffset == group * 32u);
                if ( group != 0 )
                    REQUIRE(destinationOffset == previousGroupDestination + 9u * 9u * 32u);
                previousGroupDestination = destinationOffset;
                ++inputDma3d;
            }
            if ( type == uint16_t(neuralai::CommandType::DMA2D) &&
                 Read16(data + offset + 16) == uint16_t(neuralai::Region::InputBinding) )
            {
                ++inputDma2d;
            }
            offset += commandSize;
        }
        REQUIRE(offset == 224 + commandBytes);
        REQUIRE(inputDma2d == 0);
        REQUIRE(inputDma3d == uint32_t(channels / 32));
        blob->Unmap(const_cast<uint8_t *>(data));
        blob->Release();
    }
}

TEST_CASE("Neural-AI compiler emits all depthwise C32 group and tail variants")
{
    for ( const int channels : {31, 32, 33, 48, 64, 65, 96} )
    {
        INFO("channels=" << channels);
        std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
        Compiler compiler(architecture);
        const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
        REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
        const auto model = BuildDepthwiseConvModel(3, 3, channels, 2);
        REQUIRE(compiler.LoadTflite(model.data(), model.size()));
        REQUIRE(compiler.Compile());

        IRegorBlob *blob = compiler.Output();
        REQUIRE(blob != nullptr);
        int64_t size = 0;
        const auto *data = static_cast<const uint8_t *>(blob->Map(size));
        const uint32_t commandBytes = Read32(data + 64 + 12);
        uint32_t offset = 224;
        uint32_t depthwiseCommands = 0;
        uint32_t rqLoads = 0;
        while ( offset < 224 + commandBytes )
        {
            const uint16_t type = uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
            const uint16_t commandSize = uint16_t(data[offset + 2]) |
                (uint16_t(data[offset + 3]) << 8);
            if ( type == uint16_t(neuralai::CommandType::RQLoad) )
            {
                REQUIRE(Read32(data + offset + 24) == rqLoads);
                ++rqLoads;
            }
            if ( type == uint16_t(neuralai::CommandType::DepthwiseC32) )
            {
                ++depthwiseCommands;
                const uint32_t validChannels = Read32(data + offset + 56);
                REQUIRE(commandSize == 96);
                REQUIRE(validChannels > 0);
                REQUIRE(validChannels <= 32);
                REQUIRE(Read32(data + offset + 68) == 1);
                REQUIRE(Read32(data + offset + 72) == 1);
                if ( channels % 32 != 0 && depthwiseCommands == uint32_t((channels + 31) / 32) )
                    REQUIRE(validChannels == uint32_t(channels % 32));
            }
            offset += commandSize;
        }
        REQUIRE(offset == 224 + commandBytes);
        REQUIRE(depthwiseCommands == uint32_t((channels + 31) / 32));
        REQUIRE(rqLoads == depthwiseCommands);
        blob->Unmap(const_cast<uint8_t *>(data));
        blob->Release();
    }
}

TEST_CASE("Neural-AI op groups fuse one clipping activation only")
{
    ArchNeuralAI arch;
    ArchitectureOpGroupQuery query{};
    query.type = OpType::FullyConnected;
    ArchitectureOpGroupQuery activation{};
    activation.type = OpType::Relu6;

    auto group = arch.CreateOpGroup(query);
    REQUIRE(group);
    REQUIRE(group->NeedsAllocation(1));
    REQUIRE(group->Add(query) == 0);
    REQUIRE(group->Add(activation, {-1}) == -2);
    REQUIRE(group->Add(activation, {-1}) == 0);
    REQUIRE(arch.CreateOpGroup(activation) == nullptr);
}

TEST_CASE("Neural-AI GEMM weight packing follows K-lane N-lane tile order")
{
    constexpr int dimensions[][2] = {
        {1, 65}, {31, 64}, {32, 63}, {33, 34},
        {33, 1}, {63, 33}, {64, 32}, {65, 31},
    };
    for ( const auto &dimension : dimensions )
    {
        const int depthK = dimension[0];
        const int depthN = dimension[1];
        const int kGroups = (depthK + 31) / 32;
        const int nGroups = (depthN + 31) / 32;
        INFO("K=" << depthK << ", N=" << depthN);
        std::vector<int8_t> weights(depthK * depthN);
        for ( int k = 0; k < depthK; ++k )
        {
            for ( int n = 0; n < depthN; ++n )
            {
                weights[k * depthN + n] = int8_t((k * 17 + n * 3) & 0x7f);
            }
        }

        const auto packed = neuralai::PackGEMM32Weights(weights.data(), depthK, depthN);
        REQUIRE(packed.size() == size_t(kGroups * nGroups * 32 * 32));
        for ( int nGroup = 0; nGroup < nGroups; ++nGroup )
        {
            for ( int kGroup = 0; kGroup < kGroups; ++kGroup )
            {
                for ( int kLane = 0; kLane < 32; ++kLane )
                {
                    for ( int nLane = 0; nLane < 32; ++nLane )
                    {
                        const int k = kGroup * 32 + kLane;
                        const int n = nGroup * 32 + nLane;
                        const size_t index =
                            size_t(((nGroup * kGroups + kGroup) * 32 + kLane) * 32 + nLane);
                        const uint8_t expected =
                            k < depthK && n < depthN ? uint8_t(weights[k * depthN + n]) : 0;
                        REQUIRE(packed[index] == expected);
                    }
                }
            }
        }
    }
}

TEST_CASE("Neural-AI Regor weight encoder packs OHWI matrix constants")
{
    constexpr int depthK = 33;
    constexpr int depthN = 34;
    ArchNeuralAI arch;
    NeuralAIOpConfig opConfig;
    auto *encoder = arch.WeightEncoder();
    auto config = encoder->GetEncodingConfig(
        &opConfig, nullptr, DataType::Int8, Flags<WeightFormat>(WeightFormat::Default));
    auto source = encoder->GetWeightSource(config.get(), DataType::Int8, nullptr, nullptr);

    std::vector<int8_t> ohwi(depthN * depthK);
    std::vector<int8_t> matrixKN(depthK * depthN);
    for ( int n = 0; n < depthN; ++n )
    {
        for ( int k = 0; k < depthK; ++k )
        {
            const int8_t value = int8_t(((n * 11 + k * 5) & 0x7f) - 64);
            ohwi[n * depthK + k] = value;
            matrixKN[k * depthN + n] = value;
        }
    }
    source->SetSource(ohwi.data(), 0, Shape(depthN, 1, 1, depthK),
        Shape(depthK, depthK, depthK, 1), 0);

    std::vector<uint8_t> encoded;
    const WeightsInfo info = encoder->EncodeWeights(config.get(), source.get(), encoded);
    REQUIRE(info.sourceSize == depthK * depthN);
    REQUIRE(encoded == neuralai::PackGEMM32Weights(matrixKN.data(), depthK, depthN));
}

TEST_CASE("Neural-AI generic K3 weights are grouped by IC before kernel taps")
{
    constexpr int depthK = 64;
    constexpr int depthN = 32;
    constexpr int kernel = 3;
    ArchNeuralAI arch;
    NeuralAIOpConfig opConfig(256, NeuralAIOpMode::Conv2DLinebufC32S1Requant);
    auto *encoder = arch.WeightEncoder();
    auto config = encoder->GetEncodingConfig(
        &opConfig, nullptr, DataType::Int8, Flags<WeightFormat>(WeightFormat::Default));
    auto source = encoder->GetWeightSource(config.get(), DataType::Int8, nullptr, nullptr);
    std::vector<int8_t> weights(depthN * kernel * kernel * depthK);
    for ( int o = 0; o < depthN; ++o )
        for ( int h = 0; h < kernel; ++h )
            for ( int w = 0; w < kernel; ++w )
                for ( int i = 0; i < depthK; ++i )
                    weights[((o * kernel + h) * kernel + w) * depthK + i] =
                        int8_t((h * kernel + w) * 2 + (i / 32));
    source->SetSource(weights.data(), 0, Shape(depthN, kernel, kernel, depthK),
        Shape(kernel * kernel * depthK, kernel * depthK, depthK, 1), 0);
    std::vector<uint8_t> encoded;
    REQUIRE(encoder->EncodeWeights(config.get(), source.get(), encoded).encodedSize ==
        2 * 9 * 32 * 32);
    // Tile 1 is the second kernel tap of IC group 0, while tile 9 starts
    // kernel tap 0 of IC group 1.  A linear K pack would interleave these.
    REQUIRE(int8_t(encoded[0]) == 0);
    REQUIRE(int8_t(encoded[32 * 32]) == 2);
    REQUIRE(int8_t(encoded[8 * 32 * 32]) == 16);
    REQUIRE(int8_t(encoded[9 * 32 * 32]) == 1);
}

TEST_CASE("Neural-AI generic K3 weights zero-pad a C16 input group")
{
    constexpr int depthK = 16;
    constexpr int depthN = 32;
    constexpr int kernel = 3;
    ArchNeuralAI arch;
    NeuralAIOpConfig opConfig(256, NeuralAIOpMode::Conv2DLinebufC32S2Requant);
    auto *encoder = arch.WeightEncoder();
    auto config = encoder->GetEncodingConfig(
        &opConfig, nullptr, DataType::Int8, Flags<WeightFormat>(WeightFormat::Default));
    auto source = encoder->GetWeightSource(config.get(), DataType::Int8, nullptr, nullptr);
    std::vector<int8_t> weights(depthN * kernel * kernel * depthK);
    for ( int o = 0; o < depthN; ++o )
        for ( int h = 0; h < kernel; ++h )
            for ( int w = 0; w < kernel; ++w )
                for ( int i = 0; i < depthK; ++i )
                    weights[((o * kernel + h) * kernel + w) * depthK + i] =
                        int8_t((h * kernel + w) * depthK + i);
    source->SetSource(weights.data(), 0, Shape(depthN, kernel, kernel, depthK),
        Shape(kernel * kernel * depthK, kernel * depthK, depthK, 1), 0);
    std::vector<uint8_t> encoded;
    REQUIRE(encoder->EncodeWeights(config.get(), source.get(), encoded).encodedSize ==
        9 * 32 * 32);
    for ( int tap = 0; tap < 9; ++tap )
    {
        for ( int inputLane = 0; inputLane < 32; ++inputLane )
        {
            const int8_t expected = inputLane < depthK ?
                int8_t(tap * depthK + inputLane) : int8_t(0);
            REQUIRE(int8_t(encoded[(tap * 32 + inputLane) * 32]) == expected);
        }
    }
}

TEST_CASE("Neural-AI Regor weight encoder emits ABI qparam lanes")
{
    constexpr int channels = 3;
    ArchNeuralAI arch;
    NeuralAIOpConfig opConfig;
    auto *encoder = arch.WeightEncoder();
    auto config = encoder->GetEncodingConfig(
        &opConfig, nullptr, DataType::Int8, Flags<WeightFormat>(WeightFormat::Default));
    Quantization quantization;
    quantization.scales = {QuantizedScale(101, 3), QuantizedScale(202, 4), QuantizedScale(303, 5)};
    quantization.zeroPoints = {-7, 2, 11};
    quantization.quantMin = {-100};
    quantization.quantMax = {99};
    std::vector<int32_t> biases = {1001, -2002, 3003};
    auto source = encoder->GetScaleSource(config.get(), DataType::Int32, quantization);
    source->SetSource(biases.data(), channels, 0, channels, 0);

    std::vector<uint8_t> encoded;
    REQUIRE(encoder->EncodeScales(config.get(), source.get(), encoded, false) == 32 * 32);
    REQUIRE(encoded.size() == 32 * 32);
    for ( int channel = 0; channel < channels; ++channel )
    {
        const size_t offset = size_t(channel) * 32;
        REQUIRE(int32_t(Read32(encoded, offset)) == biases[channel]);
        REQUIRE(int32_t(Read32(encoded, offset + 4)) == quantization.scales[channel].scale);
        REQUIRE(Read32(encoded, offset + 8) == uint32_t(quantization.scales[channel].shift));
        REQUIRE(int32_t(Read32(encoded, offset + 12)) == quantization.zeroPoints[channel]);
        REQUIRE(int32_t(Read32(encoded, offset + 16)) == -100);
        REQUIRE(int32_t(Read32(encoded, offset + 20)) == 99);
    }
    REQUIRE(Read32(encoded, 3 * 32 + 4) == 0);
    REQUIRE(int32_t(Read32(encoded, 3 * 32 + 16)) == -100);
    REQUIRE(int32_t(Read32(encoded, 3 * 32 + 20)) == 99);
}

TEST_CASE("Neural-AI qparam clamps saturate to the INT8 output domain")
{
    ArchNeuralAI arch;
    NeuralAIOpConfig opConfig;
    auto *encoder = arch.WeightEncoder();
    auto config = encoder->GetEncodingConfig(
        &opConfig, nullptr, DataType::Int8, Flags<WeightFormat>(WeightFormat::Default));
    Quantization quantization;
    quantization.scales = {QuantizedScale(1, 0)};
    quantization.quantMin = {std::numeric_limits<int64_t>::min()};
    quantization.quantMax = {std::numeric_limits<int64_t>::max()};
    int32_t bias = 0;
    auto source = encoder->GetScaleSource(config.get(), DataType::Int32, quantization);
    source->SetSource(&bias, 1, 0, 1, 0);
    std::vector<uint8_t> encoded;
    REQUIRE(encoder->EncodeScales(config.get(), source.get(), encoded, false) == 32 * 32);
    REQUIRE(int32_t(Read32(encoded, 16)) == -128);
    REQUIRE(int32_t(Read32(encoded, 20)) == 127);

    quantization.quantMin = {10};
    quantization.quantMax = {-10};
    source = encoder->GetScaleSource(config.get(), DataType::Int32, quantization);
    source->SetSource(&bias, 1, 0, 1, 0);
    encoded.clear();
    REQUIRE_THROWS_AS(encoder->EncodeScales(config.get(), source.get(), encoded, false), WeightEncodeException);
}

TEST_CASE("Neural-AI rejects non-uniform clamp bounds within one C32 block")
{
    ArchNeuralAI arch;
    NeuralAIOpConfig opConfig;
    auto *encoder = arch.WeightEncoder();
    auto config = encoder->GetEncodingConfig(
        &opConfig, nullptr, DataType::Int8, Flags<WeightFormat>(WeightFormat::Default));
    Quantization quantization;
    quantization.scales = {QuantizedScale(1, 0)};
    quantization.quantMin = {0, -1};
    quantization.quantMax = {127, 127};
    int32_t bias = 0;
    auto source = encoder->GetScaleSource(config.get(), DataType::Int32, quantization);
    source->SetSource(&bias, 1, 0, 32, 0);
    std::vector<uint8_t> encoded;
    REQUIRE_THROWS_AS(encoder->EncodeScales(config.get(), source.get(), encoded, false), WeightEncodeException);
}
