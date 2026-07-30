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
#include "compiler/neural_ai_command_generator.hpp"
#include "compiler/neural_ai_graph_optimiser.hpp"
#include "compiler/neural_ai_writer.hpp"
#include "compiler/scheduler.hpp"
#include "compiler/scheduler_packing.hpp"
#include "tflite/tflite_schema_generated.hpp"
#include "tflite/tflite_supported_operators.hpp"
#include "util.hpp"

#include <catch_all.hpp>

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

flatbuffers::DetachedBuffer BuildK3ConvModel(int height, int width, int depthK, int depthN, int stride)
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
    std::vector<uint8_t> weightData(size_t(depthK) * depthN * 9, 1);
    std::vector<int32_t> bias(depthN, 0);
    std::vector<uint8_t> biasData(bias.size() * sizeof(int32_t));
    std::memcpy(biasData.data(), bias.data(), biasData.size());
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
        tflite::CreateBufferDirect(builder, &weightData),
        tflite::CreateBufferDirect(builder, &biasData),
    };
    const int outputHeight = (height + stride - 1) / stride;
    const int outputWidth = (width + stride - 1) / stride;
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
        builder, tflite::Padding::SAME, stride, stride, tflite::ActivationFunctionType::NONE,
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

flatbuffers::DetachedBuffer BuildDepthwiseConvModel(int height, int width, int channels, int stride)
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
    REQUIRE(arch.FeatureMapMemory().memory->Name() == "tcdm");
    REQUIRE(arch.FeatureMapMemory().memory->SizeBytes() == ArchNeuralAI::AllocatableTCDMBytes);
    REQUIRE(arch.ReadonlyMemory().memory->Name() == "model");
    REQUIRE(arch.ModelBindingFormat(TensorUsage::IFM) == TensorFormat::NHWC);
    REQUIRE(arch.DefaultInternalTensorFormat(TensorUsage::IFM, false) == TensorFormat::Row32);
    REQUIRE(arch.DefaultInternalTensorFormat(TensorUsage::OFM, true) == TensorFormat::NHWC);

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

TEST_CASE("Neural-AI constraints accept only raw-safe Add quantization")
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
    REQUIRE(constraints->OperatorQuery(OpType::Add, &query) == QueryResult::Unsupported);
    inputQuant.zeroPoints = {0};
    outputQuant.quantMin = {0};
    REQUIRE(constraints->OperatorQuery(OpType::Add, &query) == QueryResult::Unsupported);
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

TEST_CASE("Neural-AI command generator rejects sliced MemoryCopy")
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
    REQUIRE(error == "Neural-AI sliced MemoryCopy is not implemented");
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

TEST_CASE("Neural-AI compiler rejects Add modes that need requantization or activation")
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

    rejects(BuildAddModel(1.0f, 0.5f, 1.0f));
    rejects(BuildAddModel(1.0f, 1.0f, 1.0f, 1));
    rejects(BuildAddModel(1.0f, 1.0f, 1.0f, 0, tflite::ActivationFunctionType::RELU));
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

TEST_CASE("Neural-AI compiler splits pointwise M stripes at the 256-row ABI limit")
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
        while ( offset < 224 + commandBytes )
        {
            const uint16_t type = uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
            const uint16_t commandSize = uint16_t(data[offset + 2]) |
                (uint16_t(data[offset + 3]) << 8);
            if ( type == uint16_t(neuralai::CommandType::PointwiseC32) )
            {
                ++pointwiseCommands;
                REQUIRE(Read32(data + offset + 48) > 0);
                REQUIRE(Read32(data + offset + 48) <= 256);
            }
            offset += commandSize;
        }
        REQUIRE(offset == 224 + commandBytes);
        REQUIRE(pointwiseCommands == uint32_t((width + 255) / 256));
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
    const auto model = BuildK3ConvModel(8, 8, 64, 64, 1);
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
    std::vector<uint32_t> weightOffsets;
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
            weightOffsets.push_back(Read32(data + offset + 16 + 80));
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(linebufferJobs == 4);
    REQUIRE(firstAccumMode == 1);
    REQUIRE(finalAccumMode == 2);
    REQUIRE(weightOffsets.size() == 4);
    REQUIRE(weightOffsets[1] - weightOffsets[0] == 9u * 32u * 32u);
    REQUIRE(weightOffsets[2] - weightOffsets[1] == 9u * 32u * 32u);
    REQUIRE(weightOffsets[3] - weightOffsets[2] == 9u * 32u * 32u);
    REQUIRE(Read32(data + 36) >= weightOffsets.front() + 4u * 9u * 32u * 32u);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
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
            REQUIRE(Read32(data + offset + 16 + 80 + 16) <= 256u);
            const uint16_t inputW = uint16_t(data[offset + 16 + 8]) |
                (uint16_t(data[offset + 16 + 9]) << 8);
            REQUIRE(inputW <= 640u);
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(jobs == 3);
    REQUIRE(Read32(data + 36) <= ArchNeuralAI::AllocatableTCDMBytes);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler stages direct RGB input for the K3 stem")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildK3ConvModel(7, 7, 3, 32, 2);
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
    bool sawShortDma = false;
    while ( offset < 224 + commandBytes )
    {
        const uint16_t type = uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
        const uint16_t commandSize = uint16_t(data[offset + 2]) |
            (uint16_t(data[offset + 3]) << 8);
        if ( type == uint16_t(neuralai::CommandType::LineBufferJob) ) ++linebufferJobs;
        if ( type == uint16_t(neuralai::CommandType::CopyLayout) ) ++copyLayouts;
        if ( type == uint16_t(neuralai::CommandType::DMA2D) )
        {
            REQUIRE(Read32(data + offset + 48) ==
                    uint32_t(neuralai::DMADirection::ExternalToLocal));
            sawShortDma |= Read32(data + offset + 32) == 3;
        }
        offset += commandSize;
    }
    REQUIRE(offset == 224 + commandBytes);
    REQUIRE(linebufferJobs == 1);
    REQUIRE(copyLayouts == 1);
    REQUIRE(sawShortDma);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
}

TEST_CASE("Neural-AI compiler emits group-scoped depthwise C32 commands")
{
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildDepthwiseConvModel(7, 7, 33, 2);
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

TEST_CASE("Neural-AI op groups do not fuse matrix operations")
{
    ArchNeuralAI arch;
    ArchitectureOpGroupQuery query{};
    query.type = OpType::FullyConnected;

    auto group = arch.CreateOpGroup(query);
    REQUIRE(group);
    REQUIRE(group->NeedsAllocation(1));
    REQUIRE(group->Add(query) == 0);
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
