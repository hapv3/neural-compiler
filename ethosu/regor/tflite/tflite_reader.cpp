//
// SPDX-FileCopyrightText: Copyright 2021-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "tflite_reader.hpp"

#include "common/logging.hpp"

#include "common/buffer_view.hpp"
#include "common/data_type.hpp"
#include "common/numeric_util.hpp"
#include "common/reverse_type.hpp"
#include "common/scaling.hpp"
#include "common/shape.hpp"
#include "compiler/graph.hpp"
#include "compiler/op_type.hpp"
#include "compiler/operation.hpp"
#include "compiler/operation_util.hpp"
#include "compiler/tensor.hpp"
#include "compiler/tflite_graph_optimiser.hpp"
#include "flatbuffer_utils.hpp"
#include "tflite_mapping.hpp"
#include "tflite_model_semantics.hpp"
#include "tflite_schema_generated.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace regor
{


static int64_t Quantize(float value, const Quantization &quant)
{
    float scale = quant.scales.empty() ? 1.0f : float(quant.scales[0].Dequantize());
    int64_t zp = quant.zeroPoints.empty() ? 0 : quant.zeroPoints[0];
    return zp + int64_t(std::round(double(value / scale)));
}

static void ClampActivation(const std::shared_ptr<Operation> &operation)
{
    OpType opType = operation->Type();
    Quantization &quant = operation->Output(TensorUsage::OFM)->quantization;
    if ( opType == OpType::Relu )
    {
        quant.quantMin = {Quantize(0, quant)};
    }
    else if ( opType == OpType::Relu6 )
    {
        quant.quantMin = {Quantize(0, quant)};
        quant.quantMax = {Quantize(6, quant)};
    }
    else if ( opType == OpType::ReluN1To1 )
    {
        quant.quantMin = {Quantize(-1, quant)};
        quant.quantMax = {Quantize(1, quant)};
    }
}

static void SetKernel(const std::shared_ptr<Operation> &operation, const Point2i &size, const Point2i &stride,
    const Point2i &dilation, tflite::Padding padding, int depthMultiplier = 1)
{
    const auto &inputShape = operation->IFM(0)->StorageShape();
    const auto &outputShape = operation->OFM()->StorageShape();
    Margin pad;
    if ( operation->Type() == OpType::TransposeConv2D )
    {
        // Calculate upscaled ifm height/width by multiplying with stride
        auto ifmWH = inputShape.WH<int>() * stride;
        int ypad = NeededTotalPadding(ifmWH.y, outputShape.Height(), 1, size.y);
        int xpad = NeededTotalPadding(ifmWH.x, outputShape.Width(), 1, size.x);
        if ( stride == Point2i(2, 2) || (stride == Point2i(1, 2) && ifmWH.x == 1 && size.x == 1) ||
             (stride == Point2i(2, 1) && ifmWH.y == 1 && size.y == 1) )
        {
            // Padding for upscaled IFM
            if ( padding == tflite::Padding::SAME )
            {
                int bottom = std::max(((ypad + 1) / stride.y) - 1, 0);
                int top = std::max(size.y - 1 - bottom, 0);
                int right = std::max(((xpad + 1) / stride.x) - 1, 0);
                int left = std::max(size.x - 1 - right, 0);
                pad = Margin(top, left, bottom, right);
            }
            else
            {
                pad = Margin(size.y - 1, size.x - 1, std::max(size.y - 2, 0), std::max(size.x - 2, 0));
            }
        }
        else
        {
            pad = Margin((ypad + 1) / 2, (xpad + 1) / 2, ypad / 2, xpad / 2);
        }
    }
    else if ( padding == tflite::Padding::SAME )
    {
        auto dWH = dilation * (size - Point2i(1, 1)) + Point2i(1, 1);
        int xpad = NeededTotalPadding(inputShape.Width(), stride.x, dWH.x);
        int ypad = NeededTotalPadding(inputShape.Height(), stride.y, dWH.y);
        pad = Margin(ypad / 2, xpad / 2, (ypad + 1) / 2, (xpad + 1) / 2);
    }
    auto kernel = std::make_unique<Kernel>(size, stride, dilation, depthMultiplier, pad);
    operation->SetKernel(std::move(kernel));
}

const tflite::Model *TfLiteReader::LoadModel(const void *input, size_t size)
{
    const uint8_t *buffer = static_cast<const uint8_t *>(input);
    flatbuffers::Verifier::Options options;
    flatbuffers::Verifier verifier(buffer, size, options);

    if ( !tflite::VerifyModelBuffer(verifier) )
    {
        LOG_ERROR("Failed to load TfLite model. Buffer contents inconsistent with generated schema.\n");
        return nullptr;
    }
    return tflite::GetModel(buffer);
}

void TfLiteReader::LoadGraphs(const uint8_t *input, const tflite::Model *model,
    std::vector<std::unique_ptr<Graph>> &graphs, OptimiserDatabase *optDb, IArchitectureConstraints *constraints)
{
    assert(model);

    auto semanticsChecker = tflite::TFLiteModelSemantics(model);
    semanticsChecker.Check();

    std::unordered_map<UniqueId, Quantization> tensorQuantization{};
    std::vector<tflite::BuiltinOperator> opcodes;
    auto tflite_operator_codes = model->operator_codes();
    assert(tflite_operator_codes);
    opcodes.reserve(tflite_operator_codes->size());

    for ( const auto &opcode : *tflite_operator_codes )
    {
        if ( unsigned(opcode->builtin_code()) )
        {
            opcodes.push_back(opcode->builtin_code());
        }
        else  // See https://github.com/tensorflow/tensorflow/blob/bb13f5bb9c9c55/tensorflow/lite/schema/schema_utils.cc
        {
            opcodes.push_back(tflite::BuiltinOperator(opcode->deprecated_builtin_code()));
        }
    }

    std::vector<std::shared_ptr<Buffer>> buffers;
    auto tflite_buffers = model->buffers();
    assert(tflite_buffers);
    buffers.reserve(tflite_buffers->size());

    for ( const auto &tflite_buffer : *tflite_buffers )
    {
        if ( tflite_buffer->offset() > 1 )
        {
            const uint8_t *data = &input[tflite_buffer->offset()];
            buffers.push_back(std::make_shared<Buffer>(tflite_buffer->size(), data, true));
        }
        else if ( tflite_buffer->data() )
        {
            const uint8_t *data = tflite_buffer->data()->data();
            buffers.push_back(std::make_shared<Buffer>(tflite_buffer->data()->size(), data, true));
        }
        else
        {
            buffers.push_back(nullptr);  // Preserves indexing
        }
    }

    auto tflite_subgraphs = model->subgraphs();
    assert(tflite_subgraphs);
    for ( const auto &tflite_subgraph : *tflite_subgraphs )
    {
        std::vector<std::shared_ptr<Tensor>> tensors;
        std::vector<std::shared_ptr<Tensor>> persistent;
        std::vector<std::shared_ptr<Tensor>> placeholder;
        std::vector<std::shared_ptr<Operation>> operations;
        assert(tflite_subgraph);
        auto tflite_tensors = tflite_subgraph->tensors();
        assert(tflite_tensors);
        auto tflite_operators = tflite_subgraph->operators();
        assert(tflite_operators);
        tensors.reserve(tflite_tensors->size());
        operations.reserve(tflite_operators->size());

        // Operators refer to tensors, so create tensors before operations
        for ( const auto &tflite_tensor : *tflite_tensors )
        {
            tensors.push_back(ParseTensor(tflite_tensor, buffers.at(tflite_tensor->buffer()), tensorQuantization));
            if ( tflite_tensor->is_variable() ) persistent.push_back(tensors.back());
        }

        // Create operations
        int ext_key = 0;
        for ( const auto &tflite_operator : *tflite_operators )
        {
            const OpType op_type = TfLiteMapping::BuiltinOperatorToOpType(opcodes.at(tflite_operator->opcode_index()));
            auto operation = std::make_shared<Operation>(op_type);

            // Connect operation to its input tensors
            assert(tflite_operator);
            auto tflite_inputs = tflite_operator->inputs();
            assert(tflite_inputs);
            auto tflite_outputs = tflite_operator->outputs();
            assert(tflite_outputs);
            const auto &input_tensors = *tflite_inputs;  // A vector of indices into the `tensors` vector
            int indirect_index = 0;                      // An index into `input_tensors`
            int ifm_count = 0;
            bool shapelessTensors = false;
            for ( const auto &map_entry : TfLiteMapping::InputTensorIndices(op_type) )
            {
                const TensorUsage usage = map_entry.second;
                if ( indirect_index < int(input_tensors.size()) )  // Missing index means optional tensor not present
                {
                    const int direct_index = input_tensors[indirect_index++];
                    if ( direct_index >= 0 )  // -1 indicates an optional tensor is not present
                    {
                        auto &tensor = tensors.at(direct_index);
                        shapelessTensors = shapelessTensors || !tensor->StorageShape();
                        assert(tensorQuantization.count(tensor->Uid()) > 0);
                        operation->ConnectInput(usage, tensor).Set(tensorQuantization[tensor->Uid()]);
                    }
                    if ( IsIFM(usage) )
                    {
                        ifm_count++;
                    }
                }
            }
            while ( indirect_index < int(input_tensors.size()) )
            {
                const int direct_index = input_tensors[indirect_index++];
                if ( direct_index >= 0 )
                {
                    auto &tensor = tensors.at(direct_index);
                    shapelessTensors = shapelessTensors || !tensor->StorageShape();
                    if ( IsVariadic(op_type) )
                    {
                        // Treat all input tensors beyond those specified in the indices map as IFMs.
                        assert(tensorQuantization.count(tensor->Uid()) > 0);
                        operation->ConnectInput(MakeTensorUsage(TensorUsage::IFM, ifm_count++), tensor)
                            .Set(tensorQuantization[tensor->Uid()]);
                    }
                    else
                    {
                        operation->ConnectInput(MakeTensorUsage(TensorUsage::IFM, ifm_count++), tensor)
                            .Set(tensorQuantization[tensor->Uid()]);
                    }
                }
            }
            if ( ifm_count == 0 )
            {
                // There's no IFMs -- Add a shapeless placeholder tensor because GraphIR requires IFM on all operations.
                // Also add it to the list of placeholder tensors so we can avoid writing this tensors out later on.
                auto tensor = std::make_shared<Tensor>(fmt::format("placeholder-for-{}-IFM", ext_key), DataType::None);
                operation->ConnectInput(TensorUsage::IFM, tensor);
                placeholder.push_back(std::move(tensor));
            }

            // Connect operation to its output tensors
            int ofm_count = 0;
            for ( const int tensor_index : *tflite_outputs )
            {
                const auto &ofm = tensors.at(tensor_index);
                if ( !ofm->StorageShape() )
                {
                    // Try to figure out the OFM shape if the OFM shape is unknown
                    if ( IsUnaryElementwise(op_type) || op_type == OpType::Quantize )
                    {
                        auto ifm = operation->IFM(0);
                        assert(ifm);
                        ofm->SetStorageShape(ifm->StorageShape());
                    }
                    else if ( IsBinaryElementwise(op_type) )
                    {
                        auto ifm0 = operation->IFM(0);
                        auto ifm1 = operation->IFM(1);
                        assert(ifm0 && ifm1);
                        ofm->SetStorageShape(Shape::Max(ifm0->StorageShape(), ifm1->StorageShape()));
                    }
                }
                shapelessTensors = shapelessTensors || !ofm->StorageShape();
                assert(tensorQuantization.count(ofm->Uid()) > 0);
                operation->ConnectOutput(MakeTensorUsage(TensorUsage::OFM, ofm_count++), ofm).Set(tensorQuantization[ofm->Uid()]);
            }
            if ( ofm_count == 0 )
            {
                // There's no OFM -- Add a shapeless placeholder tensor because GraphIR requires OFM on all operations.
                // Also add it to the list of placeholder tensors so we can avoid writing this tensors out later on.
                auto tensor = std::make_shared<Tensor>(fmt::format("placeholder-for-{}-OFM", ext_key), DataType::None);
                operation->ConnectOutput(TensorUsage::OFM, tensor);
                placeholder.push_back(std::move(tensor));
            }

            if ( ifm_count == 0 || ofm_count == 0 )
            {
                // NPU operations must have IFM and OFM
                operation->SetPassthroughOp();
            }

            if ( shapelessTensors )
            {
                operation->SetPassthroughOp();
            }

            if ( optDb )
            {
                optDb->SourceOp(operation.get(), ext_key);
            }

            // Interpretation of operator options may depend on input/output tensor information,
            // so the operation must be connected to its tensors before parsing operator options.
            ParseOperatorOptions(operation, tflite_operator, optDb, constraints);

            // Set rounding according to reference
            SetOFMRounding(operation);

            operations.push_back(std::move(operation));
            ext_key++;
        }

        // Create graph
        auto graph = std::make_unique<Graph>(GraphNotation::TFLite);
        for ( const auto &index : *tflite_subgraph->inputs() )
        {
            graph->AddInput(tensors.at(index));
        }
        for ( const auto &index : *tflite_subgraph->outputs() )
        {
            graph->AddOutput(tensors.at(index));
        }
        for ( auto &tensor : persistent )
        {
            graph->AddPersistent(tensor);
        }
        for ( auto &tensor : placeholder )
        {
            graph->AddPlaceholder(tensor);
            graph->AddOutput(tensor);
        }

        // Find and disconnect any operations which do not precede a graph output. Otherwise they might persist beyond
        // the life of the Graph because the Graph destructor only disconnects operations which precede its outputs.
        std::vector<Operation *> predecessors;
        graph->GetAllOperations(predecessors);
        for ( auto &operation : operations )
        {
            if ( std::find(predecessors.begin(), predecessors.end(), operation.get()) == predecessors.end() )
            {
                if ( TfLiteMapping::CanFuseActivationFunction(operation.get()) )
                {
                    operation->OFM()->Readers().front()->Disconnect();
                }
                operation->Disconnect();
            }
        }

        // Save a pointer to the model table so we can look up operator_code later
        graph->SetPassthrough(model);
        graph->SetName(GetString(tflite_subgraph->name()));

        // Give graph to caller
        graphs.push_back(std::move(graph));

        // Any operations which do not precede a graph output are destroyed here,
        // Most tensors which do not precede a graph output are also destroyed here.
        //  - Tensors which are themselves an input or output of a graph will persist.
        //  - Tensors which do not precede a graph output but are written to by an operation which does will persist.
    }
}

void TfLiteReader::LoadGraphs(const void *input, size_t size, std::vector<std::unique_ptr<Graph>> &graphs,
    OptimiserDatabase *optDb, IArchitectureConstraints *constraints)
{
    LoadGraphs(reinterpret_cast<const uint8_t *>(input), LoadModel(input, size), graphs, optDb, constraints);
}

std::shared_ptr<Tensor> TfLiteReader::ParseTensor(const tflite::Tensor *tflite_tensor,
    const std::shared_ptr<Buffer> &buffer, std::unordered_map<UniqueId, Quantization> &tensorQuantization)
{
    const std::string name = tflite_tensor->name() ? tflite_tensor->name()->str() : "<unnamed>";
    const DataType type = TfLiteMapping::TensorTypeToDataType(tflite_tensor->type());

    auto tensor = std::make_shared<Tensor>(name, type);

    Shape shape;  // Defaults to shapeless
    auto signature = tflite_tensor->shape_signature();
    if ( tflite_tensor->shape() && tflite_tensor->shape()->size() )
    {
        shape = Shape(tflite_tensor->shape()->data(), tflite_tensor->shape()->size());
    }
    if ( signature && signature->size() )
    {
        // Signature trumps shape, but default to shape if signature is dynamic
        if ( std::find(signature->begin(), signature->end(), -1) == signature->end() )
        {
            shape = Shape(signature->data(), signature->size());
        }
        else
        {
            LOG_WARN(
                "Tensor '{}' has a dynamic shape signature, which is not supported. "
                "Attempting to proceed with a fixed shape.\n",
                name);
        }
    }

    // Fix missing shapes on constant inputs
    if ( shape.Size() == 0 && buffer )
    {
        shape = Shape(DataTypeElements(type, buffer->Size()));
    }
    tensor->SetStorageShape(shape);
    tensor->SetBuffer(buffer);
    tensorQuantization[tensor->Uid()] = {};

    if ( tflite_tensor->quantization() )
    {
        if ( tflite_tensor->quantization()->details() )
        {
            LOG_WARN(
                "Tensor '{}' specifies custom quantization, which is not supported. "
                "Attempting to proceed with standard quantization only.\n",
                name);
        }
        if ( tflite_tensor->quantization()->scale() && tflite_tensor->quantization()->zero_point() )
        {
            Quantization &quantization = tensorQuantization[tensor->Uid()];
            quantization.type = QuantizationType::TFLITE;
            std::vector<float> scale_f32 = FlatbufferUtils::LoadVector<float>(tflite_tensor->quantization()->scale());
            for ( float scale : scale_f32 )
            {
                quantization.scales.push_back(QuantizedScale(scale));
            }
            quantization.zeroPoints = FlatbufferUtils::LoadVector<int64_t>(tflite_tensor->quantization()->zero_point());
            quantization.dimension = tflite_tensor->quantization()->quantized_dimension();
        }
    }

    if ( tflite_tensor->sparsity() )
    {
        LOG_WARN("Tensor '{}' contains sparsity information, which is not supported and will be ignored.\n", name);
    }

    tensor->SetPassthrough(tflite_tensor);

    return tensor;
}

template<typename T>
static const T *GetBuiltinOptions(const tflite::Operator *tflite_operator)
{
    const auto options = tflite_operator->builtin_options_as<T>();
    assert(options);
    return options;
}

void TfLiteReader::ParseOperatorOptions(const std::shared_ptr<Operation> &operation,
    const tflite::Operator *tflite_operator, OptimiserDatabase *optDb, IArchitectureConstraints *constraints)
{
    const auto type = tflite_operator->builtin_options_type();
    auto activation_function = tflite::ActivationFunctionType::NONE;

    switch ( type )
    {
        case tflite::BuiltinOptions::Conv2DOptions:
        {
            const auto options = GetBuiltinOptions<tflite::Conv2DOptions>(tflite_operator);
            auto weight_tensor = operation->Input(TensorUsage::Weights)->tensor;
            weight_tensor->SetAxisOrder(AxisOrder::OHWI);
            SetKernel(operation, Point2i(weight_tensor->StorageShape().Width(), weight_tensor->StorageShape().Height()),
                Point2i(options->stride_w(), options->stride_h()),
                Point2i(options->dilation_w_factor(), options->dilation_h_factor()), options->padding());
            activation_function = options->fused_activation_function();
        }
        break;

        case tflite::BuiltinOptions::DepthwiseConv2DOptions:
        {
            const auto options = GetBuiltinOptions<tflite::DepthwiseConv2DOptions>(tflite_operator);
            auto weight_tensor = operation->Input(TensorUsage::Weights)->tensor;
            weight_tensor->SetAxisOrder(AxisOrder::IHWO);
            Shape weightShape = weight_tensor->StorageShape();
            int depth_multiplier = options->depth_multiplier();
            if ( depth_multiplier == 0 )  // Depth multiplier is implicit. Derive it from tensor dimensions.
            {
                const int input_depth = operation->Input(TensorUsage::IFM)->tensor->StorageShape().Depth();
                depth_multiplier = weightShape.Depth() / input_depth;
            }
            SetKernel(operation, weightShape.WH<int>(), Point2i(options->stride_w(), options->stride_h()),
                Point2i(options->dilation_w_factor(), options->dilation_h_factor()), options->padding(), depth_multiplier);
            activation_function = options->fused_activation_function();
        }
        break;

        case tflite::BuiltinOptions::TransposeConvOptions:
        {
            const auto options = GetBuiltinOptions<tflite::TransposeConvOptions>(tflite_operator);
            auto weight_tensor = operation->Input(TensorUsage::Weights)->tensor;
            weight_tensor->SetAxisOrder(AxisOrder::OHWI);
            SetKernel(operation, Point2i(weight_tensor->StorageShape().Width(), weight_tensor->StorageShape().Height()),
                Point2i(options->stride_w(), options->stride_h()), Point2i(1, 1) /* no dilation */, options->padding());
            activation_function = options->fused_activation_function();
            auto attr = operation->Attribute<transpose_conv2d_attr_t>();
            attr->outShape = operation->Output(TensorUsage::OFM)->shape;
            attr->outPadTBLR = Shape(0, 0, 0, 0);  // TFLite has no out-padding
        }
        break;

        case tflite::BuiltinOptions::Pool2DOptions:
        {
            const auto options = GetBuiltinOptions<tflite::Pool2DOptions>(tflite_operator);
            SetKernel(operation, Point2i(options->filter_width(), options->filter_height()),
                Point2i(options->stride_w(), options->stride_h()), Point2i(1, 1),  // no dilation
                options->padding());
            activation_function = options->fused_activation_function();
        }
        break;

        case tflite::BuiltinOptions::FullyConnectedOptions:
        {
            const auto options = GetBuiltinOptions<tflite::FullyConnectedOptions>(tflite_operator);
            activation_function = options->fused_activation_function();

            // TODO: Are `weights_format`, `keep_num_dims` or `asymmetric_quantize_inputs` used?

            auto weight_tensor = operation->Input(TensorUsage::Weights)->tensor;
            if ( weight_tensor->AxisOrder() == AxisOrder::Unknown )
            {
                // Reshape weight tensor from (num_outputs, ..., num_inputs) to (num_outputs, 1, 1, num_inputs)
                weight_tensor->SetAxisOrder(AxisOrder::OHWI);
                const auto &shape = weight_tensor->StorageShape();
                for ( int i = 1; i < shape.Size() - 1; i++ )
                {
                    if ( shape[i] != 1 ) operation->SetPassthroughOp();
                }
                weight_tensor->Reshape(Shape(shape[0], 1, 1, shape[-1]));
            }
            else
            {
                // Weight tensor has already been reshaped
                assert(weight_tensor->AxisOrder() == AxisOrder::OHWI);
            }
            if ( operation->Input(TensorUsage::Scales) == nullptr )
            {
                // Op has no bias; add bias tensor filled with zeros
                int elems = weight_tensor->StorageShape().Batch();
                auto ifm = operation->Input(TensorUsage::IFM)->tensor;
                DataType biasType;
                std::shared_ptr<Buffer> buf;
                if ( ifm->Type() == DataType::Int16 )
                {
                    biasType = DataType::Int64;
                    std::vector<int64_t> data(ToUnsigned(elems));
                    buf = std::make_shared<Buffer>(std::move(data));
                }
                else
                {
                    biasType = DataType::Int32;
                    std::vector<int32_t> data(ToUnsigned(elems));
                    buf = std::make_shared<Buffer>(std::move(data));
                }
                auto biasTens = std::make_shared<Tensor>(weight_tensor->Name() + "_bias", biasType, Shape(1, 1, 1, elems), buf);
                operation->ConnectInput(TensorUsage::Scales, biasTens);
            }
        }
        break;

        case tflite::BuiltinOptions::SoftmaxOptions:
        {
            const auto options = GetBuiltinOptions<tflite::SoftmaxOptions>(tflite_operator);
            operation->Attribute<softmax_attr_t>()->beta = options->beta();
        }
        break;

        case tflite::BuiltinOptions::ConcatenationOptions:
        {
            const auto options = GetBuiltinOptions<tflite::ConcatenationOptions>(tflite_operator);
            operation->Attribute<axis_attr_t>()->axis = options->axis();
            activation_function = options->fused_activation_function();
        }
        break;

        case tflite::BuiltinOptions::AddOptions:
        {
            const auto options = GetBuiltinOptions<tflite::AddOptions>(tflite_operator);
            activation_function = options->fused_activation_function();
        }
        break;

        case tflite::BuiltinOptions::SubOptions:
        {
            const auto options = GetBuiltinOptions<tflite::SubOptions>(tflite_operator);
            activation_function = options->fused_activation_function();
        }
        break;

        case tflite::BuiltinOptions::DivOptions:
        {
            const auto options = GetBuiltinOptions<tflite::DivOptions>(tflite_operator);
            activation_function = options->fused_activation_function();
        }
        break;

        case tflite::BuiltinOptions::MulOptions:
        {
            const auto options = GetBuiltinOptions<tflite::MulOptions>(tflite_operator);
            activation_function = options->fused_activation_function();
        }
        break;

        case tflite::BuiltinOptions::L2NormOptions:
        {
            const auto options = GetBuiltinOptions<tflite::L2NormOptions>(tflite_operator);
            activation_function = options->fused_activation_function();
        }
        break;

        case tflite::BuiltinOptions::ReshapeOptions:
        {
            const auto conn = operation->Input(TensorUsage::Params);
            if ( conn == nullptr )
            {
                const auto options = tflite_operator->builtin_options_as<tflite::ReshapeOptions>();
                auto new_shape = options ? options->new_shape() : nullptr;
                if ( new_shape )
                {
                    // New shape specified as option. Convert to input tensor.
                    auto tensor = std::make_shared<Tensor>("new_shape", DataType::Int32);
                    tensor->SetStorageShape(Shape(new_shape->size()));
                    auto buffer_base = new_shape->Data();
                    int buffer_size = int(new_shape->size() * (sizeof(int32_t) / sizeof(uint8_t)));
                    tensor->SetBuffer(std::make_shared<Buffer>(buffer_size, buffer_base, true));
                    operation->ConnectInput(TensorUsage::Params, tensor);
                }
            }
        }
        break;

        case tflite::BuiltinOptions::PackOptions:
        {
            const auto options = GetBuiltinOptions<tflite::PackOptions>(tflite_operator);
            operation->Attribute<axis_attr_t>()->axis = options->axis();
        }
        break;

        case tflite::BuiltinOptions::UnpackOptions:
        {
            const auto options = GetBuiltinOptions<tflite::UnpackOptions>(tflite_operator);
            operation->Attribute<axis_attr_t>()->axis = options->axis();
        }
        break;

        case tflite::BuiltinOptions::LeakyReluOptions:
        {
            const auto options = GetBuiltinOptions<tflite::LeakyReluOptions>(tflite_operator);
            operation->Attribute<leaky_relu_attr_t>()->alpha = options->alpha();
        }
        break;

        case tflite::BuiltinOptions::StridedSliceOptions:
            break;

        case tflite::BuiltinOptions::SplitOptions:
        {
            int num_splits = GetBuiltinOptions<tflite::SplitOptions>(tflite_operator)->num_splits();
            if ( size_t(num_splits) != operation->Outputs().size() ) operation->SetPassthroughOp();
        }
        break;

        case tflite::BuiltinOptions::SplitVOptions:
        {
            int num_splits = GetBuiltinOptions<tflite::SplitVOptions>(tflite_operator)->num_splits();
            if ( size_t(num_splits) != operation->Outputs().size() ) operation->SetPassthroughOp();
        }
        break;

        case tflite::BuiltinOptions::SVDFOptions:
        {
            const auto options = GetBuiltinOptions<tflite::SVDFOptions>(tflite_operator);
            activation_function = options->fused_activation_function();
        }
        break;

        case tflite::BuiltinOptions::ArgMaxOptions:
        {
            // Create axis attribute from parameter-tensor
            auto *ifmConn = operation->Input(TensorUsage::IFM0);
            auto *params = operation->Input(TensorUsage::Params);
            assert(ifmConn);
            assert(params);
            int axis = 0;
            if ( params->tensor->Type() == DataType::Int64 )
            {
                assert(params->tensor->View().Values<int64_t>()[0] < std::numeric_limits<int32_t>::max() && "Too large Argmax axis attribute");
                axis = ClampToType<int32_t>(params->tensor->View().Values<int64_t>()[0]);
            }
            else
            {
                axis = params->tensor->View().Values<int32_t>()[0];
            }
            if ( axis < 0 )
            {
                axis += ifmConn->shape.Size();
            }
            operation->Attribute<axis_attr_t>()->axis = axis;
        }
        break;

        case tflite::BuiltinOptions::MirrorPadOptions:
        {
            const auto options = GetBuiltinOptions<tflite::MirrorPadOptions>(tflite_operator);
            operation->Attribute<mirror_pad_mode_attr_t>()->mode = options->mode();
        }
        break;

        case tflite::BuiltinOptions::PadOptions:
        {
            operation->Attribute<pad_attr_t>()->pad_const = 0;
        }
        break;

        case tflite::BuiltinOptions::ResizeBilinearOptions:
        case tflite::BuiltinOptions::ResizeNearestNeighborOptions:
            break;

        // Options that are not used by the compiler are not loaded in, but can be written out again via passthrough
        case tflite::BuiltinOptions::BatchMatMulOptions:
        case tflite::BuiltinOptions::GatherOptions:
        case tflite::BuiltinOptions::ShapeOptions:
        case tflite::BuiltinOptions::SqueezeOptions:
        case tflite::BuiltinOptions::ReducerOptions:
        case tflite::BuiltinOptions::CallOnceOptions:
        case tflite::BuiltinOptions::VarHandleOptions:
            break;

        // Empty option sets require no parsing
        case tflite::BuiltinOptions::NONE:
        case tflite::BuiltinOptions::HardSwishOptions:
        case tflite::BuiltinOptions::MaximumMinimumOptions:
        case tflite::BuiltinOptions::PadV2Options:
        case tflite::BuiltinOptions::DequantizeOptions:
        case tflite::BuiltinOptions::QuantizeOptions:
        case tflite::BuiltinOptions::TransposeOptions:
        case tflite::BuiltinOptions::GatherNdOptions:
        case tflite::BuiltinOptions::ScatterNdOptions:
        case tflite::BuiltinOptions::ReadVariableOptions:
        case tflite::BuiltinOptions::AssignVariableOptions:
        case tflite::BuiltinOptions::SelectOptions:
        case tflite::BuiltinOptions::SelectV2Options:
            break;

        case tflite::BuiltinOptions::ConcatEmbeddingsOptions:
        case tflite::BuiltinOptions::LSHProjectionOptions:
        case tflite::BuiltinOptions::RNNOptions:
        case tflite::BuiltinOptions::LocalResponseNormalizationOptions:
        case tflite::BuiltinOptions::LSTMOptions:
        case tflite::BuiltinOptions::CallOptions:
        case tflite::BuiltinOptions::SkipGramOptions:
        case tflite::BuiltinOptions::SpaceToDepthOptions:
        case tflite::BuiltinOptions::EmbeddingLookupSparseOptions:
        case tflite::BuiltinOptions::BatchToSpaceNDOptions:
        case tflite::BuiltinOptions::SpaceToBatchNDOptions:
        case tflite::BuiltinOptions::SequenceRNNOptions:
        case tflite::BuiltinOptions::ExpOptions:
        case tflite::BuiltinOptions::TopKV2Options:
        case tflite::BuiltinOptions::LogSoftmaxOptions:
        case tflite::BuiltinOptions::CastOptions:
        case tflite::BuiltinOptions::LessOptions:
        case tflite::BuiltinOptions::NegOptions:
        case tflite::BuiltinOptions::GreaterOptions:
        case tflite::BuiltinOptions::GreaterEqualOptions:
        case tflite::BuiltinOptions::LessEqualOptions:
        case tflite::BuiltinOptions::SliceOptions:
        case tflite::BuiltinOptions::SparseToDenseOptions:
        case tflite::BuiltinOptions::TileOptions:
        case tflite::BuiltinOptions::ExpandDimsOptions:
        case tflite::BuiltinOptions::EqualOptions:
        case tflite::BuiltinOptions::NotEqualOptions:
        case tflite::BuiltinOptions::PowOptions:
        case tflite::BuiltinOptions::ArgMinOptions:
        case tflite::BuiltinOptions::FakeQuantOptions:
        case tflite::BuiltinOptions::LogicalOrOptions:
        case tflite::BuiltinOptions::OneHotOptions:
        case tflite::BuiltinOptions::LogicalAndOptions:
        case tflite::BuiltinOptions::LogicalNotOptions:
        case tflite::BuiltinOptions::FloorDivOptions:
        case tflite::BuiltinOptions::SquareOptions:
        case tflite::BuiltinOptions::ZerosLikeOptions:
        case tflite::BuiltinOptions::FillOptions:
        case tflite::BuiltinOptions::BidirectionalSequenceLSTMOptions:
        case tflite::BuiltinOptions::BidirectionalSequenceRNNOptions:
        case tflite::BuiltinOptions::UnidirectionalSequenceLSTMOptions:
        case tflite::BuiltinOptions::FloorModOptions:
        case tflite::BuiltinOptions::RangeOptions:
        case tflite::BuiltinOptions::SquaredDifferenceOptions:
        case tflite::BuiltinOptions::AbsOptions:
        case tflite::BuiltinOptions::UniqueOptions:
        case tflite::BuiltinOptions::ReverseV2Options:
        case tflite::BuiltinOptions::AddNOptions:
        case tflite::BuiltinOptions::CosOptions:
        case tflite::BuiltinOptions::WhereOptions:
        case tflite::BuiltinOptions::RankOptions:
        case tflite::BuiltinOptions::ReverseSequenceOptions:
        case tflite::BuiltinOptions::MatrixDiagOptions:
        case tflite::BuiltinOptions::MatrixSetDiagOptions:
        case tflite::BuiltinOptions::IfOptions:
        case tflite::BuiltinOptions::WhileOptions:
        case tflite::BuiltinOptions::DepthToSpaceOptions:
        case tflite::BuiltinOptions::NonMaxSuppressionV4Options:
        case tflite::BuiltinOptions::NonMaxSuppressionV5Options:
        case tflite::BuiltinOptions::DensifyOptions:
        case tflite::BuiltinOptions::SegmentSumOptions:
        case tflite::BuiltinOptions::CumsumOptions:
        case tflite::BuiltinOptions::BroadcastToOptions:
        case tflite::BuiltinOptions::Rfft2dOptions:
        case tflite::BuiltinOptions::Conv3DOptions:
        case tflite::BuiltinOptions::HashtableOptions:
        case tflite::BuiltinOptions::HashtableFindOptions:
        case tflite::BuiltinOptions::HashtableImportOptions:
        case tflite::BuiltinOptions::HashtableSizeOptions:
            // TODO
            LOG_WARN("TfLiteReader: Built-in options type '{}' is not yet implemented and will be ignored.\n",
                tflite::EnumNameBuiltinOptions(type));
            break;
        default:
            LOG_ERROR("TfLiteReader: Unrecognised built-in options type '{}'\n", int(type));
            break;
    }
    operation->SetPassthrough(tflite_operator);

    ExecutionQuery query{};
    bool isValidQuery = true;
    try
    {
        query = OperationToExecQuery(*operation);
    }
    catch ( std::invalid_argument &e )
    {
        LOG_WARN("ExecutionQuery not buildable for operation {}: {}\n", EnumToString(operation->Type()), e.what())
        isValidQuery = false;
    }
    if ( operation->Type() == OpType::None )
    {
        operation->SetPassthroughOp();
    }
    else if ( operation->Type() != OpType::Passthrough )
    {
        if ( !isValidQuery || !(constraints->CanExecute(query)) )
        {
            operation->SetPassthroughOp();
        }
        else
        {
            UnFuseActivation(operation, activation_function, optDb);
        }
    }
}

void TfLiteReader::SetOFMRounding(const std::shared_ptr<Operation> &operation)
{
    auto ifm = operation->Input(TensorUsage::IFM)->tensor;
    auto opType = operation->Type();

    // Default rounding mode
    RoundMode roundMode = RoundMode::DBL;

    // Change according to reference
    if ( ifm->Type() == DataType::Int16 && (IsConvolution(opType) || IsVectorProduct(opType)) )
    {
        roundMode = RoundMode::NATURAL;
    }
    else if ( IsPooling(opType) )
    {
        roundMode = RoundMode::NATURAL;
    }
    operation->Output(TensorUsage::OFM)->Set(roundMode);
}

void TfLiteReader::UnFuseActivation(const std::shared_ptr<Operation> &operation, tflite::ActivationFunctionType type, OptimiserDatabase *optDb)
{
    if ( type == tflite::ActivationFunctionType::NONE )
    {
        return;
    }

    assert(operation->Outputs().size() == 1);

    // Before: upstream -> operation --------------------------------------> output_tensor -> downstream
    // After:  upstream -> operation -> intermediate_tensor -> activation -> output_tensor -> downstream

    auto activation = std::make_shared<Operation>(TfLiteMapping::ActivationFunctionToOpType(type));
    auto &output_tensor = operation->Outputs().front().tensor;
    Quantization quantization = operation->Outputs().front().quantization;
    std::shared_ptr<Tensor> intermediate_tensor = output_tensor->Clone();
    activation->ConnectOutput(TensorUsage::OFM, output_tensor).Set(quantization);
    output_tensor->RemoveWriter(operation);
    operation->ConnectOutput(TensorUsage::OFM, intermediate_tensor).Set(quantization);
    activation->ConnectInput(TensorUsage::IFM, intermediate_tensor).Set(quantization);
    ClampActivation(activation);
    if ( optDb )
    {
        optDb->AddOptimised(operation.get(), activation.get());
    }
}

namespace
{

ResizeSupportQuery CalculateResizeSupportQuery(const Operation &operation)
{
    auto ifmConn = operation.Input(TensorUsage::IFM);
    auto ofmConn = operation.Output(TensorUsage::OFM);
    assert(ifmConn);
    assert(ofmConn);

    // Get numerators(n) and denominators(d) for the scale fractions
    int width_n = ofmConn->shape.Width();
    int width_d = ifmConn->shape.Width();
    int height_n = ofmConn->shape.Height();
    int height_d = ifmConn->shape.Height();
    int heightOffset = 0;
    int widthOffset = 0;

    const tflite::Operator *tflite_operator = static_cast<const tflite::Operator *>(operation.Passthrough());
    assert(tflite_operator);
    bool halfPixelCenters = false;
    bool alignCorners = false;
    if ( operation.Type() == OpType::ResizeBilinear )
    {
        const auto *opt = tflite_operator->builtin_options_as_ResizeBilinearOptions();
        assert(opt);
        alignCorners = opt->align_corners();
        halfPixelCenters = opt->half_pixel_centers();
    }
    else
    {
        const auto *opt = tflite_operator->builtin_options_as_ResizeNearestNeighborOptions();
        assert(opt);
        alignCorners = opt->align_corners();
        // Use half-pixel-centers if align-corners is false.
        // This aligns with reference kernels
        halfPixelCenters = !alignCorners || opt->half_pixel_centers();
    }

    // Compute scaling fractions
    // align-corners use a scale-factor of (n-1)/(d-1)
    if ( alignCorners )
    {
        if ( width_d > 1 )
        {
            width_n -= 1;
            width_d -= 1;
        }
        if ( height_d > 1 )
        {
            height_n -= 1;
            height_d -= 1;
        }
    }

    // reduce scaling fractions with gcd
    int gcd_w = std::gcd(width_n, width_d);
    width_n = (width_n / gcd_w);
    width_d = (width_d / gcd_w);

    int gcd_h = std::gcd(height_n, height_d);
    height_n = (height_n / gcd_h);
    height_d = (height_d / gcd_h);

    if ( halfPixelCenters )
    {
        // make sure fractions are evenly divisible by 2
        width_n = width_n * 2;
        width_d = width_d * 2;
        height_n = height_n * 2;
        height_d = height_d * 2;
        // adjust offset for half-pixel-centers
        widthOffset = (width_d / 2) - (width_n / 2);
        heightOffset = (height_d / 2) - (height_n / 2);
    }

    // set up op-support query
    ResizeSupportQuery resizeQuery;
    resizeQuery.scaleX = {int16_t(width_n), int16_t(width_d)};
    resizeQuery.scaleY = {int16_t(height_n), int16_t(height_d)};
    resizeQuery.offsetX = widthOffset;
    resizeQuery.offsetY = heightOffset;
    resizeQuery.ifmShape = ifmConn->shape;
    resizeQuery.mode = (operation.Type() == OpType::ResizeBilinear) ? ArchResizeMode::Bilinear : ArchResizeMode::Nearest;
    return resizeQuery;
}

}  // namespace

ExecutionQuery TfLiteReader::OperationToExecQuery(const Operation &operation)
{
    ExecutionQuery query{};
    query.opType = operation.Type();
    query.ifmType = operation.IFM(0)->Type();
    query.ofmType = operation.OFM()->Type();
    query.ifmShape = operation.Input(TensorUsage::IFM0)->shape;
    if ( operation.Input(TensorUsage::IFM1) )
    {
        query.ifm2Type = operation.Input(TensorUsage::IFM1)->tensor->Type();
        query.ifm2Shape = operation.Input(TensorUsage::IFM1)->shape;
    }
    query.ofmShape = operation.Output(TensorUsage::OFM)->shape;

    switch ( query.opType )
    {
        case OpType::LeakyRelu:
        {
            auto *ifmConn = operation.Input(TensorUsage::IFM0);
            auto *ofmConn = operation.Output(TensorUsage::OFM);
            query.quantScalingInvalidOrUnequal = !IsScalingValidAndEqual(*ifmConn, *ofmConn);
            break;
        }
        case OpType::Transpose:
        {
            query.ifmShape = operation.Input(TensorUsage::IFM0)->shape;
            query.targetType = OpType::MemoryCopy;
            query.transposeType = CalculateTransposeType(operation);
            break;
        }
        case OpType::ReverseV2:
        {
            query.targetType = OpType::MemoryCopy;
            auto paramsConn = operation.Input(TensorUsage::Params);
            assert(paramsConn);
            assert(paramsConn->tensor->Type() == DataType::Int32);
            if ( !paramsConn->tensor->IsConstant() )
            {
                query.reverseTypeMask = ReverseType::Dynamic;
            }
            else
            {
                // non-dynamic reverseType, we convert it to bitmask
                auto view = paramsConn->tensor->View();
                Shape axes = Shape(view.Buffer()->Data<int32_t>(), view.ViewShape().Elements());
                query.reverseTypeMask = ToReverseMask(axes, query.ofmShape.Size());
            }
            break;
        }
        case OpType::ResizeBilinear:
        case OpType::ResizeNearestNeighbor:
            query.opType = OpType::Resize;
            query.resizeQuery = CalculateResizeSupportQuery(operation);
            break;
        default:
            break;
    }
    return query;
}

}  // namespace regor
