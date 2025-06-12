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

static void SetKernel(const std::shared_ptr<Operation> &operation, const Point2i &size, const Point2i &stride,
    const Point2i &dilation, tflite::Padding padding, int depthMultiplier = 1)
{
    const auto &inputShape = operation->IFM(0)->StorageShape();
    const auto &outputShape = operation->OFM()->StorageShape();
    Margin pad;
    if ( operation->Type() == OpType::TransposeConv2D && inputShape && outputShape )
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
    else if ( padding == tflite::Padding::SAME && inputShape )
    {
        auto dWH = dilation * (size - Point2i(1, 1)) + Point2i(1, 1);
        int xpad = NeededTotalPadding(inputShape.Width(), stride.x, dWH.x);
        int ypad = NeededTotalPadding(inputShape.Height(), stride.y, dWH.y);
        pad = Margin(ypad / 2, xpad / 2, (ypad + 1) / 2, (xpad + 1) / 2);
    }
    auto kernel = std::make_unique<Kernel>(size, stride, dilation, depthMultiplier, pad);
    operation->SetKernel(std::move(kernel));
}

static void ReshapeFullyConnectedWeights(const std::shared_ptr<Operation> &operation, TensorUsage weightUsage)
{
    auto weight_tensor = operation->Input(weightUsage)->tensor;
    if ( weight_tensor->AxisOrder() == AxisOrder::Unknown )
    {
        const auto &shape = weight_tensor->StorageShape();
        // Reshape weight tensor from (num_outputs, ..., num_inputs) to (num_outputs, 1, 1, num_inputs)
        if ( shape.Size() >= 2 && shape.Elements() == (shape[0] * shape[-1]) )
        {
            weight_tensor->Reshape(Shape(shape[0], 1, 1, shape[-1]));
            weight_tensor->SetAxisOrder(AxisOrder::OHWI);
            operation->Input(weightUsage)->shape = weight_tensor->StorageShape();
        }
    }
    else
    {
        // Weight tensor has already been reshaped
        assert(weight_tensor->AxisOrder() == AxisOrder::OHWI);
    }
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
    std::vector<std::unique_ptr<Graph>> &graphs, OptimiserDatabase *optDb, bool skipSemanticsCheck)
{
    assert(model);

    if ( !skipSemanticsCheck )
    {
        auto semanticsChecker = tflite::TFLiteModelSemantics(model);
        semanticsChecker.Check();
    }

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
            auto tflite_intermediates = tflite_operator->intermediates();
            const auto &input_tensors = *tflite_inputs;  // A vector of indices into the `tensors` vector
            int indirect_index = 0;                      // An index into `input_tensors`
            int ifm_count = 0;
            for ( const auto &map_entry : TfLiteMapping::InputTensorIndices(op_type) )
            {
                const TensorUsage usage = map_entry.second;
                if ( indirect_index < int(input_tensors.size()) )  // Missing index means optional tensor not present
                {
                    const int direct_index = input_tensors[indirect_index++];
                    if ( direct_index >= 0 )  // -1 indicates an optional tensor is not present
                    {
                        auto &tensor = tensors.at(direct_index);
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

            if ( tflite_intermediates )
            {
                // Connect operation to its intermediate tensors. They are added as inputs with usage Intermediate.
                int intermediate_count = 0;
                for ( const int tensor_index : *tflite_intermediates )
                {
                    const auto &intermediate = tensors.at(tensor_index);
                    assert(tensorQuantization.count(intermediate->Uid()) > 0);
                    operation->ConnectInput(MakeTensorUsage(TensorUsage::Scratch, intermediate_count++), intermediate)
                        .Set(tensorQuantization[intermediate->Uid()]);
                }
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
                        if ( ifm0->StorageShape() && ifm1->StorageShape() )
                        {
                            ofm->SetStorageShape(Shape::Max(ifm0->StorageShape(), ifm1->StorageShape()));
                        }
                    }
                }
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

            if ( optDb )
            {
                optDb->SourceOp(operation.get(), ext_key);
            }

            // Interpretation of operator options may depend on input/output tensor information,
            // so the operation must be connected to its tensors before parsing operator options.
            ParseOperatorOptions(operation, tflite_operator, optDb);

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
    OptimiserDatabase *optDb, bool skipSemanticsCheck)
{
    LoadGraphs(reinterpret_cast<const uint8_t *>(input), LoadModel(input, size), graphs, optDb, skipSemanticsCheck);
}

std::shared_ptr<Tensor> TfLiteReader::ParseTensor(const tflite::Tensor *tflite_tensor, std::shared_ptr<Buffer> &buffer,
    std::unordered_map<UniqueId, Quantization> &tensorQuantization)
{
    const std::string name = tflite_tensor->name() ? tflite_tensor->name()->str() : "<unnamed>";
    const DataType type = TfLiteMapping::TensorTypeToDataType(tflite_tensor->type());

    auto tensor = std::make_shared<Tensor>(name, type);

    // Regor requires buffers to be aligned based on tensor-datatype
    if ( buffer )
    {
        const Buffer *constBuf = buffer.get();
        const uint8_t *data = constBuf->Data<uint8_t>();
        // Realign tensor if needed
        if ( uintptr_t(data) % (DataTypeSizeBits(type) / 8) != 0 )
        {
            buffer = std::make_shared<Buffer>(buffer->Size(), data, false);
            assert(uintptr_t(buffer->Data<uint8_t>()) % (DataTypeSizeBits(type) / 8) == 0);
        }
    }
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

    if ( tflite_tensor->is_variable() )
    {
        // Create an empty buffer for variable tensor
        assert(buffer == nullptr && "Unexpected buffer for variable tensor!");
        auto emptyBuffer = std::make_shared<Buffer>(std::vector<int>{});
        tensor->SetBuffer(emptyBuffer);
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

void TfLiteReader::ParseOperatorOptions(
    const std::shared_ptr<Operation> &operation, const tflite::Operator *tflite_operator, OptimiserDatabase *optDb)
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
            ReshapeFullyConnectedWeights(operation, TensorUsage::Weights);
            auto weight_tensor = operation->Input(TensorUsage::Weights)->tensor;
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
            break;

        case tflite::BuiltinOptions::SplitVOptions:
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
                assert(Scalar<int64_t>(*params->tensor) < std::numeric_limits<int32_t>::max() && "Too large Argmax axis attribute");
                axis = ClampToType<int32_t>(Scalar<int64_t>(*params->tensor));
            }
            else
            {
                axis = Scalar<int32_t>(*params->tensor);
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

        case tflite::BuiltinOptions::UnidirectionalSequenceLSTMOptions:
        {
            const auto options = GetBuiltinOptions<tflite::UnidirectionalSequenceLSTMOptions>(tflite_operator);
            operation->Attribute<unidirectional_sequence_lstm_attr_t>()->cell_clip = options->cell_clip();
            operation->Attribute<unidirectional_sequence_lstm_attr_t>()->projection_clip = options->proj_clip();
            operation->Attribute<unidirectional_sequence_lstm_attr_t>()->time_major = options->time_major();

            for ( int i = 0; i < 12; i++ )
            {
                if ( operation->Input(MakeTensorUsage(TensorUsage::Weights, i)) )
                {
                    ReshapeFullyConnectedWeights(operation, MakeTensorUsage(TensorUsage::Weights, i));
                }
            }
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
    UnFuseActivation(operation, activation_function, optDb);
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
    if ( optDb )
    {
        optDb->AddOptimised(*operation, activation.get());
    }
}

}  // namespace regor
