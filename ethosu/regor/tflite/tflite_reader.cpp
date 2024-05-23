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

#include "tflite_reader.hpp"

#include "common/logging.hpp"

#include "common/buffer_view.hpp"
#include "common/data_type.hpp"
#include "common/numeric_util.hpp"
#include "common/scaling.hpp"
#include "common/shape.hpp"
#include "compiler/graph.hpp"
#include "compiler/op_type.hpp"
#include "compiler/operation.hpp"
#include "compiler/tensor.hpp"
#include "flatbuffer_utils.hpp"
#include "tflite_mapping.hpp"
#include "tflite_model_semantics.hpp"
#include "tflite_schema_generated.hpp"

#include <cassert>
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
        // Calculate TOSA TRANSPOSE_CONV2D out_pad
        int totalPaddingWidth = (inputShape.Width() - 1) * stride.x + size.x - outputShape.Width();
        totalPaddingWidth = totalPaddingWidth > 0 ? totalPaddingWidth : 0;
        int totalPaddingHeight = (inputShape.Height() - 1) * stride.y + size.y - outputShape.Height();
        totalPaddingHeight = totalPaddingHeight > 0 ? totalPaddingHeight : 0;
        int padTop = totalPaddingHeight / 2;
        int padLeft = totalPaddingWidth / 2;
        int padBottom = totalPaddingHeight - padTop;
        int padRight = totalPaddingWidth - padLeft;
        pad = Margin(padTop, padLeft, padBottom, padRight);
    }
    else
    {
        if ( padding == tflite::Padding::SAME )
        {
            int dw = dilation.x * (size.x - 1) + 1;
            int xpad = NeededTotalPadding(inputShape.Width(), stride.x, dw);
            int dh = dilation.y * (size.y - 1) + 1;
            int ypad = NeededTotalPadding(inputShape.Height(), stride.y, dh);
            pad = Margin(ypad / 2, xpad / 2, (ypad + 1) / 2, (xpad + 1) / 2);
        }
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

static Shape OperationOFMShape(OpType type, const Shape &input)
{
    assert(type == OpType::Quantize);
    UNUSED(type);
    return input;
}

void TfLiteReader::LoadGraphs(const tflite::Model *model, std::vector<std::unique_ptr<Graph>> &graphs, OptimiserDatabase *optDb)
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
        if ( tflite_buffer->data() )
        {
            uint8_t *data = const_cast<uint8_t *>(tflite_buffer->data()->data());
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
            Shape largestInput;
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
                        largestInput = Shape::Max(tensor->StorageShape(), largestInput);
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
                        LOG_WARN("TfLiteReader: Unexpected input tensor {} of operator {} will be ignored.\n",
                            tensor->Name(), OpTypeToString(operation->Type()));
                    }
                }
            }

            // Connect operation to its output tensors
            int ofm_count = 0;
            for ( const int tensor_index : *tflite_outputs )
            {
                const auto &ofm = tensors.at(tensor_index);
                // Correct OFMs with no dimension
                if ( ofm->StorageShape().Size() == 0 )
                {
                    ofm->SetStorageShape(OperationOFMShape(operation->Type(), largestInput));
                }
                assert(tensorQuantization.count(ofm->Uid()) > 0);
                operation->ConnectOutput(MakeTensorUsage(TensorUsage::OFM, ofm_count++), ofm).Set(tensorQuantization[ofm->Uid()]);
            }
            if ( optDb )
            {
                optDb->SourceOp(operation.get(), ext_key);
            }
            // Interpretation of operator options may depend on input/output tensor information,
            // so the operation must be connected to its tensors before parsing operator options.
            ParseOperatorOptions(operation, tflite_operator, optDb);

            // Set rounding according to reference
            SetOperatorRounding(operation);

            operations.push_back(operation);
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

        // Give graph to caller
        graphs.push_back(std::move(graph));

        // Any operations which do not precede a graph output are destroyed here,
        // Most tensors which do not precede a graph output are also destroyed here.
        //  - Tensors which are themselves an input or output of a graph will persist.
        //  - Tensors which do not precede a graph output but are written to by an operation which does will persist.
    }
}

void TfLiteReader::LoadGraphs(const void *input, size_t size, std::vector<std::unique_ptr<Graph>> &graphs, OptimiserDatabase *optDb)
{
    LoadGraphs(LoadModel(input, size), graphs, optDb);
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

void TfLiteReader::ParseOperatorOptions(
    const std::shared_ptr<Operation> &operation, const tflite::Operator *tflite_operator, OptimiserDatabase *optDb)
{
    const auto type = tflite_operator->builtin_options_type();

    assert((type == TfLiteMapping::OpTypeToBuiltinOptions(operation->Type())) || (type == tflite::BuiltinOptions::NONE));

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
            UnFuseActivation(operation, options->fused_activation_function(), optDb);
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
            UnFuseActivation(operation, options->fused_activation_function(), optDb);
        }
        break;

        case tflite::BuiltinOptions::TransposeConvOptions:
        {
            const auto options = GetBuiltinOptions<tflite::TransposeConvOptions>(tflite_operator);
            auto weight_tensor = operation->Input(TensorUsage::Weights)->tensor;
            weight_tensor->SetAxisOrder(AxisOrder::OHWI);
            SetKernel(operation, Point2i(weight_tensor->StorageShape().Width(), weight_tensor->StorageShape().Height()),
                Point2i(options->stride_w(), options->stride_h()), Point2i(1, 1) /* no dilation */, options->padding());
            UnFuseActivation(operation, options->fused_activation_function(), optDb);
        }
        break;

        case tflite::BuiltinOptions::Pool2DOptions:
        {
            const auto options = GetBuiltinOptions<tflite::Pool2DOptions>(tflite_operator);
            SetKernel(operation, Point2i(options->filter_width(), options->filter_height()),
                Point2i(options->stride_w(), options->stride_h()), Point2i(1, 1),  // no dilation
                options->padding());
            UnFuseActivation(operation, options->fused_activation_function(), optDb);
        }
        break;

        case tflite::BuiltinOptions::FullyConnectedOptions:
        {
            const auto options = GetBuiltinOptions<tflite::FullyConnectedOptions>(tflite_operator);
            UnFuseActivation(operation, options->fused_activation_function(), optDb);
            // TODO: Are `weights_format`, `keep_num_dims` or `asymmetric_quantize_inputs` used?

            auto weight_tensor = operation->Input(TensorUsage::Weights)->tensor;
            if ( weight_tensor->AxisOrder() == AxisOrder::Unknown )
            {
                // Reshape weight tensor from (num_outputs, ..., num_inputs) to (num_outputs, 1, 1, num_inputs)
                weight_tensor->SetAxisOrder(AxisOrder::OHWI);
                const auto &shape = weight_tensor->StorageShape();
                for ( int i = 1; i < shape.Size() - 1; i++ )
                {
                    assert(shape[i] == 1);
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
            operation->Parameters().softmax.beta = options->beta();
        }
        break;

        case tflite::BuiltinOptions::ConcatenationOptions:
        {
            const auto options = GetBuiltinOptions<tflite::ConcatenationOptions>(tflite_operator);
            operation->Parameters().concat.axis = options->axis();
            UnFuseActivation(operation, options->fused_activation_function(), optDb);
        }
        break;

        case tflite::BuiltinOptions::AddOptions:
        {
            const auto options = GetBuiltinOptions<tflite::AddOptions>(tflite_operator);
            UnFuseActivation(operation, options->fused_activation_function(), optDb);
        }
        break;

        case tflite::BuiltinOptions::SubOptions:
        {
            const auto options = GetBuiltinOptions<tflite::SubOptions>(tflite_operator);
            UnFuseActivation(operation, options->fused_activation_function(), optDb);
        }
        break;

        case tflite::BuiltinOptions::DivOptions:
        {
            const auto options = GetBuiltinOptions<tflite::DivOptions>(tflite_operator);
            UnFuseActivation(operation, options->fused_activation_function(), optDb);
        }
        break;

        case tflite::BuiltinOptions::MulOptions:
        {
            const auto options = GetBuiltinOptions<tflite::MulOptions>(tflite_operator);
            UnFuseActivation(operation, options->fused_activation_function(), optDb);
        }
        break;

        case tflite::BuiltinOptions::L2NormOptions:
        {
            const auto options = GetBuiltinOptions<tflite::L2NormOptions>(tflite_operator);
            UnFuseActivation(operation, options->fused_activation_function(), optDb);
        }
        break;

        case tflite::BuiltinOptions::ReshapeOptions:
        {
            const auto options = GetBuiltinOptions<tflite::ReshapeOptions>(tflite_operator);
            const auto conn = operation->Input(TensorUsage::Params);

            if ( conn == nullptr )
            {
                // New shape specified as option. Convert to input tensor.
                auto new_shape = options->new_shape();
                assert(new_shape);
                auto tensor = std::make_shared<Tensor>("new_shape", DataType::Int32);
                tensor->SetStorageShape(Shape(new_shape->size()));
                auto buffer_base = new_shape->Data();
                int buffer_size = int(new_shape->size() * (sizeof(int32_t) / sizeof(uint8_t)));
                tensor->SetBuffer(std::make_shared<Buffer>(buffer_size, buffer_base, true));
                operation->ConnectInput(TensorUsage::Params, tensor);
            }
        }
        break;

        case tflite::BuiltinOptions::PackOptions:
        {
            const auto options = GetBuiltinOptions<tflite::PackOptions>(tflite_operator);
            operation->Parameters().pack_unpack.axis = options->axis();
        }
        break;

        case tflite::BuiltinOptions::UnpackOptions:
        {
            const auto options = GetBuiltinOptions<tflite::UnpackOptions>(tflite_operator);
            operation->Parameters().pack_unpack.axis = options->axis();
        }
        break;

        case tflite::BuiltinOptions::LeakyReluOptions:
        {
            const auto options = GetBuiltinOptions<tflite::LeakyReluOptions>(tflite_operator);
            operation->Parameters().leaky_relu.alpha = options->alpha();
        }
        break;

        case tflite::BuiltinOptions::StridedSliceOptions:
        {
            const auto options = GetBuiltinOptions<tflite::StridedSliceOptions>(tflite_operator);
            operation->Parameters().strided_slice.begin_mask = options->begin_mask();
            operation->Parameters().strided_slice.end_mask = options->end_mask();
            operation->Parameters().strided_slice.ellipsis_mask = options->ellipsis_mask();
            operation->Parameters().strided_slice.new_axis_mask = options->new_axis_mask();
            operation->Parameters().strided_slice.shrink_axis_mask = options->shrink_axis_mask();
        }
        break;

        case tflite::BuiltinOptions::SplitOptions:
        {
            int num_splits = GetBuiltinOptions<tflite::SplitOptions>(tflite_operator)->num_splits();
            assert(num_splits == operation->Outputs().size());
        }
        break;

        case tflite::BuiltinOptions::SplitVOptions:
        {
            int num_splits = GetBuiltinOptions<tflite::SplitVOptions>(tflite_operator)->num_splits();
            assert(num_splits == operation->Outputs().size());
        }
        break;

        case tflite::BuiltinOptions::SVDFOptions:
        {
            const auto options = GetBuiltinOptions<tflite::SVDFOptions>(tflite_operator);
            UnFuseActivation(operation, options->fused_activation_function(), optDb);
        }
        break;

        case tflite::BuiltinOptions::ResizeBilinearOptions:
        {
            // Convert Parameters tensor to operation attributes
            const auto options = GetBuiltinOptions<tflite::ResizeBilinearOptions>(tflite_operator);
            operation->Parameters().resize.alignCorners = options->align_corners();
            operation->Parameters().resize.halfPixelCenters = options->half_pixel_centers();
        }
        break;

        case tflite::BuiltinOptions::ResizeNearestNeighborOptions:
        {
            // Convert Parameters tensor to operation attributes
            const auto options = GetBuiltinOptions<tflite::ResizeNearestNeighborOptions>(tflite_operator);
            operation->Parameters().resize.alignCorners = options->align_corners();
            operation->Parameters().resize.halfPixelCenters = options->half_pixel_centers();
            if ( !options->align_corners() )
            {
                // Use half-pixel-centers if align-corners is false.
                // This aligns with reference kernels
                operation->Parameters().resize.halfPixelCenters = true;
            }
        }
        break;

        // Options that are not used by the compiler are not loaded in, but can be written out again via passthrough
        case tflite::BuiltinOptions::BatchMatMulOptions:
        case tflite::BuiltinOptions::GatherOptions:
        case tflite::BuiltinOptions::ShapeOptions:
        case tflite::BuiltinOptions::SqueezeOptions:
        case tflite::BuiltinOptions::ReducerOptions:
            break;

        // Empty option sets require no parsing
        case tflite::BuiltinOptions::NONE:
        case tflite::BuiltinOptions::HardSwishOptions:
        case tflite::BuiltinOptions::MaximumMinimumOptions:
        case tflite::BuiltinOptions::PadOptions:
        case tflite::BuiltinOptions::DequantizeOptions:
        case tflite::BuiltinOptions::QuantizeOptions:
        case tflite::BuiltinOptions::TransposeOptions:
        case tflite::BuiltinOptions::GatherNdOptions:
        case tflite::BuiltinOptions::ScatterNdOptions:
        case tflite::BuiltinOptions::ArgMaxOptions:
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
        case tflite::BuiltinOptions::PadV2Options:
        case tflite::BuiltinOptions::GreaterOptions:
        case tflite::BuiltinOptions::GreaterEqualOptions:
        case tflite::BuiltinOptions::LessEqualOptions:
        case tflite::BuiltinOptions::SelectOptions:
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
        case tflite::BuiltinOptions::MirrorPadOptions:
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
        case tflite::BuiltinOptions::SelectV2Options:
        case tflite::BuiltinOptions::DensifyOptions:
        case tflite::BuiltinOptions::SegmentSumOptions:
        case tflite::BuiltinOptions::CumsumOptions:
        case tflite::BuiltinOptions::CallOnceOptions:
        case tflite::BuiltinOptions::BroadcastToOptions:
        case tflite::BuiltinOptions::Rfft2dOptions:
        case tflite::BuiltinOptions::Conv3DOptions:
        case tflite::BuiltinOptions::HashtableOptions:
        case tflite::BuiltinOptions::HashtableFindOptions:
        case tflite::BuiltinOptions::HashtableImportOptions:
        case tflite::BuiltinOptions::HashtableSizeOptions:
        case tflite::BuiltinOptions::VarHandleOptions:
        case tflite::BuiltinOptions::ReadVariableOptions:
        case tflite::BuiltinOptions::AssignVariableOptions:
            // TODO
            LOG_WARN("TfLiteReader: Built-in options type '{}' is not yet implemented and will be ignored.\n",
                tflite::EnumNameBuiltinOptions(type));
            break;
        default:
            LOG_ERROR("TfLiteReader: Unrecognised built-in options type '{}'\n", int(type));
            break;
    }

    operation->SetPassthrough(tflite_operator);
}

void TfLiteReader::SetOperatorRounding(const std::shared_ptr<Operation> &operation)
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
    operation->SetRounding(roundMode);
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

}  // namespace regor
