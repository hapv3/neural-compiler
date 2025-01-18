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

#include "tflite_writer.hpp"

#include "common/logging.hpp"

#include "architecture/architecture.hpp"
#include "flatbuffer_utils.hpp"
#include "tflite_mapping.hpp"

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Specialization for sparsity dimension metadata
template<>
flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::DimensionMetadata>>> FlatbufferUtils::CopyVector(
    flatbuffers::FlatBufferBuilder &dst, const flatbuffers::Vector<flatbuffers::Offset<tflite::DimensionMetadata>> *src)
{
    if ( src )
    {
        std::vector<flatbuffers::Offset<tflite::DimensionMetadata>> offsets;
        for ( const auto &dimension_metadata : *src )
        {
            // TODO: offsets.push_back(tflite::CreateDimensionMetadata(...))
        }
        return dst.CreateVector<flatbuffers::Offset<tflite::DimensionMetadata>>(offsets);
    }
    return 0;
}

namespace regor
{

struct FlatBuffersSizeException : public std::runtime_error
{
    FlatBuffersSizeException() : std::runtime_error("FlatBuffers size") {}
};

std::unique_ptr<const uint8_t[]> TfLiteWriter::Serialise(const std::vector<std::unique_ptr<Graph>> &graphs,
    const std::vector<std::unordered_map<const Tensor *, Address>> &tensor_address_maps, int64_t &output_buffer_offset, size_t &output_buffer_size)
{
    std::unique_ptr<const uint8_t[]> ret;
    bool retryWithBufferOffset = false;

    try
    {
        ret = SerialiseImpl(graphs, tensor_address_maps, output_buffer_offset, output_buffer_size);
    }
    catch ( const FlatBuffersSizeException & )
    {
        retryWithBufferOffset = true;
    }

    if ( retryWithBufferOffset )
    {
        _opcodes.clear();
        _buffers.clear();
        _serialised_opcodes.clear();
        _serialised_subgraphs.clear();
        _serialised_buffers.clear();
        _tensors.clear();
        _serialised_operations.clear();
        _serialised_tensors.clear();
        _tensor_addresses.clear();
        _offset_buffers.clear();
        _flatbuffer.Clear();

        _useBufferOffset = true;
        ret = SerialiseImpl(graphs, tensor_address_maps, output_buffer_offset, output_buffer_size);
    }

    return ret;
}

void TfLiteWriter::CheckFlatBufferSize()
{
    if ( _flatbuffer.GetSize() >= _fbSizeCap )
    {
        throw FlatBuffersSizeException();
    }
}

std::unique_ptr<const uint8_t[]> TfLiteWriter::SerialiseImpl(const std::vector<std::unique_ptr<Graph>> &graphs,
    const std::vector<std::unordered_map<const Tensor *, Address>> &tensor_address_maps, int64_t &output_buffer_offset, size_t &output_buffer_size)
{
    // The zeroth buffer is always present and always empty
    _buffers[BufferDesc()] = 0;
    _serialised_buffers.push_back(tflite::CreateBufferDirect(_flatbuffer, nullptr));
    std::vector<flatbuffers::Offset<tflite::Metadata>> serialised_metadata;  // TODO: passthrough metadata

    for ( const auto &graph : graphs )
    {
        const auto &tensor_address_map = tensor_address_maps.at(_serialised_subgraphs.size());
        std::set<const Operation *> skip;

        for ( const auto &operation : graph->ScheduledOrder() )
        {
            if ( skip.count(operation) )
            {
                continue;
            }

            const auto tflite_operator = static_cast<const tflite::Operator *>(operation->Passthrough());
            const auto tflite_model = static_cast<const tflite::Model *>(graph->Passthrough());

            OpType type = operation->Type();
            tflite::BuiltinOperator builtin_code;
            tflite::BuiltinOptions builtin_options_type;
            if ( type == OpType::Passthrough )
            {
                assert(tflite_model);
                assert(tflite_operator);
                auto operator_codes = tflite_model->operator_codes();
                assert(operator_codes);
                builtin_code = operator_codes->Get(tflite_operator->opcode_index())->builtin_code();
                if ( builtin_code == tflite::BuiltinOperator(0) )
                {
                    int8_t deprecated_builtin_code = operator_codes->Get(tflite_operator->opcode_index())->deprecated_builtin_code();
                    builtin_code = static_cast<tflite::BuiltinOperator>(deprecated_builtin_code);
                }
                builtin_options_type = tflite_operator->builtin_options_type();
            }
            else
            {
                builtin_code = TfLiteMapping::OpTypeToBuiltinOperator(type);
                builtin_options_type = TfLiteMapping::OpTypeToBuiltinOptions(type);
            }

            // Set deprecated_builtin_code for backwards compatibility
            int8_t deprecated_builtin_code = int32_t(builtin_code) < 127 ? int8_t(builtin_code) : 127;
            OperatorCodeDesc opcode_desc = {deprecated_builtin_code, nullptr, 1, builtin_code};
            if ( tflite_model && tflite_operator )
            {
                assert(tflite_model->operator_codes());
                assert(tflite_operator->opcode_index() < tflite_model->operator_codes()->size());
                const auto opcode = tflite_model->operator_codes()->Get(tflite_operator->opcode_index());

                assert(opcode);
                opcode_desc = {opcode->deprecated_builtin_code(), opcode->custom_code() ? opcode->custom_code()->c_str() : nullptr,
                    opcode->version(), opcode->builtin_code()};
            }
            else if ( type == OpType::CustomNpuOp )
            {
                opcode_desc = {deprecated_builtin_code, "ethos-u", 1, builtin_code};
            }
            else
            {
                assert(false && "Can't handle non-CustomNpuOp ops without passthrough data");
            }

            int opcode_index;
            auto cached_opcode_desc = _opcodes.find(opcode_desc);
            if ( cached_opcode_desc != _opcodes.end() )
            {
                // Used cached OperatorCode index
                opcode_index = cached_opcode_desc->second;
            }
            else
            {
                opcode_index = int(_serialised_opcodes.size());
                _serialised_opcodes.push_back(tflite::CreateOperatorCodeDirect(_flatbuffer,
                    opcode_desc.deprecated_builtin_code, opcode_desc.custom_code, opcode_desc.version, opcode_desc.type));

                // Cache the OperatorCode index
                _opcodes[opcode_desc] = opcode_index;
            }

            std::vector<int> inputs, outputs;
            for ( const auto &tensor : SortedInputTensors(operation, type) )
            {
                // Skip placeholder tensors
                if ( graph->IsPlaceholder(tensor) ) continue;
                inputs.push_back(SerialisedTensorIndex(tensor, tensor_address_map, *graph));
            }
            for ( const auto &connection : operation->Outputs() )
            {
                const Tensor *tensor = connection.tensor.get();
                // Skip placeholder tensors
                if ( graph->IsPlaceholder(tensor) ) continue;
                outputs.push_back(SerialisedTensorIndex(tensor, tensor_address_map, *graph));
            }

            // Unused parameters are set to default or, if present in the input model, passed through unmodified.
            tflite::CustomOptionsFormat custom_options_format = tflite::CustomOptionsFormat::FLEXBUFFERS;
            flatbuffers::Offset<flatbuffers::Vector<uint8_t>> custom_options = 0;
            flatbuffers::Offset<flatbuffers::Vector<uint8_t>> mvi = 0;  // mutating_variable_inputs
            flatbuffers::Offset<flatbuffers::Vector<int32_t>> intermediates = 0;

            if ( type == OpType::CustomNpuOp )
            {
                // Could construct the flexbuffer using flexbuffers.h like this...
                // {
                //     flexbuffers::Builder builder;
                //     builder.Int(1); // CO_TYPE = 1
                //     builder.Finish();
                //     flexbuffer = builder.GetBuffer()
                // }

                // But the result would always be the same, so just jump straight there.
                std::vector<uint8_t> flexbuffer({1, 4, 1});

                custom_options = _flatbuffer.CreateVector<uint8_t>(flexbuffer);
            }
            else if ( tflite_operator )
            {
                custom_options_format = tflite_operator->custom_options_format();
                custom_options = FlatbufferUtils::CopyVector<uint8_t>(_flatbuffer, tflite_operator->custom_options());
                mvi = FlatbufferUtils::CopyVector<uint8_t>(_flatbuffer, tflite_operator->mutating_variable_inputs());
                intermediates = FlatbufferUtils::CopyVector<int32_t>(_flatbuffer, tflite_operator->intermediates());
            }

            auto serialised_inputs = _flatbuffer.CreateVector<int32_t>(inputs);
            auto serialised_outputs = _flatbuffer.CreateVector<int32_t>(outputs);
            auto serialised_options = SerialiseOptions(operation, type);

            _serialised_operations.push_back(tflite::CreateOperator(_flatbuffer, opcode_index, serialised_inputs, serialised_outputs,
                builtin_options_type, serialised_options, custom_options, custom_options_format, mvi, intermediates));
        }

        std::vector<int> inputs, outputs;

        for ( const auto &tensor : graph->Inputs() )
        {
            inputs.push_back(SerialisedTensorIndex(tensor.get(), tensor_address_map, *graph));
        }
        for ( const auto &tensor : graph->Outputs() )
        {
            if ( graph->IsPlaceholder(tensor.get()) ) continue;

            outputs.push_back(SerialisedTensorIndex(tensor.get(), tensor_address_map, *graph));
        }

        const char *subGraphName = graph->Name().empty() ? nullptr : graph->Name().c_str();
        _serialised_subgraphs.push_back(tflite::CreateSubGraphDirect(
            _flatbuffer, &_serialised_tensors, &inputs, &outputs, &_serialised_operations, subGraphName));

        _serialised_operations.clear();
        _serialised_tensors.clear();
    }

    serialised_metadata.push_back(SerialiseTensorAddresses(int(_serialised_subgraphs.size())));

    _tensors.clear();
    _tensor_addresses.clear();

    const char *_description = "Vela Optimised";

    const auto model = tflite::CreateModelDirect(_flatbuffer,
        3,  // version
        &_serialised_opcodes, &_serialised_subgraphs, _description, &_serialised_buffers,
        nullptr,  // deprecated metadata_buffer
        &serialised_metadata
        // TODO: signature_defs
    );

    tflite::FinishModelBuffer(_flatbuffer, model);

    CheckFlatBufferSize();

    // Transfer ownership of the finished buffer from the flatbuffer builder to the caller
    ResultBuffer ret(_flatbuffer);

    // Following the model, place offset tensor buffers at the end of the file
    if ( _useBufferOffset )
    {
        // Serialise buffers at the end of the file
        auto offsetBufferOffset = SerialiseOffsetBuffers(ret);

        // Fixup indirect buffer offsets via the mutable API
        FixupFbBuffers(ret.begin(), offsetBufferOffset);
    }

    return ret.release(output_buffer_size, output_buffer_offset);
}


std::vector<size_t> TfLiteWriter::SerialiseOffsetBuffers(ResultBuffer &res)
{
    // Reserve buffer
    auto align = [](size_t sz) { return (sz + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1); };

    size_t newSize = res.pos() + BUFFER_ALIGNMENT;
    for ( const auto &buf : _offset_buffers )
    {
        newSize = align(newSize) + buf.size();
    }
    res.reserve(newSize);

    std::vector<size_t> offsetBufferOffset;
    offsetBufferOffset.reserve(_offset_buffers.size());

    for ( const auto &buf : _offset_buffers )
    {
        res.align(BUFFER_ALIGNMENT);
        offsetBufferOffset.push_back(res.push(buf.data(), buf.size()));
    }
    return offsetBufferOffset;
}


void TfLiteWriter::FixupFbBuffers(uint8_t *model, const std::vector<size_t> &offsetBufferOffset)
{
    auto tflite_buffers = tflite::GetMutableModel(model)->mutable_buffers();
    assert(tflite_buffers);
    assert(tflite_buffers->size() == (offsetBufferOffset.size() + 1));
    assert(_offset_buffers.size() == offsetBufferOffset.size());
    for ( size_t i = 0; i < offsetBufferOffset.size(); i++ )
    {
        auto tflite_buffer = tflite_buffers->GetMutableObject(flatbuffers::uoffset_t(i + 1));
        tflite_buffer->mutate_offset(offsetBufferOffset[i]);
        tflite_buffer->mutate_size(_offset_buffers[i].size());
    }
}


std::vector<const Tensor *> TfLiteWriter::SortedInputTensors(const Operation *operation, OpType type)
{
    std::vector<const Tensor *> tensors;

    const auto tensorIndices = TfLiteMapping::InputTensorIndices(type);
    if ( tensorIndices.begin() != tensorIndices.end() )
    {
        // If we have tensor indices for this op type, use that tensor order
        int ifm = 0;
        for ( const auto &[type_, usage] : tensorIndices )
        {
            const auto conn = operation->Input(usage);
            tensors.push_back(conn ? conn->tensor.get() : nullptr);
            ifm += IsIFM(usage);
        }
        while ( operation->Input(MakeTensorUsage(TensorUsage::IFM, ifm)) )
        {
            tensors.push_back(operation->IFM(ifm));
            ifm++;
        }
    }
    else
    {
        // If we don't have tensor indices for this op type, use the tensor order we have
        for ( const auto &[usage, conn] : operation->Inputs().pairs() )
        {
            tensors.push_back(conn.tensor.get());
        }
    }
    return tensors;
}


int TfLiteWriter::SerialisedTensorIndex(const Tensor *tensor, const std::unordered_map<const Tensor *, Address> &addresses, const Graph &graph)
{
    if ( !tensor )  // Optional tensor not present
    {
        return -1;
    }
    else if ( _tensors.count(tensor) )  // Already serialised
    {
        return _tensors.at(tensor);
    }
    else  // Needs serialising
    {
        const int index = int(_serialised_tensors.size());
        _tensors[tensor] = index;
        _serialised_tensors.push_back(SerialiseTensor(tensor, graph));

        auto address = addresses.find(tensor);
        if ( address == addresses.end() )
        {
            _tensor_addresses.push_back(-1);
        }
        else
        {
            assert(std::abs(address->second) <= Address(std::numeric_limits<int32_t>::max()) && "Tensor address overflow");
            _tensor_addresses.push_back(int32_t(address->second));
        }
        return index;
    }
}


flatbuffers::Offset<tflite::Tensor> TfLiteWriter::SerialiseTensor(const Tensor *tensor, const Graph &graph)
{
    auto tflite_shape = tensor->StorageShape().ToList<int>();
    std::vector<float> quant_min;
    std::vector<float> quant_max;
    std::vector<float> scale_f32;
    std::vector<int64_t> zeroPoints;
    int dimension = 0;

    // Unused parameters are set to default or, if present in the input model, passed through unmodified
    tflite::QuantizationDetails custom_quantization = tflite::QuantizationDetails::NONE;
    flatbuffers::Offset<void> custom_quantization_details = 0;
    bool is_variable = graph.IsPersistent(tensor);
    flatbuffers::Offset<tflite::SparsityParameters> sparsity = 0;
    std::vector<int> shape_signature;

    if ( tensor->Passthrough() )
    {
        const auto tflite_tensor = static_cast<const tflite::Tensor *>(tensor->Passthrough());
        const DataType type = TfLiteMapping::TensorTypeToDataType(tflite_tensor->type());

        if ( tflite_tensor->quantization() )
        {
            if ( tflite_tensor->quantization()->scale() && tflite_tensor->quantization()->zero_point() )
            {
                quant_min = FlatbufferUtils::LoadVector<float>(tflite_tensor->quantization()->min());
                quant_max = FlatbufferUtils::LoadVector<float>(tflite_tensor->quantization()->max());
                scale_f32 = FlatbufferUtils::LoadVector<float>(tflite_tensor->quantization()->scale());
                zeroPoints = FlatbufferUtils::LoadVector<int64_t>(tflite_tensor->quantization()->zero_point());
                dimension = tflite_tensor->quantization()->quantized_dimension();
            }

            custom_quantization = tflite_tensor->quantization()->details_type();
            if ( custom_quantization == tflite::QuantizationDetails::CustomQuantization )
            {
                if ( tflite_tensor->quantization()->details() )
                {
                    // TODO: custom_quantization_details
                }
            }
        }

        is_variable = tflite_tensor->is_variable();

        if ( tflite_tensor->sparsity() )
        {
            auto traversal_order = FlatbufferUtils::CopyVector<int32_t>(_flatbuffer, tflite_tensor->sparsity()->traversal_order());
            auto block_map = FlatbufferUtils::CopyVector<int32_t>(_flatbuffer, tflite_tensor->sparsity()->block_map());
            auto dim_metadata = FlatbufferUtils::CopyVector<flatbuffers::Offset<tflite::DimensionMetadata>>(
                _flatbuffer, tflite_tensor->sparsity()->dim_metadata());

            sparsity = tflite::CreateSparsityParameters(_flatbuffer, traversal_order, block_map, dim_metadata);
        }

        shape_signature = FlatbufferUtils::LoadVector<int>(tflite_tensor->shape_signature());
        tflite_shape = FlatbufferUtils::LoadVector<int>(tflite_tensor->shape());
    }

    int buffer_index = 0;  // Default to the empty buffer at index 0
    if ( tensor->IsConstant() )
    {
        const auto buffer = tensor->View().Buffer();
        const auto descriptor = BufferDesc(buffer);
        const auto it = _buffers.find(descriptor);
        if ( it == _buffers.end() )
        {
            buffer_index = int(_serialised_buffers.size());
            _buffers[descriptor] = buffer_index;
            SerialiseTensorBuffer(tensor);
        }
        else  // Buffer has already been serialised - just reference it
        {
            buffer_index = it->second;
        }
    }

    flatbuffers::Offset<tflite::QuantizationParameters> quantization = 0;
    if ( !scale_f32.empty() && !zeroPoints.empty() )
    {
        quantization = tflite::CreateQuantizationParametersDirect(_flatbuffer, &quant_min, &quant_max, &scale_f32,
            &zeroPoints, custom_quantization, custom_quantization_details, dimension);
    }

    return tflite::CreateTensorDirect(_flatbuffer, tflite_shape.size() ? &tflite_shape : nullptr,
        TfLiteMapping::DataTypeToTensorType(tensor->Type()), buffer_index, tensor->Name().c_str(), quantization,
        is_variable, sparsity, shape_signature.size() ? &shape_signature : nullptr);
}

template<typename T>
static const T *GetBuiltinOptions(const tflite::Operator *tflite_operator)
{
    const auto options = tflite_operator->builtin_options_as<T>();
    assert(options);
    return options;
}

flatbuffers::Offset<void> TfLiteWriter::SerialiseOptions(const Operation *operation, OpType opType)
{
    if ( opType == OpType::CustomNpuOp )
    {
        return 0;
    }

    flatbuffers::Offset<void> offset = 0;
    const tflite::Operator *const passthrough = static_cast<const tflite::Operator *>(operation->Passthrough());
    assert(passthrough);
    const auto type = passthrough->builtin_options_type();

    switch ( type )
    {
        case tflite::BuiltinOptions::NONE:
            break;

        case tflite::BuiltinOptions::Conv2DOptions:
        {
            assert(passthrough->builtin_options_as_Conv2DOptions());
            tflite::ActivationFunctionType fused_activation_function = passthrough->builtin_options_as_Conv2DOptions()->fused_activation_function();
            const auto kernel = TfLiteKernel(*operation->Kernel());
            const auto typed_offset = tflite::CreateConv2DOptions(_flatbuffer, kernel.padding, kernel.stride_w,
                kernel.stride_h, fused_activation_function, kernel.dilation_w_factor, kernel.dilation_h_factor);
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::DepthwiseConv2DOptions:
        {
            assert(passthrough->builtin_options_as_DepthwiseConv2DOptions());
            tflite::ActivationFunctionType fused_activation_function =
                passthrough->builtin_options_as_DepthwiseConv2DOptions()->fused_activation_function();
            const auto kernel = TfLiteKernel(*operation->Kernel());
            const auto typed_offset = tflite::CreateDepthwiseConv2DOptions(_flatbuffer, kernel.padding, kernel.stride_w,
                kernel.stride_h, kernel.depth_multiplier, fused_activation_function, kernel.dilation_w_factor, kernel.dilation_h_factor);
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::TransposeConvOptions:
        {
            assert(passthrough->builtin_options_as_TransposeConvOptions());
            tflite::ActivationFunctionType fused_activation_function =
                passthrough->builtin_options_as_TransposeConvOptions()->fused_activation_function();

            const auto kernel = TfLiteKernel(*operation->Kernel());
            const auto typed_offset = tflite::CreateTransposeConvOptions(
                _flatbuffer, kernel.padding, kernel.stride_w, kernel.stride_h, fused_activation_function);
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::Pool2DOptions:
        {
            assert(passthrough->builtin_options_as_Pool2DOptions());
            tflite::ActivationFunctionType fused_activation_function = passthrough->builtin_options_as_Pool2DOptions()->fused_activation_function();
            const auto kernel = TfLiteKernel(*operation->Kernel());
            const auto typed_offset = tflite::CreatePool2DOptions(_flatbuffer, kernel.padding, kernel.stride_w,
                kernel.stride_h, kernel.filter_w, kernel.filter_h, fused_activation_function);
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::FullyConnectedOptions:
        {
            assert(passthrough->builtin_options_as_FullyConnectedOptions());
            tflite::ActivationFunctionType fused_activation_function =
                passthrough->builtin_options_as_FullyConnectedOptions()->fused_activation_function();
            const auto typed_offset = tflite::CreateFullyConnectedOptions(_flatbuffer, fused_activation_function
                // TODO: weights_format,
                // TODO: keep_num_dims,
                // TODO: asymmetric_quantize_inputs
            );
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::SoftmaxOptions:
        {
            const auto options = GetBuiltinOptions<tflite::SoftmaxOptions>(passthrough);
            offset = tflite::CreateSoftmaxOptions(_flatbuffer, options->beta()).Union();
        }
        break;

        case tflite::BuiltinOptions::ConcatenationOptions:
        {
            assert(passthrough->builtin_options_as_ConcatenationOptions());
            tflite::ActivationFunctionType fused_activation_function =
                passthrough->builtin_options_as_ConcatenationOptions()->fused_activation_function();
            const auto options = GetBuiltinOptions<tflite::ConcatenationOptions>(passthrough);
            const auto typed_offset = tflite::CreateConcatenationOptions(_flatbuffer, options->axis(), fused_activation_function);
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::AddOptions:
        {
            assert(passthrough->builtin_options_as_AddOptions());
            tflite::ActivationFunctionType fused_activation_function = passthrough->builtin_options_as_AddOptions()->fused_activation_function();
            const auto typed_offset = tflite::CreateAddOptions(_flatbuffer, fused_activation_function,
                GetBuiltinOptions<tflite::AddOptions>(passthrough)->pot_scale_int16());
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::SubOptions:
        {
            assert(passthrough->builtin_options_as_SubOptions());
            tflite::ActivationFunctionType fused_activation_function = passthrough->builtin_options_as_SubOptions()->fused_activation_function();
            const auto typed_offset = tflite::CreateSubOptions(_flatbuffer, fused_activation_function,
                GetBuiltinOptions<tflite::SubOptions>(passthrough)->pot_scale_int16());
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::MulOptions:
        {
            assert(passthrough->builtin_options_as_MulOptions());
            tflite::ActivationFunctionType fused_activation_function = passthrough->builtin_options_as_MulOptions()->fused_activation_function();
            offset = tflite::CreateMulOptions(_flatbuffer, fused_activation_function).Union();
        }
        break;

        case tflite::BuiltinOptions::DivOptions:
        {
            assert(passthrough->builtin_options_as_DivOptions());
            tflite::ActivationFunctionType fused_activation_function = passthrough->builtin_options_as_DivOptions()->fused_activation_function();
            offset = tflite::CreateDivOptions(_flatbuffer, fused_activation_function).Union();
        }
        break;

        case tflite::BuiltinOptions::L2NormOptions:
        {
            assert(passthrough->builtin_options_as_L2NormOptions());
            tflite::ActivationFunctionType fused_activation_function = passthrough->builtin_options_as_L2NormOptions()->fused_activation_function();
            offset = tflite::CreateL2NormOptions(_flatbuffer, fused_activation_function).Union();
        }
        break;

        case tflite::BuiltinOptions::ReshapeOptions:
        {
            // Replicate parameter tensor as ReshapeOptions
            const auto tensor = operation->Input(TensorUsage::Params)->tensor;
            if ( tensor->Type() != DataType::Int32 ) throw std::runtime_error("unexpected tensor type");

            BufferReader<int> reader = tensor->View().Values<int32_t, int>();
            std::vector<int> temp;
            std::copy(reader.begin(), reader.end(), std::back_inserter(temp));
            const auto new_shape = _flatbuffer.CreateVector<int>(temp);
            offset = tflite::CreateReshapeOptions(_flatbuffer, new_shape).Union();
        }
        break;

        case tflite::BuiltinOptions::SqueezeOptions:
        {
            const auto options = GetBuiltinOptions<tflite::SqueezeOptions>(passthrough);
            const auto typed_offset = tflite::CreateSqueezeOptions(
                _flatbuffer, FlatbufferUtils::CopyVector<int32_t>(_flatbuffer, options->squeeze_dims()));
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::PackOptions:
        {
            const auto options = GetBuiltinOptions<tflite::PackOptions>(passthrough);
            const auto typed_offset = tflite::CreatePackOptions(_flatbuffer, options->values_count(), options->axis());
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::UnpackOptions:
        {
            const auto options = GetBuiltinOptions<tflite::UnpackOptions>(passthrough);
            const auto typed_offset = tflite::CreateUnpackOptions(_flatbuffer, options->num(), options->axis());
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::LeakyReluOptions:
        {
            const auto options = GetBuiltinOptions<tflite::LeakyReluOptions>(passthrough);
            offset = tflite::CreateLeakyReluOptions(_flatbuffer, options->alpha()).Union();
        }
        break;

        case tflite::BuiltinOptions::ShapeOptions:
        {
            const auto out_type = GetBuiltinOptions<tflite::ShapeOptions>(passthrough)->out_type();
            offset = tflite::CreateShapeOptions(_flatbuffer, out_type).Union();
        }
        break;

        case tflite::BuiltinOptions::StridedSliceOptions:
        {
            const auto options = GetBuiltinOptions<tflite::StridedSliceOptions>(passthrough);
            const auto typed_offset = tflite::CreateStridedSliceOptions(_flatbuffer, options->begin_mask(),
                options->end_mask(), options->ellipsis_mask(), options->new_axis_mask(), options->shrink_axis_mask());
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::SplitOptions:
        {
            offset = tflite::CreateSplitOptions(_flatbuffer, int(operation->Outputs().size())).Union();
        }
        break;

        case tflite::BuiltinOptions::SplitVOptions:
        {
            offset = tflite::CreateSplitVOptions(_flatbuffer, int(operation->Outputs().size())).Union();
        }
        break;

        case tflite::BuiltinOptions::ReducerOptions:
        {
            const auto options = GetBuiltinOptions<tflite::ReducerOptions>(passthrough);
            const auto typed_offset = tflite::CreateReducerOptions(_flatbuffer, options->keep_dims());
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::SVDFOptions:
        {
            assert(passthrough->builtin_options_as_SVDFOptions());
            tflite::ActivationFunctionType fused_activation_function = passthrough->builtin_options_as_SVDFOptions()->fused_activation_function();
            const auto options = GetBuiltinOptions<tflite::SVDFOptions>(passthrough);
            const auto typed_offset = tflite::CreateSVDFOptions(
                _flatbuffer, options->rank(), fused_activation_function, options->asymmetric_quantize_inputs());
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::BatchMatMulOptions:
        {
            const auto options = GetBuiltinOptions<tflite::BatchMatMulOptions>(passthrough);
            if ( options )
            {
                const auto typed_offset = tflite::CreateBatchMatMulOptions(
                    _flatbuffer, options->adj_x(), options->adj_y(), options->asymmetric_quantize_inputs());
                offset = typed_offset.Union();
            }
        }
        break;
        case tflite::BuiltinOptions::GatherOptions:
        {
            const auto options = GetBuiltinOptions<tflite::GatherOptions>(passthrough);
            const auto typed_offset = tflite::CreateGatherOptions(_flatbuffer, options->axis(), options->batch_dims());
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::ResizeBilinearOptions:
        {
            const auto options = GetBuiltinOptions<tflite::ResizeBilinearOptions>(passthrough);
            if ( options )
            {
                const auto typed_offset = tflite::CreateResizeBilinearOptions(
                    _flatbuffer, options->align_corners(), options->half_pixel_centers());
                offset = typed_offset.Union();
            }
            else
            {
                offset = tflite::CreateResizeBilinearOptions(_flatbuffer).Union();
            }
        }
        break;

        case tflite::BuiltinOptions::CallOnceOptions:
        {
            const auto options = GetBuiltinOptions<tflite::CallOnceOptions>(passthrough);
            const auto typed_offset = tflite::CreateCallOnceOptions(_flatbuffer, options->init_subgraph_index());
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::VarHandleOptions:
        {
            const auto options = GetBuiltinOptions<tflite::VarHandleOptions>(passthrough);
            const auto container = _flatbuffer.CreateString(options->container());
            const auto shared_name = _flatbuffer.CreateString(options->shared_name());
            const auto typed_offset = tflite::CreateVarHandleOptions(_flatbuffer, container, shared_name);
            offset = typed_offset.Union();
        }
        break;

        case tflite::BuiltinOptions::WhileOptions:
        {
            const auto options = GetBuiltinOptions<tflite::WhileOptions>(passthrough);
            if ( options )
            {
                const auto typed_offset = tflite::CreateWhileOptions(
                    _flatbuffer, options->cond_subgraph_index(), options->body_subgraph_index());
                offset = typed_offset.Union();
            }
        }
        break;

        // Empty option sets can all be written as if they were QuantizeOptions
        case tflite::BuiltinOptions::HardSwishOptions:
        case tflite::BuiltinOptions::MaximumMinimumOptions:
        case tflite::BuiltinOptions::PadOptions:
        case tflite::BuiltinOptions::DequantizeOptions:
        case tflite::BuiltinOptions::QuantizeOptions:
        case tflite::BuiltinOptions::TransposeOptions:
        case tflite::BuiltinOptions::GatherNdOptions:
        case tflite::BuiltinOptions::ScatterNdOptions:
        case tflite::BuiltinOptions::ArgMaxOptions:
        case tflite::BuiltinOptions::AssignVariableOptions:
        case tflite::BuiltinOptions::ReadVariableOptions:
        case tflite::BuiltinOptions::SelectOptions:
        case tflite::BuiltinOptions::SelectV2Options:
        {
            offset = tflite::CreateQuantizeOptions(_flatbuffer).Union();
        }
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
        case tflite::BuiltinOptions::ResizeNearestNeighborOptions:
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
            LOG_WARN("TfLiteWriter: Built-in options type '{}' is not yet implemented and will be set to default.\n",
                tflite::EnumNameBuiltinOptions(type));
            break;
        default:
            LOG_ERROR("TfLiteWriter: Unrecognised built-in options type '{}'\n", int(type));
            break;
    }
    return offset;
}

flatbuffers::Offset<tflite::Metadata> TfLiteWriter::SerialiseTensorAddresses(int subgraphs)
{
    const int32_t version = 0;
    const auto num_tensors = int32_t(_tensor_addresses.size());
    const auto buffer_index = int32_t(_serialised_buffers.size());

    _tensor_addresses.insert(_tensor_addresses.begin(), {version, subgraphs, num_tensors});

    const auto buffer_base = reinterpret_cast<uint8_t *>(_tensor_addresses.data());
    const auto buffer_size = _tensor_addresses.size() * (sizeof(int32_t) / sizeof(uint8_t));
    _serialised_buffers.push_back(SerialiseBuffer(buffer_base, buffer_size));
    if ( _useBufferOffset )
    {
        _offset_buffers.emplace_back(buffer_base, buffer_size);
    }

    return tflite::CreateMetadataDirect(_flatbuffer, "OfflineMemoryAllocation", uint32_t(buffer_index));
}

void TfLiteWriter::SerialiseTensorBuffer(const Tensor *tensor)
{
    if ( tensor->Type() == DataType::Int48 )
    {  // Translate values
        const auto values = tensor->View().Values<int48_t, int64_t>();
        auto v = std::make_unique<std::vector<int64_t>>(values.begin(), values.end());
        const auto size = v->size() * sizeof(int64_t);
        _serialised_buffers.emplace_back(SerialiseBuffer(reinterpret_cast<const uint8_t *>(v->data()), size));
        if ( _useBufferOffset )
        {
            _offset_buffers.emplace_back(std::move(v));
        }
    }
    else
    {
        const auto buffer = tensor->View().Buffer();
        _serialised_buffers.emplace_back(SerialiseBuffer(buffer));
        if ( _useBufferOffset )
        {
            _offset_buffers.emplace_back(buffer);
        }
    }
}

flatbuffers::Offset<tflite::Buffer> TfLiteWriter::SerialiseBuffer(const Buffer *buffer)
{
    return SerialiseBuffer(buffer->Data<uint8_t>(), buffer->Size());
}

flatbuffers::Offset<tflite::Buffer> TfLiteWriter::SerialiseBuffer(const uint8_t *data, size_t size)
{
    flatbuffers::Offset<tflite::Buffer> ret;

    _flatbuffer.ForceVectorAlignment(size, sizeof(uint8_t), BUFFER_ALIGNMENT);  // 16-byte alignment
    if ( _useBufferOffset )
    {
        _flatbuffer.ForceDefaults(true);
        ret = tflite::CreateBuffer(_flatbuffer);
        _flatbuffer.ForceDefaults(false);
    }
    else
    {
        ret = tflite::CreateBuffer(_flatbuffer, _flatbuffer.CreateVector<uint8_t>(data, size));
    }
    CheckFlatBufferSize();

    return ret;
}

}  // namespace regor
