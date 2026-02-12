//
// SPDX-FileCopyrightText: Copyright 2021-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "flatbuffers/minireflect.h"

namespace regor
{


int64_t TfLiteWriter::PrepareGraph(Graph *graph, int64_t &bufferSize, int64_t &operatorSize, int64_t &tensorSize)
{
    const size_t TENSOR_SIZE_ESTIMATE = tflite::TensorTypeTable()->num_elems * (sizeof(uint32_t) + sizeof(int16_t));
    const size_t OPERATOR_SIZE_ESTIMATE = tflite::OperatorTypeTable()->num_elems * (sizeof(uint32_t) + sizeof(int16_t));

    // Add in model's passthrough metadata size
    const auto *tflite_model = static_cast<const tflite::Model *>(graph->Passthrough());
    if ( tflite_model && (bufferSize == 0) )
    {
        const auto *tflite_metadata = tflite_model->metadata();
        const auto *tflite_buffers = tflite_model->buffers();
        if ( tflite_metadata && tflite_buffers )
        {
            for ( auto it = tflite_metadata->begin(); it != tflite_metadata->end(); it++ )
            {
                const auto buffer = (*it)->buffer();
                if ( buffer >= tflite_buffers->size() ) continue;  // non present buffer reference
                bufferSize += tflite_buffers->Get(buffer)->size();
            }
        }
    }

    // Prepare by extracting the unique tensors and unique buffers from the graph
    for ( const auto &op : graph->ScheduledOrder() )
    {
        const tflite::Operator *tfliteOperator = static_cast<const tflite::Operator *>(op->Passthrough());
        if ( tfliteOperator )
        {
            operatorSize += FlatbufferUtils::MeasureTable(
                reinterpret_cast<const flatbuffers::Table *>(tfliteOperator), tflite::Operator::MiniReflectTypeTable());
        }
        else
        {
            operatorSize += OPERATOR_SIZE_ESTIMATE;
        }
        for ( const ordered_map<TensorUsage, TensorConnection> *list : {&op->Inputs(), &op->Outputs()} )
        {
            for ( const auto &conn : *list )
            {
                Tensor *tensor = conn.tensor.get();
                if ( tensor->IsConstant() )
                {
                    BufferDesc desc(tensor->Buffer());
                    if ( !_buffers.contains(desc) )
                    {
                        _buffers.emplace(desc, -1);
                        bufferSize += tensor->Buffer()->Size();
                    }
                }
                if ( !graph->IsPlaceholder(tensor) )
                {
                    auto inserted = _tensors.emplace(tensor, -1);
                    if ( inserted.second )
                    {
                        tensorSize += TENSOR_SIZE_ESTIMATE;

                        if ( tensor->Passthrough() )
                        {
                            const auto tflite_tensor = static_cast<const tflite::Tensor *>(tensor->Passthrough());
                            auto *quant = tflite_tensor->quantization();
                            if ( quant )
                            {
                                tensorSize += quant->zero_point() ? quant->zero_point()->size() * sizeof(int64_t) : 0;
                                tensorSize += quant->max() ? quant->max()->size() * sizeof(float) : 0;
                                tensorSize += quant->min() ? quant->min()->size() * sizeof(float) : 0;
                                tensorSize += quant->scale() ? quant->scale()->size() * sizeof(float) : 0;
                            }
                        }
                    }
                }
            }
        }
    }

    return operatorSize + tensorSize + bufferSize;
}

std::unique_ptr<const uint8_t[]> TfLiteWriter::Serialise(const std::vector<std::unique_ptr<Graph>> &graphs,
    const std::vector<TensorAddressMap> &tensor_address_maps, int64_t &output_buffer_offset, size_t &output_buffer_size)
{
    const size_t STATIC_STRUCTURAL_ESTIMATE = 16;

    std::unique_ptr<const uint8_t[]> ret;
    bool retryWithBufferOffset = false;

    int64_t bufferSize = 0;
    int64_t operatorSize = 0;
    int64_t tensorSize = 0;
    int64_t estimatedSize = STATIC_STRUCTURAL_ESTIMATE;

    // The zeroth buffer is always present and always empty
    _buffers.emplace(BufferDesc(), 0);
    _serialised_buffers.push_back(tflite::CreateBufferDirect(_flatbuffer, nullptr));

    // Prepare for serialisation by measurnig the size of graph components.
    for ( const auto &graph : graphs )
    {
        estimatedSize += PrepareGraph(graph.get(), bufferSize, operatorSize, tensorSize);
    }

    // Pre-serialise all the tensor buffers
    const size_t structureSize = estimatedSize - bufferSize;
    const size_t sizeLimit = (_fbSizeCap > structureSize) ? _fbSizeCap - structureSize : _fbSizeCap;

    for ( auto buf : _buffers.pairs() )
    {
        // Skip the zeroth buffer during serialisation fixup
        if ( buf.second >= 0 ) continue;
        // Buffers can be represented outside the flatbuffer. Write buffers early, ensuring that
        // there's still space within the size limit for the remaining data and structural elements.
        bool useOffset = size_t(_flatbuffer.GetSize()) + buf.first.size > sizeLimit;
        buf.second = int(_serialised_buffers.size());
        _serialised_buffers.push_back(SerialiseBuffer(buf.first.data, buf.first.size, useOffset));
    }

    ret = SerialiseImpl(graphs, tensor_address_maps, output_buffer_offset, output_buffer_size);

    return ret;
}

std::unique_ptr<const uint8_t[]> TfLiteWriter::SerialiseImpl(const std::vector<std::unique_ptr<Graph>> &graphs,
    const std::vector<TensorAddressMap> &tensor_address_maps, int64_t &output_buffer_offset, size_t &output_buffer_size)
{
    std::vector<flatbuffers::Offset<tflite::Metadata>> serialised_metadata;

    for ( const auto &graph : graphs )
    {
        const auto &tensor_address_map = tensor_address_maps.at(_serialised_subgraphs.size());

        for ( const auto &operation : graph->ScheduledOrder() )
        {
            const auto tflite_operator = static_cast<const tflite::Operator *>(operation->Passthrough());
            const auto tflite_model = static_cast<const tflite::Model *>(graph->Passthrough());

            OpType type = operation->Type();
            tflite::BuiltinOperator builtin_code;
            tflite::BuiltinOptions builtin_options_type;
            tflite::BuiltinOptions2 builtin_options_2_type;
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
                builtin_options_2_type = tflite_operator->builtin_options_2_type();
            }
            else
            {
                builtin_code = TfLiteMapping::OpTypeToBuiltinOperator(type);
                builtin_options_type = TfLiteMapping::BuiltinOperatorToBuiltinOptions(builtin_code);
                builtin_options_2_type = TfLiteMapping::BuiltinOperatorToBuiltinOptions2(builtin_code);
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

            std::vector<int> inputs, outputs, intermediates;
            for ( const auto &tensor : SortedInputTensors(operation, type) )
            {
                // Skip placeholder tensors
                if ( graph->IsPlaceholder(tensor) ) continue;
                if ( (operation->UsageOfTensor(tensor) & TensorUsage::TypeMask) == TensorUsage::Scratch )
                {
                    // Scratch usage means this is an intermediate tensor
                    intermediates.push_back(SerialisedTensorIndex(tensor, tensor_address_map, *graph));
                }
                else
                {
                    inputs.push_back(SerialisedTensorIndex(tensor, tensor_address_map, *graph));
                }
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
            uint64_t large_custom_options_offset = 0;
            uint64_t large_custom_options_size = 0;

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
            }

            auto serialised_inputs = _flatbuffer.CreateVector<int32_t>(inputs);
            auto serialised_outputs = _flatbuffer.CreateVector<int32_t>(outputs);
            auto serialised_options = SerialiseOptions(operation, type);
            auto serialised_options2 = SerialiseOptions2(operation, type);

            // Flatbuffer vectors have a length prefix before the payload. If the op doesn't have any intermediates
            // the field can be omitted entirely by setting below variable to 0 instead of creating an empty vector.
            auto serialised_intermediates = !intermediates.empty() ? _flatbuffer.CreateVector<int32_t>(intermediates) : 0;

            _serialised_operations.push_back(tflite::CreateOperator(_flatbuffer, opcode_index, serialised_inputs, serialised_outputs,
                builtin_options_type, serialised_options, custom_options, custom_options_format, mvi, serialised_intermediates,
                large_custom_options_offset, large_custom_options_size, builtin_options_2_type, serialised_options2));
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

    bool hasOfflineMemoryAllocation = false;
    if ( graphs.size() > 0 )
    {
        const auto tflite_model = static_cast<const tflite::Model *>(graphs[0]->Passthrough());
        if ( tflite_model )
        {
            const auto *tflite_metadata = tflite_model->metadata();
            const auto *tflite_buffers = tflite_model->buffers();
            if ( tflite_metadata && tflite_buffers )
            {
                for ( auto it = tflite_metadata->begin(); it != tflite_metadata->end(); it++ )
                {
                    const auto buffer = (*it)->buffer();
                    if ( buffer >= tflite_buffers->size() ) continue;  // Invalid buffer
                    const auto name = (*it)->name();
                    if ( !name ) continue;  // Invalid name
                    const auto data = FlatbufferUtils::CopyVector(_flatbuffer, tflite_buffers->Get(buffer)->data());
                    const auto offset = tflite_buffers->Get(buffer)->offset();
                    const auto size = tflite_buffers->Get(buffer)->size();
                    // Copy buffer
                    _serialised_buffers.push_back(tflite::CreateBuffer(_flatbuffer, data, offset, size));
                    // Copy metadata
                    serialised_metadata.push_back(tflite::CreateMetadata(
                        _flatbuffer, _flatbuffer.CreateString(name), uint32_t(_serialised_buffers.size() - 1)));
                    // If we copied a OfflineMemoryAllocation, don't create a new one later on
                    if ( name->str() == "OfflineMemoryAllocation" ) hasOfflineMemoryAllocation = true;
                }
            }
        }
    }

    if ( !_skipOfflineMemoryAllocation && !hasOfflineMemoryAllocation )
    {
        serialised_metadata.push_back(SerialiseTensorAddresses(_serialised_subgraphs.size()));
    }

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

    // Transfer ownership of the finished buffer from the flatbuffer builder to the caller
    ResultBuffer ret(_flatbuffer);

    // Following the model, place offset tensor buffers at the end of the file
    if ( !_offset_buffers.empty() )
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
        for ( const auto &tensorIndex : tensorIndices )
        {
            const auto &usage = tensorIndex.second;
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
        for ( const auto &tensorEntry : operation->Inputs().pairs() )
        {
            tensors.push_back(tensorEntry.second.tensor.get());
        }
    }
    return tensors;
}


int TfLiteWriter::SerialisedTensorIndex(const Tensor *tensor, const TensorAddressMap &addresses, const Graph &graph)
{
    if ( !tensor )  // Optional tensor not present
    {
        return -1;
    }

    auto inserted = _tensors.emplace(tensor, -1);
    int index = inserted.first->second;
    if ( index < 0 )
    {
        inserted.first->second = index = int(_serialised_tensors.size());
        _serialised_tensors.push_back(SerialiseTensor(tensor, graph));

        auto address = addresses.find(tensor->Uid());
        if ( address == addresses.end() )
        {
            _tensor_addresses.push_back(-1);
        }
        else
        {
            Address x = address->second.address;
            assert(std::abs(x) <= Address(std::numeric_limits<int32_t>::max()) && "Tensor address overflow");
            _tensor_addresses.push_back(int32_t(x));
        }
    }

    return index;
}


flatbuffers::Offset<tflite::Tensor> TfLiteWriter::SerialiseTensor(const Tensor *tensor, const Graph &graph)
{
    auto tflite_shape = tensor->StorageShape().ToList<int>();

    // Unused parameters are set to default or, if present in the input model, passed through unmodified
    bool is_variable = graph.IsPersistent(tensor);
    flatbuffers::Offset<tflite::SparsityParameters> sparsity = 0;
    std::vector<int> shape_signature;
    bool has_rank = false;
    std::vector<flatbuffers::Offset<tflite::VariantSubType>> variant_tensors;
    flatbuffers::Offset<tflite::QuantizationParameters> quantization = 0;

    if ( tensor->Passthrough() )
    {
        const auto tflite_tensor = static_cast<const tflite::Tensor *>(tensor->Passthrough());

        if ( tflite_tensor->quantization() )
        {
            quantization = FlatbufferUtils::CopyTable(_flatbuffer, tflite_tensor->quantization());
        }

        is_variable = tflite_tensor->is_variable();

        if ( tflite_tensor->sparsity() )
        {
            sparsity = FlatbufferUtils::CopyTable(_flatbuffer, tflite_tensor->sparsity());
        }

        if ( tflite_tensor->variant_tensors() )
        {
            variant_tensors = FlatbufferUtils::CopyVectorOfTables(_flatbuffer, tflite_tensor->variant_tensors());
        }

        shape_signature = FlatbufferUtils::LoadVector<int>(tflite_tensor->shape_signature());
        tflite_shape = FlatbufferUtils::LoadVector<int>(tflite_tensor->shape());
        has_rank = tflite_tensor->has_rank();
    }

    int buffer_index = 0;  // Default to the empty buffer at index 0
    if ( tensor->IsConstant() )
    {
        const auto it = _buffers.find(BufferDesc(tensor->Buffer()));
        if ( it != _buffers.end() ) buffer_index = *it;
        assert(buffer_index >= 0 && "Buffer not yet serialised");
    }

    return tflite::CreateTensorDirect(_flatbuffer, tflite_shape.size() ? &tflite_shape : nullptr,
        TfLiteMapping::DataTypeToTensorType(tensor->Type()), buffer_index, tensor->Name().c_str(), quantization,
        is_variable, sparsity, shape_signature.size() ? &shape_signature : nullptr, has_rank, &variant_tensors);
}

template<typename T>
static const T *GetBuiltinOptions(const tflite::Operator *tflite_operator)
{
    const auto options = tflite_operator->builtin_options_as<T>();
    assert(options);
    return options;
}

// Serialize builtin_options and return offset to it
flatbuffers::Offset<void> TfLiteWriter::SerialiseOptions(const Operation *operation, OpType opType)
{
    if ( opType == OpType::CustomNpuOp )
    {
        return 0;
    }

    const auto tfliteOperator = static_cast<const tflite::Operator *>(operation->Passthrough());
    assert(tfliteOperator);
    const tflite::BuiltinOptions unionMemberType = tfliteOperator->builtin_options_type();
    if ( unionMemberType == tflite::BuiltinOptions::NONE )
    {
        return 0;
    }

    const auto *unionTypeTable = tflite::BuiltinOptionsTypeTable();
    assert(unionTypeTable->st == flatbuffers::SequenceType::ST_UNION);
    const size_t unionMemberIndex = size_t(unionMemberType) - 1;  // 0 is reserved for NONE
    assert(unionMemberIndex < unionTypeTable->num_elems);
    const auto *unionMemberTypeTable = unionTypeTable->type_refs[unionMemberIndex]();

    // The builtin options union member
    const auto *unionMember = static_cast<const flatbuffers::Table *>(tfliteOperator->builtin_options());
    assert(unionMember);

    return FlatbufferUtils::CopyTable(_flatbuffer, unionMember, unionMemberTypeTable).Union();
}

flatbuffers::Offset<void> TfLiteWriter::SerialiseOptions2(const Operation *operation, OpType opType)
{
    if ( opType == OpType::CustomNpuOp )
    {
        return 0;
    }

    const auto tfliteOperator = static_cast<const tflite::Operator *>(operation->Passthrough());
    assert(tfliteOperator);
    const tflite::BuiltinOptions2 unionMemberType = tfliteOperator->builtin_options_2_type();
    if ( unionMemberType == tflite::BuiltinOptions2::NONE )
    {
        return 0;
    }

    const auto *unionTypeTable = tflite::BuiltinOptions2TypeTable();
    assert(unionTypeTable->st == flatbuffers::SequenceType::ST_UNION);
    const size_t unionMemberIndex = size_t(unionMemberType) - 1;  // 0 is reserved for NONE
    assert(unionMemberIndex < unionTypeTable->num_elems);
    const auto *unionMemberTypeTable = unionTypeTable->type_refs[unionMemberIndex]();

    // The builtin options union member
    const auto *unionMember = static_cast<const flatbuffers::Table *>(tfliteOperator->builtin_options_2());
    assert(unionMember);

    return FlatbufferUtils::CopyTable(_flatbuffer, unionMember, unionMemberTypeTable).Union();
}

flatbuffers::Offset<tflite::Metadata> TfLiteWriter::SerialiseTensorAddresses(size_t subgraphs)
{
    const int32_t version = 0;
    const auto num_tensors = int32_t(_tensor_addresses.size());
    const auto buffer_index = int32_t(_serialised_buffers.size());

    _tensor_addresses.insert(_tensor_addresses.begin(), {version, int(subgraphs), num_tensors});

    const auto buffer_base = reinterpret_cast<uint8_t *>(_tensor_addresses.data());
    const auto buffer_size = _tensor_addresses.size() * sizeof(int32_t);

    _serialised_buffers.push_back(SerialiseBuffer(buffer_base, buffer_size, false));

    return tflite::CreateMetadataDirect(_flatbuffer, "OfflineMemoryAllocation", uint32_t(buffer_index));
}

flatbuffers::Offset<tflite::Buffer> TfLiteWriter::SerialiseBuffer(const Buffer *buffer, bool useOffset)
{
    return SerialiseBuffer(buffer->Data<uint8_t>(), buffer->Size(), useOffset);
}

flatbuffers::Offset<tflite::Buffer> TfLiteWriter::SerialiseBuffer(const uint8_t *data, size_t size, bool useOffset)
{
    const auto tflite_buffer_structure_size = sizeof(flatbuffers::soffset_t) * 4;

    flatbuffers::Offset<tflite::Buffer> ret;

    _flatbuffer.ForceVectorAlignment(size, sizeof(uint8_t), BUFFER_ALIGNMENT);  // 16-byte alignment
    if ( !useOffset )
    {
        useOffset = size_t(_flatbuffer.GetSize()) + size > (_fbSizeCap - tflite_buffer_structure_size);
    }

    if ( useOffset )
    {
        _flatbuffer.ForceDefaults(true);
        ret = tflite::CreateBuffer(_flatbuffer, 0, 0, size);
        _flatbuffer.ForceDefaults(false);
        _offset_buffers.emplace_back(data, size);
    }
    else
    {
        ret = tflite::CreateBuffer(_flatbuffer, _flatbuffer.CreateVector<uint8_t>(data, size));
    }

    return ret;
}

}  // namespace regor
