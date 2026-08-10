//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "tflite_model_inventory.hpp"

#include "common/hash.hpp"
#include "flatbuffer_utils.hpp"
#include "tflite_schema_generated.hpp"

#include <flatbuffers/flatbuffers.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace regor
{
namespace
{

std::string EscapeJson(std::string_view value)
{
    std::ostringstream os;
    for ( const unsigned char ch : value )
    {
        switch ( ch )
        {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\b': os << "\\b"; break;
            case '\f': os << "\\f"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if ( ch < 0x20 )
                {
                    os << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(ch) << std::dec;
                }
                else
                {
                    os << char(ch);
                }
        }
    }
    return os.str();
}

template<typename T>
void WriteVector(std::ostringstream &os, const flatbuffers::Vector<T> *values)
{
    os << '[';
    if ( values )
    {
        for ( unsigned i = 0; i < values->size(); ++i )
        {
            if ( i ) os << ',';
            os << values->Get(i);
        }
    }
    os << ']';
}

void WriteFloatVector(std::ostringstream &os, const flatbuffers::Vector<float> *values)
{
    os << '[';
    if ( values )
    {
        os << std::setprecision(std::numeric_limits<float>::max_digits10);
        for ( unsigned i = 0; i < values->size(); ++i )
        {
            if ( i ) os << ',';
            const float value = values->Get(i);
            if ( std::isfinite(value) ) os << value;
            else os << "null";
        }
    }
    os << ']';
}

tflite::BuiltinOperator GetBuiltinCode(const tflite::OperatorCode *code)
{
    return unsigned(code->builtin_code()) ? code->builtin_code() : tflite::BuiltinOperator(code->deprecated_builtin_code());
}

std::string OperatorName(const tflite::OperatorCode *code)
{
    const auto builtin = GetBuiltinCode(code);
    if ( builtin == tflite::BuiltinOperator::CUSTOM && code->custom_code() ) return code->custom_code()->str();
    return tflite::EnumNameBuiltinOperator(builtin);
}

void WriteOperatorOptions(std::ostringstream &os, const tflite::Operator *op)
{
    os << "{\"type\":\"" << tflite::EnumNameBuiltinOptions(op->builtin_options_type()) << '"';
    if ( const auto *conv = op->builtin_options_as_Conv2DOptions() )
    {
        os << ",\"padding\":\"" << tflite::EnumNamePadding(conv->padding()) << "\",\"stride_h\":"
           << conv->stride_h() << ",\"stride_w\":" << conv->stride_w() << ",\"dilation_h\":"
           << conv->dilation_h_factor() << ",\"dilation_w\":" << conv->dilation_w_factor()
           << ",\"activation\":\"" << tflite::EnumNameActivationFunctionType(conv->fused_activation_function()) << '"';
    }
    else if ( const auto *depthwise = op->builtin_options_as_DepthwiseConv2DOptions() )
    {
        os << ",\"padding\":\"" << tflite::EnumNamePadding(depthwise->padding()) << "\",\"stride_h\":"
           << depthwise->stride_h() << ",\"stride_w\":" << depthwise->stride_w() << ",\"depth_multiplier\":"
           << depthwise->depth_multiplier() << ",\"dilation_h\":" << depthwise->dilation_h_factor()
           << ",\"dilation_w\":" << depthwise->dilation_w_factor() << ",\"activation\":\""
           << tflite::EnumNameActivationFunctionType(depthwise->fused_activation_function()) << '"';
    }
    else if ( const auto *pool = op->builtin_options_as_Pool2DOptions() )
    {
        os << ",\"padding\":\"" << tflite::EnumNamePadding(pool->padding()) << "\",\"stride_h\":"
           << pool->stride_h() << ",\"stride_w\":" << pool->stride_w() << ",\"filter_h\":"
           << pool->filter_height() << ",\"filter_w\":" << pool->filter_width() << ",\"activation\":\""
           << tflite::EnumNameActivationFunctionType(pool->fused_activation_function()) << '"';
    }
    else if ( const auto *concat = op->builtin_options_as_ConcatenationOptions() )
    {
        os << ",\"axis\":" << concat->axis() << ",\"activation\":\""
           << tflite::EnumNameActivationFunctionType(concat->fused_activation_function()) << '"';
    }
    else if ( const auto *add = op->builtin_options_as_AddOptions() )
    {
        os << ",\"activation\":\"" << tflite::EnumNameActivationFunctionType(add->fused_activation_function()) << '"';
    }
    else if ( const auto *mul = op->builtin_options_as_MulOptions() )
    {
        os << ",\"activation\":\"" << tflite::EnumNameActivationFunctionType(mul->fused_activation_function()) << '"';
    }
    else if ( const auto *resize = op->builtin_options_as_ResizeNearestNeighborOptions() )
    {
        os << ",\"align_corners\":" << (resize->align_corners() ? "true" : "false")
           << ",\"half_pixel_centers\":" << (resize->half_pixel_centers() ? "true" : "false");
    }
    os << '}';
}

void WriteTensor(std::ostringstream &os, const tflite::Tensor *tensor, unsigned index)
{
    os << "{\"index\":" << index << ",\"name\":\""
       << EscapeJson(tensor->name() ? tensor->name()->string_view() : std::string_view{}) << "\",\"shape\":";
    WriteVector(os, tensor->shape());
    os << ",\"shape_signature\":";
    WriteVector(os, tensor->shape_signature());
    os << ",\"type\":\"" << tflite::EnumNameTensorType(tensor->type()) << "\",\"buffer\":" << tensor->buffer()
       << ",\"is_variable\":" << (tensor->is_variable() ? "true" : "false") << ",\"quantization\":";
    const auto *quant = tensor->quantization();
    if ( !quant )
    {
        os << "null";
    }
    else
    {
        os << "{\"scales\":";
        WriteFloatVector(os, quant->scale());
        os << ",\"zero_points\":";
        WriteVector(os, quant->zero_point());
        os << ",\"quantized_dimension\":" << quant->quantized_dimension() << '}';
    }
    os << '}';
}

std::string Md5Hex(const uint8_t *data, size_t size)
{
    MD5 md5;
    while ( size )
    {
        const int chunk = int(std::min(size, size_t(std::numeric_limits<int>::max())));
        md5.Combine(data, chunk);
        data += chunk;
        size -= size_t(chunk);
    }
    Hash128 hash;
    md5.Get(hash);
    std::ostringstream os;
    os << std::hex << std::setfill('0');
    for ( int i = 0; i < hash.Size(); ++i ) os << std::setw(2) << unsigned(hash.v8[i]);
    return os.str();
}

}  // namespace

TfLiteModelInventory BuildTfLiteModelInventory(const uint8_t *data, size_t size, std::string_view artifactName)
{
    if ( !data || size < 8 ) return {{}, "Input is too small to be a TFLite model"};

    flatbuffers::Verifier verifier(data, size);
    if ( !tflite::VerifyModelBuffer(verifier) ) return {{}, "Input is not a valid TFLite FlatBuffer"};

    const auto *model = tflite::GetModel(data);
    const auto *codes = model->operator_codes();
    const auto *subgraphs = model->subgraphs();
    if ( !codes || !subgraphs ) return {{}, "TFLite model has no operator-code or subgraph table"};

    std::map<std::string, unsigned> operatorCounts;
    for ( const auto *subgraph : *subgraphs )
    {
        if ( !subgraph->operators() ) continue;
        for ( const auto *op : *subgraph->operators() )
        {
            if ( op->opcode_index() >= codes->size() ) return {{}, "Operator references an invalid opcode index"};
            ++operatorCounts[OperatorName(codes->Get(op->opcode_index()))];
        }
    }

    std::ostringstream os;
    os << "{\"inventory_schema_version\":1,\"artifact\":{\"name\":\"" << EscapeJson(artifactName)
       << "\",\"byte_size\":" << size << ",\"hash_algorithm\":\"md5\",\"hash\":\"" << Md5Hex(data, size)
       << "\"},\"tflite_schema_version\":" << model->version() << ",\"description\":\""
       << EscapeJson(model->description() ? model->description()->string_view() : std::string_view{})
       << "\",\"operator_counts\":[";
    bool first = true;
    for ( const auto &[name, count] : operatorCounts )
    {
        if ( !first ) os << ',';
        first = false;
        os << "{\"name\":\"" << EscapeJson(name) << "\",\"count\":" << count << '}';
    }
    os << "],\"subgraphs\":[";
    for ( unsigned graphIndex = 0; graphIndex < subgraphs->size(); ++graphIndex )
    {
        if ( graphIndex ) os << ',';
        const auto *subgraph = subgraphs->Get(graphIndex);
        os << "{\"index\":" << graphIndex << ",\"name\":\""
           << EscapeJson(subgraph->name() ? subgraph->name()->string_view() : std::string_view{}) << "\",\"inputs\":";
        WriteVector(os, subgraph->inputs());
        os << ",\"outputs\":";
        WriteVector(os, subgraph->outputs());
        os << ",\"tensors\":[";
        if ( subgraph->tensors() )
        {
            for ( unsigned tensorIndex = 0; tensorIndex < subgraph->tensors()->size(); ++tensorIndex )
            {
                if ( tensorIndex ) os << ',';
                WriteTensor(os, subgraph->tensors()->Get(tensorIndex), tensorIndex);
            }
        }
        os << "],\"operators\":[";
        if ( subgraph->operators() )
        {
            for ( unsigned opIndex = 0; opIndex < subgraph->operators()->size(); ++opIndex )
            {
                if ( opIndex ) os << ',';
                const auto *op = subgraph->operators()->Get(opIndex);
                const auto *code = codes->Get(op->opcode_index());
                const auto builtin = GetBuiltinCode(code);
                os << "{\"index\":" << opIndex << ",\"name\":\"" << EscapeJson(OperatorName(code))
                   << "\",\"builtin_code\":" << int(builtin) << ",\"version\":" << code->version()
                   << ",\"inputs\":";
                WriteVector(os, op->inputs());
                os << ",\"outputs\":";
                WriteVector(os, op->outputs());
                os << ",\"options\":";
                WriteOperatorOptions(os, op);
                os << '}';
            }
        }
        os << "]}";
    }
    os << "]}\n";
    return {os.str(), {}};
}

TfLiteTopologyMicrograph BuildTfLiteTopologyMicrograph(const uint8_t *data, size_t size,
    unsigned subgraphIndex, const std::vector<unsigned> &operatorIndices, std::string_view artifactName)
{
    if ( !data || size < 8 ) return {{}, {}, "Input is too small to be a TFLite model"};
    flatbuffers::Verifier verifier(data, size);
    if ( !tflite::VerifyModelBuffer(verifier) ) return {{}, {}, "Input is not a valid TFLite FlatBuffer"};
    if ( operatorIndices.empty() ) return {{}, {}, "At least one source operator index is required"};

    const auto *model = tflite::GetModel(data);
    const auto *subgraphs = model->subgraphs();
    const auto *codes = model->operator_codes();
    const auto *buffers = model->buffers();
    if ( !subgraphs || subgraphIndex >= subgraphs->size() ) return {{}, {}, "Subgraph index is out of range"};
    if ( !codes || !buffers ) return {{}, {}, "TFLite model has no operator-code or buffer table"};

    const auto *subgraph = subgraphs->Get(subgraphIndex);
    const auto *sourceOps = subgraph->operators();
    const auto *sourceTensors = subgraph->tensors();
    if ( !sourceOps || !sourceTensors ) return {{}, {}, "Selected subgraph has no operators or tensors"};

    std::set<unsigned> selectedOps(operatorIndices.begin(), operatorIndices.end());
    if ( selectedOps.size() != operatorIndices.size() ) return {{}, {}, "Source operator indices must be unique"};
    for ( const unsigned index : selectedOps )
    {
        if ( index >= sourceOps->size() ) return {{}, {}, "Source operator index is out of range"};
        if ( sourceOps->Get(index)->opcode_index() >= codes->size() )
            return {{}, {}, "Operator references an invalid opcode index"};
        if ( sourceOps->Get(index)->large_custom_options_size() != 0 )
            return {{}, {}, "Large external custom options are not supported in mapping micro-graphs"};
    }

    std::map<unsigned, unsigned> producer;
    std::map<unsigned, std::set<unsigned>> consumers;
    for ( unsigned opIndex = 0; opIndex < sourceOps->size(); ++opIndex )
    {
        const auto *op = sourceOps->Get(opIndex);
        if ( op->outputs() )
        {
            for ( const int32_t tensorIndex : *op->outputs() )
                if ( tensorIndex >= 0 ) producer[unsigned(tensorIndex)] = opIndex;
        }
        if ( op->inputs() )
        {
            for ( const int32_t tensorIndex : *op->inputs() )
                if ( tensorIndex >= 0 ) consumers[unsigned(tensorIndex)].insert(opIndex);
        }
    }

    std::set<unsigned> selectedTensors;
    std::set<unsigned> boundaryInputs;
    std::set<unsigned> boundaryOutputs;
    const auto hasConstantPayload = [&](const tflite::Tensor *tensor)
    {
        if ( tensor->buffer() >= buffers->size() ) return false;
        const auto *buffer = buffers->Get(tensor->buffer());
        return (buffer->data() && buffer->data()->size() != 0) || buffer->size() != 0;
    };
    const auto collectTensor = [&](int32_t tensorIndex, std::set<unsigned> &target) -> bool
    {
        if ( tensorIndex < 0 ) return true;
        if ( unsigned(tensorIndex) >= sourceTensors->size() ) return false;
        target.insert(unsigned(tensorIndex));
        return true;
    };
    for ( const unsigned opIndex : selectedOps )
    {
        const auto *op = sourceOps->Get(opIndex);
        if ( op->inputs() )
        {
            for ( const int32_t tensorIndex : *op->inputs() )
            {
                if ( !collectTensor(tensorIndex, selectedTensors) )
                    return {{}, {}, "Operator references an invalid input tensor index"};
                if ( tensorIndex < 0 ) continue;
                const auto writer = producer.find(unsigned(tensorIndex));
                const bool producedInside = writer != producer.end() && selectedOps.count(writer->second) != 0;
                const auto *tensor = sourceTensors->Get(unsigned(tensorIndex));
                if ( !producedInside && !hasConstantPayload(tensor) ) boundaryInputs.insert(unsigned(tensorIndex));
            }
        }
        if ( op->outputs() )
        {
            for ( const int32_t tensorIndex : *op->outputs() )
            {
                if ( !collectTensor(tensorIndex, selectedTensors) )
                    return {{}, {}, "Operator references an invalid output tensor index"};
                if ( tensorIndex < 0 ) continue;
                bool readOutside = false;
                const auto readers = consumers.find(unsigned(tensorIndex));
                if ( readers != consumers.end() )
                    readOutside = std::any_of(readers->second.begin(), readers->second.end(),
                        [&](unsigned reader) { return selectedOps.count(reader) == 0; });
                const bool readInside = readers != consumers.end() && std::any_of(readers->second.begin(), readers->second.end(),
                    [&](unsigned reader) { return selectedOps.count(reader) != 0; });
                if ( readOutside || !readInside ) boundaryOutputs.insert(unsigned(tensorIndex));
            }
        }
        if ( op->intermediates() )
            for ( const int32_t tensorIndex : *op->intermediates() )
                if ( !collectTensor(tensorIndex, selectedTensors) )
                    return {{}, {}, "Operator references an invalid intermediate tensor index"};
    }

    std::map<unsigned, unsigned> tensorMap;
    unsigned nextTensor = 0;
    for ( const unsigned sourceIndex : selectedTensors ) tensorMap[sourceIndex] = nextTensor++;

    std::set<unsigned> selectedBuffers{0};
    for ( const unsigned sourceIndex : selectedTensors )
    {
        const unsigned bufferIndex = sourceTensors->Get(sourceIndex)->buffer();
        if ( bufferIndex >= buffers->size() ) return {{}, {}, "Tensor references an invalid buffer index"};
        const auto *buffer = buffers->Get(bufferIndex);
        if ( buffer->size() != 0 &&
             (buffer->offset() > size || buffer->size() > size - size_t(buffer->offset())) )
            return {{}, {}, "External tensor buffer is outside the source model"};
        selectedBuffers.insert(bufferIndex);
    }
    std::map<unsigned, unsigned> bufferMap;
    unsigned nextBuffer = 0;
    for ( const unsigned sourceIndex : selectedBuffers ) bufferMap[sourceIndex] = nextBuffer++;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<tflite::OperatorCode>> copiedCodes;
    for ( const auto *code : *codes )
    {
        const auto copied = FlatbufferUtils::CopyTable(
            builder, reinterpret_cast<const flatbuffers::Table *>(code), tflite::OperatorCode::MiniReflectTypeTable());
        copiedCodes.emplace_back(copied.o);
    }
    std::vector<flatbuffers::Offset<tflite::Buffer>> copiedBuffers;
    for ( const auto &[sourceIndex, unused] : bufferMap )
    {
        (void)unused;
        const auto *sourceBuffer = buffers->Get(sourceIndex);
        if ( sourceBuffer->size() != 0 )
        {
            const auto begin = data + sourceBuffer->offset();
            const std::vector<uint8_t> payload(begin, begin + sourceBuffer->size());
            copiedBuffers.push_back(tflite::CreateBufferDirect(builder, &payload));
        }
        else
        {
            const auto copied = FlatbufferUtils::CopyTable(builder,
                reinterpret_cast<const flatbuffers::Table *>(sourceBuffer), tflite::Buffer::MiniReflectTypeTable());
            copiedBuffers.emplace_back(copied.o);
        }
    }
    std::vector<flatbuffers::Offset<tflite::Tensor>> copiedTensors;
    for ( const auto &[sourceIndex, unused] : tensorMap )
    {
        (void)unused;
        const auto copied = FlatbufferUtils::CopyTable(builder,
            reinterpret_cast<const flatbuffers::Table *>(sourceTensors->Get(sourceIndex)), tflite::Tensor::MiniReflectTypeTable());
        copiedTensors.emplace_back(copied.o);
    }
    std::vector<flatbuffers::Offset<tflite::Operator>> copiedOps;
    for ( const unsigned sourceIndex : selectedOps )
    {
        const auto copied = FlatbufferUtils::CopyTable(builder,
            reinterpret_cast<const flatbuffers::Table *>(sourceOps->Get(sourceIndex)), tflite::Operator::MiniReflectTypeTable());
        copiedOps.emplace_back(copied.o);
    }

    const auto remapBoundary = [&](const std::set<unsigned> &source)
    {
        std::vector<int32_t> result;
        for ( const unsigned index : source ) result.push_back(int32_t(tensorMap.at(index)));
        return result;
    };
    const auto graphInputs = remapBoundary(boundaryInputs);
    const auto graphOutputs = remapBoundary(boundaryOutputs);
    std::vector<flatbuffers::Offset<tflite::SubGraph>> copiedSubgraphs{
        tflite::CreateSubGraphDirect(builder, &copiedTensors, &graphInputs, &graphOutputs, &copiedOps, "mapping_micrograph"),
    };
    std::ostringstream description;
    description << "Neural-AI topology micrograph from " << artifactName << " subgraph " << subgraphIndex;
    const auto copiedModel = tflite::CreateModelDirect(
        builder, model->version(), &copiedCodes, &copiedSubgraphs, description.str().c_str(), &copiedBuffers);
    tflite::FinishModelBuffer(builder, copiedModel);

    auto *mutableModel = tflite::GetMutableModel(builder.GetBufferPointer());
    auto *mutableGraph = mutableModel->mutable_subgraphs()->GetMutableObject(0);
    unsigned localTensor = 0;
    for ( const auto &[sourceIndex, unused] : tensorMap )
    {
        (void)unused;
        mutableGraph->mutable_tensors()->GetMutableObject(localTensor++)->mutate_buffer(
            bufferMap.at(sourceTensors->Get(sourceIndex)->buffer()));
    }
    unsigned localOp = 0;
    for ( const unsigned sourceIndex : selectedOps )
    {
        const auto patchTensorIndices = [&](flatbuffers::Vector<int32_t> *indices)
        {
            if ( !indices ) return;
            for ( unsigned i = 0; i < indices->size(); ++i )
            {
                const int32_t sourceTensor = indices->Get(i);
                if ( sourceTensor >= 0 ) indices->Mutate(i, int32_t(tensorMap.at(unsigned(sourceTensor))));
            }
        };
        auto *mutableOp = mutableGraph->mutable_operators()->GetMutableObject(localOp++);
        patchTensorIndices(mutableOp->mutable_inputs());
        patchTensorIndices(mutableOp->mutable_outputs());
        patchTensorIndices(mutableOp->mutable_intermediates());
        (void)sourceIndex;
    }

    std::vector<uint8_t> result(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
    std::ostringstream provenance;
    provenance << "{\"micrograph_schema_version\":1,\"source_artifact\":{\"name\":\""
               << EscapeJson(artifactName) << "\",\"byte_size\":" << size
               << ",\"hash_algorithm\":\"md5\",\"hash\":\"" << Md5Hex(data, size)
               << "\"},\"subgraph_index\":" << subgraphIndex << ",\"source_operator_indices\":[";
    bool first = true;
    for ( const unsigned index : selectedOps )
    {
        if ( !first ) provenance << ',';
        first = false;
        provenance << index;
    }
    provenance << "],\"source_input_tensor_indices\":[";
    first = true;
    for ( const unsigned index : boundaryInputs )
    {
        if ( !first ) provenance << ',';
        first = false;
        provenance << index;
    }
    provenance << "],\"source_output_tensor_indices\":[";
    first = true;
    for ( const unsigned index : boundaryOutputs )
    {
        if ( !first ) provenance << ',';
        first = false;
        provenance << index;
    }
    provenance << "],\"micrograph\":{\"byte_size\":" << result.size()
               << ",\"hash_algorithm\":\"md5\",\"hash\":\"" << Md5Hex(result.data(), result.size()) << "\"}}\n";
    return {std::move(result), provenance.str(), {}};
}

}  // namespace regor
