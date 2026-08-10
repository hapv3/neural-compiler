//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "tflite_model_inventory.hpp"

#include "common/hash.hpp"
#include "tflite_schema_generated.hpp"

#include <flatbuffers/flatbuffers.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>

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

}  // namespace regor
