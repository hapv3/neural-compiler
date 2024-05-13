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

#pragma once

#include "compiler/graph.hpp"
#include "compiler/op_type.hpp"
#include "compiler/operation.hpp"
#include "compiler/tensor.hpp"
#include "tflite_schema_generated.hpp"

#include <flatbuffers/flatbuffers.h>
#include <map>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace regor
{

class TfLiteWriter
{
public:
    TfLiteWriter()
    {
        _flatbuffer = flatbuffers::FlatBufferBuilder();  // TODO: Determine sensible starting size (default is 1KB)
    }

    std::unique_ptr<const uint8_t[]> Serialise(const std::vector<std::unique_ptr<Graph>> &graphs,
        const std::vector<std::unordered_map<const Tensor *, Address>> &tensor_address_maps,
        int64_t &output_buffer_offset, size_t &output_buffer_size);

private:
    struct BufferDesc
    {
        const uint8_t *data = nullptr;
        size_t size = 0;
        BufferDesc() = default;
        BufferDesc(const Buffer *buffer) : data(buffer->Data<uint8_t>()), size(buffer->Size()) {}
        bool operator==(const BufferDesc &other) const { return (other.data == data) && (other.size == size); }
        struct hash
        {
            size_t operator()(const BufferDesc &desc) const { return std::hash<const uint8_t *>{}(desc.data); }
        };
    };

    struct OperatorCodeDesc
    {
        int8_t deprecated_builtin_code;
        const char *custom_code;
        int32_t version;
        tflite::BuiltinOperator type;
        OperatorCodeDesc() = default;
        bool operator==(const OperatorCodeDesc &other) const
        {
            const std::string_view custom_code_a(other.custom_code ? other.custom_code : "");
            const std::string_view custom_code_b(custom_code ? custom_code : "");
            return other.deprecated_builtin_code == deprecated_builtin_code && custom_code_a == custom_code_b &&
                   other.version == version && other.type == type;
        }
        struct hash
        {
            size_t operator()(const OperatorCodeDesc &desc) const
            {
                const size_t a = std::hash<tflite::BuiltinOperator>{}(desc.type);
                const size_t b = std::hash<std::string_view>{}(desc.custom_code ? desc.custom_code : "");
                return a ^ (b << 1);
            }
        };
    };

    // per-model
    flatbuffers::FlatBufferBuilder _flatbuffer;
    std::unordered_map<OperatorCodeDesc, int, OperatorCodeDesc::hash> _opcodes;
    std::unordered_map<BufferDesc, int, BufferDesc::hash> _buffers;
    std::vector<flatbuffers::Offset<tflite::OperatorCode>> _serialised_opcodes;
    std::vector<flatbuffers::Offset<tflite::SubGraph>> _serialised_subgraphs;
    std::vector<flatbuffers::Offset<tflite::Buffer>> _serialised_buffers;

    // per-subgraph
    std::unordered_map<const Tensor *, int> _tensors;
    std::vector<flatbuffers::Offset<tflite::Operator>> _serialised_operations;
    std::vector<flatbuffers::Offset<tflite::Tensor>> _serialised_tensors;
    std::vector<int32_t> _tensor_addresses;  // Keep as 32 bits - required by runtime

    int SerialisedTensorIndex(const Tensor *tensor, const std::unordered_map<const Tensor *, Address> &addresses);

    flatbuffers::Offset<tflite::Tensor> SerialiseTensor(const Tensor *tensor);
    flatbuffers::Offset<void> SerialiseOptions(const Operation *operation, OpType type);
    flatbuffers::Offset<tflite::Metadata> SerialiseTensorAddresses(int subgraph_index);

    flatbuffers::Offset<tflite::Buffer> SerialiseBuffer(const Buffer *buffer);
    flatbuffers::Offset<tflite::Buffer> SerialiseBuffer(const uint8_t *data, int size);

    struct TfLiteKernel
    {
        const tflite::Padding padding;
        const int filter_w;
        const int filter_h;
        const int stride_w;
        const int stride_h;
        const int dilation_w_factor;
        const int dilation_h_factor;
        const int depth_multiplier;

        TfLiteKernel(const Kernel &kernel) :
                padding(kernel.Padding().IsZero() ? tflite::Padding::VALID : tflite::Padding::SAME),
                filter_w(kernel.Size().x), filter_h(kernel.Size().y), stride_w(kernel.Stride().x),
                stride_h(kernel.Stride().y), dilation_w_factor(kernel.Dilation().x),
                dilation_h_factor(kernel.Dilation().y), depth_multiplier(kernel.DepthMultiplier())
        {
        }
    };

    static std::vector<const Tensor *> SortedInputTensors(const Operation *operation, OpType type);
};

}  // namespace regor
