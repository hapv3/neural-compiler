//
// SPDX-FileCopyrightText: Copyright 2021-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2024 Meta Platforms, Inc. and affiliates.
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

#include "architecture/architecture.hpp"
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
    TfLiteWriter(size_t fbSizeCap = size_t{1U << 31}) :
            _flatbuffer(flatbuffers::FlatBufferBuilder()), _fbSizeCap(fbSizeCap)
    {
    }

    std::unique_ptr<const uint8_t[]> Serialise(const std::vector<std::unique_ptr<Graph>> &graphs,
        const std::vector<std::unordered_map<const Tensor *, Address>> &tensor_address_maps,
        int64_t &output_buffer_offset, size_t &output_buffer_size);

private:
    std::unique_ptr<const uint8_t[]> SerialiseImpl(const std::vector<std::unique_ptr<Graph>> &graphs,
        const std::vector<std::unordered_map<const Tensor *, Address>> &tensor_address_maps,
        int64_t &output_buffer_offset, size_t &output_buffer_size);

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
        OperatorCodeDesc(int8_t _deprecated_builtin_code, const char *_custom_code, int32_t _version, tflite::BuiltinOperator _type) :
                deprecated_builtin_code(_deprecated_builtin_code), custom_code(_custom_code), version(_version), type(_type)
        {
        }
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

    int SerialisedTensorIndex(const Tensor *tensor, const std::unordered_map<const Tensor *, Address> &addresses, const Graph &graph);

    flatbuffers::Offset<tflite::Tensor> SerialiseTensor(const Tensor *tensor, const Graph &graph);
    flatbuffers::Offset<void> SerialiseOptions(const Operation *operation, OpType type);
    flatbuffers::Offset<tflite::Metadata> SerialiseTensorAddresses(int subgraphs);
    void SerialiseTensorBuffer(const Tensor *tensor);

    class ResultBuffer
    {
        std::unique_ptr<uint8_t[]> _buf;
        size_t _reserved = 0;
        size_t _offset = 0;
        size_t _wr = 0;

    public:
        ResultBuffer(flatbuffers::FlatBufferBuilder &fb)
        {
            // Can convert to unique_ptr because std::default_delete() is equivalent to
            // flatbuffer::DefaultAllocator::deallocate()
            //  - i.e. they both do `delete base`
            size_t size, offset;
            auto ptr = fb.ReleaseRaw(size, offset);

            _buf.reset(ptr);
            _reserved = size;
            _offset = offset;
            _wr = size;
        }

        uint8_t *begin() { return &_buf[_offset]; }

        size_t reserved() const { return _reserved; }

        void reserve(size_t size)
        {
            if ( reserved() >= size ) return;

            auto buf = std::make_unique<uint8_t[]>(size);
            std::copy_n(begin(), _reserved - _offset, &buf[0]);
            _buf.reset(buf.release());
            _reserved = size;
            _offset = 0;
        }

        size_t push(const uint8_t *buf, size_t size)
        {
            reserve(_wr + size);

            auto wr = _wr;
            std::copy_n(buf, size, &_buf[_wr]);
            _wr += size;
            return wr;
        }

        size_t pos() const { return _wr; }

        void align(size_t alignment)
        {
            _wr = (_wr + alignment - 1) & ~(alignment - 1);
            reserve(_wr);
        }

        std::unique_ptr<const uint8_t[]> release(size_t &size, int64_t &offset)
        {
            size = _wr - _offset;
            offset = int64_t(_offset);

            _reserved = 0;
            _offset = 0;
            _wr = 0;
            return std::unique_ptr<const uint8_t[]>(_buf.release());
        }
    };

    std::vector<size_t> SerialiseOffsetBuffers(ResultBuffer &res);
    void FixupFbBuffers(uint8_t *model, const std::vector<size_t> &offsetBufferOffset);

    class OffsetBufferDesc
    {
        typedef void (*DeleteFunc)(void *);
        DeleteFunc _deleter = nullptr;
        void *_obj = nullptr;
        const uint8_t *_data = nullptr;
        size_t _size = 0;

        template<typename TYPE>
        static inline void DeleteVector(void *v)
        {
            using vec = std::vector<TYPE>;
            delete static_cast<vec *>(v);
        }

    public:
        template<typename T>
        OffsetBufferDesc(std::unique_ptr<std::vector<T>> &&buf)
        {
            auto *vec = buf.release();
            assert(vec);
            _deleter = &OffsetBufferDesc::DeleteVector<T>;
            _obj = vec;
            _data = reinterpret_cast<const uint8_t *>(vec->data());
            _size = vec->size() * sizeof(T);
        }

        OffsetBufferDesc(const Buffer *buffer) : OffsetBufferDesc(buffer->Data<uint8_t>(), buffer->Size()) {}

        OffsetBufferDesc(const uint8_t *data, size_t size) : _data(data), _size(size) {}

        ~OffsetBufferDesc()
        {
            if ( _deleter )
            {
                assert(_obj);
                _deleter(_obj);
            }
        }

        const uint8_t *data() const { return _data; }
        size_t size() const { return _size; }
    };

    std::vector<OffsetBufferDesc> _offset_buffers;
    bool _useBufferOffset = false;
    const size_t _fbSizeCap;

    static constexpr size_t BUFFER_ALIGNMENT = 16ULL;

    void CheckFlatBufferSize();
    flatbuffers::Offset<tflite::Buffer> SerialiseBuffer(const Buffer *buffer);
    flatbuffers::Offset<tflite::Buffer> SerialiseBuffer(const uint8_t *data, size_t size);

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
