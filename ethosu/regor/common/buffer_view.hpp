//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common.hpp"
#include "data_type.hpp"
#include "hash.hpp"
#include "shape.hpp"

#include <cassert>
#include <iterator>
#include <memory>
#include <vector>

namespace regor
{

template<typename TYPE>
struct extended_make_unsigned
{
    using type = std::make_unsigned_t<TYPE>;
};

template<>
struct extended_make_unsigned<int48_t>
{
    using type = int48_t;
};

template<typename TYPE>
using extended_make_unsigned_t = typename extended_make_unsigned<TYPE>::type;

/// <summary>
/// Buffer mechanism for local/remote data storage
/// </summary>
class Buffer : public std::enable_shared_from_this<Buffer>
{
    typedef void (*DeleteFunc)(void *);

#define FOR_ALL_INT_TYPES(functor, sep) \
    functor(uint8_t) sep functor(uint16_t) \
    sep functor(uint32_t) \
    sep functor(uint64_t) \
    sep functor(int8_t) \
    sep functor(int16_t) \
    sep functor(int32_t) \
    sep functor(int48_t) \
    sep functor(int64_t)

    union LocalStorage
    {
        LocalStorage() {}
        ~LocalStorage() {}
#define TYPE_FUNC(x) std::vector<x> as_##x
        FOR_ALL_INT_TYPES(TYPE_FUNC, ;);
#undef TYPE_FUNC
    };

    template<typename TYPE>
    struct IsSupportedIntegral
    {
#define TYPE_FUNC(x) std::is_same<TYPE, x>::value
        static constexpr bool value = FOR_ALL_INT_TYPES(TYPE_FUNC, ||);
#undef TYPE_FUNC
    };

    template<typename TYPE>
    struct IsByte
    {
        static constexpr bool value =
            std::is_same<TYPE, char>::value || std::is_same<TYPE, unsigned char>::value || std::is_same<TYPE, std::byte>::value;
    };

    // TODO : make a proper type hash
    template<typename TYPE>
    struct TypeHash
    {
        static constexpr uint32_t value = (std::is_signed<TYPE>::value ? 1U << 16 : 0) | sizeof(TYPE);
    };

    union RefData
    {
        void *data;
        const void *cdata;
    };

private:
    RefData _refData = {};
    size_t _sizeBytes = 0;
    const uint32_t _typeHash;
    const uint32_t _utypeHash;
    bool _isLocal = false;
    LocalStorage _localStorage;
    DeleteFunc _deleter = nullptr;
    Hash128 _dataHash;

public:
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;

    template<typename TYPE, std::enable_if_t<IsSupportedIntegral<TYPE>::value, int> = 0>
    Buffer(size_t sizeElements, const TYPE *buffer = nullptr, bool alias = false) :
            _typeHash(TypeHash<TYPE>::value), _utypeHash(TypeHash<std::make_unsigned_t<TYPE>>::value)
    {
        _sizeBytes = sizeof(TYPE) * sizeElements;
        if ( buffer == nullptr || !alias )
        {
            assert(sizeElements > 0);
            auto ref = new TYPE[sizeElements];
            if ( buffer )
            {
                std::copy_n(buffer, sizeElements, ref);
            }
            _refData.data = ref;
            _deleter = &Buffer::DeleteArray<TYPE>;
        }
        else
        {
            assert(alias && buffer);
            _refData.cdata = buffer;
        }

        Rehash();
    }

    template<typename TYPE, std::enable_if_t<IsSupportedIntegral<TYPE>::value, int> = 0>
    Buffer(std::unique_ptr<TYPE> ptr) :
            _typeHash(TypeHash<TYPE>::value), _utypeHash(TypeHash<std::make_unsigned_t<TYPE>>::value)
    {
        _refData.data = ptr.release();
        _sizeBytes = sizeof(TYPE);
        _deleter = &Buffer::Delete<TYPE>;

        Rehash();
    }

    template<typename TYPE, std::enable_if_t<IsSupportedIntegral<TYPE>::value, int> = 0>
    Buffer(std::unique_ptr<TYPE[]> ptr, int sizeElements) :
            _typeHash(TypeHash<TYPE>::value), _utypeHash(TypeHash<std::make_unsigned_t<TYPE>>::value)
    {
        _refData.data = ptr.release();
        assert(sizeElements > 0);
        assert(INT_MAX / int(sizeof(TYPE)) >= sizeElements);
        _sizeBytes = sizeof(TYPE) * sizeElements;
        _deleter = &Buffer::DeleteArray<TYPE>;

        Rehash();
    }

    template<typename TYPE, std::enable_if_t<IsSupportedIntegral<TYPE>::value, int> = 0>
    Buffer(std::vector<TYPE> &&buffer) :
            _typeHash(TypeHash<TYPE>::value), _utypeHash(TypeHash<std::make_unsigned_t<TYPE>>::value)
    {
        new (&GetLocalVector<TYPE>()) std::vector<TYPE>(std::move(buffer));
        _deleter = &Buffer::DeleteVector<TYPE>;
        _refData.data = &GetLocalVector<TYPE>();
        _isLocal = true;

        Rehash();
    }

    ~Buffer()
    {
        if ( _deleter )
        {
            _deleter(_refData.data);
        }
    }

public:
    template<typename T>
    T *Data()
    {
        // Follow strict reinterpret_cast type aliasing rules
        assert(IsByte<T>::value || (TypeHash<std::make_unsigned_t<T>>::value == _utypeHash));
        if ( _isLocal )
        {
            if constexpr ( IsByte<T>::value )
            {
                switch ( _typeHash )
                {
#define TYPE_FUNC(x) \
    case TypeHash<x>::value: \
        return reinterpret_cast<T *>(GetLocalVector<x>().data())
                    FOR_ALL_INT_TYPES(TYPE_FUNC, ;);
#undef TYPE_FUNC
                    default:
                        assert(false);
                        return nullptr;
                }
            }
            else
            {
                using S = std::make_signed_t<T>;
                using U = std::make_unsigned_t<T>;
                switch ( _typeHash )
                {
                    case TypeHash<S>::value:
                        return reinterpret_cast<T *>(GetLocalVector<S>().data());
                    case TypeHash<U>::value:
                        return reinterpret_cast<T *>(GetLocalVector<U>().data());
                    default:
                        assert(false);
                        return nullptr;
                }
            }
        }
        else
        {
            assert(_deleter);
            return reinterpret_cast<T *>(_refData.data);
        }
    }

    template<typename T>
    const T *Data() const
    {
        if ( _isLocal )
        {
            // Follow strict reinterpret_cast type aliasing rules
            assert(IsByte<T>::value || (TypeHash<extended_make_unsigned_t<T>>::value == _utypeHash));
            if constexpr ( IsByte<T>::value )
            {
                switch ( _typeHash )
                {
#define TYPE_FUNC(x) \
    case TypeHash<x>::value: \
        return reinterpret_cast<const T *>(GetLocalVector<x>().data())
                    FOR_ALL_INT_TYPES(TYPE_FUNC, ;);
#undef TYPE_FUNC
                    default:
                        assert(false);
                        return nullptr;
                }
            }
            else if constexpr ( std::is_same<T, int48_t>::value )
            {
                return reinterpret_cast<const T *>(GetLocalVector<int48_t>().data());
            }
            else
            {
                using S = std::make_signed_t<T>;
                using U = std::make_unsigned_t<T>;
                switch ( _typeHash )
                {
                    case TypeHash<S>::value:
                        return reinterpret_cast<const T *>(GetLocalVector<S>().data());
                    case TypeHash<U>::value:
                        return reinterpret_cast<const T *>(GetLocalVector<U>().data());
                    default:
                        assert(false);
                        return nullptr;
                }
            }
        }
        else
        {
            assert(uintptr_t(_deleter ? _refData.data : _refData.cdata) % alignof(T) == 0);
            return reinterpret_cast<const T *>(_deleter ? _refData.data : _refData.cdata);
        }
    }

    int Size() const
    {
        if ( _isLocal )
        {
            switch ( _typeHash )
            {
#define TYPE_FUNC(x) \
    case TypeHash<x>::value: \
        return int(GetLocalVector<x>().size() * sizeof(x))
                FOR_ALL_INT_TYPES(TYPE_FUNC, ;);
#undef TYPE_FUNC
                default:
                    assert(false);
                    return 0;
            }
        }
        else
        {
            return int(_sizeBytes);
        }
    }

    const Hash128 &Hash() const { return _dataHash; }

    void Rehash()
    {
        // Calculate MD5 hash of data, prefixed by the size of data
        const auto buffer = const_cast<const Buffer *>(this);
        std::string sizeStr("<");
        sizeStr += std::to_string(buffer->Size());
        sizeStr += '>';
        MD5 hash;
        hash.Combine(reinterpret_cast<uint8_t *>(sizeStr.data()), int(sizeStr.size()));
        hash.Combine(buffer->Data<uint8_t>(), buffer->Size());
        hash.Get(_dataHash);
    }

private:
    template<typename TYPE>
    std::vector<TYPE> &GetLocalVector()
    {
        if constexpr ( false )
        {
        }
#define TYPE_FUNC(x) else if constexpr ( std::is_same<TYPE, x>::value ) return _localStorage.as_##x
        FOR_ALL_INT_TYPES(TYPE_FUNC, ;);
#undef TYPE_FUNC
        else
        {
            static_assert(IsSupportedIntegral<TYPE>::value, "");
            return _localStorage.as_uint8_t;
        }
    }
    template<typename TYPE>
    const std::vector<TYPE> &GetLocalVector() const
    {
        if constexpr ( false )
        {
        }
#define TYPE_FUNC(x) else if constexpr ( std::is_same<TYPE, x>::value ) return _localStorage.as_##x
        FOR_ALL_INT_TYPES(TYPE_FUNC, ;);
#undef TYPE_FUNC
        else
        {
            static_assert(IsSupportedIntegral<TYPE>::value, "");
            return _localStorage.as_uint8_t;
        }
    }

    template<typename TYPE>
    static inline void Delete(void *p)
    {
        delete reinterpret_cast<TYPE *>(p);
    }
    template<typename TYPE>
    static inline void DeleteArray(void *p)
    {
        delete[] reinterpret_cast<TYPE *>(p);
    }
    template<typename TYPE>
    static inline void DeleteVector(void *v)
    {
        using vec = std::vector<TYPE>;
        static_cast<vec *>(v)->~vec();
    }
#undef FOR_ALL_INT_TYPES
};

/// <summary>
/// Transient read-only access to a buffer view's elements
/// - Polymorphically compatible with same-typed readers
///   using different-typed data sources.
/// </summary>
template<typename TYPE>
struct BufferReader
{
    typedef TYPE (*GetFunc)(const void *p, size_t index);

private:
    Shape _strideBytes;
    const void *_data = nullptr;
    size_t _count = 0;
    GetFunc _get = nullptr;

public:
    BufferReader() = default;

    BufferReader(BufferReader<TYPE> &&other) noexcept { *this = std::move(other); }

    BufferReader(const BufferReader<TYPE> &other)
    {
        _strideBytes = other._strideBytes;
        _data = other._data;
        _count = other._count;
        _get = other._get;
    }

    BufferReader(const Shape &strideBytes, const void *p, size_t count, GetFunc fn) :
            _strideBytes(strideBytes), _data(p), _count(count), _get(fn)
    {
    }

    BufferReader<TYPE> &operator=(BufferReader<TYPE> &&other) noexcept
    {
        _strideBytes = other._strideBytes;
        _data = other._data;
        _count = other._count;
        _get = other._get;
        return *this;
    }

    TYPE operator[](size_t index) const
    {
        assert(index < _count);
        assert(((index == 0) || (_strideBytes.Size() <= 2)) && "View does not guarantee linear access");
        return _get(_data, index * _strideBytes.Depth());
    }

    TYPE operator[](const Shape &offset) const
    {
        size_t index = offset.Dot(_strideBytes);
        return _get(_data, index);
    }

    // Simple wrapping iterator
    template<typename VALUE>
    struct iterator_base_t
    {
    private:
        GetFunc _get;
        const void *_data = nullptr;
        size_t _offset;
        size_t _strideBytes;

    public:
        using value_type = VALUE;
        using pointer = VALUE *;
        using reference = VALUE &;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

    public:
        iterator_base_t() = default;
        iterator_base_t(const iterator_base_t<VALUE> &other) = default;
        iterator_base_t(GetFunc fn, const void *p, size_t index, size_t strideBytes) :
                _get(fn), _data(p), _offset(index * strideBytes), _strideBytes(strideBytes)
        {
        }

        // Value-only access
        VALUE operator*() { return _get(_data, _offset); }

        iterator_base_t<VALUE> &operator++()
        {
            _offset += _strideBytes;
            return *this;
        }

        iterator_base_t<VALUE> operator++(int)
        {
            iterator_base_t<VALUE> tmp = *this;
            _offset += _strideBytes;
            return tmp;
        }

        iterator_base_t<VALUE> &operator=(const iterator_base_t<VALUE> &other) = default;

        bool operator==(const iterator_base_t<VALUE> &other) const
        {
            assert(_data == other._data);
            return other._offset == _offset;
        }

        bool operator!=(const iterator_base_t<VALUE> &other) const
        {
            assert(_data == other._data);
            return other._offset != _offset;
        }
    };

    using iterator_t = iterator_base_t<TYPE>;
    iterator_t begin() const { return iterator_t(_get, _data, 0, _strideBytes.Depth()); }
    iterator_t end() const { return iterator_t(_get, _data, _count, _strideBytes.Depth()); }
};

/// <summary>
/// Transient read/write access to a buffer view's elements
/// </summary>
template<typename TYPE>
class BufferWriter
{
private:
    Shape _strideBytes;
    TYPE *_data = nullptr;
    size_t _count = 0;

public:
    BufferWriter() = default;

    BufferWriter(BufferWriter<TYPE> &&other) noexcept { *this = std::move(other); }

    BufferWriter(const BufferWriter<TYPE> &other)
    {
        _strideBytes = other._strideBytes;
        _data = other._data;
        _count = other._count;
    }

    BufferWriter(const Shape &strideBytes, TYPE *data, size_t count) :
            _strideBytes(strideBytes), _data(data), _count(count)
    {
    }

    BufferWriter<TYPE> &operator=(BufferWriter<TYPE> &&other) noexcept
    {
        _strideBytes = other._strideBytes;
        _data = other._data;
        _count = other._count;
        return *this;
    }

    const TYPE &operator[](size_t index) const
    {
        assert(index < _count);
        assert(_strideBytes[-1] == sizeof(TYPE));
        assert(((index == 0) || (_strideBytes.Size() <= 2)) && "View does not guarantee linear access");
        return _data[index];
    }

    TYPE &operator[](size_t index)
    {
        assert(index < _count);
        assert(_strideBytes[-1] == sizeof(TYPE));
        assert(((index == 0) || (_strideBytes.Size() <= 2)) && "View does not guarantee linear access");
        return _data[index];
    }

    const TYPE &operator[](const Shape &offset) const
    {
        size_t index = offset.Dot(_strideBytes) / sizeof(TYPE);
        assert(index < _count);
        return _data[index];
    }

    TYPE &operator[](const Shape &offset)
    {
        size_t index = offset.Dot(_strideBytes) / sizeof(TYPE);
        assert(index < _count);
        return _data[index];
    }
};

template<typename FROM, typename TO>
TO BufferReaderValueGet(const void *p, size_t offset)
{
    assert(((uintptr_t(p) + offset) % alignof(FROM)) == 0);
    return TO(*static_cast<const FROM *>(static_cast<const void *>(static_cast<const uint8_t *>(p) + offset)));
}

/// <summary>
/// View of buffer memory
/// </summary>
class BufferView
{
protected:
    std::shared_ptr<class Buffer> _buffer;
    int _elementBits = 0;
    int _elements = 0;
    int _baseOffset = 0;
    Shape _axisElements;
    Shape _strideBytes;

public:
    BufferView() = default;

    BufferView(const std::shared_ptr<Buffer> &buffer, int firstElement, int elementBits, const Shape &axisElements, const Shape &strideBytes)
    {
        assert((elementBits >= 8 && elementBits % 8 == 0) || (elementBits == 4 && !strideBytes));
        _buffer = buffer;
        _elementBits = elementBits;
        _baseOffset = firstElement;
        _axisElements = axisElements;
        if ( strideBytes.IsEmpty() && elementBits > 4 )
        {
            // Calculate byte strides
            int sz = axisElements.Size();
            if ( sz > 0 )
            {
                std::vector<int> strides(sz);
                int v = 1;
                for ( int i = sz - 1; i >= 0; --i )
                {
                    strides[i] = (v * elementBits) / 8;
                    v *= axisElements[i];
                }

                _strideBytes = Shape(&strides[0], sz);
            }
        }
        else
        {
            _strideBytes = strideBytes;
        }
        _elements = _axisElements.Elements();
    }

    BufferView(const std::shared_ptr<Buffer> &buffer, const BufferView &other)
    {
        _buffer = buffer;
        _elementBits = other._elementBits;
        _baseOffset = 0;
        _axisElements = other._axisElements;
        _elements = other._elements;
        _strideBytes = other._strideBytes;
    }

public:
    bool HasBuffer() const { return _buffer != nullptr; }
    const Shape &ViewShape() const { return _axisElements; }
    const Shape &StrideBytes() const { return _strideBytes; }
    int Elements() const { return _elements; }

    BufferView Reshape(const Shape &size) const
    {
        assert(size.Elements() == _axisElements.Elements());
        return BufferView(_buffer, 0, _elementBits, size, Shape());
    }

    BufferView SubView(const Shape &offset, const Shape &size) const
    {
        assert(_strideBytes && size.Elements() < _axisElements.Elements());
        int linearOffset = (offset.Dot(_strideBytes) * 8) / _elementBits;
        return BufferView(_buffer, linearOffset, _elementBits, size, _strideBytes);
    }

    template<typename STORAGE_TYPE, typename AS_TYPE = STORAGE_TYPE,
        AS_TYPE (*FUNC)(const void *p, size_t offset) = &BufferReaderValueGet<STORAGE_TYPE, AS_TYPE>>
    BufferReader<AS_TYPE> Values() const
    {
        assert(HasBuffer() && _strideBytes);
        const auto *start = const_cast<const class Buffer *>(_buffer.get())->Data<STORAGE_TYPE>() + _baseOffset;
        return BufferReader<AS_TYPE>(_strideBytes, start, _elements, FUNC);
    }

    template<typename STORAGE_TYPE>
    BufferWriter<STORAGE_TYPE> WritableValues()
    {
        assert(HasBuffer() && _strideBytes);
        auto *start = _buffer->Data<STORAGE_TYPE>() + _baseOffset;
        return BufferWriter<STORAGE_TYPE>(_strideBytes, start, _elements);
    }

    template<typename STORAGE_TYPE>
    const STORAGE_TYPE *RawData() const
    {
        const auto *start = const_cast<const class Buffer *>(_buffer.get())->Data<STORAGE_TYPE>();
        return start + _baseOffset;
    }

    template<typename TYPE>
    BufferReader<TYPE> Values(DataType dataType)
    {
        switch ( dataType )
        {
            case DataType::Int8:
                return Values<int8_t, TYPE>();
            case DataType::UInt8:
                return Values<uint8_t, TYPE>();
            case DataType::Int16:
                return Values<int16_t, TYPE>();
            case DataType::UInt16:
                return Values<uint16_t, TYPE>();
            case DataType::Int32:
                return Values<int32_t, TYPE>();
            case DataType::UInt32:
                return Values<uint32_t, TYPE>();
            case DataType::Int64:
                return Values<int64_t, TYPE>();
            case DataType::UInt64:
                return Values<uint64_t, TYPE>();
            default:
                assert(false && "Unexepected DataType");
                return Values<TYPE>();
        }
    }

    const class Buffer *Buffer() const { return _buffer.get(); }
};

}  // namespace regor
