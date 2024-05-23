//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "hash.hpp"
#include "shape.hpp"

#include <cassert>
#include <memory>
#include <vector>

namespace regor
{

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
    int _sizeBytes = 0;
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
    Buffer(int sizeElements, const TYPE *buffer = nullptr, bool alias = false) :
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
            assert(IsByte<T>::value || (TypeHash<std::make_unsigned_t<T>>::value == _utypeHash));
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
            return _sizeBytes;
        }
    }

    const Hash128 &Hash() const { return _dataHash; }

    void Rehash()
    {
        // Calculate MD5 hash of data, prefixed by the size of data
        const auto buffer = const_cast<const Buffer *>(this);
        auto sizeStr = fmt::format("<{}>", buffer->Size());
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
    static void Delete(void *p)
    {
        delete reinterpret_cast<TYPE *>(p);
    }
    template<typename TYPE>
    static void DeleteArray(void *p)
    {
        delete[] reinterpret_cast<TYPE *>(p);
    }
    template<typename TYPE>
    static void DeleteVector(void *v)
    {
        using vec = std::vector<TYPE>;
        static_cast<vec *>(v)->~vec();
    }
#undef FOR_ALL_INT_TYPES
};


/// <summary>
/// Access proxy for processing values within a buffer
/// </summary>
template<typename TYPE, bool IS_CONST>
class BufferValues
{
    using PTR_TYPE = typename std::conditional_t<IS_CONST, const TYPE *, TYPE *>;

private:
    PTR_TYPE _data;
    Shape _strideBytes;

public:
    BufferValues(PTR_TYPE data, const Shape &strideBytes) : _data(data), _strideBytes(strideBytes) {}

    template<bool CONSTNESS = IS_CONST, typename std::enable_if_t<!CONSTNESS, int> = 0>
    TYPE &operator[](int index)
    {
        return _data[index];
    }

    const TYPE &operator[](int index) const { return _data[index]; }

    int ElementIndex(const Shape &offset) const
    {
        int index = offset.Dot(_strideBytes) / sizeof(TYPE);
        return index;
    }
};


/// <summary>
/// View of buffer memory
/// </summary>
class BufferView
{
protected:
    std::shared_ptr<class Buffer> _buffer;
    int _elementBits = 0;
    int _baseOffset = 0;
    Shape _axisElements;
    Shape _strideBytes;

public:
    BufferView() {}

    BufferView(const std::shared_ptr<Buffer> &buffer, int firstElement, int elementBits, const Shape &axisElements, const Shape &strideBytes)
    {
        assert(elementBits >= 8 && elementBits % 8 == 0);
        _buffer = buffer;
        _elementBits = elementBits;
        _baseOffset = firstElement;
        _axisElements = axisElements;
        if ( strideBytes.IsEmpty() )
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
    }

    BufferView(const std::shared_ptr<Buffer> &buffer, const BufferView &other)
    {
        _buffer = buffer;
        _elementBits = other._elementBits;
        _baseOffset = 0;
        _axisElements = other._axisElements;
        _strideBytes = other._strideBytes;
    }

public:
    bool HasBuffer() const { return _buffer != nullptr; }
    const Shape &ViewShape() const { return _axisElements; }
    const Shape &StrideBytes() const { return _strideBytes; }

    BufferView Reshape(const Shape &size) const
    {
        assert(size.Elements() == _axisElements.Elements());
        return BufferView(_buffer, 0, _elementBits, size, Shape());
    }

    BufferView SubView(const Shape &offset, const Shape &size) const
    {
        assert(size.Elements() < _axisElements.Elements());
        int linearOffset = offset.Dot(_strideBytes);
        return BufferView(_buffer, linearOffset, _elementBits, size, _strideBytes);
    }

    template<typename TYPE>
    BufferValues<TYPE, true> Values() const
    {
        assert(HasBuffer());
        auto start = const_cast<const class Buffer *>(_buffer.get())->Data<TYPE>() + _baseOffset;
        return BufferValues<TYPE, true>(start, _strideBytes);
    }

    template<typename TYPE>
    BufferValues<TYPE, false> WritableValues()
    {
        assert(HasBuffer());
        auto start = _buffer->Data<TYPE>() + _baseOffset;
        return BufferValues<TYPE, false>(start, _strideBytes);
    }

    int BufferSize() const { return _buffer->Size(); }

    const class Buffer *Buffer() const { return _buffer.get(); }
};

}  // namespace regor
