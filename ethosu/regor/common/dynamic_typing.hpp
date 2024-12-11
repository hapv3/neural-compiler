//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/shape.hpp"

#include <cstdint>

namespace regor
{

template<typename T, typename ENABLE = void>
struct FieldTypeId
{
    static constexpr uint8_t TYPEID = 0;
};
// Ordinal types
template<>
struct FieldTypeId<bool>
{
    static constexpr uint8_t TYPEID = 1;
};
template<>
struct FieldTypeId<uint8_t>
{
    static constexpr uint8_t TYPEID = 3;
};
template<>
struct FieldTypeId<int8_t>
{
    static constexpr uint8_t TYPEID = 4;
};
template<>
struct FieldTypeId<uint16_t>
{
    static constexpr uint8_t TYPEID = 5;
};
template<>
struct FieldTypeId<int16_t>
{
    static constexpr uint8_t TYPEID = 6;
};
template<>
struct FieldTypeId<uint32_t>
{
    static constexpr uint8_t TYPEID = 7;
};
template<>
struct FieldTypeId<int32_t>
{
    static constexpr uint8_t TYPEID = 8;
};
template<>
struct FieldTypeId<uint64_t>
{
    static constexpr uint8_t TYPEID = 9;
};
template<>
struct FieldTypeId<int64_t>
{
    static constexpr uint8_t TYPEID = 10;
};
template<>
struct FieldTypeId<float>
{
    static constexpr uint8_t TYPEID = 11;
};
template<>
struct FieldTypeId<double>
{
    static constexpr uint8_t TYPEID = 12;
};
template<>
struct FieldTypeId<std::string>
{
    static constexpr uint8_t TYPEID = 13;
};
// Class types
template<>
struct FieldTypeId<Point2i>
{
    static constexpr uint8_t TYPEID = 0x20;
};
template<>
struct FieldTypeId<Shape>
{
    static constexpr uint8_t TYPEID = 0x21;
};
template<>
struct FieldTypeId<Fraction<int>>
{
    static constexpr uint8_t TYPEID = 0x22;
};
// Enum types
template<typename T>
struct FieldTypeId<T, typename std::enable_if<std::is_enum<T>::value>::type>
{
    static constexpr uint8_t TYPEID = FieldTypeId<typename std::underlying_type<T>::type>::TYPEID;
};

template<typename TYPE>
struct TypeHash
{
    static constexpr uint32_t HASH = PlatformTypeHash<TYPE, true>();
};

struct FieldInfo
{
    size_t offset = 0;
    uint32_t id = 0;
    uint8_t typeId = 0;
};

struct TypeInfo
{
    uint32_t _hash;
    const FieldInfo *_fields;
    size_t _fieldCount;
    void (*_deleter)(void *);
    void (*_addref)(void *);

public:
    uint32_t Hash() const { return _hash; }
    void *AddRef(void *p) const
    {
        if ( !_addref ) return nullptr;
        _addref(p);
        return p;
    }
    void Delete(void *p) const { _deleter(p); }
    const FieldInfo *Fields(size_t &length) const
    {
        length = _fieldCount;
        return _fields;
    }
};

// Dynamic allocation with type-erasure. Use to handle anonymous TYPE
// allocations by passing around a void pointer and the type information
// separately at runtime.
template<typename TYPE>
struct TypeInfoOf
{
    static void *DefaultNew() { return new TYPE(); }
    static void DefaultDeleter(void *p) { delete static_cast<TYPE *>(p); }

    struct SharedType
    {
        TYPE instance;
        unsigned ref = 1;
    };
    static void *SharedNew()
    {
        auto *p = new SharedType();
        assert(static_cast<void *>(p) == static_cast<void *>(&p->instance));
        return &p->instance;
    }
    static void SharedAddRef(void *p)
    {
        assert(p);
        static_cast<SharedType *>(p)->ref++;
    }
    static void SharedDeleter(void *p)
    {
        auto *shared = static_cast<SharedType *>(p);
        if ( --shared->ref == 0 ) delete shared;
    }

    static const TypeInfo *Get(bool sharedInstancing)
    {
        size_t len;
        const FieldInfo *f = TYPE::FieldTable(len);
        static const TypeInfo s_infoDefault{PlatformTypeHash<TYPE, true>(), f, len, &DefaultDeleter, nullptr};
        static const TypeInfo s_infoShared{PlatformTypeHash<TYPE, true>(), f, len, &SharedDeleter, &SharedAddRef};
        return sharedInstancing ? &s_infoShared : &s_infoDefault;
    }
};


// Container for dynamically typed instances
struct DynamicRef
{
private:
    const TypeInfo *_info = nullptr;
    void *_instance = nullptr;

public:
    DynamicRef() = default;
    DynamicRef(const TypeInfo *info, void *inst) : _info(info), _instance(inst) {}
    DynamicRef(const DynamicRef &other) { *this = other; }
    DynamicRef(DynamicRef &&other) noexcept { *this = std::move(other); }
    ~DynamicRef()
    {
        if ( _instance )
        {
            assert(_info);
            _info->Delete(_instance);
        }
    }

    DynamicRef &operator=(const DynamicRef &other)
    {
        if ( &other != this )
        {
            auto tmp = other._info ? other._info->AddRef(other._instance) : nullptr;
            if ( _instance )
            {
                assert(_info);
                _info->Delete(_instance);
                _instance = nullptr;
            }
            _info = other._info;
            _instance = tmp;
        }
        return *this;
    }

    DynamicRef &operator=(DynamicRef &&other) noexcept
    {
        if ( &other != this )
        {
            if ( _instance )
            {
                assert(_info);
                _info->Delete(_instance);
                _instance = nullptr;
            }
            _info = other._info;
            _instance = other._instance;
            other._instance = nullptr;
        }
        return *this;
    }
    operator bool() const { return _info && _instance; }
    void *Instance() { return _instance; }
    const void *Instance() const { return _instance; }
    const TypeInfo *Info() const { return _info; }
};


// clang-format off

#define BEGIN_FIELD_TABLE(CLASS_) \
    static const FieldInfo *FieldTable(size_t &len) { \
        typedef CLASS_ thisclass_t; \
        static const FieldInfo s_fieldTable[] = {

#define END_FIELD_TABLE() }; \
    len = std::size(s_fieldTable); \
    return s_fieldTable; }

// clang-format on

#define REGOR_FIELD_TYPE(TYPE_) FieldTypeId<std::decay<TYPE_>::type>::TYPEID


}  // namespace regor
