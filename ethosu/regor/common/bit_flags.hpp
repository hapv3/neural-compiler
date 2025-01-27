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

#include "common/common.hpp"

#include "common/lexer.hpp"

#include <algorithm>
#include <type_traits>

struct EnumNameEntry
{
    unsigned value;
    const char *name;
};

template<typename ENUM>
static ENUM OrFlags(ENUM a)
{
    return a;
}

template<typename ENUM, typename... ARGS>
static ENUM OrFlags(ENUM a, ARGS... rest)
{
    return ENUM(typename std::underlying_type<ENUM>::type(OrFlags(std::forward<ARGS>(rest)...)) | typename std::underlying_type<ENUM>::type(a));
}

static std::string EnumFlagsToString(unsigned value, const EnumNameEntry *table, int tableLength);
static bool StringToEnumFlags(unsigned &value, const std::string &text, const EnumNameEntry *table, int tableLength);

/// <summary>
/// Enumerated flags wrapper
/// </summary>
template<typename ENUM>
struct Flags
{
    using TYPE = typename std::underlying_type<ENUM>::type;

private:
    TYPE _raw = 0;

public:
    Flags() {}
    Flags(ENUM val) : _raw(TYPE(val)) {}

    template<typename... ARGS>
    Flags(ENUM val, ARGS... rest)
    {
        _raw = TYPE(val) | TYPE(OrFlags(std::forward<ARGS>(rest)...));
    }

    explicit Flags(TYPE val) : _raw(val) {}
    Flags(const Flags<ENUM> &) = default;

public:
    bool operator==(ENUM val) const { return _raw == TYPE(val); }
    bool operator!=(ENUM val) const { return _raw != TYPE(val); }
    bool operator<(ENUM val) const { return _raw < TYPE(val); }
    operator ENUM() const { return ENUM(_raw); }
    explicit operator unsigned() const { return unsigned(_raw); }

    Flags<ENUM> &operator=(ENUM val)
    {
        _raw = TYPE(val);
        return *this;
    }
    Flags<ENUM> &operator=(const Flags<ENUM> &other)
    {
        _raw = other._raw;
        return *this;
    }
    Flags<ENUM> &operator&=(ENUM val)
    {
        _raw &= TYPE(val);
        return *this;
    }
    Flags<ENUM> &operator&=(const Flags<ENUM> &other)
    {
        _raw &= other._raw;
        return *this;
    }
    Flags<ENUM> &operator|=(ENUM val)
    {
        _raw |= TYPE(val);
        return *this;
    }
    Flags<ENUM> &operator|=(const Flags<ENUM> &other)
    {
        _raw |= other._raw;
        return *this;
    }
    Flags<ENUM> &operator^=(ENUM val)
    {
        _raw ^= TYPE(val);
        return *this;
    }
    Flags<ENUM> &operator^=(const Flags<ENUM> &other)
    {
        _raw ^= other._raw;
        return *this;
    }

    Flags<ENUM> operator&(ENUM val) const { return Flags<ENUM>(ENUM(_raw & TYPE(val))); }
    Flags<ENUM> operator|(ENUM val) const { return Flags<ENUM>(ENUM(_raw | TYPE(val))); }
    Flags<ENUM> operator^(ENUM val) const { return Flags<ENUM>(ENUM(_raw ^ TYPE(val))); }
    Flags<ENUM> operator~() const { return Flags<ENUM>(ENUM(~_raw)); }
    // Flag presence check. Use as if an infix replacement for '&' for testing flags.
    bool operator%(ENUM val) const { return (_raw & TYPE(val)) != 0; }
    bool operator!() const { return _raw == 0; }

    // Extract non-bitfield item
    unsigned GetUInt(ENUM offset, int bits) { return (_raw >> int(offset)) & ((1u << bits) - 1u); }

    // Set multiple flags
    Flags<ENUM> &Set(TYPE val)
    {
        _raw |= TYPE(val);
        return *this;
    }
    Flags<ENUM> &Set(ENUM val) { return Set(TYPE(val)); }
    template<typename... ARGS>
    Flags<ENUM> &Set(ENUM val, ARGS... rest)
    {
        return Set(TYPE(val) | TYPE(OrFlags(std::forward<ARGS>(rest)...)));
    }

    // Unset multiple flags
    Flags<ENUM> &Unset(TYPE val)
    {
        _raw &= ~val;
        return *this;
    }
    Flags<ENUM> &Unset(ENUM val) { return Unset(TYPE(val)); }
    template<typename... ARGS>
    Flags<ENUM> &Unset(ENUM val, ARGS... rest)
    {
        return Unset(TYPE(val) | TYPE(OrFlags(std::forward<ARGS>(rest)...)));
    }

    // Test any set
    bool Any(TYPE val) const { return (_raw & TYPE(val)) != 0; }
    bool Any(ENUM val) const { return Any(TYPE(val)); }
    bool Any(const Flags<ENUM> &val) const { return Any(ENUM(val)); }
    template<typename... ARGS>
    bool Any(ENUM val, ARGS... rest) const
    {
        TYPE mask = TYPE(val) | TYPE(OrFlags(std::forward<ARGS>(rest)...));
        return (_raw & mask) != 0;
    }

    // Test all set
    bool All(TYPE val) const { return (_raw & TYPE(val)) == TYPE(val); }
    bool All(ENUM val) const { return All(TYPE(val)); }
    bool All(const Flags<ENUM> &val) const { return All(ENUM(val)); }
    template<typename... ARGS>
    bool All(ENUM val, ARGS... rest) const
    {
        TYPE mask = TYPE(val) | TYPE(OrFlags(std::forward<ARGS>(rest)...));
        return (_raw & mask) == mask;
    }

    std::string ToString() const
    {
        int length = 0;
        const EnumNameEntry *table = GetTable(length);
        return EnumFlagsToString(_raw, table, length);
    }

    bool Parse(const std::string &text)
    {
        unsigned value = 0;
        bool ok = text.empty();
        if ( !ok )
        {
            int length = 0;
            const EnumNameEntry *table = GetTable(length);
            ok = StringToEnumFlags(value, text, table, length);
        }
        _raw = TYPE(value);
        return ok;
    }

private:
    // Proxy function for type erasure
    static const EnumNameEntry *GetTable(int &length)
    {
        extern const EnumNameEntry *GetEnumTable(ENUM, int &);
        length = 0;
        return GetEnumTable(ENUM(0), length);
    }
};

template<typename ENUM>
static std::string AllFlagsToString()
{
    std::string tmp;
    tmp.reserve(32);
    int length = 0;
    extern const EnumNameEntry *GetEnumTable(ENUM, int &);
    const EnumNameEntry *table = GetEnumTable(ENUM(0), length);
    for ( int i = 0; i < length; i++ )
    {
        const auto *sz = table[i].name;
        assert(sz);
        tmp += sz;
        tmp += '|';
    }
    return tmp;
}


static std::string EnumToString(unsigned value, const EnumNameEntry *table, int tableLength)
{
    auto pos = std::find_if(table, table + tableLength, [&value](const EnumNameEntry &v) { return v.value == value; });
    if ( pos != table + tableLength )
    {
        return pos->name;
    }
    return std::to_string(value);
}

static std::string EnumFlagsToString(unsigned value, const EnumNameEntry *table, int tableLength)
{
    if ( value == 0 ) return EnumToString(value, table, tableLength);
    unsigned mask = 1;
    std::string text;
    while ( mask <= value )
    {
        if ( value & mask )
        {
            if ( !text.empty() )
            {
                text += '|';
            }

            auto pos = std::find_if(
                table, table + tableLength, [&mask](const EnumNameEntry &v) { return v.value == mask; });
            if ( pos != table + tableLength )
            {
                text += pos->name;
            }
            else
            {
                text += std::to_string(mask);
            }
        }
        mask = mask << 1;
    }
    return text;
}

template<typename ENUM>
static std::string EnumToString(ENUM value)
{
    extern const EnumNameEntry *GetEnumTable(ENUM, int &);
    int length = 0;
    auto table = GetEnumTable(ENUM(0), length);
    return EnumToString(unsigned(value), table, length);
}

static bool StringToEnumFlags(unsigned &value, const std::string &text, const EnumNameEntry *table, int tableLength)
{
    Lexer lexer(text.data(), text.size());
    value = 0;
    std::string ident;
    bool isXor = false;
    while ( true )
    {
        if ( !lexer.SkipSpace() )
        {
            break;
        }
        if ( lexer.GetIdent(ident, false) )
        {
            auto pos = std::find_if(table, table + tableLength,
                [&ident](const EnumNameEntry &v)
                {
                    assert(v.name);
                    return ident.compare(v.name) == 0;
                });

            if ( pos == table + tableLength ) return false;

            value = isXor ? (value ^ pos->value) : (value | pos->value);
            isXor = false;
        }
        if ( !lexer.SkipSpace() )
        {
            break;
        }
        if ( lexer.Expect('^') )
        {
            isXor = true;
        }
        else if ( !lexer.Expect('|') )
        {
            return false;
        }
    }
    return true;
}

template<typename ENUM, std::enable_if_t<std::is_enum_v<ENUM>, int> = 0>
inline std::string format_as(const Flags<ENUM> &flags) noexcept
{
    return flags.ToString();
}

// Use to treat enumerations as flags when defined as single-bit
// numeric values:
//
//  enum class Type
//  {
//     First=1,
//     Second=2,
//     Third=4
//  };
//
// Wrap enumerations with Flags template and use as a bitset:
//
//  Flags<Type> flags(Type::First, Type::Second);
//  flags |= Type::Third;
//
// To convert to/from string, use macros to build a mapping table
//
//  BEGIN_ENUM_TABLE(Type)
//    ADD_ENUM_NAME(First)
//    ADD_ENUM_NAME(Second)
//    ADD_ENUM_NAME(Third)
//  END_ENUM_TABLE()
//
// Then use flags.ToString() or flags.Parse() to convert between
// representations.

#define BEGIN_ENUM_TABLE(TYPE) \
    const EnumNameEntry *GetEnumTable(TYPE disc, int &length); \
    const EnumNameEntry *GetEnumTable(TYPE disc, int &length) \
    { \
        (void)disc; \
        using ENUM_TYPE = TYPE; \
        static EnumNameEntry table[] = {
#define ADD_ENUM_NAME(ENUM) {unsigned(ENUM_TYPE::ENUM), #ENUM},
#define END_ENUM_TABLE() \
    } \
    ; \
    length = int(std::size(table)); \
    return table; \
    }
