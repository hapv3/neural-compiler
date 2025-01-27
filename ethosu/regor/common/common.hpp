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

// Macro to mark variables as unused
#if !defined UNUSED
#define UNUSED(x) (void)(x)
#endif

#define MACRO_EXPAND(a) a
#define MACRO_CONCAT_IMPL(a, b) a##b
#define MACRO_CONCAT(a, b) MACRO_CONCAT_IMPL(a, b)

// clang-format off
#if __GNUC__
    #define DLL_EXPORT __attribute__((visibility("default")))
    #define _strnicmp strncasecmp
#elif _WIN32
    #if TARGET_WIN32_DLL
        #define DLL_EXPORT __declspec(dllexport)
    #else
        #define DLL_EXPORT
    #endif
    #ifndef ssize_t
        #define ssize_t ptrdiff_t
    #endif
#else
    #error "undefined export semantics"
#endif
// clang-format on

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <cassert>
#include <cstdint>
#include <functional>
#include <string_view>
#include <type_traits>

namespace regor
{
using fmt::enums::format_as;
}  // namespace regor

#define DECLARE_ENUM_AS_FLAGS(ENUM_) \
    static constexpr inline ENUM_ operator&(ENUM_ a, ENUM_ b) \
    { \
        return ENUM_(std::underlying_type<ENUM_>::type(a) & std::underlying_type<ENUM_>::type(b)); \
    } \
    static constexpr inline ENUM_ operator|(ENUM_ a, ENUM_ b) \
    { \
        return ENUM_(std::underlying_type<ENUM_>::type(a) | std::underlying_type<ENUM_>::type(b)); \
    } \
    static constexpr inline ENUM_ operator^(ENUM_ a, ENUM_ b) \
    { \
        return ENUM_(std::underlying_type<ENUM_>::type(a) ^ std::underlying_type<ENUM_>::type(b)); \
    }

using UniqueId = uint32_t;

UniqueId GenerateUniqueId();

#define VERIFY(x_) (assert(x_), (x_))

namespace regor
{

#if defined __GNUC__

template<typename TYPE>
constexpr const char *PlatformRootName()
{
    const char *p = __PRETTY_FUNCTION__;
    return p;
}

template<typename TYPE>
constexpr std::string_view PlatformTypeName()
{
    const char *p = PlatformRootName<TYPE>();
    while ( (*p != 0) && (*p != '=') )
        p++;
    while ( *p == '=' || *p == ' ' )
        p++;
    const char *s = p;
    while ( (*p != 0) && (*p != ']') && (*p != ';') )
        p++;
    return std::string_view(s, size_t(p - s));
}

#elif _MSC_VER

template<typename TYPE>
constexpr const char *PlatformRootName()
{
    const char *p = __FUNCSIG__;
    return p;
}

template<typename TYPE>
constexpr std::string_view PlatformTypeName()
{
    const char *p = PlatformRootName<TYPE>();
    while ( *p++ != '<' )
        ;
    if ( (*p == 'c') || (*p == 'u') || (*p == 's') || (*p == 'e') )
    {
        if ( std::string_view(p, 7) == "struct " ) p += 7;
        else if ( std::string_view(p, 6) == "class " ) p += 6;
        else if ( std::string_view(p, 6) == "union " ) p += 6;
        else if ( std::string_view(p, 5) == "enum " ) p += 5;
    }
    std::size_t i = 0;
    while ( p[i] != '(' )
        i++;
    return std::string_view(p, i - 1);
}

#else
#error No type hash for this target
#endif

static constexpr uint32_t REGOR_FNV_SEED = 0x811c9dc5;
static constexpr uint32_t REGOR_FNV_PRIME = 0x01000193;

constexpr inline uint32_t FNVHashBytes(const char *p, int length, uint32_t hash = REGOR_FNV_SEED)
{
    while ( length-- )
    {
        hash ^= uint8_t(*p++);
        hash *= REGOR_FNV_PRIME;
    }
    return hash;
}

template<typename TYPE, bool NO_NAMESPACE>
static constexpr uint32_t PlatformTypeHash()
{
    const std::string_view name = PlatformTypeName<TYPE>();
    auto p = name.data();
    auto e = p + name.length();
    if constexpr ( NO_NAMESPACE )
    {
        while ( (*p != 0) && (*p != ':') )
            p++;
        while ( *p == ':' )
            p++;
        if ( *p == 0 ) p = name.data();  // No namespace
    }
    return FNVHashBytes(p, int(e - p));
}

// <algorithm> version not constexpr until C++20
template<typename TYPE, size_t SIZE, typename LESS>
constexpr bool is_sorted(const TYPE (&list)[SIZE], LESS func)
{
    if constexpr ( SIZE > 1 )
    {
        const TYPE *v = list;
        for ( size_t i = 1; i < SIZE; i++ )
        {
            if ( func(list[i], *v) ) return false;
            v = list + i;
        }
    }
    return true;
}

template<typename TYPE, size_t SIZE>
constexpr bool is_sorted(const TYPE (&list)[SIZE])
{
    return is_sorted(list, std::less<TYPE>());
}


}  // namespace regor
