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

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>
#if _MSC_VER
#include <intrin.h>
#endif

template<typename T>
bool CheckSafeAdd(T a, T b)
{
#ifdef __GNUC__
    T res;
    return !__builtin_add_overflow(a, b, &res);
#else
    return (!(
        (a > 0 && b > std::numeric_limits<T>::max() - a) ||
        ((std::is_signed<T>::value && (a < 0 && b < std::numeric_limits<T>::min() - a)))));
#endif
}

template<typename T>
bool CheckSafeSub(T a, T b)
{
#ifdef __GNUC__
    T res;
    return !__builtin_sub_overflow(a, b, &res);
#else
    return !(
        (a > 0 && b < std::numeric_limits<T>::min() + a) ||
        (std::is_signed<T>::value && (a < 0 && b > std::numeric_limits<T>::max() + a)) || (std::is_unsigned<T>::value && a < b));
#endif
}

template<typename T>
bool CheckSafeMul(T a, T b)
{
#ifdef __GNUC__
    T res;
    return !__builtin_mul_overflow(a, b, &res);
#else
    return !(
        (std::max(a, b) == -1 && std::min(a, b) == std::numeric_limits<T>::min()) ||
        (b != 0 && (a > std::numeric_limits<T>::max() / b || (std::is_unsigned<T>::value && (a < std::numeric_limits<T>::min() / b)))));
#endif
}

template<typename T>
T SafeAdd(T a, T b)
{
    if ( !CheckSafeAdd(a, b) ) throw std::overflow_error("Addition overflow");
    return a + b;
}

template<typename T>
T SafeSub(T a, T b)
{
    if ( !CheckSafeSub(a, b) ) throw std::overflow_error("Subtraction overflow");
    return a - b;
}

template<typename T>
T SafeMul(T a, T b)
{
    if ( !CheckSafeMul(a, b) ) throw std::overflow_error("Multiplication overflow");
    return a * b;
}

template<typename T>
T AssertAdd(T a, T b)
{
    assert(CheckSafeAdd(a, b));
    return a + b;
}
template<typename T>
T AssertSub(T a, T b)
{
    assert(CheckSafeSub(a, b));
    return a - b;
}

template<typename T>
T AssertMul(T a, T b)
{
    assert(CheckSafeMul(a, b));
    return a * b;
}

template<typename TYPE>
class Point2
{
public:
    TYPE x = 0, y = 0;

public:
    Point2(TYPE xx = 0, TYPE yy = 0) : x(xx), y(yy) {}

public:
    TYPE AreaXY() const { return x * y; }

    Point2<TYPE> operator+(const Point2<TYPE> &pt) const
    {
        return Point2<TYPE>(AssertAdd(x, pt.x), AssertAdd(y, pt.y));
    }
    Point2<TYPE> operator-(const Point2<TYPE> &pt) const
    {
        return Point2<TYPE>(AssertSub(x, pt.x), AssertSub(y, pt.y));
    }
    Point2<TYPE> operator*(const Point2<TYPE> &pt) const
    {
        return Point2<TYPE>(AssertMul(x, pt.x), AssertMul(y, pt.y));
    }
    Point2<TYPE> operator/(const Point2<TYPE> &pt) const { return Point2<TYPE>(x / pt.x, y / pt.y); }

    bool operator==(const Point2<TYPE> &pt) const { return (x == pt.x) && (y == pt.y); }
    bool operator!=(const Point2<TYPE> &pt) const { return !((*this) == pt); }
    bool operator<(const Point2<TYPE> &pt) const { return (x < pt.x) || ((x == pt.x) && (y < pt.y)); }

    static Point2<TYPE> Min(const Point2<TYPE> &a, const Point2<TYPE> &b)
    {
        return Point2<TYPE>(std::min(a.x, b.x), std::min(a.y, b.y));
    }

    static Point2<TYPE> Max(const Point2<TYPE> &a, const Point2<TYPE> &b)
    {
        return Point2<TYPE>(std::max(a.x, b.x), std::max(a.y, b.y));
    }

    explicit operator uint32_t() const { return (uint32_t(x) << 16) ^ y; }

    explicit operator uint64_t() const { return (uint64_t(x) << 16) ^ y; }
};

template<typename TYPE>
struct Point2Hash
{
    size_t operator()(const Point2<TYPE> &pt) const { return (pt.x * 8191) ^ pt.y; }
};

using Point2i = Point2<int>;

template<typename TYPE>
class Point3
{
public:
    TYPE x = 0, y = 0, z = 0;

public:
    Point3(TYPE xx = 0, TYPE yy = 0, TYPE zz = 0) : x(xx), y(yy), z(zz) {}

public:
    TYPE AreaXY() const { return x * y; }

    Point3<TYPE> operator+(const Point3<TYPE> &pt) const { return Point3<TYPE>(x + pt.x, y + pt.y, z + pt.z); }
    Point3<TYPE> operator-(const Point3<TYPE> &pt) const { return Point3<TYPE>(x - pt.x, y - pt.y, z - pt.z); }
    Point3<TYPE> operator*(const Point3<TYPE> &pt) const { return Point3<TYPE>(x * pt.x, y * pt.y, z * pt.z); }
    Point3<TYPE> operator/(const Point3<TYPE> &pt) const { return Point3<TYPE>(x / pt.x, y / pt.y, z / pt.z); }

    bool operator==(const Point3<TYPE> &pt) const { return (x == pt.x) && (y == pt.y) && (z == pt.z); }
    bool operator!=(const Point3<TYPE> &pt) const { return !((*this) == pt); }
    bool operator<(const Point3<TYPE> &pt) const
    {
        return (x < pt.x) || ((x == pt.x) && (y < pt.y)) || ((x == pt.x) && (y == pt.y) && (z < pt.z));
    }
};

template<typename TYPE>
struct Fraction
{
public:
    TYPE n = 0, d = 0;

public:
    Fraction() = default;
    Fraction(TYPE numerator, TYPE denominator) : n(numerator), d(denominator) {}
    template<typename TYPE2>
    Fraction(const Point2<TYPE2> pt) : n(pt.x), d(pt.y)
    {
    }

public:
    operator Point2<TYPE>() { return Point2<TYPE>(n, d); }
};

template<typename TYPE>
TYPE RoundAway(TYPE value, TYPE align)
{
    assert(align > 0);
    TYPE rem = value % align;
    if ( rem == 0 )
    {
        return value;
    }
    else if ( rem < 0 )
    {
        return value - (align + rem);
    }
    return value + (align - rem);
}

inline float RoundAway(float value, float align)
{
    assert(align > 0);
    if ( value < 0 )
    {
        value = value - align + 1;
    }
    else
    {
        value = value + align - 1;
    }
    return std::truncf(value / align) * align;
}

template<typename TYPE>
TYPE RoundZero(TYPE value, TYPE align)
{
    assert(align > 0);
    return value - (value % align);
}

inline float RoundZero(float value, float align)
{
    return std::truncf(value / align) * align;
}


template<typename TYPE>
TYPE DivRoundUp(TYPE a, TYPE b)
{
    return TYPE((a + b - 1) / b);
}

// Checks if the ranges overlap, to0 and to1 are exclusive
template<typename TYPE>
bool Overlaps(TYPE from0, TYPE to0, TYPE from1, TYPE to1)
{
    return from0 < to1 && from1 < to0;
}

template<typename TYPE>
TYPE ClampSigmoid(TYPE x, TYPE limit)
{
    if ( x <= -limit )
    {
        return TYPE(0);
    }
    else if ( x >= limit )
    {
        return TYPE(1);
    }
    else
    {
        return TYPE(1 / (1 + std::exp(-x)));
    }
}

inline int NeededTotalPadding(int inputSize, int outputSize, int stride, int filterSize)
{
    int outSize = DivRoundUp(outputSize, stride);
    int neededInput = (outSize - 1) * stride + filterSize;
    return std::max(0, neededInput - inputSize);
}

inline int NeededTotalPadding(int inputSize, int stride, int filterSize)
{
    return NeededTotalPadding(inputSize, inputSize, stride, filterSize);
}

template<typename TYPE>
std::make_unsigned_t<TYPE> ToUnsigned(TYPE x)
{
    assert(x >= 0);
    return static_cast<std::make_unsigned_t<TYPE>>(x);
}

inline int Clz32(uint32_t x)
{
#if __GNUC__
    return x == 0 ? 32 : __builtin_clz(x);
#elif _MSC_VER
    unsigned long leading_zero = 0;
    return _BitScanReverse(&leading_zero, x) ? 31 - leading_zero : 32;
#else
#error "Unsupported toolchain"
#endif
}

inline int Clz64(uint64_t x)
{
#if __GNUC__
    return x == 0 ? 64 : __builtin_clzll(x);
#elif _MSC_VER
    unsigned long leading_zero = 0;

#if defined(_WIN64)
    return _BitScanReverse64(&leading_zero, x) ? 63 - leading_zero : 64;
#elif defined(_WIN32)
    // WIN32 does not contain _BitScanReverse64 so split into high and low parts
    uint32_t x_h = x >> 32;
    uint32_t x_l = x & 0xffffffff;
    if ( x_h != 0 )
    {
        // There is a set bit in the high bits
        return _BitScanReverse(&leading_zero, x_h) ? 31 - leading_zero : 32;
    }
    else
    {
        // There is no set bit in the high bits, try lower bits
        return _BitScanReverse(&leading_zero, x_l) ? 63 - leading_zero : 64;
    }
#else
#error "Unsupported MSVC platform"
#endif
#else
#error "Unsupported toolchain"
#endif
}

template<typename T>
int IntLog2(T x)
{
    if ( x <= T(0) ) return 0;

    static_assert(std::is_arithmetic_v<T> && (sizeof(T) <= 8), "");
    using itype = std::conditional_t<std::is_floating_point_v<T> || sizeof(T) == 8, uint64_t, uint32_t>;
    itype n;

    if constexpr ( std::is_floating_point_v<T> )
    {
        n = itype(std::ceil(x));
    }
    else
    {
        n = itype(x);
    }

    if constexpr ( std::is_same_v<itype, uint64_t> )
    {
        return 63 - Clz64(ToUnsigned(n));
    }
    else
    {
        return 31 - Clz32(ToUnsigned(n));
    }
}

template<typename T>
constexpr bool IsPowerOfTwo(T x)
{
    static_assert(std::is_integral_v<T>, "");
    return x > 0 && (x & (x - 1)) == 0;
}

// Count the number of nonzero nybbles
inline unsigned NonZeroNybbles(unsigned mask)
{
    mask |= mask >> 2;   // technically & with 0xCCCC and 0x2222
    mask |= mask >> 1;   // but bits don't tavel far enough to leak
    mask &= 0x11111111;  // =...A...A...A...A...A...A...A...A
    mask += mask >> 16;  // +...x...x...x...x...B...B...B...B
    mask += mask >> 8;   // +...x...x...x...x...x...x...C...C
    mask += mask >> 4;   // +...x...x...x...x...x...x...x...D
    return mask & 0xF;
};

template<typename OUT, typename IN>
OUT ClampToType(IN x)
{
    static_assert(std::is_floating_point_v<IN> == std::is_floating_point_v<OUT>, "");
    IN hi = std::numeric_limits<OUT>::max();
    IN lo;
    if constexpr ( std::is_floating_point_v<OUT> )
    {
        lo = -std::numeric_limits<OUT>::max();
    }
    else
    {
        lo = std::numeric_limits<OUT>::min();
    }
    return OUT(std::clamp(x, lo, hi));
}
