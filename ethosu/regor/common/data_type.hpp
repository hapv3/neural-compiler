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
#include "common/bit_flags.hpp"

#include <cassert>
#include <cstdint>
#include <limits>
#include <string>

namespace regor
{
class int48_t
{
public:
    int48_t() = default;
    int48_t(const int48_t &obj) = default;
    int48_t(int48_t &&obj) noexcept = default;
    int48_t(const int64_t val)
    {
        for ( int i = 0; i < 6; i++ )
        {
            _data[i] = (val & (uint64_t(0xFF) << i * 8)) >> (i * 8);
        }
    }

    int48_t(const uint64_t val) { int48_t(static_cast<int64_t>(val)); }

    operator int64_t() const
    {
        int64_t res = 0;
        for ( int i = 0; i < 6; i++ )
        {
            res |= uint64_t(_data[i]) << (16 + i * 8);
        }

        return res >> 16;
    }

    operator uint64_t() const { return static_cast<uint64_t>(operator int64_t()); }

private:
    uint8_t _data[6]{0};
};

enum class DataType : uint16_t
{
    None = 0,
    // Bits 1 and 2 reserved for disambiguating variably sized types
    Bits4 = 1 << 2,
    Bits8 = 1 << 3,
    Bits16 = 1 << 4,
    Bits32 = 1 << 5,
    Bits48 = Bits32 | Bits16,
    Bits64 = 1 << 6,
    Bits128 = 1 << 7,
    Signed = 1 << 8,
    Packed = 1 << 9,
    Asymmetric = 1 << 10,
    Int = 1 << 11,
    SignedInt = Signed | Int,
    Int4 = SignedInt | Bits4,
    Int4Packed8 = SignedInt | Bits8 | Bits4 | Packed,
    Int8 = SignedInt | Bits8,
    Int16 = SignedInt | Bits16,
    Int32 = SignedInt | Bits32,
    Int48 = SignedInt | Bits48,
    Int64 = SignedInt | Bits64,
    UInt8 = Int | Bits8,
    UInt16 = Int | Bits16,
    UInt32 = Int | Bits32,
    UInt48 = Int | Bits48,
    UInt64 = Int | Bits64,
    QInt = Asymmetric | SignedInt,
    QInt4 = QInt | Bits4,
    QInt8 = QInt | Bits8,
    QInt12 = QInt | Bits8 | Bits4,
    QInt16 = QInt | Bits16,
    QInt32 = QInt | Bits32,
    QUInt = Asymmetric,
    QUInt4 = QUInt | Bits4,
    QUInt8 = QUInt | Bits8,
    QUInt12 = QUInt | Bits8 | Bits4,
    QUInt16 = QUInt | Bits16,
    QUInt32 = QUInt | Bits32,
    Float = 1 << 12,
    BFloat16 = Float | Bits16 | Packed,
    Float16 = Float | Bits16,
    Float32 = Float | Bits32,
    Float64 = Float | Bits64,
    Bool = 1 << 13,
    Bool8 = Bool | Bits8,
    Complex = 1 << 14,
    Complex64 = Complex | Bits64,
    Complex128 = Complex | Bits128,
    VariablySized = 1 << 15,
    String = VariablySized | 1,
    Resource = VariablySized | 2,
    Variant = VariablySized | 3,
};

inline constexpr DataType operator&(DataType type, DataType mask)
{
    return DataType(unsigned(type) & unsigned(mask));
}
inline constexpr DataType operator&(DataType type, unsigned mask)
{
    return DataType(unsigned(type) & mask);
}
inline constexpr DataType operator|(DataType type, DataType mask)
{
    return DataType(unsigned(type) | unsigned(mask));
}
inline constexpr DataType operator|(DataType type, unsigned mask)
{
    return DataType(unsigned(type) | mask);
}
inline constexpr bool operator!(DataType type)
{
    return type == DataType::None;
}


static inline int Clz(uint32_t value)
{
    // Ensure all CLZ implementations return '32 zeroes'
    // for a zero value input.
    if ( value == 0 )
    {
        return 32;
    }
#if defined(__GNUC__)
    return __builtin_clz(value);
#elif defined(_MSC_VER)
    unsigned long index;
    _BitScanReverse(&index, value);
    return int(31 - index);
#else
#error "Missing platform CLZ32 implementation"
#endif
}

inline constexpr bool IsPacked(DataType type)
{
    return (type & DataType::Packed) == DataType::Packed;
}

inline constexpr bool IsVariablySized(DataType type)
{
    return (type & DataType::VariablySized) == DataType::VariablySized;
}

inline constexpr int DataTypeStorageSizeBits(DataType type)
{
    assert(!IsVariablySized(type));
    unsigned bits = unsigned(type & 0x00FFu);
    // Interpret packed word size as the largest set bit
    if ( IsPacked(type) )
    {
        assert(bits > 0);
        return 1 << (31 - Clz(bits));
    }
    return std::max(int(bits), 8);
}

inline constexpr int DataTypeSizeBits(DataType type)
{
    assert(!IsVariablySized(type));
    unsigned bits = unsigned(type & 0x00FFu);
    if ( IsPacked(type) )
    {
        assert(bits > 0);
        bits ^= 1 << (31 - Clz(bits));  // Strip container word
    }
    assert(bits > 0);
    return int(bits);
}

inline constexpr int DataTypeStorageSizeBytes(DataType type, int elements)
{
    const int storageBits = DataTypeStorageSizeBits(type);
    const int bits = IsPacked(type) ? DataTypeSizeBits(type) : storageBits;
    assert(storageBits >= 8);
    int64_t result = (((int64_t(elements) * bits) + storageBits - 1) / storageBits) * (storageBits / 8);
    assert(result < std::numeric_limits<int>::max());
    return int(result);
}


inline constexpr int DataTypeElements(DataType type, int size)
{
    const int bits = IsPacked(type) ? DataTypeSizeBits(type) : DataTypeStorageSizeBits(type);
    assert(size <= std::numeric_limits<int>::max() / 8);
    return 8 * size / bits;
}

inline constexpr DataType DataTypeBase(DataType type)
{
    return (!IsVariablySized(type) ? type & 0xFF00u : type);
}

inline std::string DataTypeToString(const DataType type)
{
    return EnumToString<DataType>(type);
}

inline constexpr bool IsInteger(DataType type)
{
    return (type & DataType::Int) == DataType::Int;
}

inline constexpr bool IsSignedInteger(DataType type)
{
    return (type & DataType::SignedInt) == DataType::SignedInt;
}

inline constexpr bool IsFloat(DataType type)
{
    return (type & DataType::Float) == DataType::Float;
}

inline constexpr bool IsBool(DataType type)
{
    return (type & DataType::Bool) == DataType::Bool;
}

inline constexpr uint64_t IntegerMax(DataType type)
{
    assert(IsInteger(type) || IsBool(type));
    return ~0ULL >> (64 - DataTypeSizeBits(type) + int(IsSignedInteger(type) || IsBool(type)));
}

static_assert(uint64_t(std::numeric_limits<int8_t>::max()) == IntegerMax(DataType::Int8));
static_assert(uint64_t(std::numeric_limits<int16_t>::max()) == IntegerMax(DataType::Int16));
static_assert(uint64_t(std::numeric_limits<int32_t>::max()) == IntegerMax(DataType::Int32));
static_assert(uint64_t(std::numeric_limits<int64_t>::max()) == IntegerMax(DataType::Int64));

inline constexpr int64_t IntegerMin(DataType type)
{
    assert(IsInteger(type) || IsBool(type));
    if ( IsSignedInteger(type) || IsBool(type) )
    {
        int size = DataTypeSizeBits(type);
        return -1 * (1ULL << (size - 1));
    }
    return 0;
}

static_assert(std::numeric_limits<int8_t>::min() == IntegerMin(DataType::Int8));
static_assert(std::numeric_limits<int16_t>::min() == IntegerMin(DataType::Int16));
static_assert(std::numeric_limits<int32_t>::min() == IntegerMin(DataType::Int32));
static_assert(std::numeric_limits<int64_t>::min() == IntegerMin(DataType::Int64));

template<typename T>
struct DataTypeOf
{
    static constexpr DataType value = DataType::None;
};
template<>
struct DataTypeOf<int8_t>
{
    static constexpr DataType value = DataType::Int8;
};
template<>
struct DataTypeOf<int16_t>
{
    static constexpr DataType value = DataType::Int16;
};
template<>
struct DataTypeOf<int32_t>
{
    static constexpr DataType value = DataType::Int32;
};
template<>
struct DataTypeOf<int64_t>
{
    static constexpr DataType value = DataType::Int64;
};
template<>
struct DataTypeOf<uint8_t>
{
    static constexpr DataType value = DataType::UInt8;
};
template<>
struct DataTypeOf<uint16_t>
{
    static constexpr DataType value = DataType::UInt16;
};
template<>
struct DataTypeOf<uint32_t>
{
    static constexpr DataType value = DataType::UInt32;
};
template<>
struct DataTypeOf<uint64_t>
{
    static constexpr DataType value = DataType::UInt64;
};

}  // namespace regor
