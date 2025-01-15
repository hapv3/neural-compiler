//
// SPDX-FileCopyrightText: Copyright 2021-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "numeric_util.hpp"

#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <vector>

/// <summary>
/// Multi-axis shape description (stores axis dimensions backwards)
/// </summary>
class Shape
{
private:
    static constexpr int MAX_STATIC_AXES = 4;
    union
    {
        int32_t axes[MAX_STATIC_AXES];
        int32_t *ptr;
    } _storage;
    int8_t _last = -1;  // Invalid
    bool _dynamic = false;

public:
    Shape() { _storage.ptr = nullptr; }

    Shape(int c)
    {
        Init(1);
        At(0) = c;
    }

    Shape(int w, int c)
    {
        Init(2);
        At(0) = c;
        At(1) = w;
    }

    Shape(int h, int w, int c)
    {
        Init(3);
        At(0) = c;
        At(1) = w;
        At(2) = h;
    }

    Shape(int n, int h, int w, int c)
    {
        Init(4);
        At(0) = c;
        At(1) = w;
        At(2) = h;
        At(3) = n;
    }

    template<typename TYPE>
    Shape(const TYPE *axes, size_t length)
    {
        assert(length < size_t(std::numeric_limits<int>::max()));
        Init(int(length));
        if ( axes != nullptr )
        {
            auto *local = Storage();
            // Reverses input into position
            assert(size_t(_last) == length - 1);
            for ( size_t i = 0; i < length; i++ )
            {
                local[_last - i] = int32_t(axes[i]);
            }
        }
    }

    Shape(std::nullptr_t, int length, int fillValue = 0) { Init(length, fillValue); }

    Shape(const Shape &other)
    {
        if ( other.IsValid() )
        {
            Init(other.Size());
            std::copy_n(other.Storage(), Size(), Storage());
        }
    }

    explicit Shape(Shape &&other)
    {
        _storage = other._storage;
        _dynamic = other._dynamic;
        _last = other._last;
        other._storage.ptr = nullptr;
        other._dynamic = false;
    }

    Shape(const Shape &other, int length, int padValue = 0)
    {
        if ( other.IsValid() )
        {
            Init(length, padValue);
            std::copy_n(other.Storage(), std::min(other.Size(), length), Storage());
        }
    }

    ~Shape() { Free(); }

public:
    int &operator[](int index)
    {
        int offset = ToOffset(index);
        assert((offset >= 0) && (offset <= _last));
        return At(offset);
    }

    int operator[](int index) const
    {
        int offset = ToOffset(index);
        assert((offset >= 0) && (offset <= _last));
        return At(offset);
    }

    Shape &operator=(const Shape &other)
    {
        if ( &other != this )
        {
            Free();
            if ( other.IsValid() )
            {
                Init(other.Size());
                std::copy_n(other.Storage(), Size(), Storage());
            }
        }
        return *this;
    }

    Shape &operator=(Shape &&other)
    {
        if ( &other != this )
        {
            Free();
            _storage = other._storage;
            _dynamic = other._dynamic;
            _last = other._last;
            other._storage.ptr = nullptr;
            other._dynamic = false;
            other._last = -1;
        }
        return *this;
    }

    bool operator==(const Shape &other) const
    {
        if ( other._last != _last )
        {
            return false;
        }

        auto *from = other.Storage();
        auto *local = Storage();
        for ( int i = 0; i <= _last; i++ )
        {
            if ( local[i] != from[i] )
            {
                return false;
            }
        }
        return true;
    }

    explicit operator uint32_t() const { return At(0) ^ (At(1) << 8) ^ (At(2) << 16) ^ (At(3) << 24); }

    explicit operator uint64_t() const { return uint64_t(uint32_t(*this)); }

    explicit operator bool() const { return IsValid(); }

    bool operator!=(const Shape &other) const { return !((*this) == other); }

    // Required for use in maps/sets
    bool operator<(const Shape &other) const
    {
        auto *from = other.Storage();
        auto *local = Storage();

        for ( int i = std::min(other._last, _last); i >= 0; i-- )
        {
            if ( local[i] < from[i] )
            {
                return true;
            }
            else if ( local[i] > from[i] )
            {
                return false;
            }
        }

        return false;
    }

    bool operator<=(const Shape &other) const { return *this < other || *this == other; }

    bool operator>=(const Shape &other) const { return !(*this < other); }

    Shape operator+(const Shape &other) const { return Shape::MaxFunc<std::plus<int32_t>, false, 0>(*this, other); }

    Shape operator-(const Shape &other) const { return Shape::MaxFunc<std::minus<int32_t>, false, 0>(*this, other); }

    Shape operator%(const Shape &other) const { return Shape::MaxFunc<std::modulus<int32_t>, false, 1>(*this, other); }

    Shape operator/(const Shape &other) const { return Shape::MaxFunc<std::divides<int32_t>, false, 1>(*this, other); }

    Shape operator*(const Shape &other) const
    {
        return Shape::MaxFunc<std::multiplies<int32_t>, false, 1>(*this, other);
    }

    Shape operator*(int scale) const { return Shape::ScalarFunc<std::multiplies<int32_t>>(*this, scale); }

    Shape operator/(int scale) const { return Shape::ScalarFunc<std::divides<int32_t>>(*this, scale); }

    Shape &operator+=(const Shape &other)
    {
        Shape tmp = *this + other;
        *this = std::move(tmp);
        return *this;
    }

    Shape &operator-=(const Shape &other)
    {
        Shape tmp = *this - other;
        *this = std::move(tmp);
        return *this;
    }

    int Dot(const Shape &other) const
    {
        int result = 0;
        if ( VERIFY(other.Size() >= Size()) )
        {
            auto *from = other.Storage();
            auto *local = Storage();
            for ( int i = 0; i <= _last; i++ )
            {
                result += local[i] * from[i];
            }
        }
        return result;
    }

    // Compute a product of axes from start to end
    int AxisProduct(int start, int end) const
    {
        if ( end > Size() ) end = Size();
        if ( start == end ) return 0;
        int inner = ToOffset(end - 1);
        int outer = ToOffset(start);
        int tmp = 1;
        for ( int i = inner; i <= outer; i++ )
        {
            tmp *= At(i);
        }
        return tmp;
    }

    Shape With(int index, int value) const
    {
        Shape tmp(*this, std::max(Size(), index + 1));
        tmp.At(ToOffset(index)) = value;
        return tmp;
    }

    Shape WithBatch(int n) const
    {
        Shape tmp(*this, std::max(Size(), 4));
        tmp.At(3) = n;
        return tmp;
    }

    Shape WithHeight(int h) const
    {
        Shape tmp(*this, std::max(Size(), 3));
        tmp.At(2) = h;
        return tmp;
    }

    Shape WithWidth(int w) const
    {
        Shape tmp(*this, std::max(Size(), 2));
        tmp.At(1) = w;
        return tmp;
    }

    Shape WithDepth(int d) const
    {
        Shape tmp(*this, std::max(Size(), 1));
        tmp.At(0) = d;
        return tmp;
    }

    Shape WithHW(int h, int w) const
    {
        Shape tmp(*this, std::max(Size(), 3));
        tmp.At(2) = h;
        tmp.At(1) = w;
        return tmp;
    }

    Shape WithZeros() const { return Shape(nullptr, Size()); }

    Shape WithOnes() const { return Shape(nullptr, Size(), 1); }

    Shape Insert(int index, int value) const
    {
        Shape tmp(nullptr, Size() + 1);
        auto *result = tmp.Storage();
        auto *local = Storage();

        index = tmp.ToOffset(index);
        for ( int i = 0; i < index; i++ )
            result[i] = local[i];
        result[index] = value;
        for ( int i = index; i <= _last; i++ )
            result[i + 1] = local[i];

        return tmp;
    }

    Shape Erase(int index) const
    {
        Shape tmp(nullptr, Size() - 1);
        auto *result = tmp.Storage();
        auto *local = Storage();

        index = ToOffset(index);
        for ( int i = 0; i < index; i++ )
            result[i] = local[i];
        for ( int i = index + 1; i <= _last; i++ )
            result[i - 1] = local[i];

        return tmp;
    }

    Shape Extract(int a, int b, int c) const { return Extract({a, b, c}); }

    Shape Extract(int a, int b, int c, int d) const { return Extract({a, b, c, d}); }

    Shape Extract(std::initializer_list<int32_t> axes) const
    {
        Shape tmp(nullptr, int(axes.size()));
        auto *local = Storage();
        auto *result = tmp.Storage() + tmp.Size() - 1;
        for ( auto axis : axes )
        {
            int from = ToOffset(axis);
            assert(from < Size());
            *result-- = local[from];
        }
        return tmp;
    }

    Shape Extract(const Shape &axes) const
    {
        Shape tmp(nullptr, axes.Size());
        auto *local = Storage();
        auto *result = tmp.Storage() + tmp.Size() - 1;
        for ( int i = 0; i < axes.Size(); i++ )
        {
            int from = ToOffset(axes[i]);
            assert(from < Size());
            *result-- = local[from];
        }
        return tmp;
    }

    // Permute using 4-bit-per-axis mask with depth in the LSB
    Shape Permute(uint32_t reverseAxisMask4b) const
    {
        int length = Size();
        if ( length == 0 ) return *this;
        assert(length <= 8 && "Can only permute shapes with a most 8 axes");

        Shape tmp(nullptr, length);
        auto *local = Storage();
        auto *result = tmp.Storage();

        while ( length-- )
        {
            int from = reverseAxisMask4b & 0xF;
            assert(from < Size());
            *result++ = local[from];
            reverseAxisMask4b = reverseAxisMask4b >> 4;
        }
        assert((tmp.Elements64() == Elements64()) && "Possible bad permute (volume differs)");
        return tmp;
    }

    // Reverse permute using 4-bit-per-axis mask with depth in the LSB
    Shape Unpermute(uint32_t reverseAxisMask4b) const
    {
        int length = Size();
        if ( length == 0 ) return *this;
        assert(length <= 8 && "Can only unpermute shapes with a most 8 axes");

        Shape tmp(nullptr, length);
        auto *local = Storage();
        auto *result = tmp.Storage();

        while ( length-- )
        {
            int to = reverseAxisMask4b & 0xF;
            assert(to >= 0 && to < tmp.Size());
            result[to] = *local++;
            reverseAxisMask4b = reverseAxisMask4b >> 4;
        }
        assert((tmp.Elements64() == Elements64()) && "Possible bad unpermute (volume differs)");
        return tmp;
    }

    int Size() const { return _last + 1; }

    int Depth() const
    {
        assert(_last >= 0);
        return At(0);
    }
    int Width() const
    {
        assert(_last >= 1);
        return At(1);
    }
    int Height() const
    {
        assert(_last >= 2);
        return At(2);
    }
    int Batch() const
    {
        assert(_last >= 3);
        return At(3);
    }

    template<typename TYPE>
    Point2<TYPE> WC() const
    {
        assert(Size() >= 2);
        return Point2<TYPE>(TYPE(Width()), TYPE(Depth()));
    }

    template<typename TYPE>
    Point2<TYPE> WH() const
    {
        assert(Size() >= 3);
        return Point2<TYPE>(TYPE(Width()), TYPE(Height()));
    }

    template<typename TYPE>
    Point3<TYPE> HWC() const
    {
        assert(Size() >= 3);
        return Point3<TYPE>(TYPE(Height()), TYPE(Width()), TYPE(Depth()));
    }

    int ElementsWH() const
    {
        int64_t result = int64_t(Width()) * Height();
        assert(result <= std::numeric_limits<int>::max());
        return int(result);
    }

    int Elements() const
    {
        int64_t result = Elements64();
        assert(result <= std::numeric_limits<int>::max());
        return int(result);
    }

    int64_t Elements64() const
    {
        int64_t result = 0;
        if ( IsValid() )
        {
            auto *local = Storage();
            result = local[0];
            for ( int i = 1; i <= _last; i++ )
            {
                result *= local[i];
            }
        }
        return result;
    }

    unsigned LessMask(const Shape &other) const { return MinAxisFunc<std::less<int32_t>>(*this, other); }

    unsigned GreaterMask(const Shape &other) const { return MinAxisFunc<std::greater<int32_t>>(*this, other); }

    unsigned EqualMask(const Shape &other) const { return MinAxisFunc<std::equal_to<int32_t>>(*this, other); }

    unsigned ShapeMask() const
    {
        unsigned shift = unsigned(_last);
        shift = (shift < 32) ? (31u - shift) : 31u;
        return ~0u >> shift;
    }

    bool IsValid() const { return _last >= 0; }

    bool IsDynamic() const { return _dynamic; }

    bool IsEmpty() const
    {
        auto *local = Storage();
        for ( int i = 0; i <= _last; i++ )
        {
            if ( local[i] != 0 )
            {
                return false;
            }
        }
        return true;
    }

    bool IsSubShapeOf(const Shape &other) const
    {
        if ( Size() > other.Size() )
        {
            return false;
        }
        auto *bounds = other.Storage();
        auto *local = Storage();
        for ( int i = _last; i >= 0; i-- )
        {
            if ( local[i] > bounds[i] )
            {
                return false;
            }
        }
        return true;
    }

    // Returns true if two shapes are equal, ignoring leading dimensions that are 1
    static bool IsReducedEqual(const Shape &a, const Shape &b)
    {
        return MaxAxisFunc<std::not_equal_to<int32_t>, 1>(a, b) == 0;
    }

    template<typename TYPE>
    int ToNHWC(TYPE *buffer, size_t length) const
    {
        length = std::min(length, size_t(Size()));
        auto *local = Storage() + _last;
        for ( size_t i = 0; i < length; i++ )
        {
            *buffer++ = TYPE(local[-int(i)]);
        }
        return int(length);
    }

    template<typename TYPE>
    std::vector<TYPE> ToList() const
    {
        return std::vector<TYPE>(std::reverse_iterator<const int *>(Storage() + Size()), std::reverse_iterator<const int *>(Storage()));
    }

    uint32_t ToMask() const
    {
        uint32_t mask = 0;
        auto *local = Storage();
        for ( int i = _last; i >= 0; i-- )
        {
            int offset = ToOffset(local[i]);
            assert(offset <= 0xF);
            mask = (mask << 4) | offset;
        }
        return mask;
    }

    std::string ToString() const
    {
        std::string tmp;
        tmp.reserve(16);
        auto *local = Storage();
        for ( int i = _last; i >= 0; i-- )
        {
            tmp += std::to_string(local[i]);
            if ( i > 0 )
            {
                tmp += ", ";
            }
        }
        return tmp;
    }

private:
    void Init(int size, int fillValue = 0)
    {
        assert(size > 0);
        assert(size <= 127);
        _last = size - 1;
        _dynamic = (size > MAX_STATIC_AXES);
        int32_t *p = _dynamic ? (_storage.ptr = new int32_t[size]) : _storage.axes;
        std::fill_n(p, size, fillValue);
    }

    void Free()
    {
        if ( _dynamic ) delete[] _storage.ptr;
        _last = -1;  // Becomes invalid
        _dynamic = false;
        _storage.ptr = nullptr;
    }

    int32_t &At(int index) { return Storage()[index]; }

    const int32_t &At(int index) const { return Storage()[index]; }

    int32_t *Storage() { return _dynamic ? _storage.ptr : _storage.axes; }

    const int32_t *Storage() const { return _dynamic ? _storage.ptr : _storage.axes; }

    int ToOffset(int index) const { return (index < 0) ? (-index - 1) : (_last - index); }

    template<typename FUNC>
    static unsigned MinAxisFunc(const Shape &a, const Shape &b)
    {
        int size = std::min(a.Size(), b.Size());
        assert(size < 32);
        auto *pa = a.Storage();
        auto *pb = b.Storage();
        unsigned axisMask = 0;
        for ( int i = 0; i < size; i++ )
        {
            if ( FUNC()(pa[i], pb[i]) ) axisMask |= 1 << i;
        }
        return axisMask;
    }

    // Apply a function to the minimum number of axes between two shapes.
    template<typename FUNC>
    static Shape MinFunc(const Shape &a, const Shape &b)
    {
        int size = std::min(a.Size(), b.Size());
        Shape tmp(nullptr, size);
        auto *pa = a.Storage();
        auto *pb = b.Storage();
        auto *result = tmp.Storage();
        for ( int i = 0; i < size; i++ )
        {
            result[i] = FUNC()(pa[i], pb[i]);
        }
        return tmp;
    }

    template<typename FUNC, int MISSING_VALUE = 0>
    static unsigned MaxAxisFunc(const Shape &a, const Shape &b)
    {
        bool a_longer = a.Size() >= b.Size();
        int length = a_longer ? a.Size() : b.Size();
        assert(length < 32);
        int shortest = a_longer ? b.Size() : a.Size();
        assert(shortest < 32);

        auto *pa = a.Storage();
        auto *pb = b.Storage();
        unsigned axisMask = 0;

        int i = 0;
        for ( ; i < shortest; i++ )
        {
            if ( FUNC()(pa[i], pb[i]) ) axisMask |= 1 << i;
        }
        for ( ; i < length; i++ )
        {
            if ( a_longer && FUNC()(pa[i], MISSING_VALUE) ) axisMask |= 1 << i;
            else if ( !a_longer && FUNC()(MISSING_VALUE, pb[i]) ) axisMask |= 1 << i;
        }
        return axisMask;
    }

    // Apply a function to the maximum number of axes between two shapes. For missing
    // axes either take from the longest shape, or substitute a constant value.
    template<typename FUNC, bool TAKE_LONGEST, int MISSING_VALUE = 0>
    static Shape MaxFunc(const Shape &a, const Shape &b)
    {
        bool a_longer = a.Size() >= b.Size();
        int length = a_longer ? a.Size() : b.Size();
        int shortest = a_longer ? b.Size() : a.Size();

        Shape tmp(nullptr, length);
        auto *pa = a.Storage();
        auto *pb = b.Storage();
        auto *result = tmp.Storage();

        int i = 0;
        for ( ; i < shortest; i++ )
        {
            result[i] = FUNC()(pa[i], pb[i]);
        }
        for ( ; i < length; i++ )
        {
            if ( TAKE_LONGEST )
            {
                result[i] = a_longer ? pa[i] : pb[i];
            }
            else
            {
                result[i] = a_longer ? FUNC()(pa[i], MISSING_VALUE) : FUNC()(MISSING_VALUE, pb[i]);
            }
        }
        return tmp;
    }

    // Apply a scalar function to all axes
    template<typename FUNC>
    static Shape ScalarFunc(const Shape &a, int value)
    {
        Shape tmp(nullptr, a.Size());
        auto *pa = a.Storage();
        auto *result = tmp.Storage();
        for ( int i = 0; i < a.Size(); i++ )
        {
            result[i] = FUNC()(pa[i], value);
        }
        return tmp;
    }

    // Proxy for pointer-to-functions
    template<typename T, T (*FUNC)(T, T)>
    struct func_proxy
    {
        std::remove_reference_t<T> operator()(T a, T b) const { return FUNC(a, b); }
    };

    template<typename T>
    struct op_wrap
    {
        T operator()(const T a, const T b) const
        {
            assert(b > 0);
            return (a >= b) ? (a % b) : a;
        }
    };

public:
    template<typename TYPE>
    static Shape FromVector(const std::vector<TYPE> &from)
    {
        return from.empty() ? Shape() : Shape(from.data(), from.size());
    }

    static Shape PadAxes(const Shape &shape, int axes, int padValue)
    {
        if ( shape.Size() == axes ) return shape;
        return Shape(shape, std::max(axes, shape.Size()), padValue);
    }

    static Shape Min(const Shape &a, const Shape &b)
    {
        return Shape::MinFunc<func_proxy<const int32_t &, std::min<int32_t>>>(a, b);
    }

    static Shape Max(const Shape &a, const Shape &b)
    {
        return Shape::MaxFunc<func_proxy<const int32_t &, std::max<int32_t>>, true>(a, b);
    }

    static Shape RoundAway(const Shape &a, const Shape &b)
    {
        return Shape::MinFunc<func_proxy<int32_t, ::RoundAway<int32_t>>>(a, b);
    }

    static Shape RoundZero(const Shape &a, const Shape &b)
    {
        return Shape::MinFunc<func_proxy<int32_t, ::RoundZero<int32_t>>>(a, b);
    }

    static Shape DivRoundUp(const Shape &a, const Shape &b)
    {
        return Shape::MinFunc<func_proxy<int32_t, ::DivRoundUp<int32_t>>>(a, b);
    }

    static Shape Wrap(const Shape &a, const Shape &b) { return Shape::MinFunc<op_wrap<int32_t>>(a, b); }

    static Shape GetStridesForShape(const Shape &shape, int elementBytes)
    {
        return GetStridesForShape(shape, Shape(elementBytes));
    }

    static Shape GetStridesForShape(const Shape &shape, const Shape &granularity)
    {
        Shape tmp(nullptr, shape.Size());
        if ( shape.IsValid() )
        {
            auto *gran = granularity.Storage();
            auto *from = shape.Storage();
            auto *result = tmp.Storage();
            int lastGranule = std::min(shape._last, granularity._last);
            result[0] = gran[0];
            int i = 1;
            for ( ; i <= lastGranule; i++ )
                result[i] = ::RoundAway(result[i - 1] * from[i - 1], gran[i]);
            for ( ; i <= shape._last; i++ )
                result[i] = result[i - 1] * from[i - 1];
        }
        return tmp;
    }
};
