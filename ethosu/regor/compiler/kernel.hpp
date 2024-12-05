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

#include "common/common.hpp"

#include "common/numeric_util.hpp"
#include "include/graphapi.hpp"

#include <cassert>
#include <string>

namespace regor
{

class Margin
{
private:
    int _top = 0;
    int _left = 0;
    int _bottom = 0;
    int _right = 0;
    int _near = 0;
    int _far = 0;

public:
    Margin(int top, int left, int bottom, int right) : _top(top), _left(left), _bottom(bottom), _right(right) {}
    Margin(int top, int left, int bottom, int right, int near, int far) :
            _top(top), _left(left), _bottom(bottom), _right(right), _near(near), _far(far)
    {
    }

    Margin() = default;

    int Top() const { return _top; }
    int Left() const { return _left; }
    int Bottom() const { return _bottom; }
    int Right() const { return _right; }
    int Near() const { return _near; }
    int Far() const { return _far; }

    bool IsZero() const { return !(_top | _left | _bottom | _right | _near | _far); }

    std::string ToString() const
    {
        return fmt::format("[t:{},l:{},b:{},r:{},n:{},f:{}]", _top, _left, _bottom, _right, _near, _far);
    }
};

/// <summary>
/// Kernel parameters
/// </summary>
class Kernel
{
private:
    Point2i _size;
    Point2i _stride;
    Point2i _dilation;
    int _sizeZ = 0;
    int _strideZ = 0;
    int _dilationZ = 0;
    Margin _padding;
    int _depthMultiplier = 0;


public:
    Kernel(const GraphApi::GraphKernel *kernel)
    {
        _size = Point2i(kernel->sizeYXZ[1], kernel->sizeYXZ[0]);
        _sizeZ = kernel->sizeYXZ[2];
        _stride = Point2i(kernel->strideYXZ[1], kernel->strideYXZ[0]);
        _strideZ = kernel->strideYXZ[2];
        _dilation = Point2i(kernel->dilationYXZ[1], kernel->dilationYXZ[0]);
        _dilationZ = kernel->dilationYXZ[2];
        _padding = Margin(kernel->paddingTBLRNF[0], kernel->paddingTBLRNF[2], kernel->paddingTBLRNF[1],
            kernel->paddingTBLRNF[3], kernel->paddingTBLRNF[4], kernel->paddingTBLRNF[5]);
        _depthMultiplier = 0;
        assert(_size.x > 0 && _size.y > 0);
    }

    Kernel(Point2i size, Point2i stride, Point2i dilation, int depthMultiplier = 1, Margin padding = Margin(0, 0, 0, 0))
    {
        assert(size.x > 0 && size.y > 0);
        assert(stride.x > 0 && stride.y > 0);
        _size = size;
        _stride = stride;
        _dilation = dilation;
        _depthMultiplier = depthMultiplier;
        _padding = padding;
    }
    Kernel() = default;

    int ElementsWH() const { return _size.x * _size.y; }
    const Point3<int> Size3D() const { return {_size.x, _size.y, _sizeZ}; }
    const Point3<int> Stride3D() const { return {_stride.x, _stride.y, _strideZ}; }
    const Point3<int> Dilation3D() const { return {_dilation.x, _dilation.y, _dilationZ}; }
    const Point2i &Size() const { return _size; }
    const Point2i &Stride() const { return _stride; }
    const Point2i &Dilation() const { return _dilation; }
    int DepthMultiplier() const { return _depthMultiplier; }
    const Margin &Padding() const { return _padding; }

    Kernel WithSize(Point2i size) const
    {
        Kernel tmp(*this);
        tmp._size = size;
        return tmp;
    }

    Kernel WithStride(Point2i stride) const
    {
        Kernel tmp(*this);
        tmp._stride = stride;
        return tmp;
    }

    Kernel WithDilation(Point2i dilation) const
    {
        Kernel tmp(*this);
        tmp._dilation = dilation;
        return tmp;
    }

    Kernel WithPadding(Margin padding) const
    {
        Kernel tmp(*this);
        tmp._padding = padding;
        return tmp;
    }

    Kernel WithDepthMultiplier(int depthMultiplier) const
    {
        Kernel tmp(*this);
        tmp._depthMultiplier = depthMultiplier;
        return tmp;
    }

    Point2i DilatedWH() const { return (_dilation * (_size - Point2i(1, 1))) + Point2i(1, 1); }

    std::string ToString() const
    {
        return fmt::format("size={},{} stride={},{}, dilation={},{} padding={}", _size.x, _size.y, _stride.x, _stride.y,
            _dilation.x, _dilation.y, _padding.ToString());
    }

    static Kernel UnitKernel() { return Kernel({1, 1}, {1, 1}, {1, 1}); }
};

static inline int RequiredInputSize(int value, int stride, int border, int upscale, int rounding = 0)
{
    return int(std::ceil(float((value - 1) * stride + border + rounding) / float(upscale)));
}

}  // namespace regor
