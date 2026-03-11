//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "common/transpose_type.hpp"

#include <assert.h>
#include <numeric>

namespace regor
{

// Convert a permutation shape (up to 8 elements) to a TransposeType
// For example:
// [0, 1, 2, 3] -> 0x0123 ("NHWC")
// [0, 1, 2] -> 0x0123 ("NHWC")
// [0, 1] -> 0x0123 ("NHWC")
// [0] -> 0x0123 ("NHWC")
// [0, 2, 1, 3] -> 0x0213 ("NWHC")
// [1, 0, 2] -> 0x0213 ("NWHC")
inline TransposeType TransposeTypeFromShape(const Shape &perm)
{
    const int n = perm.Size();
    // We can only handle permutation vectors up 8 elements
    if ( n > 8 ) throw std::invalid_argument("Permutation shape has more than 8 elements");
    uint32_t mask = perm.ToMask();
    uint32_t offset = 0x76543210 & ~(0xFFFFFFFF >> (4 * (8 - n)));
    uint32_t mask8D = mask + offset;
    return TransposeType(mask8D);
}

// Checks if a permutation vector is a no-op for a given shape, ignoring unit axes
inline bool IsNoOpPermuteForShape(const Shape &shape, const Shape &perm)
{
    assert(shape.Size() == perm.Size());

    int lastAxis = 0;
    for ( int i = 0; i < perm.Size(); i++ )
    {
        const int axis = perm[i];
        if ( shape[axis] == 1 ) continue;     // Ignore unit axes
        if ( axis < lastAxis ) return false;  // Permute vector is not ordered
        lastAxis = axis;
    }
    return true;
}

// Reshape for example (A, B, N, H, W, C) + (3, 2, 1) -> (A*B*N, H*W, C)
inline Shape ReshapeTo3D(const Shape &shape, const Shape &axes, int minAxis = 1)
{
    assert(axes.Size() == 3);
    assert(axes[0] + axes[1] + axes[2] == shape.Size());
    int h = std::max(minAxis, shape.AxisProduct(0, axes[0]));
    int w = std::max(minAxis, shape.AxisProduct(axes[0], axes[0] + axes[1]));
    int c = std::max(minAxis, shape.AxisProduct(axes[0] + axes[1], axes[0] + axes[1] + axes[2]));
    return Shape(h, w, c);
}

// Reshape for example (B, N, H, W, C) + W -> (B*N*H, W, C)
inline Shape ReshapeTo3DAroundAxis(const Shape &shape, int axis, int minAxis = 1)
{
    assert(axis >= 0);
    assert(axis < shape.Size());
    int outer = axis;
    int inner = shape.Size() - axis - 1;
    return ReshapeTo3D(shape, {outer, 1, inner}, minAxis);
}

// Reshape (B, N, H, W, C) -> (B, N*H*W, C)
inline Shape ReshapeTo3DAroundEdges(const Shape &shape, int minAxis = 1)
{
    assert(shape.Size() > 1);
    return ReshapeTo3D(shape, {1, shape.Size() - 2, 1}, minAxis);
}

inline Shape ReshapeToNHWC(const Shape &shape)
{
    if ( shape.Size() == 4 ) return shape;
    else if ( shape.Size() < 4 ) return Shape::PadAxes(shape, 4, 1);
    else return Shape::MergeAxes(shape, -4, 0xFFFFFFF8, false);
}

inline Shape ReshapeToHWC(const Shape &shape)
{
    if ( shape.Size() == 3 ) return shape;
    else if ( shape.Size() < 3 ) return Shape::PadAxes(shape, 3, 1);
    else return Shape::MergeAxes(shape, -3, 0xFFFFFFFC, false);
}

inline bool IsContiguousStrides(const Shape &strides)
{
    for ( int i = 0; i < strides.Size() - 1; i++ )
    {
        if ( strides[i] % strides[i + 1] != 0 )
        {
            return false;
        }
    }
    return true;
}

// Return a copy of the shape with all unit (== 1) axes removed.
// For example:
//   [1, 3, 1, 4] -> [3, 4]
//   [1, 1]       -> []
inline Shape Squeeze(const Shape &shape)
{
    int nonUnitAxes = 0;
    for ( int i = 0; i < shape.Size(); i++ )
    {
        if ( shape[i] != 1 )
        {
            nonUnitAxes++;
        }
    }

    if ( nonUnitAxes == 0 )
    {
        return Shape();
    }

    Shape result(nullptr, nonUnitAxes);
    int write = 0;
    for ( int i = 0; i < shape.Size(); i++ )
    {
        if ( shape[i] != 1 )
        {
            result[write++] = shape[i];
        }
    }
    return result;
}

// Reshape a shape so that the depth axis is at least a certain size
inline Shape ReshapeToIncreaseDepth(const Shape &shape, int minDepth)
{
    if ( !shape ) return shape;

    // Factor shape into primes
    std::array<int, 32> primes;
    int primesCount = 0;
    int elements = shape.Elements();
    if ( elements == 0 ) return {0, 0, 0, 0};
    for ( ; elements % 2 == 0; elements /= 2 )
    {
        // Factor it in 2s
        primes[primesCount++] = 2;
    }
    const int sqrt = int(std::floor(std::sqrt(std::abs(elements))));
    for ( int factor = 3; std::abs(elements) > 1 && factor <= sqrt; factor += 2 )
    {
        // Factor it in 3s, 5s, 7s, ...
        for ( ; elements % factor == 0; elements /= factor )
        {
            primes[primesCount++] = factor;
        }
    }
    if ( elements != 1 ) primes[primesCount++] = elements;
    assert(primesCount <= 32);
    assert(std::accumulate(primes.begin(), primes.begin() + primesCount, 1, std::multiplies<int>()) == shape.Elements());

    // Build a depth at least as large as the depth granule
    int depth = 1;
    int primesIdx = 0;
    for ( ; primesIdx < primesCount && depth < minDepth; primesIdx++ )
    {
        depth *= primes[primesIdx];
    }

    // Divide the rest between height and width, but prefer height > width
    int height = 1;
    int width = 1;
    for ( ; primesIdx < primesCount; primesIdx++ )
    {
        if ( width < height ) width *= primes[primesIdx];
        else height *= primes[primesIdx];
    }
    if ( height < width ) std::swap(height, width);
    assert(height * width * depth == shape.Elements());

    return {1, height, width, depth};
}

}  // namespace regor
