//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

inline Shape ReshapeToNHWC(Shape shape)
{
    if ( !shape.IsValid() )
    {
        shape = {0, 0, 0, 0};
    }
    int batch = shape.AxisProduct(0, shape.Size() - 3);
    shape = Shape::PadAxes(shape, 4, 1).Extract(0, -3, -2, -1);
    shape[0] = batch;
    return shape;
}


}  // namespace regor
