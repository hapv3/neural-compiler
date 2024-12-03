//
// SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

// PART OF EXTERNAL INTERFACE
// REQUIRED: NO NAMESPACE, NO HEADER GUARDS

namespace detail
{

constexpr inline uint32_t FNVHash(const char *p)
{
    uint32_t hash = 0x811c9dc5;  // FNV-1a SEED
    while ( *p )
        hash = (hash ^ uint8_t(*p++)) * 0x01000193;  // FNV-1a PRIME
    return hash;
}

constexpr inline uint32_t MakeAttributeId(uint32_t hash, unsigned type, unsigned index)
{
    return (hash << 12) | (index & 0xFF) << 4 | (type & 0xF);
}

}  // namespace detail

#ifndef GRAPHAPI_FUNCTION_DETAIL_ONLY

constexpr uint32_t GRAPHAPI_MAKE_ATTR_ID(const char *name, unsigned type, unsigned index)
{
    return detail::MakeAttributeId(detail::FNVHash(name), type, index);
}

// Validation type-codes (invalid mappings will be rejected)
static constexpr unsigned GRAPHAPI_TYPECODE_bool = (1);
static constexpr unsigned GRAPHAPI_TYPECODE_int32 = (2);
static constexpr unsigned GRAPHAPI_TYPECODE_double = (3);
static constexpr unsigned GRAPHAPI_TYPECODE_string = (4);
static constexpr unsigned GRAPHAPI_TYPECODE_GraphShape = (5);
static constexpr unsigned GRAPHAPI_TYPECODE_FractionND = (6);
static constexpr unsigned GRAPHAPI_TYPECODE_Point2 = (7);

// NAME: Name of the internal attribute
// TYPE: Type that we want to you to pass in (not the internal type)
// UNIQUE: Discriminator value for this named attribute
#define GRAPHAPI_MAKE_ATTR(NAME_, TYPE_, UNIQUE_) \
    GRAPHAPI_MAKE_ATTR_ID(#NAME_ "_attr_t", GRAPHAPI_TYPECODE_##TYPE_, UNIQUE_)


enum class OpAttr : uint32_t
{
    // Pooling
    POOL_PRECISION = GRAPHAPI_MAKE_ATTR(pooling, int32, 0),
    // Axis
    AXIS_SELECT = GRAPHAPI_MAKE_ATTR(axis, int32, 0),
    // Reshape
    RESHAPE_SHAPE = GRAPHAPI_MAKE_ATTR(reshape, GraphShape, 0),
    // Slice
    SLICE_BEGIN = GRAPHAPI_MAKE_ATTR(slice, GraphShape, 0),
    SLICE_SIZE = GRAPHAPI_MAKE_ATTR(slice, GraphShape, 1),
    // Resize
    RESIZE_SCALEX = GRAPHAPI_MAKE_ATTR(resize, FractionND, 0),
    RESIZE_SCALEY = GRAPHAPI_MAKE_ATTR(resize, FractionND, 1),
    RESIZE_OFFSET = GRAPHAPI_MAKE_ATTR(resize, Point2, 2),
    RESIZE_BORDER = GRAPHAPI_MAKE_ATTR(resize, Point2, 3),
    RESIZE_MODE = GRAPHAPI_MAKE_ATTR(resize, int32, 4),
    // Clamp
    CLAMP_MIN = GRAPHAPI_MAKE_ATTR(clamp, double, 0),
    CLAMP_MAX = GRAPHAPI_MAKE_ATTR(clamp, double, 1),
    // Rescale
    RESCALE_SCALE32 = GRAPHAPI_MAKE_ATTR(rescale, bool, 0),
    RESCALE_DOUBLE_ROUND = GRAPHAPI_MAKE_ATTR(rescale, bool, 1),
    RESCALE_PER_CHANNEL = GRAPHAPI_MAKE_ATTR(rescale, bool, 2),
    // Sign
    RESCALE_INPUT_UNSIGNED = GRAPHAPI_MAKE_ATTR(sign, bool, 0),
    RESCALE_OUTPUT_UNSIGNED = GRAPHAPI_MAKE_ATTR(sign, bool, 1),
    // Mul
    MUL_SHIFT = GRAPHAPI_MAKE_ATTR(mul, int32, 0),
    // Asr
    ASR_ROUND = GRAPHAPI_MAKE_ATTR(asr, bool, 0),
    // Conditional branch
    COND_IF = GRAPHAPI_MAKE_ATTR(cond, string, 0),
    COND_ELSE = GRAPHAPI_MAKE_ATTR(cond, string, 1),
    // While loop
    WHILE_COND = GRAPHAPI_MAKE_ATTR(while, string, 0),
    WHILE_BODY = GRAPHAPI_MAKE_ATTR(while, string, 1),
    // Transpose Conv2D
    TRANSPOSE_CONV2D_OUTSHAPE = GRAPHAPI_MAKE_ATTR(transpose_conv2d, GraphShape, 0),
    TRANSPOSE_CONV2D_OUTPAD = GRAPHAPI_MAKE_ATTR(transpose_conv2d, GraphShape, 1),
    // Transpose
    TRANSPOSE_PERM = GRAPHAPI_MAKE_ATTR(transpose, GraphShape, 0),
    // FFT
    FFT_INVERSE = GRAPHAPI_MAKE_ATTR(fft, bool, 0),
    // CUSTOM
    CUSTOM_NAME = GRAPHAPI_MAKE_ATTR(custom, string, 0),
    CUSTOM_DOMAIN = GRAPHAPI_MAKE_ATTR(custom, string, 1),
    // PAD
    PAD_PAD_CONST = GRAPHAPI_MAKE_ATTR(pad, double, 0),
};

#endif
