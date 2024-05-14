//
// SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include <cstdint>

namespace tosa
{


enum class DType : uint32_t
{
    UNKNOWN = 0,
    BOOL = 1,
    UINT8 = 2,
    INT4 = 3,
    INT8 = 4,
    INT16 = 5,
    INT32 = 6,
    INT48 = 7,
    FP32 = 8,
    UINT16 = 9,
    FP16 = 10,
    BF16 = 11,
};

enum class ResizeMode : uint32_t
{
    UNKNOWN = 0,
    NEAREST = 1,
    BILINEAR = 2,
};

enum class Op : uint32_t
{
    UNKNOWN = 0,
    ARGMAX = 1,
    AVG_POOL2D = 2,
    CONV2D = 3,
    CONV3D = 4,
    DEPTHWISE_CONV2D = 5,
    FULLY_CONNECTED = 6,
    MATMUL = 7,
    MAX_POOL2D = 8,
    TRANSPOSE_CONV2D = 9,
    CLAMP = 10,
    RESERVED = 11,
    SIGMOID = 12,
    TANH = 13,
    ADD = 14,
    ARITHMETIC_RIGHT_SHIFT = 15,
    BITWISE_AND = 16,
    BITWISE_OR = 17,
    BITWISE_XOR = 18,
    INTDIV = 19,
    LOGICAL_AND = 20,
    LOGICAL_LEFT_SHIFT = 21,
    LOGICAL_RIGHT_SHIFT = 22,
    LOGICAL_OR = 23,
    LOGICAL_XOR = 24,
    MAXIMUM = 25,
    MINIMUM = 26,
    MUL = 27,
    POW = 28,
    SUB = 29,
    TABLE = 30,
    ABS = 31,
    BITWISE_NOT = 32,
    CEIL = 33,
    CLZ = 34,
    EXP = 35,
    FLOOR = 36,
    LOG = 37,
    LOGICAL_NOT = 38,
    NEGATE = 39,
    RECIPROCAL = 40,
    RSQRT = 41,
    SELECT = 42,
    EQUAL = 43,
    GREATER = 44,
    GREATER_EQUAL = 45,
    REDUCE_ANY = 46,
    REDUCE_ALL = 47,
    REDUCE_MAX = 48,
    REDUCE_MIN = 49,
    REDUCE_PRODUCT = 50,
    REDUCE_SUM = 51,
    CONCAT = 52,
    PAD = 53,
    RESHAPE = 54,
    REVERSE = 55,
    SLICE = 56,
    TILE = 57,
    TRANSPOSE = 58,
    GATHER = 59,
    SCATTER = 60,
    RESIZE = 61,
    CAST = 62,
    RESCALE = 63,
    CONST = 64,
    IDENTITY = 65,
    CUSTOM = 66,
    COND_IF = 67,
    WHILE_LOOP = 68,
    FFT2D = 69,
    RFFT2D = 70,
    ERF = 71,
    DIM = 72,
};

}  // namespace tosa
