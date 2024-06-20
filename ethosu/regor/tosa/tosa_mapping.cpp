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

#include "tosa_mapping.hpp"

#include "include/graphapi_tosa_types.hpp"
#include "tosa/tosa_schema_generated.hpp"

#include <stdexcept>
#include <type_traits>

namespace regor
{


static constexpr std::pair<tosaFb::DType, GraphApi::GraphDataType> s_tensorTypeToDataType[] = {
    // clang-format off
    {tosaFb::DType::BOOL,          GraphApi::GraphDataType::Bool8},
    {tosaFb::DType::UINT8,         GraphApi::GraphDataType::UInt8},
    {tosaFb::DType::INT4,          GraphApi::GraphDataType::Int4Packed8},
    {tosaFb::DType::INT8,          GraphApi::GraphDataType::Int8},
    {tosaFb::DType::INT16,         GraphApi::GraphDataType::Int16},
    {tosaFb::DType::INT32,         GraphApi::GraphDataType::Int32},
    {tosaFb::DType::INT48,         GraphApi::GraphDataType::Int48},
    {tosaFb::DType::FP32,          GraphApi::GraphDataType::Float32},
    {tosaFb::DType::UINT16,        GraphApi::GraphDataType::UInt16},
    {tosaFb::DType::FP16,          GraphApi::GraphDataType::Float16},
    {tosaFb::DType::BF16,          GraphApi::GraphDataType::BFloat16},
    // clang-format on
};

static constexpr std::pair<tosaFb::Op, tosa::Op> s_FBOpToOp[] = {
    // clang-format off
    {tosaFb::Op::UNKNOWN, tosa::Op::UNKNOWN},
    {tosaFb::Op::ARGMAX, tosa::Op::ARGMAX},
    {tosaFb::Op::AVG_POOL2D, tosa::Op::AVG_POOL2D},
    {tosaFb::Op::CONV2D, tosa::Op::CONV2D},
    {tosaFb::Op::CONV3D, tosa::Op::CONV3D},
    {tosaFb::Op::DEPTHWISE_CONV2D, tosa::Op::DEPTHWISE_CONV2D},
    {tosaFb::Op::FULLY_CONNECTED, tosa::Op::FULLY_CONNECTED},
    {tosaFb::Op::MATMUL, tosa::Op::MATMUL},
    {tosaFb::Op::MAX_POOL2D, tosa::Op::MAX_POOL2D},
    {tosaFb::Op::TRANSPOSE_CONV2D, tosa::Op::TRANSPOSE_CONV2D},
    {tosaFb::Op::CLAMP, tosa::Op::CLAMP},
    {tosaFb::Op::RESERVED, tosa::Op::RESERVED},
    {tosaFb::Op::SIGMOID, tosa::Op::SIGMOID},
    {tosaFb::Op::TANH, tosa::Op::TANH},
    {tosaFb::Op::ADD, tosa::Op::ADD},
    {tosaFb::Op::ARITHMETIC_RIGHT_SHIFT, tosa::Op::ARITHMETIC_RIGHT_SHIFT},
    {tosaFb::Op::BITWISE_AND, tosa::Op::BITWISE_AND},
    {tosaFb::Op::BITWISE_OR, tosa::Op::BITWISE_OR},
    {tosaFb::Op::BITWISE_XOR, tosa::Op::BITWISE_XOR},
    {tosaFb::Op::INTDIV, tosa::Op::INTDIV},
    {tosaFb::Op::LOGICAL_AND, tosa::Op::LOGICAL_AND},
    {tosaFb::Op::LOGICAL_LEFT_SHIFT, tosa::Op::LOGICAL_LEFT_SHIFT},
    {tosaFb::Op::LOGICAL_RIGHT_SHIFT, tosa::Op::LOGICAL_RIGHT_SHIFT},
    {tosaFb::Op::LOGICAL_OR, tosa::Op::LOGICAL_OR},
    {tosaFb::Op::LOGICAL_XOR, tosa::Op::LOGICAL_XOR},
    {tosaFb::Op::MAXIMUM, tosa::Op::MAXIMUM},
    {tosaFb::Op::MINIMUM, tosa::Op::MINIMUM},
    {tosaFb::Op::MUL, tosa::Op::MUL},
    {tosaFb::Op::POW, tosa::Op::POW},
    {tosaFb::Op::SUB, tosa::Op::SUB},
    {tosaFb::Op::TABLE, tosa::Op::TABLE},
    {tosaFb::Op::ABS, tosa::Op::ABS},
    {tosaFb::Op::BITWISE_NOT, tosa::Op::BITWISE_NOT},
    {tosaFb::Op::CEIL, tosa::Op::CEIL},
    {tosaFb::Op::CLZ, tosa::Op::CLZ},
    {tosaFb::Op::EXP, tosa::Op::EXP},
    {tosaFb::Op::FLOOR, tosa::Op::FLOOR},
    {tosaFb::Op::LOG, tosa::Op::LOG},
    {tosaFb::Op::LOGICAL_NOT, tosa::Op::LOGICAL_NOT},
    {tosaFb::Op::NEGATE, tosa::Op::NEGATE},
    {tosaFb::Op::RECIPROCAL, tosa::Op::RECIPROCAL},
    {tosaFb::Op::RSQRT, tosa::Op::RSQRT},
    {tosaFb::Op::SELECT, tosa::Op::SELECT},
    {tosaFb::Op::EQUAL, tosa::Op::EQUAL},
    {tosaFb::Op::GREATER, tosa::Op::GREATER},
    {tosaFb::Op::GREATER_EQUAL, tosa::Op::GREATER_EQUAL},
    {tosaFb::Op::REDUCE_ANY, tosa::Op::REDUCE_ANY},
    {tosaFb::Op::REDUCE_ALL, tosa::Op::REDUCE_ALL},
    {tosaFb::Op::REDUCE_MAX, tosa::Op::REDUCE_MAX},
    {tosaFb::Op::REDUCE_MIN, tosa::Op::REDUCE_MIN},
    {tosaFb::Op::REDUCE_PRODUCT, tosa::Op::REDUCE_PRODUCT},
    {tosaFb::Op::REDUCE_SUM, tosa::Op::REDUCE_SUM},
    {tosaFb::Op::CONCAT, tosa::Op::CONCAT},
    {tosaFb::Op::PAD, tosa::Op::PAD},
    {tosaFb::Op::RESHAPE, tosa::Op::RESHAPE},
    {tosaFb::Op::REVERSE, tosa::Op::REVERSE},
    {tosaFb::Op::SLICE, tosa::Op::SLICE},
    {tosaFb::Op::TILE, tosa::Op::TILE},
    {tosaFb::Op::TRANSPOSE, tosa::Op::TRANSPOSE},
    {tosaFb::Op::GATHER, tosa::Op::GATHER},
    {tosaFb::Op::SCATTER, tosa::Op::SCATTER},
    {tosaFb::Op::RESIZE, tosa::Op::RESIZE},
    {tosaFb::Op::CAST, tosa::Op::CAST},
    {tosaFb::Op::RESCALE, tosa::Op::RESCALE},
    {tosaFb::Op::CONST, tosa::Op::CONST},
    {tosaFb::Op::IDENTITY, tosa::Op::IDENTITY},
    {tosaFb::Op::CUSTOM, tosa::Op::CUSTOM},
    {tosaFb::Op::COND_IF, tosa::Op::COND_IF},
    {tosaFb::Op::WHILE_LOOP, tosa::Op::WHILE_LOOP},
    {tosaFb::Op::FFT2D, tosa::Op::FFT2D},
    {tosaFb::Op::RFFT2D, tosa::Op::RFFT2D},
    {tosaFb::Op::ERF, tosa::Op::ERF},
    {tosaFb::Op::DIM, tosa::Op::DIM},
    // clang-format on
};

static constexpr std::pair<tosaFb::ResizeMode, tosa::ResizeMode> s_FBResizeModeToResizeMode[] = {
    // clang-format off
    {tosaFb::ResizeMode::UNKNOWN, tosa::ResizeMode::UNKNOWN},
    {tosaFb::ResizeMode::NEAREST, tosa::ResizeMode::NEAREST},
    {tosaFb::ResizeMode::BILINEAR, tosa::ResizeMode::BILINEAR},
    // clang-format on
};

template<typename A, typename B, size_t SIZE>
constexpr bool is_sorted(const std::pair<A, B> (&list)[SIZE])
{
    A v = list[0].first;
    for ( size_t i = 1; i < SIZE; i++ )
    {
        if ( list[i].first < v ) return false;
        v = list[i].first;
    }
    return true;
}

static_assert(is_sorted(s_tensorTypeToDataType), "TOSA mapping must be sorted");
static_assert(is_sorted(s_FBOpToOp), "TOSA mapping must be sorted");
static_assert(is_sorted(s_FBResizeModeToResizeMode), "TOSA mapping must be sorted");

template<typename KEY, typename VALUE, size_t SZ>
constexpr VALUE Lookup(const std::pair<KEY, VALUE> (&arr)[SZ], KEY type)
{
    auto pos = std::equal_range(std::begin(arr), std::end(arr), std::pair<KEY, VALUE>(type, {}),
        [](const auto &a, const auto &b) { return a.first < b.first; });
    assert(pos.first != std::end(arr));
    return pos.first->second;
}

GraphApi::GraphDataType TosaMapping::TensorTypeToDataType(tosaFb::DType type)
{
    return Lookup(s_tensorTypeToDataType, type);
}

tosa::Op TosaMapping::FBOpToOp(tosaFb::Op op)
{
    return Lookup(s_FBOpToOp, op);
}

tosa::ResizeMode TosaMapping::FBResizeModeToResizeMode(tosaFb::ResizeMode mode)
{
    return Lookup(s_FBResizeModeToResizeMode, mode);
}

}  // namespace regor
