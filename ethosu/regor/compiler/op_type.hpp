//
// SPDX-FileCopyrightText: Copyright 2021, 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include <cstdint>
#include <string>

namespace regor
{

enum class OpType : uint16_t
{
    None = 0,
    // TOSA equivalent ops
    ArgMax,
    AvgPool,
    Conv2D,
    Conv3D,
    DepthwiseConv2D,
    FullyConnected,
    MatMul,
    MaxPool,
    TransposeConv2D,
    Clamp,
    Sigmoid,
    Tanh,
    Add,
    Asr,
    And,
    Or,
    Xor,
    Div,
    LogicalAnd,
    SHL,
    SHR,
    LogicalOr,
    LogicalXor,
    Maximum,
    Minimum,
    Mul,
    Pow,
    Sub,
    Table,
    Abs,
    Not,
    Ceil,
    CLZ,
    Exp,
    Floor,
    LogicalNot,
    Neg,
    Reciprocal,
    Rsqrt,
    Select,
    Equal,
    Greater,
    GreaterEqual,
    ReduceAny,
    ReduceAll,
    ReduceMax,
    ReduceMin,
    ReduceProduct,
    ReduceSum,
    Concat,
    Pad,
    Reshape,
    Reverse,
    Slice,
    Tile,
    Transpose,
    Gather,
    Scatter,
    Resize,
    Cast,
    Rescale,
    Identity,
    If,
    While,

    // Regor Internal Operators
    MemoryCopy,
    Passthrough,
    LUT,
    AndNot,

    // Compatibility Operators
    AddN,
    ArgMin,
    BatchMatMul,
    BatchToSpaceND,
    BidirectionalSequenceLstm,
    BidirectionalSequenceRnn,
    BlockLSTM,
    Call,
    ConcatEmbeddings,
    Const,
    Cos,
    Cumsum,
    Custom,
    CustomNpuOp,
    Delegate,
    Densify,
    DepthToSpace,
    Dequantize,
    Elu,
    EmbeddingLookup,
    EmbeddingLookupSparse,
    ExpandDims,
    FakeQuantWithMinMaxArgs,
    Fill,
    FloorDiv,
    FloorMod,
    GatherNd,
    GatherV2,
    HardSwish,
    HashtableLookup,
    L2Norm,
    L2Pool2D,
    LRN,
    LSHProjection,
    LeakyRelu,
    Less,
    LessEqual,
    Log,
    LogSoftmax,
    Lstm,
    MatrixDiag,
    MatrixSetDiag,
    Mean,
    MirrorPad,
    NonMaxSuppressionV4,
    NonMaxSuppressionV5,
    NotEqual,
    OneHot,
    Pack,
    PadV2,
    Placeholder,
    Prelu,
    Quantize,
    QuantizedAvgPool,
    QuantizedConv2D,
    QuantizedMatMul,
    QuantizedMaxPool,
    QuantizedReshape,
    Range,
    Rank,
    Relu,
    Relu6,
    ReluN1To1,
    ReluN,
    ResizeBilinear,
    ResizeNearestNeighbor,
    ReverseSequence,
    ReverseV2,
    Rnn,
    Round,
    ScatterNd,
    SegmentSum,
    SelectV2,
    Shape,
    SignBit,
    Sin,
    SkipGram,
    Softmax,
    SpaceToBatchND,
    SpaceToDepth,
    SparseToDense,
    Split,
    SplitV,
    Sqrt,
    Square,
    SquaredDifference,
    Squeeze,
    StridedSlice,
    SubgraphInput,
    Svdf,
    TopKV2,
    UnidirectionalSequenceLstm,
    UnidirectionalSequenceRnn,
    Unique,
    Unpack,
    Where,
    ZerosLike,
    LookupTable,
    ENUM_END
};

inline std::string OpTypeToString(const OpType type)
{
    return EnumToString<OpType>(type);
}

constexpr inline bool IsUnaryElementwise(OpType opType)
{
    return opType == OpType::Abs || opType == OpType::LeakyRelu || opType == OpType::CLZ ||
           opType == OpType::LogicalNot || opType == OpType::Not || opType == OpType::Neg;
}

constexpr inline bool IsBinaryElementwise(OpType opType)
{
    return opType == OpType::Add || opType == OpType::Sub || opType == OpType::Mul || opType == OpType::Minimum ||
           opType == OpType::Maximum || opType == OpType::SHL || opType == OpType::SHR || opType == OpType::Div ||
           opType == OpType::LogicalAnd || opType == OpType::LogicalOr || opType == OpType::LogicalXor || opType == OpType::Xor ||
           opType == OpType::And || opType == OpType::Or || opType == OpType::Asr || opType == OpType::Equal ||
           opType == OpType::Greater || opType == OpType::GreaterEqual || opType == OpType::NotEqual || opType == OpType::AndNot;
}

constexpr inline bool IsElementwise(OpType opType)
{
    return IsUnaryElementwise(opType) || IsBinaryElementwise(opType);
}

constexpr inline bool DecomposeAsElementwise(OpType opType)
{
    return IsElementwise(opType) || opType == OpType::Rescale || opType == OpType::LUT || opType == OpType::Table ||
           opType == OpType::Clamp || opType == OpType::Cast;
}

constexpr inline bool IsDepthwise(OpType opType)
{
    return opType == OpType::DepthwiseConv2D;
}

constexpr inline bool IsConvolution(OpType opType)
{
    return opType == OpType::Conv2D || opType == OpType::DepthwiseConv2D || opType == OpType::TransposeConv2D;
}

constexpr inline bool IsPooling(OpType opType)
{
    return opType == OpType::MaxPool || opType == OpType::AvgPool || opType == OpType::QuantizedAvgPool ||
           opType == OpType::QuantizedMaxPool || opType == OpType::ReduceSum || opType == OpType::ReduceMin ||
           opType == OpType::ReduceMax || opType == OpType::ReduceAny || opType == OpType::ReduceAll || opType == OpType::ArgMax;
}

constexpr inline bool IsVectorProduct(OpType opType)
{
    return opType == OpType::FullyConnected || opType == OpType::BidirectionalSequenceLstm || opType == OpType::BidirectionalSequenceRnn ||
           opType == OpType::BlockLSTM || opType == OpType::Lstm || opType == OpType::MatMul || opType == OpType::Rnn ||
           opType == OpType::UnidirectionalSequenceLstm || opType == OpType::UnidirectionalSequenceRnn;
}

constexpr inline bool IsActivation(OpType opType)
{
    return opType == OpType::Relu || opType == OpType::Relu6 || opType == OpType::ReluN || opType == OpType::ReluN1To1 ||
           opType == OpType::Prelu || opType == OpType::Clamp || opType == OpType::Sigmoid || opType == OpType::Tanh || opType == OpType::LUT;
}

constexpr inline bool IsConcatenation(OpType opType)
{
    return opType == OpType::Concat || opType == OpType::ConcatEmbeddings;
}

constexpr inline bool IsVariadic(OpType opType)
{
    return IsConcatenation(opType) || opType == OpType::Pack || opType == OpType::Maximum || opType == OpType::Minimum ||
           opType == OpType::AddN || opType == OpType::Custom || opType == OpType::CustomNpuOp;
}

constexpr inline bool IsReshape(OpType opType)
{
    // The Reshape like operations: Reshape, Squeeze, and ExpandDims
    return opType == OpType::Reshape || opType == OpType::QuantizedReshape || opType == OpType::Squeeze ||
           opType == OpType::ExpandDims || opType == OpType::Identity;
}

}  // namespace regor
