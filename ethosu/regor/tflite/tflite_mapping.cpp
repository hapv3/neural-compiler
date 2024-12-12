//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "tflite_mapping.hpp"

#include "common/data_type.hpp"
#include "compiler/op_type.hpp"
#include "flatbuffer_utils.hpp"
#include "tflite_schema_generated.hpp"

#include <map>
#include <set>

namespace regor
{

const std::map<tflite::TensorType, DataType> TfLiteMapping::_tensorTypeToDataType = {
    // clang-format off
    {tflite::TensorType::FLOAT32,       DataType::Float32},
    {tflite::TensorType::FLOAT16,       DataType::Float16},
    {tflite::TensorType::INT32,         DataType::Int32},
    {tflite::TensorType::UINT8,         DataType::UInt8},
    {tflite::TensorType::INT64,         DataType::Int64},
    {tflite::TensorType::STRING,        DataType::String},
    {tflite::TensorType::BOOL,          DataType::Bool8},
    {tflite::TensorType::INT16,         DataType::Int16},
    {tflite::TensorType::COMPLEX64,     DataType::Complex64},
    {tflite::TensorType::INT8,          DataType::Int8},
    {tflite::TensorType::FLOAT64,       DataType::Float64},
    {tflite::TensorType::COMPLEX128,    DataType::Complex128},
    {tflite::TensorType::UINT64,        DataType::UInt64},
    {tflite::TensorType::RESOURCE,      DataType::Resource},
    {tflite::TensorType::VARIANT,       DataType::Variant},
    {tflite::TensorType::UINT32,        DataType::UInt32},
    {tflite::TensorType::UINT16,        DataType::UInt16}
    // clang-format on
};

const std::map<DataType, tflite::TensorType> TfLiteMapping::_dataTypeToTensorType = InvertMap<tflite::TensorType, DataType>(_tensorTypeToDataType);

const std::map<tflite::ActivationFunctionType, OpType> TfLiteMapping::_activationFunctionToOpType = {
    // clang-format off
    {tflite::ActivationFunctionType::NONE,          OpType::None},
    {tflite::ActivationFunctionType::RELU,          OpType::Relu},
    {tflite::ActivationFunctionType::RELU_N1_TO_1,  OpType::ReluN1To1},
    {tflite::ActivationFunctionType::RELU6,         OpType::Relu6},
    {tflite::ActivationFunctionType::TANH,          OpType::Tanh},
    {tflite::ActivationFunctionType::SIGN_BIT,      OpType::SignBit}
    // clang-format on
};

const std::map<OpType, tflite::ActivationFunctionType> TfLiteMapping::_opTypeToActivationFunction = InvertMap<
    tflite::ActivationFunctionType, OpType>(_activationFunctionToOpType);

const std::map<tflite::BuiltinOperator, OpType> TfLiteMapping::_builtinOperatorToOpType = {
    // clang-format off
    {tflite::BuiltinOperator::ADD,                              OpType::Add},
    {tflite::BuiltinOperator::AVERAGE_POOL_2D,                  OpType::AvgPool},
    {tflite::BuiltinOperator::CONCATENATION,                    OpType::Concat},
    {tflite::BuiltinOperator::CONV_2D,                          OpType::Conv2D},
    {tflite::BuiltinOperator::DEPTHWISE_CONV_2D,                OpType::DepthwiseConv2D},
    {tflite::BuiltinOperator::DEPTH_TO_SPACE,                   OpType::DepthToSpace},
    {tflite::BuiltinOperator::DEQUANTIZE,                       OpType::Dequantize},
    {tflite::BuiltinOperator::EMBEDDING_LOOKUP,                 OpType::EmbeddingLookup},
    {tflite::BuiltinOperator::FLOOR,                            OpType::Floor},
    {tflite::BuiltinOperator::FULLY_CONNECTED,                  OpType::FullyConnected},
    {tflite::BuiltinOperator::HASHTABLE_LOOKUP,                 OpType::HashtableLookup},
    {tflite::BuiltinOperator::L2_NORMALIZATION,                 OpType::L2Norm},
    {tflite::BuiltinOperator::L2_POOL_2D,                       OpType::L2Pool2D},
    {tflite::BuiltinOperator::LOCAL_RESPONSE_NORMALIZATION,     OpType::LRN},
    {tflite::BuiltinOperator::LOGISTIC,                         OpType::Sigmoid},
    {tflite::BuiltinOperator::LSH_PROJECTION,                   OpType::LSHProjection},
    {tflite::BuiltinOperator::LSTM,                             OpType::Lstm},
    {tflite::BuiltinOperator::MAX_POOL_2D,                      OpType::MaxPool},
    {tflite::BuiltinOperator::MUL,                              OpType::Mul},
    {tflite::BuiltinOperator::RELU,                             OpType::Relu},
    {tflite::BuiltinOperator::RELU_N1_TO_1,                     OpType::ReluN1To1},
    {tflite::BuiltinOperator::RELU6,                            OpType::Relu6},
    {tflite::BuiltinOperator::RESHAPE,                          OpType::Reshape},
    {tflite::BuiltinOperator::RESIZE_BILINEAR,                  OpType::ResizeBilinear},
    {tflite::BuiltinOperator::RNN,                              OpType::Rnn},
    {tflite::BuiltinOperator::SOFTMAX,                          OpType::Softmax},
    {tflite::BuiltinOperator::SPACE_TO_DEPTH,                   OpType::SpaceToDepth},
    {tflite::BuiltinOperator::SVDF,                             OpType::Svdf},
    {tflite::BuiltinOperator::TANH,                             OpType::Tanh},
    {tflite::BuiltinOperator::CONCAT_EMBEDDINGS,                OpType::ConcatEmbeddings},
    {tflite::BuiltinOperator::SKIP_GRAM,                        OpType::SkipGram},
    {tflite::BuiltinOperator::CALL,                             OpType::Call},
    {tflite::BuiltinOperator::CUSTOM,                           OpType::Custom},
    {tflite::BuiltinOperator::EMBEDDING_LOOKUP_SPARSE,          OpType::EmbeddingLookupSparse},
    {tflite::BuiltinOperator::PAD,                              OpType::Pad},
    {tflite::BuiltinOperator::UNIDIRECTIONAL_SEQUENCE_RNN,      OpType::UnidirectionalSequenceRnn},
    {tflite::BuiltinOperator::GATHER,                           OpType::GatherV2},
    {tflite::BuiltinOperator::BATCH_TO_SPACE_ND,                OpType::BatchToSpaceND},
    {tflite::BuiltinOperator::SPACE_TO_BATCH_ND,                OpType::SpaceToBatchND},
    {tflite::BuiltinOperator::TRANSPOSE,                        OpType::Transpose},
    {tflite::BuiltinOperator::MEAN,                             OpType::Mean},
    {tflite::BuiltinOperator::SUB,                              OpType::Sub},
    {tflite::BuiltinOperator::DIV,                              OpType::Div},
    {tflite::BuiltinOperator::SQUEEZE,                          OpType::Squeeze},
    {tflite::BuiltinOperator::UNIDIRECTIONAL_SEQUENCE_LSTM,     OpType::UnidirectionalSequenceLstm},
    {tflite::BuiltinOperator::STRIDED_SLICE,                    OpType::StridedSlice},
    {tflite::BuiltinOperator::BIDIRECTIONAL_SEQUENCE_RNN,       OpType::BidirectionalSequenceRnn},
    {tflite::BuiltinOperator::EXP,                              OpType::Exp},
    {tflite::BuiltinOperator::TOPK_V2,                          OpType::TopKV2},
    {tflite::BuiltinOperator::SPLIT,                            OpType::Split},
    {tflite::BuiltinOperator::LOG_SOFTMAX,                      OpType::LogSoftmax},
    {tflite::BuiltinOperator::DELEGATE,                         OpType::Delegate},
    {tflite::BuiltinOperator::BIDIRECTIONAL_SEQUENCE_LSTM,      OpType::BidirectionalSequenceLstm},
    {tflite::BuiltinOperator::CAST,                             OpType::Cast},
    {tflite::BuiltinOperator::PRELU,                            OpType::Prelu},
    {tflite::BuiltinOperator::MAXIMUM,                          OpType::Maximum},
    {tflite::BuiltinOperator::ARGMAX,                           OpType::ArgMax},
    {tflite::BuiltinOperator::MINIMUM,                          OpType::Minimum},
    {tflite::BuiltinOperator::LESS,                             OpType::Less},
    {tflite::BuiltinOperator::NEG,                              OpType::Neg},
    {tflite::BuiltinOperator::PADV2,                            OpType::PadV2},
    {tflite::BuiltinOperator::GREATER,                          OpType::Greater},
    {tflite::BuiltinOperator::GREATER_EQUAL,                    OpType::GreaterEqual},
    {tflite::BuiltinOperator::LESS_EQUAL,                       OpType::LessEqual},
    {tflite::BuiltinOperator::SELECT,                           OpType::Select},
    {tflite::BuiltinOperator::SLICE,                            OpType::Slice},
    {tflite::BuiltinOperator::SIN,                              OpType::Sin},
    {tflite::BuiltinOperator::TRANSPOSE_CONV,                   OpType::TransposeConv2D},
    {tflite::BuiltinOperator::SPARSE_TO_DENSE,                  OpType::SparseToDense},
    {tflite::BuiltinOperator::TILE,                             OpType::Tile},
    {tflite::BuiltinOperator::EXPAND_DIMS,                      OpType::ExpandDims},
    {tflite::BuiltinOperator::EQUAL,                            OpType::Equal},
    {tflite::BuiltinOperator::NOT_EQUAL,                        OpType::NotEqual},
    {tflite::BuiltinOperator::LOG,                              OpType::Log},
    {tflite::BuiltinOperator::SUM,                              OpType::ReduceSum},
    {tflite::BuiltinOperator::SQRT,                             OpType::Sqrt},
    {tflite::BuiltinOperator::RSQRT,                            OpType::Rsqrt},
    {tflite::BuiltinOperator::SHAPE,                            OpType::Shape},
    {tflite::BuiltinOperator::POW,                              OpType::Pow},
    {tflite::BuiltinOperator::ARG_MIN,                          OpType::ArgMin},
    {tflite::BuiltinOperator::FAKE_QUANT,                       OpType::FakeQuantWithMinMaxArgs},
    {tflite::BuiltinOperator::REDUCE_PROD,                      OpType::ReduceProduct},
    {tflite::BuiltinOperator::REDUCE_MAX,                       OpType::ReduceMax},
    {tflite::BuiltinOperator::PACK,                             OpType::Pack},
    {tflite::BuiltinOperator::LOGICAL_OR,                       OpType::LogicalOr},
    {tflite::BuiltinOperator::ONE_HOT,                          OpType::OneHot},
    {tflite::BuiltinOperator::LOGICAL_AND,                      OpType::LogicalAnd},
    {tflite::BuiltinOperator::LOGICAL_NOT,                      OpType::LogicalNot},
    {tflite::BuiltinOperator::UNPACK,                           OpType::Unpack},
    {tflite::BuiltinOperator::REDUCE_MIN,                       OpType::ReduceMin},
    {tflite::BuiltinOperator::FLOOR_DIV,                        OpType::FloorDiv},
    {tflite::BuiltinOperator::REDUCE_ANY,                       OpType::ReduceAny},
    {tflite::BuiltinOperator::SQUARE,                           OpType::Square},
    {tflite::BuiltinOperator::ZEROS_LIKE,                       OpType::ZerosLike},
    {tflite::BuiltinOperator::FILL,                             OpType::Fill},
    {tflite::BuiltinOperator::FLOOR_MOD,                        OpType::FloorMod},
    {tflite::BuiltinOperator::RANGE,                            OpType::Range},
    {tflite::BuiltinOperator::RESIZE_NEAREST_NEIGHBOR,          OpType::ResizeNearestNeighbor},
    {tflite::BuiltinOperator::LEAKY_RELU,                       OpType::LeakyRelu},
    {tflite::BuiltinOperator::SQUARED_DIFFERENCE,               OpType::SquaredDifference},
    {tflite::BuiltinOperator::MIRROR_PAD,                       OpType::MirrorPad},
    {tflite::BuiltinOperator::ABS,                              OpType::Abs},
    {tflite::BuiltinOperator::SPLIT_V,                          OpType::SplitV},
    {tflite::BuiltinOperator::UNIQUE,                           OpType::Unique},
    {tflite::BuiltinOperator::CEIL,                             OpType::Ceil},
    {tflite::BuiltinOperator::REVERSE_V2,                       OpType::ReverseV2},
    {tflite::BuiltinOperator::ADD_N,                            OpType::AddN},
    {tflite::BuiltinOperator::GATHER_ND,                        OpType::GatherNd},
    {tflite::BuiltinOperator::COS,                              OpType::Cos},
    {tflite::BuiltinOperator::WHERE,                            OpType::Where},
    {tflite::BuiltinOperator::RANK,                             OpType::Rank},
    {tflite::BuiltinOperator::ELU,                              OpType::Elu},
    {tflite::BuiltinOperator::REVERSE_SEQUENCE,                 OpType::ReverseSequence},
    {tflite::BuiltinOperator::MATRIX_DIAG,                      OpType::MatrixDiag},
    {tflite::BuiltinOperator::QUANTIZE,                         OpType::Quantize},
    {tflite::BuiltinOperator::MATRIX_SET_DIAG,                  OpType::MatrixSetDiag},
    {tflite::BuiltinOperator::ROUND,                            OpType::Round},
    {tflite::BuiltinOperator::HARD_SWISH,                       OpType::HardSwish},
    {tflite::BuiltinOperator::IF,                               OpType::If},
    {tflite::BuiltinOperator::WHILE,                            OpType::While},
    {tflite::BuiltinOperator::NON_MAX_SUPPRESSION_V4,           OpType::NonMaxSuppressionV4},
    {tflite::BuiltinOperator::NON_MAX_SUPPRESSION_V5,           OpType::NonMaxSuppressionV5},
    {tflite::BuiltinOperator::SCATTER_ND,                       OpType::ScatterNd},
    {tflite::BuiltinOperator::SELECT_V2,                        OpType::SelectV2},
    {tflite::BuiltinOperator::DENSIFY,                          OpType::Densify},
    {tflite::BuiltinOperator::SEGMENT_SUM,                      OpType::SegmentSum},
    {tflite::BuiltinOperator::BATCH_MATMUL,                     OpType::BatchMatMul},
    {tflite::BuiltinOperator::CUMSUM,                           OpType::Cumsum},
    {tflite::BuiltinOperator::REDUCE_ALL,                       OpType::ReduceAll}
    // clang-format on
};

const std::map<OpType, tflite::BuiltinOperator>
    TfLiteMapping::_opTypeToBuiltinOperator = InvertMap<tflite::BuiltinOperator, OpType>(_builtinOperatorToOpType);

const std::map<tflite::BuiltinOperator, tflite::BuiltinOptions> TfLiteMapping::_builtinOperatorToBuiltinOptions = {
    // clang-format off
    {tflite::BuiltinOperator::ADD,                              tflite::BuiltinOptions::AddOptions},
    {tflite::BuiltinOperator::AVERAGE_POOL_2D,                  tflite::BuiltinOptions::Pool2DOptions},
    {tflite::BuiltinOperator::CONCATENATION,                    tflite::BuiltinOptions::ConcatenationOptions},
    {tflite::BuiltinOperator::CONV_2D,                          tflite::BuiltinOptions::Conv2DOptions},
    {tflite::BuiltinOperator::DEPTHWISE_CONV_2D,                tflite::BuiltinOptions::DepthwiseConv2DOptions},
    {tflite::BuiltinOperator::DEPTH_TO_SPACE,                   tflite::BuiltinOptions::DepthToSpaceOptions},
    {tflite::BuiltinOperator::DEQUANTIZE,                       tflite::BuiltinOptions::DequantizeOptions},
    {tflite::BuiltinOperator::EMBEDDING_LOOKUP,                 tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::FLOOR,                            tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::FULLY_CONNECTED,                  tflite::BuiltinOptions::FullyConnectedOptions},
    {tflite::BuiltinOperator::HASHTABLE_LOOKUP,                 tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::L2_NORMALIZATION,                 tflite::BuiltinOptions::L2NormOptions},
    {tflite::BuiltinOperator::L2_POOL_2D,                       tflite::BuiltinOptions::Pool2DOptions},
    {tflite::BuiltinOperator::LOCAL_RESPONSE_NORMALIZATION,     tflite::BuiltinOptions::LocalResponseNormalizationOptions},
    {tflite::BuiltinOperator::LOGISTIC,                         tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::LSH_PROJECTION,                   tflite::BuiltinOptions::LSHProjectionOptions},
    {tflite::BuiltinOperator::LSTM,                             tflite::BuiltinOptions::LSTMOptions},
    {tflite::BuiltinOperator::MAX_POOL_2D,                      tflite::BuiltinOptions::Pool2DOptions},
    {tflite::BuiltinOperator::MUL,                              tflite::BuiltinOptions::MulOptions},
    {tflite::BuiltinOperator::RELU,                             tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::RELU_N1_TO_1,                     tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::RELU6,                            tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::RESHAPE,                          tflite::BuiltinOptions::ReshapeOptions},
    {tflite::BuiltinOperator::RESIZE_BILINEAR,                  tflite::BuiltinOptions::ResizeBilinearOptions},
    {tflite::BuiltinOperator::RNN,                              tflite::BuiltinOptions::RNNOptions},
    {tflite::BuiltinOperator::SOFTMAX,                          tflite::BuiltinOptions::SoftmaxOptions},
    {tflite::BuiltinOperator::SPACE_TO_DEPTH,                   tflite::BuiltinOptions::SpaceToDepthOptions},
    {tflite::BuiltinOperator::SVDF,                             tflite::BuiltinOptions::SVDFOptions},
    {tflite::BuiltinOperator::TANH,                             tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::CONCAT_EMBEDDINGS,                tflite::BuiltinOptions::ConcatEmbeddingsOptions},
    {tflite::BuiltinOperator::SKIP_GRAM,                        tflite::BuiltinOptions::SkipGramOptions},
    {tflite::BuiltinOperator::CALL,                             tflite::BuiltinOptions::CallOptions},
    {tflite::BuiltinOperator::CUSTOM,                           tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::EMBEDDING_LOOKUP_SPARSE,          tflite::BuiltinOptions::EmbeddingLookupSparseOptions},
    {tflite::BuiltinOperator::PAD,                              tflite::BuiltinOptions::PadOptions},
    {tflite::BuiltinOperator::UNIDIRECTIONAL_SEQUENCE_RNN,      tflite::BuiltinOptions::SequenceRNNOptions},
    {tflite::BuiltinOperator::GATHER,                           tflite::BuiltinOptions::GatherOptions},
    {tflite::BuiltinOperator::BATCH_TO_SPACE_ND,                tflite::BuiltinOptions::BatchToSpaceNDOptions},
    {tflite::BuiltinOperator::SPACE_TO_BATCH_ND,                tflite::BuiltinOptions::SpaceToBatchNDOptions},
    {tflite::BuiltinOperator::TRANSPOSE,                        tflite::BuiltinOptions::TransposeOptions},
    {tflite::BuiltinOperator::MEAN,                             tflite::BuiltinOptions::ReducerOptions},
    {tflite::BuiltinOperator::SUB,                              tflite::BuiltinOptions::SubOptions},
    {tflite::BuiltinOperator::DIV,                              tflite::BuiltinOptions::DivOptions},
    {tflite::BuiltinOperator::SQUEEZE,                          tflite::BuiltinOptions::SqueezeOptions},
    {tflite::BuiltinOperator::UNIDIRECTIONAL_SEQUENCE_LSTM,     tflite::BuiltinOptions::UnidirectionalSequenceLSTMOptions},
    {tflite::BuiltinOperator::STRIDED_SLICE,                    tflite::BuiltinOptions::StridedSliceOptions},
    {tflite::BuiltinOperator::BIDIRECTIONAL_SEQUENCE_RNN,       tflite::BuiltinOptions::BidirectionalSequenceRNNOptions},
    {tflite::BuiltinOperator::EXP,                              tflite::BuiltinOptions::ExpOptions},
    {tflite::BuiltinOperator::TOPK_V2,                          tflite::BuiltinOptions::TopKV2Options},
    {tflite::BuiltinOperator::SPLIT,                            tflite::BuiltinOptions::SplitOptions},
    {tflite::BuiltinOperator::LOG_SOFTMAX,                      tflite::BuiltinOptions::LogSoftmaxOptions},
    {tflite::BuiltinOperator::DELEGATE,                         tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::BIDIRECTIONAL_SEQUENCE_LSTM,      tflite::BuiltinOptions::BidirectionalSequenceLSTMOptions},
    {tflite::BuiltinOperator::CAST,                             tflite::BuiltinOptions::CastOptions},
    {tflite::BuiltinOperator::PRELU,                            tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::MAXIMUM,                          tflite::BuiltinOptions::MaximumMinimumOptions},
    {tflite::BuiltinOperator::ARGMAX,                           tflite::BuiltinOptions::ArgMaxOptions},
    {tflite::BuiltinOperator::MINIMUM,                          tflite::BuiltinOptions::MaximumMinimumOptions},
    {tflite::BuiltinOperator::LESS,                             tflite::BuiltinOptions::LessOptions},
    {tflite::BuiltinOperator::NEG,                              tflite::BuiltinOptions::NegOptions},
    {tflite::BuiltinOperator::PADV2,                            tflite::BuiltinOptions::PadV2Options},
    {tflite::BuiltinOperator::GREATER,                          tflite::BuiltinOptions::GreaterOptions},
    {tflite::BuiltinOperator::GREATER_EQUAL,                    tflite::BuiltinOptions::GreaterEqualOptions},
    {tflite::BuiltinOperator::LESS_EQUAL,                       tflite::BuiltinOptions::LessEqualOptions},
    {tflite::BuiltinOperator::SELECT,                           tflite::BuiltinOptions::SelectOptions},
    {tflite::BuiltinOperator::SLICE,                            tflite::BuiltinOptions::SliceOptions},
    {tflite::BuiltinOperator::SIN,                              tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::TRANSPOSE_CONV,                   tflite::BuiltinOptions::TransposeConvOptions},
    {tflite::BuiltinOperator::SPARSE_TO_DENSE,                  tflite::BuiltinOptions::SparseToDenseOptions},
    {tflite::BuiltinOperator::TILE,                             tflite::BuiltinOptions::TileOptions},
    {tflite::BuiltinOperator::EXPAND_DIMS,                      tflite::BuiltinOptions::ExpandDimsOptions},
    {tflite::BuiltinOperator::EQUAL,                            tflite::BuiltinOptions::EqualOptions},
    {tflite::BuiltinOperator::NOT_EQUAL,                        tflite::BuiltinOptions::NotEqualOptions},
    {tflite::BuiltinOperator::LOG,                              tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::SUM,                              tflite::BuiltinOptions::ReducerOptions},
    {tflite::BuiltinOperator::SQRT,                             tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::RSQRT,                            tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::SHAPE,                            tflite::BuiltinOptions::ShapeOptions},
    {tflite::BuiltinOperator::POW,                              tflite::BuiltinOptions::PowOptions},
    {tflite::BuiltinOperator::ARG_MIN,                          tflite::BuiltinOptions::ArgMinOptions},
    {tflite::BuiltinOperator::FAKE_QUANT,                       tflite::BuiltinOptions::FakeQuantOptions},
    {tflite::BuiltinOperator::REDUCE_PROD,                      tflite::BuiltinOptions::ReducerOptions},
    {tflite::BuiltinOperator::REDUCE_MAX,                       tflite::BuiltinOptions::ReducerOptions},
    {tflite::BuiltinOperator::PACK,                             tflite::BuiltinOptions::PackOptions},
    {tflite::BuiltinOperator::LOGICAL_OR,                       tflite::BuiltinOptions::LogicalOrOptions},
    {tflite::BuiltinOperator::ONE_HOT,                          tflite::BuiltinOptions::OneHotOptions},
    {tflite::BuiltinOperator::LOGICAL_AND,                      tflite::BuiltinOptions::LogicalAndOptions},
    {tflite::BuiltinOperator::LOGICAL_NOT,                      tflite::BuiltinOptions::LogicalNotOptions},
    {tflite::BuiltinOperator::UNPACK,                           tflite::BuiltinOptions::UnpackOptions},
    {tflite::BuiltinOperator::REDUCE_MIN,                       tflite::BuiltinOptions::ReducerOptions},
    {tflite::BuiltinOperator::FLOOR_DIV,                        tflite::BuiltinOptions::FloorDivOptions},
    {tflite::BuiltinOperator::REDUCE_ANY,                       tflite::BuiltinOptions::ReducerOptions},
    {tflite::BuiltinOperator::SQUARE,                           tflite::BuiltinOptions::SquareOptions},
    {tflite::BuiltinOperator::ZEROS_LIKE,                       tflite::BuiltinOptions::ZerosLikeOptions},
    {tflite::BuiltinOperator::FILL,                             tflite::BuiltinOptions::FillOptions},
    {tflite::BuiltinOperator::FLOOR_MOD,                        tflite::BuiltinOptions::FloorModOptions},
    {tflite::BuiltinOperator::RANGE,                            tflite::BuiltinOptions::RangeOptions},
    {tflite::BuiltinOperator::RESIZE_NEAREST_NEIGHBOR,          tflite::BuiltinOptions::ResizeNearestNeighborOptions},
    {tflite::BuiltinOperator::LEAKY_RELU,                       tflite::BuiltinOptions::LeakyReluOptions},
    {tflite::BuiltinOperator::SQUARED_DIFFERENCE,               tflite::BuiltinOptions::SquaredDifferenceOptions},
    {tflite::BuiltinOperator::MIRROR_PAD,                       tflite::BuiltinOptions::MirrorPadOptions},
    {tflite::BuiltinOperator::ABS,                              tflite::BuiltinOptions::AbsOptions},
    {tflite::BuiltinOperator::SPLIT_V,                          tflite::BuiltinOptions::SplitVOptions},
    {tflite::BuiltinOperator::UNIQUE,                           tflite::BuiltinOptions::UniqueOptions},
    {tflite::BuiltinOperator::CEIL,                             tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::REVERSE_V2,                       tflite::BuiltinOptions::ReverseV2Options},
    {tflite::BuiltinOperator::ADD_N,                            tflite::BuiltinOptions::AddNOptions},
    {tflite::BuiltinOperator::GATHER_ND,                        tflite::BuiltinOptions::GatherNdOptions},
    {tflite::BuiltinOperator::COS,                              tflite::BuiltinOptions::CosOptions},
    {tflite::BuiltinOperator::WHERE,                            tflite::BuiltinOptions::WhereOptions},
    {tflite::BuiltinOperator::RANK,                             tflite::BuiltinOptions::RankOptions},
    {tflite::BuiltinOperator::ELU,                              tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::REVERSE_SEQUENCE,                 tflite::BuiltinOptions::ReverseSequenceOptions},
    {tflite::BuiltinOperator::MATRIX_DIAG,                      tflite::BuiltinOptions::MatrixDiagOptions},
    {tflite::BuiltinOperator::QUANTIZE,                         tflite::BuiltinOptions::QuantizeOptions},
    {tflite::BuiltinOperator::MATRIX_SET_DIAG,                  tflite::BuiltinOptions::MatrixSetDiagOptions},
    {tflite::BuiltinOperator::ROUND,                            tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::HARD_SWISH,                       tflite::BuiltinOptions::HardSwishOptions},
    {tflite::BuiltinOperator::IF,                               tflite::BuiltinOptions::IfOptions},
    {tflite::BuiltinOperator::WHILE,                            tflite::BuiltinOptions::WhileOptions},
    {tflite::BuiltinOperator::NON_MAX_SUPPRESSION_V4,           tflite::BuiltinOptions::NonMaxSuppressionV4Options},
    {tflite::BuiltinOperator::NON_MAX_SUPPRESSION_V5,           tflite::BuiltinOptions::NonMaxSuppressionV5Options},
    {tflite::BuiltinOperator::SCATTER_ND,                       tflite::BuiltinOptions::ScatterNdOptions},
    {tflite::BuiltinOperator::SELECT_V2,                        tflite::BuiltinOptions::SelectV2Options},
    {tflite::BuiltinOperator::DENSIFY,                          tflite::BuiltinOptions::DensifyOptions},
    {tflite::BuiltinOperator::SEGMENT_SUM,                      tflite::BuiltinOptions::SegmentSumOptions},
    {tflite::BuiltinOperator::BATCH_MATMUL,                     tflite::BuiltinOptions::BatchMatMulOptions},
    {tflite::BuiltinOperator::PLACEHOLDER_FOR_GREATER_OP_CODES, tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::CUMSUM,                           tflite::BuiltinOptions::CumsumOptions},
    {tflite::BuiltinOperator::CALL_ONCE,                        tflite::BuiltinOptions::CallOnceOptions},
    {tflite::BuiltinOperator::BROADCAST_TO,                     tflite::BuiltinOptions::BroadcastToOptions},
    {tflite::BuiltinOperator::RFFT2D,                           tflite::BuiltinOptions::Rfft2dOptions},
    {tflite::BuiltinOperator::CONV_3D,                          tflite::BuiltinOptions::Conv3DOptions},
    {tflite::BuiltinOperator::IMAG,                             tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::REAL,                             tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::COMPLEX_ABS,                      tflite::BuiltinOptions::NONE},
    {tflite::BuiltinOperator::HASHTABLE,                        tflite::BuiltinOptions::HashtableOptions},
    {tflite::BuiltinOperator::HASHTABLE_FIND,                   tflite::BuiltinOptions::HashtableFindOptions},
    {tflite::BuiltinOperator::HASHTABLE_IMPORT,                 tflite::BuiltinOptions::HashtableImportOptions},
    {tflite::BuiltinOperator::HASHTABLE_SIZE,                   tflite::BuiltinOptions::HashtableSizeOptions},
    {tflite::BuiltinOperator::REDUCE_ALL,                       tflite::BuiltinOptions::ReducerOptions},
    // TODO: {tflite::BuiltinOperator::CONV_3D_TRANSPOSE,       tflite::BuiltinOptions::???},
    {tflite::BuiltinOperator::VAR_HANDLE,                       tflite::BuiltinOptions::VarHandleOptions},
    {tflite::BuiltinOperator::READ_VARIABLE,                    tflite::BuiltinOptions::ReadVariableOptions},
    {tflite::BuiltinOperator::ASSIGN_VARIABLE,                  tflite::BuiltinOptions::AssignVariableOptions},
    {tflite::BuiltinOperator::BROADCAST_ARGS,                   tflite::BuiltinOptions::NONE},
    // clang-format on
};

const std::multimap<OpType, TensorUsage> TfLiteMapping::_inputTensorIndices = {
    // clang-format off
    {OpType::Abs,                               TensorUsage::IFM0},
    {OpType::Add,                               TensorUsage::IFM0},
    {OpType::Add,                               TensorUsage::IFM1},
    // AddN is variadic                         None
    {OpType::ArgMax,                            TensorUsage::IFM0},
    {OpType::ArgMax,                            TensorUsage::Params},
    {OpType::ArgMin,                            TensorUsage::IFM0},
    {OpType::ArgMin,                            TensorUsage::Params},
    {OpType::AvgPool,                           TensorUsage::IFM0},
    {OpType::BatchMatMul,                       TensorUsage::IFM0},
    {OpType::BatchMatMul,                       TensorUsage::IFM1},
    {OpType::BatchToSpaceND,                    TensorUsage::IFM0},
    {OpType::BatchToSpaceND,                    TensorUsage::Params},
    {OpType::BatchToSpaceND,                    MakeTensorUsage(TensorUsage::Params, 1)},
    {OpType::BidirectionalSequenceLstm,         TensorUsage::IFM0},
    {OpType::BidirectionalSequenceLstm,         TensorUsage::Weights},
    {OpType::BidirectionalSequenceRnn,          TensorUsage::IFM0},
    {OpType::BidirectionalSequenceRnn,          TensorUsage::Weights},
    {OpType::CLZ,                               TensorUsage::IFM0},
    // Call                                     None
    {OpType::Cast,                              TensorUsage::IFM0},
    {OpType::Ceil,                              TensorUsage::IFM0},
    // Clip                                     None
    {OpType::Clamp,                             TensorUsage::IFM0},
    // Concat ops are variadic                  None
    // Const                                    None
    {OpType::Conv2D,                            TensorUsage::IFM0},
    {OpType::Conv2D,                            TensorUsage::Weights},
    {OpType::Conv2D,                            TensorUsage::Scales},
    {OpType::TransposeConv2D,                   TensorUsage::Params},
    {OpType::TransposeConv2D,                   TensorUsage::Weights},
    {OpType::TransposeConv2D,                   TensorUsage::IFM},
    {OpType::TransposeConv2D,                   TensorUsage::Scales},
    {OpType::Cos,                               TensorUsage::IFM0},
    {OpType::Cumsum,                            TensorUsage::IFM0},
    // Custom ops are variadic
    {OpType::CustomNpuOp,                       TensorUsage::Params},                     // Register command stream
    {OpType::CustomNpuOp,                       MakeTensorUsage(TensorUsage::Params, 1)}, // Read only constants
    {OpType::CustomNpuOp,                       TensorUsage::State},                      // Feature map area
    {OpType::CustomNpuOp,                       MakeTensorUsage(TensorUsage::State, 1)},  // Staging area
    // Delegate                                 None
    {OpType::Densify,                           TensorUsage::IFM0},
    {OpType::DepthToSpace,                      TensorUsage::IFM0},
    {OpType::DepthwiseConv2D,                   TensorUsage::IFM0},
    {OpType::DepthwiseConv2D,                   TensorUsage::Weights},
    {OpType::DepthwiseConv2D,                   TensorUsage::Scales},
    {OpType::Dequantize,                        TensorUsage::IFM0},
    {OpType::Div,                               TensorUsage::IFM0},
    {OpType::Div,                               TensorUsage::IFM1},
    {OpType::Elu,                               TensorUsage::IFM0},
    // EmbeddingLookup, EmbeddingLookupSparse   None
    {OpType::Equal,                             TensorUsage::IFM0},
    {OpType::Equal,                             TensorUsage::IFM1},
    {OpType::Exp,                               TensorUsage::IFM0},
    {OpType::ExpandDims,                        TensorUsage::IFM0},
    {OpType::FakeQuantWithMinMaxArgs,           TensorUsage::IFM0},
    // Fill                                     None
    {OpType::Floor,                             TensorUsage::IFM0},
    {OpType::FloorDiv,                          TensorUsage::IFM0},
    {OpType::FloorDiv,                          TensorUsage::IFM1},
    {OpType::FloorMod,                          TensorUsage::IFM0},
    {OpType::FloorMod,                          TensorUsage::IFM1},
    {OpType::FullyConnected,                    TensorUsage::IFM0},
    {OpType::FullyConnected,                    TensorUsage::Weights},
    {OpType::FullyConnected,                    TensorUsage::Scales},
    {OpType::GatherNd,                          TensorUsage::IFM0},
    {OpType::GatherNd,                          TensorUsage::IFM1},
    {OpType::GatherV2,                          TensorUsage::IFM0},
    {OpType::GatherV2,                          TensorUsage::IFM1},
    {OpType::Greater,                           TensorUsage::IFM0},
    {OpType::Greater,                           TensorUsage::IFM1},
    {OpType::GreaterEqual,                      TensorUsage::IFM0},
    {OpType::GreaterEqual,                      TensorUsage::IFM1},
    {OpType::HardSwish,                         TensorUsage::IFM0},
    {OpType::HashtableLookup,                   TensorUsage::IFM0},
    {OpType::Identity,                          TensorUsage::IFM0},
    // If                                       None
    {OpType::L2Norm,                            TensorUsage::IFM0},
    {OpType::L2Pool2D,                          TensorUsage::IFM0},
    {OpType::LRN,                               TensorUsage::IFM0},
    {OpType::LSHProjection,                     TensorUsage::Params},
    {OpType::LSHProjection,                     TensorUsage::IFM0},
    {OpType::LSHProjection,                     MakeTensorUsage(TensorUsage::Params, 1)},
    {OpType::LeakyRelu,                         TensorUsage::IFM0},
    {OpType::Less,                              TensorUsage::IFM0},
    {OpType::Less,                              TensorUsage::IFM1},
    {OpType::LessEqual,                         TensorUsage::IFM0},
    {OpType::LessEqual,                         TensorUsage::IFM1},
    {OpType::Log,                               TensorUsage::IFM0},
    {OpType::LogSoftmax,                        TensorUsage::IFM0},
    {OpType::LogicalAnd,                        TensorUsage::IFM0},
    {OpType::LogicalAnd,                        TensorUsage::IFM1},
    {OpType::LogicalNot,                        TensorUsage::IFM0},
    {OpType::LogicalNot,                        TensorUsage::IFM1},
    {OpType::LogicalOr,                         TensorUsage::IFM0},
    {OpType::LogicalOr,                         TensorUsage::IFM1},
    // LUT                                      None
    {OpType::Lstm,                              TensorUsage::IFM0},
    {OpType::Lstm,                              TensorUsage::Weights},
    {OpType::MatMul,                            TensorUsage::IFM0},
    {OpType::MatMul,                            TensorUsage::Weights},
    {OpType::MatrixDiag,                        TensorUsage::IFM0},
    {OpType::MatrixSetDiag,                     TensorUsage::IFM0},
    {OpType::MatrixSetDiag,                     TensorUsage::IFM1},
    {OpType::MaxPool,                           TensorUsage::IFM0},
    // Maximum and Minimum are variadic (not to be confused with Max and Min which are reduction operators)
    {OpType::Mean,                              TensorUsage::IFM0},
    {OpType::Mean,                              TensorUsage::Params},
    {OpType::MirrorPad,                         TensorUsage::IFM0},
    {OpType::MirrorPad,                         TensorUsage::Params},
    {OpType::Mul,                               TensorUsage::IFM0},
    {OpType::Mul,                               TensorUsage::IFM1},
    {OpType::Neg,                               TensorUsage::IFM0},
    {OpType::NonMaxSuppressionV4,               TensorUsage::IFM0},
    {OpType::NonMaxSuppressionV4,               TensorUsage::IFM1},
    {OpType::NonMaxSuppressionV4,               TensorUsage::Params},
    {OpType::NonMaxSuppressionV4,               MakeTensorUsage(TensorUsage::Params, 1)},
    {OpType::NonMaxSuppressionV4,               MakeTensorUsage(TensorUsage::Params, 2)},
    {OpType::NonMaxSuppressionV5,               TensorUsage::IFM0},
    {OpType::NonMaxSuppressionV5,               TensorUsage::IFM1},
    {OpType::NonMaxSuppressionV5,               TensorUsage::Params},
    {OpType::NonMaxSuppressionV5,               MakeTensorUsage(TensorUsage::Params, 1)},
    {OpType::NonMaxSuppressionV5,               MakeTensorUsage(TensorUsage::Params, 2)},
    {OpType::NonMaxSuppressionV5,               MakeTensorUsage(TensorUsage::Params, 3)},
    {OpType::NotEqual,                          TensorUsage::IFM0},
    {OpType::NotEqual,                          TensorUsage::IFM1},
    {OpType::OneHot,                            TensorUsage::IFM0},
    {OpType::OneHot,                            TensorUsage::Params},
    {OpType::Pack,                              TensorUsage::IFM0},
    {OpType::Pad,                               TensorUsage::IFM0},
    {OpType::Pad,                               TensorUsage::Params},
    {OpType::PadV2,                             TensorUsage::IFM0},
    {OpType::PadV2,                             TensorUsage::Params},
    {OpType::PadV2,                             MakeTensorUsage(TensorUsage::Params, 1)},
    // Placeholder                              None
    {OpType::Pow,                               TensorUsage::IFM0},
    {OpType::Pow,                               TensorUsage::IFM1},
    {OpType::Prelu,                             TensorUsage::IFM0},
    {OpType::Prelu,                             TensorUsage::Params},
    {OpType::Quantize,                          TensorUsage::IFM0},
    // Range                                    None
    {OpType::Rank,                              TensorUsage::IFM0},
    {OpType::ReduceAll,                         TensorUsage::IFM0},
    {OpType::ReduceAll,                         TensorUsage::Params},
    {OpType::ReduceAny,                         TensorUsage::IFM0},
    {OpType::ReduceAny,                         TensorUsage::Params},
    {OpType::ReduceMax,                         TensorUsage::IFM0},
    {OpType::ReduceMax,                         TensorUsage::Params},
    {OpType::ReduceMin,                         TensorUsage::IFM0},
    {OpType::ReduceMin,                         TensorUsage::Params},
    {OpType::ReduceProduct,                     TensorUsage::IFM0},
    {OpType::ReduceProduct,                     TensorUsage::Params},
    {OpType::ReduceSum,                         TensorUsage::IFM0},
    {OpType::ReduceSum,                         TensorUsage::Params},
    {OpType::Relu,                              TensorUsage::IFM0},
    {OpType::Relu6,                             TensorUsage::IFM0},
    {OpType::ReluN1To1,                         TensorUsage::IFM0},
    {OpType::ReluN,                             TensorUsage::IFM0},
    {OpType::Rescale,                           TensorUsage::IFM0},
    {OpType::Reshape,                           TensorUsage::IFM0},
    {OpType::Reshape,                           TensorUsage::Params},
    {OpType::ResizeBilinear,                    TensorUsage::IFM0},
    {OpType::ResizeBilinear,                    TensorUsage::Params},
    {OpType::ResizeNearestNeighbor,             TensorUsage::IFM0},
    {OpType::ResizeNearestNeighbor,             TensorUsage::Params},
    {OpType::Round,                             TensorUsage::IFM0},
    {OpType::Rsqrt,                             TensorUsage::IFM0},
    {OpType::ReverseSequence,                   TensorUsage::IFM0},
    {OpType::ReverseSequence,                   TensorUsage::Params},
    {OpType::ReverseV2,                         TensorUsage::IFM0},
    {OpType::ReverseV2,                         TensorUsage::Params},
    {OpType::Rnn,                               TensorUsage::IFM0},
    {OpType::Rnn,                               TensorUsage::Weights},
    {OpType::ScatterNd,                         TensorUsage::IFM0},
    {OpType::ScatterNd,                         TensorUsage::IFM1},
    {OpType::ScatterNd,                         TensorUsage::Params},
    {OpType::SegmentSum,                        TensorUsage::IFM0},
    {OpType::SegmentSum,                        TensorUsage::Params},
    {OpType::Select,                            TensorUsage::IFM0},
    {OpType::Select,                            TensorUsage::IFM1},
    {OpType::Select,                            TensorUsage::IFM2},
    {OpType::SelectV2,                          TensorUsage::IFM0},
    {OpType::SelectV2,                          TensorUsage::IFM1},
    {OpType::SelectV2,                          TensorUsage::IFM2},
    {OpType::Shape,                             TensorUsage::IFM0},
    {OpType::SignBit,                           TensorUsage::IFM0},
    {OpType::Sin,                               TensorUsage::IFM0},
    {OpType::SkipGram,                          TensorUsage::IFM0},
    {OpType::Sigmoid,                           TensorUsage::IFM0},
    {OpType::Slice,                             TensorUsage::IFM0},
    {OpType::Slice,                             TensorUsage::Params},
    {OpType::Slice,                             MakeTensorUsage(TensorUsage::Params, 1)},
    {OpType::Softmax,                           TensorUsage::IFM0},
    {OpType::SpaceToBatchND,                    TensorUsage::IFM0},
    {OpType::SpaceToBatchND,                    TensorUsage::Params},
    {OpType::SpaceToBatchND,                    MakeTensorUsage(TensorUsage::Params, 1)},
    {OpType::SpaceToDepth,                      TensorUsage::IFM0},
    {OpType::SparseToDense,                     TensorUsage::IFM0},
    {OpType::Split,                             TensorUsage::Params},
    {OpType::Split,                             TensorUsage::IFM0},
    {OpType::SplitV,                            TensorUsage::IFM0},
    {OpType::SplitV,                            TensorUsage::Params},
    {OpType::SplitV,                            MakeTensorUsage(TensorUsage::Params, 1)},
    {OpType::Sqrt,                              TensorUsage::IFM0},
    {OpType::Square,                            TensorUsage::IFM0},
    {OpType::SquaredDifference,                 TensorUsage::IFM0},
    {OpType::SquaredDifference,                 TensorUsage::IFM1},
    {OpType::Squeeze,                           TensorUsage::IFM0},
    {OpType::StridedSlice,                      TensorUsage::IFM0},
    {OpType::StridedSlice,                      TensorUsage::Params},
    {OpType::StridedSlice,                      MakeTensorUsage(TensorUsage::Params, 1)},
    {OpType::StridedSlice,                      MakeTensorUsage(TensorUsage::Params, 2)},
    {OpType::Sub,                               TensorUsage::IFM0},
    {OpType::Sub,                               TensorUsage::IFM1},
    // SubgraphInput                            None
    {OpType::Svdf,                              TensorUsage::IFM0},
    {OpType::Svdf,                              TensorUsage::Weights},
    {OpType::Svdf,                              MakeTensorUsage(TensorUsage::Weights, 1)},
    {OpType::Svdf,                              TensorUsage::Scales},
    {OpType::Svdf,                              TensorUsage::State},
    {OpType::Tanh,                              TensorUsage::IFM0},
    {OpType::Tile,                              TensorUsage::IFM0},
    {OpType::Tile,                              TensorUsage::Params},
    {OpType::TopKV2,                            TensorUsage::IFM0},
    {OpType::TopKV2,                            TensorUsage::Params},
    {OpType::Transpose,                         TensorUsage::IFM0},
    {OpType::Transpose,                         TensorUsage::Params},
    {OpType::Unique,                            TensorUsage::IFM0},
    {OpType::UnidirectionalSequenceLstm,        TensorUsage::IFM0},
    {OpType::UnidirectionalSequenceLstm,        TensorUsage::Weights},
    {OpType::UnidirectionalSequenceRnn,         TensorUsage::IFM0},
    {OpType::UnidirectionalSequenceRnn,         TensorUsage::Weights},
    {OpType::Unpack,                            TensorUsage::IFM0},
    {OpType::Where,                             TensorUsage::IFM0},
    // While                                    None
    {OpType::ZerosLike,                         TensorUsage::IFM0},
    // clang-format on
};

template<typename T>
static tflite::ActivationFunctionType GetBuiltinFaf(const tflite::Operator *tflite_operator)
{
    const auto options = tflite_operator->builtin_options_as<T>();
    assert(options);
    return options->fused_activation_function();
}

bool TfLiteMapping::CanFuseActivationFunction(const Operation *operation)
{
    if ( operation->Passthrough() == nullptr )
    {
        return false;  // Only fuse operators which came in fused
    }

    const auto type = TfLiteMapping::OpTypeToBuiltinOptions(operation->Type());
    const tflite::Operator *const passthrough = static_cast<const tflite::Operator *>(operation->Passthrough());
    tflite::ActivationFunctionType activation;

    if ( type == tflite::BuiltinOptions::Conv2DOptions )
    {
        activation = GetBuiltinFaf<tflite::Conv2DOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::Conv3DOptions )
    {
        activation = GetBuiltinFaf<tflite::Conv3DOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::Pool2DOptions )
    {
        activation = GetBuiltinFaf<tflite::Pool2DOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::DepthwiseConv2DOptions )
    {
        activation = GetBuiltinFaf<tflite::DepthwiseConv2DOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::SVDFOptions )
    {
        activation = GetBuiltinFaf<tflite::SVDFOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::RNNOptions )
    {
        activation = GetBuiltinFaf<tflite::RNNOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::SequenceRNNOptions )
    {
        activation = GetBuiltinFaf<tflite::SequenceRNNOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::BidirectionalSequenceRNNOptions )
    {
        activation = GetBuiltinFaf<tflite::BidirectionalSequenceRNNOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::FullyConnectedOptions )
    {
        activation = GetBuiltinFaf<tflite::FullyConnectedOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::ConcatenationOptions )
    {
        activation = GetBuiltinFaf<tflite::ConcatenationOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::AddOptions )
    {
        activation = GetBuiltinFaf<tflite::AddOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::MulOptions )
    {
        activation = GetBuiltinFaf<tflite::MulOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::L2NormOptions )
    {
        activation = GetBuiltinFaf<tflite::L2NormOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::LSTMOptions )
    {
        activation = GetBuiltinFaf<tflite::LSTMOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::UnidirectionalSequenceLSTMOptions )
    {
        activation = GetBuiltinFaf<tflite::UnidirectionalSequenceLSTMOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::BidirectionalSequenceLSTMOptions )
    {
        activation = GetBuiltinFaf<tflite::BidirectionalSequenceLSTMOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::SubOptions )
    {
        activation = GetBuiltinFaf<tflite::SubOptions>(passthrough);
    }
    else if ( type == tflite::BuiltinOptions::DivOptions )
    {
        activation = GetBuiltinFaf<tflite::DivOptions>(passthrough);
    }
    else
    {
        return false;  // Operator type does not support fused activations
    }

    if ( activation == tflite::ActivationFunctionType::NONE )
    {
        return false;  // Only fuse operators which came in fused
    }

    if ( (operation->Outputs().size() != 1) || (operation->OFM()->Readers().size() != 1) )
    {
        return false;  // Only fuse operators with a single direct successor
    }

    const auto &successor = operation->OFM()->Readers().front();
    if ( activation != TfLiteMapping::OpTypeToActivationFunction(successor->Type()) )
    {
        return false;  // Only fuse operators to the activation functions they arrived fused to
    }

    if ( operation->Output(TensorUsage::OFM)->quantization != successor->Output(TensorUsage::OFM)->quantization )
    {
        return false;  // Intermediate tensor contains quantization information that cannot be discarded
    }

    return true;
}

}  // namespace regor
