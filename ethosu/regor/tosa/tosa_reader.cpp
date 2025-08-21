//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "tosa_reader.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "common/shape.hpp"
#include "compiler/attributes.hpp"
#include "compiler/graph_builder.hpp"
#include "include/graphapi.hpp"
#include "tosa_mapping.hpp"
#include "tosa_schema_generated.hpp"

#include <compiler/tensor.hpp>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace tosaFb
{
struct NONE
{
};
}  // namespace tosaFb

namespace regor
{

namespace
{

struct TensorInfo
{
    GraphApi::GraphTensor *tensor;
    GraphApi::GraphShape shape;
    GraphApi::GraphDataType type;
    bool constant;

    TensorInfo() = default;

    TensorInfo(GraphApi::GraphTensor *t, const GraphApi::GraphShape &s, GraphApi::GraphDataType d, bool c) :
            tensor(t), shape(s), type(d), constant(c)
    {
    }
};


template<class TO, class FROM>
std::enable_if_t<sizeof(TO) == sizeof(FROM) && std::is_trivially_copyable_v<FROM> && std::is_trivially_copyable_v<TO> && std::is_trivially_constructible_v<TO>, TO>
BitCast(const FROM &src) noexcept
{
    TO dst;
    std::memcpy(&dst, &src, sizeof(TO));
    return dst;
}

inline void TosaAssert(bool cond, const char *msg = nullptr)
{
    if ( !cond )
    {
        throw std::runtime_error("TOSA FB Reader error: " + std::string(msg ? msg : "Failed to load TOSA model. Buffer contents inconsistent with generated schema"));
    }
}

inline void BuilderAssert(bool cond, const std::string &msg)
{
    if ( !cond )
    {
        throw std::runtime_error("TOSA builder error: " + msg);
    }
}

template<typename T>
const T &SafeDeref(const T *ptr, const char *msg = nullptr)
{
    TosaAssert(ptr, msg);
    const T &ret = *ptr;
    if constexpr ( std::is_pointer_v<T> )
    {
        TosaAssert(ret, msg);
    }
    return ret;
}

template<GraphApi::GraphDataType = GraphApi::GraphDataType::Int32, typename ARG>
double ToDouble(ARG v)
{
    return double(v);
}

template<>
double ToDouble<GraphApi::GraphDataType::Int8, const ::flatbuffers::Vector<uint8_t> *>(const ::flatbuffers::Vector<uint8_t> *v)
{
    const auto &buf = SafeDeref(v, "No Int8 buffer");
    TosaAssert(buf.size() >= 1, "Malformed constant buffer");
    int8_t r = buf[0];
    return double(r);
}

template<>
double ToDouble<GraphApi::GraphDataType::Int16, const ::flatbuffers::Vector<uint8_t> *>(const ::flatbuffers::Vector<uint8_t> *v)
{
    const auto &buf = SafeDeref(v, "No Int16 buffer");
    TosaAssert(buf.size() >= 2, "Malformed constant buffer");
    int16_t r = 0;
    for ( int i = 0; i < 2; i++ )
    {
        r |= uint16_t(buf[i]) << (i * 8);
    }
    return double(r);
}

template<>
double ToDouble<GraphApi::GraphDataType::Int48, const ::flatbuffers::Vector<uint8_t> *>(const ::flatbuffers::Vector<uint8_t> *v)
{
    const auto &buf = SafeDeref(v, "No Int48 buffer");
    TosaAssert(buf.size() >= 6, "Malformed constant buffer");
    int64_t r = 0;
    for ( int i = 0; i < 6; i++ )
    {
        r |= uint64_t(buf[i]) << (16 + i * 8);
    }
    return double(r);
}

template<>
double ToDouble<GraphApi::GraphDataType::Float32, const ::flatbuffers::Vector<uint8_t> *>(const ::flatbuffers::Vector<uint8_t> *v)
{
    const auto &buf = SafeDeref(v, "No Float32 buffer");
    TosaAssert(buf.size() >= 4, "Malformed constant buffer");
    uint32_t u = 0;
    for ( int i = 0; i < 4; i++ )
    {
        u |= uint32_t(buf[i]) << (i * 8);
    }
    return double(BitCast<float>(u));
}

template<>
double ToDouble<GraphApi::GraphDataType::Float16, const ::flatbuffers::Vector<uint8_t> *>(const ::flatbuffers::Vector<uint8_t> *v)
{
    const auto &buf = SafeDeref(v, "No Float16 buffer");
    TosaAssert(buf.size() >= 2, "Malformed constant buffer");
    uint32_t u = 0;
    for ( int i = 0; i < 2; i++ )
    {
        u |= uint32_t(buf[i]) << (i * 8);
    }
    auto sign = u >> 15;
    auto exp = (u >> 10) & 0x1F;
    auto mant = u & 0x3FF;
    if ( exp == 0x1F )
    {
        exp = 0xFF - 112;
    }
    u = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    return double(BitCast<float>(u));
}

template<>
double ToDouble<GraphApi::GraphDataType::BFloat16, const ::flatbuffers::Vector<uint8_t> *>(const ::flatbuffers::Vector<uint8_t> *v)
{
    const auto &buf = SafeDeref(v, "No BFloat16 buffer");
    TosaAssert(buf.size() >= 2, "Malformed constant buffer");
    uint32_t u = 0;
    for ( int i = 0; i < 2; i++ )
    {
        u |= uint32_t(buf[i]) << (i * 8);
    }
    u <<= 16;
    return double(BitCast<float>(u));
}

// Check that zero points are compile time constants
inline void CheckConstantZeroPoint(TensorInfo &tensorInfo, const char *tensorRole, const char *opName, int tosaOpIndex)
{
    TosaAssert(tensorInfo.constant,
        fmt::format("{} zero point tensor must be a compile time constant on {} operator at index {}.", tensorRole, opName, tosaOpIndex)
            .c_str());
}

template<typename ALLOC_TYPE = void, typename FB_TYPE>
GraphApi::GraphTensor *CreateParamTensor(const ::flatbuffers::Vector<FB_TYPE> *attr, GraphApi::IGraphBuilder *builder,
    const std::string &name, GraphApi::GraphShape *shape = nullptr)
{
    using ACTUAL_ALLOC_TYPE = std::conditional_t<std::is_same_v<ALLOC_TYPE, void>, FB_TYPE, ALLOC_TYPE>;

    const auto &buf = SafeDeref(attr, "No attribute buffer");
    GraphApi::GraphBuffer *buffer;

    if constexpr ( std::is_same_v<FB_TYPE, ACTUAL_ALLOC_TYPE> )
    {
        buffer = builder->CreateBuffer(buf.size() * sizeof(buf[0]), GraphApi::BufferMapping::Alias, buf.Data());
    }
    else
    {
        std::vector<ACTUAL_ALLOC_TYPE> vbuf(buf.begin(), buf.end());
        buffer = builder->CreateBuffer(vbuf.size() * sizeof(vbuf[0]), GraphApi::BufferMapping::Allocate, vbuf.data());
    }
    GraphApi::GraphShape tosaShape = shape ? *shape : GraphApi::GraphShape{1, {int(buf.size())}};
    GraphApi::GraphDataType type;
    if constexpr ( std::is_same_v<ACTUAL_ALLOC_TYPE, bool> ) type = GraphApi::GraphDataType::Bool8;
    else if constexpr ( std::is_same_v<ACTUAL_ALLOC_TYPE, int8_t> ) type = GraphApi::GraphDataType::Int8;
    else if constexpr ( std::is_same_v<ACTUAL_ALLOC_TYPE, int16_t> ) type = GraphApi::GraphDataType::Int16;
    else if constexpr ( std::is_same_v<ACTUAL_ALLOC_TYPE, int32_t> ) type = GraphApi::GraphDataType::Int32;
    else if constexpr ( std::is_same_v<ACTUAL_ALLOC_TYPE, int64_t> ) type = GraphApi::GraphDataType::Int64;
    else if constexpr ( std::is_same_v<ACTUAL_ALLOC_TYPE, uint8_t> ) type = GraphApi::GraphDataType::UInt8;
    else if constexpr ( std::is_same_v<ACTUAL_ALLOC_TYPE, uint16_t> ) type = GraphApi::GraphDataType::UInt16;
    else if constexpr ( std::is_same_v<ACTUAL_ALLOC_TYPE, uint32_t> ) type = GraphApi::GraphDataType::UInt32;
    else if constexpr ( std::is_same_v<ACTUAL_ALLOC_TYPE, uint64_t> ) type = GraphApi::GraphDataType::UInt64;
    else if constexpr ( std::is_same_v<ACTUAL_ALLOC_TYPE, float> ) type = GraphApi::GraphDataType::Float32;
    else
        static_assert(std::is_integral_v<ACTUAL_ALLOC_TYPE> || std::is_same_v<ACTUAL_ALLOC_TYPE, float>, "Make this more generic");
    auto tensor = builder->CreateTensor(name.c_str(), tosaShape, GraphApi::GraphTensorLayout::Linear, type, buffer);
    builder->SetAxisStrides(tensor, nullptr);  // Autocalculate
    return tensor;
}

const tosaFb::TosaGraph *LoadModel(const void *input, size_t size)
{
    const uint8_t *buffer = static_cast<const uint8_t *>(input);
    flatbuffers::Verifier verifier(buffer, size);

    TosaAssert(tosaFb::VerifyTosaGraphBuffer(verifier));
    return tosaFb::GetTosaGraph(buffer);
}

template<tosaFb::Op OP>
struct TosaAttr
{
};

std::unordered_map<tosaFb::Op, std::vector<GraphApi::GraphTensorUsage>> s_tosaTensorUsage;

GraphApi::GraphTensorUsage GetTosaTensorUsage(const tosaFb::TosaOperator &op, int index)
{
    const auto &v = s_tosaTensorUsage.at(op.op());

    return index >= int(v.size()) ? GraphApi::GraphTensorUsage::IFM : GraphApi::GraphTensorUsage(uint32_t(v[index]) & uint32_t(GraphApi::GraphTensorUsage::TypeMask));
}

#define TOSA_REGISTER_OP(OP_ENUM, ATTR_PREFIX, ...) \
    template<> \
    struct TosaAttr<tosaFb::Op::OP_ENUM> \
    { \
        static const tosaFb::ATTR_PREFIX &Get(const tosaFb::TosaOperator &op) \
        { \
            TosaAssert(op.attribute_type() == tosaFb::Attribute::ATTR_PREFIX, "Malformed TOSA Flatbuffer " #ATTR_PREFIX); \
            auto attr = op.attribute_as<tosaFb::ATTR_PREFIX>(); \
            return SafeDeref(attr, "Malformed TOSA Flatbuffer " #ATTR_PREFIX); \
        } \
\
        TosaAttr() \
        { \
            s_tosaTensorUsage[tosaFb::Op::OP_ENUM] = {__VA_ARGS__}; \
        } \
    }; \
    TosaAttr<tosaFb::Op::OP_ENUM> s_Init_##OP_ENUM

// clang-format off
TOSA_REGISTER_OP(ARGMAX,                  ArgMaxAttribute,               GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(AVG_POOL2D,              AvgPool2dAttribute,            GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Params0, GraphApi::GraphTensorUsage::Params1);
TOSA_REGISTER_OP(CONV2D,                  Conv2dAttribute,               GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Weights, GraphApi::GraphTensorUsage::Scales, GraphApi::GraphTensorUsage::Params0, GraphApi::GraphTensorUsage::Params1);
TOSA_REGISTER_OP(CONV3D,                  Conv3dAttribute,               GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Weights, GraphApi::GraphTensorUsage::Scales, GraphApi::GraphTensorUsage::Params0, GraphApi::GraphTensorUsage::Params1);
TOSA_REGISTER_OP(DEPTHWISE_CONV2D,        DepthwiseConv2dAttribute,      GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Weights, GraphApi::GraphTensorUsage::Scales, GraphApi::GraphTensorUsage::Params0, GraphApi::GraphTensorUsage::Params1);
TOSA_REGISTER_OP(FFT2D,                   FFT2dAttribute,                GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(MATMUL,                  MatMulAttribute,               GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1, GraphApi::GraphTensorUsage::Params0, GraphApi::GraphTensorUsage::Params1);
TOSA_REGISTER_OP(MAX_POOL2D,              MaxPool2dAttribute,            GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(RFFT2D,                  RFFT2dAttribute,               GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(TRANSPOSE_CONV2D,        TransposeConv2dAttribute,      GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Weights, GraphApi::GraphTensorUsage::Scales, GraphApi::GraphTensorUsage::Params0, GraphApi::GraphTensorUsage::Params1);
TOSA_REGISTER_OP(CLAMP,                   ClampAttribute,                GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(ERF,                     ErfAttribute,                  GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(SIGMOID,                 SigmoidAttribute,              GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(TANH,                    TanhAttribute,                 GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(ADD,                     AddAttribute,                  GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(ARITHMETIC_RIGHT_SHIFT,  ArithmeticRightShiftAttribute, GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(BITWISE_AND,             BitwiseAndAttribute,           GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(BITWISE_OR,              BitwiseOrAttribute,            GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(BITWISE_XOR,             BitwiseXorAttribute,           GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(INTDIV,                  IntDivAttribute,               GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(LOGICAL_AND,             LogicalAndAttribute,           GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(LOGICAL_LEFT_SHIFT,      LogicalLeftShiftAttribute,     GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(LOGICAL_RIGHT_SHIFT,     LogicalRightShiftAttribute,    GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(LOGICAL_OR,              LogicalOrAttribute,            GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(LOGICAL_XOR,             LogicalXorAttribute,           GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(MAXIMUM,                 MaximumAttribute,              GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(MINIMUM,                 MinimumAttribute,              GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(MUL,                     MulAttribute,                  GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1, GraphApi::GraphTensorUsage::Params);
TOSA_REGISTER_OP(POW,                     PowAttribute,                  GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(SUB,                     SubAttribute,                  GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(TABLE,                   TableAttribute,                GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Params);
TOSA_REGISTER_OP(ABS,                     AbsAttribute,                  GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(BITWISE_NOT,             BitwiseNotAttribute,           GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(CEIL,                    CeilAttribute,                 GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(CLZ,                     ClzAttribute,                  GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(COS,                     CosAttribute,                  GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(EXP,                     ExpAttribute,                  GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(FLOOR,                   FloorAttribute,                GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(LOG,                     LogAttribute,                  GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(LOGICAL_NOT,             LogicalNotAttribute,           GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(NEGATE,                  NegateAttribute,               GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Params0, GraphApi::GraphTensorUsage::Params1);
TOSA_REGISTER_OP(RECIPROCAL,              ReciprocalAttribute,           GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(RSQRT,                   RsqrtAttribute,                GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(SIN,                     SinAttribute,                  GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(SELECT,                  SelectAttribute,               GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1, GraphApi::GraphTensorUsage::IFM2);
TOSA_REGISTER_OP(EQUAL,                   EqualAttribute,                GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(GREATER,                 GreaterAttribute,              GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(GREATER_EQUAL,           GreaterEqualAttribute,         GraphApi::GraphTensorUsage::IFM0, GraphApi::GraphTensorUsage::IFM1);
TOSA_REGISTER_OP(REDUCE_ALL,              ReduceAllAttribute,            GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(REDUCE_ANY,              ReduceAnyAttribute,            GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(REDUCE_MAX,              ReduceMaxAttribute,            GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(REDUCE_MIN,              ReduceMinAttribute,            GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(REDUCE_PRODUCT,          ReduceProductAttribute,        GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(REDUCE_SUM,              ReduceSumAttribute,            GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(CONCAT,                  ConcatAttribute,               GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(PAD,                     PadAttribute,                  GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Params0, GraphApi::GraphTensorUsage::Params1);
TOSA_REGISTER_OP(RESHAPE,                 ReshapeAttribute,              GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Params);
TOSA_REGISTER_OP(REVERSE,                 ReverseAttribute,              GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(SLICE,                   SliceAttribute,                GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Params0, GraphApi::GraphTensorUsage::Params1);
TOSA_REGISTER_OP(TILE,                    TileAttribute,                 GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Params);
TOSA_REGISTER_OP(TRANSPOSE,               TransposeAttribute,            GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(GATHER,                  GatherAttribute,               GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(SCATTER,                 ScatterAttribute,              GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(RESIZE,                  ResizeAttribute,               GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Params0, GraphApi::GraphTensorUsage::Params1, GraphApi::GraphTensorUsage::Params2);
TOSA_REGISTER_OP(CAST,                    CastAttribute,                 GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(RESCALE,                 RescaleAttribute,              GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Params0, GraphApi::GraphTensorUsage::Params1, GraphApi::GraphTensorUsage::Params2, GraphApi::GraphTensorUsage::Params3);
TOSA_REGISTER_OP(CONST,                   ConstAttribute,                );
TOSA_REGISTER_OP(IDENTITY,                IdentityAttribute,             GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(CUSTOM,                  CustomAttribute,               GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(COND_IF,                 CondIfAttribute,               GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(WHILE_LOOP,              WhileLoopAttribute,            GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(VARIABLE,                VariableAttribute,             );
TOSA_REGISTER_OP(VARIABLE_WRITE,          VariableWriteAttribute,        GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(VARIABLE_READ,           VariableReadAttribute,         );
TOSA_REGISTER_OP(CONST_SHAPE,             ConstShapeAttribute,           );

#define FOR_ALL_AXIS_SELECT_TYPES(functor, sep) \
    functor(ARGMAX) sep \
    functor(REDUCE_ANY) sep \
    functor(REDUCE_ALL) sep \
    functor(REDUCE_MAX) sep \
    functor(REDUCE_MIN) sep \
    functor(REDUCE_PRODUCT) sep \
    functor(REDUCE_SUM) sep \
    functor(CONCAT) sep \
    functor(REVERSE)
// clang-format on

}  // namespace

void TosaReader::LoadGraphs(const tosaFb::TosaGraph *model, std::list<GraphBuilder> &builders)
{
    using GraphApi::OpAttr;

    TosaAssert(model, "No model");
    const auto &version = SafeDeref(model->version(), "Could not find version");
    const uint32_t ver_word = (uint32_t(version._major()) << 24) | (uint32_t(version._minor()) << 8) | uint32_t(version._patch());

    for ( const auto &tosaRegion : SafeDeref(model->regions(), "No regions") )
    {
        for ( const auto &tosaBasicBlock : SafeDeref(tosaRegion->blocks(), "No blocks") )
        {
            const char *bbName = SafeDeref(tosaBasicBlock->name(), "No basic block name").c_str();
            TosaAssert(bbName, "Basic block needs a valid name");
            builders.emplace_back(bbName);
            GraphApi::IGraphBuilder *builder = &builders.back();

            TosaAssert(builder->RequireSyntaxVersion(ver_word, GraphApi::PROFILE_BASELINE), "Tosa version mismatch");

            std::unordered_map<std::string, TensorInfo> tensors;
            tensors.reserve(SafeDeref(tosaBasicBlock->tensors(), "No tensors").size());

            // Vector to API shape
            auto ToApiShape = [](const ::flatbuffers::Vector<int32_t> *in) -> GraphApi::GraphShape
            {
                GraphApi::GraphShape out{};
                // Interpret a missing shape vector as scalar, rank 0 shape.
                if ( in == nullptr )
                {
                    out.count = 0;
                    return out;
                }
                const auto &buf = SafeDeref(in, "No shape vector");
                TosaAssert(buf.size() <= std::size(out.axisNHWC), "Shape rank exceeds maximum allowed");
                for ( int i = 0; i < int(buf.size()); i++ )
                {
                    out.axisNHWC[i] = buf[i];
                }
                out.count = buf.size();
                return out;
            };

            for ( const auto &tosaTensor : SafeDeref(tosaBasicBlock->tensors()) )
            {
                GraphApi::GraphBuffer *buffer = nullptr;

                const char *name = SafeDeref(tosaTensor->name(), "No tensor name").c_str();
                TosaAssert(name, "Tensor needs a valid name");
                const auto type = TosaMapping::TensorTypeToDataType(tosaTensor->type());
                TosaAssert(type != GraphApi::GraphDataType::Unknown, "Unknown data type");

                const bool variable = tosaTensor->variable();
                const bool isUnranked = tosaTensor->is_unranked();
                TosaAssert(!isUnranked, "Unranked tensors not supported");

                const auto &tensorData = tosaTensor->data();
                if ( tensorData && tensorData->size() )
                {
                    GraphApi::BufferMapping mapping = GraphApi::BufferMapping::Alias;
                    if ( reinterpret_cast<uintptr_t>(tensorData->Data()) % 8 != 0 )
                    {
                        LOG_DEBUG("{} tensor buffer is not 8 byte aligned\n", name);
                        mapping = GraphApi::BufferMapping::Allocate;
                    }
                    buffer = builder->CreateBuffer(tensorData->size(), mapping, tensorData->Data());
                    BuilderAssert(buffer, "Failed to create buffer");
                }

                auto tosaShape = ToApiShape(tosaTensor->shape());
                auto tensor = builder->CreateTensor(name, tosaShape, GraphApi::GraphTensorLayout::Linear, type, buffer);
                BuilderAssert(tensor, "Failed to create tensor");

                tensors.emplace(std::string(name), TensorInfo(tensor, tosaShape, type, buffer != nullptr));

                builder->SetAxisStrides(tensor, nullptr);  // Autocalculate
                if ( variable ) builder->AddPersistent(tensor);
            }

            // Decode shape objects as tensors
            // TODO: MLBEDSW-10904 Improve support for TosaShape
            if ( tosaBasicBlock->shapes() )
            {
                for ( const auto &tosaShape : SafeDeref(tosaBasicBlock->shapes()) )
                {
                    GraphApi::GraphBuffer *buffer = nullptr;

                    const char *name = SafeDeref(tosaShape->name(), "No shape name").c_str();
                    TosaAssert(name, "Shape needs a valid name");
                    const auto type = GraphApi::GraphDataType::Int64;

                    const auto &shapeData = tosaShape->data();
                    if ( shapeData && shapeData->size() )
                    {
                        GraphApi::BufferMapping mapping = GraphApi::BufferMapping::Alias;
                        if ( reinterpret_cast<uintptr_t>(shapeData->Data()) % 8 != 0 )
                        {
                            LOG_DEBUG("{} tensor buffer is not 8 byte aligned\n", name);
                            mapping = GraphApi::BufferMapping::Allocate;
                        }
                        buffer = builder->CreateBuffer(shapeData->size(), mapping, shapeData->Data());
                        BuilderAssert(buffer, "Failed to create buffer");
                    }

                    GraphApi::GraphShape shape;
                    shape.count = 1;
                    shape.axisNHWC[0] = tosaShape->rank();
                    auto tensor = builder->CreateTensor(name, shape, GraphApi::GraphTensorLayout::Linear, type, buffer);
                    BuilderAssert(tensor, "Failed to create tensor");

                    TosaAssert(tensors.count(name) == 0, "Shape and Tensor name collision");
                    tensors.emplace(name, TensorInfo(tensor, shape, GraphApi::GraphDataType::Int32, buffer != nullptr));
                }
            }

            const auto &tosaOperators = SafeDeref(tosaBasicBlock->operators(), "No operators");
            for ( int tosaOpIndex = 0; tosaOpIndex < int(tosaOperators.size()); tosaOpIndex++ )
            {
                const auto &tosaOperator = SafeDeref(tosaOperators[tosaOpIndex], "Invalid operator");
                if ( tosaOperator.op() == tosaFb::Op::CONST_SHAPE ) continue;
                // Connect operation to its input tensors
                std::vector<std::string> inputTensors;
                if ( tosaOperator.inputs() )
                {
                    const auto &inputTensorsFb = SafeDeref(tosaOperator.inputs(), "No inputs");
                    inputTensors.reserve(inputTensorsFb.size());
                    for ( const auto &ten : inputTensorsFb )
                        inputTensors.push_back(SafeDeref(ten, "Invalid tensor name").str());
                }
                const auto &outputTensors = SafeDeref(tosaOperator.outputs(), "No outputs");
                if ( outputTensors.empty() )
                {
                    auto message = fmt::format("WARNING: Operator: {} at index {} does not output to any tensor.\n",
                        tosaFb::EnumNameOp(tosaOperator.op()), tosaOpIndex);
                    LOG_WARN(message);
                }

                // Kernel
                GraphApi::GraphKernel kernel = {};
                GraphApi::GraphKernel *kernelPtr = nullptr;
                switch ( tosaOperator.op() )
                {
                    case tosaFb::Op::DEPTHWISE_CONV2D:
                    {
                        kernelPtr = &kernel;
                        TosaAssert(inputTensors.size() > 1, "Missing DEPTHWISE_CONV2D input tensor");
                        const auto &shape = tensors.at(inputTensors[1]).shape;
                        kernel.sizeYXZ[0] = shape.axisNHWC[0];
                        kernel.sizeYXZ[1] = shape.axisNHWC[1];
                        kernel.sizeYXZ[2] = 1;
                        const auto &attr = TosaAttr<tosaFb::Op::DEPTHWISE_CONV2D>::Get(tosaOperator);
                        TosaAssert(attr.pad(), "Missing DEPTHWISE_CONV2D pad attribute");
                        TosaAssert(attr.pad()->size() == 4, "Invalid DEPTHWISE_CONV2D pad attribute");
                        kernel.paddingTBLRNF[0] = (*attr.pad())[0];
                        kernel.paddingTBLRNF[1] = (*attr.pad())[1];
                        kernel.paddingTBLRNF[2] = (*attr.pad())[2];
                        kernel.paddingTBLRNF[3] = (*attr.pad())[3];
                        TosaAssert(attr.stride(), "Missing DEPTHWISE_CONV2D stride attribute");
                        TosaAssert(attr.stride()->size() == 2, "Invalid DEPTHWISE_CONV2D stride attribute");
                        kernel.strideYXZ[0] = (*attr.stride())[0];
                        kernel.strideYXZ[1] = (*attr.stride())[1];
                        kernel.strideYXZ[2] = 1;
                        TosaAssert(attr.dilation(), "Missing DEPTHWISE_CONV2D dilation attribute");
                        TosaAssert(attr.dilation()->size() == 2, "Invalid DEPTHWISE_CONV2D dilation attribute");
                        kernel.dilationYXZ[0] = (*attr.dilation())[0];
                        kernel.dilationYXZ[1] = (*attr.dilation())[1];
                        kernel.dilationYXZ[2] = 1;
                        // Check for illegal accumulator type
                        auto accType = attr.acc_type();
                        TosaAssert(accType == tosaFb::DType::INT32 || accType == tosaFb::DType::INT48,
                            fmt::format("Invalid acc_type attribute on DEPTHWISE_CONV2D operator with index {}.", tosaOpIndex)
                                .c_str());

                        CheckConstantZeroPoint(tensors.at(inputTensors[3]), "Input", "DEPTHWISE_CONV2D", tosaOpIndex);
                        CheckConstantZeroPoint(tensors.at(inputTensors[4]), "Weight", "DEPTHWISE_CONV2D", tosaOpIndex);
                    }
                    break;
                    case tosaFb::Op::CONV2D:
                    {
                        kernelPtr = &kernel;
                        TosaAssert(inputTensors.size() > 1, "Missing CONV2D input tensor");
                        const auto &shape = tensors.at(inputTensors[1]).shape;
                        kernel.sizeYXZ[0] = shape.axisNHWC[1];
                        kernel.sizeYXZ[1] = shape.axisNHWC[2];
                        kernel.sizeYXZ[2] = 1;
                        const auto &attr = TosaAttr<tosaFb::Op::CONV2D>::Get(tosaOperator);
                        TosaAssert(attr.pad(), "Missing CONV2D pad attribute");
                        TosaAssert(attr.pad()->size() == 4, "Invalid CONV2D pad attribute");
                        kernel.paddingTBLRNF[0] = (*attr.pad())[0];
                        kernel.paddingTBLRNF[1] = (*attr.pad())[1];
                        kernel.paddingTBLRNF[2] = (*attr.pad())[2];
                        kernel.paddingTBLRNF[3] = (*attr.pad())[3];
                        TosaAssert(attr.stride(), "Missing CONV2D stride attribute");
                        TosaAssert(attr.stride()->size() == 2, "Invalid CONV2D stride attribute");
                        kernel.strideYXZ[0] = (*attr.stride())[0];
                        kernel.strideYXZ[1] = (*attr.stride())[1];
                        kernel.strideYXZ[2] = 1;
                        TosaAssert(attr.dilation(), "Missing CONV2D dilation attribute");
                        TosaAssert(attr.dilation()->size() == 2, "Invalid CONV2D dilation attribute");
                        kernel.dilationYXZ[0] = (*attr.dilation())[0];
                        kernel.dilationYXZ[1] = (*attr.dilation())[1];
                        kernel.dilationYXZ[2] = 1;
                        // Check for illegal accumulator type
                        auto accType = attr.acc_type();
                        TosaAssert(accType == tosaFb::DType::INT32 || accType == tosaFb::DType::INT48,
                            fmt::format("Invalid acc_type attribute on CONV2D operator with index {}.", tosaOpIndex).c_str());
                        CheckConstantZeroPoint(tensors.at(inputTensors[3]), "Input", "CONV2D", tosaOpIndex);
                        CheckConstantZeroPoint(tensors.at(inputTensors[4]), "Weight", "CONV2D", tosaOpIndex);
                    }
                    break;
                    case tosaFb::Op::CONV3D:
                    {
                        kernelPtr = &kernel;
                        TosaAssert(inputTensors.size() > 1, "Missing CONV3D input tensor");
                        const auto &shape = tensors.at(inputTensors[1]).shape;
                        TosaAssert(shape.count == 5, "Invalid CONV3D input rank");
                        kernel.sizeYXZ[0] = shape.axisNHWC[2];
                        kernel.sizeYXZ[1] = shape.axisNHWC[3];
                        kernel.sizeYXZ[2] = shape.axisNHWC[1];
                        const auto &attr = TosaAttr<tosaFb::Op::CONV3D>::Get(tosaOperator);
                        TosaAssert(attr.pad(), "Missing CONV3D pad attribute");
                        TosaAssert(attr.pad()->size() == 6, "Invalid CONV3D pad attribute");
                        kernel.paddingTBLRNF[0] = (*attr.pad())[2];
                        kernel.paddingTBLRNF[1] = (*attr.pad())[3];
                        kernel.paddingTBLRNF[2] = (*attr.pad())[4];
                        kernel.paddingTBLRNF[3] = (*attr.pad())[5];
                        kernel.paddingTBLRNF[4] = (*attr.pad())[0];
                        kernel.paddingTBLRNF[5] = (*attr.pad())[1];
                        TosaAssert(attr.stride(), "Missing CONV3D stride attribute");
                        TosaAssert(attr.stride()->size() == 3, "Invalid CONV3D stride attribute");
                        kernel.strideYXZ[0] = (*attr.stride())[1];
                        kernel.strideYXZ[1] = (*attr.stride())[2];
                        kernel.strideYXZ[2] = (*attr.stride())[0];
                        TosaAssert(attr.dilation(), "Missing CONV3D dilation attribute");
                        TosaAssert(attr.dilation()->size() == 3, "Invalid CONV3D dilation attribute");
                        kernel.dilationYXZ[0] = (*attr.dilation())[1];
                        kernel.dilationYXZ[1] = (*attr.dilation())[2];
                        kernel.dilationYXZ[2] = (*attr.dilation())[0];
                        // Check for illegal accumulator type
                        auto accType = attr.acc_type();
                        TosaAssert(accType == tosaFb::DType::INT32 || accType == tosaFb::DType::INT48,
                            fmt::format("Invalid acc_type attribute on CONV3D operator with index {}.", tosaOpIndex).c_str());
                        CheckConstantZeroPoint(tensors.at(inputTensors[3]), "Input", "CONV3D", tosaOpIndex);
                        CheckConstantZeroPoint(tensors.at(inputTensors[4]), "Weight", "CONV3D", tosaOpIndex);
                    }
                    break;
                    case tosaFb::Op::TRANSPOSE_CONV2D:
                    {
                        kernelPtr = &kernel;
                        TosaAssert(inputTensors.size() > 1, "Missing TRANSPOSE_CONV2D input tensor");
                        const auto &shape = tensors.at(inputTensors[1]).shape;
                        kernel.sizeYXZ[0] = shape.axisNHWC[1];
                        kernel.sizeYXZ[1] = shape.axisNHWC[2];
                        kernel.sizeYXZ[2] = 1;
                        const auto &attr = TosaAttr<tosaFb::Op::TRANSPOSE_CONV2D>::Get(tosaOperator);

                        // Default-pad IFM with kernel-size-1
                        // Might be adjusted when rewriting OFM-padding (see graphir_optimiser)
                        kernel.paddingTBLRNF[0] = kernel.paddingTBLRNF[1] = shape.axisNHWC[1] - 1;
                        kernel.paddingTBLRNF[2] = kernel.paddingTBLRNF[3] = shape.axisNHWC[2] - 1;

                        TosaAssert(attr.stride(), "Missing TRANSPOSE_CONV2D stride attribute");
                        TosaAssert(attr.stride()->size() == 2, "Invalid TRANSPOSE_CONV2D stride attribute");
                        kernel.strideYXZ[0] = (*attr.stride())[0];
                        kernel.strideYXZ[1] = (*attr.stride())[1];
                        kernel.strideYXZ[2] = 1;
                        kernel.dilationYXZ[0] = 1;
                        kernel.dilationYXZ[1] = 1;
                        kernel.dilationYXZ[2] = 1;
                        // Check for illegal accumulator type
                        auto accType = attr.acc_type();
                        TosaAssert(accType == tosaFb::DType::INT32 || accType == tosaFb::DType::INT48,
                            fmt::format("Invalid acc_type attribute on TRANSPOSE_CONV2D operator with index {}.", tosaOpIndex)
                                .c_str());
                        CheckConstantZeroPoint(tensors.at(inputTensors[3]), "Input", "TRANSPOSE_CONV2D", tosaOpIndex);
                        CheckConstantZeroPoint(tensors.at(inputTensors[4]), "Weight", "TRANSPOSE_CONV2D", tosaOpIndex);
                    }
                    break;
                    case tosaFb::Op::AVG_POOL2D:
                    {
                        kernelPtr = &kernel;
                        const auto &attr = TosaAttr<tosaFb::Op::AVG_POOL2D>::Get(tosaOperator);
                        TosaAssert(attr.kernel(), "Missing AVG_POOL2D kernel attribute");
                        TosaAssert(attr.kernel()->size() == 2, "Invalid AVG_POOL2D kernel attribute");
                        kernel.sizeYXZ[0] = (*attr.kernel())[0];
                        kernel.sizeYXZ[1] = (*attr.kernel())[1];
                        kernel.sizeYXZ[2] = 1;
                        TosaAssert(attr.pad(), "Missing AVG_POOL2D pad attribute");
                        TosaAssert(attr.pad()->size() == 4, "Invalid AVG_POOL2D pad attribute");
                        kernel.paddingTBLRNF[0] = (*attr.pad())[0];
                        kernel.paddingTBLRNF[1] = (*attr.pad())[1];
                        kernel.paddingTBLRNF[2] = (*attr.pad())[2];
                        kernel.paddingTBLRNF[3] = (*attr.pad())[3];
                        TosaAssert(attr.stride(), "Missing AVG_POOL2D stride attribute");
                        TosaAssert(attr.stride()->size() == 2, "Invalid AVG_POOL2D stride attribute");
                        kernel.strideYXZ[0] = (*attr.stride())[0];
                        kernel.strideYXZ[1] = (*attr.stride())[1];
                        kernel.strideYXZ[2] = 1;
                        kernel.dilationYXZ[0] = 1;
                        kernel.dilationYXZ[1] = 1;
                        kernel.dilationYXZ[2] = 1;
                        // Check for illegal accumulator type
                        auto accType = attr.acc_type();
                        TosaAssert(accType == tosaFb::DType::INT32,
                            fmt::format("Invalid acc_type attribute on AVG_POOL2D operator with index {}.", tosaOpIndex).c_str());
                        CheckConstantZeroPoint(tensors.at(inputTensors[1]), "Input", "AVG_POOL2D", tosaOpIndex);
                        CheckConstantZeroPoint(tensors.at(inputTensors[2]), "Output", "AVG_POOL2D", tosaOpIndex);
                    }
                    break;
                    case tosaFb::Op::MAX_POOL2D:
                    {
                        kernelPtr = &kernel;
                        const auto &attr = TosaAttr<tosaFb::Op::MAX_POOL2D>::Get(tosaOperator);
                        TosaAssert(attr.kernel(), "Missing MAX_POOL2D kernel attribute");
                        TosaAssert(attr.kernel()->size() == 2, "Invalid MAX_POOL2D kernel attribute");
                        kernel.sizeYXZ[0] = (*attr.kernel())[0];
                        kernel.sizeYXZ[1] = (*attr.kernel())[1];
                        kernel.sizeYXZ[2] = 1;
                        TosaAssert(attr.pad(), "Missing MAX_POOL2D pad attribute");
                        TosaAssert(attr.pad()->size() == 4, "Invalid MAX_POOL2D pad attribute");
                        kernel.paddingTBLRNF[0] = (*attr.pad())[0];
                        kernel.paddingTBLRNF[1] = (*attr.pad())[1];
                        kernel.paddingTBLRNF[2] = (*attr.pad())[2];
                        kernel.paddingTBLRNF[3] = (*attr.pad())[3];
                        TosaAssert(attr.stride(), "Missing MAX_POOL2D stride attribute");
                        TosaAssert(attr.stride()->size() == 2, "Invalid MAX_POOL2D stride attribute");
                        kernel.strideYXZ[0] = (*attr.stride())[0];
                        kernel.strideYXZ[1] = (*attr.stride())[1];
                        kernel.strideYXZ[2] = 1;
                        kernel.dilationYXZ[0] = 1;
                        kernel.dilationYXZ[1] = 1;
                        kernel.dilationYXZ[2] = 1;
                    }
                    break;
                    case tosaFb::Op::MATMUL:
                    {
                        CheckConstantZeroPoint(tensors.at(inputTensors[2]), "InputA", "MATMUL", tosaOpIndex);
                        CheckConstantZeroPoint(tensors.at(inputTensors[3]), "InputB", "MATMUL", tosaOpIndex);
                    }
                    break;
                    default:
                        break;
                }
                auto opType = TosaMapping::FBOpToOp(tosaOperator.op());
                TosaAssert(opType != tosa::Op::UNKNOWN, "Unknown data type");
                auto op = builder->CreateOp(opType, kernelPtr);
                BuilderAssert(op, fmt::format("Failed to create {} operation", tosaFb::EnumNameOp(tosaOperator.op())));
                builder->SetExternalId(op, tosaOpIndex);

                switch ( tosaOperator.op() )
                {
                    case tosaFb::Op::ARITHMETIC_RIGHT_SHIFT:
                    {
                        const auto &tosaAttr = TosaAttr<tosaFb::Op::ARITHMETIC_RIGHT_SHIFT>::Get(tosaOperator);
                        BuilderAssert(builder->Set(op, OpAttr::ASR_ROUND, tosaAttr.round()), "Failed to set ASR_ROUND attribute on ARITHMETIC_RIGHT_SHIFT");
                    }
                    break;
                    case tosaFb::Op::CLAMP:
                    {
                        const auto &tosaAttr = TosaAttr<tosaFb::Op::CLAMP>::Get(tosaOperator);
                        double clampMin = 0;
                        double clampMax = 0;
                        TosaAssert(!inputTensors.empty(), "Missing CLAMP input tensor");
                        auto type = tensors.at(inputTensors[0]).type;
                        switch ( type )
                        {
                            case GraphApi::GraphDataType::Int8:
                                clampMin = ToDouble<GraphApi::GraphDataType::Int8>(tosaAttr.min_val());
                                clampMax = ToDouble<GraphApi::GraphDataType::Int8>(tosaAttr.max_val());
                                break;
                            case GraphApi::GraphDataType::Int16:
                                clampMin = ToDouble<GraphApi::GraphDataType::Int16>(tosaAttr.min_val());
                                clampMax = ToDouble<GraphApi::GraphDataType::Int16>(tosaAttr.max_val());
                                break;
                            case GraphApi::GraphDataType::Float32:
                                clampMin = ToDouble<GraphApi::GraphDataType::Float32>(tosaAttr.min_val());
                                clampMax = ToDouble<GraphApi::GraphDataType::Float32>(tosaAttr.max_val());
                                break;
                            case GraphApi::GraphDataType::Float16:
                                clampMin = ToDouble<GraphApi::GraphDataType::Float16>(tosaAttr.min_val());
                                clampMax = ToDouble<GraphApi::GraphDataType::Float16>(tosaAttr.max_val());
                                break;
                            case GraphApi::GraphDataType::BFloat16:
                                clampMin = ToDouble<GraphApi::GraphDataType::BFloat16>(tosaAttr.min_val());
                                clampMax = ToDouble<GraphApi::GraphDataType::BFloat16>(tosaAttr.max_val());
                                break;
                            default:  // empty
                                break;
                        }
                        BuilderAssert(builder->Set(op, OpAttr::CLAMP_MIN, clampMin), "Failed to set CLAMP_MIN attribute on CLAMP");
                        BuilderAssert(builder->Set(op, OpAttr::CLAMP_MAX, clampMax), "Failed to set CLAMP_MAX attribute on CLAMP");
                    }
                    break;
                    case tosaFb::Op::TRANSPOSE:
                    {
                        const auto &tosaAttr = TosaAttr<tosaFb::Op::TRANSPOSE>::Get(tosaOperator);
                        BuilderAssert(builder->Set(op, OpAttr::TRANSPOSE_PERM, ToApiShape(tosaAttr.perms())),
                            "Failed to set TRANSPOSE_PERM attribute on TRANSPOSE");
                    }
                    break;
                    case tosaFb::Op::COND_IF:
                    {
                        const auto &tosaAttr = TosaAttr<tosaFb::Op::COND_IF>::Get(tosaOperator);
                        BuilderAssert(
                            builder->Set(op, OpAttr::COND_IF, SafeDeref(tosaAttr.then_graph(), "COND_IF: No then graph").c_str()),
                            "Failed to set COND_IF attribute on COND_IF");
                        BuilderAssert(
                            builder->Set(op, OpAttr::COND_ELSE,
                                SafeDeref(tosaAttr.else_graph(), "COND_IF: No else graph").c_str()),
                            "Failed to set COND_ELSE attribute on COND_IF");
                    }
                    break;
                    case tosaFb::Op::WHILE_LOOP:
                    {
                        const auto &tosaAttr = TosaAttr<tosaFb::Op::WHILE_LOOP>::Get(tosaOperator);
                        BuilderAssert(
                            builder->Set(op, OpAttr::WHILE_BODY,
                                SafeDeref(tosaAttr.body_graph(), "WHILE_LOOP: No body graph").c_str()),
                            "Failed to set WHILE_BODY attribute on WHILE_LOOP");
                        BuilderAssert(
                            builder->Set(op, OpAttr::WHILE_COND,
                                SafeDeref(tosaAttr.cond_graph(), "WHILE_LOOP: No cond graph").c_str()),
                            "Failed to set WHILE_COND attribute on WHILE_LOOP");
                    }
                    break;
                    case tosaFb::Op::RESCALE:
                    {
                        const auto &tosaAttr = TosaAttr<tosaFb::Op::RESCALE>::Get(tosaOperator);
                        BuilderAssert(builder->Set(op, GraphApi::OpAttr::RESCALE_SCALE32, tosaAttr.scale32()),
                            "Failed to set RESCALE_SCALE32 attribute on RESCALE");
                        BuilderAssert(builder->Set(op, GraphApi::OpAttr::RESCALE_DOUBLE_ROUND, tosaAttr.rounding_mode() == tosaFb::RoundingMode::DOUBLE_ROUND),
                            "Failed to set RESCALE_DOUBLE_ROUND attribute on RESCALE");
                        BuilderAssert(builder->Set(op, GraphApi::OpAttr::RESCALE_PER_CHANNEL, tosaAttr.per_channel()),
                            "Failed to set RESCALE_PER_CHANNEL attribute on RESCALE");
                        BuilderAssert(builder->Set(op, GraphApi::OpAttr::RESCALE_INPUT_UNSIGNED, tosaAttr.input_unsigned()),
                            "Failed to set RESCALE_INPUT_UNSIGNED attribute on RESCALE");
                        BuilderAssert(builder->Set(op, GraphApi::OpAttr::RESCALE_OUTPUT_UNSIGNED, tosaAttr.output_unsigned()),
                            "Failed to set RESCALE_OUTPUT_UNSIGNED attribute on RESCALE");

                        if ( tosaAttr.per_channel() )
                        {
                            const auto &rescaleInputTensor = inputTensors[0];
                            TosaAssert(tensors.at(rescaleInputTensor).shape.count != 0,
                                fmt::format("RESCALE input tensor {} needs to have rank > 0 when per channel attribute is set.", rescaleInputTensor)
                                    .c_str());
                        }
                    }
                    break;
                    case tosaFb::Op::RESIZE:
                    {
                        const auto &tosaAttr = TosaAttr<tosaFb::Op::RESIZE>::Get(tosaOperator);
                        const auto mode = TosaMapping::FBResizeModeToResizeMode(tosaAttr.mode());
                        TosaAssert(mode != tosa::ResizeMode::UNKNOWN, "Unknown resize mode");
                        BuilderAssert(builder->Set(op, GraphApi::OpAttr::RESIZE_MODE, int(mode)), "Failed to set RESIZE_MODE attribute on RESIZE");
                    }
                    break;
#define TYPE_FUNC(opType) \
    case tosaFb::Op::opType: \
    { \
        const auto &tosa_attr = TosaAttr<tosaFb::Op::opType>::Get(tosaOperator); \
        BuilderAssert(builder->Set(op, GraphApi::OpAttr::AXIS_SELECT, tosa_attr.axis()), "Failed to set AXIS_SELECT attribute on " #opType); \
    } \
    break
                        FOR_ALL_AXIS_SELECT_TYPES(TYPE_FUNC, ;);
#undef TYPE_FUNC
#undef FOR_ALL_AXIS_SELECT_TYPES
                    case tosaFb::Op::TRANSPOSE_CONV2D:
                    {
                        const auto &tosaAttr = TosaAttr<tosaFb::Op::TRANSPOSE_CONV2D>::Get(tosaOperator);
                        TosaAssert(tosaAttr.out_pad());
                        TosaAssert(tosaAttr.out_pad()->size() == 4);
                        BuilderAssert(builder->Set(op, OpAttr::TRANSPOSE_CONV2D_OUTPAD, ToApiShape(tosaAttr.out_pad())),
                            "Failed to set OUTPAD attribute on TRANSPOSE_CONV2D");
                    }
                    break;
                    case tosaFb::Op::MUL:
                    {
                        const auto &mulShiftTensor = inputTensors[2];
                        TosaAssert(tensors.at(mulShiftTensor).shape.count == 1,
                            ("MUL shift tensor " + mulShiftTensor + " needs to have rank 1.").c_str());
                    }
                    break;
                    default:
                        break;
                }

                // Collect input usage
                std::vector<GraphApi::GraphTensorUsage> usages;
                usages.reserve(inputTensors.size());
                for ( int i = 0; i < int(inputTensors.size()); i++ )
                {
                    GraphApi::GraphTensorUsage usage = GetTosaTensorUsage(tosaOperator, i);
                    int count = 0;
                    for ( auto u : usages )
                    {
                        if ( GraphApi::GraphTensorUsage(uint32_t(u) & uint32_t(GraphApi::GraphTensorUsage::TypeMask)) == usage )
                        {
                            count++;
                        }
                    }
                    usages.push_back(GraphApi::MakeTensorUsage(usage, count));
                }
                // Add inputs
                for ( int i = 0; i < int(inputTensors.size()); i++ )
                {
                    const auto &ten = inputTensors[i];
                    TosaAssert(tensors.count(ten),
                        fmt::format("{} operator input tensor '{}' not found", tosaFb::EnumNameOp(tosaOperator.op()), ten)
                            .c_str());
                    auto usage = usages[i];
                    auto tensor = tensors.at(ten).tensor;

                    // Axis order
                    if ( usage == GraphApi::GraphTensorUsage::Weights )
                    {
                        if ( tosaOperator.op() == tosaFb::Op::DEPTHWISE_CONV2D )
                        {
                            builder->SetAxisOrder(tensor, GraphApi::AxisOrder::HWCM);
                        }
                        else if ( tosaOperator.op() == tosaFb::Op::CONV2D )
                        {
                            builder->SetAxisOrder(tensor, GraphApi::AxisOrder::OHWI);
                        }
                    }

                    builder->AddInput(op, usage, tensor);
                }
                // Add outputs
                for ( int i = 0; i < int(outputTensors.size()); i++ )
                {
                    const auto &ten = SafeDeref(outputTensors[i], "Invalid output tensor name");
                    GraphApi::GraphTensorUsage usage = GraphApi::MakeTensorUsage(GraphApi::GraphTensorUsage::OFM, i);
                    TosaAssert(tensors.count(ten.str()),
                        fmt::format("{} operator output tensor '{}' not found", tosaFb::EnumNameOp(tosaOperator.op()), ten.str())
                            .c_str());
                    builder->AddOutput(op, usage, tensors.at(ten.str()).tensor);
                }
            }

            // Add graph inputs and outputs
            if ( tosaBasicBlock->inputs() )
            {
                for ( auto ten : SafeDeref(tosaBasicBlock->inputs(), "No BasicBlock inputs") )
                {
                    TosaAssert(tensors.count(ten->str()),
                        fmt::format("BasicBlock input tensor '{}' not found", ten->str()).c_str());
                    builder->AddInput(tensors.at(ten->str()).tensor);
                }
            }
            for ( auto ten : SafeDeref(tosaBasicBlock->outputs(), "No BasicBlock outputs") )
            {
                TosaAssert(tensors.count(ten->str()),
                    fmt::format("BasicBlock output tensor '{}' not found.", ten->str()).c_str());
                auto outputTensor = static_cast<Tensor *>(tensors.at(ten->str()).tensor);  // NOLINT(*-pro-type-static-cast-downcast)
                TosaAssert(!outputTensor->Writers().empty(),
                    fmt::format("Output tensor {} does not have any writers.", ten->str()).c_str());
                builder->AddOutput(outputTensor);
            }
        }
    }
}

void TosaReader::LoadGraphs(const void *input, size_t size, std::list<GraphBuilder> &builders)
{
    LoadGraphs(LoadModel(input, size), builders);
}


}  // namespace regor
