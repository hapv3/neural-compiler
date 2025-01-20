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

#include "common/shape.hpp"
#include "compiler/attributes.hpp"
#include "compiler/graph_builder.hpp"
#include "include/graphapi.hpp"
#include "tosa_mapping.hpp"
#include "tosa_schema_generated.hpp"

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

template<class TO, class FROM>
std::enable_if_t<sizeof(TO) == sizeof(FROM) && std::is_trivially_copyable_v<FROM> && std::is_trivially_copyable_v<TO> && std::is_trivially_constructible_v<TO>, TO>
BitCast(const FROM &src) noexcept
{
    TO dst;
    std::memcpy(&dst, &src, sizeof(TO));
    return dst;
}

inline void tosa_assert(bool cond, const char *msg = nullptr)
{
    if ( !cond )
    {
        throw std::runtime_error("TOSA FB Reader error : " + std::string(msg ? msg : "Failed to load TOSA model. Buffer contents inconsistent with generated schema"));
    }
}

inline void builder_assert(bool cond, const std::string &msg)
{
    if ( !cond )
    {
        throw std::runtime_error("TOSA builder error : " + msg);
    }
}

template<typename T>
const T &SafeDeref(const T *ptr, const char *msg = nullptr)
{
    tosa_assert(ptr, msg);
    const T &ret = *ptr;
    if constexpr ( std::is_pointer_v<T> )
    {
        tosa_assert(ret, msg);
    }
    return ret;
}

template<GraphApi::GraphDataType = GraphApi::GraphDataType::Int32, typename ARG>
double ToDouble(ARG v)
{
    return double(v);
}

template<>
double ToDouble<GraphApi::GraphDataType::Int48, const ::flatbuffers::Vector<uint8_t> *>(const ::flatbuffers::Vector<uint8_t> *v)
{
    const auto &buf = SafeDeref(v);
    tosa_assert(buf.size() == 6, "Malformed constant buffer");
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
    const auto &buf = SafeDeref(v);
    tosa_assert(buf.size() == 4, "Malformed constant buffer");
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
    const auto &buf = SafeDeref(v);
    tosa_assert(buf.size() == 2, "Malformed constant buffer");
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
    const auto &buf = SafeDeref(v);
    tosa_assert(buf.size() == 2, "Malformed constant buffer");
    uint32_t u = 0;
    for ( int i = 0; i < 2; i++ )
    {
        u |= uint32_t(buf[i]) << (i * 8);
    }
    u <<= 16;
    return double(BitCast<float>(u));
}

template<typename ALLOC_TYPE = void, typename FB_TYPE>
GraphApi::GraphTensor *CreateParamTensor(const ::flatbuffers::Vector<FB_TYPE> *attr, GraphApi::IGraphBuilder *builder,
    const std::string &name, GraphApi::GraphShape *shape = nullptr)
{
    using ACTUAL_ALLOC_TYPE = std::conditional_t<std::is_same_v<ALLOC_TYPE, void>, FB_TYPE, ALLOC_TYPE>;

    const auto &buf = SafeDeref(attr);
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

    tosa_assert(tosaFb::VerifyTosaGraphBuffer(verifier));
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
            tosa_assert(op.attribute_type() == tosaFb::Attribute::ATTR_PREFIX, "Malformed TOSA Flatbuffer attribute"); \
            auto attr = op.attribute_as<tosaFb::ATTR_PREFIX>(); \
            return SafeDeref(attr, "Malformed TOSA Flatbuffer attribute"); \
        } \
\
        TosaAttr() \
        { \
            s_tosaTensorUsage[tosaFb::Op::OP_ENUM] = {__VA_ARGS__}; \
        } \
    }; \
    TosaAttr<tosaFb::Op::OP_ENUM> s_Init_##OP_ENUM

// clang-format off
TOSA_REGISTER_OP(ARGMAX,                  AxisAttribute,                 GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(AVG_POOL2D,              PoolAttribute,                 GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(CONV2D,                  ConvAttribute,                 GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Weights, GraphApi::GraphTensorUsage::Scales);
TOSA_REGISTER_OP(CONV3D,                  ConvAttribute,                 GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Weights, GraphApi::GraphTensorUsage::Scales);
TOSA_REGISTER_OP(DEPTHWISE_CONV2D,        ConvAttribute,                 GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Weights, GraphApi::GraphTensorUsage::Scales);
TOSA_REGISTER_OP(FULLY_CONNECTED,         FullyConnectedAttribute,       GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Weights, GraphApi::GraphTensorUsage::Scales);
TOSA_REGISTER_OP(MATMUL,                  MatMulAttribute,               GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(MAX_POOL2D,              PoolAttribute,                 GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(TRANSPOSE_CONV2D,        TransposeConvAttribute,        GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Weights, GraphApi::GraphTensorUsage::Scales);
TOSA_REGISTER_OP(CLAMP,                   ClampAttribute,                GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(SIGMOID,                 NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(TANH,                    NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(ADD,                     NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(ARITHMETIC_RIGHT_SHIFT,  ArithmeticRightShiftAttribute, GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(BITWISE_AND,             NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(BITWISE_OR,              NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(BITWISE_XOR,             NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(INTDIV,                  NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(LOGICAL_AND,             NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(LOGICAL_LEFT_SHIFT,      NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(LOGICAL_RIGHT_SHIFT,     NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(LOGICAL_OR,              NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(LOGICAL_XOR,             NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(MAXIMUM,                 NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(MINIMUM,                 NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(MUL,                     MulAttribute,                  GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(POW,                     NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(SUB,                     NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(TABLE,                   TableAttribute,                GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Params);
TOSA_REGISTER_OP(ABS,                     NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(BITWISE_NOT,             NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(CEIL,                    NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(CLZ,                     NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(EXP,                     NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(FLOOR,                   NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(LOG,                     NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(LOGICAL_NOT,             NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(NEGATE,                  NegateAttribute,               GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(RECIPROCAL,              NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(RSQRT,                   NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(SELECT,                  NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(EQUAL,                   NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(GREATER,                 NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(GREATER_EQUAL,           NONE,                          GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(REDUCE_ANY,              AxisAttribute,                 GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(REDUCE_ALL,              AxisAttribute,                 GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(REDUCE_MAX,              AxisAttribute,                 GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(REDUCE_MIN,              AxisAttribute,                 GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(REDUCE_PRODUCT,          AxisAttribute,                 GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(REDUCE_SUM,              AxisAttribute,                 GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(CONCAT,                  AxisAttribute,                 GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(PAD,                     PadAttribute,                  GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Params);
TOSA_REGISTER_OP(RESHAPE,                 ReshapeAttribute,              GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Params);
TOSA_REGISTER_OP(REVERSE,                 AxisAttribute,                 GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(SLICE,                   SliceAttribute,                GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(TILE,                    TileAttribute,                 GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Params);
TOSA_REGISTER_OP(TRANSPOSE,               TransposeAttribute,            GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(GATHER,                  NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(SCATTER,                 NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(RESIZE,                  ResizeAttribute,               GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Params, GraphApi::GraphTensorUsage::Params1, GraphApi::GraphTensorUsage::Params2);
TOSA_REGISTER_OP(CAST,                    NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(RESCALE,                 RescaleAttribute,              GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::Params, GraphApi::GraphTensorUsage::Params1);
TOSA_REGISTER_OP(CONST,                   NONE,                          );
TOSA_REGISTER_OP(IDENTITY,                NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(CUSTOM,                  NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(COND_IF,                 CondIfAttribute,               GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(WHILE_LOOP,              WhileLoopAttribute,            GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(FFT2D,                   FFTAttribute,                  GraphApi::GraphTensorUsage::IFM, GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(RFFT2D,                  NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(ERF,                     NONE,                          GraphApi::GraphTensorUsage::IFM);
TOSA_REGISTER_OP(DIM,                     AxisAttribute,                 GraphApi::GraphTensorUsage::IFM);
// clang-format on

}  // namespace

void TosaReader::LoadGraphs(const tosaFb::TosaGraph *model, std::list<GraphBuilder> &builders)
{
    using GraphApi::OpAttr;

    tosa_assert(model);
    const auto &version = SafeDeref(model->version());
    const uint32_t ver_word = (uint32_t(version._major()) << 24) | (uint32_t(version._minor()) << 8) | uint32_t(version._patch());

    for ( const auto &tosa_region : SafeDeref(model->regions()) )
    {
        for ( const auto &tosa_basicblock : SafeDeref(tosa_region->blocks()) )
        {
            const char *bbName = SafeDeref(tosa_basicblock->name()).c_str();
            tosa_assert(bbName, "Basic block needs a valid name");
            builders.emplace_back(bbName);
            GraphApi::IGraphBuilder *builder = &builders.back();

            tosa_assert(builder->RequireSyntaxVersion(ver_word, GraphApi::PROFILE_BASELINE), "Tosa version mismatch");

            std::unordered_map<std::string, GraphApi::GraphTensor *> tensors;
            std::unordered_map<std::string, GraphApi::GraphShape> shapes;
            std::unordered_map<std::string, GraphApi::GraphDataType> types;
            tensors.reserve(SafeDeref(tosa_basicblock->tensors()).size());

            for ( const auto &tosa_tensor : SafeDeref(tosa_basicblock->tensors()) )
            {
                GraphApi::GraphBuffer *buffer = nullptr;

                const char *name = SafeDeref(tosa_tensor->name()).c_str();
                tosa_assert(name, "Tensor needs a valid name");
                const auto type = TosaMapping::TensorTypeToDataType(tosa_tensor->type());

                const bool variable = tosa_tensor->variable();
                const bool is_unranked = tosa_tensor->is_unranked();
                tosa_assert(!is_unranked, "Unranked tensors not supported");

                Shape shape;  // Defaults to shapeless
                const auto &tensorShape = tosa_tensor->shape();
                if ( tensorShape && tensorShape->size() )
                {
                    shape = Shape(tensorShape->data(), tensorShape->size());
                }
                const auto &tensorData = tosa_tensor->data();
                if ( tensorData && tensorData->size() )
                {
                    buffer = builder->CreateBuffer(tensorData->size(), GraphApi::BufferMapping::Alias, tensorData->Data());
                    builder_assert(buffer, "Failed to create buffer");
                }

                GraphApi::GraphShape tosaShape;
                tosaShape.count = shape.ToNHWC(tosaShape.axisNHWC, std::size(tosaShape.axisNHWC));

                auto tensor = builder->CreateTensor(name, tosaShape, GraphApi::GraphTensorLayout::Linear, type, buffer);
                builder_assert(tensor, "Failed to create tensor");

                tensors[name] = tensor;
                shapes[name] = std::move(tosaShape);
                types[name] = type;

                builder->SetAxisStrides(tensor, nullptr);  // Autocalculate
                if ( variable ) builder->AddPersistent(tensor);
            }

            const auto &tosa_operators = SafeDeref(tosa_basicblock->operators());
            for ( int tosa_op_index = 0; tosa_op_index < int(tosa_operators.size()); tosa_op_index++ )
            {
                const auto &tosa_operator = SafeDeref(tosa_operators[tosa_op_index]);
                if ( tosa_operator.op() == tosaFb::Op::CONST ) continue;
                // Connect operation to its input tensors
                const auto &input_tensors_fb = SafeDeref(tosa_operator.inputs());
                std::vector<std::string> input_tensors;
                input_tensors.reserve(input_tensors_fb.size());
                for ( const auto &ten : input_tensors_fb )
                    input_tensors.push_back(SafeDeref(ten).str());
                const auto &output_tensors = SafeDeref(tosa_operator.outputs());

                // Kernel
                GraphApi::GraphKernel kernel = {};
                GraphApi::GraphKernel *kernelPtr = nullptr;
                switch ( tosa_operator.op() )
                {
                    case tosaFb::Op::DEPTHWISE_CONV2D:
                    {
                        kernelPtr = &kernel;
                        tosa_assert(input_tensors.size() > 1);
                        const auto &shape = shapes.at(input_tensors[1]);
                        kernel.sizeYXZ[0] = shape.axisNHWC[0];
                        kernel.sizeYXZ[1] = shape.axisNHWC[1];
                        kernel.sizeYXZ[2] = 1;
                        const auto &attr = TosaAttr<tosaFb::Op::DEPTHWISE_CONV2D>::Get(tosa_operator);
                        tosa_assert(attr.pad());
                        tosa_assert(attr.pad()->size() == 4);
                        kernel.paddingTBLRNF[0] = (*attr.pad())[0];
                        kernel.paddingTBLRNF[1] = (*attr.pad())[1];
                        kernel.paddingTBLRNF[2] = (*attr.pad())[2];
                        kernel.paddingTBLRNF[3] = (*attr.pad())[3];
                        tosa_assert(attr.stride());
                        tosa_assert(attr.stride()->size() == 2);
                        kernel.strideYXZ[0] = (*attr.stride())[0];
                        kernel.strideYXZ[1] = (*attr.stride())[1];
                        kernel.strideYXZ[2] = 1;
                        tosa_assert(attr.dilation());
                        tosa_assert(attr.dilation()->size() == 2);
                        kernel.dilationYXZ[0] = (*attr.dilation())[0];
                        kernel.dilationYXZ[1] = (*attr.dilation())[1];
                        kernel.dilationYXZ[2] = 1;
                    }
                    break;
                    case tosaFb::Op::CONV2D:
                    {
                        kernelPtr = &kernel;
                        tosa_assert(input_tensors.size() > 1);
                        const auto &shape = shapes.at(input_tensors[1]);
                        kernel.sizeYXZ[0] = shape.axisNHWC[1];
                        kernel.sizeYXZ[1] = shape.axisNHWC[2];
                        kernel.sizeYXZ[2] = 1;
                        const auto &attr = TosaAttr<tosaFb::Op::CONV2D>::Get(tosa_operator);
                        tosa_assert(attr.pad());
                        tosa_assert(attr.pad()->size() == 4);
                        kernel.paddingTBLRNF[0] = (*attr.pad())[0];
                        kernel.paddingTBLRNF[1] = (*attr.pad())[1];
                        kernel.paddingTBLRNF[2] = (*attr.pad())[2];
                        kernel.paddingTBLRNF[3] = (*attr.pad())[3];
                        tosa_assert(attr.stride());
                        tosa_assert(attr.stride()->size() == 2);
                        kernel.strideYXZ[0] = (*attr.stride())[0];
                        kernel.strideYXZ[1] = (*attr.stride())[1];
                        kernel.strideYXZ[2] = 1;
                        tosa_assert(attr.dilation());
                        tosa_assert(attr.dilation()->size() == 2);
                        kernel.dilationYXZ[0] = (*attr.dilation())[0];
                        kernel.dilationYXZ[1] = (*attr.dilation())[1];
                        kernel.dilationYXZ[2] = 1;
                    }
                    break;
                    case tosaFb::Op::CONV3D:
                    {
                        kernelPtr = &kernel;
                        tosa_assert(input_tensors.size() > 1);
                        const auto &shape = shapes.at(input_tensors[1]);
                        tosa_assert(shape.count == 5);
                        kernel.sizeYXZ[0] = shape.axisNHWC[2];
                        kernel.sizeYXZ[1] = shape.axisNHWC[3];
                        kernel.sizeYXZ[2] = shape.axisNHWC[1];
                        const auto &attr = TosaAttr<tosaFb::Op::CONV3D>::Get(tosa_operator);
                        tosa_assert(attr.pad());
                        tosa_assert(attr.pad()->size() == 6);
                        kernel.paddingTBLRNF[0] = (*attr.pad())[2];
                        kernel.paddingTBLRNF[1] = (*attr.pad())[3];
                        kernel.paddingTBLRNF[2] = (*attr.pad())[4];
                        kernel.paddingTBLRNF[3] = (*attr.pad())[5];
                        kernel.paddingTBLRNF[4] = (*attr.pad())[0];
                        kernel.paddingTBLRNF[5] = (*attr.pad())[1];
                        tosa_assert(attr.stride());
                        tosa_assert(attr.stride()->size() == 3);
                        kernel.strideYXZ[0] = (*attr.stride())[1];
                        kernel.strideYXZ[1] = (*attr.stride())[2];
                        kernel.strideYXZ[2] = (*attr.stride())[0];
                        tosa_assert(attr.dilation());
                        tosa_assert(attr.dilation()->size() == 3);
                        kernel.dilationYXZ[0] = (*attr.dilation())[1];
                        kernel.dilationYXZ[1] = (*attr.dilation())[2];
                        kernel.dilationYXZ[2] = (*attr.dilation())[0];
                    }
                    break;
                    case tosaFb::Op::TRANSPOSE_CONV2D:
                    {
                        kernelPtr = &kernel;
                        tosa_assert(input_tensors.size() > 1);
                        const auto &shape = shapes.at(input_tensors[1]);
                        kernel.sizeYXZ[0] = shape.axisNHWC[1];
                        kernel.sizeYXZ[1] = shape.axisNHWC[2];
                        kernel.sizeYXZ[2] = 1;
                        const auto &attr = TosaAttr<tosaFb::Op::TRANSPOSE_CONV2D>::Get(tosa_operator);

                        // Default-pad IFM with kernel-size-1
                        // Might be adjusted when rewriting OFM-padding (see graphir_optimiser)
                        kernel.paddingTBLRNF[0] = kernel.paddingTBLRNF[1] = shape.axisNHWC[1] - 1;
                        kernel.paddingTBLRNF[2] = kernel.paddingTBLRNF[3] = shape.axisNHWC[2] - 1;

                        tosa_assert(attr.stride());
                        tosa_assert(attr.stride()->size() == 2);
                        kernel.strideYXZ[0] = (*attr.stride())[0];
                        kernel.strideYXZ[1] = (*attr.stride())[1];
                        kernel.strideYXZ[2] = 1;
                        kernel.dilationYXZ[0] = 1;
                        kernel.dilationYXZ[1] = 1;
                        kernel.dilationYXZ[2] = 1;
                    }
                    break;
                    case tosaFb::Op::AVG_POOL2D:
                        [[fallthrough]];
                    case tosaFb::Op::MAX_POOL2D:
                    {
                        kernelPtr = &kernel;
                        const auto &attr = TosaAttr<tosaFb::Op::AVG_POOL2D>::Get(tosa_operator);
                        tosa_assert(attr.kernel());
                        tosa_assert(attr.kernel()->size() == 2);
                        kernel.sizeYXZ[0] = (*attr.kernel())[0];
                        kernel.sizeYXZ[1] = (*attr.kernel())[1];
                        kernel.sizeYXZ[2] = 1;
                        tosa_assert(attr.pad());
                        tosa_assert(attr.pad()->size() == 4);
                        kernel.paddingTBLRNF[0] = (*attr.pad())[0];
                        kernel.paddingTBLRNF[1] = (*attr.pad())[1];
                        kernel.paddingTBLRNF[2] = (*attr.pad())[2];
                        kernel.paddingTBLRNF[3] = (*attr.pad())[3];
                        tosa_assert(attr.stride());
                        tosa_assert(attr.stride()->size() == 2);
                        kernel.strideYXZ[0] = (*attr.stride())[0];
                        kernel.strideYXZ[1] = (*attr.stride())[1];
                        kernel.strideYXZ[2] = 1;
                        kernel.dilationYXZ[0] = 1;
                        kernel.dilationYXZ[1] = 1;
                        kernel.dilationYXZ[2] = 1;
                    }
                    break;
                    default:
                        break;
                }

                auto op = builder->CreateOp(TosaMapping::FBOpToOp(tosa_operator.op()), kernelPtr);
                builder_assert(op, "Failed to create operation");

                // Fix op Attributes
                auto ToApiShape = [](const ::flatbuffers::Vector<int32_t> *in) -> GraphApi::GraphShape
                {
                    GraphApi::GraphShape out;
                    const auto &buf = SafeDeref(in);
                    tosa_assert(buf.size() <= std::size(out.axisNHWC), "Shape rank exceeds maximum allowed");
                    for ( int i = 0; i < int(buf.size()); i++ )
                    {
                        out.axisNHWC[i] = buf[i];
                    }
                    out.count = buf.size();
                    return out;
                };

                switch ( tosa_operator.op() )
                {
                    case tosaFb::Op::ARITHMETIC_RIGHT_SHIFT:
                    {
                        const auto &tosa_attr = TosaAttr<tosaFb::Op::ARITHMETIC_RIGHT_SHIFT>::Get(tosa_operator);
                        builder_assert(builder->Set(op, OpAttr::ASR_ROUND, tosa_attr.round()),
                            "Failed to set ASR_ROUND attribute on ARITHMETIC_RIGHT_SHIFT");
                    }
                    break;
                    case tosaFb::Op::CLAMP:
                    {
                        const auto &tosa_attr = TosaAttr<tosaFb::Op::CLAMP>::Get(tosa_operator);
                        double clamp_min = tosa_attr.min_int();
                        double clamp_max = tosa_attr.max_int();
                        if ( tosa_attr.min_fp() != nullptr )
                        {
                            tosa_assert(input_tensors.size() > 0);
                            auto type = types.at(input_tensors[0]);
                            switch ( type )
                            {
                                case GraphApi::GraphDataType::Int48:
                                    clamp_min = ToDouble<GraphApi::GraphDataType::Int48>(tosa_attr.min_fp());
                                    clamp_max = ToDouble<GraphApi::GraphDataType::Int48>(tosa_attr.max_fp());
                                    break;
                                case GraphApi::GraphDataType::Float32:
                                    clamp_min = ToDouble<GraphApi::GraphDataType::Float32>(tosa_attr.min_fp());
                                    clamp_max = ToDouble<GraphApi::GraphDataType::Float32>(tosa_attr.max_fp());
                                    break;
                                case GraphApi::GraphDataType::Float16:
                                    clamp_min = ToDouble<GraphApi::GraphDataType::Float16>(tosa_attr.min_fp());
                                    clamp_max = ToDouble<GraphApi::GraphDataType::Float16>(tosa_attr.max_fp());
                                    break;
                                case GraphApi::GraphDataType::BFloat16:
                                    clamp_min = ToDouble<GraphApi::GraphDataType::BFloat16>(tosa_attr.min_fp());
                                    clamp_max = ToDouble<GraphApi::GraphDataType::BFloat16>(tosa_attr.max_fp());
                                    break;
                                default:  // empty
                                    break;
                            }
                        }
                        builder_assert(builder->Set(op, OpAttr::CLAMP_MIN, clamp_min), "Failed to set CLAMP_MIN attribute on CLAMP");
                        builder_assert(builder->Set(op, OpAttr::CLAMP_MAX, clamp_max), "Failed to set CLAMP_MAX attribute on CLAMP");
                    }
                    break;
                    case tosaFb::Op::SLICE:
                    {
                        const auto &tosa_attr = TosaAttr<tosaFb::Op::SLICE>::Get(tosa_operator);
                        builder_assert(builder->Set(op, OpAttr::SLICE_BEGIN, ToApiShape(tosa_attr.start())),
                            "Failed to set SLICE_BEGIN attribute on SLICE");
                        builder_assert(builder->Set(op, OpAttr::SLICE_SIZE, ToApiShape(tosa_attr.size())), "Failed to set SLICE_SIZE attribute on SLICE");
                    }
                    break;
                    case tosaFb::Op::MUL:
                    {
                        const auto &tosa_attr = TosaAttr<tosaFb::Op::MUL>::Get(tosa_operator);
                        builder_assert(builder->Set(op, OpAttr::MUL_SHIFT, tosa_attr.shift()), "Failed to set MUL_SHIFT attribute on MUL");
                    }
                    break;
                    case tosaFb::Op::TRANSPOSE:
                    {
                        const auto &tosa_attr = TosaAttr<tosaFb::Op::TRANSPOSE>::Get(tosa_operator);
                        builder_assert(builder->Set(op, OpAttr::TRANSPOSE_PERM, ToApiShape(tosa_attr.perms())),
                            "Failed to set TRANSPOSE_PERM attribute on TRANSPOSE");
                    }
                    break;
                    case tosaFb::Op::COND_IF:
                    {
                        const auto &tosa_attr = TosaAttr<tosaFb::Op::COND_IF>::Get(tosa_operator);
                        builder_assert(builder->Set(op, OpAttr::COND_IF, SafeDeref(tosa_attr.then_branch()).c_str()),
                            "Failed to set COND_IF attribute on COND_IF");
                        builder_assert(builder->Set(op, OpAttr::COND_ELSE, SafeDeref(tosa_attr.else_branch()).c_str()),
                            "Failed to set COND_ELSE attribute on COND_IF");
                    }
                    break;
                    case tosaFb::Op::WHILE_LOOP:
                    {
                        const auto &tosa_attr = TosaAttr<tosaFb::Op::WHILE_LOOP>::Get(tosa_operator);
                        builder_assert(builder->Set(op, OpAttr::WHILE_BODY, SafeDeref(tosa_attr.body_branch()).c_str()),
                            "Failed to set WHILE_BODY attribute on WHILE_LOOP");
                        builder_assert(builder->Set(op, OpAttr::WHILE_COND, SafeDeref(tosa_attr.cond_branch()).c_str()),
                            "Failed to set WHILE_COND attribute on WHILE_LOOP");
                    }
                    break;
                    case tosaFb::Op::RESCALE:
                    {
                        const auto &tosa_attr = TosaAttr<tosaFb::Op::RESCALE>::Get(tosa_operator);
                        builder_assert(builder->Set(op, GraphApi::OpAttr::RESCALE_SCALE32, tosa_attr.scale32()),
                            "Failed to set RESCALE_SCALE32 attribute on RESCALE");
                        builder_assert(builder->Set(op, GraphApi::OpAttr::RESCALE_DOUBLE_ROUND, tosa_attr.double_round()),
                            "Failed to set RESCALE_DOUBLE_ROUND attribute on RESCALE");
                        builder_assert(builder->Set(op, GraphApi::OpAttr::RESCALE_PER_CHANNEL, tosa_attr.per_channel()),
                            "Failed to set RESCALE_PER_CHANNEL attribute on RESCALE");
                        builder_assert(builder->Set(op, GraphApi::OpAttr::RESCALE_INPUT_UNSIGNED, tosa_attr.input_unsigned()),
                            "Failed to set RESCALE_INPUT_UNSIGNED attribute on RESCALE");
                        builder_assert(builder->Set(op, GraphApi::OpAttr::RESCALE_OUTPUT_UNSIGNED, tosa_attr.output_unsigned()),
                            "Failed to set RESCALE_OUTPUT_UNSIGNED attribute on RESCALE");

                        if ( input_tensors.size() == 1 )
                        {
                            std::string name = "multiplier_param" + std::to_string(tosa_op_index);
                            if ( tosa_attr.scale32() )
                                tensors[name] = CreateParamTensor<int32_t>(tosa_attr.multiplier(), builder, name);
                            else tensors[name] = CreateParamTensor<int16_t>(tosa_attr.multiplier(), builder, name);
                            input_tensors.push_back(name);
                            name = "shift_param" + std::to_string(tosa_op_index);
                            tensors[name] = CreateParamTensor<int8_t>(tosa_attr.shift(), builder, name);
                            input_tensors.push_back(std::move(name));
                        }
                    }
                    break;
                    case tosaFb::Op::RESHAPE:
                    {
                        const auto &tosa_attr = TosaAttr<tosaFb::Op::RESHAPE>::Get(tosa_operator);
                        builder_assert(builder->Set(op, OpAttr::RESHAPE_SHAPE, ToApiShape(tosa_attr.new_shape())),
                            "Failed to set RESHAPE_SHAPE attribute on RESHAPE");
                    }
                    break;
                    case tosaFb::Op::RESIZE:
                    {
                        const auto &tosa_attr = TosaAttr<tosaFb::Op::RESIZE>::Get(tosa_operator);
                        tosa_assert(tosa_attr.scale());
                        tosa_assert(tosa_attr.scale()->size() == 4);
                        tosa_assert(tosa_attr.offset());
                        tosa_assert(tosa_attr.offset()->size() == 2);
                        tosa_assert(tosa_attr.border());
                        tosa_assert(tosa_attr.border()->size() == 2);

                        builder_assert(
                            builder->Set(op, GraphApi::OpAttr::RESIZE_SCALEY,
                                GraphApi::FractionND{(*tosa_attr.scale())[0], (*tosa_attr.scale())[1]}),
                            "Failed to set RESIZE_SCALEY attribute on RESIZE");
                        builder_assert(
                            builder->Set(op, GraphApi::OpAttr::RESIZE_SCALEX,
                                GraphApi::FractionND{(*tosa_attr.scale())[2], (*tosa_attr.scale())[3]}),
                            "Failed to set RESIZE_SCALEX attribute on RESIZE");
                        builder_assert(
                            builder->Set(op, GraphApi::OpAttr::RESIZE_OFFSET,
                                GraphApi::Point2{(*tosa_attr.offset())[1], (*tosa_attr.offset())[0]}),
                            "Failed to set RESIZE_OFFSET attribute on RESIZE");
                        builder_assert(
                            builder->Set(op, GraphApi::OpAttr::RESIZE_BORDER,
                                GraphApi::Point2{(*tosa_attr.border())[1], (*tosa_attr.border())[0]}),
                            "Failed to set RESIZE_BORDER attribute on RESIZE");
                        builder_assert(
                            builder->Set(op, GraphApi::OpAttr::RESIZE_MODE,
                                int(TosaMapping::FBResizeModeToResizeMode(tosa_attr.mode()))),
                            "Failed to RESIZE_MODE attribute on RESIZE");

                        // If no input tensors for scale/offset/border, create them to be backwards compatible
                        if ( input_tensors.size() == 1 )
                        {
                            std::string name = "scale_param" + std::to_string(tosa_op_index);
                            tensors[name] = CreateParamTensor<int32_t>(tosa_attr.scale(), builder, name);
                            input_tensors.push_back(name);
                            name = "offset_param" + std::to_string(tosa_op_index);
                            tensors[name] = CreateParamTensor<int32_t>(tosa_attr.offset(), builder, name);
                            input_tensors.push_back(std::move(name));
                            name = "border_param" + std::to_string(tosa_op_index);
                            tensors[name] = CreateParamTensor<int32_t>(tosa_attr.border(), builder, name);
                            input_tensors.push_back(std::move(name));
                        }
                    }
                    break;
                    case tosaFb::Op::ARGMAX:
                        [[fallthrough]];
                    case tosaFb::Op::REDUCE_ANY:
                        [[fallthrough]];
                    case tosaFb::Op::REDUCE_ALL:
                        [[fallthrough]];
                    case tosaFb::Op::REDUCE_MAX:
                        [[fallthrough]];
                    case tosaFb::Op::REDUCE_MIN:
                        [[fallthrough]];
                    case tosaFb::Op::REDUCE_PRODUCT:
                        [[fallthrough]];
                    case tosaFb::Op::REDUCE_SUM:
                        [[fallthrough]];
                    case tosaFb::Op::CONCAT:
                        [[fallthrough]];
                    case tosaFb::Op::REVERSE:
                    {
                        const auto &tosa_attr = TosaAttr<tosaFb::Op::ARGMAX>::Get(tosa_operator);
                        builder_assert(builder->Set(op, GraphApi::OpAttr::AXIS_SELECT, tosa_attr.axis()), "Failed to set AXIS_SELECT attribute on REVERSE");
                        break;
                    }
                    case tosaFb::Op::TILE:
                    {
                        // If no input tensors for multiples, convert multiples attribute to param tensor
                        if ( input_tensors.size() == 1 )
                        {
                            const auto &tosa_attr = TosaAttr<tosaFb::Op::TILE>::Get(tosa_operator);
                            std::string name = "multiples" + std::to_string(tosa_op_index);
                            tensors[name] = CreateParamTensor<int32_t>(tosa_attr.multiples(), builder, name);
                            input_tensors.push_back(std::move(name));
                        }
                        break;
                    }
                    break;
                    case tosaFb::Op::PAD:
                    {
                        const auto &tosa_attr = TosaAttr<tosaFb::Op::PAD>::Get(tosa_operator);
                        double pad_const = tosa_attr.pad_const_int();
                        if ( tosa_attr.pad_const_fp() != nullptr )
                        {
                            tosa_assert(input_tensors.size() > 0);
                            auto type = types.at(input_tensors[0]);
                            switch ( type )
                            {
                                case GraphApi::GraphDataType::Int48:
                                    pad_const = ToDouble<GraphApi::GraphDataType::Int48>(tosa_attr.pad_const_fp());
                                    break;
                                case GraphApi::GraphDataType::Float32:
                                    pad_const = ToDouble<GraphApi::GraphDataType::Float32>(tosa_attr.pad_const_fp());
                                    break;
                                case GraphApi::GraphDataType::Float16:
                                    pad_const = ToDouble<GraphApi::GraphDataType::Float16>(tosa_attr.pad_const_fp());
                                    break;
                                case GraphApi::GraphDataType::BFloat16:
                                    pad_const = ToDouble<GraphApi::GraphDataType::BFloat16>(tosa_attr.pad_const_fp());
                                    break;
                                default:  // empty
                                    break;
                            }
                        }
                        builder_assert(builder->Set(op, OpAttr::PAD_PAD_CONST, pad_const), "Failed to set PAD_CONST attribute on PAD");

                        // If no input tensors for padding, convert padding attribute to param tensor
                        if ( input_tensors.size() == 1 )
                        {
                            // Padding tensor has 2D shape, but padding attribute has 1D shape
                            Shape shape(SafeDeref(tosa_attr.padding()).size() / 2, 2);
                            GraphApi::GraphShape tosaShape;
                            tosaShape.count = shape.ToNHWC(tosaShape.axisNHWC, std::size(tosaShape.axisNHWC));
                            std::string name = "padding_param" + std::to_string(tosa_op_index);
                            tensors[name] = CreateParamTensor(tosa_attr.padding(), builder, name, &tosaShape);
                            input_tensors.push_back(std::move(name));
                        }
                    }
                    break;
                    case tosaFb::Op::TABLE:
                    {
                        const auto &tosa_attr = TosaAttr<tosaFb::Op::TABLE>::Get(tosa_operator);
                        if ( input_tensors.size() == 1 )
                        {
                            std::string name = "table_param" + std::to_string(tosa_op_index);
                            auto type = types.at(input_tensors[0]);
                            assert(type == GraphApi::GraphDataType::Int8 || type == GraphApi::GraphDataType::Int16);
                            if ( type == GraphApi::GraphDataType::Int8 )
                            {
                                tensors[name] = CreateParamTensor<int8_t>(tosa_attr.table(), builder, name);
                            }
                            else
                            {
                                tensors[name] = CreateParamTensor<int16_t>(tosa_attr.table(), builder, name);
                            }
                            input_tensors.push_back(std::move(name));
                        }
                    }
                    break;
                    case tosaFb::Op::TRANSPOSE_CONV2D:
                    {
                        const auto &tosa_attr = TosaAttr<tosaFb::Op::TRANSPOSE_CONV2D>::Get(tosa_operator);
                        tosa_assert(tosa_attr.out_pad());
                        tosa_assert(tosa_attr.out_pad()->size() == 4);
                        tosa_assert(tosa_attr.output_shape());
                        tosa_assert(tosa_attr.output_shape()->size() == 4);
                        builder_assert(builder->Set(op, OpAttr::TRANSPOSE_CONV2D_OUTSHAPE, ToApiShape(tosa_attr.output_shape())),
                            "Failed to set OUTSHAPE attribute on TRANSPOSE_CONV2D");
                        builder_assert(builder->Set(op, OpAttr::TRANSPOSE_CONV2D_OUTPAD, ToApiShape(tosa_attr.out_pad())),
                            "Failed to set OUTPAD attribute on TRANSPOSE_CONV2D");
                    }
                    break;
                    default:
                        break;
                }

                // Collect input usage
                std::vector<GraphApi::GraphTensorUsage> usages;
                usages.reserve(input_tensors.size());
                for ( int i = 0; i < int(input_tensors.size()); i++ )
                {
                    GraphApi::GraphTensorUsage usage = GetTosaTensorUsage(tosa_operator, i);
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
                for ( int i = 0; i < int(input_tensors.size()); i++ )
                {
                    auto usage = usages[i];
                    auto tensor = tensors.at(input_tensors[i]);

                    // Axis order
                    if ( usage == GraphApi::GraphTensorUsage::Weights )
                    {
                        if ( tosa_operator.op() == tosaFb::Op::DEPTHWISE_CONV2D )
                        {
                            builder->SetAxisOrder(tensor, GraphApi::AxisOrder::HWCM);
                        }
                        else if ( tosa_operator.op() == tosaFb::Op::CONV2D )
                        {
                            builder->SetAxisOrder(tensor, GraphApi::AxisOrder::OHWI);
                        }
                        else if ( tosa_operator.op() == tosaFb::Op::FULLY_CONNECTED )
                        {
                            builder->SetAxisOrder(tensor, GraphApi::AxisOrder::OI);
                        }
                    }

                    builder->AddInput(op, usage, tensor);

                    // Zero point
                    switch ( tosa_operator.op() )
                    {
                        case tosaFb::Op::AVG_POOL2D:
                            if ( usage == GraphApi::GraphTensorUsage::IFM )
                            {
                                const auto &tosa_attr = TosaAttr<tosaFb::Op::AVG_POOL2D>::Get(tosa_operator);
                                builder->SetZeroPoint(op, usage, double(tosa_attr.input_zp()));
                            }
                            break;
                        case tosaFb::Op::CONV2D:
                            [[fallthrough]];
                        case tosaFb::Op::CONV3D:
                            [[fallthrough]];
                        case tosaFb::Op::DEPTHWISE_CONV2D:
                            if ( usage == GraphApi::GraphTensorUsage::IFM )
                            {
                                const auto &tosa_attr = TosaAttr<tosaFb::Op::CONV2D>::Get(tosa_operator);
                                builder->SetZeroPoint(op, usage, double(tosa_attr.input_zp()));
                            }
                            if ( usage == GraphApi::GraphTensorUsage::Weights )
                            {
                                const auto &tosa_attr = TosaAttr<tosaFb::Op::CONV2D>::Get(tosa_operator);
                                builder->SetZeroPoint(op, usage, double(tosa_attr.weight_zp()));
                            }
                            break;
                        case tosaFb::Op::FULLY_CONNECTED:
                            if ( usage == GraphApi::GraphTensorUsage::IFM )
                            {
                                const auto &tosa_attr = TosaAttr<tosaFb::Op::FULLY_CONNECTED>::Get(tosa_operator);
                                builder->SetZeroPoint(op, usage, double(tosa_attr.input_zp()));
                            }
                            if ( usage == GraphApi::GraphTensorUsage::Weights )
                            {
                                const auto &tosa_attr = TosaAttr<tosaFb::Op::FULLY_CONNECTED>::Get(tosa_operator);
                                builder->SetZeroPoint(op, usage, double(tosa_attr.weight_zp()));
                            }
                            break;
                        case tosaFb::Op::MATMUL:
                            if ( usage == GraphApi::GraphTensorUsage::IFM0 )
                            {
                                const auto &tosa_attr = TosaAttr<tosaFb::Op::MATMUL>::Get(tosa_operator);
                                builder->SetZeroPoint(op, usage, double(tosa_attr.a_zp()));
                            }
                            if ( usage == GraphApi::GraphTensorUsage::IFM1 )
                            {
                                const auto &tosa_attr = TosaAttr<tosaFb::Op::MATMUL>::Get(tosa_operator);
                                builder->SetZeroPoint(op, usage, double(tosa_attr.b_zp()));
                            }
                            break;
                        case tosaFb::Op::TRANSPOSE_CONV2D:
                            if ( usage == GraphApi::GraphTensorUsage::IFM )
                            {
                                const auto &tosa_attr = TosaAttr<tosaFb::Op::TRANSPOSE_CONV2D>::Get(tosa_operator);
                                builder->SetZeroPoint(op, usage, double(tosa_attr.input_zp()));
                            }
                            if ( usage == GraphApi::GraphTensorUsage::Weights )
                            {
                                const auto &tosa_attr = TosaAttr<tosaFb::Op::TRANSPOSE_CONV2D>::Get(tosa_operator);
                                builder->SetZeroPoint(op, usage, double(tosa_attr.weight_zp()));
                            }
                            break;
                        case tosaFb::Op::NEGATE:
                            if ( usage == GraphApi::GraphTensorUsage::IFM )
                            {
                                const auto &tosa_attr = TosaAttr<tosaFb::Op::NEGATE>::Get(tosa_operator);
                                builder->SetZeroPoint(op, usage, double(tosa_attr.input1_zp()));
                            }
                            break;
                        case tosaFb::Op::RESCALE:
                            if ( usage == GraphApi::GraphTensorUsage::IFM )
                            {
                                const auto &tosa_attr = TosaAttr<tosaFb::Op::RESCALE>::Get(tosa_operator);
                                builder->SetZeroPoint(op, usage, double(tosa_attr.input_zp()));
                            }
                            break;
                        default:
                            break;
                    }
                }
                // Add outputs
                for ( int i = 0; i < int(output_tensors.size()); i++ )
                {
                    const auto &ten = SafeDeref(output_tensors[i]);
                    GraphApi::GraphTensorUsage usage = GraphApi::MakeTensorUsage(GraphApi::GraphTensorUsage::OFM, i);
                    builder->AddOutput(op, usage, tensors.at(ten.str()));

                    // Zero point
                    switch ( tosa_operator.op() )
                    {
                        case tosaFb::Op::AVG_POOL2D:
                            if ( usage == GraphApi::GraphTensorUsage::OFM )
                            {
                                const auto &tosa_attr = TosaAttr<tosaFb::Op::AVG_POOL2D>::Get(tosa_operator);
                                builder->SetZeroPoint(op, usage, double(tosa_attr.output_zp()));
                            }
                            break;
                        case tosaFb::Op::NEGATE:
                            if ( usage == GraphApi::GraphTensorUsage::OFM )
                            {
                                const auto &tosa_attr = TosaAttr<tosaFb::Op::NEGATE>::Get(tosa_operator);
                                builder->SetZeroPoint(op, usage, double(tosa_attr.output_zp()));
                            }
                            break;
                        case tosaFb::Op::RESCALE:
                            if ( usage == GraphApi::GraphTensorUsage::OFM )
                            {
                                const auto &tosa_attr = TosaAttr<tosaFb::Op::RESCALE>::Get(tosa_operator);
                                builder->SetZeroPoint(op, usage, double(tosa_attr.output_zp()));
                            }
                            break;
                        default:
                            break;
                    }
                }
            }

            // Add graph inputs and outputs
            if ( tosa_basicblock->inputs() )
            {
                for ( auto ten : SafeDeref(tosa_basicblock->inputs()) )
                {
                    builder->AddInput(tensors.at(ten->str()));
                }
            }
            for ( auto ten : SafeDeref(tosa_basicblock->outputs()) )
            {
                builder->AddOutput(tensors.at(ten->str()));
            }
        }
    }
}

void TosaReader::LoadGraphs(const void *input, size_t size, std::list<GraphBuilder> &builders)
{
    LoadGraphs(LoadModel(input, size), builders);
}


}  // namespace regor
