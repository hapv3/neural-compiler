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

#include "graph_builder.hpp"

#include "attributes.hpp"
#include "common/numeric_util.hpp"
#include "graph.hpp"
#include "operation.hpp"

#include <algorithm>
#include <memory>
#include <unordered_set>


namespace regor
{

namespace
{

TensorUsage GraphAPIUsageToTensorUsage(GraphApi::GraphTensorUsage usage)
{
    return regor::TensorUsage(usage);  // currently 1:1 mapping required
}

// clang-format off
static constexpr std::pair<tosa::Op, regor::OpType> s_aTosaMapping[] = {
    {tosa::Op::ARGMAX,     OpType::ArgMax},
    {tosa::Op::AVG_POOL2D, OpType::AvgPool},
    {tosa::Op::CONV2D,     OpType::Conv2D},
    {tosa::Op::CONV3D,     OpType::Conv3D},
    {tosa::Op::DEPTHWISE_CONV2D, OpType::DepthwiseConv2D},
    {tosa::Op::FULLY_CONNECTED,  OpType::FullyConnected},
    {tosa::Op::MATMUL,     OpType::MatMul},
    {tosa::Op::MAX_POOL2D, OpType::MaxPool},
    {tosa::Op::TRANSPOSE_CONV2D, OpType::TransposeConv2D},
    {tosa::Op::CLAMP,    OpType::Clamp},
    {tosa::Op::SIGMOID,  OpType::Sigmoid},
    {tosa::Op::TANH,     OpType::Tanh},
    {tosa::Op::ADD,      OpType::Add},
    {tosa::Op::ARITHMETIC_RIGHT_SHIFT, OpType::Asr},
    {tosa::Op::BITWISE_AND, OpType::And},
    {tosa::Op::BITWISE_OR,  OpType::Or},
    {tosa::Op::BITWISE_XOR, OpType::Xor},
    {tosa::Op::INTDIV,      OpType::Div},
    {tosa::Op::LOGICAL_AND, OpType::LogicalAnd},
    {tosa::Op::LOGICAL_LEFT_SHIFT, OpType::SHL},
    {tosa::Op::LOGICAL_RIGHT_SHIFT, OpType::SHR},
    {tosa::Op::LOGICAL_OR,  OpType::LogicalOr},
    {tosa::Op::LOGICAL_XOR, OpType::LogicalXor},
    {tosa::Op::MAXIMUM, OpType::Maximum},
    {tosa::Op::MINIMUM, OpType::Minimum},
    {tosa::Op::MUL,   OpType::Mul},
    {tosa::Op::POW,   OpType::Pow},
    {tosa::Op::SUB,   OpType::Sub},
    {tosa::Op::TABLE, OpType::Table},
    {tosa::Op::ABS,   OpType::Abs},
    {tosa::Op::BITWISE_NOT, OpType::Not},
    {tosa::Op::CEIL,  OpType::Ceil},
    {tosa::Op::CLZ,   OpType::CLZ},
    {tosa::Op::EXP,   OpType::Exp},
    {tosa::Op::FLOOR, OpType::Floor},
    {tosa::Op::LOG,   OpType::Log},
    {tosa::Op::LOGICAL_NOT, OpType::LogicalNot},
    {tosa::Op::NEGATE,      OpType::Neg},
    {tosa::Op::RECIPROCAL,  OpType::Reciprocal},
    {tosa::Op::RSQRT,   OpType::Rsqrt},
    {tosa::Op::SELECT,  OpType::Select},
    {tosa::Op::EQUAL,   OpType::Equal},
    {tosa::Op::GREATER, OpType::Greater},
    {tosa::Op::GREATER_EQUAL,  OpType::GreaterEqual},
    {tosa::Op::REDUCE_ANY,     OpType::ReduceAny},
    {tosa::Op::REDUCE_ALL,     OpType::ReduceAll},
    {tosa::Op::REDUCE_MAX,     OpType::ReduceMax},
    {tosa::Op::REDUCE_MIN,     OpType::ReduceMin},
    {tosa::Op::REDUCE_PRODUCT, OpType::ReduceProduct},
    {tosa::Op::REDUCE_SUM, OpType::ReduceSum},
    {tosa::Op::CONCAT,     OpType::Concat},
    {tosa::Op::PAD,        OpType::Pad},
    {tosa::Op::RESHAPE,    OpType::Reshape},
    {tosa::Op::REVERSE,    OpType::Reverse},
    {tosa::Op::SLICE,      OpType::Slice},
    {tosa::Op::TILE,       OpType::Tile},
    {tosa::Op::TRANSPOSE,  OpType::Transpose},
    {tosa::Op::GATHER,     OpType::Gather},
    {tosa::Op::SCATTER,    OpType::Scatter},
    {tosa::Op::RESIZE,     OpType::Resize},
    {tosa::Op::CAST,       OpType::Cast},
    {tosa::Op::RESCALE,    OpType::Rescale},
    {tosa::Op::CONST,      OpType::Const},
    {tosa::Op::IDENTITY,   OpType::Identity},
    {tosa::Op::CUSTOM,     OpType::Custom},
    {tosa::Op::COND_IF,    OpType::If},
    {tosa::Op::WHILE_LOOP, OpType::While},
    //{tosa::Op::FFT2D,      OpType::CurrentlyUnsupported},
    //{tosa::Op::RFFT2D,     OpType::CurrentlyUnsupported},
    //{tosa::Op::ERF,        OpType::CurrentlyUnsupported},
    //{tosa::Op::DIM,        OpType::CurrentlyUnsupported},
};

static constexpr std::pair<GraphApi::GraphDataType, regor::DataType> s_aTypeMapping[] = {
    {GraphApi::GraphDataType::Bool8,      DataType::Bool8},
    {GraphApi::GraphDataType::Int4Packed8, DataType::Int4Packed8},
    {GraphApi::GraphDataType::Int8,       DataType::Int8},
    {GraphApi::GraphDataType::Int16,      DataType::Int16},
    {GraphApi::GraphDataType::Int32,      DataType::Int32},
    {GraphApi::GraphDataType::Int48,      DataType::Int48},
    {GraphApi::GraphDataType::Int64,      DataType::Int64},
    {GraphApi::GraphDataType::UInt8,      DataType::UInt8},
    {GraphApi::GraphDataType::UInt16,     DataType::UInt16},
    {GraphApi::GraphDataType::UInt32,     DataType::UInt32},
    {GraphApi::GraphDataType::UInt48,     DataType::UInt48},
    {GraphApi::GraphDataType::UInt64,     DataType::UInt64},
    {GraphApi::GraphDataType::BFloat16,   DataType::BFloat16},
    {GraphApi::GraphDataType::Float16,    DataType::Float16},
    {GraphApi::GraphDataType::Float32,    DataType::Float32},
};
// clang-format on

static_assert(is_sorted(s_aTosaMapping, [](const auto &a, const auto &b) { return a.first < b.first; }), "TOSA mapping must be sorted");

bool map_tosa_op(tosa::Op op, regor::OpType &tosaOp)
{
    auto pos = std::equal_range(std::begin(s_aTosaMapping), std::end(s_aTosaMapping),
        std::pair<tosa::Op, regor::OpType>(op, {}), [](const auto &a, const auto &b) { return a.first < b.first; });
    if ( pos.first == pos.second )
    {
        return false;
    }

    tosaOp = pos.first->second;
    return true;
}

static_assert(is_sorted(s_aTypeMapping, [](const auto &a, const auto &b) { return a.first < b.first; }), "Type mapping must be sorted");

bool map_data_type(GraphApi::GraphDataType type, regor::DataType &out)
{
    auto pos = std::equal_range(std::begin(s_aTypeMapping), std::end(s_aTypeMapping),
        std::pair<GraphApi::GraphDataType, regor::DataType>(type, {}),
        [](const auto &a, const auto &b) { return a.first < b.first; });
    if ( pos.first == pos.second )
    {
        return false;
    }

    out = pos.first->second;
    return true;
}

}  // namespace


GraphBuilder::GraphBuilder(const std::string &name) : _graphName(name)
{
}

GraphBuilder::~GraphBuilder()
{
    FreeUnconnected();
}

bool GraphBuilder::RequireSyntaxVersion(uint32_t version, int32_t level)
{
    _syntaxVersion = (version & 0xFFFFFF00) | uint32_t(level);

    if ( _syntaxVersion > (GraphApi::VERSION_TOSA_0_80 | GraphApi::PROFILE_BASELINE) )  // 0.80.Baseline
    {
        return false;
    }

    return true;
}

GraphApi::GraphOperation *GraphBuilder::CreateOp(tosa::Op tosaType, const GraphKernel *kernel)
{
    OpType type = OpType::None;
    if ( !map_tosa_op(tosaType, type) )
    {
        return nullptr;
    }

    auto op = std::make_shared<Operation>(type);
    if ( kernel )
    {
        op->SetKernel(std::make_unique<Kernel>(kernel));
    }
    else
    {
        op->SetKernel(std::make_unique<Kernel>(Point2i(1, 1), Point2i(1, 1), Point2i(1, 1)));
    }
    _operations.push_back(op);

    return op.get();
}


struct GraphBuilderBuffer : public Buffer, public GraphApi::GraphBuffer
{
    template<typename TYPE>
    GraphBuilderBuffer(int sizeBytes, const TYPE *p, bool alias) : Buffer(sizeBytes, p, alias)
    {
    }
};


GraphApi::GraphBuffer *GraphBuilder::CreateBuffer(size_t sizeBytes, GraphApi::BufferMapping mapping, const void *initialData)
{
    auto buffer = std::make_shared<GraphBuilderBuffer>(
        int(std::clamp<size_t>(sizeBytes, 0, unsigned(std::numeric_limits<int>::max()))),
        reinterpret_cast<const uint8_t *>(initialData), (mapping == BufferMapping::Alias));
    _buffers.push_back(buffer);
    return buffer.get();
}

GraphApi::GraphTensor *GraphBuilder::CreateTensor(
    const char *name, const GraphShape &shape, GraphTensorLayout layout, GraphDataType dataType, GraphBuffer *buffer)
{
    DataType type;
    if ( !map_data_type(dataType, type) )
    {
        return nullptr;
    }

    auto tensor = std::make_shared<Tensor>(name, type);
    if ( shape.count > 0 )
    {
        tensor->SetStorageShape(Shape(shape.axisNHWC, size_t(shape.count)));
    }
    else
    {
        // Is scalar -- use shape [1]
        tensor->SetStorageShape(Shape(1));
    }
    // TODO: Handle external tensor format specification - tensor->SetStorageLayout(layout);
    if ( buffer )
    {
        assert(uintptr_t(buffer) % alignof(GraphBuilderBuffer) == 0);
        auto graphBuffer = static_cast<GraphBuilderBuffer *>(buffer)->shared_from_this();
        int reqBytes = DataTypeStorageSizeBytes(type, tensor->StorageShape().Elements());
        assert(layout == GraphTensorLayout::Linear);
        UNUSED(layout);
        assert(reqBytes <= graphBuffer->Size());
        if ( reqBytes > graphBuffer->Size() )
        {
            return nullptr;
        }
        tensor->SetBuffer(graphBuffer);
    }
    _tensors.push_back(tensor);
    return tensor.get();
}

void GraphBuilder::AddInput(GraphTensor *graphTensor)
{
    auto tensor = static_cast<Tensor *>(graphTensor);
    _inputs.push_back(tensor->shared_from_this());
}

void GraphBuilder::AddOutput(GraphTensor *graphTensor)
{
    auto tensor = static_cast<Tensor *>(graphTensor);
    _outputs.push_back(tensor->shared_from_this());
}

void GraphBuilder::AddPersistent(GraphTensor *graphTensor)
{
    auto tensor = static_cast<Tensor *>(graphTensor);
    _persistent.push_back(tensor->shared_from_this());
}

void GraphBuilder::AddInput(GraphOperation *graphOp, GraphTensorUsage usage, GraphTensor *graphTensor)
{
    auto op = static_cast<Operation *>(graphOp);
    // TODO check cross graph contamination - assert( std::find(_operations.begin(), _operations.end(), op) !=
    // _operations.end() );
    auto tensor = static_cast<Tensor *>(graphTensor);
    auto tmp = GraphAPIUsageToTensorUsage(usage);
    int count = op->CountInputs(tmp);
    op->ConnectInput(MakeTensorUsage(tmp, count), tensor->shared_from_this()).Set(Quantization::Unit());
    if ( usage == GraphTensorUsage::Weights && tensor->AxisOrder() == AxisOrder::HWCM )
    {  // Update kernel depth multiplier
        op->SetKernel(std::make_unique<Kernel>(op->Kernel()->WithDepthMultiplier(tensor->StorageShape().Depth())));
    }
}

void GraphBuilder::AddOutput(GraphOperation *graphOp, GraphTensorUsage usage, GraphTensor *graphTensor)
{
    auto op = static_cast<Operation *>(graphOp);
    // TODO check cross graph contamination -  assert( std::find(_operations.begin(), _operations.end(), op) !=
    // _operations.end() );
    auto tensor = static_cast<Tensor *>(graphTensor);
    auto tmp = GraphAPIUsageToTensorUsage(usage);
    int count = op->CountOutputs(tmp);
    op->ConnectOutput(MakeTensorUsage(tmp, count), tensor->shared_from_this()).Set(Quantization::Unit());
    if ( IsPooling(op->Type()) )
    {
        op->Output(MakeTensorUsage(tmp, count))->Set(RoundMode::NATURAL);
    }
}

namespace
{

const FieldInfo *FindField(const TypeInfo *info, uint32_t id)
{
    assert(info);
    size_t length;
    const FieldInfo *table = info->Fields(length);
    for ( size_t i = 0; i < length; i++ )
    {
        if ( table[i].id == id ) return &table[i];
    }
    return nullptr;
}

template<typename TYPE, typename DTYPE = TYPE>
void WriteField(void *p, const TYPE &value)
{
    assert(p);
    *reinterpret_cast<DTYPE *>(p) = value;
}

template<typename TYPE>
bool ConvertToType(void *p, [[maybe_unused]] uint8_t destType, const TYPE &value)
{
    if ( destType == FieldTypeId<TYPE>::TYPEID )
    {
        WriteField<TYPE>(p, value);
        return true;
    }
    // Convert From Point2
    else if constexpr ( FieldTypeId<TYPE>::TYPEID == FieldTypeId<Point2i>::TYPEID )
    {
        // To Fraction
        if ( destType == FieldTypeId<Fraction<int>>::TYPEID )
        {
            WriteField<TYPE, Fraction<int>>(p, value);
            return true;
        }
    }
    return false;
}

template<>
bool ConvertToType(void *p, uint8_t destType, const int32_t &value)
{
    assert(p);
    switch ( destType )
    {
        case FieldTypeId<uint8_t>::TYPEID:
            [[fallthrough]];
        case FieldTypeId<int8_t>::TYPEID:
            WriteField<uint8_t>(p, uint8_t(value));
            break;
        case FieldTypeId<uint16_t>::TYPEID:
            [[fallthrough]];
        case FieldTypeId<int16_t>::TYPEID:
            WriteField<uint16_t>(p, uint16_t(value));
            break;
        case FieldTypeId<uint32_t>::TYPEID:
            [[fallthrough]];
        case FieldTypeId<int32_t>::TYPEID:
            WriteField<uint32_t>(p, uint32_t(value));
            break;
        default:
            assert(false);
            return false;
            break;
    }
    return true;
}

template<>
bool ConvertToType(void *p, uint8_t destType, const double &value)
{
    assert(p);
    switch ( destType )
    {
        case FieldTypeId<float>::TYPEID:
            WriteField<float>(p, float(value));
            break;
        case FieldTypeId<double>::TYPEID:
            WriteField<double>(p, value);
            break;
        default:
            assert(false);
            return false;
            break;
    }
    return true;
}

template<typename TYPE>
bool WriteAttributeValue(Operation *op, GraphApi::OpAttr attrId, const TYPE &value)
{
    DynamicRef *attr = op->AttributeByKey(uint32_t(attrId) >> 12);
    if ( attr )
    {
        const auto *field = FindField(attr->Info(), (uint32_t(attrId) >> 4) & 0x0FFFFFFF);
        if ( field )
        {
            void *to = reinterpret_cast<uint8_t *>(attr->Instance()) + field->offset;
            return ConvertToType(to, field->typeId, value);
        }
    }
    return false;
}

}  // namespace

bool GraphBuilder::Set(GraphOperation *graphOp, GraphApi::OpAttr attr, bool value)
{
    auto op = static_cast<Operation *>(graphOp);
    if ( (unsigned(attr) & 0xF) != GraphApi::GRAPHAPI_TYPECODE_bool )
    {
        assert(false && "Attribute type not bool");
        return false;
    }

    return WriteAttributeValue(op, attr, value);
}

bool GraphBuilder::Set(GraphOperation *graphOp, GraphApi::OpAttr attr, int32_t value)
{
    auto op = static_cast<Operation *>(graphOp);
    if ( (unsigned(attr) & 0xF) != GraphApi::GRAPHAPI_TYPECODE_int32 )
    {
        assert(false && "Attribute type not int32");
        return false;
    }
    return WriteAttributeValue(op, attr, value);
}

bool GraphBuilder::Set(GraphOperation *graphOp, GraphApi::OpAttr attr, double value)
{
    auto op = static_cast<Operation *>(graphOp);
    if ( (unsigned(attr) & 0xF) != GraphApi::GRAPHAPI_TYPECODE_double )
    {
        assert(false && "Attribute type not double");
        return false;
    }
    return WriteAttributeValue(op, attr, value);
}

bool GraphBuilder::Set(GraphOperation *graphOp, GraphApi::OpAttr attr, const GraphApi::GraphShape &value)
{
    auto op = static_cast<Operation *>(graphOp);
    if ( (unsigned(attr) & 0xF) != GraphApi::GRAPHAPI_TYPECODE_GraphShape )
    {
        assert(false && "Attribute type not Shape");
        return false;
    }

    Shape shape;
    if ( value.count > 0 )
    {
        shape = Shape(value.axisNHWC, size_t(value.count));
    }
    else
    {
        // Is scalar -- use shape [1]
        shape = Shape(1);
    }
    return WriteAttributeValue(op, attr, shape);
}

bool GraphBuilder::Set(GraphOperation *graphOp, GraphApi::OpAttr attr, const GraphApi::FractionND &value)
{
    auto op = static_cast<Operation *>(graphOp);
    if ( (unsigned(attr) & 0xF) != GraphApi::GRAPHAPI_TYPECODE_FractionND )
    {
        assert(false && "Attribute type not FractionND");
        return false;
    }

    Point2i xy(value.n, value.d);
    return WriteAttributeValue(op, attr, xy);
}

bool GraphBuilder::Set(GraphOperation *graphOp, GraphApi::OpAttr attr, const GraphApi::Point2 &value)
{
    auto op = static_cast<Operation *>(graphOp);
    if ( (unsigned(attr) & 0xF) != GraphApi::GRAPHAPI_TYPECODE_Point2 )
    {
        assert(false && "Attribute type not Point2");
        return false;
    }
    Point2i xy(value.x, value.y);
    return WriteAttributeValue(op, attr, xy);
}

bool GraphBuilder::Set(GraphOperation *graphOp, GraphApi::OpAttr attr, const char *value)
{
    auto op = static_cast<Operation *>(graphOp);
    if ( (unsigned(attr) & 0xF) != GraphApi::GRAPHAPI_TYPECODE_string )
    {
        assert(false && "Attribute type not string");
        return false;
    }

    std::string str(value);
    return WriteAttributeValue(op, attr, str);
}

void GraphBuilder::SetZeroPoint(GraphOperation *graphOp, GraphTensorUsage tensorUsage, double zeroPoint)
{
    auto op = static_cast<Operation *>(graphOp);
    auto usage = GraphAPIUsageToTensorUsage(tensorUsage);
    auto *conn = (usage & regor::TensorUsage::TypeMask) == regor::TensorUsage::OFM ? op->Output(usage) : op->Input(usage);
    if ( conn )
    {
        conn->quantization.zeroPoints = {int64_t(zeroPoint)};
    }
}

void GraphBuilder::SetAxisOrder(GraphTensor *graphTensor, GraphApi::AxisOrder order)
{
    auto tensor = static_cast<Tensor *>(graphTensor);
    assert(tensor->Readers().size() == 0 || tensor->AxisOrder() == regor::AxisOrder(order));
    tensor->SetAxisOrder(regor::AxisOrder(order));
}

void GraphBuilder::SetAxisStrides([[maybe_unused]] GraphTensor *graphTensor, [[maybe_unused]] const GraphApi::GraphShape *axisStrides)
{
    assert(axisStrides == nullptr && "Not currently implemented");
}

void GraphBuilder::FreeUnconnected()
{
    try
    {
        // In case somebody added self-supporting graph fragments
        std::unordered_set<Operation *> connected;
        Graph::TraverseGraphFromEnd(_outputs,
            [&](Operation *op) -> bool
            {
                connected.insert(op);
                return true;
            });
        for ( auto &op : _operations )
        {
            if ( !connected.count(op.get()) )
            {
                op->Disconnect();
            }
        }
    }
    catch ( std::bad_weak_ptr & )
    {
        // ignored
    }
}


}  // namespace regor
