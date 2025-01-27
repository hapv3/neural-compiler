//
// SPDX-FileCopyrightText: Copyright 2021-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "architecture/architecture.hpp"
#include "architecture/architecture_constraints.hpp"
#include "common/buffer_view.hpp"
#include "operation.hpp"
#include "quantization.hpp"
#include "shape_util.hpp"
#include "tensor.hpp"

#include <numeric>

namespace regor
{

#define FOR_ALL_INT_TYPES(functor, sep) \
    functor(uint8_t) sep functor(uint16_t) \
    sep functor(uint32_t) \
    sep functor(uint64_t) \
    sep functor(int8_t) \
    sep functor(int16_t) \
    sep functor(int32_t) \
    sep functor(int64_t)

inline std::shared_ptr<Tensor> CreateConstTensor(
    const std::string &name, DataType type, const std::shared_ptr<Buffer> &buffer, const Shape *shape = nullptr)
{
    Shape tensorShape;
    if ( shape == nullptr )
    {
        tensorShape = Shape(DataTypeElements(type, buffer->Size()));
    }
    else
    {
        tensorShape = *shape;
    }
    auto tensor = std::make_shared<Tensor>(name, type, tensorShape, buffer);
    return tensor;
}

template<typename T>
std::shared_ptr<Tensor> CreateConstTensor(const std::string &name, T value)
{
    auto buf = std::make_shared<Buffer>(std::vector<T>{value});
    return CreateConstTensor(name, DataTypeOf<T>::value, buf);
}

// Create a single element constant tensor with the specified data type and value (value is not bounds-checked)
inline std::shared_ptr<Tensor> CreateConstTensor(const std::string &name, DataType type, int value)
{
    switch ( type )
    {
#define TYPE_FUNC(x) \
    case DataTypeOf<x>::value: \
        return CreateConstTensor(name, type, std::make_shared<Buffer>(std::vector<x>{x(value)}));
        FOR_ALL_INT_TYPES(TYPE_FUNC, ;);
#undef TYPE_FUNC
        default:
            return CreateConstTensor(name, value);
    };
}

// Convert a constant Tensor to a Shape
// Parameters:
// - tensor: Tensor to convert to shape.
// - size: Number of elements to read from tensor.
// - stride: Number of elements to step after each read.
// - offset:  Number of elements to step before first read.
inline Shape TensorToShape(Tensor *tensor, int size, int stride = 1, int offset = 0)
{
    Shape shape(nullptr, size);
    switch ( tensor->Type() )
    {
#define TYPE_FUNC(x) \
    case DataTypeOf<x>::value: \
    { \
        const auto values = tensor->View().Values<x>(); \
        for ( int i = 0; i < size; i++ ) \
            shape[i] = int(values[stride * i + offset]); \
    } \
    break;
        FOR_ALL_INT_TYPES(TYPE_FUNC, ;);
#undef TYPE_FUNC
        default:
            assert(false);
    }
#undef FOR_ALL_INT_TYPES
    return shape;
}

inline Operation *CreateLUT(const std::shared_ptr<Tensor> &ifm, const std::shared_ptr<Tensor> &lut, const Quantization &ifmQuantization,
    const Quantization &ofmQuantization, DataType dtype = DataType::None, const Shape *ifmShape = nullptr,
    std::shared_ptr<Tensor> ofm = nullptr, TensorSlice ifmSlice = {}, TensorSlice ofmSlice = {})
{
    auto op = std::make_shared<Operation>(OpType::LUT);
    if ( dtype == DataType::None )
    {
        dtype = lut->Type();
    }
    if ( ifmShape == nullptr )
    {
        ifmShape = &ifm->StorageShape();
    }
    op->ConnectInput(TensorUsage::IFM, ifm).Set(*ifmShape).Set(ifmQuantization).Set(ifmSlice);

    op->ConnectInput(TensorUsage::LUT, lut);
    if ( ofm == nullptr )
    {
        ofm = std::make_shared<Tensor>(ifm->Name() + "/lut", dtype);
        ofm->SetStorageShape(*ifmShape);
    }
    op->ConnectOutput(TensorUsage::OFM, ofm).Set(ofm->StorageShape()).Set(ofmQuantization).Set(ofmSlice);
    return op.get();
}

inline Operation *CreateDepthwiseMaxpool(const std::shared_ptr<Tensor> &ifm, const Shape &ifmShape,
    const Quantization &ifmQuantization, const Quantization &ofmQuantization)
{
    auto op = std::make_shared<Operation>(OpType::MaxPool);
    int height = ifmShape.ElementsWH();
    int width = ifmShape.Depth();
    auto kernel = std::make_unique<Kernel>(Point2i(width, 1), Point2i(1, 1), Point2i(1, 1), 1);
    auto ofm = std::make_shared<Tensor>(ifm->Name() + "/maxpool", ifm->Type());
    ofm->SetStorageShape(Shape(1, ifmShape.Height(), ifmShape.Width(), 1));
    op->SetKernel(std::move(kernel));

    op->ConnectInput(TensorUsage::IFM, ifm).Set(ifmQuantization);
    op->Input(TensorUsage::IFM)->shape = Shape(1, height, width, 1);
    op->ConnectOutput(TensorUsage::OFM, ofm).Set(ofmQuantization);
    op->Output(TensorUsage::OFM)->shape = Shape(1, height, 1, 1);
    return op.get();
}

inline Operation *CreateReduceSum(const std::shared_ptr<Tensor> &ifm, const Quantization &ifmQuantization, const Quantization &ofmQuantization)
{
    const auto &ifmShape = ifm->StorageShape();
    auto op = std::make_shared<Operation>(OpType::ReduceSum);
    auto attr = op->Attribute<axis_attr_t>();
    attr->axis = ifmShape.Size() - 1;  // Depth dimension
    auto ofm = std::make_shared<Tensor>(ifm->Name() + "/reducesum", DataType::Int32);
    ofm->SetStorageShape(ifmShape.WithDepth(1));
    op->ConnectInput(TensorUsage::IFM, ifm).Set(ifmQuantization);
    op->ConnectOutput(TensorUsage::OFM, ofm).Set(ofmQuantization);
    return op.get();
}

inline Operation *CreateElementwise(OpType type, const std::shared_ptr<Tensor> &ifm, const std::shared_ptr<Tensor> &ifm2,
    const Quantization &ifmQuantization, const Quantization &ifm2Quantization, const Quantization &ofmQuantization,
    DataType dtype = DataType::None, const Shape *ifmShape = nullptr, const Shape *ifm2Shape = nullptr)
{
    assert(IsElementwise(type));
    auto op = std::make_shared<Operation>(type);
    op->ConnectInput(TensorUsage::IFM, ifm).Set(ifmQuantization);
    if ( ifmShape ) op->Input(TensorUsage::IFM)->shape = *ifmShape;
    Shape ofmShape;
    if ( ifm2 )
    {
        op->ConnectInput(TensorUsage::IFM1, ifm2).Set(ifm2Quantization);
        if ( ifm2Shape ) op->Input(TensorUsage::IFM1)->shape = *ifm2Shape;

        // Compute the broadcasted OFM shape
        const Shape &a = op->Input(TensorUsage::IFM0)->shape;
        const Shape &b = op->Input(TensorUsage::IFM1)->shape;
        const unsigned dims = std::clamp(std::min(a.Size(), b.Size()), 0, 31);
        assert(dims < 32);
        const unsigned mask = (1u << dims) - 1u;
        assert(((a.EqualMask(a.WithOnes()) | b.EqualMask(b.WithOnes()) | a.EqualMask(b)) & mask) == mask);
        ofmShape = Shape::Max(a, b);
    }
    else
    {
        ofmShape = op->Input(TensorUsage::IFM)->shape;
    }

    if ( dtype == DataType::None ) dtype = ifm->Type();
    auto ofm = std::make_shared<Tensor>(ifm->Name() + "/" + OpTypeToString(type), dtype);
    ofm->SetStorageShape(ofmShape);
    op->ConnectOutput(TensorUsage::OFM, ofm).Set(ofmQuantization);
    return op.get();
}

inline Operation *CreateBinaryElementwise(OpType type, const std::shared_ptr<Tensor> &ifm, const std::shared_ptr<Tensor> &ifm2,
    const Quantization &ifmQuantization, const Quantization &ifm2Quantization, const Quantization &ofmQuantization,
    DataType dtype = DataType::None, const Shape *ifmShape = nullptr, const Shape *ifm2Shape = nullptr)
{
    assert(IsBinaryElementwise(type));
    return CreateElementwise(type, ifm, ifm2, ifmQuantization, ifm2Quantization, ofmQuantization, dtype, ifmShape, ifm2Shape);
}

inline Operation *CreateUnaryElementwise(OpType type, const std::shared_ptr<Tensor> &ifm, const Quantization &ifmQuantization,
    const Quantization &ofmQuantization, DataType dtype = DataType::None, const Shape *ifmShape = nullptr)
{
    assert(IsUnaryElementwise(type));
    return CreateElementwise(type, ifm, nullptr, ifmQuantization, {}, ofmQuantization, dtype, ifmShape);
}

inline Operation *CreateClz(const std::shared_ptr<Tensor> &ifm, const Quantization &ifmQuantization,
    const Quantization &ofmQuantization, DataType dtype = DataType::None, const Shape *ifmShape = nullptr)
{
    return CreateUnaryElementwise(OpType::CLZ, ifm, ifmQuantization, ofmQuantization, dtype, ifmShape);
}

inline Operation *CreateAdd(const std::shared_ptr<Tensor> &ifm, const std::shared_ptr<Tensor> &ifm2,
    const Quantization &ifmQuantization, const Quantization &ifm2Quantization, const Quantization &ofmQuantization,
    DataType dtype = DataType::None, const Shape *ifmShape = nullptr, const Shape *ifm2Shape = nullptr)
{
    return CreateBinaryElementwise(OpType::Add, ifm, ifm2, ifmQuantization, ifm2Quantization, ofmQuantization, dtype, ifmShape, ifm2Shape);
}

inline Operation *CreateMul(const std::shared_ptr<Tensor> &ifm, const std::shared_ptr<Tensor> &ifm2,
    const Quantization &ifmQuantization, const Quantization &ifm2Quantization, const Quantization &ofmQuantization,
    DataType dtype = DataType::None, const Shape *ifmShape = nullptr, const Shape *ifm2Shape = nullptr)
{
    return CreateBinaryElementwise(OpType::Mul, ifm, ifm2, ifmQuantization, ifm2Quantization, ofmQuantization, dtype, ifmShape, ifm2Shape);
}

inline Operation *CreateSub(const std::shared_ptr<Tensor> &ifm, const std::shared_ptr<Tensor> &ifm2,
    const Quantization &ifmQuantization, const Quantization &ifm2Quantization, const Quantization &ofmQuantization,
    DataType dtype = DataType::None, const Shape *ifmShape = nullptr, const Shape *ifm2Shape = nullptr)
{
    return CreateBinaryElementwise(OpType::Sub, ifm, ifm2, ifmQuantization, ifm2Quantization, ofmQuantization, dtype, ifmShape, ifm2Shape);
}

inline Operation *CreateShl(const std::shared_ptr<Tensor> &ifm, const std::shared_ptr<Tensor> &ifm2,
    const Quantization &ifmQuantization, const Quantization &ifm2Quantization, const Quantization &ofmQuantization,
    DataType dtype = DataType::None, const Shape *ifmShape = nullptr, const Shape *ifm2Shape = nullptr)
{
    return CreateBinaryElementwise(OpType::SHL, ifm, ifm2, ifmQuantization, ifm2Quantization, ofmQuantization, dtype, ifmShape, ifm2Shape);
}

inline Operation *CreateAsr(const std::shared_ptr<Tensor> &ifm, const std::shared_ptr<Tensor> &ifm2,
    const Quantization &ifmQuantization, const Quantization &ifm2Quantization, const Quantization &ofmQuantization,
    DataType dtype = DataType::None, const Shape *ifmShape = nullptr, const Shape *ifm2Shape = nullptr)
{
    return CreateBinaryElementwise(OpType::Asr, ifm, ifm2, ifmQuantization, ifm2Quantization, ofmQuantization, dtype, ifmShape, ifm2Shape);
}

inline Operation *CreateRescaleAdd(const std::shared_ptr<Tensor> &ifm, const std::shared_ptr<Tensor> &ifm2, const Quantization &ifmQuantization,
    const Quantization &ifm2Quantization, const Quantization &ofmQuantization, int32_t scale, int shift)
{
    auto op = CreateBinaryElementwise(OpType::Add, ifm, ifm2, ifmQuantization, ifm2Quantization, ofmQuantization);
    op->Output(TensorUsage::OFM)->quantization.scales.push_back(QuantizedScale(scale, shift));
    return op;
}

inline TransposeType CalculateTransposeType(const Operation &operation)
{
    const auto *paramsConn = operation.Input(TensorUsage::Params);
    assert(paramsConn);
    // We can only handle permutation vectors up 8 elements
    if ( paramsConn->shape.Depth() > 8 ) throw std::invalid_argument("Permutation vector has more than 8 elements");
    // We can only handle constant permutation vectors
    if ( !paramsConn->tensor->IsConstant() ) throw std::invalid_argument("Permutation vector is non-constant");
    Shape perm = TensorToShape(paramsConn->tensor.get(), paramsConn->shape.Depth(), 1, 0);
    return TransposeTypeFromShape(perm);
}

// Is the scaling of Tensor connection a and b valid and equal.
inline bool IsScalingValidAndEqual(const TensorConnection &a, const TensorConnection &b)
{
    return (a.quantization.IsValid() && b.quantization.IsValid() && a.quantization.scales == b.quantization.scales &&
            a.quantization.zeroPoints == b.quantization.zeroPoints);
}

#undef FOR_ALL_INT_TYPES

inline ArchFM &Set(ArchFM &fm, const Tensor *src)
{
    if ( src )
    {
        fm.type = src->Type();
        fm.shape = src->StorageShape();
    }
    return fm;
}

}  // namespace regor
