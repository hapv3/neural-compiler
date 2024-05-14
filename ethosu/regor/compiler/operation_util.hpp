//
// SPDX-FileCopyrightText: Copyright 2021-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "common/buffer_view.hpp"
#include "operation.hpp"
#include "quantization.hpp"
#include "tensor.hpp"

namespace regor
{

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

template<typename T>
std::shared_ptr<Buffer> NewBufferFromView(const BufferView &src)
{
    auto size = src.ViewShape().Elements();
    auto buf = std::make_shared<Buffer>(size, static_cast<T *>(nullptr));
    auto bufView = BufferView(buf, src);
    const auto srcData = src.Values<T>();
    auto dstData = bufView.template WritableValues<T>();
    for ( int i = 0; i < size; i++ )
    {
        dstData[i] = srcData[i];
    }
    return buf;
}

inline std::shared_ptr<Tensor> CreateSliceCopy(const std::string &name, const Tensor *src, const TensorSlice &slice)
{
    assert(src->IsConstant());
    auto sliceView = src->View().SubView(slice.offset, slice.shape);
    std::shared_ptr<Buffer> buffer;
    switch ( src->Type() )
    {
        case DataType::Int8:
            buffer = NewBufferFromView<int8_t>(sliceView);
            break;
        case DataType::UInt8:
            buffer = NewBufferFromView<uint8_t>(sliceView);
            break;
        case DataType::Int16:
            buffer = NewBufferFromView<int16_t>(sliceView);
            break;
        case DataType::Int32:
            buffer = NewBufferFromView<int32_t>(sliceView);
            break;
        default:
            assert(false);
            break;
    }
    auto tensor = CreateConstTensor(name, src->Type(), buffer, &slice.shape);
    return tensor;
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
    auto ofm = std::make_shared<Tensor>(ifm->Name() + "/reducesum", DataType::Int32);
    ofm->SetStorageShape(Shape(1, ifmShape.Height(), ifmShape.Width(), 1));
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
    if ( ifm2 )
    {
        op->ConnectInput(TensorUsage::IFM1, ifm2).Set(ifm2Quantization);
        if ( ifm2Shape ) op->Input(TensorUsage::IFM1)->shape = *ifm2Shape;
    }

    if ( dtype == DataType::None ) dtype = ifm->Type();

    Shape ofmShape = op->Input(TensorUsage::IFM)->shape;
    // If reverse operands use ifm2 shape as ofm shape
    if ( ifm2 && ((ofmShape.Elements() == 1 && ifm->IsConstant()) || ofmShape.IsSubShapeOf(op->Input(TensorUsage::IFM1)->shape)) )
    {
        ofmShape = ifm2->StorageShape();
    }

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

}  // namespace regor
