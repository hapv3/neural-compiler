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

#include "compiler/graphir_optimiser.hpp"

#include "operation_util.hpp"
#include "optimiser_utils.hpp"
#include "tflite/tflite_schema_generated.hpp"


namespace regor
{

using namespace GraphOptimisation;

Tensor *GraphIrOptimiser::ConvertInt48Tensors(Graph *, Tensor *tensor)
{
    if ( tensor->Type() == DataType::Int48 || tensor->Type() == DataType::UInt48 )
    {
        if ( tensor->IsConstant() )
        {
            // Unpack 48-bit to 64-bit values
            const auto values = tensor->View().Values<int48_t, int64_t>();
            std::vector<int64_t> unpackedValues(values.begin(), values.end());
            // Replace the tensor's buffer with the new buffer containing the 64-bit values
            tensor->SetBuffer(nullptr);
            tensor->ChangeType(DataType::Int64);
            tensor->SetBuffer(std::make_shared<Buffer>(std::move(unpackedValues)));
        }
        else
        {
            tensor->ChangeType(IsSignedInteger(tensor->Type()) ? DataType::Int64 : DataType::UInt64);
        }
    }
    return tensor;
}

// The internal boolean representation is -1 (true) and 0 (false). To be able to handle the TFLite representation 1
// (true) and 0 (false), or the TOSA representation non-zero (true) and 0 (false), we need to insert ops that converts
// graph inputs/outputs.
Tensor *GraphIrOptimiser::ConvertBool8Tensors(Graph *graph, Tensor *tensor)
{
    Tensor *returnTensor = tensor;
    if ( tensor->Type() == DataType::Bool8 )
    {
        if ( !tensor->StorageShape() )
        {
            // don't convert shapeless tensors
            return returnTensor;
        }
        if ( tensor->IsConstant() )
        {
            const auto oldView = tensor->View();
            const auto oldValues = oldView.RawData<int8_t>();
            const auto size = oldView.Buffer()->Size();

            // Replace this tensor's buffer with a new buffer since we don't know if the current buffer is writable
            auto newValues = std::make_unique<uint8_t[]>(size);
            for ( int i = 0; i < size; i++ )
            {
                // Convert each element to the internal representation -1 (true) and 0 (false)
                newValues[i] = oldValues[i] == 0 ? 0 : -1;
            }
            tensor->SetBuffer(std::make_shared<Buffer>(std::move(newValues), size));
        }
        else if ( graph->IsInput(tensor) )
        {
            // Replace the IFM of ops consuming the graph input tensor
            std::shared_ptr<Tensor> graphInputTensor = tensor->shared_from_this();
            std::shared_ptr<Tensor> newTensor = tensor->Clone();
            newTensor->SetBuffer(nullptr);
            newTensor->SetName(newTensor->Name() + "_int8");
            ReplaceConsumerInput(nullptr, graphInputTensor->Readers(), graphInputTensor.get(), newTensor);

            // Create and insert an elementwise CMP_NE to convert to internal bool representation
            auto newOp = std::make_shared<Operation>(OpType::NotEqual);
            newOp->ConnectInput(TensorUsage::IFM0, graphInputTensor);
            newOp->ConnectInput(TensorUsage::IFM1, CreateConstTensor("const_zero", bool(false)));
            newOp->ConnectOutput(TensorUsage::OFM, newTensor);
            RecordOptimisation(graph, newOp.get());
            returnTensor = graphInputTensor.get();
        }
        else if ( graph->IsOutput(tensor) )
        {
            // Replace the OFM of ops producing the graph output tensor
            std::shared_ptr<Tensor> newTensor = tensor->Clone();
            newTensor->SetBuffer(nullptr);
            newTensor->SetName(newTensor->Name() + "_int8");
            std::shared_ptr<Tensor> graphOutputTensor = tensor->shared_from_this();
            ReplaceProducerOutput(graphOutputTensor->Writers(), graphOutputTensor.get(), newTensor);

            // Create and insert an elementwise BITWISE_AND to convert from internal bool representation
            auto newOp = std::make_shared<Operation>(OpType::And);
            newOp->ConnectInput(TensorUsage::IFM0, newTensor);
            newOp->ConnectInput(TensorUsage::IFM1, CreateConstTensor("const_one", bool(true)));
            newOp->ConnectOutput(TensorUsage::OFM, graphOutputTensor);
            RecordOptimisation(graph, newOp.get());
            returnTensor = newTensor.get();
        }
    }
    return returnTensor;
}

Tensor *GraphIrOptimiser::ConvertInt4Tensors(Graph *graph, Tensor *tensor)
{
    if ( tensor->Type() == DataType::Int4Packed8 )
    {
        if ( tensor->IsConstant() )
        {
            const auto oldView = tensor->View();
            const auto oldValues = oldView.RawData<uint8_t>();
            const auto newSize = oldView.Buffer()->Size() * 2;

            auto data = std::make_unique<uint8_t[]>(newSize);
            for ( int i = 0; i < newSize; i++ )
            {  // Convert each element to Int8
                const auto &nibbles = oldValues[i >> 1];
                uint8_t val = i & 1 ? (nibbles & 0xF0) >> 4 : nibbles & 0x0F;
                data[i] = val > 7 ? val - 16 : val;
            }

            tensor->SetBuffer(nullptr);
            tensor->ChangeType(DataType::Int8);
            // Replace this tensor's buffer with a new buffer
            tensor->SetBuffer(std::make_shared<Buffer>(std::move(data), newSize));
        }
        else
        {
            tensor->ChangeType(DataType::Int8);
        }
    }
    return tensor;
}

Operation *GraphIrOptimiser::ConvertAttributes(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    OpType opType = operation->Type();
    if ( opType == OpType::Asr )
    {
        const auto *attr = operation->Attribute<asr_attr_t>();
        auto roundMode = attr->round ? RoundMode::NATURAL : RoundMode::TRUNCATE_TO_LOWER;
        operation->Output(TensorUsage::OFM)->Set(roundMode);
    }
    else if ( opType == OpType::Rescale )
    {
        const auto *attr = operation->Attribute<rescale_attr_t>();
        auto roundMode = attr->double_round ? RoundMode::DBL : RoundMode::NATURAL;
        operation->Output(TensorUsage::OFM)->Set(roundMode);
    }
    else if ( opType == OpType::Clamp )
    {
        const auto *attr = operation->Attribute<clamp_attr_t>();
        TensorConnection *ofmConn = operation->Output(TensorUsage::OFM);
        ofmConn->quantization.quantMin = {int64_t(attr->min)};
        ofmConn->quantization.quantMax = {int64_t(attr->max)};
    }
    else if ( opType == OpType::SHL || opType == OpType::SHR )
    {
        TensorConnection *ofmConn = operation->Output(TensorUsage::OFM);
        ofmConn->quantization.quantMin = {std::numeric_limits<int64_t>::min()};
        ofmConn->quantization.quantMax = {std::numeric_limits<int64_t>::max()};
    }
    else if ( opType == OpType::Mul )
    {
        auto *attr = operation->Attribute<mul_attr_t>();
        TensorConnection *ofmConn = operation->Output(TensorUsage::OFM);
        // A non-zero shift attribute is only supported with explicit quantization
        assert(attr->shift == 0 || ofmConn->quantization.type == QuantizationType::EXPLICIT);
        if ( !ofmConn->quantization.scales.size() )
        {
            ofmConn->quantization.scales.push_back({1, 0});
        }
        // Move shift attribute to OFM quantization
        ofmConn->quantization.scales[0].shift += attr->shift;
        attr->shift = 0;
    }
    else if ( opType == OpType::Reverse )
    {
        // Convert TOSA axis attribute to ReverseType representation
        TensorConnection *ofmConn = operation->Output(TensorUsage::OFM);
        int ofmRank = ofmConn->shape.Size();
        const auto *attr = operation->Attribute<axis_attr_t>();
        auto mask = ToReverseMask({attr->axis}, ofmRank);
        assert(mask != ReverseType::Dynamic && "Unexpected dynamic reverse axis.");
        assert((mask == ReverseType::None || IsPowerOfTwo(unsigned(mask))) && "Reverse operation can only have one axis");
        ofmConn->reverse = mask;
    }
    else if ( opType == OpType::ReduceMin || opType == OpType::ReduceMax || opType == OpType::ReduceAny || opType == OpType::ReduceAll )
    {
        TensorConnection *ifmConn = operation->Input(TensorUsage::IFM);
        auto *attr = operation->Attribute<axis_attr_t>();
        auto axis = attr->axis;
        if ( axis < 0 ) axis = ifmConn->shape.Size() + axis;
        assert(axis >= 0);
        assert(axis < ifmConn->shape.Size());
        // Create a reduce kernel, if reducing in H or W
        Kernel kernel = *operation->Kernel();
        if ( axis == ifmConn->shape.Size() - 3 )
            kernel = operation->Kernel()->WithSize({1 /* W */, ifmConn->shape.Height() /* H */});
        else if ( axis == ifmConn->shape.Size() - 2 )
            kernel = operation->Kernel()->WithSize({ifmConn->shape.Width() /* W */, 1 /* H */});
        operation->SetKernel(std::make_unique<Kernel>(std::move(kernel)));
    }

    return operation;
}

Operation *GraphIrOptimiser::ConvertAttributeTensors(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    OpType opType = operation->Type();
    if ( opType == OpType::Mul )
    {
        auto *attr = operation->Attribute<mul_attr_t>();
        // Shift can be a compile time constant tensor
        if ( auto *shiftConn = operation->Input(TensorUsage::Params) )
        {
            assert(shiftConn->tensor->IsConstant());
            attr->shift = Scalar<int32_t>(*shiftConn->tensor);
        }
    }
    else if ( opType == OpType::Pad )
    {
        auto *attr = operation->Attribute<pad_attr_t>();
        // Pad value can be a compile time constant tensor
        if ( auto padConstConn = operation->Input(TensorUsage::Params1) )
        {
            assert(padConstConn->tensor->IsConstant());
            attr->pad_const = Scalar<double>(*padConstConn->tensor);
        }
    }
    else if ( opType == OpType::Slice )
    {
        auto *attr = operation->Attribute<slice_attr_t>();
        // Start shape can be a compile time constant tensor
        if ( auto startConn = operation->Input(TensorUsage::Params0) )
        {
            assert(startConn->tensor->IsConstant());
            attr->begin = TensorToShape(startConn->tensor.get(), startConn->shape.Elements());
        }
        // Size shape can be a compile time constant tensor
        if ( auto sizeConn = operation->Input(TensorUsage::Params1) )
        {
            assert(sizeConn->tensor->IsConstant());
            attr->size = TensorToShape(sizeConn->tensor.get(), sizeConn->shape.Elements());
        }
    }
    else if ( opType == OpType::Resize )
    {
        auto *attr = operation->Attribute<resize_attr_t>();
        // Scale can be a compile time constant tensor
        if ( const auto scaleConn = operation->Input(TensorUsage::Params0) )
        {
            assert(scaleConn->tensor->IsConstant());
            auto scale = TensorToShape(scaleConn->tensor.get(), scaleConn->shape.Elements());
            attr->scaleY.n = scale[0];
            attr->scaleY.d = scale[1];
            attr->scaleX.n = scale[2];
            attr->scaleX.d = scale[3];
        }
        // Offset can be a compile time constant tensor
        if ( const auto offsetConn = operation->Input(TensorUsage::Params1) )
        {
            assert(offsetConn->tensor->IsConstant());
            auto offset = TensorToShape(offsetConn->tensor.get(), offsetConn->shape.Elements());
            attr->offset.x = offset[1];
            attr->offset.y = offset[0];
        }
        // Border can be a compile time constant tensor
        if ( const auto borderConn = operation->Input(TensorUsage::Params2) )
        {
            assert(borderConn->tensor->IsConstant());
            auto border = TensorToShape(borderConn->tensor.get(), borderConn->shape.Elements());
            attr->border.x = border[1];
            attr->border.y = border[0];
        }
    }

    return operation;
}

Operation *GraphIrOptimiser::ConvertResizeOffsets(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    // Reduce positive offset parameters that are larger than scale_n
    // If offset >= scale_n, we can create an ifm-slice to start on offset/scale_n.
    // The offset parameters are updated to the remainder of the fraction.
    Operation *returnOp = operation;
    OpType opType = operation->Type();
    if ( opType == OpType::Resize )
    {
        auto *attr = operation->Attribute<resize_attr_t>();
        TensorConnection *ifmConn = operation->Input(TensorUsage::IFM);
        Shape ifmStart = ifmConn->shape.WithZeros();
        Shape ifmShape = ifmConn->shape;
        int offset_h = attr->offset.y;
        int offset_w = attr->offset.x;
        int scale_nh = attr->scaleY.n;
        int scale_nw = attr->scaleX.n;
        if ( offset_h >= scale_nh )
        {
            ifmStart[1] += offset_h / scale_nh;
            ifmShape[1] -= ifmStart[1];
            attr->offset.y = offset_h % scale_nh;
        }
        if ( offset_w >= scale_nw )
        {
            ifmStart[2] += offset_w / scale_nw;
            ifmShape[2] -= ifmStart[2];
            attr->offset.x = offset_w % scale_nw;
        }
        TensorSlice slice{std::move(ifmStart), std::move(ifmShape)};
        ifmConn->Set(slice);
    }
    return returnOp;
}

template<typename T>
static T Scale(int64_t v, const Quantization &quant, RoundMode rounding = RoundMode::AUTO, int64_t ofmZp = 0, int doubleRound = 0)
{
    assert(doubleRound >= 0 && doubleRound < 31 && "Illegal double round");
    if ( !quant.IsUnitScale() )
    {
        // TODO MLBEDSW-11390: Support per-channel quantization
        const auto &qs = quant.scales.front();
        assert(qs.shift >= 0 && qs.shift <= 63);
        int64_t round = 1ll << (qs.shift - 1);                                            // Natural round
        const int64_t D = (qs.shift > 31 - doubleRound) ? 1ll << (30 - doubleRound) : 0;  // Double round
        switch ( rounding )
        {
            case RoundMode::AUTO:
            case RoundMode::NATURAL:
                break;
            case RoundMode::SYMMETRIC:
                if ( v < 0 ) round -= 1;
                break;
            case RoundMode::TRUNCATE:
                round = v >= 0 ? 0 : (1ll << qs.shift) - 1;
                break;
            case RoundMode::TRUNCATE_TO_LOWER:
                round = 0;
                break;
            case RoundMode::DBL:
                round += (v >= 0 ? D : -D);
                break;
            case RoundMode::DOUBLE_ASYMMETRIC:
                round += D;
                break;
            default:
                assert(false && "Unsupported rounding");
        }
        v *= qs.scale;
        v = (v + round) >> qs.shift;
        v = v + ofmZp;
        if ( quant.quantMin.size() && quant.quantMax.size() )
        {
            v = std::clamp<int64_t>(v, quant.quantMin[0], quant.quantMax[0]);
        }
    }
    return T(v);
}

template<typename T>
struct EwShl
{
    int64_t operator()(T a, T b)
    {
        assert(b >= 0);
        return int64_t(uint64_t(a) << std::make_unsigned_t<T>(b));
    }
};

template<typename T>
struct EwMul
{
    int64_t operator()(T a, T b) { return int64_t(a) * int64_t(b); }
};

template<typename T>
struct EwMemoryCopy
{
    int64_t operator()(T a) { return int64_t(a); }
};

template<typename T>
static std::vector<T> BroadcastValues(const Tensor *in, const Shape &iShape, const Shape &oShape)
{
    const auto &iData = in->View().Values<T>(in->Type());
    const int elementCnt = oShape.Elements();

    std::vector<T> ret(elementCnt);
    auto opos = oShape.WithZeros();
    auto ipos = opos;

    auto posIncr = [&]()
    {
        for ( int i = opos.Size() - 1; i >= 0; i-- )
        {
            opos[i]++;
            if ( iShape[i] == oShape[i] )
            {
                ipos[i]++;
            }

            if ( opos[i] < oShape[i] )
            {
                return false;
            }

            opos[i] = 0;
            ipos[i] = 0;
        }
        return true;
    };

    for ( int i = 0; i < elementCnt; i++ )
    {
        ret[i] = iData[ipos];
        bool done = posIncr();
        UNUSED(done);
        assert(done == (i == (elementCnt - 1)));
    }

    return ret;
}

// Compute the output for a binary elementwise operation with constant inputs
template<template<typename> typename F, typename T>
std::shared_ptr<Buffer> ConstPropBinEw(Operation *const operation)
{
    auto ifmConn0 = operation->Input(TensorUsage::IFM);
    auto ifmConn1 = operation->Input(TensorUsage::IFM1);
    auto ofmConn = operation->Output(TensorUsage::OFM);
    const auto &oShape = ofmConn->SliceShape();
    const auto &iShape0 = ifmConn0->SliceShape();
    const auto &iShape1 = ifmConn1->SliceShape();
    auto *ifm0 = ifmConn0->tensor.get();
    auto *ifm1 = ifmConn1->tensor.get();
    auto *ofm = ofmConn->tensor.get();

    auto v0 = BroadcastValues<T>(ifm0, iShape0, oShape);
    auto v1 = BroadcastValues<T>(ifm1, iShape1, oShape);
    std::vector<T> c(oShape.Elements());

    for ( int i = 0; i < oShape.Elements(); i++ )
    {
        c[i] = Scale<T>(F<T>()(v0[i], v1[i]), ofmConn->quantization, ofmConn->rounding);
    }

    return std::make_shared<Buffer>(std::move(c));
}

// Compute the output for a binary elementwise operation with constant inputs
template<template<typename> typename F>
std::shared_ptr<Buffer> ConstPropBinEw(Operation *const operation)
{
    auto dataType = operation->Output(TensorUsage::OFM)->tensor->Type();

    switch ( dataType )
    {
        case DataType::Int8:
            return ConstPropBinEw<F, int8_t>(operation);
        case DataType::Int16:
            return ConstPropBinEw<F, int16_t>(operation);
        case DataType::Int32:
            return ConstPropBinEw<F, int32_t>(operation);
        default:
            return {};
    }
}

// Compute the output for a unary elementwise operation with constant input
template<template<typename> typename F, typename T>
std::shared_ptr<Buffer> ConstPropUnEw(Operation *const operation)
{
    auto ifmConn = operation->Input(TensorUsage::IFM);
    auto ofmConn = operation->Output(TensorUsage::OFM);
    const auto &oShape = ofmConn->SliceShape();
    const auto &iShape = ifmConn->SliceShape();
    auto *ifm0 = ifmConn->tensor.get();
    auto *ofm = ofmConn->tensor.get();

    assert(ofmConn->quantization.zeroPoints.size() <= 1);
    const auto zpIfm = ifmConn->quantization.zeroPoints.size() ? ifmConn->quantization.zeroPoints[0] : 0;
    const auto zpOfm = ofmConn->quantization.zeroPoints.size() ? ofmConn->quantization.zeroPoints[0] : 0;

    auto v0 = BroadcastValues<T>(ifm0, iShape, oShape);
    std::vector<T> c(oShape.Elements());

    for ( int i = 0; i < oShape.Elements(); i++ )
    {
        c[i] = Scale<T>(F<T>()(v0[i] - zpIfm), ofmConn->quantization, ofmConn->rounding, zpOfm);
    }

    return std::make_shared<Buffer>(std::move(c));
}

// Compute the output for a unary elementwise operation with constant input
template<template<typename> typename F>
std::shared_ptr<Buffer> ConstPropUnEw(Operation *const operation)
{
    auto dataType = operation->Output(TensorUsage::OFM)->tensor->Type();

    switch ( dataType )
    {
        case DataType::Int8:
            return ConstPropUnEw<F, int8_t>(operation);
        case DataType::Int16:
            return ConstPropUnEw<F, int16_t>(operation);
        case DataType::Int32:
            return ConstPropUnEw<F, int32_t>(operation);
        default:
            return {};
    }
}

Operation *GraphIrOptimiser::ConstPropagation(Graph *const graph, Operation *const operation)
{
    Operation *returnOp = operation;

    for ( auto [usage, ifmConn] : operation->Inputs().pairs() )
    {
        if ( !IsIFM(usage) ) continue;

        if ( !ifmConn.tensor->IsConstant() || !ifmConn.quantization.IsUnitScale() )
        {
            return operation;
        }
    }

    auto *ofmConn = operation->Output(TensorUsage::OFM);
    if ( ofmConn->quantization.type != QuantizationType::EXPLICIT )
    {
        // TODO: Remove this restriction when MLBEDSW-10086 is implemented
        return operation;
    }

    // Don't try to remove memory copy operations that produce graph output, regardless if it has constant input or not.
    if ( operation->Type() == OpType::MemoryCopy && graph->IsOutput(ofmConn->tensor.get()) )
    {
        return operation;
    }

    // Op has only constant input and result can be computed
    std::shared_ptr<Buffer> ofmBuf;
    switch ( operation->Type() )
    {
        case OpType::SHL:
            ofmBuf = ConstPropBinEw<EwShl>(operation);
            break;
        case OpType::Mul:
            ofmBuf = ConstPropBinEw<EwMul>(operation);
            break;
        case OpType::MemoryCopy:
            ofmBuf = ConstPropUnEw<EwMemoryCopy>(operation);
            break;
        case OpType::Quantize:
            ofmBuf = ConstPropUnEw<EwMemoryCopy>(operation);
            break;
        default:
            break;
    }

    if ( ofmBuf )
    {
        auto *ofm = ofmConn->tensor.get();
        if ( graph->IsOutput(ofm) )
        {
            // Don't propagate the computed result to the graph output. If we do that and remove this operation, we will
            // not have any operation left in the graph that writes the graph output. Instead, replace it with a memory
            // copy.
            auto constant = std::make_shared<Tensor>("constprop", ofm->Type(), ofm->StorageShape(), ofmBuf);
            auto copy = std::make_shared<Operation>(OpType::MemoryCopy);
            copy->ConnectInput(TensorUsage::IFM, constant);
            copy->ConnectOutput(TensorUsage::OFM, ofmConn->tensor);
            returnOp = copy.get();
            RecordOptimisation(*operation, copy.get());
            operation->Disconnect();
        }
        else
        {
            ofm->SetBuffer(ofmBuf);

            // Remove op from ifm readers and ofm writers.
            // Note the Inputs/Outputs on operation should still be intact to not break the traversal.
            for ( auto [usage, ifmConn] : operation->Inputs().pairs() )
            {
                ifmConn.tensor->RemoveReader(operation->shared_from_this());
            }
            ofm->RemoveWriter(operation->shared_from_this());
        }
    }

    return returnOp;
}

/*
 * This pass replaces the Const operator with an Identity operator (to be removed in RemoveReshape) and moves the
 * "values" attribute tensor to an input tensor instead to enable us to treat it as a normal reshape-like operator.
 */
Operation *GraphIrOptimiser::RewriteConst(Graph *const graph, Operation *const operation)
{
    Operation *returnOp = operation;
    OpType opType = operation->Type();
    if ( opType == OpType::Const )
    {
        const auto *ofmConn = operation->Output(TensorUsage::OFM);
        // Clone tensor to create input tensor with the constant values and remove constant values from output
        std::shared_ptr<Tensor> constIfm = ofmConn->tensor->Clone();
        constIfm->SetName("const_values");
        ofmConn->tensor->SetBuffer(nullptr);

        // Create new identity operator (to be removed in RemoveReshape) and set constant values as input
        auto identityOp = std::make_shared<Operation>(OpType::Identity);
        identityOp->ConnectInput(TensorUsage::IFM0, constIfm);
        identityOp->CopyOutput(TensorUsage::OFM, *ofmConn);

        returnOp = identityOp.get();
        RecordOptimisation(*operation, returnOp);
        operation->Disconnect();
    }
    return returnOp;
}

Operation *GraphIrOptimiser::RewriteIdentityResize(Graph *const graph, Operation *const operation)
{
    Operation *returnOp = operation;
    OpType opType = operation->Type();
    if ( opType == OpType::Resize )
    {
        auto *attr = operation->Attribute<resize_attr_t>();
        const auto ofmConn = operation->Output(TensorUsage::OFM);
        bool noBorder = attr->border.x == 0 && attr->border.y == 0;
        bool noOffset = attr->offset.x == 0 && attr->offset.y == 0;
        bool unitUpscale = attr->scaleX.n == 1 && attr->scaleX.d == 1 && attr->scaleY.n == 1 && attr->scaleY.d == 1;
        bool identityUpscaleNoRescaleRequired =
            attr->mode == tosa::ResizeMode::NEAREST && attr->scaleX.n == attr->scaleX.d &&
            attr->scaleY.n == attr->scaleY.d;

        bool isIdentity = noBorder && noOffset && (unitUpscale || identityUpscaleNoRescaleRequired);
        if ( isIdentity )
        {
            const auto ifmConn = operation->Input(TensorUsage::IFM);

            auto identityOp = std::make_shared<Operation>(OpType::MemoryCopy);
            identityOp->ConnectInput(TensorUsage::IFM, ifmConn->tensor).Set(Quantization::Unit()).Set(ifmConn->tensor->StorageShape());
            identityOp->ConnectOutput(TensorUsage::OFM, ofmConn->tensor).Set(Quantization::Unit()).Set(ofmConn->tensor->StorageShape());


            returnOp = identityOp.get();
            RecordOptimisation(*operation, returnOp);
            operation->Disconnect();
        }
    }
    return returnOp;
}

Operation *GraphIrOptimiser::RewriteFullyConnected(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    OpType opType = operation->Type();
    auto ifm = operation->Input(TensorUsage::IFM0);
    const auto kernel = operation->Kernel();

    // Batched Conv2D with kernel 1x1 can be handled the same way as FullyConnected
    if ( opType == OpType::FullyConnected ||
         (opType == OpType::Conv2D && ifm->shape.Batch() > 1 && kernel->Size().AreaXY() == 1 &&
             kernel->Stride().AreaXY() == 1 && kernel->DilatedWH().AreaXY() == 1 && kernel->Padding().IsZero()) )
    {
        const auto &weights = operation->Input(TensorUsage::Weights);
        const auto &scales = operation->Input(TensorUsage::Scales);
        if ( !weights->tensor->IsConstant() || !scales->tensor->IsConstant() )
        {
            // Do not rewrite if bias or weights are non-constant
            return returnOp;
        }

        const auto &shape = weights->tensor->StorageShape();
        if ( shape.Size() == 2 )
        {
            // Reshape weight tensor from (num_outputs, ..., num_inputs) to (num_outputs, 1, 1, num_inputs)
            weights->tensor->Reshape(Shape(shape[0], 1, 1, shape[-1]));
        }

        // Rewrite input shape to batched shape
        auto nInElems = weights->shape.Depth();
        auto &ifmShape = ifm->slice.shape.IsEmpty() ? ifm->shape : ifm->slice.shape;
        auto elems = ifmShape.Elements();
        auto batchSize = elems / nInElems;
        assert(batchSize * nInElems == elems);
        ifmShape = Shape(batchSize, 1, 1, nInElems);

        // Check if the first dimension indicates batching
        int n = ifmShape.Batch();
        if ( n > 1 )
        {
            // More square H/W gives better performance up to a point
            int w = std::max(n / 16, int(std::ceil(std::sqrt(n))));
            while ( n % w != 0 )
                w++;
            int h = n / w;

            ifmShape = Shape(1, h, w, ifmShape.Depth());
            auto ofm = operation->Output(TensorUsage::OFM);
            ofm->shape = Shape(1, h, w, ofm->shape.Depth());
        }
    }
    return returnOp;
}

/*
 * Lower Rescale into one (or more) 32-bit elementwise MUL operations.
 * Multipliers are moved to a constant-tensor, while the shift value is keps as ofm-quantization
 *
 *                            Cast to 32-bit (if necessary)
 *                                     |
 *         IFM                   IFM (32-bit)  Multipliers (32-bit)
 *          |                             \   /
 *       Rescale           --->            MUL
 *          |                               |
 *         OFM                             OFM
 *
 * Global-scaling (one global multiplier):
 *      Converted into one MUL operation
 *
 * Per-Channel scaling (one multiplier per channel):
 *      The algorithm will attempt to adjust scales to a common shift representation
 *      to pack consecutive channels into the same MUL operation.
 *      This can be done as long as the adjustment can be made without precision-loss.
 *      Worst-case, per-channel scaling is handled with one MUL-operation per channel
 */
Operation *GraphIrOptimiser::RewriteRescale(Graph *const, Operation *const operation)
{
    Operation *returnOp = operation;
    OpType opType = operation->Type();
    if ( opType == OpType::Rescale )
    {
        const auto &ifmConn = operation->Input(TensorUsage::IFM0);
        const auto &ofmConn = operation->Output(TensorUsage::OFM);
        const Quantization &quant = ofmConn->quantization;
        DataType ifmType = ifmConn->tensor->Type();
        DataType ofmType = ofmConn->tensor->Type();
        const auto rescaleAttr = operation->Attribute<rescale_attr_t>();
        auto signAttr = operation->Attribute<sign_attr_t>();
        if ( signAttr->input_unsigned )
        {
            ifmType = ifmType & ~unsigned(DataType::Signed);
        }
        if ( signAttr->output_unsigned )
        {
            ofmType = ofmType & ~unsigned(DataType::Signed);
        }
        if ( ifmType != DataType::Int32 && !_constraints->SupportsRescale(ifmType, ofmType) )
        {
            // create cast op to convert to 32-bit ifm
            if ( ifmConn->tensor->Type() != DataType::Int32 )
            {
                auto castOp = std::make_shared<Operation>(OpType::Cast);
                auto ifm32Tens = std::make_shared<Tensor>(ifmConn->tensor->Name() + "_int32", DataType::Int32, ifmConn->shape);

                // move zero point to cast input
                auto castInQuant = Quantization::Unit();
                castInQuant.zeroPoints = ifmConn->quantization.zeroPoints;
                ifmConn->quantization.zeroPoints.clear();
                ifmConn->quantization.zeroPoints.push_back(0);

                // connect cast before the rescale
                castOp->ConnectInput(TensorUsage::IFM, ifmConn->tensor).Set(ifmConn->shape).Set(castInQuant);
                castOp->ConnectOutput(TensorUsage::OFM, ifm32Tens);

                // move input_unsigned to cast input
                auto castAttr = castOp->Attribute<sign_attr_t>();
                castAttr->input_unsigned = signAttr->input_unsigned;
                signAttr->input_unsigned = false;

                RecordOptimisation(*operation, castOp.get());
                operation->ConnectInput(TensorUsage::IFM, ifm32Tens);
                ifmType = DataType::Int32;
            }
        }
        if ( ifmType == DataType::Int32 && !_constraints->SupportsRescale(ifmType, ofmType) )
        {
            auto CreateRescalingMul = [ifmConn, ofmConn](int startChannel, int endChannel, std::vector<int32_t> &scales, int shift)
            {
                Shape sliceOffset = ifmConn->shape.WithZeros().WithDepth(startChannel);
                Shape sliceShape = ifmConn->shape.WithDepth(endChannel - startChannel);
                TensorSlice slice{sliceOffset, sliceShape};

                auto mulOp = std::make_shared<Operation>(OpType::Mul);
                auto buf = std::make_shared<Buffer>(scales.size(), scales.data());
                auto scaleTensor = CreateConstTensor(fmt::format("multipliers_{}_{}", startChannel, endChannel - 1), DataType::Int32, buf);

                Quantization scaleQuant = Quantization::Unit();
                scaleQuant.type = QuantizationType::EXPLICIT;

                Quantization ifmQuant = ifmConn->quantization;
                ifmQuant.scales.clear();
                ifmQuant.scales.push_back({1, 0});
                ifmQuant.type = QuantizationType::EXPLICIT;

                Quantization ofmQuant = ofmConn->quantization;
                ofmQuant.scales.clear();
                ofmQuant.scales.push_back({1, shift});
                ofmQuant.type = QuantizationType::EXPLICIT;

                mulOp->ConnectInput(TensorUsage::IFM1, scaleTensor);
                mulOp->CopyInput(TensorUsage::IFM0, *ifmConn);
                mulOp->CopyOutput(TensorUsage::OFM, *ofmConn);
                mulOp->Output(TensorUsage::OFM)->Set(ofmConn->rounding);

                mulOp->Input(TensorUsage::IFM1)->Set(scaleQuant);
                mulOp->Input(TensorUsage::IFM0)->Set(ifmQuant).Set(slice);
                mulOp->Output(TensorUsage::OFM)->Set(ofmQuant).Set(slice);
                return mulOp;
            };

            // Use the first channels shift-value as reference shift
            // try to adjust multipliers to pack as many consecutive channels in the same mul-operation
            int shift = quant.scales[0].shift;
            std::vector<int32_t> scales;
            int startChannel = 0;
            for ( auto qscale : quant.scales )
            {
                int shiftDiff = qscale.shift - shift;
                // Double-rounding with shift > 31 cannot be reduced as doing so would affect the significance of the
                // rounding.
                bool correctRounding = (ofmConn->rounding != RoundMode::DBL || qscale.shift <= 31 || shiftDiff == 0);
                // try to right-shift scale without precision-loss
                // This can be done if the scale is evenly divisible by 2^shiftdiff
                if ( (shiftDiff >= 0) && correctRounding && (qscale.scale % (1ULL << shiftDiff) == 0) )
                {
                    scales.push_back(qscale.scale >> shiftDiff);
                }
                else
                {
                    // Could not adjust the scale without precision loss.
                    // Create elementwise mul operation to handle all the previous scales
                    int endChannel = startChannel + scales.size();
                    auto mulOp = CreateRescalingMul(startChannel, endChannel, scales, shift);
                    auto mulAttr = mulOp->Attribute<sign_attr_t>();
                    mulAttr->output_unsigned = signAttr->output_unsigned;
                    RecordOptimisation(*operation, mulOp.get());

                    // reset scales and startChannel
                    startChannel = endChannel;
                    scales.clear();
                    scales.push_back(qscale.scale);

                    // update target shift to the current shift-value
                    shift = qscale.shift;
                }
            }

            // Emit the final mul operation (or the only one for global scaling)
            int endChannel = ifmConn->shape.Depth();
            auto mulOp = CreateRescalingMul(startChannel, endChannel, scales, shift);
            auto mulAttr = mulOp->Attribute<sign_attr_t>();
            mulAttr->output_unsigned = signAttr->output_unsigned;
            RecordOptimisation(*operation, mulOp.get());
            returnOp = mulOp.get();
            operation->Disconnect();
        }
    }
    return returnOp;
}

Operation *GraphIrOptimiser::MakeFillOperation(TensorConnection *const ofmConn, const Shape &ofmShape,
    const TensorSlice &ofmSlice, std::shared_ptr<Tensor> padTensor)
{
    auto fillOp = std::make_shared<Operation>(OpType::MemoryCopy);
    fillOp->ConnectInput(TensorUsage::IFM, padTensor).Set(ofmSlice.shape).Set(ofmConn->quantization);
    fillOp->CopyOutput(TensorUsage::OFM, *ofmConn);
    fillOp->Output(TensorUsage::OFM)->Set(ofmShape).Set(ofmSlice).Set(RoundMode::NATURAL);
    return fillOp.get();
}

// Tries to completely remove a PAD operator by using explicit padding.
// E.g. a PAD operation that pads 1, followed by a CONV with VALID padding and kernel size 3
// is rewritten such that the PAD is removed, and the CONV uses explicit padding.
// Converts tens1 -> PAD -> tens2 -> CONV to tens1 -> CONV
// This is the most efficient way to implement PAD, but cannot be done for all pad sizes.
Operation *GraphIrOptimiser::ReplacePadByExplicitPadding(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    const OpType opType = operation->Type();
    if ( IsConvolution(opType) && opType != OpType::TransposeConv2D && operation->Kernel()->Padding().IsZero() )
    {
        const auto &producers = operation->IFM(0)->Writers();
        if ( producers.size() != 1 )
        {
            // IFM has multiple producers
            return operation;
        }

        const auto &padOp = producers.front();
        if ( padOp->Type() != OpType::Pad || padOp->Attribute<pad_attr_t>()->pad_const != 0 )
        {
            // Not a pad or not padding with zeros
            return operation;
        }

        const auto padIfmConn = padOp->Input(TensorUsage::IFM0);
        const auto padOfmConn = padOp->Output(TensorUsage::OFM);
        const auto &padIfm = padIfmConn->tensor;
        const auto &padOfm = padOfmConn->tensor;
        if ( padIfm->Type() != padOfm->Type() || !IsScalingValidAndEqual(*padIfmConn, *padOfmConn) )
        {
            // Different data types or different scaling
            return operation;
        }

        const auto padParamConn = padOp->Input(TensorUsage::Params);
        const auto &padIfmShape = padIfmConn->SliceShape();
        const auto beforePad = TensorToShape(padParamConn->tensor.get(), padIfmShape.Size(), 2, 0);
        const auto afterPad = TensorToShape(padParamConn->tensor.get(), padIfmShape.Size(), 2, 1);
        if ( beforePad.WithHW(0, 0) != beforePad.WithZeros() || afterPad.WithHW(0, 0) != afterPad.WithZeros() )
        {
            // Pad in other dimensions than height and width
            return operation;
        }

        int top = beforePad.Height();
        int left = beforePad.Width();
        int bottom = afterPad.Height();
        int right = afterPad.Width();
        const auto &k = operation->Kernel();
        const auto &kwh = k->DilatedWH();
        auto CalcPadAfter = [](int inputSize, int stride, int filterSize, int padBefore, int padAfter) -> int
        {
            const int totalPadding = NeededTotalPadding(inputSize, stride, filterSize);
            // The bottom/right padding might need downward adjustment depending on stride/input size
            const int remainderDiff = padAfter % stride - (totalPadding - padBefore) % stride;
            return std::max(0, padAfter - remainderDiff - (remainderDiff >= 0 ? 0 : stride));
        };
        // Adjust the padding attributes of the convolution operator
        bottom = CalcPadAfter(padIfmShape.Height(), k->Stride().y, kwh.y, top, bottom);
        right = CalcPadAfter(padIfmShape.Width(), k->Stride().x, kwh.x, left, right);
        if ( left >= kwh.x || right >= kwh.x || top >= kwh.y || bottom >= kwh.y )
        {
            // Pad greater than or equal to kernel
            return operation;
        }

        const auto kernel = k->WithPadding({top, left, bottom, right});
        operation->SetKernel(std::make_unique<Kernel>(std::move(kernel)));
        operation->CopyInput(TensorUsage::IFM0, *padIfmConn);
        if ( padOfm->Readers().empty() )
        {
            // Bypass the PAD operator
            padOp->Disconnect();
        }
    }
    return operation;
}

Operation *GraphIrOptimiser::RewritePad(Graph *const, Operation *const operation)
{
    Operation *returnOp = operation;
    OpType opType = operation->Type();
    if ( opType == OpType::Pad )
    {
        const auto &ifmConn = operation->Input(TensorUsage::IFM0);
        const auto &ofmConn = operation->Output(TensorUsage::OFM);
        const Shape ofmShape = ofmConn->shape;
        const auto &paramsConn = operation->Input(TensorUsage::Params);
        const auto &attr = operation->Attribute<pad_attr_t>();
        const int zeroPoint = ofmConn->quantization.IsValid() ? static_cast<int>(ofmConn->quantization.zeroPoints[0]) : 0;
        const int padConst = int(attr->pad_const) + zeroPoint;

        // Decode the padding before and after each dimension as two shapes
        assert(paramsConn->shape.Elements() == 2 * ifmConn->shape.Size());
        Shape paddingBefore = TensorToShape(paramsConn->tensor.get(), ifmConn->shape.Size(), 2, 0);
        Shape paddingAfter = TensorToShape(paramsConn->tensor.get(), ifmConn->shape.Size(), 2, 1);

        std::shared_ptr<Tensor> padTensor;
        DataType dataType = ofmConn->tensor->Type();

        // Find the largest required pad area and create a constant of that size filled
        // with the padding value. Then memcopy slices of this tensor to the different
        // axes to be padded.
        int maxElements = 0;
        for ( int axis = 0; axis < ofmShape.Size(); axis++ )
        {
            int padElements = (ofmShape.Elements() / ofmShape[axis]) * std::max(paddingBefore[axis], paddingAfter[axis]);
            maxElements = std::max(maxElements, padElements);
        }

        int bits = DataTypeSizeBits(dataType);
        // Mask out the bits from the original constant to force a zero extension regardless
        // of signedness.
        uint32_t fillPattern = uint32_t(padConst) & (~0u >> std::max(32 - bits, 0));
        // Then replicate the bits from the original constant to the rest of the 32-bit value if needed.
        // So for example the 8-bit value -2 (0xfe) is replicated to 0xfefefefe, while the 16-bit value
        // -2 (0xfffe) becomes 0xfffefffe.
        if ( bits < 16 )
        {
            fillPattern |= fillPattern << 8;
        }
        if ( bits < 32 )
        {
            fillPattern |= fillPattern << 16;
        }
        std::vector<uint32_t> buffer(DivRoundUp(DataTypeStorageSizeBytes(dataType, maxElements), 4), fillPattern);
        const Shape padShape = Shape(maxElements);
        padTensor = CreateConstTensor("pad_const", dataType, std::make_shared<Buffer>(std::move(buffer)), &padShape);

        // Padding tensors of higher than rank 4 or rank 4 with a batch larger than 1 requires reshaping to a 3D shape
        // (HWC) where W is the dimension to pad. Only use this strategy when necessary since it is often slower.
        const Shape ifmShape = ifmConn->shape;
        bool reshapeAndPadW = ifmShape.Size() > 4 || (ifmShape.Size() == 4 && ifmShape.Batch() > 1);
        const Shape zeroShape = reshapeAndPadW ? Shape(0, 0, 0) : ofmShape.WithZeros();
        for ( int axis = 0; axis < ifmShape.Size(); axis++ )
        {
            Shape newOfmShape = reshapeAndPadW ? ReshapeTo3DAroundAxis(ofmShape, axis) : ofmShape;
            int padAxis = reshapeAndPadW ? 1 : axis;

            const int padBefore = paddingBefore[axis];
            if ( padBefore )
            {
                TensorSlice newOfmSlice = {zeroShape, newOfmShape.With(padAxis, padBefore)};
                auto fillOp = MakeFillOperation(ofmConn, newOfmShape, newOfmSlice, padTensor);
                RecordOptimisation(*operation, fillOp);
            }

            const int padAfter = paddingAfter[axis];
            if ( padAfter )
            {
                TensorSlice newOfmSlice = {zeroShape.With(padAxis, newOfmShape[padAxis] - padAfter), newOfmShape.With(padAxis, padAfter)};
                auto fillOp = MakeFillOperation(ofmConn, newOfmShape, newOfmSlice, padTensor);
                RecordOptimisation(*operation, fillOp);
            }
        }

        // Copy original IFM to OFM
        auto copyOp = std::make_shared<Operation>(OpType::MemoryCopy);
        copyOp->CopyInput(TensorUsage::IFM, *ifmConn);
        copyOp->CopyOutput(TensorUsage::OFM, *ofmConn);
        copyOp->Output(TensorUsage::OFM)->Set({paddingBefore, ifmShape}).Set(RoundMode::NATURAL);
        RecordOptimisation(*operation, copyOp.get());
        returnOp = copyOp.get();

        // Remove original pad
        operation->Disconnect();
    }
    return returnOp;
}

Operation *GraphIrOptimiser::UnrollKernelStrides(Graph *const, Operation *const operation)
{
    auto returnOp = operation;

    if ( operation->Type() == OpType::Conv2D || operation->Type() == OpType::AvgPool || operation->Type() == OpType::MaxPool )
    {
        const auto ifmConn = operation->Input(TensorUsage::IFM);
        assert(ifmConn);
        TensorConnection *weightsConn = nullptr;
        TensorConnection *scalesConn = nullptr;

        if ( operation->Type() == OpType::Conv2D )
        {
            weightsConn = operation->Input(TensorUsage::Weights);
            assert(weightsConn);
            scalesConn = operation->Input(TensorUsage::Scales);
            assert(scalesConn);
            if ( !weightsConn->tensor->IsConstant() || !scalesConn->tensor->IsConstant() )
            {
                // Do not unroll kernel if bias or weights are non-constant
                return returnOp;
            }
        }
        const auto ofmConn = operation->Output(TensorUsage::OFM);
        assert(ofmConn);

        const auto kernel = operation->Kernel();
        assert(kernel);
        const int32_t kernel_h = kernel->Size().y;
        assert(kernel_h > 0);
        const int32_t kernel_w = kernel->Size().x;
        assert(kernel_w > 0);
        const int32_t stride_h = kernel->Stride().y;
        assert(stride_h > 0);
        const int32_t stride_w = kernel->Stride().x;
        assert(stride_w > 0);
        const int32_t dilation_h = kernel->Dilation().y;
        assert(dilation_h > 0);
        const int32_t dilation_w = kernel->Dilation().x;
        assert(dilation_w > 0);
        const bool hasPadding = !kernel->Padding().IsZero();
        const bool hasIfmSlice = ifmConn->slice.shape || ifmConn->slice.offset;
        const bool hasOfmSlice = ofmConn->slice.shape || ofmConn->slice.offset;

        // Figure out if op needs to be unrolled
        const bool needUnrollH = stride_h > 3;
        const bool needUnrollW = stride_w > 3;

        // Figure out if op can be unrolled
        const bool canUnroll = !hasPadding && !hasIfmSlice && !hasOfmSlice && kernel->Padding().IsZero();
        const bool canUnrollH = dilation_h == 1 && canUnroll;
        const bool canUnrollW = dilation_w == 1 && canUnroll;

        if ( (needUnrollH || needUnrollW) && canUnrollH && canUnrollW )
        {
            const Shape inputGridCell = ifmConn->shape.WithHW(kernel_h, kernel_w);
            const Shape outputGridCell = ofmConn->shape.WithHW(1, 1);
            const Point2i gridSize = ofmConn->shape.WH();

            for ( int h = 0; h < gridSize.y; h++ )
            {
                for ( int w = 0; w < gridSize.x; w++ )
                {
                    TensorSlice ifmSlice;
                    ifmSlice.shape = inputGridCell;
                    ifmSlice.offset = Shape(0, h * stride_h, w * stride_w, 0);

                    TensorSlice ofmSlice;
                    ofmSlice.shape = outputGridCell;
                    ofmSlice.offset = Shape(0, h, w, 0);

                    // Add new for this grid cell
                    auto op = std::make_shared<Operation>(operation->Type());
                    op->SetKernel(std::make_unique<Kernel>(kernel->WithStride({1, 1})));
                    op->CopyInput(TensorUsage::IFM, *ifmConn);
                    op->Input(TensorUsage::IFM)->Set(ifmSlice);
                    if ( weightsConn )
                    {
                        op->CopyInput(TensorUsage::Weights, *weightsConn);
                    }
                    if ( scalesConn )
                    {
                        op->CopyInput(TensorUsage::Scales, *scalesConn);
                    }
                    op->CopyOutput(TensorUsage::OFM, *ofmConn);
                    op->Output(TensorUsage::OFM)->Set(ofmSlice);
                    RecordOptimisation(*operation, op.get());

                    returnOp = op.get();
                }
            }

            // Remove original op
            operation->Disconnect();
        }
    }

    return returnOp;
}

namespace
{
// Reduce scales to a minimal shift-representation as this can improve fusing for Ethos-U55/U65
std::vector<QuantizedScale> ReduceScales(const std::vector<QuantizedScale> &scales, bool doubleRound)
{
    auto reducedScales = scales;
    for ( auto &qs : reducedScales )
    {
        if ( doubleRound && qs.shift > 31 )
        {
            // cannot reduce double-rounded scales with shift > 31
            continue;
        }
        qs = QuantizedScale::ReduceScale(qs);
    }
    return reducedScales;
}

// Generic constraints to determine whether a Rescale should be considered for fusing.
bool CanBeFused(Graph *const graph, Operation *const operation, bool ontoConsumers)
{
    assert(operation->Type() == OpType::Rescale);
    auto ifmConn = operation->Input(TensorUsage::IFM);
    auto ofmConn = operation->Output(TensorUsage::OFM);
    assert(ifmConn);
    assert(ofmConn);
    auto fusedConn = ontoConsumers ? ofmConn : ifmConn;
    auto fusedTensor = fusedConn->tensor;
    auto *signAttr = operation->Attribute<sign_attr_t>();

    // TODO MLBEDSW-11216: Support fusing of unsigned rescales
    if ( signAttr && (signAttr->input_unsigned || signAttr->output_unsigned) )
    {
        return false;
    }
    // TODO MLBEDSW-11218: Support fusing onto multiple consumers
    if ( fusedTensor->Readers().size() != 1 )
    {
        return false;
    }
    // TODO MLBEDSW-11218: Support fusing onto multiple producers
    if ( fusedTensor->Writers().size() != 1 )
    {
        return false;
    }

    // Validate that the rescale can be performed without clipping
    auto ifmType = ifmConn->tensor->Type();
    auto ofmType = ofmConn->tensor->Type();
    auto &ifmQuant = ifmConn->quantization;
    auto &ofmQuant = ofmConn->quantization;
    if ( ontoConsumers )
    {
        // Find the most significant scale-factor
        QuantizedScale qs = QuantizedScale::Unit();
        auto itr = std::max_element(std::begin(ofmQuant.scales), std::end(ofmQuant.scales));
        if ( itr != std::end(ofmQuant.scales) )
        {
            qs = *itr;
        }
        assert(ifmQuant.zeroPoints.size() <= 1 && "Rescale with multiple zeroPoints");
        int64_t zp = ifmQuant.zeroPoints.size() ? ifmQuant.zeroPoints.front() : 0;
        int64_t value = (zp < 0 ? int64_t(IntegerMax(ifmType)) : IntegerMin(ifmType));
        value = value - zp;
        assert(qs.shift >= 0 && "QuantizedScale with negative shift");
        assert(qs.shift < 64 && "Shift is out of bounds");
        value = (value * qs.scale) >> qs.shift;
        if ( value < IntegerMin(ofmType) || value > int64_t(IntegerMax(ofmType)) )
        {
            return false;
        }
    }

    // Cannot fuse-away non-unit zeroPoint
    if ( fusedConn->quantization.zeroPoints != Quantization::Unit().zeroPoints )
    {
        return false;
    }
    // Cannot fuse-away graph input/output tensors
    if ( graph->IsOutput(fusedTensor.get()) )
    {
        return false;
    }
    if ( graph->IsInput(fusedTensor.get()) )
    {
        return false;
    }
    return true;
}
}  // namespace

// Check if a rescale can be fused onto a specific consumer
bool GraphIrOptimiser::CanFuseRescaleOnConsumer(Operation *const consumer, TensorUsage usage, Quantization &newQuant, DataType newType)
{
    OpType opType = consumer->Type();
    // Don't fuse to dataLayout opTypes as those are not typically
    // associated with quantization
    // This avoids incorrect propagation of quantization
    // as dataLayout operations are often fused or removed
    if ( IsDataLayout(opType) )
    {
        return false;
    }
    // Connection to the fused-away tensor
    auto fusedConn = consumer->Input(usage);
    auto fusedType = fusedConn->tensor->Type();
    const auto &ofmConn = consumer->Output(TensorUsage::OFM);
    const auto &ofmQuant = ofmConn->quantization;
    auto ofmType = ofmConn->tensor->Type();
    assert(consumer->Input(usage));

    // Cannot fuse-away non-unit quantization
    if ( !consumer->Input(usage)->quantization.EqualScales(Quantization::Unit()) )
    {
        return false;
    }
    if ( IsBinaryElementwise(consumer->Type()) )
    {
        TensorUsage otherInputUsage = usage == TensorUsage::IFM0 ? TensorUsage::IFM1 : TensorUsage::IFM0;
        auto otherInputConn = consumer->Input(otherInputUsage);
        const auto &otherQuant = otherInputConn->quantization;
        auto otherType = otherInputConn->tensor->Type();
        return _constraints->SupportsQuantization(consumer->Type(), newQuant, newType, otherQuant, otherType, ofmQuant, ofmType);
    }
    return _constraints->SupportsQuantization(consumer->Type(), newQuant, newType, ofmQuant, ofmType);
}

// Check if multiple rescales can be fused together onto a specific consumer
bool GraphIrOptimiser::CanFuseMultipleRescalesOnConsumer(Operation *const consumer, Operation *const rescale1,
    Operation *const rescale2, const Quantization &q1, const Quantization &q2)
{
    OpType opType = consumer->Type();
    // Don't fuse to dataLayout opTypes as those are not typically
    // associated with quantization
    // This avoids incorrect propagation of quantization
    // as dataLayout operations are often fused or removed
    if ( IsDataLayout(opType) )
    {
        return false;
    }
    assert(rescale1->Type() == OpType::Rescale);
    assert(rescale2->Type() == OpType::Rescale);
    auto ofmConn1 = rescale1->Output(TensorUsage::OFM);
    auto ofmConn2 = rescale2->Output(TensorUsage::OFM);
    auto consumerOfmConn = consumer->Output(TensorUsage::OFM);
    assert(ofmConn1->tensor->Writers().size() == 1 && "rescale1 must be the only producer when IFM-fusing");
    assert(ofmConn2->tensor->Writers().size() == 1 && "rescale2 must be the only producer when IFM-fusing");
    assert(consumer->UsageOfTensor(ofmConn1->tensor.get()) != TensorUsage::None && "fused tensor1 is not connected to consumer");
    assert(consumer->UsageOfTensor(ofmConn2->tensor.get()) != TensorUsage::None && "fused tensor2 is not connected to consumer");
    auto newType1 = rescale1->Input(TensorUsage::IFM)->tensor->Type();
    auto newType2 = rescale2->Input(TensorUsage::IFM)->tensor->Type();
    // Cannot fuse-away non-unit quantization
    if ( !consumer->Input(TensorUsage::IFM0)->quantization.EqualScales(Quantization::Unit()) )
    {
        return false;
    }
    if ( !consumer->Input(TensorUsage::IFM1)->quantization.EqualScales(Quantization::Unit()) )
    {
        return false;
    }
    auto fusedType1 = rescale1->Output(TensorUsage::OFM)->tensor->Type();
    auto fusedType2 = rescale2->Output(TensorUsage::OFM)->tensor->Type();
    auto consumerOfmType = consumer->Output(TensorUsage::OFM)->tensor->Type();
    return _constraints->SupportsQuantization(
        consumer->Type(), q1, newType1, q2, newType2, consumerOfmConn->quantization, consumerOfmConn->tensor->Type());
}

// Check if a rescale can be fused onto a specific producer
bool GraphIrOptimiser::CanFuseRescaleOnProducer(Operation *const producer, Quantization &newQuant, DataType newType)
{
    OpType opType = producer->Type();
    // Don't fuse to dataLayout opTypes as those are not typically
    // associated with quantization
    // This avoids incorrect propagation of quantization
    // as dataLayout operations are often fused or removed
    if ( IsDataLayout(opType) )
    {
        return false;
    }
    // Connection to the fused-away tensor
    auto fusedConn = producer->Output(TensorUsage::OFM);
    auto fusedType = fusedConn->tensor->Type();
    assert(producer->Input(TensorUsage::IFM));
    // Cannot fuse-away non-unit quantization
    if ( !fusedConn->quantization.EqualScales(Quantization::Unit()) )
    {
        return false;
    }
    auto ifmConn = producer->Input(TensorUsage::IFM);
    auto ifm2Conn = producer->Input(TensorUsage::IFM1);
    auto ifmType = ifmConn->tensor->Type();
    const auto &ifmQuant = ifmConn->quantization;
    auto ifm2Type = ifm2Conn ? ifm2Conn->tensor->Type() : DataType::None;
    const auto &ifm2Quant = ifm2Conn ? ifm2Conn->quantization : Quantization::Unit();
    return _constraints->SupportsQuantization(producer->Type(), ifmQuant, ifmType, ifm2Quant, ifm2Type, newQuant, newType);
}

/// @brief Moves Rescale operations to the output of the previous operation
///        or the input of the next operation when possible.
///
/// @param operation Operation to optimise
/// @return (Possibly) optimised operation
Operation *GraphIrOptimiser::FuseRescale(Graph *const graph, Operation *const operation)
{
    Operation *returnOp = operation;
    OpType opType = operation->Type();
    if ( opType != OpType::Rescale )
    {
        return returnOp;
    }
    // Create new quantization after IFM-fusing a rescale
    auto QuantAfterIFMFusing = [](Operation *const rescale)
    {
        assert(rescale->Type() == OpType::Rescale);
        auto *attr = rescale->Attribute<rescale_attr_t>();
        assert(attr);
        auto ifmConn = rescale->Input(TensorUsage::IFM);
        auto ofmConn = rescale->Output(TensorUsage::OFM);
        // The resulting quantization is ofm-scales, but with ifm ZP and clamping
        //  -----------------------> Rescale -------------------->
        //  ^ we want to keep this              ^ but the multiplier and shift
        //    connection when IFM-fusing        are stored here in GraphIR
        auto quant = ifmConn->quantization;
        assert(quant.type == QuantizationType::EXPLICIT && "Rescale without explicit scaling");
        quant.scales = ReduceScales(ofmConn->quantization.scales, attr->double_round);
        return quant;
    };
    auto ofmConn = operation->Output(TensorUsage::OFM);
    auto ifmConn = operation->Input(TensorUsage::IFM);
    // Fusing onto consumers
    if ( CanBeFused(graph, operation, /*ontoConsumers */ true) )
    {
        auto newQuant = QuantAfterIFMFusing(operation);
        // TODO MLBEDSW-11218: Support fusing onto multiple consumers
        assert(ofmConn->tensor->Readers().size() == 1);
        auto consumer = ofmConn->tensor->Readers()[0];
        TensorUsage consumerUsage = consumer->UsageOfTensor(ofmConn->tensor.get());
        // Check if we can IFM-fuse directly on the consumer without involving any other Rescales
        if ( CanFuseRescaleOnConsumer(consumer.get(), consumerUsage, newQuant, ifmConn->tensor->Type()) )
        {
            ReplaceConsumerInput(nullptr, ofmConn->tensor->Readers(), ofmConn->tensor.get(), ifmConn->tensor);
            consumer->Input(consumerUsage)->Set(ofmConn->rounding).Set(newQuant);
            returnOp = consumer.get();
            operation->Disconnect();
        }
        else if ( IsBinaryElementwise(consumer->Type()) )
        {
            // Might be possible to IFM-fuse if there is also a rescale producing the other IFM-connection.
            // e.g.
            //
            //      int8          int8
            //       \             /
            //      RESCALE    RESCALE
            //         \         /
            //        int32    int32
            //            \    /
            //             MUL
            //
            // Here we'd need to IFM-fuse both Rescales
            // Otherwise we'd end up with a mix of input-dataTypes.
            TensorUsage otherInputUsage = consumerUsage == TensorUsage::IFM0 ? TensorUsage::IFM1 : TensorUsage::IFM0;
            auto otherInputConn = consumer->Input(otherInputUsage);
            // Other connection must have one producer, and it must be a rescale
            if ( otherInputConn->tensor->Writers().size() == 1 && otherInputConn->tensor->Writers()[0]->Type() == OpType::Rescale )
            {
                auto otherRescale = otherInputConn->tensor->Writers()[0];
                auto otherRescaleIfmConn = otherRescale->Input(TensorUsage::IFM);
                auto otherRescaleOfmConn = otherRescale->Output(TensorUsage::OFM);
                auto newQuant2 = QuantAfterIFMFusing(otherRescale.get());
                if ( CanBeFused(graph, otherRescale.get(), /*ontoConsumers*/ true) )
                {
                    // TODO MLBEDSW-11218: Support fusing onto multiple consumers
                    assert(otherRescale->Output(TensorUsage::OFM)->tensor->Readers().size() == 1);
                    if ( CanFuseMultipleRescalesOnConsumer(consumer.get(), operation, otherRescale.get(), newQuant, newQuant2) )
                    {
                        // Fuse current rescale
                        ReplaceConsumerInput(nullptr, ofmConn->tensor->Readers(), ofmConn->tensor.get(), ifmConn->tensor);
                        consumer->Input(consumerUsage)->Set(ofmConn->rounding).Set(newQuant);
                        // Fuse other rescale
                        ReplaceConsumerInput(nullptr, otherRescaleOfmConn->tensor->Readers(),
                            otherRescaleOfmConn->tensor.get(), otherRescaleIfmConn->tensor);
                        consumer->Input(otherInputUsage)->Set(otherRescaleOfmConn->rounding).Set(newQuant2);
                        returnOp = consumer.get();
                        operation->Disconnect();
                        otherRescale->Disconnect();
                    }
                }
            }
        }
    }
    // Fusing onto producers
    if ( returnOp == operation && CanBeFused(graph, operation, /* ontoConsumers */ false) )
    {
        auto newOFMQuant = ofmConn->quantization;
        auto *attr = operation->Attribute<rescale_attr_t>();
        // Normalize scales to shift 0 if possible
        newOFMQuant.scales = ReduceScales(ofmConn->quantization.scales, attr->double_round);
        assert(newOFMQuant.type == QuantizationType::EXPLICIT && "Rescale without explicit scaling");
        // TODO MLBEDSW-11218: Support fusing onto multiple producers
        assert(ifmConn->tensor->Writers().size() == 1);
        auto producer = ifmConn->tensor->Writers()[0];
        if ( CanFuseRescaleOnProducer(producer.get(), newOFMQuant, ofmConn->tensor->Type()) )
        {
            // Propagate rescaling to output of previous op
            ReplaceProducerOutput({producer}, ifmConn->tensor.get(), ofmConn->tensor);
            producer->Output(TensorUsage::OFM)->Set(ofmConn->rounding).Set(newOFMQuant);
            returnOp = producer.get();
            operation->Disconnect();
        }
    }
    return returnOp;
}

// Fixup Pool strides when the kernel size, IFM shape and stride are equal.
Operation *GraphIrOptimiser::FixupPoolStrides(Graph *const, Operation *const operation)
{
    if ( IsPooling(operation->Type()) )
    {
        auto kernel = operation->Kernel();
        const auto ifm = operation->Input(TensorUsage::IFM);
        if ( kernel->Size() == kernel->Stride() && ifm->shape.Size() >= 3 && kernel->Stride() == ifm->shape.WH() &&
             kernel->Padding().IsZero() )
        {
            operation->SetKernel(std::make_unique<Kernel>(kernel->WithStride({1, 1})));
        }
    }
    return operation;
}

// Rewrite TOSA Table to GraphIR LUT
Operation *GraphIrOptimiser::RewriteTable(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType == OpType::Table )
    {
        const auto ifmConn = operation->Input(TensorUsage::IFM);
        const auto lutConn = operation->Input(TensorUsage::Params);
        const auto ofmConn = operation->Output(TensorUsage::OFM);
        assert(ifmConn);
        assert(lutConn);
        assert(ofmConn);

        std::shared_ptr<Tensor> newLutTensor;
        const auto newLutTensorType = lutConn->tensor->Type();
        assert(newLutTensorType == DataType::Int8 || newLutTensorType == DataType::Int16);
        if ( newLutTensorType == DataType::Int8 )
        {
            // For int8, TOSA Table is same as GraphIR LUT
            newLutTensor = lutConn->tensor;
        }
        else
        {
            // For int16, we need to recalculate the LUT tensor
            const auto view = lutConn->tensor->View();
            assert(view.ViewShape() == Shape(513));
            const auto values = view.Values<int16_t>();
            auto newLut = std::make_unique<int16_t[]>(1024);
            for ( int i = 0; i < 512; i++ )
            {
                newLut[2 * i] = values[i];                      // Base
                newLut[2 * i + 1] = values[i + 1] - values[i];  // Slope
            }
            newLutTensor = CreateConstTensor("LUT", newLutTensorType, std::make_shared<Buffer>(std::move(newLut), 1024));
        }

        // Check for native support for this Table operation
        ArchOperatorQuery query;
        ArchRequirements req;
        query.ifm[0].type = ifmConn->tensor->Type();
        query.ofm.type = ofmConn->tensor->Type();
        auto res = _constraints->OperatorQuery(OpType::LUT, &query, &req);

        if ( newLutTensorType == DataType::Int16 && res.Any(QueryResult::HasRequirements) && req.req.Any(ArchRequirement::OpSubstitution) )
        {
            // Need to break down into three different LUTs and combine the results
            const auto values = newLutTensor->View().Values<int16_t>();
            auto baseLut = std::make_unique<int16_t[]>(1024);
            auto slopeLut = std::make_unique<int16_t[]>(1024);
            auto fractionLut = std::make_unique<int16_t[]>(1024);
            for ( int i = 0; i < 512; i++ )
            {
                baseLut[2 * i] = values[2 * i];       // Base
                baseLut[2 * i + 1] = 0;               // Slope = 0 to get base
                slopeLut[2 * i] = values[2 * i + 1];  // Base  = Slope to get slope
                slopeLut[2 * i + 1] = 0;              // Slope = 0
                fractionLut[2 * i] = 0;               // Base  = 0
                fractionLut[2 * i + 1] = (1 << 7);    // Slope = 1 << 7 to get fraction
            }
            auto baseLutTensor = CreateConstTensor("BaseLUT", DataType::Int16, std::make_shared<Buffer>(std::move(baseLut), 1024));
            auto slopeLutTensor = CreateConstTensor("SlopeLUT", DataType::Int16, std::make_shared<Buffer>(std::move(slopeLut), 1024));
            auto fractionLutTensor = CreateConstTensor(
                "FractionLUT", DataType::Int16, std::make_shared<Buffer>(std::move(fractionLut), 1024));

            // Get the base value tensor using the LUT created above
            auto baseLutOp = CreateLUT(ifmConn->tensor, baseLutTensor, ifmConn->quantization, Quantization::Unit(),
                baseLutTensor->Type(), &ifmConn->shape, nullptr, ifmConn->slice);
            auto baseTensor = baseLutOp->OFM()->shared_from_this();
            RecordOptimisation(*operation, baseLutOp);

            // Get the slope value tensor using the LUT created above
            auto slopeLutOp = CreateLUT(ifmConn->tensor, slopeLutTensor, ifmConn->quantization, Quantization::Unit(),
                slopeLutTensor->Type(), &ifmConn->shape, nullptr, ifmConn->slice);
            auto slopeTensor = slopeLutOp->OFM()->shared_from_this();
            RecordOptimisation(*operation, slopeLutOp);

            // Get the fraction value tensor using the LUT created above
            auto fractionLutOp = CreateLUT(ifmConn->tensor, fractionLutTensor, ifmConn->quantization,
                Quantization::Unit(), fractionLutTensor->Type(), &ifmConn->shape, nullptr, ifmConn->slice);
            auto fractionTensor = fractionLutOp->OFM()->shared_from_this();
            RecordOptimisation(*operation, fractionLutOp);

            // Multiply slope * fraction
            auto mulSlopeFractionOp = CreateMul(slopeTensor, fractionTensor, Quantization::Unit(), Quantization::Unit(),
                Quantization::Unit(), DataType::Int32);
            RecordOptimisation(*operation, mulSlopeFractionOp);

            // Shift base << 7
            auto shiftLeftBaseOp = CreateMul(baseTensor, CreateConstTensor("shl_seven", int16_t(1 << 7)),
                Quantization::Unit(), Quantization::Unit(), Quantization::Unit(), DataType::Int32);
            RecordOptimisation(*operation, shiftLeftBaseOp);

            // Add (base << 7) + (slope * fraction)
            auto addOp = std::make_shared<Operation>(OpType::Add);
            addOp->ConnectInput(TensorUsage::IFM0, shiftLeftBaseOp->OFM()->shared_from_this());
            addOp->ConnectInput(TensorUsage::IFM1, mulSlopeFractionOp->OFM()->shared_from_this());
            addOp->CopyOutput(TensorUsage::OFM, *ofmConn);
            returnOp = addOp.get();
        }
        else
        {
            // Replace TOSA Table op with GraphIR LUT op
            assert(ofmConn->quantization.IsUnitScale() ||
                   (ofmConn->quantization.scales.size() == 1 && ofmConn->quantization.scales.front() == QuantizedScale(1, 7)));
            returnOp = CreateLUT(ifmConn->tensor, newLutTensor, ifmConn->quantization, Quantization::Unit(),
                newLutTensor->Type(), &ifmConn->shape, ofmConn->tensor, ifmConn->slice, ofmConn->slice);
        }
    }
    if ( returnOp != operation )
    {
        returnOp->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
        RecordOptimisation(*operation, returnOp);
        operation->Disconnect();
    }
    return returnOp;
}

// Rewrite TOSA Cast and int64 cast to other ops
Operation *GraphIrOptimiser::RewriteCast(Graph *const, Operation *const operation)
{
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType == OpType::Cast )
    {
        const auto ifmConn = operation->Input(TensorUsage::IFM);
        const auto ofmConn = operation->Output(TensorUsage::OFM);

        auto ofmType = ofmConn->tensor->Type();
        /* Casting to int32 is hardware supported, but casting to int64 is not. We solve this by converting
         * the int64 cast to a series of operations in the following if statement. This does not work for int32 input.
         * 1. Cast the input to an int32 tensor.
         * The tensor size is kept the same (WxHxC -> WxHxC) but the memory size is doubled.
         * 2. Reinterpret the tensor as an int16 tensor.
         * The tensor size is doubled (WxHxC -> WxHx2C), where every second element is 0xFFFF / 0x0000 for
         * negative / positive numbers. Memory size is unchanged.
         * 3. Cast the reinterpreted input to an int32 tensor again.
         * The tensor size is again the same (WxHx2C -> WxHx2C) but the size in memory is double.
         * 4. Finally, reinterpret the result as an int64 tensor.
         * The 0x0000FFFF / 0x00000000 elements becomes most significant bits of the int64 values.
         * Tensor size (WxHx2C -> WxHxC) */
        if ( (ofmType == DataType::Int64) || (ofmType == DataType::UInt64) )
        {
            bool allowedDataType = ifmConn->tensor->Type() != DataType::Int32 && ifmConn->tensor->Type() != DataType::UInt32;
            assert(allowedDataType && "Casting from int32 to int64 is not supported.");

            const int c = ifmConn->shape.Depth();

            // Create intermediate tensor for the casting
            const auto intermediate32Bit = std::make_shared<Tensor>("intermediate_32bit", DataType::Int32, ifmConn->shape);

            // Create double size intermediate tensor for the casting
            const auto intermediate16Bit2xSize = std::make_shared<Tensor>(
                "intermediate16Bit2xSize", DataType::Int16, ifmConn->shape.WithDepth(2 * c));

            // Create double size intermediate tensor for the casting
            const auto intermediate32Bit2xSize = std::make_shared<Tensor>(
                "intermediate32Bit2xSize", DataType::Int32, ifmConn->shape.WithDepth(2 * c));

            // Connect the cast output to the newly created tensor
            const auto castOp1 = std::make_shared<Operation>(OpType::Cast);
            castOp1->CopyInput(TensorUsage::IFM, *ifmConn);
            castOp1->ConnectOutput(TensorUsage::OFM, intermediate32Bit);
            RecordOptimisation(*operation, castOp1.get());

            // Create reinterpret cast op to reinterpret to 16 bit, double size
            const auto reinterpretOp1 = std::make_shared<Operation>(OpType::ReinterpretCast);
            reinterpretOp1->ConnectInput(TensorUsage::IFM, intermediate32Bit);
            reinterpretOp1->ConnectOutput(TensorUsage::OFM, intermediate16Bit2xSize);
            RecordOptimisation(*operation, reinterpretOp1.get());

            // Create additional cast op
            const auto castOp2 = std::make_shared<Operation>(OpType::Cast);
            castOp2->ConnectInput(TensorUsage::IFM, intermediate16Bit2xSize).Set(ifmConn->shape.WithDepth(2 * c));
            castOp2->ConnectOutput(TensorUsage::OFM, intermediate32Bit2xSize).Set(ifmConn->shape.WithDepth(2 * c));
            RecordOptimisation(*operation, castOp2.get());

            // Create the final reinterpret cast to reinterpret the result as an int64 tensor
            const auto reinterpretOp2 = std::make_shared<Operation>(OpType::ReinterpretCast);
            reinterpretOp2->ConnectInput(TensorUsage::IFM, intermediate32Bit2xSize).Set(ifmConn->shape.WithDepth(2 * c));
            reinterpretOp2->CopyOutput(TensorUsage::OFM, *ofmConn);
            RecordOptimisation(*operation, reinterpretOp2.get());

            ofmConn->quantization = Quantization::Unit();
            operation->Disconnect();
            returnOp = reinterpretOp2.get();
            return returnOp;
        }

        if ( IsBool(ifmConn->tensor->Type()) && IsInteger(ofmConn->tensor->Type()) )
        {
            // Replace CAST with BITWISE_AND to convert from internal bool representation to integer
            auto newOp = std::make_shared<Operation>(OpType::And);
            newOp->CopyInput(TensorUsage::IFM0, *ifmConn);
            newOp->ConnectInput(TensorUsage::IFM1, CreateConstTensor("const_one", int8_t(1)));
            newOp->CopyOutput(TensorUsage::OFM, *ofmConn);
            RecordOptimisation(*operation, newOp.get());
            operation->Disconnect();
            returnOp = newOp.get();
        }
        else if ( IsInteger(ifmConn->tensor->Type()) && IsBool(ofmConn->tensor->Type()) )
        {
            // Replace CAST with CMP_NE to convert from integer to internal bool representation
            auto newOp = std::make_shared<Operation>(OpType::NotEqual);
            newOp->CopyInput(TensorUsage::IFM0, *ifmConn);
            newOp->ConnectInput(TensorUsage::IFM1, CreateConstTensor("const_zero", ifmConn->tensor->Type(), 0));
            newOp->CopyOutput(TensorUsage::OFM, *ofmConn);
            RecordOptimisation(*operation, newOp.get());
            operation->Disconnect();
            returnOp = newOp.get();
        }
        else
        {
            // Replace CAST with ADD
            auto copyOp = std::make_shared<Operation>(OpType::Add);
            auto type = ifmConn->tensor->Type();
            ReplaceOperation(operation, copyOp.get());
            copyOp->ConnectInput(TensorUsage::IFM1, CreateConstTensor("const_zero", type, 0));
            RecordOptimisation(*operation, copyOp.get());
            returnOp = copyOp.get();

            // Set max range to disable clipping
            auto copyOpConn = copyOp->Output(TensorUsage::OFM);
            copyOpConn->quantization.quantMin = {std::numeric_limits<int64_t>::min()};
            copyOpConn->quantization.quantMax = {std::numeric_limits<int64_t>::max()};
        }

        // Copy sign attribute to new operation
        if ( operation->HasAttribute<sign_attr_t>() )
        {
            auto signAttr = operation->Attribute<sign_attr_t>();
            auto newAttr = returnOp->Attribute<sign_attr_t>();
            *newAttr = *signAttr;
        }
    }
    return returnOp;
}

// Rewrite TOSA Concat to one MemoryCopy per IFM
Operation *GraphIrOptimiser::RewriteConcat(Graph *const graph, Operation *const operation)
{
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType == OpType::Concat )
    {
        const auto *ofmConn = operation->Output(TensorUsage::OFM);
        const auto *attr = operation->Attribute<axis_attr_t>();
        auto axis = attr->axis;
        if ( axis < 0 ) axis = ofmConn->shape.Size() + axis;

        // Replace CONCAT with a memory copy per IFM that copies IFM to an offset into OFM
        Shape ofmSliceOffset = ofmConn->shape.WithZeros();
        for ( auto [usage, ifmConn] : operation->Inputs().pairs() )
        {
            if ( !IsIFM(usage) ) continue;

            auto copyOp = std::make_shared<Operation>(OpType::MemoryCopy);
            copyOp->CopyInput(TensorUsage::IFM, ifmConn);
            copyOp->CopyOutput(TensorUsage::OFM, *ofmConn);
            copyOp->Output(TensorUsage::OFM)->Set({ofmSliceOffset, ifmConn.shape});
            copyOp->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
            RecordOptimisation(*operation, copyOp.get());
            returnOp = copyOp.get();

            ofmSliceOffset[axis] += ifmConn.shape[axis];
        }
        operation->Disconnect();
    }
    return returnOp;
}

// Rewrite TOSA Slice to a MemoryCopy
Operation *GraphIrOptimiser::RewriteSlice(Graph *const graph, Operation *const operation)
{
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType == OpType::Slice )
    {
        const auto *ifmConn = operation->Input(TensorUsage::IFM);
        const auto *ofmConn = operation->Output(TensorUsage::OFM);
        auto *attr = operation->Attribute<slice_attr_t>();
        const Shape begin = attr->begin;
        const Shape size = attr->size;

        // Replace SLICE with a memory copy with IFM slice
        auto copyOp = std::make_shared<Operation>(OpType::MemoryCopy);
        copyOp->CopyInput(TensorUsage::IFM, *ifmConn);
        copyOp->Input(TensorUsage::IFM)->Set({begin, size});
        copyOp->CopyOutput(TensorUsage::OFM, *ofmConn);
        copyOp->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
        RecordOptimisation(*operation, copyOp.get());
        returnOp = copyOp.get();
        operation->Disconnect();
    }
    return returnOp;
}

// Rewrite TOSA Negate to TOSA Sub
Operation *GraphIrOptimiser::RewriteNegate(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType == OpType::Neg )
    {
        const auto ifmConn = operation->Input(TensorUsage::IFM);
        const auto ofmConn = operation->Output(TensorUsage::OFM);

        // Replace NEG(x) with SUB(0, x)
        auto newOp = std::make_shared<Operation>(OpType::Sub);
        newOp->ConnectInput(TensorUsage::IFM0, CreateConstTensor("const_zero", ifmConn->tensor->Type(), 0));
        newOp->CopyInput(TensorUsage::IFM1, *ifmConn);
        newOp->CopyOutput(TensorUsage::OFM, *ofmConn);
        newOp->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
        RecordOptimisation(*operation, newOp.get());
        returnOp = newOp.get();
        operation->Disconnect();
    }
    return returnOp;
}

// Rewrite TOSA Select as ((ifm1 & ifm0) | (ifm2 & ~ifm0))
Operation *GraphIrOptimiser::RewriteSelect(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType == OpType::Select || opType == OpType::SelectV2 )
    {
        auto selectorConn = operation->Input(TensorUsage::IFM0);
        const auto ifm1Conn = operation->Input(TensorUsage::IFM1);  // Used if selector is true
        const auto ifm2Conn = operation->Input(TensorUsage::IFM2);  // Used if selector is false
        const auto ofmConn = operation->Output(TensorUsage::OFM);

        // Cast selector IFM (bool8) to same data type as the OFM (if needed)
        if ( DataTypeSizeBits(selectorConn->tensor->Type()) != DataTypeSizeBits(ofmConn->tensor->Type()) )
        {
            assert(selectorConn->tensor->Type() == DataType::Bool8);
            auto addOp = CreateAdd(selectorConn->tensor, CreateConstTensor("const_zero", DataType::Int8, 0),
                selectorConn->quantization, Quantization::Unit(), Quantization::Unit(), ofmConn->tensor->Type());
            selectorConn = addOp->Output(TensorUsage::OFM);
            RecordOptimisation(*operation, addOp);
        }

        // Break down SELECT(selector, a, b) into OR(AND(a, selector), AND_NOT(b, selector))
        auto andOp = CreateBinaryElementwise(OpType::And, ifm1Conn->tensor, selectorConn->tensor,
            ifm1Conn->quantization, selectorConn->quantization, ofmConn->quantization, ofmConn->tensor->Type());
        auto andNotOp = CreateBinaryElementwise(OpType::AndNot, ifm2Conn->tensor, selectorConn->tensor,
            ifm2Conn->quantization, selectorConn->quantization, ofmConn->quantization, ofmConn->tensor->Type());
        auto orOp = CreateBinaryElementwise(OpType::Or, andOp->Output(TensorUsage::OFM)->tensor,
            andNotOp->Output(TensorUsage::OFM)->tensor, ofmConn->quantization, ofmConn->quantization,
            ofmConn->quantization, ofmConn->tensor->Type());
        orOp->CopyOutput(TensorUsage::OFM, *ofmConn);
        RecordOptimisation(*operation, andOp);
        RecordOptimisation(*operation, andNotOp);
        RecordOptimisation(*operation, orOp);
        returnOp = orOp;

        // Remove old select op
        operation->Disconnect();
    }
    return returnOp;
}

// If ReduceMin is not HW supported, but ReduceMax is, then rewrite the operation.
// The operation is replaced by [Sub -> ReduceMax -> Sub] that inverts the sign of
// the input, runs ReduceMax, then inverts it back.
Operation *GraphIrOptimiser::RewriteReduceMin(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    if ( operation->Type() != OpType::ReduceMin )
    {
        return operation;
    }

    // If ReduceMin is HW supported, abort.
    ArchOperatorQuery hwSupportedQuery{};
    if ( _constraints->OperatorQuery(OpType::ReduceMin, &hwSupportedQuery, nullptr).Any(QueryResult::Native) )
    {
        return operation;
    }

    auto *ifmConn = operation->Input(TensorUsage::IFM);
    auto *ofmConn = operation->Output(TensorUsage::OFM);
    auto *attr = operation->Attribute<axis_attr_t>();

    int axis = attr->axis;
    if ( axis < 0 )
    {
        axis += ifmConn->shape.Size();
    }
    assert(axis >= 0);
    assert(axis < ifmConn->shape.Size());

    // Create a pivot, a single constant value tensor.
    // The pivot becomes the max value for unsigned values and -1 for signed types.
    const DataType type = ifmConn->tensor->Type();
    auto constTensorName = ifmConn->tensor->Name() + "_reducemin_pivot";
    // Pass an integer with all bits set to CreateConstTensor.
    // This corresponds to -1 for signed integer types and the integer max value for unsigned integer types.
    auto pivotConst = CreateConstTensor(constTensorName, type, int(~0u));

    // Create the first sub op that reverses the input
    auto sub1 = std::make_shared<Operation>(OpType::Sub);
    sub1->ConnectInput(TensorUsage::IFM0, pivotConst).Set(Quantization::Unit());
    sub1->ConnectInput(TensorUsage::IFM1, ifmConn->tensor).Set(Quantization::Unit());

    // Create the first intermediate tensor
    auto reduceMaxInputTensor = std::make_shared<Tensor>(ifmConn->tensor->Name() + "_reducemin_invert", type, ifmConn->shape);
    sub1->ConnectOutput(TensorUsage::OFM, reduceMaxInputTensor).Set(Quantization::Unit());
    RecordOptimisation(*operation, sub1.get());

    // Create the ReduceMax operator
    auto reduceMaxOp = std::make_shared<Operation>(OpType::ReduceMax);
    auto reduceMaxAttr = reduceMaxOp->Attribute<axis_attr_t>();
    reduceMaxAttr->axis = attr->axis;
    reduceMaxOp->SetKernel(std::make_unique<Kernel>(*operation->Kernel()));
    reduceMaxOp->ConnectInput(TensorUsage::IFM, reduceMaxInputTensor)
        .Set(ifmConn->shape)
        .Set(ifmConn->quantization)
        .Set(ifmConn->rounding);

    // Create the second intermediate tensor
    auto reduceMaxOutputTensor = std::make_shared<Tensor>(
        ofmConn->tensor->Name() + "_reducemin_invert_back", ofmConn->tensor->Type(), ofmConn->shape);
    reduceMaxOp->ConnectOutput(TensorUsage::OFM, reduceMaxOutputTensor)
        .Set(ofmConn->shape)
        .Set(ofmConn->quantization)
        .Set(ofmConn->rounding);
    RecordOptimisation(*operation, reduceMaxOp.get());

    // Create the last sub op that reverses back the output
    auto sub2 = std::make_shared<Operation>(OpType::Sub);
    sub2->ConnectInput(TensorUsage::IFM0, pivotConst).Set(Quantization::Unit());
    sub2->ConnectInput(TensorUsage::IFM1, reduceMaxOutputTensor).Set(Quantization::Unit());
    sub2->ConnectOutput(TensorUsage::OFM, ofmConn->tensor).Set(Quantization::Unit());
    RecordOptimisation(*operation, sub2.get());

    operation->Disconnect();
    return sub2.get();
}

// Rewrite REDUCE_SUM with any axis into a REDUCE_SUM with C axis
Operation *GraphIrOptimiser::RewriteReduceSum(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType == OpType::ReduceSum )
    {
        auto *ifmConn = operation->Input(TensorUsage::IFM);
        auto *ofmConn = operation->Output(TensorUsage::OFM);
        auto *attr = operation->Attribute<axis_attr_t>();
        auto axis = attr->axis;
        if ( axis < 0 ) axis = ifmConn->shape.Size() + axis;
        assert(axis >= 0);
        assert(axis < ifmConn->shape.Size());

        if ( axis != ifmConn->shape.Size() - 1 )
        {
            // Replace ReduceSum (axis != C) with a Reshape, Transpose and ReduceSum (axis = C):
            //
            // 1. Reshape to 3D shape (HWC) where W dimension is the dimension to reduce.
            // 2. Transpose HCW: HxWxC -> HxCxW.
            // 3. ReduceSum axis C: HxCxW -> HxCx1.

            // Calculate 3D shape of IFM where 2nd dimension is the dimension to reduce
            const Shape ifmShape3D = ReshapeTo3DAroundAxis(ifmConn->shape, axis);

            // Create intermediate tensor between Transpose and ReduceSum
            std::shared_ptr<Tensor> transposeTens = ifmConn->tensor->Clone();
            transposeTens->SetBuffer(nullptr);
            transposeTens->SetName(ifmConn->tensor->Name() + "_transpose");
            transposeTens->SetStorageShape(ifmShape3D.Extract(0, 2, 1));

            // Create Transpose op
            auto transposeOp = std::make_shared<Operation>(OpType::Transpose);
            auto transposeAttr = transposeOp->Attribute<transpose_attr_t>();
            transposeAttr->perm = {0, 2, 1};  // HCW
            transposeOp->CopyInput(TensorUsage::IFM, *ifmConn);
            transposeOp->Input(TensorUsage::IFM)->Set(ifmShape3D).Set(Quantization::Unit());
            transposeOp->ConnectOutput(TensorUsage::OFM, transposeTens);
            RecordOptimisation(*operation, transposeOp.get());

            // Create ReduceSum op
            auto reduceSumOp = std::make_shared<Operation>(OpType::ReduceSum);
            auto reduceAttr = reduceSumOp->Attribute<axis_attr_t>();
            reduceAttr->axis = 2;  // C
            reduceSumOp->ConnectInput(TensorUsage::IFM, transposeTens).Set(ifmConn->quantization).Set(ifmConn->rounding);
            reduceSumOp->CopyOutput(TensorUsage::OFM, *ofmConn);
            reduceSumOp->Output(TensorUsage::OFM)->Set(transposeTens->StorageShape().WithDepth(1)).Set(ofmConn->rounding);
            RecordOptimisation(*operation, reduceSumOp.get());
            returnOp = reduceSumOp.get();

            // Remove old ReduceSum op
            operation->Disconnect();
        }
        else
        {
            const int64_t zp = ifmConn->quantization.zeroPoints.empty() ? 0 : ifmConn->quantization.zeroPoints[0];
            if ( zp != 0 )
            {
                const auto &ofmQuant = ofmConn->quantization;
                auto ofmType = ofmConn->tensor->Type();
                if ( _constraints->SupportsQuantization(OpType::Sub, Quantization::Unit(), DataType::Int32,
                         Quantization::Unit(), DataType::Int32, ofmQuant, ofmType) )
                {
                    // Replace ReduceSum (zp != 0) with ReduceSum->Sub(zp):
                    // Temporary tensor between ReduceSum and Sub
                    std::shared_ptr<Tensor> reduceSumTens = ofmConn->tensor->Clone();
                    reduceSumTens->SetBuffer(nullptr);
                    reduceSumTens->SetName(ofmConn->tensor->Name() + "_reducesum");
                    reduceSumTens->ChangeType(DataType::Int32);
                    reduceSumTens->SetStorageShape(ofmConn->shape);

                    // Sub op with zero point
                    auto zpTens = CreateConstTensor("zero_point", DataType::Int32, int(ifmConn->shape.Depth() * zp));
                    auto subOp = std::make_shared<Operation>(OpType::Sub);
                    subOp->ConnectInput(TensorUsage::IFM, reduceSumTens).Set(Quantization::Unit());
                    subOp->ConnectInput(TensorUsage::IFM1, zpTens).Set(Quantization::Unit());
                    subOp->CopyOutput(TensorUsage::OFM, *ofmConn);
                    subOp->Output(TensorUsage::OFM)->Set(ofmConn->rounding);
                    RecordOptimisation(*operation, subOp.get());
                    returnOp = subOp.get();

                    // Connect temporary tensor to reduceSum and remove the zero point
                    operation->ConnectOutput(TensorUsage::OFM, reduceSumTens).Set(Quantization::Unit());
                    ifmConn->quantization.zeroPoints[0] = 0;
                }
                else
                {
                    // Replace ReduceSum (zp != 0) with 1x1 Conv2D:
                    //
                    // 1. Reshape to 3D shape (HWC) where C dimension is the dimension to reduce.
                    // 2. 1x1 Conv2D (1x1x1xC weights): HxWxC -> HxWx1.

                    // Reshape to 4D shape (NHWC) where C dimension is the dimension to reduce
                    Shape paddedIfmShape = Shape::PadAxes(ifmConn->shape, 3, 1);
                    const Shape ifmShape3D = ReshapeTo3D(paddedIfmShape, {paddedIfmShape.Size() - 2, 1, 1});
                    const Shape ifmShape4D = Shape::PadAxes(ifmShape3D, 4, 1);

                    // Create an identity 1x1x1xC weights tensor
                    auto weightsBuffer = std::make_shared<Buffer>(std::vector<int8_t>(ifmShape4D.Depth(), 1));
                    auto weightsTens = CreateConstTensor("weights", DataType::Int8, weightsBuffer);
                    weightsTens->SetStorageShape({1, 1, 1, ifmShape4D.Depth()});
                    auto weightsQuant = ifmConn->quantization;
                    weightsQuant.quantMin = {IntegerMin(DataType::Int8)};
                    weightsQuant.quantMax = {IntegerMax(DataType::Int8)};
                    weightsQuant.zeroPoints = {0};
                    weightsQuant.scales = {{1, 0}};  // Identity

                    // Create an identity bias tensor
                    auto biasTens = CreateConstTensor("bias", DataType::Int32, 0);
                    auto biasQuant = ifmConn->quantization;
                    biasQuant.zeroPoints = {0};

                    // Replace ReduceSum with a 1x1 Conv2D
                    Kernel kernel({1, 1}, {1, 1}, {1, 1});
                    auto convOp = std::make_shared<Operation>(OpType::Conv2D);
                    convOp->SetKernel(std::make_unique<Kernel>(kernel));
                    convOp->CopyInput(TensorUsage::IFM, *ifmConn);
                    convOp->Input(TensorUsage::IFM)->Set(ifmShape4D).Set(ifmConn->rounding);

                    convOp->ConnectInput(TensorUsage::Weights, weightsTens).Set(weightsQuant);
                    convOp->ConnectInput(TensorUsage::Scales, biasTens).Set(biasQuant);
                    convOp->CopyOutput(TensorUsage::OFM, *ofmConn);
                    convOp->Output(TensorUsage::OFM)->Set(ifmShape4D.WithDepth(1)).Set(ofmConn->rounding);
                    RecordOptimisation(*operation, convOp.get());
                    returnOp = convOp.get();

                    // Remove old ReduceSum op
                    operation->Disconnect();
                }
            }
            else if ( ifmConn->shape.Size() > 3 )
            {
                // Replace >3D ReduceSum (axis = C) with 3D ReduceSum:
                //
                // 1. Reshape to 3D shape (HWC) where C dimension is the dimension to reduce. For example, 3x5x7x11x13
                //    (5D) becomes 105x11x13 (3D).
                // 2. ReduceSum: HxWxC -> HxWx1.

                // Reshape to 3D shape (HWC) where C dimension is the dimension to reduce
                const Shape ifmShape3D = ReshapeTo3D(ifmConn->shape, {ifmConn->shape.Size() - 2, 1, 1});

                operation->Input(TensorUsage::IFM)->Set(ifmShape3D);
                operation->Output(TensorUsage::OFM)->Set(ifmShape3D.WithDepth(1));
                attr->axis = 2;  // C
            }
        }
    }

    return returnOp;
}

// Decompose Tile with more than one tiled axis
// into several tile operations, each with one tiled axis
Operation *GraphIrOptimiser::RewriteTile(Graph *const, Operation *const operation)
{
    Operation *returnOp = operation;

    const OpType opType = operation->Type();
    if ( opType != OpType::Tile )
    {
        return returnOp;
    }

    auto *ofmConn = operation->Output(TensorUsage::OFM);
    auto *ifmConn = operation->Input(TensorUsage::IFM);
    auto *params = operation->Input(TensorUsage::Params);
    auto *ofm = ofmConn->tensor.get();
    auto *ifm = ifmConn->tensor.get();

    assert(ifmConn);
    assert(ofmConn);
    assert(params);

    // Convert params tensor to vector
    Shape multiples = TensorToShape(params->tensor.get(), params->shape.Elements());

    // axisMask contains ones for every axis that needs to be tiled.
    // e.g. if H,W are tiled, axisMask will be 0110
    unsigned axisMask = multiples.GreaterMask(multiples.WithOnes());

    // We only need to decompose if there is more than one tiled axis
    if ( axisMask == 0 || IsPowerOfTwo(axisMask) )
    {
        return returnOp;
    }

    auto inputConn = ifmConn;
    int axis = ifmConn->shape.Size() - 1;

    while ( axisMask )
    {
        // tile only if the LSB>0
        if ( axisMask & 1 )
        {
            // Create new tile operation that only tiles one of the axes
            int multiplier = multiples[axis];

            // The shape of the intermediate tensor is same as its input-tensor
            // but with one tiled axis (taken from ofm-shape)
            Shape outShape = inputConn->shape;
            outShape[axis] = ofmConn->shape[axis];

            std::vector<int32_t> newMultiples(multiples.Size(), 1);
            newMultiples[axis] = multiplier;

            std::shared_ptr<Tensor> outTens = ofmConn->tensor;
            // create intermediate tensor if this is not the last tiled axis
            if ( (axisMask >> 1) > 0 )
            {
                std::string name(fmt::format("{}_tiled_axis_{}", ofm->Name(), axis));
                outTens = std::make_shared<Tensor>(name, ofm->Type(), outShape);
            }

            auto tileOp = std::make_shared<Operation>(OpType::Tile);
            tileOp->CopyInput(TensorUsage::IFM, *inputConn);
            tileOp->ConnectOutput(TensorUsage::OFM, outTens).Set(outShape);
            // create new param tensor
            auto newParamtensor = CreateConstTensor(
                "multiples", DataType::Int32, std::make_shared<Buffer>(newMultiples.size(), newMultiples.data()));
            tileOp->ConnectInput(TensorUsage::Params, newParamtensor);

            RecordOptimisation(*operation, tileOp.get());
            returnOp = tileOp.get();

            inputConn = tileOp->Output(TensorUsage::OFM);
        }
        axis--;
        axisMask >>= 1;
    }

    operation->Disconnect();
    return returnOp;
}

// Merge adjacent transposes
Operation *GraphIrOptimiser::MergeTransposes(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType == OpType::Transpose )
    {
        auto *ifmConn = operation->Input(TensorUsage::IFM);
        auto *ofmConn = operation->Output(TensorUsage::OFM);
        auto *ifm = ifmConn->tensor.get();
        const auto &ofm = ofmConn->tensor;
        auto *prevOp = ifm->Writers().empty() ? nullptr : ifm->Writers().front().get();

        auto *attr = operation->Attribute<transpose_attr_t>();
        auto curTranspose = TransposeTypeFromShape(attr->perm);
        bool opHasQuant = ofmConn->quantization.IsValid() && !ofmConn->quantization.IsUnitScale();

        // Remove no-op transposes if possible
        if ( IsNone(curTranspose) )
        {
            assert(ofmConn->shape == ifmConn->shape);
            // Transpose is the only operator, it may be peforming memory copy duties.
            if ( !prevOp && ofm->Readers().empty() )
            {
                auto newOp = std::make_shared<Operation>(OpType::MemoryCopy);
                newOp->CopyInput(TensorUsage::IFM0, *ifmConn);
                newOp->CopyOutput(TensorUsage::OFM, *ofmConn);
                operation->Disconnect();
                returnOp = newOp.get();
                RecordOptimisation(*operation, returnOp);
            }
            // Disconnect from surrounding ops, if this is a graph input
            // or output it remains untouched.
            else if ( ifm->IsSinglePath() && !opHasQuant && prevOp )
            {
                ifm->RemoveWriter(prevOp->shared_from_this());
                prevOp->ConnectOutput(TensorUsage::OFM, ofm).Set(ofmConn->slice);
                operation->Disconnect();
                returnOp = prevOp;
            }
            return returnOp;
        }

        // Transpose is fed by a preceding transpose (single writer, single reader)
        if ( prevOp && (prevOp->Type() == OpType::Transpose) && ifm->IsSinglePath() )
        {
            const auto *prevConn = prevOp->Output(TensorUsage::OFM);
            assert(prevConn);

            // Can't merge if predecessor reverses or reshapes
            if ( prevConn->reverse != ReverseType::None || prevConn->shape != ifmConn->shape ) return returnOp;

            // Can't merge if both apply quantization
            bool prevHasQuant = prevConn->quantization.IsValid() && !prevConn->quantization.IsUnitScale();

            if ( opHasQuant && prevHasQuant ) return returnOp;

            // Examine previous op's transpose
            auto *prevAttr = prevOp->Attribute<transpose_attr_t>();
            auto prevTranspose = TransposeTypeFromShape(prevAttr->perm);

            // Apply both transposes to default axes and examine the resulting transpose
            static std::array<int, 8> nhwcDefault = {0, 1, 2, 3, 4, 5, 6, 7};
            int activeAxes = std::min(int(nhwcDefault.size()), ifmConn->shape.Size());

            Shape axes(nhwcDefault.data(), activeAxes);
            Shape prevMapping = axes.Permute(unsigned(prevTranspose));
            Shape finalMapping = prevMapping.Permute(unsigned(curTranspose));
            TransposeType mergedTranspose = TransposeTypeFromShape(finalMapping);

            ArchOperatorQuery query;
            ArchRequirements req;
            query.transposeMask = mergedTranspose;
            Set(query.ifm[0], ifmConn);
            Set(query.ofm, ofmConn);
            if ( _constraints->OperatorQuery(OpType::Transpose, &query, &req).Any(QueryResult::Native) )
            {
                // only merge the transpose if the new mask is natively supported
                // without mask-decomp
                if ( !req.decomposeProps.Any(ArchProperty::TransposeMask) )
                {
                    // Change the transpose attribute on the preceding transpose and remove this one
                    prevAttr->perm = finalMapping;
                    TensorConnection &newConn = prevOp->ConnectOutput(TensorUsage::OFM, ofm);
                    newConn.Set(ofmConn->slice).Set(ofmConn->reverse).Set(ofmConn->shape);
                    if ( !prevHasQuant && opHasQuant ) newConn.Set(ofmConn->quantization);
                    operation->Disconnect();
                    return prevOp;
                }
            }
        }
    }

    return returnOp;
}

// Rearrange transpose
Operation *GraphIrOptimiser::RearrangeTranspose(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType == OpType::Transpose )
    {
        auto *ifmConn = operation->Input(TensorUsage::IFM);
        auto *ofmConn = operation->Output(TensorUsage::OFM);
        auto *attr = operation->Attribute<transpose_attr_t>();

        // If the transpose type is not supported (for example it's transposing in the batch dimension), try to
        // rearrange the IFM and OFM shapes by moving any dimension that is 1 to the left. Then recalculate the
        // transpose mask to match the new shapes.
        //
        // Example 1:
        // Original, with unsupported permutation vector:
        // 128x1x8x128 + [1, 2, 0, 3] -> 1x8x128x128
        // Compact, with supported permutation vector:
        // 1x128x8x128 + [0, 2, 1, 3] ("NWHC") -> 1x8x128x128
        //
        // Example 2:
        // Original, with unsupported permutation vector:
        // 1x8x128x32 + [2, 0, 1, 3] -> 128x1x8x32
        // Compact, with supported permutation vector:
        // 1x8x128x32 + [0, 2, 1, 3] ("NWHC") -> 1x128x8x32
        Shape perm = attr->perm;

        // Don't bother with rearrangement if transpose type is already supported
        ArchOperatorQuery query;
        query.transposeMask = TransposeTypeFromShape(perm);
        if ( _constraints->OperatorQuery(OpType::Transpose, &query, nullptr).Any(QueryResult::Native) )
        {
            return returnOp;
        }

        Shape ifmShape = ifmConn->shape;
        Shape ofmShape = ofmConn->shape;
        int ofmDim = perm.Size() - 1;
        for ( auto onesMask = ofmShape.EqualMask(ofmShape.WithOnes()); onesMask; onesMask >>= 1 )
        {
            if ( onesMask & 1 )
            {
                // Find matching dimension to remove from IFM
                int ifmDim = perm[ofmDim];

                // Remove dimensions from IFM, OFM, perm
                ofmShape = ofmShape.Erase(ofmDim);
                ifmShape = ifmShape.Erase(ifmDim);
                perm = perm.Erase(ofmDim);
                for ( int i = 0; i < perm.Size(); i++ )
                {
                    if ( perm[i] > ifmDim ) perm[i]--;
                }
            }
            ofmDim--;
        }

        attr->perm = perm;
        ifmConn->shape = ifmShape;
        ofmConn->shape = ofmShape;
    }

    return returnOp;
}

// Rewrite Matmul by adding a NHCW transpose for the IFM2-tensor
// Also reshape all Non-WC axes into the height axis.
Operation *GraphIrOptimiser::RewriteMatmul(Graph *const graph, Operation *const operation)
{
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType != OpType::MatMul )
    {
        return returnOp;
    }

    auto *ifmConn = operation->Input(TensorUsage::IFM);
    auto *ifm1Conn = operation->Input(TensorUsage::IFM1);
    auto *ofmConn = operation->Output(TensorUsage::OFM);

    // Reshape non-WC axes into height
    auto ReshapeFunc = [](const Shape &s)
    {
        int height = s.Elements() / s.ElementsWC();
        return Shape(1, height, s.Width(), s.Depth());
    };

    ifmConn->shape = ReshapeFunc(ifmConn->shape);
    ifm1Conn->shape = ReshapeFunc(ifm1Conn->shape);
    ofmConn->shape = ReshapeFunc(ofmConn->shape);

    // If IFM2 producer is already a NHCW transpose
    // and there are no other producers/consumers of ifm2
    // we remove the transpose instead of adding another
    const auto &ifm1Writers = ifm1Conn->tensor->Writers();
    const auto &ifm1Readers = ifm1Conn->tensor->Readers();
    // TODO MLBEDSW-9620: Remove inverse transpose sequences
    if ( (ifm1Readers.size() == 1) && (ifm1Writers.size() == 1) )
    {
        auto producer = ifm1Writers[0];
        if ( producer->Type() == OpType::Transpose )
        {

            auto *attr = producer->Attribute<transpose_attr_t>();
            TransposeType transposeType = TransposeType::None;
            if ( attr->perm.Size() <= 4 )
            {
                transposeType = TransposeTypeFromShape(attr->perm);
            }
            if ( transposeType == TransposeType::NHCW )
            {
                auto *producerIfm = producer->Input(TensorUsage::IFM0);
                operation->ConnectInput(TensorUsage::IFM1, producerIfm->tensor).Set(ifm1Conn->quantization);
                operation->Input(TensorUsage::IFM1)->shape = ReshapeFunc(producerIfm->shape);
                producer->Disconnect();
                return returnOp;
            }
        }
    }

    // Otherwise create new transpose op
    auto transposeOp = std::make_shared<Operation>(OpType::Transpose);
    auto *attr = transposeOp->Attribute<transpose_attr_t>();
    attr->perm = Shape(0, 1, 3, 2);
    const auto &ifm1Shape = ifm1Conn->shape;
    auto transposedIfm1Shape = ifm1Shape.WithWidth(ifm1Shape.Depth()).WithDepth(ifm1Shape.Width());
    auto transposedIfm1 = std::make_shared<Tensor>(ifm1Conn->tensor->Name() + "/" + OpTypeToString(transposeOp->Type()),
        ifm1Conn->tensor->Type(), transposedIfm1Shape);
    transposeOp->ConnectInput(TensorUsage::IFM0, ifm1Conn->tensor).Set(ifm1Shape);
    transposeOp->ConnectOutput(TensorUsage::OFM, transposedIfm1);
    RecordOptimisation(*operation, transposeOp.get());

    // replace IFM2 with transposed output
    operation->ConnectInput(TensorUsage::IFM1, transposedIfm1).Set(ifm1Conn->quantization);

    return returnOp;
}

// Convert depthwise convolutions with a depth multiplier greater than 1 into a single Conv2D if:
// - the input depth is 1; and
// - the output depth equals the depth multiplier; and
// - the weights and bias are constant
Operation *GraphIrOptimiser::RewriteDepthwise(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    if ( operation->Type() == OpType::DepthwiseConv2D )
    {
        const auto weights = operation->Input(TensorUsage::Weights);
        const auto scales = operation->Input(TensorUsage::Scales);
        if ( !weights->tensor->IsConstant() || !scales->tensor->IsConstant() )
        {
            // Do not rewrite if bias or weights are non-constant
            return returnOp;
        }
        const auto ifm = operation->Input(TensorUsage::IFM0);
        const auto ofm = operation->Output(TensorUsage::OFM);
        assert(ifm && ofm && weights && ifm->shape.Depth() > 0);
        const auto wshape = weights->shape;
        const auto multiplier = wshape.Depth() / ifm->shape.Depth();

        if ( ifm && (ifm->shape.Depth() == 1) && (multiplier != 1) && ofm && (ofm->shape.Depth() == multiplier) )
        {
            auto newOp = std::make_shared<Operation>(OpType::Conv2D);
            auto kernel = std::make_unique<Kernel>(*operation->Kernel());
            // Use striding to avoid having to permute the constant data
            weights->shape = Shape(multiplier, wshape.Height(), wshape.Width(), 1);
            weights->slice.shape = weights->shape;
            weights->slice.stride = Shape::GetStridesForShape(wshape, 1).Extract(3, 1, 2, 0);
            newOp->SetKernel(std::move(kernel));

            ReplaceOperation(operation, newOp.get());
            newOp->Output(TensorUsage::OFM)->Set(ofm->rounding);
            returnOp = newOp.get();
            RecordOptimisation(*operation, returnOp);
        }
    }

    return returnOp;
}

// This step does 3 things to fixup WHILE/IF operations to simplify scheduling and HLC generation:
//
// 1. Connect constant inputs to WHILE through a MemoryCopy op to make sure all input tensors to a WHILE has writable
//    memory. This is important later when this operator is converted to a loop with HLCBranch ops in HLCS generator.
//    After each iteration we copy the output back to the input. This wouldn't work if any input is in read-only memory.
// 2. Connect first input to IF through a MemoryCopy op, if it's constant or a graph input, to make sure this input is
//    produced by an operation. This is important so that the HW COND_STATUS register is properly written. This register
//    controls the behavior if OP_BRANCH instruction.
// 3. Reshape inputs and outputs to 3D to ensure the ops will pass the constraints checks.
Operation *GraphIrOptimiser::FixupControlFlow(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    if ( IsControlFlow(operation->Type()) )
    {
        const bool isWhile = operation->Type() == OpType::While;
        const bool isIf = operation->Type() == OpType::If;

        for ( const auto &[ifmUsage, ifmConn] : operation->Inputs().pairs() )
        {
            if ( !IsIFM(ifmUsage) ) continue;

            // Reshape IFM to 3D
            ifmConn.shape = ReshapeToHWC(ifmConn.shape);

            const bool isConstant = ifmConn.tensor->IsConstant();
            const bool isGraphInput = graph->IsInput(ifmConn.tensor.get());
            const bool isIFM0 = ifmUsage == TensorUsage::IFM0;
            if ( (isWhile && isConstant) || (isIf && isIFM0 && (isConstant || isGraphInput)) )
            {
                // Create a MemoryCopy with identity quantization
                auto tmp = std::make_shared<Tensor>("non-constant", ifmConn.tensor->Type(), ifmConn.tensor->StorageShape());
                auto copyOp = std::make_shared<Operation>(OpType::MemoryCopy);
                copyOp->ConnectInput(TensorUsage::IFM, ifmConn.tensor);                // Constant input
                copyOp->ConnectOutput(TensorUsage::OFM, tmp).Set(RoundMode::NATURAL);  // Non-constant output
                RecordOptimisation(*operation, copyOp.get());

                // Replace input with the MemoryCopy output
                operation->ConnectInput(ifmUsage, tmp);
            }
        }

        for ( const auto &[ofmUsage, ofmConn] : operation->Outputs().pairs() )
        {
            if ( IsOFM(ofmUsage) )
            {
                // Reshape OFM to 3D
                ofmConn.shape = ReshapeToHWC(ofmConn.shape);
            }
        }
    }
    return returnOp;
}

// Reshape Reverse with unsupported shape or axis
// If a Reverse has >4D shape, or unsupported axis-parameter
// reshape to a 3D-tensor where W is the reversed axis
Operation *GraphIrOptimiser::ReshapeReverse(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    auto *ifmConn = operation->Input(TensorUsage::IFM);
    auto *ofmConn = operation->Output(TensorUsage::OFM);
    const auto &ofmShape = ofmConn->shape;
    const auto &ifmShape = ifmConn->shape;
    int ofmRank = ofmConn->shape.Size();

    if ( opType != OpType::Reverse )
    {
        return returnOp;
    }

    auto *attr = operation->Attribute<axis_attr_t>();
    int axis = attr->axis;
    // We need to reshape the operation if any of the following are true:
    //     OFM is >4D
    //     OFM is 4D with batch > 1
    //     OFM is 4D with reversed batch
    // TODO MLBEDSW-9621: Use HW-constraint-check instead
    if ( ofmRank > 4 || (ofmRank == 4 && (ofmShape.Batch() > 1 || axis == 0)) )
    {
        assert(ifmShape == ofmShape);
        assert(axis < ofmRank);
        // Reshape reversed axis into W
        // All predecing axes into H
        // All succeeding axes into C
        auto newShape = ReshapeTo3DAroundAxis(ofmShape, axis);
        ifmConn->shape = newShape;
        ofmConn->shape = newShape;
        attr->axis = 1;
        ofmConn->reverse = ReverseType::W;
    }
    return returnOp;
}

/**
 * Rewrite the argmax operation (WxHxC -> WxHx1) to the OpTypes: DepthwiseConv2D -> MaxPool -> LUT -> Cast
 *
 * DepthwiseConv2D:
 * 1x1 kernel with weights that bit shift each value in the input tensor 7 bits to the left. (Weights equal to 1 << 7)
 * A bias corresponding channel index in reverse order is added to the kernel to pack channel information into the
 * value. [0x04, 0x13, 0x0a, 0x02] = [0000 0100, 0001 0011, 0000 1010, 0000 0010] ->
 * [...0010 0000 0000, ...1001 1000 0000, ...0101 0000 0000, ...0001 0000 0000] + [...0011, ...0010, ...0001, ...0000]
 * ->
 * [...0010 0000 0011, ...1001 1000 0010, ...0101 0000 0001, ...0001 0000 0000]
 *
 * MaxPool:
 * The max pool operation selects the maximum value along the channel and flattens the depth to 1 (WxHxC -> WxHx1)
 * [...0010 0000 0011, ...1001 1000 0010, ...0101 0000 0001, ...0001 0000 0000] - > [...1001 1000 0010]
 *
 * LUT:
 * The lookup table is used to retrieve the channel information from the reverse order channel index value in the
 * max pool output tensor (the least significant bytes). [0000 1001 1000 0010] -> [0000 0000 0000 0001] = 0x0001
 *
 * Cast:
 * Finally the value is cast to the correct output type. 0x0001 -> 0x00000001
 */
Operation *GraphIrOptimiser::RewriteArgmax(Graph *const graph, Operation *const operation)
{
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType != OpType::ArgMax )
    {
        return returnOp;
    }
    auto attr = operation->Attribute<axis_attr_t>();
    auto *ifmConn = operation->Input(TensorUsage::IFM);
    auto *ofmConn = operation->Output(TensorUsage::OFM);
    auto &ifmShape = ifmConn->shape;
    auto &ofmShape = ofmConn->shape;

    // If native support exists for argmax, we return the argmax op without decomposing.
    ArchOperatorQuery query;
    ArchRequirements req;
    query.axis = attr->axis;
    query.ifm[0].shape = ifmConn->shape;
    query.ifm[0].type = ifmConn->tensor->Type();
    query.ofm.type = ofmConn->tensor->Type();
    query.ofm.shape = ofmShape;
    auto res = _constraints->OperatorQuery(OpType::ArgMax, &query, &req);

    // Extend OfmShape to match ifmRank
    if ( ofmShape.Size() != ifmShape.Size() )
    {
        ofmShape = ofmShape.Insert(attr->axis, 1);
        assert(ofmShape.Size() == ifmShape.Size());
    }

    if ( res.Any(QueryResult::Unsupported) )
    {
        // Unsupported argmax, there is nothing we can do here.
        ofmShape = query.ofm.shape;  // Restore original ofmShape
        return returnOp;
    }
    else if ( res.Any(QueryResult::Native) && !req.req.Any(ArchRequirement::OpSubstitution) )
    {
        // Reshape IFM and OFM to 3D-tensors where W is the reduced axis
        if ( attr->axis != 1 || ifmConn->shape.Size() != 3 )
        {
            ifmShape = ReshapeTo3DAroundAxis(ifmShape, attr->axis);
            ofmShape = ifmShape.WithWidth(1);
            attr->axis = 1;
        }
        operation->Output(TensorUsage::OFM)->Set(RoundMode::TRUNCATE_TO_LOWER);
        // Update kernel based on reshapes
        std::unique_ptr<Kernel> kernel = std::make_unique<Kernel>(Point2i(ifmShape[1], 1), Point2i(1, 1), Point2i(1, 1));
        operation->SetKernel(std::move(kernel));

        return returnOp;
    }

    // Pad both OFM and IFM to 3D before extracting c, w, h
    ifmShape = Shape::PadAxes(ifmShape, 3, 1);
    ofmShape = Shape::PadAxes(ofmShape, 3, 1);
    int c = ifmShape.Depth();
    int w = ifmShape.Width();
    int h = ifmShape.Height();

    // Create tensors to hold intermediate values for conv, max pool and lut output.
    std::shared_ptr convOutput = std::make_shared<Tensor>("convOutputTensor", DataType::Int16, Shape(h, w, c));
    std::shared_ptr maxPoolOutput = std::make_shared<Tensor>("maxPoolOutputTensor", DataType::Int16, Shape(h, w, 1));
    std::shared_ptr lutOutput = std::make_shared<Tensor>("lutOutputTensor", DataType::Int16, Shape(h, w, 1));

    // Create values for the channel information for the Conv2D bias.
    std::vector<int64_t> reverse_idx(c);
    std::iota(reverse_idx.begin(), reverse_idx.end(), 0);  // reverse_idxs = [0, 1, 2, 3, 4]
    std::reverse(reverse_idx.begin(), reverse_idx.end());  // reverse_idxs = [4, 3, 2, 1, 0]

    // Create a constant tensor with the channel information values.
    auto biasBuffer = std::make_shared<Buffer>(std::move(reverse_idx));
    Shape biasShape(1, 1, 1, c);
    auto convBias = CreateConstTensor("bias", DataType::Int64, biasBuffer, &biasShape);

    // Create weights-tensor with 1x1 kernel.
    Shape weightShape(1, 1, 1, c);
    std::vector<uint8_t> values(weightShape.Elements(), 1 << 7);  // Weights are 128 to mimic the bit shift.
    auto weightBuf = std::make_shared<Buffer>(std::move(values));
    const auto weightTensor = std::make_shared<Tensor>("convOp_unitWeights", DataType::UInt8, weightShape, weightBuf);

    // Create a convolution operation for shifting values 7 bits (multiplying by 2**7)
    auto convOp = std::make_shared<Operation>(OpType::DepthwiseConv2D);
    convOp->SetKernel(std::make_unique<Kernel>(Point2i(1, 1), Point2i(1, 1), Point2i(1, 1)));
    convOp->CopyInput(TensorUsage::IFM, *ifmConn);
    convOp->ConnectInput(TensorUsage::Weights, weightTensor).Set(Quantization::Unit());
    // Add bias to the convolution corresponding to the input channel
    convOp->ConnectInput(TensorUsage::Scales, convBias).Set(Quantization::Unit());
    convOp->ConnectOutput(TensorUsage::OFM, convOutput).Set(ifmConn->quantization);

    // Max pool op. Squash the width dimension into height since max pool can only work in 2D.  1xHxWxC -> 1x(WxH)xCx1
    auto maxPoolOp = std::make_shared<Operation>(OpType::MaxPool);
    int newHeight = h * w;
    maxPoolOp->ConnectInput(TensorUsage::IFM, convOutput).Set(ifmConn->quantization).Set(Shape(1, newHeight, c, 1));
    maxPoolOp->SetKernel(std::make_unique<Kernel>(Point2i(c, 1), Point2i(1, 1), Point2i(1, 1)));
    maxPoolOp->ConnectOutput(TensorUsage::OFM, maxPoolOutput).Set(Quantization::Unit()).Set(Shape(1, newHeight, 1, 1));

    // Create new tensor for LUT
    Shape lutShape(1, 1, 1, 512);
    uint32_t slope = (-128 & 0xFFFF) << 16;
    uint32_t base = c - 1;
    std::vector lutValues(512, slope + base);
    auto lutBuf = std::make_shared<Buffer>(std::move(lutValues));
    auto lutTensor = CreateConstTensor("lutTensor", DataType::UInt32, lutBuf, &lutShape);

    // Use the LUT operator to extract the channel information from the lower 7 bits
    Operation *lutOp = CreateLUT(maxPoolOutput, lutTensor, ifmConn->quantization, ofmConn->quantization, DataType::UInt32);
    lutOp->ConnectInput(TensorUsage::IFM, maxPoolOutput).Set(Quantization::Unit());
    lutOp->ConnectOutput(TensorUsage::OFM, lutOutput).Set(Quantization::Unit()).Set(Shape(h, w, 1));

    // Cast the LUT output back to the OFM type
    auto castOp = std::make_shared<Operation>(OpType::Cast);
    castOp->ConnectInput(TensorUsage::IFM, lutOutput).Set(Quantization::Unit());
    castOp->CopyOutput(TensorUsage::OFM, *ofmConn);

    // Set the return OP to the cast and disconnect the input operation before we return from the function
    returnOp = castOp.get();
    operation->Disconnect();

    return returnOp;
}

// Rewrite 1x1 Resize to Add op with input broadcasted to OFM
Operation *GraphIrOptimiser::RewriteResize(Graph *const, Operation *const operation)
{
    if ( operation->Type() != OpType::Resize )
    {
        return operation;
    }
    auto *ifmConn = operation->Input(TensorUsage::IFM);

    ArchOperatorQuery query;
    query.ifm[0].shape = ifmConn->shape;
    if ( !_constraints->OperatorQuery(OpType::Resize, &query).Any(QueryResult::Unsupported) || ifmConn->shape.ElementsWH() != 1 )
    {
        return operation;
    }

    auto *ofmConn = operation->Output(TensorUsage::OFM);
    auto dtype = ofmConn->tensor->Type();
    int zp = ifmConn->quantization.zeroPoints.empty() ? 0 : ifmConn->quantization.zeroPoints[0];

    std::vector<int8_t> zeroVector(DataTypeStorageSizeBytes(dtype, ofmConn->shape.Elements()), zp);
    auto zeroBuffer = std::make_shared<Buffer>(std::move(zeroVector));
    auto zeroTensor = CreateConstTensor("const_zeros", dtype, zeroBuffer, &ofmConn->shape);

    auto copyOp = std::make_shared<Operation>(OpType::Add);
    ReplaceOperation(operation, copyOp.get());
    copyOp->ConnectInput(TensorUsage::IFM1, zeroTensor).Set(copyOp->Input(TensorUsage::IFM)->quantization);
    copyOp->Output(TensorUsage::OFM)->rounding = RoundMode::DBL;
    RecordOptimisation(*operation, copyOp.get());
    return copyOp.get();
}


static std::shared_ptr<Operation> CreatePadForKernelPadding(OpType type, const Margin &padding, const TensorConnection &ifmConn)
{
    // Translate the kernel padding into a Pad op parameter tensor
    std::vector<int> padVec = {0, 0};
    if ( type == OpType::Conv3D )
    {
        padVec.insert(padVec.end(), {padding.Near(), padding.Far()});
    }
    padVec.insert(padVec.end(), {padding.Top(), padding.Bottom(), padding.Left(), padding.Right(), 0, 0});

    Shape padParamShape = Shape(padVec.size());
    auto buffer = std::make_shared<Buffer>(std::move(padVec));
    auto paddingParams = CreateConstTensor("padding_params", DataType::Int32, buffer, &padParamShape);

    // Create intermediate tensor with the IFM + padding shape
    const auto &ifmShape = ifmConn.shape;
    Shape paddedIfmShape = ifmShape.WithHW(ifmShape.WH() + padding.TL() + padding.BR());
    if ( type == OpType::Conv3D && paddedIfmShape.Size() > 3 )
    {
        paddedIfmShape[-4] += padding.Near() + padding.Far();
    }
    const auto intermediateTensor = std::make_shared<Tensor>("padded_ifm", ifmConn.tensor->Type(), paddedIfmShape);

    // Create Pad operation
    auto padOp = std::make_shared<Operation>(OpType::Pad);
    padOp->CopyInput(TensorUsage::IFM, ifmConn);
    padOp->ConnectInput(TensorUsage::Params, paddingParams);
    padOp->ConnectOutput(TensorUsage::OFM, intermediateTensor);

    // Reset to no quantization on the Pad IFM
    padOp->Input(TensorUsage::IFM)->quantization = Quantization();

    // Pad with the zero point
    const auto &attr = padOp->Attribute<pad_attr_t>();
    attr->pad_const = ifmConn.quantization.zeroPoints[0];

    return padOp;
}

// This function does two things to legalize Convolution-like ops with non-constant weights and/or bias.
//  1. Transposes the IFM to swap height and width if height is significantly larger than width.
//  2. Replaces kernel padding with zero-point values padded around the IFM tensor.
Operation *GraphIrOptimiser::RewriteNonConstWeightOp(Graph *const, Operation *const operation)
{
    OpType opType = operation->Type();
    if ( !IsConvolution(opType) && (opType != OpType::Conv3D) )
    {
        return operation;
    }

    auto *weightsConn = operation->Input(TensorUsage::Weights);
    auto *scalesConn = operation->Input(TensorUsage::Scales);
    if ( weightsConn->tensor->IsConstant() && scalesConn->tensor->IsConstant() )
    {
        return operation;
    }

    auto *ifmConn = operation->Input(TensorUsage::IFM);
    auto *ofmConn = operation->Output(TensorUsage::OFM);

    // Swap height and width if the effective height is more than twice the effective width and the
    // height is not shallow.
    static constexpr int HEIGHT_THRESHOLD = 32;
    auto kernel = operation->Kernel();
    auto effectiveHeight = ifmConn->shape.Height() / kernel->Stride().y;
    auto effectiveWidth = ifmConn->shape.Width() / kernel->Stride().x;
    if ( effectiveHeight > HEIGHT_THRESHOLD && (effectiveHeight > 2 * effectiveWidth) )
    {
        // Transpose IFM
        auto ifmTransposeOp = CreateTranspose(Shape(0, 2, 1, 3), ifmConn->tensor, ifmConn->shape, TensorUsage::IFM);
        operation->ConnectInput(TensorUsage::IFM, ifmTransposeOp->Output(TensorUsage::OFM)->tensor);
        RecordOptimisation(*operation, ifmTransposeOp);

        // Transpose Weight tensor
        auto weightsTransposeOp = CreateTranspose(Shape(0, 2, 1, 3), weightsConn->tensor, weightsConn->shape, TensorUsage::IFM);
        operation->ConnectInput(TensorUsage::Weights, weightsTransposeOp->Output(TensorUsage::OFM)->tensor);
        RecordOptimisation(*operation, weightsTransposeOp);

        // Transpose OFM
        auto ofmTransposeOp = CreateTranspose(Shape(0, 2, 1, 3), ofmConn->tensor, ofmConn->shape, TensorUsage::OFM);
        operation->ConnectOutput(TensorUsage::OFM, ofmTransposeOp->Input(TensorUsage::IFM)->tensor);
        RecordOptimisation(*operation, ofmTransposeOp);

        // Swap kernel parameters
        Point2i newSize(kernel->Size().y, kernel->Size().x);
        Point2i newStride(kernel->Stride().y, kernel->Stride().x);
        Point2i newDilation(kernel->Dilation().y, kernel->Dilation().x);
        auto &pad = operation->Kernel()->Padding();
        Margin newPadding(pad.Left(), pad.Top(), pad.Right(), pad.Bottom());
        auto newKernel = std::make_unique<Kernel>(
            kernel->WithSize(newSize).WithStride(newStride).WithDilation(newDilation).WithPadding(newPadding));
        operation->SetKernel(std::move(newKernel));
    }

    auto &padding = operation->Kernel()->Padding();
    if ( !padding.IsZero() )
    {
        auto padOp = CreatePadForKernelPadding(operation->Type(), padding, *ifmConn);
        assert(padOp->OFM());

        // Remove kernel padding and replace IFM with the padded intermediate tensor
        operation->SetKernel(std::make_unique<Kernel>(operation->Kernel()->WithPadding({})));
        operation->ConnectInput(TensorUsage::IFM, padOp->OFM()->shared_from_this());
        RecordOptimisation(*operation, padOp.get());
    }

    return operation;
}

Operation *GraphIrOptimiser::RewriteConv3D(Graph *const, Operation *const operation)
{
    if ( operation->Type() == OpType::Conv3D )
    {
        const auto kernel = operation->Kernel();
        // A Conv3D can be rewritten as a Conv2D if:
        // - the kernel depth dimension is 1
        // - the stride in the depth dimension is 1
        // - there is no padding in the depth dimension
        if ( kernel->Size3D().z == 1 && kernel->Stride3D().z == 1 && kernel->Padding().Near() == 0 && kernel->Padding().Far() == 0 )
        {
            auto conv2dOp = std::make_shared<Operation>(OpType::Conv2D);
            ReplaceOperation(operation, conv2dOp.get());
            conv2dOp->SetKernel(std::make_unique<Kernel>(kernel->As2D()));
            auto ifmConn = conv2dOp->Input(TensorUsage::IFM0);
            auto weightsConn = conv2dOp->Input(TensorUsage::Weights);
            auto ofmConn = conv2dOp->Output(TensorUsage::OFM);
            // Move the depth dimension to batch
            int batch = ifmConn->shape[0] * ifmConn->shape[1];
            ifmConn->shape = ifmConn->shape.Erase(1).WithBatch(batch);
            ofmConn->shape = ofmConn->shape.Erase(1).WithBatch(batch);
            // Reshape weights to remove depth dimension
            weightsConn->shape = weightsConn->shape.Erase(1);
            weightsConn->tensor->Reshape(weightsConn->shape);
            RecordOptimisation(*operation, conv2dOp.get());
            return conv2dOp.get();
        }
    }
    return operation;
}

Operation *GraphIrOptimiser::ReplaceBroadcastWithAdd(Graph *const, Operation *const operation)
{
    if ( IsBinaryElementwise(operation->Type()) && !_constraints->SupportsDoubleBroadcast() )
    {
        auto *ifm0Conn = operation->Input(TensorUsage::IFM0);
        auto *ifm1Conn = operation->Input(TensorUsage::IFM1);
        auto *ofmConn = operation->Output(TensorUsage::OFM);
        auto &ofmShape = ofmConn->SliceShape();
        auto ifm0ShapePadded = Shape::PadAxes(ifm0Conn->SliceShape(), ofmShape.Size(), 1);
        auto ifm1ShapePadded = Shape::PadAxes(ifm1Conn->SliceShape(), ofmShape.Size(), 1);
        if ( ifm0ShapePadded != ofmShape && ifm1ShapePadded != ofmShape )
        {
            std::shared_ptr<Tensor> newTensor = ofmConn->tensor->Clone();
            newTensor->ChangeType(ifm0Conn->tensor->Type());
            newTensor->SetName(fmt::format("{}_broadcasted", ifm0Conn->tensor->Name()));
            newTensor->SetStorageShape(ofmShape);

            auto broadcastOp = std::make_shared<Operation>(OpType::Add);
            broadcastOp->ConnectInput(TensorUsage::IFM1, ifm0Conn->tensor);

            std::vector<int8_t> zeroVector(DataTypeStorageSizeBytes(ifm0Conn->tensor->Type(), ofmShape.Elements()), 0);
            auto zeroBuffer = std::make_shared<Buffer>(std::move(zeroVector));
            auto constTensor = CreateConstTensor("const_zero", ifm0Conn->tensor->Type(), zeroBuffer, &ofmShape);

            broadcastOp->ConnectInput(TensorUsage::IFM0, constTensor);
            broadcastOp->ConnectOutput(TensorUsage::OFM, newTensor);

            operation->ConnectInput(TensorUsage::IFM0, newTensor);
            RecordOptimisation(*operation, broadcastOp.get());
        }
    }

    return operation;
}

Operation *GraphIrOptimiser::RealiseKernelPadding(Graph *const, Operation *const operation)
{
    if ( !IsConvolution(operation->Type()) && operation->Type() != OpType::Conv3D )
    {
        return operation;
    }

    auto *kernel = operation->Kernel();
    auto *ifmConn = operation->Input(TensorUsage::IFM);
    const auto &kSize = kernel->DilatedWH();
    const auto &padding = kernel->Padding();
    // If the kernel padding places the kernel ENTIRELY in the padded region
    // then that padding must be realised as actual tensor padding for later
    // decomposition to succeed.
    int maxLR = std::max(padding.Left(), padding.Right());
    int maxTB = std::max(padding.Top(), padding.Bottom());
    if ( kSize.x <= maxLR || kSize.y <= maxTB )
    {
        ArchOperatorQuery query{};
        ArchRequirements req{};
        Set(query.ifm[0], operation->Input(TensorUsage::IFM0));
        Set(query.weights, operation->Input(TensorUsage::Weights));
        Set(query.ofm, operation->Output(TensorUsage::OFM));
        if ( _constraints->OperatorQuery(operation->Type(), &query, &req).Any(QueryResult::Native) )
        {
            if ( req.decomposeProps.Any(ArchProperty::TensorAxis, ArchProperty::KernelStride, ArchProperty::KernelDilation) )
            {
                auto padOp = CreatePadForKernelPadding(operation->Type(), padding, *ifmConn);
                assert(padOp->OFM());

                // Remove kernel padding and replace IFM with the padded intermediate tensor
                operation->SetKernel(std::make_unique<Kernel>(operation->Kernel()->WithPadding({})));
                operation->ConnectInput(TensorUsage::IFM, padOp->OFM()->shared_from_this());
                RecordOptimisation(*operation, padOp.get());
            }
        }
    }

    return operation;
}

// Move Split/slice op to consumer
void GraphIrOptimiser::MoveToConsumer(const Operation *const operation, Operation *const cons)
{
    auto *ifmConn = operation->Input(TensorUsage::IFM0);
    auto *ofm = operation->OFM();
    auto *consIfm0 = cons->IFM(0);
    auto *consIfm1 = cons->IFM(1);

    if ( consIfm0 == ofm )
    {
        cons->ConnectInput(TensorUsage::IFM0, ifmConn->tensor).Set(ifmConn->shape).Set(ifmConn->slice);
    }
    else if ( consIfm1 != nullptr && IsBinaryElementwise(cons->Type()) && consIfm1 == ofm )
    {
        cons->ConnectInput(TensorUsage::IFM1, ifmConn->tensor).Set(ifmConn->shape).Set(ifmConn->slice);
    }
}

Operation *GraphIrOptimiser::MoveSplitSliceToConsumer(Graph *const, Operation *const operation)
{
    auto *ifmConn = operation->Input(TensorUsage::IFM0);
    auto *ofmConn = operation->Output(TensorUsage::OFM);

    if ( operation->Type() == OpType::MemoryCopy && ifmConn->slice.offset.Size() > 0 && (ofmConn->reverse == ReverseType::None) )
    {
        auto *ofm = ofmConn->tensor.get();

        if ( ofm->Readers().size() == 1 )
        {
            auto cons = ofm->Readers().front();
            auto *consIfm0 = cons->IFM(0);
            auto *consIfm1 = cons->IFM(1);

            bool ifmShapeEqual = false;
            bool bothHaveIfmStride = false;

            // Don't move to CPU, Reshape or Tile operations
            // low-level implementation of TILE requires unsliced inputs
            if ( cons->Type() == OpType::Passthrough || IsReshape(cons->Type()) || cons->Type() == OpType::Tile ||
                 IsControlFlow(cons->Type()) )
            {
                return operation;
            }

            if ( consIfm0 == ofm )
            {
                // Check if ifm0 consumer has correct shape
                auto *consIfm0Conn = cons->Input(TensorUsage::IFM0);
                ifmShapeEqual = Shape::IsReducedEqual(consIfm0Conn->shape, ofmConn->shape);

                // Check if both ifm and ifm0 consumer have stride
                const auto &ifmStride = ifmConn->slice.stride;
                const auto &conIfmStride = consIfm0Conn->slice.stride;
                bothHaveIfmStride =
                    ifmStride && ifmStride != ifmStride.WithOnes() && conIfmStride && conIfmStride != conIfmStride.WithOnes();
            }
            else if ( consIfm1 != nullptr && consIfm1 == ofm )
            {
                // Check if ifm1 consumer has correct shape
                auto *consIfm1Conn = cons->Input(TensorUsage::IFM1);
                ifmShapeEqual = Shape::IsReducedEqual(consIfm1Conn->shape, ofmConn->shape);

                // Check if both ifm and ifm1 consumer have stride
                const auto &ifmStride = ifmConn->slice.stride;
                const auto &conIfmStride = consIfm1Conn->slice.stride;
                bothHaveIfmStride =
                    ifmStride && ifmStride != ifmStride.WithOnes() && conIfmStride && conIfmStride != conIfmStride.WithOnes();
            }

            // Calculate the consumer transpose type
            TransposeType consumerTranspose = TransposeType::None;
            if ( cons->Type() == OpType::Transpose )
            {
                consumerTranspose = TransposeTypeFromShape(cons->Attribute<transpose_attr_t>()->perm);
            }

            // We can only move to consumer if there is no transpose on the op that we move to,
            // otherwise the IFM shape may change and transposition will be wrong.
            if ( Shape::IsReducedEqual(ofmConn->shape, ofm->StorageShape()) && IsNone(consumerTranspose) && ifmShapeEqual && !bothHaveIfmStride )
            {
                // Split/Slice can be performed by tensor consumer
                MoveToConsumer(operation, cons.get());
            }
        }
    }

    return operation;
}

GraphIrOptimiser::GraphIrOptimiser(IArchitectureConstraints *constraints, const GraphOptimiserOptions &options, OptimiserDatabase *db) :
        GraphOptimiser(constraints, options, db)
{
}

void GraphIrOptimiser::OptimiseGraph(Graph *graph)
{
    for ( auto iOpt = GraphOptimisationSteps().begin(); iOpt != GraphOptimisationSteps().end(); ++iOpt )
    {
        LOG_TRACE1("GraphOptimiser {0}/{1}\n", std::distance(GraphOptimisationSteps().begin(), iOpt) + 1,
            GraphOptimisationSteps().size());
        // Check if function lists are empty. Do not call for step that only contain disabled debug functions.
        if ( !iOpt->opFunction.empty() || !iOpt->tensorFunction.empty() )
        {
            RewriteGraph<GraphIrOptimiser>(graph, *iOpt);
        }
    }
}

}  // namespace regor
