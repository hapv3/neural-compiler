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
    if ( tensor->Type() == DataType::Int48 && !tensor->IsConstant() )
    {
        tensor->ChangeType(DataType::Int64);
    }
    else if ( tensor->Type() == DataType::UInt48 && !tensor->IsConstant() )
    {
        tensor->ChangeType(DataType::UInt64);
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
            newTensor->SetName(newTensor->Name() + "_int8");
            ReplaceConsumerInput(nullptr, graphInputTensor->Readers(), graphInputTensor.get(), newTensor);

            // Create and insert an elementwise CMP_NE to convert to internal bool representation
            auto newOp = std::make_shared<Operation>(OpType::NotEqual);
            newOp->ConnectInput(TensorUsage::IFM0, graphInputTensor);
            newOp->ConnectInput(TensorUsage::IFM1, CreateConstTensor("const_zero", int8_t(0)));
            newOp->ConnectOutput(TensorUsage::OFM, newTensor);
            RecordOptimisation(graph, newOp.get());
            returnTensor = graphInputTensor.get();
        }
        else if ( graph->IsOutput(tensor) )
        {
            // Replace the OFM of ops producing the graph output tensor
            std::shared_ptr<Tensor> newTensor = tensor->Clone();
            newTensor->SetName(newTensor->Name() + "_int8");
            std::shared_ptr<Tensor> graphOutputTensor = tensor->shared_from_this();
            ReplaceProducerOutput(graphOutputTensor->Writers(), graphOutputTensor.get(), newTensor);

            // Create and insert an elementwise BITWISE_AND to convert from internal bool representation
            auto newOp = std::make_shared<Operation>(OpType::And);
            newOp->ConnectInput(TensorUsage::IFM0, newTensor);
            newOp->ConnectInput(TensorUsage::IFM1, CreateConstTensor("const_one", int8_t(1)));
            newOp->ConnectOutput(TensorUsage::OFM, graphOutputTensor);
            RecordOptimisation(graph, newOp.get());
            returnTensor = newTensor.get();
        }
    }
    return returnTensor;
}

Tensor *GraphIrOptimiser::ConvertInt4Tensors(Graph *graph, Tensor *tensor)
{
    Tensor *returnTensor = tensor;
    if ( tensor->Type() == DataType::Int4Packed8 && tensor->IsConstant() )
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
    return returnTensor;
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
        auto mask = ToReverseMask(attr->axis, ofmRank);
        assert(mask != ReverseType::Dynamic && "Unexpected dynamic reverse axis.");
        assert((mask == ReverseType::None || IsPowerOfTwo(unsigned(mask))) && "Reverse operation can only have one axis");
        ofmConn->reverse = mask;
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
struct EwShl
{
    T operator()(T a, T b)
    {
        assert(b >= 0);
        return T(std::make_unsigned_t<T>(a) << std::make_unsigned_t<T>(b));
    }
};

template<typename T>
static std::vector<T> BroadcastValues(const Tensor *in, const Shape &oShape)
{
    const Shape &iShape = in->StorageShape();
    const auto &iData = in->View().Values<T>();
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

template<template<typename> typename F, typename T>
std::shared_ptr<Buffer> ConstPropEw(Operation *const operation)
{
    auto ifmConn0 = operation->Input(TensorUsage::IFM);
    auto ifmConn1 = operation->Input(TensorUsage::IFM1);
    auto ofmConn = operation->Output(TensorUsage::OFM);
    const auto &oShape = ofmConn->tensor->StorageShape();
    auto *ifm0 = ifmConn0->tensor.get();
    auto *ifm1 = ifmConn1->tensor.get();
    auto *ofm = ofmConn->tensor.get();

    auto v0 = BroadcastValues<T>(ifm0, oShape);
    auto v1 = BroadcastValues<T>(ifm1, oShape);
    std::vector<T> c(oShape.Elements());

    for ( int i = 0; i < oShape.Elements(); i++ )
    {
        c[i] = F<T>()(v0[i], v1[i]);
    }

    return std::make_shared<Buffer>(std::move(c));
}

template<template<typename> typename F>
std::shared_ptr<Buffer> ConstPropEw(Operation *const operation)
{
    auto dataType = operation->Output(TensorUsage::OFM)->tensor->Type();

    switch ( dataType )
    {
        case DataType::Int8:
        {
            return ConstPropEw<F, int8_t>(operation);
        }
        break;
        case DataType::Int16:
        {
            return ConstPropEw<F, int16_t>(operation);
        }
        break;
        case DataType::Int32:
        {
            return ConstPropEw<F, int32_t>(operation);
        }
        break;
        default:
            return {};
            break;
    }
}

Operation *GraphIrOptimiser::ConstPropagation(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    for ( auto [usage, ifmConn] : operation->Inputs().pairs() )
    {
        if ( !IsIFM(usage) ) continue;

        if ( !ifmConn.tensor->IsConstant() )
        {
            return operation;
        }
    }

    // Op has only constant input and result can be computed
    std::shared_ptr<Buffer> ofmBuf;
    switch ( operation->Type() )
    {
        case OpType::SHL:
        {
            ofmBuf = ConstPropEw<EwShl>(operation);
        }
        break;
        default:
            break;
    }

    if ( ofmBuf )
    {
        auto *ofmConn = operation->Output(TensorUsage::OFM);
        auto *ofm = ofmConn->tensor.get();
        ofm->SetBuffer(ofmBuf);

        // Remove op from ifm readers and ofm writers.
        // Note the Inputs/Outputs on operation should still be intact to not break the traversal.
        for ( auto [usage, ifmConn] : operation->Inputs().pairs() )
        {
            ifmConn.tensor->RemoveReader(operation->shared_from_this());
        }
        ofm->RemoveWriter(operation->shared_from_this());
    }

    return operation;
}

Operation *GraphIrOptimiser::RemoveReshape(Graph *const graph, Operation *const operation)
{
    Operation *returnOp = operation;
    OpType opType = operation->Type();

    if ( IsReshape(opType) )
    {
        auto *ifmConn = operation->Input(TensorUsage::IFM0);
        auto *ofmConn = operation->Output(TensorUsage::OFM);
        auto *ifm = ifmConn->tensor.get();
        auto *ofm = ofmConn->tensor.get();

        // Check if ifm/ofm are network ifm/ofm or constant
        bool isIfmConst = ifm->IsConstant();
        bool isIfmSgIfm = IsTensorInVector(graph->Inputs(), ifm);
        bool isOfmSgOfm = IsTensorInVector(graph->Outputs(), ofm);
        bool isIfmSgOfm = IsTensorInVector(graph->Outputs(), ifm);

        // TODO: MLBEDSW-9069: Check CPU operator producer/consumer
        // Inserts a copy op if needed before removing reshapes.
        if ( ((isIfmSgIfm || isIfmSgOfm || isIfmConst) && (isOfmSgOfm)) ||
             ((ifm->Readers().size() > 1) && (ifm->StorageShape() != ofm->StorageShape() || ifm->AxisOrder() != ofm->AxisOrder())) )
        {
            auto copyOp = InsertCopyOpAfterTensor(ifmConn->tensor, ifmConn->quantization);
            // reset the ifm to reflect the reshape's new ifm
            ifmConn = operation->Input(TensorUsage::IFM0);
            ifm = ifmConn->tensor.get();
            returnOp = copyOp.get();
            RecordOptimisation(operation, returnOp);
            // Reshape still needs to be removed.
        }

        // Remove the reshape and one of the tensors.
        if ( isOfmSgOfm )
        {
            // TODO: This path should also be used for ofm tensors consumed by CPU ops.

            // The OFM is in graph outputs, do not remove this tensor.
            // Bypass by replacing ifm with ofm.
            // Set OFM as output for IFM producers
            ReplaceProducerOutput(ifm->Writers(), ifm, ofmConn->tensor);

            // Set OFM as input to other IFM consumers.
            ReplaceConsumerInput(operation, ifm->Readers(), ifm, ofmConn->tensor);
        }
        else
        {
            // Bypass by replacing ofm with ifm.
            // Set IFM as input to OFM consumers.
            ReplaceConsumerInput(nullptr, ofm->Readers(), ofm, ifmConn->tensor);
            assert(ifm->AxisOrder() == AxisOrder::Unknown || ifm->AxisOrder() == ofm->AxisOrder());
            // This is needed as we use the weight tensor, and not the tensor connection,
            // during weight encode. MLBEDSW-9267
            ifmConn->tensor->SetAxisOrder(ofm->AxisOrder());
            ifmConn->tensor->Reshape(ofm->StorageShape());
        }
        // Remove the reshape from ifm readers and ofm writers.
        // Note the Inputs/Outputs on operation should still be intact to not break the traversal.
        ifm->RemoveReader(operation->shared_from_this());
        ofm->RemoveWriter(operation->shared_from_this());
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
             kernel->Stride().AreaXY() == 1 && kernel->DilatedWH().AreaXY() == 1) )
    {
        const auto &weights = operation->Input(TensorUsage::Weights);
        const auto &shape = weights->tensor->StorageShape();
        if ( weights->tensor->AxisOrder() == AxisOrder::OI && shape.Size() == 2 )
        {
            // Reshape weight tensor from (num_outputs, ..., num_inputs) to (num_outputs, 1, 1, num_inputs)
            weights->tensor->SetAxisOrder(AxisOrder::OHWI);
            weights->tensor->Reshape(Shape(shape[0], 1, 1, shape[-1]));
        }
        assert(weights->tensor->AxisOrder() == AxisOrder::OHWI);

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

Operation *GraphIrOptimiser::RewriteRescaleInputs(Graph *const, Operation *const operation)
{
    Operation *returnOp = operation;
    OpType opType = operation->Type();
    if ( opType == OpType::Rescale )
    {
        auto ofmConn = operation->Output(TensorUsage::OFM);
        auto ifmConn = operation->Input(TensorUsage::IFM);
        auto mulConn = operation->Input(TensorUsage::Params);
        auto shiftConn = operation->Input(TensorUsage::Params1);
        auto mulView = mulConn->tensor->View();
        auto shiftView = shiftConn->tensor->View();
        auto inT = ifmConn->tensor->Type();
        auto mulT = mulConn->tensor->Type();
        auto shiftT = shiftConn->tensor->Type();
        assert(mulT == DataType::Int16 || mulT == DataType::Int32);
        assert(shiftT == DataType::Int8);
        std::vector<QuantizedScale> newScale;
        auto *attr = operation->Attribute<rescale_attr_t>();
        int channels = attr->per_channel ? ofmConn->shape.Depth() : 1;
        for ( int i = 0; i < channels; i++ )
        {
            QuantizedScale qScale;
            int32_t scale = mulT == DataType::Int32 ? mulView.Values<int32_t>()[i] : mulView.Values<int16_t>()[i];
            int32_t shift = shiftView.Values<int8_t>()[i];
            assert(attr->scale32 || static_cast<int16_t>(scale) == scale);
            assert(static_cast<int8_t>(shift) == shift);

            qScale.scale = attr->scale32 ? scale : static_cast<int16_t>(scale);
            qScale.shift = shift;
            newScale.emplace_back(qScale);
        }
        ofmConn->quantization.scales = std::move(newScale);
        auto rescaleOp = operation->shared_from_this();
        rescaleOp->DisconnectInputInvalidatingInputs(TensorUsage::Params);
        rescaleOp->DisconnectInputInvalidatingInputs(TensorUsage::Params1);
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
                std::shared_ptr<Tensor> ifm32Tens = ifmConn->tensor->Clone();

                castOp->ConnectInput(TensorUsage::IFM, ifmConn->tensor).quantization.zeroPoints = ifmConn->quantization.zeroPoints;
                ifmConn->quantization.zeroPoints.clear();
                ifmConn->quantization.zeroPoints.push_back(0);

                castOp->ConnectOutput(TensorUsage::OFM, ifm32Tens);
                auto castAttr = castOp->Attribute<sign_attr_t>();

                // move input_unsigned to cast input
                castAttr->input_unsigned = signAttr->input_unsigned;
                signAttr->input_unsigned = false;

                ifm32Tens->ChangeType(DataType::Int32);
                RecordOptimisation(operation, castOp.get());
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
                // try to right-shift scale without precision-loss
                // This can be done if the scale is evenly divisible by 2^shift
                if ( (shiftDiff >= 0) && (qscale.scale % (1UL << shiftDiff) == 0) )
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
                    RecordOptimisation(operation, mulOp.get());

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
            RecordOptimisation(operation, mulOp.get());
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
        uint8_t zeroPoint = ofmConn->quantization.IsValid() ? uint8_t(ofmConn->quantization.zeroPoints[0]) : 0;
        const int padConst = int(attr->pad_const) + zeroPoint;

        // Decode the padding before and after each dimension as two shapes
        Shape paddingBefore = TensorToShape(paramsConn->tensor.get(), paramsConn->shape.Width(), 2, 0);
        Shape paddingAfter = TensorToShape(paramsConn->tensor.get(), paramsConn->shape.Width(), 2, 1);

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
                RecordOptimisation(operation, fillOp);
            }

            const int padAfter = paddingAfter[axis];
            if ( padAfter )
            {
                TensorSlice newOfmSlice = {zeroShape.With(padAxis, newOfmShape[padAxis] - padAfter), newOfmShape.With(padAxis, padAfter)};
                auto fillOp = MakeFillOperation(ofmConn, newOfmShape, newOfmSlice, padTensor);
                RecordOptimisation(operation, fillOp);
            }
        }

        // Copy original IFM to OFM
        auto copyOp = std::make_shared<Operation>(OpType::MemoryCopy);
        copyOp->CopyInput(TensorUsage::IFM, *ifmConn);
        copyOp->CopyOutput(TensorUsage::OFM, *ofmConn);
        copyOp->Output(TensorUsage::OFM)->Set({paddingBefore, ifmShape}).Set(RoundMode::NATURAL);
        RecordOptimisation(operation, copyOp.get());
        returnOp = copyOp.get();

        // Remove original pad
        operation->Disconnect();
    }
    return returnOp;
}

Operation *GraphIrOptimiser::UnrollConv(Graph *const, Operation *const operation)
{
    auto returnOp = operation;

    if ( operation->Type() == OpType::Conv2D )
    {
        const auto ifmConn = operation->Input(TensorUsage::IFM);
        assert(ifmConn);
        const auto weightsConn = operation->Input(TensorUsage::Weights);
        assert(weightsConn);
        const auto scalesConn = operation->Input(TensorUsage::Scales);
        assert(scalesConn);
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
        const bool hasIfmSlice = ifmConn->slice.shape.IsValid() || ifmConn->slice.offset.IsValid();
        const bool hasOfmSlice = ofmConn->slice.shape.IsValid() || ofmConn->slice.offset.IsValid();

        tflite::Padding paddingType = tflite::Padding::VALID;
        const tflite::Operator *const passthrough = static_cast<const tflite::Operator *>(operation->Passthrough());
        if ( passthrough )
        {
            const auto options = passthrough->builtin_options_as_Conv2DOptions();
            if ( options )
            {
                paddingType = options->padding();
            }
        }

        // Figure out if op needs to be unrolled
        const bool needUnrollH = stride_h > 3;
        const bool needUnrollW = stride_w > 3;

        // Figure out if op can be unrolled
        const bool canUnroll = !hasPadding && !hasIfmSlice && !hasOfmSlice && paddingType == tflite::Padding::VALID;
        const bool canUnrollH = dilation_h == 1 && canUnroll;
        const bool canUnrollW = dilation_w == 1 && canUnroll;

        if ( (needUnrollH || needUnrollW) && canUnrollH && canUnrollW )
        {
            const Shape inputGridCell = ifmConn->shape.WithHeight(kernel_h).WithWidth(kernel_w);
            const Shape outputGridCell = ofmConn->shape.WithHeight(1).WithWidth(1);
            const Point2i gridSize = ofmConn->shape.WH<int>();

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
                    op->CopyInput(TensorUsage::Weights, *weightsConn);
                    op->CopyInput(TensorUsage::Scales, *scalesConn);
                    op->CopyOutput(TensorUsage::OFM, *ofmConn);
                    op->Output(TensorUsage::OFM)->Set(ofmSlice);
                    RecordOptimisation(operation, op.get());

                    returnOp = op.get();
                }
            }

            // Remove original op
            operation->Disconnect();
        }
    }

    return returnOp;
}

/// @brief Moves Rescale operations to the output of the previous operation
///        or the input of the next operation when possible.
/// @param
/// @param operation Operation to optimise
/// @return (Possibly) optimised operation
Operation *GraphIrOptimiser::FuseRescale(Graph *const graph, Operation *const operation)
{
    Operation *returnOp = operation;
    OpType opType = operation->Type();
    if ( opType == OpType::Rescale )
    {
        auto *attr = operation->Attribute<rescale_attr_t>();
        auto *signAttr = operation->Attribute<sign_attr_t>();
        if ( signAttr && (signAttr->input_unsigned || signAttr->output_unsigned) )
        {
            // These type of rescales needs special handling and cannot be fused
            return returnOp;
        }

        auto ofmConn = operation->Output(TensorUsage::OFM);
        auto ifmConn = operation->Input(TensorUsage::IFM);
        auto producer = ifmConn->tensor->Writers().size() == 1 ? ifmConn->tensor->Writers().front() : nullptr;

        // Convert scales to have 0 shift if possible, since this can improve fusing for Ethos-U55/65
        auto ConvertedScales = [](const TensorConnection *conn)
        {
            auto scales = conn->quantization.scales;
            for ( auto &qs : scales )
            {
                if ( qs.shift > 0 && qs.shift < 31 && (qs.scale % (1 << qs.shift)) == 0 )
                {
                    qs = {(qs.scale >> qs.shift), 0};
                }
            }
            return scales;
        };

        // Check if there is only one consumer of the output of the rescale and try to fuse to that operation.
        // Do not fuse if the connecting tensor is also a graph-output.
        // Note (ZeroPoints): For input fusing we cannot have an output zero point on the Rescale operation (since the
        //       zero point is applied before scaling on inputs), however input zero point is fine.
        if ( ofmConn->tensor->Readers().size() == 1 && ofmConn->quantization.zeroPoints == Quantization::Unit().zeroPoints &&
             !IsTensorInVector(graph->Outputs(), ofmConn->tensor.get()) )
        {
            // Propagate rescaling to input of next op
            auto consumer = ofmConn->tensor->Readers().front();

            // Copy quantization information from ifm connection and (converted) scales from ofm connection,
            // since these are the scales we want to apply.
            auto ifmQuant = ifmConn->quantization;
            // Normalize scales to shift 0 if possible
            ifmQuant.scales = ConvertedScales(ofmConn);

            for ( auto ifm : consumer->Inputs().pairs() )
            {
                if ( ifm.second.tensor == ofmConn->tensor )
                {
                    auto consumerOfmCon = consumer->Output(TensorUsage::OFM);
                    // This is the input of the next operation that consumes the rescaled values,
                    // check that this input does not already have scaling and that fusing is allowed
                    // by the constraints of the architecture.
                    if ( ifm.second.quantization.EqualScales(Quantization::Unit()) &&
                         _constraints->SupportsFusedRescale(consumer->Type(), TensorUsage::IFM, ifmConn->tensor->Type(),
                             ofmConn->tensor->Type(), ifm.second.tensor->Type(), consumerOfmCon->tensor->Type(), ifmQuant) )
                    {
                        // If the consumer is a binary elementwise make sure that both inputs have
                        // the same data type after fusing.
                        if ( IsBinaryElementwise(consumer->Type()) )
                        {
                            auto consumerOtherIfmCon = consumer->Input(
                                ofmConn->tensor.get() == consumer->IFM(0) ? TensorUsage::IFM1 : TensorUsage::IFM0);

                            auto otherProducer =
                                consumerOtherIfmCon->tensor->Writers().size() == 1 ?
                                    consumerOtherIfmCon->tensor->Writers().front() :
                                    nullptr;

                            // Both ifms must have same type after fusing (no rescale fused to this input case)
                            bool sameType = consumerOtherIfmCon->tensor->Type() == ifmConn->tensor->Type();
                            // Is there a Rescale for the other ifm
                            if ( otherProducer && otherProducer->Type() == OpType::Rescale )
                            {
                                // Check if the other ifm rescale can be fused
                                auto opSign = otherProducer->Attribute<sign_attr_t>();
                                bool otherFusedTensorInGraphOutputs = IsTensorInVector(
                                    graph->Outputs(), consumerOtherIfmCon->tensor.get());
                                bool otherRescaleUnsigned = opSign ? (opSign->input_unsigned || opSign->output_unsigned) : false;

                                auto otherIfmQuant = otherProducer->Input(TensorUsage::IFM)->quantization;
                                otherIfmQuant.scales = ConvertedScales(otherProducer->Output(TensorUsage::OFM));
                                if ( consumerOtherIfmCon->quantization.EqualScales(Quantization::Unit()) && !otherRescaleUnsigned && !otherFusedTensorInGraphOutputs &&
                                     _constraints->SupportsFusedRescale(consumer->Type(), TensorUsage::IFM,
                                         otherProducer->IFM(0)->Type(), otherProducer->OFM()->Type(),
                                         consumerOtherIfmCon->tensor->Type(), consumerOfmCon->tensor->Type(), otherIfmQuant) )
                                {
                                    // If other input can be fused, check that both ifms will have the same type,
                                    // either when this rescale is fused or not.
                                    sameType = sameType || otherProducer->IFM(0)->Type() == ifmConn->tensor->Type();
                                }
                            }
                            // If the inputs have different types when fusing we will have to
                            // avoid performing this fuse operation.
                            if ( !sameType ) break;
                        }
                        consumer->CopyInput(ifm.first, *ifmConn);
                        ifm.second.quantization = ifmQuant;
                        consumer->Input(ifm.first)->Set(ofmConn->rounding);
                        returnOp = consumer.get();
                    }
                }
            }
        }
        // If the rescale could not be fused to the consumer of the output of the rescale, check if there
        // is only one producer of the input to the rescale operation. If this input has no zero point
        // adjustment and the producers output has unit scaling, it might be possible to fuse this rescale to
        // the producers output if the constraints of the architecture allows it

        // Normalize scales to shift 0 if possible
        auto ofmQuant = ofmConn->quantization;
        ofmQuant.scales = ConvertedScales(ofmConn);

        if ( returnOp == operation && producer && ifmConn->tensor->Readers().size() == 1 &&
             producer->Output(TensorUsage::OFM)->quantization.EqualScales(Quantization::Unit()) &&
             ifmConn->quantization.zeroPoints == Quantization::Unit().zeroPoints &&
             // fused tensor cannot be in graph-outputs
             !IsTensorInVector(graph->Outputs(), ifmConn->tensor.get()) &&
             _constraints->SupportsFusedRescale(producer->Type(), TensorUsage::OFM, ifmConn->tensor->Type(),
                 ofmConn->tensor->Type(), producer->IFM(0)->Type(), producer->OFM()->Type(), ofmQuant) )
        {
            // Propagate rescaling to output of previous op
            producer->CopyOutput(TensorUsage::OFM, *ofmConn);
            producer->Output(TensorUsage::OFM)->Set(ofmConn->rounding).Set(ofmQuant);
            returnOp = producer.get();
        }
    }
    if ( returnOp != operation )
    {
        RecordOptimisation(operation, returnOp);
        operation->Disconnect();
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
        if ( kernel->Size() == kernel->Stride() && ifm->shape.Size() >= 3 && kernel->Stride() == ifm->shape.WH<int>() &&
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

        // Replace TOSA Table op with GraphIR LUT op
        returnOp = CreateLUT(ifmConn->tensor, newLutTensor, ifmConn->quantization, ofmConn->quantization,
            newLutTensor->Type(), &ifmConn->shape, ofmConn->tensor, ifmConn->slice, ofmConn->slice);
        returnOp->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
        operation->Disconnect();
    }
    return returnOp;
}

// Rewrite TOSA Cast to other ops
Operation *GraphIrOptimiser::RewriteCast(Graph *const, Operation *const operation)
{
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType == OpType::Cast )
    {
        const auto ifmConn = operation->Input(TensorUsage::IFM);
        const auto ofmConn = operation->Output(TensorUsage::OFM);

        if ( IsBool(ifmConn->tensor->Type()) && IsInteger(ofmConn->tensor->Type()) )
        {
            // Replace CAST with BITWISE_AND to convert from internal bool representation to integer
            auto newOp = std::make_shared<Operation>(OpType::And);
            newOp->CopyInput(TensorUsage::IFM0, *ifmConn);
            newOp->ConnectInput(TensorUsage::IFM1, CreateConstTensor("const_one", int8_t(1)));
            newOp->CopyOutput(TensorUsage::OFM, *ofmConn);
            RecordOptimisation(operation, newOp.get());
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
            RecordOptimisation(operation, newOp.get());
            operation->Disconnect();
            returnOp = newOp.get();
        }
        else
        {
            // Replace CAST with ADD
            auto copyOp = std::make_shared<Operation>(OpType::Add);
            ReplaceOperation(operation, copyOp.get());
            copyOp->ConnectInput(TensorUsage::IFM1, CreateConstTensor("const_zero", ifmConn->tensor->Type(), 0));
            RecordOptimisation(operation, copyOp.get());
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

Operation *GraphIrOptimiser::OptimiseElementwise(Graph *const, Operation *const operation)
{
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( DecomposeAsElementwise(opType) )
    {
        auto ofmShape = operation->Output(TensorUsage::OFM)->shape;
        if ( ofmShape.Size() > 3 && (ofmShape.Depth() == 1 || ofmShape.Width() == 1 || ofmShape.Height() == 1) &&
             ofmShape.Elements() > ofmShape.Depth() * ofmShape.Width() * ofmShape.Height() )
        {
            auto *ofmConn = returnOp->Output(TensorUsage::OFM);
            auto *ifmConn = returnOp->Input(TensorUsage::IFM0);
            auto *ifm2Conn = returnOp->Input(TensorUsage::IFM1);
            while ( ofmShape.Size() > 3 && (ofmShape.Depth() == 1 || ofmShape.Width() == 1 || ofmShape.Height() == 1) )
            {
                auto Reshape = [&](int indexToRemove)
                {
                    ofmShape = ofmShape.Erase(indexToRemove);
                    ofmConn->Set(ofmShape);
                    if ( ifmConn->shape.Size() > indexToRemove )
                    {
                        ifmConn->Set(ifmConn->shape.Erase(indexToRemove));
                    }
                    if ( ifm2Conn && ifm2Conn->shape.Size() > indexToRemove )
                    {
                        ifm2Conn->Set(ifm2Conn->shape.Erase(indexToRemove));
                    }
                };
                if ( ofmShape.Depth() == 1 )
                {
                    Reshape(ofmShape.Size() - 1);
                }
                else if ( ofmShape.Width() == 1 )
                {
                    Reshape(ofmShape.Size() - 2);
                }
                else if ( ofmShape.Height() == 1 )
                {
                    Reshape(ofmShape.Size() - 3);
                }
            }
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
            RecordOptimisation(operation, copyOp.get());
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
        const auto *attr = operation->Attribute<slice_attr_t>();
        const Shape begin = attr->begin;
        const Shape size = attr->size;

        // Replace SLICE with a memory copy with IFM slice
        auto copyOp = std::make_shared<Operation>(OpType::MemoryCopy);
        copyOp->CopyInput(TensorUsage::IFM, *ifmConn);
        copyOp->Input(TensorUsage::IFM)->Set({begin, size});
        copyOp->CopyOutput(TensorUsage::OFM, *ofmConn);
        copyOp->Output(TensorUsage::OFM)->Set(RoundMode::NATURAL);
        RecordOptimisation(operation, copyOp.get());
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
        RecordOptimisation(operation, newOp.get());
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
            RecordOptimisation(operation, addOp);
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
        RecordOptimisation(operation, andOp);
        RecordOptimisation(operation, andNotOp);
        RecordOptimisation(operation, orOp);
        returnOp = orOp;

        // Remove old select op
        operation->Disconnect();
    }
    return returnOp;
}

// Rewrite REDUCE_{MIN,MAX,ANY,ALL} IFM/OFM shapes and set a kernel matching the axis to reduce
Operation *GraphIrOptimiser::RewriteReduceMinMaxAnyAll(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    const OpType opType = operation->Type();
    if ( opType == OpType::ReduceMin || opType == OpType::ReduceMax || opType == OpType::ReduceAny || opType == OpType::ReduceAll )
    {
        auto *ifmConn = operation->Input(TensorUsage::IFM);
        auto *ofmConn = operation->Output(TensorUsage::OFM);
        auto *attr = operation->Attribute<axis_attr_t>();
        auto axis = attr->axis;
        if ( axis < 0 ) axis = ifmConn->shape.Size() + axis;
        assert(axis >= 0);
        assert(axis < ifmConn->shape.Size());

        // Reshape IFM/OFM so IFM is HxWxC and OFM is Hx1xC
        ifmConn->shape = ReshapeTo3DAroundAxis(ifmConn->shape, axis);
        ofmConn->shape = ifmConn->shape.WithWidth(1);

        // Update the axis to reduce to match the reshapes shapes
        attr->axis = 1;

        // Set kernel to 1xW (where W is the width of the reshaped shapes)
        auto kernel = operation->Kernel()->WithSize({ifmConn->shape.Width() /* W */, 1 /* H */});
        operation->SetKernel(std::make_unique<Kernel>(std::move(kernel)));
    }
    return returnOp;
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
            transposeTens->SetName(ifmConn->tensor->Name() + "_transpose");
            transposeTens->SetStorageShape(ifmShape3D.Extract(0, 2, 1));

            // Create Transpose op
            auto transposeOp = std::make_shared<Operation>(OpType::Transpose);
            auto transposeAttr = transposeOp->Attribute<transpose_attr_t>();
            transposeAttr->perm = {0, 2, 1};  // HCW
            transposeOp->CopyInput(TensorUsage::IFM, *ifmConn);
            transposeOp->Input(TensorUsage::IFM)->Set(ifmShape3D).Set(Quantization::Unit());
            transposeOp->ConnectOutput(TensorUsage::OFM, transposeTens);
            RecordOptimisation(operation, transposeOp.get());

            // Create ReduceSum op
            auto reduceSumOp = std::make_shared<Operation>(OpType::ReduceSum);
            auto reduceAttr = reduceSumOp->Attribute<axis_attr_t>();
            reduceAttr->axis = 2;  // C
            reduceSumOp->ConnectInput(TensorUsage::IFM, transposeTens).Set(ifmConn->quantization).Set(ifmConn->rounding);
            reduceSumOp->CopyOutput(TensorUsage::OFM, *ofmConn);
            reduceSumOp->Output(TensorUsage::OFM)->Set(transposeTens->StorageShape().WithDepth(1)).Set(ofmConn->rounding);
            RecordOptimisation(operation, reduceSumOp.get());
            returnOp = reduceSumOp.get();

            // Remove old ReduceSum op
            operation->Disconnect();
        }
        else
        {
            const int64_t zp = ifmConn->quantization.zeroPoints.empty() ? 0 : ifmConn->quantization.zeroPoints[0];
            if ( zp != 0 )
            {
                if ( _constraints->SupportsFusedRescale(OpType::Sub, TensorUsage::OFM, DataType::Int32, DataType::Int32,
                         DataType::Int32, DataType::Int32, ofmConn->quantization) )
                {
                    // Replace ReduceSum (zp != 0) with ReduceSum->Sub(zp):

                    // Temporary tensor between ReduceSum and Sub
                    std::shared_ptr<Tensor> reduceSumTens = ofmConn->tensor->Clone();
                    reduceSumTens->SetName(ofmConn->tensor->Name() + "_reducesum");
                    reduceSumTens->ChangeType(DataType::Int32);
                    reduceSumTens->SetStorageShape(ofmConn->shape);

                    // Sub op with zero point
                    auto zpTens = CreateConstTensor("zero_point", DataType::Int32, int(ifmConn->shape.Depth() * zp));
                    auto subOp = std::make_shared<Operation>(OpType::Sub);
                    subOp->ConnectInput(TensorUsage::IFM, reduceSumTens);
                    subOp->ConnectInput(TensorUsage::IFM1, zpTens);
                    subOp->CopyOutput(TensorUsage::OFM, *ofmConn);
                    subOp->Output(TensorUsage::OFM)->Set(ofmConn->rounding);
                    RecordOptimisation(operation, subOp.get());
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
                    weightsTens->SetAxisOrder(AxisOrder::OHWI);
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
                    RecordOptimisation(operation, convOp.get());
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
    auto view = params->tensor->View();
    assert(params->tensor->Type() == DataType::Int32);

    Shape multiples(view.Buffer()->Data<int32_t>(), view.ViewShape().Elements());

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

            RecordOptimisation(operation, tileOp.get());
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
                RecordOptimisation(operation, returnOp);
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
            query.transposeMask = mergedTranspose;
            if ( _constraints->OperatorQuery(OpType::Transpose, &query, nullptr).Any(QueryResult::Native) )
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
        int height = s.Elements() / (s.Width() * s.Depth());
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
    RecordOptimisation(operation, transposeOp.get());

    // replace IFM2 with transposed output
    operation->ConnectInput(TensorUsage::IFM1, transposedIfm1).Set(ifm1Conn->quantization);

    return returnOp;
}

// Convert depthwise convolutions with a depth multiplier greater than 1 into a single Conv2D if:
// - the input depth is 1; and
// - the output depth equals the depth multiplier.
Operation *GraphIrOptimiser::RewriteDepthwise(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    if ( operation->Type() == OpType::DepthwiseConv2D )
    {
        const auto ifm = operation->Input(TensorUsage::IFM0);
        const auto ofm = operation->Output(TensorUsage::OFM);
        const auto multiplier = operation->Kernel()->DepthMultiplier();

        if ( ifm && (ifm->shape.Depth() == 1) && (multiplier != 1) && ofm && (ofm->shape.Depth() == multiplier) )
        {
            auto newOp = std::make_shared<Operation>(OpType::Conv2D);
            auto kernel = std::make_unique<Kernel>(operation->Kernel()->WithDepthMultiplier(1));
            newOp->SetKernel(std::move(kernel));
            const auto weights = operation->Input(TensorUsage::Weights);
            if ( weights->tensor->AxisOrder() == AxisOrder::HWCM )
            {
                const auto &shape = weights->tensor->StorageShape();
                weights->tensor->Reshape(Shape(1, shape[0], shape[1], shape[3]));
                weights->tensor->SetAxisOrder(AxisOrder::IHWO);
                weights->shape = weights->tensor->StorageShape();
            }

            ReplaceOperation(operation, newOp.get());
            newOp->Output(TensorUsage::OFM)->Set(ofm->rounding);
            returnOp = newOp.get();
            RecordOptimisation(operation, returnOp);
        }
    }

    return returnOp;
}

// Handle OFM-padding for transpose convolutions by reshaping the OFM/IFM areas.
// if ofm_pad > 0
//     * Adjust write-offset and write-shape of the OFM
//     * Insert depthwise convolution operations to copy bias to the padded OFM areas
// if ofm_pad < 0
//     * Adjust the read-shape of the IFM by reducing the ifm-padding
Operation *GraphIrOptimiser::RewriteTransposeConvOFMPadding(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    if ( !operation->HasAttribute<transpose_conv2d_attr_t>() )
    {
        return returnOp;
    }
    auto attr = operation->Attribute<transpose_conv2d_attr_t>();
    assert(attr);
    assert(attr->outPadTBLR.IsValid());
    assert(attr->outShape.IsValid());
    if ( attr->outPadTBLR.IsEmpty() )
    {
        // no out-padding
        return returnOp;
    }
    auto *ofmConn = operation->Output(TensorUsage::OFM);
    auto *ifmConn = operation->Input(TensorUsage::IFM);
    auto *biasConn = operation->Input(TensorUsage::Scales);
    auto ofmShape = ofmConn->shape;
    DataType biasType = biasConn->tensor->Type();

    ofmConn->slice.Initialize(ofmShape.WithZeros(), ofmShape);

    auto &ofmSlice = ofmConn->slice;

    const auto &padding = operation->Kernel()->Padding();
    int ifmPadTop = padding.Top();
    int ifmPadBottom = padding.Bottom();
    int ifmPadLeft = padding.Left();
    int ifmPadRight = padding.Right();

    int ofmPadTop = attr->outPadTBLR[0];
    int ofmPadBottom = attr->outPadTBLR[1];
    int ofmPadLeft = attr->outPadTBLR[2];
    int ofmPadRight = attr->outPadTBLR[3];

    // Create DW-convolutions to copy bias values into the padded OFM-region
    auto CreateDWForOFMPadding = [&](const TensorSlice &padSlice, const std::string &name)
    {
        std::shared_ptr<Tensor> inputZero;
        auto dwOp = std::make_shared<Operation>(OpType::DepthwiseConv2D);
        int sliceElements = padSlice.shape.Elements();

        // Create zero-input tensor that has same shape as the padded OFM-area
        std::string inputName = fmt::format("{}_inputZero", name);
        RoundMode rounding;
        if ( biasType == DataType::Int48 )
        {
            auto zeroBuf = std::make_shared<Buffer>(std::vector<int16_t>(sliceElements, 0));
            inputZero = std::make_shared<Tensor>(inputName, DataType::Int16, padSlice.shape, zeroBuf);
            rounding = RoundMode::NATURAL;
        }
        else
        {
            auto zeroBuf = std::make_shared<Buffer>(std::vector<int8_t>(sliceElements, 0));
            inputZero = std::make_shared<Tensor>(inputName, DataType::Int8, padSlice.shape, zeroBuf);
            rounding = RoundMode::DBL;
        }

        // Create weights-tensor with 1x1 kernel
        Shape weightShape(1, 1, 1, ofmShape.Depth());
        std::vector<int8_t> ones(weightShape.Elements(), 1);
        auto weightBuf = std::make_shared<Buffer>(std::move(ones));
        auto weightTensor = std::make_shared<Tensor>(fmt::format("{}_unitWeights", name), DataType::UInt8, weightShape, weightBuf);
        weightTensor->SetAxisOrder(AxisOrder::IHWO);

        dwOp->SetKernel(std::make_unique<Kernel>(Point2i(1, 1), Point2i(1, 1), Point2i(1, 1)));
        dwOp->ConnectInput(TensorUsage::IFM, inputZero).Set(Quantization::Unit());
        dwOp->ConnectInput(TensorUsage::Weights, weightTensor).Set(Quantization::Unit());
        dwOp->CopyInput(TensorUsage::Scales, *biasConn);
        dwOp->CopyOutput(TensorUsage::OFM, *ofmConn);
        dwOp->Output(TensorUsage::OFM)->Set(padSlice).Set(rounding);

        RecordOptimisation(operation, dwOp.get());
    };

    // Positive output-padding is handled by adjusting the write slice of the OFM
    // Negative output-padding is handled by reducing IFM-padding
    if ( ofmPadTop > 0 )
    {
        // Adjust OFM-offset and shape to account for positive top-padding
        ofmSlice.offset = ofmSlice.offset.WithHeight(ofmSlice.offset.Height() + ofmPadTop);
        ofmSlice.shape = ofmSlice.shape.WithHeight(ofmSlice.shape.Height() - ofmPadTop);

        // pad OFM-area from origo to the slice-offset height
        Shape padOffset = ofmShape.WithZeros();
        Shape padShape = ofmShape.WithHeight(ofmSlice.offset.Height());
        CreateDWForOFMPadding({padOffset, padShape}, fmt::format("{}_ofmPadTop", ofmConn->tensor->Name()));
    }
    else
    {
        // negative OFM-padding reduces ifm-padding
        ifmPadTop += ofmPadTop;
        assert(ifmPadTop >= 0 && "unexpected negative OFM-padding (top)");
    }
    if ( ofmPadBottom > 0 )
    {
        // Reduce OFM-shape to account for positive bottom-padding
        ofmSlice.shape = ofmSlice.shape.WithHeight(ofmSlice.shape.Height() - ofmPadBottom);

        // pad OFM-area from the end of the write-region until the ofm-height
        int writeEndHeight = ofmSlice.offset.Height() + ofmSlice.shape.Height();
        Shape padOffset = ofmShape.WithZeros().WithHeight(writeEndHeight);
        Shape padShape = ofmShape.WithHeight(ofmShape.Height() - writeEndHeight);
        CreateDWForOFMPadding({padOffset, padShape}, fmt::format("{}_ofmPadBottom", ofmConn->tensor->Name()));
    }
    else
    {
        // negative OFM-padding reduces ifm-padding
        ifmPadBottom += ofmPadBottom;
        assert(ifmPadBottom >= 0 && "unexpected negative OFM-padding (bottom)");
    }
    if ( ofmPadLeft > 0 )
    {
        // Adjust OFM-offset and shape to account for positive left-padding
        ofmSlice.offset = ofmSlice.offset.WithWidth(ofmSlice.offset.Width() + ofmPadLeft);
        ofmSlice.shape = ofmSlice.shape.WithWidth(ofmSlice.shape.Width() - ofmPadLeft);

        // pad OFM-area from origo to slice-offset width
        Shape padOffset = ofmShape.WithZeros();
        Shape padShape = ofmShape.WithWidth(ofmSlice.offset.Width());

        // Adjust shape and offset for already padded top/bottom regions
        // i.e. only pad along the height of the write-shape
        padOffset = padOffset.WithHeight(ofmSlice.offset.Height());
        padShape = padShape.WithHeight(ofmSlice.shape.Height());

        CreateDWForOFMPadding({padOffset, padShape}, fmt::format("{}_ofmPadLeft", ofmConn->tensor->Name()));
    }
    else
    {
        // negative OFM-padding reduces ifm-padding
        ifmPadLeft += ofmPadLeft;
        assert(ifmPadLeft >= 0 && "unexpected negative OFM-padding (left)");
    }
    if ( ofmPadRight > 0 )
    {
        // Reduce OFM-shape to account for positive right-padding
        ofmSlice.shape = ofmSlice.shape.WithWidth(ofmSlice.shape.Width() - ofmPadRight);

        // pad OFM-area from the end of the write-region until the ofm-width
        int writeEndWidth = ofmSlice.offset.Width() + ofmSlice.shape.Width();
        Shape padOffset = ofmShape.WithZeros().WithWidth(writeEndWidth);
        Shape padShape = ofmShape.WithWidth(ofmShape.Width() - writeEndWidth);

        // Adjust shape and offset for already padded top/bottom regions
        // i.e. only pad along the height of the write-shape
        padOffset = padOffset.WithHeight(ofmSlice.offset.Height());
        padShape = padShape.WithHeight(ofmSlice.shape.Height());
        CreateDWForOFMPadding({padOffset, padShape}, fmt::format("{}_ofmPadRight", ofmConn->tensor->Name()));
    }
    else
    {
        // negative OFM-padding reduces ifm-padding
        ifmPadRight += ofmPadRight;
        assert(ifmPadRight >= 0 && "unexpected negative OFM-padding (right)");
    }

    auto newKernel = std::make_unique<Kernel>(operation->Kernel()->WithPadding({ifmPadTop, ifmPadLeft, ifmPadBottom, ifmPadRight}));
    operation->SetKernel(std::move(newKernel));
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

// Reshape ArgMax input/outputs to 3D-tensors where W is the reduced axis
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
    int ifmRank = ifmConn->shape.Size();
    int axis = attr->axis;

    // Extend OfmShape to match ifmRank
    if ( ofmShape.Size() != ifmShape.Size() )
    {
        ofmShape = ofmShape.Insert(attr->axis, 1);
        assert(ofmShape.Size() == ifmShape.Size());
    }

    // Reshape IFM and OFM to 3D-tensors where W is the reduced axis
    if ( attr->axis != 1 || ifmRank != 3 )
    {
        ifmShape = ReshapeTo3DAroundAxis(ifmShape, axis);
        ofmShape = ifmShape.WithWidth(1);
        attr->axis = 1;
    }
    operation->Output(TensorUsage::OFM)->Set(RoundMode::TRUNCATE_TO_LOWER);
    // Update kernel based on reshapes
    std::unique_ptr<Kernel> kernel = std::make_unique<Kernel>(Point2i(ifmShape[1], 1), Point2i(1, 1), Point2i(1, 1));
    operation->SetKernel(std::move(kernel));
    return returnOp;
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

            // Don't move to CPU, Reshape or Tile operations
            // low-level implementation of TILE requires unsliced inputs
            if ( cons->Type() == OpType::Passthrough || IsReshape(cons->Type()) || cons->Type() == OpType::Tile )
            {
                return operation;
            }

            if ( consIfm0 == ofm )
            {
                // Check if ifm0 consumer has correct shape
                auto *consIfm0Conn = cons->Input(TensorUsage::IFM0);
                ifmShapeEqual = Shape::IsReducedEqual(consIfm0Conn->shape, ofmConn->shape);
            }
            else if ( consIfm1 != nullptr && consIfm1 == ofm )
            {
                // Check if ifm1 consumer has correct shape
                auto *consIfm1Conn = cons->Input(TensorUsage::IFM1);
                ifmShapeEqual = Shape::IsReducedEqual(consIfm1Conn->shape, ofmConn->shape);
            }

            // Calculate the consumer transpose type
            TransposeType consumerTranspose = TransposeType::None;
            if ( cons->Type() == OpType::Transpose )
            {
                consumerTranspose = TransposeTypeFromShape(cons->Attribute<transpose_attr_t>()->perm);
            }

            // We can only move to consumer if there is no transpose on the op that we move to,
            // otherwise the IFM shape may change and transposition will be wrong.
            if ( Shape::IsReducedEqual(ofmConn->shape, ofm->StorageShape()) && IsNone(consumerTranspose) && ifmShapeEqual )
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
