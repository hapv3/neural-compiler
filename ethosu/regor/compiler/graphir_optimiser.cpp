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

#include "compiler/graphir_optimiser.hpp"

#include "operation_util.hpp"
#include "optimiser_utils.hpp"

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
        if ( tensor->IsConstant() )
        {
            const auto oldView = tensor->View();
            const auto oldValues = oldView.Values<int8_t>();
            const auto size = oldView.BufferSize();

            // Replace this tensor's buffer with a new buffer since we don't know if the current buffer is writable
            auto newBuffer = std::make_shared<Buffer>(std::make_unique<uint8_t[]>(size), size);
            tensor->SetBuffer(newBuffer);
            auto view = tensor->View();
            auto &shape = view.ViewShape();
            auto values = view.WritableValues<int8_t>();
            for ( int i = 0; i < shape.Elements(); i++ )
            {
                // Convert each element to the internal representation -1 (true) and 0 (false)
                values[i] = oldValues[i] == 0 ? 0 : -1;
            }
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

Operation *GraphIrOptimiser::ConvertAttributes(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    OpType opType = operation->Type();
    if ( opType == OpType::Asr )
    {
        const auto *attr = operation->Attribute<asr_attr_t>();
        auto roundMode = attr->round ? RoundMode::NATURAL : RoundMode::TRUNCATE_TO_LOWER;
        operation->SetRounding(roundMode);
    }
    else if ( opType == OpType::Rescale )
    {
        const auto *attr = operation->Attribute<rescale_attr_t>();
        auto roundMode = attr->double_round ? RoundMode::DBL : RoundMode::NATURAL;
        operation->SetRounding(roundMode);
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
        const auto *attr = operation->Attribute<mul_attr_t>();
        TensorConnection *ofmConn = operation->Output(TensorUsage::OFM);
        // A non-zero shift attribute is only supported with explicit quantization
        assert(attr->shift == 0 || ofmConn->quantization.type == QuantizationType::EXPLICIT);
        if ( !ofmConn->quantization.scales.size() )
        {
            ofmConn->quantization.scales.push_back({1, 0});
        }
        ofmConn->quantization.scales[0].shift += attr->shift;
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

        // Check if ifm/ofm are network ifm/ofm
        bool isIfmSgIfm = IsTensorInVector(graph->Inputs(), ifm);
        bool isOfmSgOfm = IsTensorInVector(graph->Outputs(), ofm);
        bool isIfmSgOfm = IsTensorInVector(graph->Outputs(), ifm);

        // TODO: MLBEDSW-9069: Check CPU operator producer/consumer
        assert(ifm->Readers().size() == 1 || (ifm->StorageShape() == ofm->StorageShape() && ifm->AxisOrder() == ofm->AxisOrder()));
        // Inserts a copy op if needed before removing reshapes.
        if ( ((isIfmSgIfm || isIfmSgOfm) && (isOfmSgOfm)) ||
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
    if ( opType == OpType::FullyConnected )
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
        auto ifm = operation->Input(TensorUsage::IFM0);
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
            if ( h > 4 || w > 4 )
            {
                // Ended up with shape that requires the weights to be reread.
                // Convert op to conv2d since this enables weight buffering.
                auto newOp = std::make_shared<Operation>(OpType::Conv2DBias);
                newOp->SetRounding(ifm->tensor->Type() == DataType::Int16 ? RoundMode::NATURAL : RoundMode::DBL);
                ReplaceOperation(operation, newOp.get());
                returnOp = newOp.get();
                RecordOptimisation(operation, returnOp);
            }
        }
    }
    return returnOp;
}

Operation *GraphIrOptimiser::RewriteRescale(Graph *const, Operation *const operation)
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


/// @brief Moves Rescale operations to the output of the previous operation
///        or the input of the next operation when possible.
/// @param
/// @param operation Operation to optimise
/// @return (Possibly) optimised operation
Operation *GraphIrOptimiser::FuseRescale(Graph *const, Operation *const operation)
{
    Operation *returnOp = operation;
    OpType opType = operation->Type();
    if ( opType == OpType::Rescale )
    {
        auto ofmConn = operation->Output(TensorUsage::OFM);
        auto ifmConn = operation->Input(TensorUsage::IFM);
        auto producer = ifmConn->tensor->Writers().size() == 1 ? ifmConn->tensor->Writers().front() : nullptr;
        if ( producer && producer->Output(TensorUsage::OFM)->quantization.EqualScales(Quantization::Unit()) &&
             _constraints->SupportsFusedRescale(producer->Type(), TensorUsage::OFM, producer->IFM(0)->Type(),
                 ofmConn->tensor->Type(), ofmConn->quantization) )
        {
            // Propagate rescaling to output of previous op
            producer->CopyOutput(TensorUsage::OFM, *ofmConn);
            returnOp = producer.get();
        }
        else if ( ofmConn->tensor->Readers().size() == 1 && ofmConn->quantization.zeroPoints == Quantization::Unit().zeroPoints )
        {
            // Propagate rescaling to input of next op
            auto consumer = ofmConn->tensor->Readers().front();
            auto ifmQuant = ofmConn->quantization;
            // Convert scales to have 0 shift if possible, since this can
            // improve fusing for Ethos-U55/65
            for ( auto &qs : ifmQuant.scales )
            {
                if ( qs.shift > 0 && qs.shift < 31 && (qs.scale % (1 << qs.shift)) == 0 )
                {
                    qs = {(qs.scale >> qs.shift), 0};
                }
            }
            for ( auto ifm : consumer->Inputs().pairs() )
            {
                if ( ifm.second.tensor == ofmConn->tensor )
                {
                    if ( ifm.second.quantization.EqualScales(Quantization::Unit()) &&
                         _constraints->SupportsFusedRescale(consumer->Type(), TensorUsage::IFM, ifmConn->tensor->Type(),
                             ofmConn->tensor->Type(), ifmQuant) )
                    {
                        consumer->CopyInput(ifm.first, *ifmConn);
                        ifm.second.quantization.scales = ifmQuant.scales;
                        returnOp = consumer.get();
                        break;
                    }
                }
            }
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
        if ( kernel->Size() == kernel->Stride() && kernel->Stride() == ifm->shape.WH<int>() && kernel->Padding().IsZero() )
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
        returnOp->SetRounding(RoundMode::NATURAL);
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
            copyOp->ConnectInput(TensorUsage::IFM1, CreateConstTensor("const_zero", ifmConn->tensor->Type(), 0));
            RecordOptimisation(operation, copyOp.get());
            ReplaceOperation(operation, copyOp.get());
            returnOp = copyOp.get();
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
            copyOp->SetRounding(RoundMode::NATURAL);
            copyOp->CopyInput(TensorUsage::IFM, ifmConn);
            copyOp->CopyOutput(TensorUsage::OFM, *ofmConn);
            copyOp->Output(TensorUsage::OFM)->Set({ofmSliceOffset, ifmConn.shape});
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
        auto copyOp = std::make_shared<Operation>(OpType::Add);
        copyOp->SetRounding(RoundMode::NATURAL);
        copyOp->CopyInput(TensorUsage::IFM0, *ifmConn);
        copyOp->Input(TensorUsage::IFM0)->Set({begin, size});
        copyOp->ConnectInput(TensorUsage::IFM1, CreateConstTensor("const_zero", ifmConn->tensor->Type(), 0));
        copyOp->CopyOutput(TensorUsage::OFM, *ofmConn);
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
        newOp->SetRounding(RoundMode::NATURAL);
        newOp->ConnectInput(TensorUsage::IFM0, CreateConstTensor("const_zero", ifmConn->tensor->Type(), 0));
        newOp->CopyInput(TensorUsage::IFM1, *ifmConn);
        newOp->CopyOutput(TensorUsage::OFM, *ofmConn);
        RecordOptimisation(operation, newOp.get());
        returnOp = newOp.get();
        operation->Disconnect();
    }
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
        cons->CopyInput(TensorUsage::IFM0, *ifmConn);
    }
    else if ( consIfm1 != nullptr && IsBinaryElementwise(cons->Type()) && consIfm1 == ofm )
    {
        cons->CopyInput(TensorUsage::IFM1, *ifmConn);
    }
}

Operation *GraphIrOptimiser::MoveSplitSliceToConsumer(Graph *const, Operation *const operation)
{
    auto *ifmConn = operation->Input(TensorUsage::IFM0);

    if ( operation->Type() == OpType::MemoryCopy && ifmConn->slice.offset.Size() > 0 )
    {
        auto *ofmConn = operation->Output(TensorUsage::OFM);
        auto *ofm = ofmConn->tensor.get();

        // TODO: MLBEDSW-9072: Add check that moving split to consumer is valid

        // We can only move to consumer if there is no transpose on the op that we will remove,
        // otherwise we will lose that transposition.
        if ( ofm->Readers().size() == 1 && IsNone(ofmConn->transpose) )
        {
            auto cons = ofm->Readers().front();
            auto consOfmConn = cons->Output(TensorUsage::OFM);
            auto *consIfm0 = cons->IFM(0);
            auto *consIfm1 = cons->IFM(1);

            bool ifmShapeEqual = false;
            if ( consIfm0 == ofm )
            {
                // Check if ifm0 consumer has correct shape
                auto *consIfm0Conn = cons->Input(TensorUsage::IFM0);
                ifmShapeEqual = consIfm0Conn->shape == ofmConn->shape;
            }
            else if ( consIfm1 != nullptr && consIfm1 == ofm )
            {
                // Check if ifm1 consumer has correct shape
                auto *consIfm1Conn = cons->Input(TensorUsage::IFM1);
                ifmShapeEqual = consIfm1Conn->shape == ofmConn->shape;
            }

            // We can only move to consumer if there is no transpose on the op that we move to,
            // otherwise the IFM shape may change and transposition will be wrong.
            if ( !IsReshape(cons->Type()) && ofmConn->shape == Shape::PadAxes(ofm->StorageShape(), 4, 1) &&
                 IsNone(consOfmConn->transpose) && ifmShapeEqual )
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
