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

Operation *GraphIrOptimiser::ConvertAttributes(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    OpType opType = operation->Type();
    if ( opType == OpType::Asr )
    {
        auto roundMode = operation->attr.asr.round ? RoundMode::NATURAL : RoundMode::TRUNCATE_TO_LOWER;
        operation->SetRounding(roundMode);
    }
    if ( opType == OpType::Rescale )
    {
        auto roundMode = operation->attr.rescale.double_round ? RoundMode::DBL : RoundMode::NATURAL;
        operation->SetRounding(roundMode);
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
        auto &attr = operation->attr;
        TensorConnection *ifmConn = operation->Input(TensorUsage::IFM);
        Shape ifmStart = ifmConn->shape.WithZeros();
        Shape ifmShape = ifmConn->shape;
        int offset_h = attr.resize.offsetYX[0];
        int offset_w = attr.resize.offsetYX[1];
        int scale_nh = attr.resize.scaleY.n;
        int scale_nw = attr.resize.scaleX.n;
        if ( offset_h >= scale_nh )
        {
            ifmStart[1] += offset_h / scale_nh;
            ifmShape[1] -= ifmStart[1];
            attr.resize.offsetYX[0] = offset_h % scale_nh;
        }
        if ( offset_w >= scale_nw )
        {
            ifmStart[2] += offset_w / scale_nw;
            ifmShape[2] -= ifmStart[2];
            attr.resize.offsetYX[1] = offset_w % scale_nw;
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

        // Inserts a copy op if needed before removing reshapes.
        if ( (isIfmSgIfm || isIfmSgOfm) && (isOfmSgOfm) )
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
        int channels = operation->attr.rescale.per_channel ? ofmConn->shape.Depth() : 1;
        for ( int i = 0; i < channels; i++ )
        {
            QuantizedScale qScale;
            int32_t scale = mulT == DataType::Int32 ? mulView.Values<int32_t>()[i] : mulView.Values<int16_t>()[i];
            int32_t shift = shiftView.Values<int8_t>()[i];
            assert(operation->attr.rescale.scale32 || static_cast<int16_t>(scale) == scale);
            assert(static_cast<int8_t>(shift) == shift);

            qScale.scale = operation->attr.rescale.scale32 ? scale : static_cast<int16_t>(scale);
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
