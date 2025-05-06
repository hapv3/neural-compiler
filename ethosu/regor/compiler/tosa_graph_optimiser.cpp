//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "compiler/tosa_graph_optimiser.hpp"

#include "architecture/architecture_constraints.hpp"
#include "optimiser_utils.hpp"


namespace regor
{

using namespace GraphOptimisation;

// Convert compile time constant zero point tensors to quantization zero points
Operation *TosaGraphOptimiser::ConvertZeroPointTensors(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    auto SetZeroPoint = [&](TensorUsage target, TensorUsage param, bool asUnsigned = false)
    {
        if ( const auto zpConn = operation->Input(param) )
        {
            assert(zpConn->tensor->IsConstant());
            const auto targetConn = IsOFM(target) ? operation->Output(target) : operation->Input(target);
            assert(targetConn);
            auto dataType = asUnsigned ? zpConn->tensor->Type() & ~unsigned(DataType::Signed) : zpConn->tensor->Type();
            auto values = zpConn->tensor->View().Values<int64_t>(dataType);
            targetConn->quantization.zeroPoints = {values.begin(), values.end()};
        }
    };
    switch ( operation->Type() )
    {
        case OpType::AvgPool:
        case OpType::Neg:
            SetZeroPoint(TensorUsage::IFM, TensorUsage::Params0);
            SetZeroPoint(TensorUsage::OFM, TensorUsage::Params1);
            break;
        case OpType::Conv2D:
        case OpType::Conv3D:
        case OpType::DepthwiseConv2D:
        case OpType::TransposeConv2D:
            SetZeroPoint(TensorUsage::IFM, TensorUsage::Params0);
            SetZeroPoint(TensorUsage::Weights, TensorUsage::Params1);
            break;
        case OpType::MatMul:
            SetZeroPoint(TensorUsage::IFM0, TensorUsage::Params0);
            SetZeroPoint(TensorUsage::IFM1, TensorUsage::Params1);
            break;
        case OpType::Rescale:
        {
            const auto signAttr = operation->Attribute<sign_attr_t>();
            SetZeroPoint(TensorUsage::IFM, TensorUsage::Params2, signAttr->input_unsigned);
            SetZeroPoint(TensorUsage::OFM, TensorUsage::Params3, signAttr->output_unsigned);
            break;
        }
        default:
            break;
    }
    return operation;
}

Operation *TosaGraphOptimiser::RewriteTransposeConvPadding(Graph *const graph, Operation *const operation)
{
    UNUSED(graph);
    Operation *returnOp = operation;
    if ( operation->Type() != OpType::TransposeConv2D || _constraints->SupportsAccumulatorSaveRestore() )
    {
        return returnOp;
    }

    // TODO: MLBEDSW-11089 - Refactor TransposeConv padding calculation for IFM resampling path
    auto *ofmConn = operation->Output(TensorUsage::OFM);
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

    auto *attr = operation->Attribute<transpose_conv2d_attr_t>();
    const int ofmPadTop = attr->outPadTBLR[0];
    const int ofmPadBottom = attr->outPadTBLR[1];
    const int ofmPadLeft = attr->outPadTBLR[2];
    const int ofmPadRight = attr->outPadTBLR[3];

    // Create DW-convolutions to copy bias values into the padded OFM-region
    auto CreateDWForOFMPadding = [&](const TensorSlice &padSlice, const std::string &name)
    {
        std::shared_ptr<Tensor> inputZero;
        auto dwOp = std::make_shared<Operation>(OpType::DepthwiseConv2D);
        int sliceElements = padSlice.shape.Elements();

        // Create zero-input tensor that has same shape as the padded OFM-area
        std::string inputName = fmt::format("{}_inputZero", name);
        RoundMode rounding;
        if ( biasType == DataType::Int48 || biasType == DataType::Int64 )
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

        RecordOptimisation(*operation, dwOp.get());
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


TosaGraphOptimiser::TosaGraphOptimiser(IArchitectureConstraints *constraints, const GraphOptimiserOptions &options, OptimiserDatabase *db) :
        GraphOptimiser(constraints, options, db)
{
}

void TosaGraphOptimiser::OptimiseGraph(Graph *graph)
{
    for ( auto iOpt = GraphOptimisationSteps().begin(); iOpt != GraphOptimisationSteps().end(); ++iOpt )
    {
        LOG_TRACE1("GraphOptimiser {0}/{1}\n", std::distance(GraphOptimisationSteps().begin(), iOpt) + 1,
            GraphOptimisationSteps().size());
        // Check if function lists are empty. Do not call for step that only contain disabled debug functions.
        if ( !iOpt->opFunction.empty() || !iOpt->tensorFunction.empty() )
        {
            RewriteGraph<TosaGraphOptimiser>(graph, *iOpt);
        }
    }
}

}  // namespace regor
