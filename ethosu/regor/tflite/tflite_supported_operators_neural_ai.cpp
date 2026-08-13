//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "tflite_supported_operators_neural_ai.hpp"

#include "compiler/operation_util.hpp"

#include <limits>
#include <set>

namespace regor
{
namespace
{

const std::set<OpType> s_supportedOpTypes = {
    OpType::Add,
    OpType::AvgPool,
    OpType::Conv2D,
    OpType::Concat,
    OpType::DepthwiseConv2D,
    OpType::Dfl,
    OpType::ExpandDims,
    OpType::FullyConnected,
    OpType::LUT,
    OpType::MaxPool,
    OpType::Reshape,
    OpType::ResizeNearestNeighbor,
    OpType::Sigmoid,
    OpType::StridedSlice,
    OpType::Squeeze,
    OpType::Transpose,
};

const std::set<DataType> s_supportedDataTypes = {
    DataType::Int8,
    DataType::Int32,
};

bool ViewPreservesDepth(const Operation *operation)
{
    const TensorConnection *ifm = operation->Input(TensorUsage::IFM0);
    const TensorConnection *ofm = operation->Output(TensorUsage::OFM);
    if ( ifm == nullptr || ofm == nullptr ||
         ifm->shape.Elements64() != ofm->shape.Elements64() ||
         ifm->shape.Depth() != ofm->shape.Depth() )
    {
        Failure(operation,
            "Neural-AI zero-copy views must preserve the innermost channel dimension");
        return false;
    }
    return true;
}

bool PreservesQuantization(const Operation *operation)
{
    const TensorConnection *ifm = operation->Input(TensorUsage::IFM0);
    const TensorConnection *ofm = operation->Output(TensorUsage::OFM);
    if ( ifm == nullptr || ofm == nullptr || ifm->quantization != ofm->quantization )
    {
        Failure(operation,
            "Neural-AI raw-value operation requires identical IFM and OFM quantization");
        return false;
    }
    return true;
}

bool SupportedConcat(const Operation *operation)
{
    const TensorConnection *lhs = operation->Input(TensorUsage::IFM0);
    const TensorConnection *rhs = operation->Input(TensorUsage::IFM1);
    const TensorConnection *tail = operation->Input(TensorUsage::IFM2);
    const TensorConnection *ofm = operation->Output(TensorUsage::OFM);
    const auto *attr = operation->Attribute<axis_attr_t>();
    int ifmCount = 0;
    for ( const auto &[usage, connection] : operation->Inputs().pairs() )
    {
        UNUSED(connection);
        if ( IsIFM(usage) ) ++ifmCount;
    }
    int axis = attr == nullptr ? 0 : attr->axis;
    if ( lhs != nullptr && axis < 0 ) axis += lhs->shape.Size();
    const bool structuralCspConcat = lhs != nullptr && rhs != nullptr && tail != nullptr && ofm != nullptr &&
        ifmCount == 3 && lhs->shape.Size() == 4 && lhs->shape.Batch() == 1 &&
        lhs->shape.Height() > 0 && lhs->shape.Width() > 0 && lhs->shape.Depth() == 16 &&
        rhs->shape == lhs->shape && tail->shape == lhs->shape &&
        ofm->shape == lhs->shape.WithDepth(48) &&
        lhs->quantization.EqualScales(rhs->quantization) && lhs->quantization.EqualScales(tail->quantization) &&
        lhs->quantization.EqualScales(ofm->quantization) &&
        lhs->quantization.zeroPoints == rhs->quantization.zeroPoints &&
        lhs->quantization.zeroPoints == tail->quantization.zeroPoints &&
        lhs->quantization.zeroPoints == ofm->quantization.zeroPoints;
    if ( lhs == nullptr || rhs == nullptr || ofm == nullptr || (!structuralCspConcat && ifmCount != 2) ||
         axis != lhs->shape.Size() - 1 || lhs->shape.Size() != rhs->shape.Size() ||
         lhs->shape.Size() != ofm->shape.Size() ||
         lhs->shape.WithDepth(1) != rhs->shape.WithDepth(1) ||
         lhs->shape.WithDepth(1) != ofm->shape.WithDepth(1) ||
         (lhs->shape.Depth() % 32 != 0 && !structuralCspConcat) ||
         (rhs->shape.Depth() % 32 != 0 &&
             !structuralCspConcat && !(lhs->shape.Depth() == 64 && rhs->shape.Depth() == 80 &&
               (lhs->shape.Height() == 10 || lhs->shape.Height() == 20 ||
                   lhs->shape.Height() == 40) &&
               lhs->shape.Width() == lhs->shape.Height())) ||
         (!structuralCspConcat && ofm->shape.Depth() != lhs->shape.Depth() + rhs->shape.Depth()) ||
         (!structuralCspConcat &&
             (lhs->quantization != rhs->quantization || lhs->quantization != ofm->quantization)) )
    {
        Failure(operation,
            "Neural-AI Concat requires aligned two-input, selected C64+C80, or structural C16+C16+C16 inputs on the channel axis");
        return false;
    }
    return true;
}

bool SupportedHeadTranspose(const Operation *operation)
{
    const TensorConnection *ifm = operation->Input(TensorUsage::IFM0);
    const TensorConnection *permutation = operation->Input(TensorUsage::Params);
    const TensorConnection *ofm = operation->Output(TensorUsage::OFM);
    if ( ifm == nullptr || permutation == nullptr || ofm == nullptr ||
         !permutation->tensor->IsConstant() || permutation->shape.Elements64() != 4 ||
         ifm->shape.Size() != 4 || ofm->shape.Size() != 4 || ifm->shape.Batch() != 1 ||
         ifm->shape.Depth() != 144 || ifm->shape.Width() != ifm->shape.Height() ||
         (ifm->shape.Height() != 10 && ifm->shape.Height() != 20 && ifm->shape.Height() != 40) ||
         ofm->shape != Shape(1, 144, ifm->shape.Height(), ifm->shape.Width()) ||
         ifm->quantization != ofm->quantization )
    {
        Failure(operation,
            "Neural-AI Transpose requires a selected quantization-preserving C144 detection head");
        return false;
    }
    const auto values = permutation->tensor->View().Values<int32_t>();
    if ( values.Count() != 4 || values[0] != 0 || values[1] != 3 ||
         values[2] != 1 || values[3] != 2 )
    {
        Failure(operation, "Neural-AI head Transpose requires permutation [0, 3, 1, 2]");
        return false;
    }
    return true;
}

bool SupportedDfl16(const Operation *operation)
{
    const TensorConnection *ifm = operation->Input(TensorUsage::IFM0);
    const TensorConnection *ofm = operation->Output(TensorUsage::OFM);
    if ( ifm == nullptr || ofm == nullptr || ifm->tensor->Type() != DataType::Int8 ||
         ofm->tensor->Type() != DataType::Int8 || ifm->shape != Shape(1, 144, 2100) || ofm->shape != Shape(1, 1, 4, 2100) )
    {
        Failure(operation, "Neural-AI DFL requires the selected 4x16x2100 INT8 contract");
        return false;
    }
    return true;
}

bool SupportedC32DepthSlice(const Operation *operation)
{
    const TensorConnection *ifm = operation->Input(TensorUsage::IFM0);
    const TensorConnection *ofm = operation->Output(TensorUsage::OFM);
    const TensorConnection *beginParam = operation->Input(TensorUsage::Params0);
    const TensorConnection *endParam = operation->Input(TensorUsage::Params1);
    const TensorConnection *stridesParam = operation->Input(TensorUsage::Params2);
    const auto *passthrough = static_cast<const tflite::Operator *>(operation->Passthrough());
    const auto *options = passthrough != nullptr ?
        passthrough->builtin_options_as_StridedSliceOptions() : nullptr;
    if ( ifm == nullptr || ofm == nullptr || beginParam == nullptr || endParam == nullptr ||
         stridesParam == nullptr || options == nullptr || ifm->shape.Size() != 4 ||
         ofm->shape.Size() != 4 || ifm->shape.WithDepth(1) != ofm->shape.WithDepth(1) ||
         ifm->quantization != ofm->quantization || options->ellipsis_mask() != 0 ||
         options->new_axis_mask() != 0 || options->shrink_axis_mask() != 0 ||
         (options->begin_mask() & 0x7) != 0x7 || (options->end_mask() & 0x7) != 0x7 )
    {
        Failure(operation,
            "Neural-AI StridedSlice requires a full-spatial, quantization-preserving rank-4 depth slice");
        return false;
    }

    const Shape begin = TensorToShape(beginParam->tensor.get(), beginParam->shape.Elements());
    const Shape end = TensorToShape(endParam->tensor.get(), endParam->shape.Elements());
    const Shape strides = TensorToShape(stridesParam->tensor.get(), stridesParam->shape.Elements());
    if ( begin.Size() != 4 || end.Size() != 4 || strides.Size() != 4 ||
         strides != strides.WithOnes() )
    {
        Failure(operation, "Neural-AI StridedSlice requires four unit strides");
        return false;
    }

    const int depthBegin = (options->begin_mask() & 0x8) != 0 ? 0 : begin.Depth();
    const int depthEnd = (options->end_mask() & 0x8) != 0 ? ifm->shape.Depth() : end.Depth();
    const bool c32Aligned = depthBegin % 32 == 0 && ofm->shape.Depth() % 32 == 0;
    const bool c32ToC16 = ifm->shape.Depth() == 32 && ofm->shape.Depth() == 16 &&
        (depthBegin == 0 || depthBegin == 16);
    if ( depthBegin < 0 || depthEnd <= depthBegin || depthEnd > ifm->shape.Depth() ||
         (!c32Aligned && !c32ToC16) || depthEnd - depthBegin != ofm->shape.Depth() )
    {
        Failure(operation,
            "Neural-AI StridedSlice requires a C32-aligned view or a C32-to-C16 half-depth slice");
        return false;
    }
    return true;
}

}  // namespace

TfLiteSupportedOperatorsNeuralAI::TfLiteSupportedOperatorsNeuralAI() :
        TfLiteSupportedOperators(std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(),
            std::numeric_limits<int64_t>::max(), s_supportedDataTypes, s_supportedOpTypes)
{
    viewPreservesDepth = {&ViewPreservesDepth,
        "Reshape, Squeeze, and ExpandDims must preserve the innermost channel dimension."};
    for ( const OpType opType : {OpType::Reshape, OpType::Squeeze, OpType::ExpandDims} )
        opConstraints[opType].push_back(&viewPreservesDepth);
    static ConstraintCheck s_preservesQuantization = {&PreservesQuantization,
        "The operation must preserve quantization."};
    opConstraints[OpType::ResizeNearestNeighbor].push_back(&s_preservesQuantization);
    opConstraints[OpType::MaxPool].push_back(&s_preservesQuantization);
    static ConstraintCheck s_supportedConcat = {&SupportedConcat,
        "Concat must match an aligned two-input or structural CSP/head channel-axis contract."};
    opConstraints[OpType::Concat].push_back(&s_supportedConcat);
    static ConstraintCheck s_supportedHeadTranspose = {&SupportedHeadTranspose,
        "Transpose must match the selected C144 detection-head pack contract."};
    opConstraints[OpType::Transpose].push_back(&s_supportedHeadTranspose);
    static ConstraintCheck s_supportedDfl16 = {&SupportedDfl16, "DFL must match the selected four-coordinate, 16-bin, 2100-location contract."};
    opConstraints[OpType::Dfl].push_back(&s_supportedDfl16);
    static ConstraintCheck s_supportedC32DepthSlice = {&SupportedC32DepthSlice,
        "StridedSlice must match the C32-aligned view or C32-to-C16 half-depth contract."};
    opConstraints[OpType::StridedSlice].push_back(&s_supportedC32DepthSlice);
}

}  // namespace regor
