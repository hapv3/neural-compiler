//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "tflite_supported_operators_neural_ai.hpp"

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
    OpType::ExpandDims,
    OpType::FullyConnected,
    OpType::LUT,
    OpType::MaxPool,
    OpType::Reshape,
    OpType::ResizeNearestNeighbor,
    OpType::Sigmoid,
    OpType::Squeeze,
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
    if ( lhs == nullptr || rhs == nullptr || ofm == nullptr || ifmCount != 2 ||
         axis != lhs->shape.Size() - 1 || lhs->shape.Size() != rhs->shape.Size() ||
         lhs->shape.Size() != ofm->shape.Size() ||
         lhs->shape.WithDepth(1) != rhs->shape.WithDepth(1) ||
         lhs->shape.WithDepth(1) != ofm->shape.WithDepth(1) ||
         lhs->shape.Depth() % 32 != 0 || rhs->shape.Depth() % 32 != 0 ||
         ofm->shape.Depth() != lhs->shape.Depth() + rhs->shape.Depth() ||
         lhs->quantization != rhs->quantization || lhs->quantization != ofm->quantization )
    {
        Failure(operation,
            "Neural-AI Concat requires two equal-spatial, equal-quantization, C32-aligned inputs on the channel axis");
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
        "Concat must match the two-input C32 channel-axis contract."};
    opConstraints[OpType::Concat].push_back(&s_supportedConcat);
}

}  // namespace regor
