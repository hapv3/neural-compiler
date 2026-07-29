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
    OpType::Conv2D,
    OpType::DepthwiseConv2D,
    OpType::ExpandDims,
    OpType::FullyConnected,
    OpType::Reshape,
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

}  // namespace

TfLiteSupportedOperatorsNeuralAI::TfLiteSupportedOperatorsNeuralAI() :
        TfLiteSupportedOperators(std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(),
            std::numeric_limits<int64_t>::max(), s_supportedDataTypes, s_supportedOpTypes)
{
    viewPreservesDepth = {&ViewPreservesDepth,
        "Reshape, Squeeze, and ExpandDims must preserve the innermost channel dimension."};
    for ( const OpType opType : {OpType::Reshape, OpType::Squeeze, OpType::ExpandDims} )
        opConstraints[opType].push_back(&viewPreservesDepth);
}

}  // namespace regor
