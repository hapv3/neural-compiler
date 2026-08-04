//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_op_config.hpp"

#include <fmt/format.h>

namespace regor
{

const char *NeuralAIOpModeName(NeuralAIOpMode mode)
{
    switch ( mode )
    {
    case NeuralAIOpMode::Unsupported: return "Unsupported";
    case NeuralAIOpMode::FullyConnectedRow32: return "FullyConnectedRow32";
    case NeuralAIOpMode::MatMulRow32: return "MatMulRow32";
    case NeuralAIOpMode::Conv2DRgbLinebufRequant: return "Conv2DRgbLinebufRequant";
    case NeuralAIOpMode::Conv2DPointwiseC32Requant: return "Conv2DPointwiseC32Requant";
    case NeuralAIOpMode::Conv2DLinebufC32S1Requant: return "Conv2DLinebufC32S1Requant";
    case NeuralAIOpMode::Conv2DLinebufC32S2Requant: return "Conv2DLinebufC32S2Requant";
    case NeuralAIOpMode::DepthwiseC32S1Requant: return "DepthwiseC32S1Requant";
    case NeuralAIOpMode::DepthwiseC32S2Requant: return "DepthwiseC32S2Requant";
    case NeuralAIOpMode::AFULutI8: return "AFULutI8";
    case NeuralAIOpMode::AFUBinaryAddI8: return "AFUBinaryAddI8";
    case NeuralAIOpMode::AFUGlobalAvgPoolC32: return "AFUGlobalAvgPoolC32";
    case NeuralAIOpMode::UpsampleNearestC32: return "UpsampleNearestC32";
    case NeuralAIOpMode::MaxPoolK5S1P2C32: return "MaxPoolK5S1P2C32";
    default: return "Unsupported";
    }
}

std::unique_ptr<ArchitectureOpConfig> NeuralAIOpConfig::Clone()
{
    return std::make_unique<NeuralAIOpConfig>(_maxRows, _mode, _directNhwcInput, _groupStationary);
}

std::string NeuralAIOpConfig::ToString(bool full)
{
    return full ? fmt::format("Neural-AI GEMM32/pointwise Conv, max rows {}, depth granule 32", _maxRows) :
                  "GEMM32";
}

int NeuralAIOpGroup::Add(const ArchitectureOpGroupQuery &op, const std::vector<int> &dependsOn)
{
    if ( _hasOp )
    {
        if ( !_allowsActivation || _hasActivation || !IsClipping(op.type) ||
             dependsOn.size() != 1 || dependsOn[0] != -1 )
            return 0;
        _hasActivation = true;
        return -2;
    }
    if ( !dependsOn.empty() ||
         (op.type != OpType::FullyConnected && op.type != OpType::MatMul &&
             op.type != OpType::Conv2D && op.type != OpType::DepthwiseConv2D &&
             op.type != OpType::LUT && op.type != OpType::Add && op.type != OpType::AvgPool &&
             op.type != OpType::MaxPool &&
             op.type != OpType::MemoryCopy) )
    {
        return 0;
    }
    _hasOp = true;
    _allowsActivation = op.type == OpType::FullyConnected || op.type == OpType::MatMul ||
                        op.type == OpType::Conv2D || op.type == OpType::DepthwiseConv2D;
    _allowsIFMReuse = op.type != OpType::Add && op.type != OpType::LUT &&
                      op.type != OpType::AvgPool && op.type != OpType::MaxPool;
    return -1;
}

}  // namespace regor
