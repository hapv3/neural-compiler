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
    case NeuralAIOpMode::Conv2DLinebufC32TailRequant: return "Conv2DLinebufC32TailRequant";
    case NeuralAIOpMode::DepthwiseC32S1Requant: return "DepthwiseC32S1Requant";
    case NeuralAIOpMode::DepthwiseC32S2Requant: return "DepthwiseC32S2Requant";
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
    if ( _hasOp || !dependsOn.empty() ||
         (op.type != OpType::FullyConnected && op.type != OpType::MatMul && op.type != OpType::Conv2D &&
             op.type != OpType::DepthwiseConv2D &&
             op.type != OpType::MemoryCopy) )
    {
        return 0;
    }
    _hasOp = true;
    return -1;
}

}  // namespace regor
