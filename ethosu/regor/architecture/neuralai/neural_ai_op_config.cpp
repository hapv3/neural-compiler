//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_op_config.hpp"

#include <fmt/format.h>

namespace regor
{

std::unique_ptr<ArchitectureOpConfig> NeuralAIOpConfig::Clone()
{
    return std::make_unique<NeuralAIOpConfig>(_maxRows);
}

std::string NeuralAIOpConfig::ToString(bool full)
{
    return full ? fmt::format("Neural-AI GEMM32, max rows {}, depth granule 32", _maxRows) : "GEMM32";
}

int NeuralAIOpGroup::Add(const ArchitectureOpGroupQuery &op, const std::vector<int> &dependsOn)
{
    if ( _hasOp || !dependsOn.empty() ||
         (op.type != OpType::FullyConnected && op.type != OpType::MatMul && op.type != OpType::MemoryCopy) )
    {
        return 0;
    }
    _hasOp = true;
    return -1;
}

}  // namespace regor
