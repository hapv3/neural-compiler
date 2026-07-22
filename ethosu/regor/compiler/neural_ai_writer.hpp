//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "architecture/neuralai/neural_ai_abi.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace regor
{

struct CompiledNeuralAIArtifact
{
    std::vector<uint8_t> commands;
    std::vector<uint8_t> constants;
    std::vector<neuralai::TensorV1> tensors;
    std::vector<neuralai::BindingV1> bindings;
    std::vector<neuralai::QParamV1> qparams;
    uint32_t commandCount = 0;
    uint32_t requiredTCDMBytes = 0;
};

bool WriteNeuralAIModel(const CompiledNeuralAIArtifact &artifact, std::vector<uint8_t> &output, std::string &error);

}  // namespace regor
