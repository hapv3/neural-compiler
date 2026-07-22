//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

namespace regor::neuralai
{

std::vector<uint8_t> PackGEMM32Weights(const int8_t *weightsKN, int depthK, int depthN);

}  // namespace regor::neuralai
