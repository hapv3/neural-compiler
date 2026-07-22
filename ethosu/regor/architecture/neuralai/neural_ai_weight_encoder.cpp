//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_weight_encoder.hpp"

#include "common/numeric_util.hpp"

#include <stdexcept>

namespace regor::neuralai
{

std::vector<uint8_t> PackGEMM32Weights(const int8_t *weightsKN, int depthK, int depthN)
{
    if ( !weightsKN || depthK <= 0 || depthN <= 0 )
    {
        throw std::invalid_argument("Neural-AI GEMM weights require positive K and N dimensions");
    }

    constexpr int tile = 32;
    const int kGroups = RoundAway(depthK, tile) / tile;
    const int nGroups = RoundAway(depthN, tile) / tile;
    std::vector<uint8_t> packed(size_t(kGroups) * nGroups * tile * tile, 0);
    size_t output = 0;
    for ( int nGroup = 0; nGroup < nGroups; ++nGroup )
    {
        for ( int kGroup = 0; kGroup < kGroups; ++kGroup )
        {
            for ( int kLane = 0; kLane < tile; ++kLane )
            {
                const int k = kGroup * tile + kLane;
                for ( int nLane = 0; nLane < tile; ++nLane )
                {
                    const int n = nGroup * tile + nLane;
                    if ( k < depthK && n < depthN )
                    {
                        packed[output] = uint8_t(weightsKN[k * depthN + n]);
                    }
                    ++output;
                }
            }
        }
    }
    return packed;
}

}  // namespace regor::neuralai
