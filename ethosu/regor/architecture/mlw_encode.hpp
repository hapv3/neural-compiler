//
// SPDX-FileCopyrightText: Copyright 2021-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#pragma once

#include <cstdint>
#include <vector>

enum class WeightFormat : uint16_t
{
    Default = 1 << 0,
    Fast = 1 << 1,
    Sparse2_4 = 1 << 2
};

inline constexpr bool operator&(WeightFormat type, WeightFormat mask)
{
    return bool(unsigned(type) & unsigned(mask));
}

struct MlwEncodeResult
{
    int elements_read = 0;
    int bytes_written = 0;
    int zero_count = 0;
    int distinct_values = 0;
    uint64_t distinct_weights[512 / 64]{0};
};

class IWeightSource;

class IWeightSource
{
public:
    virtual ~IWeightSource() = default;
    virtual int Elements() = 0;
    virtual int Get(int16_t *buffer, int count) = 0;
};

MlwEncodeResult mle_encode_proxy(IWeightSource *source, int chunkSize, std::vector<uint8_t> &output, unsigned encodeFlags);
MlwEncodeResult mle_encode_fwd_proxy(IWeightSource *source, int chunkSize, std::vector<uint8_t> &output, unsigned encodeFlags);
