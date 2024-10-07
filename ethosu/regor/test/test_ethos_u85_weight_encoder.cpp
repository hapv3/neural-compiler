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

#include "common/common.hpp"

#include "architecture/ethosu85/ethos_u85.hpp"
#include "architecture/ethosu85/ethos_u85_weight_encoder.hpp"
#include "common/ini_reader.hpp"
#include "randomize.hpp"
#include "util.hpp"

#include <catch_all.hpp>

namespace
{

using namespace regor;

TEST_CASE("ethos_u85_weightsource")
{

    constexpr int CHUNK_SIZE = 128 * 1024;

    // Initialize Architecture
    auto arch = CreateArchDefault<ArchEthosU85>(128);
    // Initialize shapes
    Shape ifmShape{128, 128, 3};
    Shape ofmShape{64, 64, 16};
    Shape weightShape{16, 3, 3, 3};
    Kernel kernel{{3, 3}, {2, 2}, {1, 1}};
    std::vector<int> depthOffsets{0, ofmShape.Depth()};
    // Generate weights
    std::vector<uint8_t> weights;
    weights.resize(weightShape.Elements());
    randomize(weights);
    auto weightBuffer = std::make_shared<Buffer>(std::move(weights));
    Tensor weightTensor{"Weights", DataType::Int8, weightShape, weightBuffer};
    // Setup query
    ArchitectureConfigQuery query{};
    query.ifmBits = 8;
    query.ifmShape[0] = ifmShape;
    query.ofmShape = ofmShape;
    query.kernel = &kernel;
    query.lutBytes = 0;
    query.scaled = true;
    query.ifmResampling = ArchResampling::None;
    query.transpose = TransposeType::None;
    query.ofmFormat = TensorFormat::Unknown;
    // Setup weight source
    WeightTransformParam param;
    auto opCfg = arch->GetOpConfig(OpType::Conv2D, query);
    auto encoder = arch->WeightEncoder();
    auto view = weightTensor.View();
    WeightsRef weightsRef = {&view, AxisOrder::OHWI, weightTensor.Type()};
    auto config = encoder->GetEncodingConfig(opCfg.get(), weightsRef, &kernel, DataType::Int8, depthOffsets, WeightFormat::Default);
    auto transform = [](const WeightTransformParam *, int weight) { return weight; };
    auto source = encoder->GetWeightSource(config.get(), DataType::Int8, transform, &param);
    source->SetSource(view.RawData<uint8_t>(), 0, weightShape, view.StrideBytes(), 0);

    // Initialize buffers
    std::vector<int16_t> refBuffer, actBuffer;
    actBuffer.resize(CHUNK_SIZE, 0);
    refBuffer.resize(CHUNK_SIZE, 0);
    // Generate reordered weights reference
    int refSize = source->Get(refBuffer.data(), int(refBuffer.capacity()));
    REQUIRE(refSize > 0);
    REQUIRE(refSize < CHUNK_SIZE);
    refBuffer.resize(refSize);
    // Generate reordered weights and compare with reference for all possible chunk sizes
    for ( int chunkSize = refSize; chunkSize > 0; chunkSize-- )
    {
        CAPTURE(chunkSize);
        actBuffer.clear();
        actBuffer.resize(CHUNK_SIZE, 0);
        int actSize = source->Get(actBuffer.data(), chunkSize);
        int totalSize = actSize;
        while ( actSize == chunkSize )
        {
            actSize = source->Get(actBuffer.data() + totalSize, chunkSize);
            totalSize += actSize;
        }
        actBuffer.resize(totalSize);
        REQUIRE(refBuffer.size() == actBuffer.size());
        auto cmp = std::mismatch(refBuffer.begin(), refBuffer.end(), actBuffer.begin());
        if ( cmp.first != refBuffer.end() )
        {
            int index = std::distance(refBuffer.begin(), cmp.first);
            CAPTURE(index);
            REQUIRE(*cmp.first == *cmp.second);
        }
    }
}

}  // namespace
