//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "architecture/neuralai/neural_ai_constraints.hpp"
#include "architecture/neuralai/neural_ai_linebuffer_planner.hpp"
#include "architecture/neuralai/neural_ai_quantization.hpp"
#include "architecture/neuralai/neural_ai.hpp"
#include "architecture/neuralai/neural_ai_weight_encoder.hpp"

#include <catch_all.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

using namespace regor;
using namespace regor::neuralai;

TEST_CASE("Neural-AI Phase 3 classifier selects the stable Conv modes")
{
    const Kernel rgb({3, 3}, {2, 2}, {1, 1}, Margin(1, 1, 1, 1));
    auto classification = NeuralAIConstraints::Classify(OpType::Conv2D,
        Shape(1, 8, 8, 3), Shape(32, 3, 3, 3), Shape(1, 4, 4, 32), &rgb);
    REQUIRE(classification.mode == NeuralAIOpMode::Conv2DRgbLinebufRequant);
    REQUIRE(classification.directNhwcInput);

    const Kernel generic({3, 3}, {1, 1}, {1, 1}, Margin(1, 1, 1, 1));
    classification = NeuralAIConstraints::Classify(OpType::Conv2D,
        Shape(1, 8, 8, 64), Shape(33, 3, 3, 64), Shape(1, 8, 8, 33), &generic);
    REQUIRE(classification.mode == NeuralAIOpMode::Unsupported);
    REQUIRE(classification.hasIcTail == false);
    REQUIRE(classification.hasOcTail);
    REQUIRE_FALSE(classification.groupStationary);
    REQUIRE(classification.diagnostic.find("16-lane tails") != std::string::npos);

    classification = NeuralAIConstraints::Classify(OpType::Conv2D,
        Shape(1, 8, 8, 64), Shape(64, 3, 3, 64), Shape(1, 8, 8, 64), &generic);
    REQUIRE(classification.mode == NeuralAIOpMode::Conv2DLinebufC32S1Requant);
    REQUIRE(classification.groupStationary);

    classification = NeuralAIConstraints::Classify(OpType::DepthwiseConv2D,
        Shape(1, 8, 8, 33), Shape(1, 3, 3, 33), Shape(1, 4, 4, 33), &rgb);
    REQUIRE(classification.mode == NeuralAIOpMode::DepthwiseC32S2Requant);

    classification = NeuralAIConstraints::Classify(OpType::DepthwiseConv2D,
        Shape(1, 8, 8, 33), Shape(1, 3, 3, 1), Shape(1, 4, 4, 33), &rgb);
    REQUIRE(classification.mode == NeuralAIOpMode::Unsupported);
    REQUIRE(classification.diagnostic.find("Phase 3") != std::string::npos);

    const Kernel invalid({3, 3}, {3, 3}, {1, 1}, Margin(1, 1, 1, 1));
    classification = NeuralAIConstraints::Classify(OpType::Conv2D,
        Shape(1, 8, 8, 32), Shape(32, 3, 3, 32), Shape(1, 3, 3, 32), &invalid);
    REQUIRE(classification.mode == NeuralAIOpMode::Unsupported);
    REQUIRE(classification.diagnostic.find("stride") != std::string::npos);

    classification = NeuralAIConstraints::Classify(OpType::Conv2D,
        Shape(1, 8, 8, 32), Shape(32, 3, 3, 32), Shape(1, 7, 8, 32), &generic);
    REQUIRE(classification.mode == NeuralAIOpMode::Unsupported);
    REQUIRE(classification.diagnostic.find("spatial shape") != std::string::npos);
}

TEST_CASE("Neural-AI Phase 3 quantization is deterministic and symmetric")
{
    int32_t multiplier = 0;
    int32_t shift = 0;
    REQUIRE(CalculateQuantizedMultiplier(1.0, multiplier, shift));
    REQUIRE(multiplier == (1 << 30));
    REQUIRE(shift == 30);
    REQUIRE(CalculateQuantizedMultiplier(0.5, multiplier, shift));
    REQUIRE(multiplier == (1 << 30));
    REQUIRE(shift == 31);
    REQUIRE_FALSE(CalculateQuantizedMultiplier(0.0, multiplier, shift));
    REQUIRE_FALSE(CalculateQuantizedMultiplier(-1.0, multiplier, shift));

    Quantization ifm;
    ifm.scales = {QuantizedScale(0.25)};
    Quantization weights;
    weights.scales = {QuantizedScale(0.5), QuantizedScale(0.25)};
    Quantization ofm;
    ofm.scales = {QuantizedScale(0.5)};
    std::vector<QuantizedScale> scales;
    REQUIRE(GenerateQuantizationParameters(ifm, weights, ofm, scales));
    REQUIRE(scales.size() == 2);
    REQUIRE(scales[0].Dequantize() == Catch::Approx(0.25));
    REQUIRE(scales[1].Dequantize() == Catch::Approx(0.125));
    REQUIRE(Requantize(1, QuantizedScale(1, 1), 0, -128, 127) == 1);
    REQUIRE(Requantize(-1, QuantizedScale(1, 1), 0, -128, 127) == -1);
}

TEST_CASE("Neural-AI Phase 3 requantization matches an independent 10000-seed reference")
{
    uint64_t state = 0x9E3779B97F4A7C15ULL;
    const auto next = [&state]() {
        state ^= state << 7;
        state ^= state >> 9;
        state ^= state << 8;
        return state;
    };
    const auto reference = [](int64_t accumulator, int32_t multiplier, uint32_t shift,
                              int32_t zeroPoint, int32_t clampMin, int32_t clampMax) {
        const __int128 product = __int128(accumulator) * __int128(multiplier);
        const __int128 magnitude = product < 0 ? -product : product;
        const __int128 rounded = shift == 0 ? magnitude :
            ((magnitude + (__int128(1) << (shift - 1))) >> shift);
        const __int128 scaled = product < 0 ? -rounded : rounded;
        const __int128 shifted = scaled + zeroPoint;
        return int32_t(std::clamp<__int128>(shifted, clampMin, clampMax));
    };

    for ( int seed = 0; seed < 10000; ++seed )
    {
        const int64_t accumulator = int64_t(next() % 2000000001ULL) - 1000000000LL;
        const int32_t multiplier = int32_t(1 + (next() % uint64_t(std::numeric_limits<int32_t>::max())));
        const uint32_t shift = uint32_t(next() % 32u);
        const int32_t zeroPoint = int32_t(next() % 255u) - 127;
        const int32_t clampMin = -128 + int32_t(next() % 64u);
        const int32_t clampMax = clampMin + 1 + int32_t(next() % uint32_t(127 - clampMin));
        const int32_t actual = Requantize(accumulator, QuantizedScale(multiplier, int32_t(shift)),
                                          zeroPoint, clampMin, clampMax);
        REQUIRE(actual == reference(accumulator, multiplier, shift, zeroPoint, clampMin, clampMax));
    }
}

TEST_CASE("Neural-AI Phase 3 planner respects M and C32 group predicates")
{
    LinebufferPlannerInput input{};
    input.logicalIfm = Shape(1, 10, 10, 64);
    input.logicalOfm = Shape(1, 8, 8, 32);
    input.ifmBase = 0x1000;
    input.ofmBase = 0x3000;
    input.weightBase = 0x2000;
    input.kernelH = 3;
    input.kernelW = 3;
    input.strideH = 1;
    input.strideW = 1;
    input.padTop = 1;
    input.padLeft = 1;
    input.padBottom = 1;
    input.padRight = 1;
    input.ic = 64;
    input.oc = 32;
    input.groupIndex = 0;
    input.validLaneCount = 32;
    input.ifmPixelStride = 32;
    input.maxM = 256;
    input.tcdmBudget = 512 * 1024;

    const auto jobs = LinebufferPlanner().Plan(input);
    REQUIRE_FALSE(jobs.empty());
    for ( const auto &job : jobs )
    {
        REQUIRE(job.rows <= 256);
        REQUIRE(job.linebuf.c32Fast == 1);
        REQUIRE(job.linebuf.coalesce == 1);
        REQUIRE(job.linebuf.kgen == 1);
        REQUIRE(job.linebuf.c32GroupStationary == 1);
        REQUIRE(job.gemm.ofmRowStrideBytes == 8 * 32);
    }

    input.logicalIfm = Shape(1, 1, 511, 32);
    input.logicalOfm = Shape(1, 1, 511, 32);
    input.ic = 32;
    input.oc = 32;
    input.ifmPixelStride = 32;
    input.maxM = 256;
    const auto striped = LinebufferPlanner().Plan(input);
    REQUIRE(striped.size() == 2);
    REQUIRE(striped[0].rows == 256);
    REQUIRE(striped[1].rows == 255);

    input.logicalIfm = Shape(1, 8, 8, 3);
    input.logicalOfm = Shape(1, 4, 4, 32);
    input.ic = 3;
    input.oc = 32;
    input.strideH = 2;
    input.strideW = 2;
    input.ifmPixelStride = 3;
    input.validLaneCount = 3;
    const auto rgb = LinebufferPlanner().Plan(input);
    REQUIRE(rgb.size() == 1);
    REQUIRE(rgb[0].linebuf.c32Fast == 0);
    REQUIRE(rgb[0].linebuf.coalesce == 1);
    REQUIRE(rgb[0].linebuf.kgen == 0);
    REQUIRE(rgb[0].linebuf.c32GroupStationary == 0);
    REQUIRE(rgb[0].linebuf.blockValidBytes == 3);

    input.ifmBase = std::numeric_limits<uint32_t>::max();
    input.logicalIfm = Shape(1, 8, 8, 64);
    input.logicalOfm = Shape(1, 8, 8, 32);
    input.ic = 32;
    input.inputGroupIndex = 1;
    input.ifmPixelStride = 32;
    input.maxM = 64;
    REQUIRE_THROWS_AS(LinebufferPlanner().Plan(input), std::runtime_error);
}

TEST_CASE("Neural-AI Phase 3 depthwise weights are grouped by channel")
{
    ArchNeuralAI arch;
    NeuralAIOpConfig configObject(256, NeuralAIOpMode::DepthwiseC32S1Requant);
    auto config = arch.WeightEncoder()->GetEncodingConfig(&configObject, nullptr, DataType::Int8,
        Flags<WeightFormat>(WeightFormat::Default));
    auto source = arch.WeightEncoder()->GetWeightSource(config.get(), DataType::Int8, nullptr, nullptr);
    std::vector<int8_t> weights(3 * 3 * 33);
    for ( int h = 0; h < 3; ++h )
        for ( int w = 0; w < 3; ++w )
            for ( int c = 0; c < 33; ++c ) weights[(h * 3 + w) * 33 + c] = int8_t(c + h * 3 + w);
    source->SetSource(weights.data(), 0, Shape(1, 3, 3, 33), Shape(297, 99, 33, 1), 0);
    std::vector<uint8_t> encoded;
    const auto info = arch.WeightEncoder()->EncodeWeights(config.get(), source.get(), encoded);
    REQUIRE(info.sourceSize == 3 * 3 * 33);
    REQUIRE(encoded.size() == 2 * 3 * 3 * 32);
    REQUIRE(int8_t(encoded[0]) == 0);
    REQUIRE(int8_t(encoded[31]) == 31);
    REQUIRE(int8_t(encoded[32]) == 1);
    REQUIRE(int8_t(encoded[8 * 32]) == 8);
    REQUIRE(int8_t(encoded[8 * 32 + 31]) == 39);
    REQUIRE(int8_t(encoded[9 * 32]) == 32);
    REQUIRE(int8_t(encoded[9 * 32 + 1]) == 0);
}
