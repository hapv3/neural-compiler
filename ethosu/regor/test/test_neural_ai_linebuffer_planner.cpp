//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "architecture/neuralai/neural_ai_linebuffer_planner.hpp"

#include <catch_all.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

using namespace regor;
using namespace regor::neuralai;

namespace
{

LinebufferPlannerInput GoldenInput()
{
    LinebufferPlannerInput input{};
    input.logicalIfm = Shape(1, 8, 8, 64);
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
    input.ic = 32;
    input.oc = 32;
    input.inputGroupIndex = 0;
    input.outputGroupIndex = 0;
    input.validLaneCount = 32;
    input.ifmPixelStride = 32;
    input.maxM = 256;
    input.tcdmBudget = 512 * 1024;
    return input;
}

}  // namespace

namespace
{

void RequireStableJobFields(const LinebufferJob &job, uint32_t inputBase, uint16_t inputH,
    uint16_t inputW, uint16_t inputC, uint16_t outputW, uint32_t rowStride,
    uint32_t pixelStride, uint32_t owStep, uint32_t ohStep, uint16_t padH, uint16_t padW,
    uint16_t c32Fast, uint16_t groupStationary, uint16_t validBytes, uint32_t kTiles,
    uint32_t spatialM, uint32_t channelOffset, uint32_t coalesceKBytes,
    uint32_t weightAddr, uint32_t ofmAddr, uint32_t dimM, uint32_t accumEn,
    uint32_t ofmRowStride, uint32_t ofmTileCols, uint32_t psumRowStride)
{
    REQUIRE(job.linebuf.inputBase == inputBase);
    REQUIRE(job.linebuf.inputH == inputH);
    REQUIRE(job.linebuf.inputW == inputW);
    REQUIRE(job.linebuf.inputC == inputC);
    REQUIRE(job.linebuf.outputW == outputW);
    REQUIRE(job.linebuf.rowStrideBytes == rowStride);
    REQUIRE(job.linebuf.pixelStrideBytes == pixelStride);
    REQUIRE(job.linebuf.owStepBytes == owStep);
    REQUIRE(job.linebuf.ohStepBytes == ohStep);
    REQUIRE(job.linebuf.padH == padH);
    REQUIRE(job.linebuf.padW == padW);
    REQUIRE(job.linebuf.c32Fast == c32Fast);
    REQUIRE(job.linebuf.c32GroupStationary == groupStationary);
    REQUIRE(job.linebuf.blockValidBytes == validBytes);
    REQUIRE(job.linebuf.kTiles == kTiles);
    REQUIRE(job.linebuf.spatialM == spatialM);
    REQUIRE(job.linebuf.channelAddrOffset == channelOffset);
    REQUIRE(job.linebuf.coalesceKBytes == coalesceKBytes);
    REQUIRE(job.gemm.weightAddr == weightAddr);
    REQUIRE(job.gemm.ofmAddr == ofmAddr);
    REQUIRE(job.gemm.dimM == dimM);
    REQUIRE(job.gemm.accumEn == accumEn);
    REQUIRE(job.gemm.ofmRowStrideBytes == ofmRowStride);
    REQUIRE(job.gemm.ofmTileCols == ofmTileCols);
    REQUIRE(job.gemm.psumRowStrideBytes == psumRowStride);
    REQUIRE(job.rows == spatialM);
    REQUIRE(job.kTiles == kTiles);
}

LinebufferPlannerInput GoldenInputForShape(int ifmH, int ifmW, int ifmC, int ofmH, int ofmW,
    int ofmC = 32)
{
    LinebufferPlannerInput input{};
    input.logicalIfm = Shape(1, ifmH, ifmW, ifmC);
    input.logicalOfm = Shape(1, ofmH, ofmW, ofmC);
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
    input.ic = ifmC;
    input.oc = ofmC;
    input.validLaneCount = std::min(ifmC, 32);
    input.ifmPixelStride = ifmC;
    input.maxM = 256;
    input.tcdmBudget = 512 * 1024;
    return input;
}

}  // namespace

TEST_CASE("Neural-AI linebuffer planner emits the stable top-left golden")
{
    const auto jobs = LinebufferPlanner().Plan(GoldenInput());
    REQUIRE(jobs.size() == 1);
    const auto &job = jobs.front();

    // Immutable wire-field golden for an 8x8, C32, K3/S1/P1 tile.
    REQUIRE(job.linebuf.inputBase == 0x1000u);
    REQUIRE(job.linebuf.inputH == 8u);
    REQUIRE(job.linebuf.inputW == 8u);
    REQUIRE(job.linebuf.inputC == 32u);
    REQUIRE(job.linebuf.outputW == 8u);
    REQUIRE(job.linebuf.padH == 1u);
    REQUIRE(job.linebuf.padW == 1u);
    REQUIRE(job.linebuf.rowStrideBytes == 256u);
    REQUIRE(job.linebuf.pixelStrideBytes == 32u);
    REQUIRE(job.linebuf.owStepBytes == 32u);
    REQUIRE(job.linebuf.ohStepBytes == 256u);
    REQUIRE(job.linebuf.kTiles == 9u);
    REQUIRE(job.linebuf.spatialM == 64u);
    REQUIRE(job.linebuf.blockValidBytes == 32u);
    REQUIRE(job.linebuf.c32Fast == 1u);
    REQUIRE(job.linebuf.c32GroupStationary == 1u);
    REQUIRE(job.gemm.weightAddr == 0x2000u);
    REQUIRE(job.gemm.ofmAddr == 0x3000u);
    REQUIRE(job.gemm.dimM == 64u);
    REQUIRE(job.gemm.accumEn == 0u);
    REQUIRE(job.gemm.ofmRowStrideBytes == 256u);
    REQUIRE(job.gemm.ofmTileCols == 8u);
    REQUIRE(job.gemm.psumRowStrideBytes == 1024u);
}

TEST_CASE("Neural-AI linebuffer planner splits M and width at ABI limits")
{
    auto input = GoldenInput();
    input.logicalIfm = Shape(1, 1, 641, 32);
    input.logicalOfm = Shape(1, 1, 641, 32);
    input.maxM = 256;
    const auto jobs = LinebufferPlanner().Plan(input);
    REQUIRE(jobs.size() == 3);
    for ( const auto &job : jobs )
    {
        REQUIRE(job.rows <= 256u);
        REQUIRE(job.linebuf.inputW <= 640u);
        REQUIRE(job.linebuf.spatialM == job.rows);
    }
    REQUIRE(jobs[0].rows == 256u);
    REQUIRE(jobs[1].rows == 256u);
    REQUIRE(jobs[2].rows == 129u);

    input.logicalIfm = Shape(1, 1, 511, 32);
    input.logicalOfm = Shape(1, 1, 511, 32);
    const auto rows = LinebufferPlanner().Plan(input);
    REQUIRE(rows.size() == 2);
    REQUIRE(rows[0].rows == 256u);
    REQUIRE(rows[1].rows == 255u);
}

TEST_CASE("Neural-AI linebuffer planner rejects address overflow")
{
    auto input = GoldenInput();
    input.ifmBase = 0xffffffffu;
    input.logicalIfm = Shape(1, 8, 8, 64);
    input.inputGroupIndex = 1;
    REQUIRE_THROWS_AS(LinebufferPlanner().Plan(input), std::runtime_error);
}

TEST_CASE("Neural-AI linebuffer planner accounts for encoded weights and partial sums")
{
    auto input = GoldenInput();
    /* 2 KiB IFM + 2 KiB OFM + 9 KiB encoded K3 tile + 8 KiB PSum. */
    input.accumMode = 1;
    input.tcdmBudget = 21u * 1024u;
    REQUIRE(LinebufferPlanner().Plan(input).size() == 1);

    input.tcdmBudget = 21u * 1024u - 1u;
    REQUIRE_THROWS_AS(LinebufferPlanner().Plan(input), std::runtime_error);
}

TEST_CASE("Neural-AI linebuffer planner budgets a rolling RGB stem by resident rows")
{
    auto input = GoldenInputForShape(320, 320, 3, 160, 160);
    input.strideH = 2;
    input.strideW = 2;
    input.ifmPixelStride = 3;
    input.validLaneCount = 3;
    input.maxM = 1024;
    input.tcdmBudget = 64 * 1024;

    REQUIRE_THROWS_AS(LinebufferPlanner().Plan(input), std::runtime_error);

    input.rollingBuffers = true;
    const auto jobs = LinebufferPlanner().Plan(input);
    REQUIRE(jobs.size() == 27);
    const auto footprint = LinebufferPlanner().RollingFootprint(input, jobs.front());
    REQUIRE(footprint.ifmBytes == 12u * 320u * 3u);
    REQUIRE(footprint.ofmBytes == 6u * 160u * 32u);
    REQUIRE(footprint.weightBytes == 3u * 3u * 32u * 32u);
    REQUIRE(footprint.psumBytes == 0u);
    REQUIRE(footprint.totalBytes == 51456u);
}

TEST_CASE("Neural-AI linebuffer planner rejects an invalid lane count")
{
    auto input = GoldenInput();
    input.validLaneCount = 33;
    REQUIRE_THROWS_AS(LinebufferPlanner().Plan(input), std::invalid_argument);
}

TEST_CASE("Neural-AI linebuffer planner matches the interior Python golden fields")
{
    auto input = GoldenInputForShape(8, 8, 32, 4, 4);
    input.padTop = input.padLeft = input.padBottom = input.padRight = 0;
    const auto jobs = LinebufferPlanner().Plan(input);
    REQUIRE(jobs.size() == 1);
    RequireStableJobFields(jobs.front(), 0x1000u, 6u, 6u, 32u, 4u,
        256u, 32u, 32u, 256u, 0u, 0u, 1u, 1u, 32u, 9u, 16u,
        2048u, 288u, 0x2000u, 0x3000u, 16u, 0u, 128u, 4u, 512u);
}

TEST_CASE("Neural-AI linebuffer planner matches the C31 tail golden fields")
{
    auto input = GoldenInputForShape(8, 8, 31, 8, 8);
    input.ifmPixelStride = 31;
    const auto jobs = LinebufferPlanner().Plan(input);
    REQUIRE(jobs.size() == 1);
    RequireStableJobFields(jobs.front(), 0x1000u, 8u, 8u, 31u, 8u,
        248u, 31u, 31u, 248u, 1u, 1u, 0u, 0u, 31u, 9u, 64u,
        0u, 279u, 0x2000u, 0x3000u, 64u, 0u, 256u, 8u, 1024u);
}

TEST_CASE("Neural-AI linebuffer planner disables fast mode for an unaligned base")
{
    auto input = GoldenInputForShape(8, 8, 64, 8, 8);
    input.ifmPixelStride = 32;
    input.ifmBase = 0x1003;
    const auto jobs = LinebufferPlanner().Plan(input);
    REQUIRE(jobs.size() == 1);
    REQUIRE(jobs.front().linebuf.c32Fast == 0u);
    REQUIRE(jobs.front().linebuf.c32GroupStationary == 0u);
    REQUIRE(jobs.front().linebuf.channelAddrOffset == 0u);
}
