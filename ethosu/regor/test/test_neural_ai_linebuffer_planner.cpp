//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "architecture/neuralai/neural_ai_linebuffer_planner.hpp"

#include <catch_all.hpp>

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

TEST_CASE("Neural-AI linebuffer planner rejects an invalid lane count")
{
    auto input = GoldenInput();
    input.validLaneCount = 33;
    REQUIRE_THROWS_AS(LinebufferPlanner().Plan(input), std::invalid_argument);
}
