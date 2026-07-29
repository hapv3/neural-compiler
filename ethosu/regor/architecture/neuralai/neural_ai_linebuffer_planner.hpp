//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/shape.hpp"

#include <cstdint>
#include <vector>

namespace regor::neuralai
{

struct SystolicLinebufCfg
{
    uint32_t inputBase = 0;
    uint16_t inputH = 0;
    uint16_t inputW = 0;
    uint16_t inputC = 0;
    uint16_t outputW = 0;
    uint16_t strideH = 0;
    uint16_t strideW = 0;
    uint16_t padH = 0;
    uint16_t padW = 0;
    uint32_t rowStrideBytes = 0;
    uint32_t pixelStrideBytes = 0;
    uint32_t owStepBytes = 0;
    uint32_t ohStepBytes = 0;
    uint16_t kernelH = 0;
    uint16_t kernelW = 0;
    uint16_t cBase = 0;
    uint16_t laneBase = 0;
    uint16_t coalesce = 0;
    uint16_t kgen = 0;
    uint16_t pool = 0;
    uint16_t c32Fast = 0;
    uint16_t depthwise = 0;
    uint16_t c32GroupStationary = 0;
    uint16_t blockValidBytes = 0;
    uint16_t kSeedKh = 0;
    uint16_t kSeedKw = 0;
    uint16_t kSeedIc = 0;
    uint32_t kTiles = 0;
    uint32_t spatialM = 0;
    uint32_t channelAddrOffset = 0;
    uint32_t coalesceKBytes = 0;
};

struct SystolicGemm32Req
{
    uint32_t weightAddr = 0;
    uint32_t ifmAddr = 0;
    uint32_t psumAddr = 0;
    uint32_t ofmAddr = 0;
    uint32_t dimM = 0;
    uint32_t accumEn = 0;
    uint32_t ofmRowStrideBytes = 0;
    uint32_t ofmTileCols = 0;
    uint32_t psumRowStrideBytes = 0;
};

struct LinebufferJob
{
    SystolicLinebufCfg linebuf;
    SystolicGemm32Req gemm;
    uint32_t rows = 0;
    uint32_t kTiles = 0;
};

struct LinebufferPlannerInput
{
    Shape logicalIfm;
    Shape logicalOfm;
    uint32_t ifmBase = 0;
    uint32_t ofmBase = 0;
    uint32_t weightBase = 0;
    uint32_t psumBase = 0;
    int kernelH = 0;
    int kernelW = 0;
    int strideH = 0;
    int strideW = 0;
    int padTop = 0;
    int padLeft = 0;
    int padBottom = 0;
    int padRight = 0;
    int ic = 0;
    int oc = 0;
    int groupIndex = 0;
    int inputGroupIndex = 0;
    int outputGroupIndex = 0;
    int validLaneCount = 32;
    int ifmPixelStride = 0;
    int maxM = 256;
    int tcdmBudget = 0;
    int alreadyLiveBytes = 0;
    bool isDepthwise = false;
    // 0 = first/direct requant, 1 = intermediate accumulation, 2 = final
    // accumulation with requantization.  This is carried in GEMM32 accum_en.
    int accumMode = 0;
};

class LinebufferPlanner final
{
public:
    std::vector<LinebufferJob> Plan(const LinebufferPlannerInput &input) const;
};

static_assert(sizeof(SystolicLinebufCfg) == 80);
static_assert(sizeof(SystolicGemm32Req) == 36);
static_assert(sizeof(LinebufferJob) == 124);

}  // namespace regor::neuralai
