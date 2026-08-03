//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_linebuffer_planner.hpp"

#include "common/numeric_util.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <stdexcept>

namespace regor::neuralai
{
namespace
{

constexpr int C32 = 32;

uint32_t CheckedAddress(uint64_t value)
{
    if ( value > std::numeric_limits<uint32_t>::max() )
        throw std::runtime_error("Neural-AI linebuffer address exceeds the 32-bit ABI");
    return uint32_t(value);
}

uint32_t CheckedU32(uint64_t value, const char *what)
{
    if ( value > std::numeric_limits<uint32_t>::max() )
        throw std::runtime_error(std::string("Neural-AI linebuffer ") + what + " exceeds the 32-bit ABI");
    return uint32_t(value);
}

uint16_t CheckedU16(int value, const char *what)
{
    if ( value < 0 || value > std::numeric_limits<uint16_t>::max() )
        throw std::runtime_error(std::string("Neural-AI linebuffer ") + what + " exceeds the 16-bit ABI");
    return uint16_t(value);
}

LinebufferJob MakeJob(const LinebufferPlannerInput &input, int ohBase, int owBase, int tileOh, int tileOw)
{
    const int ifmH = input.logicalIfm.Height();
    const int ifmW = input.logicalIfm.Width();
    const int ofmW = input.logicalOfm.Width();
    const int inputCStride = input.ifmPixelStride > 0 ? input.ifmPixelStride : input.ic;
    const uint64_t inputRowStride64 = uint64_t(ifmW) * uint64_t(inputCStride);
    const uint32_t inputRowStride = CheckedU32(inputRowStride64, "row stride");
    const int requestedFirstY = ohBase * input.strideH - input.padTop;
    const int requestedFirstX = owBase * input.strideW - input.padLeft;
    const int requestedLastY = (ohBase + tileOh - 1) * input.strideH - input.padTop + input.kernelH - 1;
    const int requestedLastX = (owBase + tileOw - 1) * input.strideW - input.padLeft + input.kernelW - 1;
    const int firstY = std::clamp(requestedFirstY, 0, ifmH - 1);
    const int firstX = std::clamp(requestedFirstX, 0, ifmW - 1);
    const int lastY = std::clamp(requestedLastY, 0, ifmH - 1);
    const int lastX = std::clamp(requestedLastX, 0, ifmW - 1);
    const int tileInputH = std::max(0, lastY - firstY + 1);
    const int tileInputW = std::max(0, lastX - firstX + 1);
    const int padH = std::max(0, firstY - requestedFirstY);
    const int padW = std::max(0, firstX - requestedFirstX);
    const int valid = std::clamp(input.validLaneCount, 1, C32);
    const int inputGroup = input.inputGroupIndex != 0 ? input.inputGroupIndex : input.groupIndex;
    const int outputGroup = std::max(0, input.outputGroupIndex);
    const int groupBase = std::max(0, inputGroup) * C32;
    const uint64_t kTotal = uint64_t(input.kernelH) * uint64_t(input.kernelW) * uint64_t(input.ic);
    const uint64_t kTiles64 = (kTotal + C32 - 1) / C32;
    const bool c32Fast = inputCStride == C32 && input.ic >= C32 && input.ic % C32 == 0 &&
                         valid == C32 && groupBase % C32 == 0 &&
                         (input.ifmBase % C32) == 0;
    /* Coalescing is also required for NHWC direct-RGB kernels: without it
       the formatter emits only one kernel tap per systolic input vector.
       C32-fast remains the additional predicate for the optimized KGEN and
       group-stationary paths. */
    const bool coalesce = input.kernelH * input.kernelW > 1;
    const bool kgen = coalesce && kTiles64 > 1;
    const bool groupStationary = c32Fast && coalesce && kgen && input.logicalIfm.Depth() >= C32 &&
                                 inputGroup >= 0;
    const uint64_t groupPlaneBytes = uint64_t(ifmH) * uint64_t(ifmW) * C32;
    const uint64_t inputGroupOffset = c32Fast ? uint64_t(std::max(0, inputGroup)) * groupPlaneBytes :
                                               uint64_t(groupBase);
    const uint64_t inputAddress = uint64_t(input.ifmBase) + inputGroupOffset +
        uint64_t(firstY) * inputRowStride64 + uint64_t(firstX) * uint64_t(inputCStride);
    const uint64_t outputGroupOffset = uint64_t(outputGroup) *
        uint64_t(input.logicalOfm.Height()) * uint64_t(ofmW) * C32;
    const uint64_t outputAddress = uint64_t(input.ofmBase) + outputGroupOffset +
        (uint64_t(ohBase) * uint64_t(ofmW) + uint64_t(owBase)) * C32;

    LinebufferJob job{};
    job.linebuf.inputBase = CheckedAddress(inputAddress);
    job.linebuf.inputH = CheckedU16(tileInputH, "input height");
    job.linebuf.inputW = CheckedU16(tileInputW, "input width");
    job.linebuf.inputC = CheckedU16(input.ic, "input channels");
    job.linebuf.outputW = CheckedU16(tileOw, "output width");
    job.linebuf.strideH = CheckedU16(input.strideH, "height stride");
    job.linebuf.strideW = CheckedU16(input.strideW, "width stride");
    job.linebuf.padH = CheckedU16(padH, "height padding");
    job.linebuf.padW = CheckedU16(padW, "width padding");
    job.linebuf.rowStrideBytes = inputRowStride;
    job.linebuf.pixelStrideBytes = CheckedU32(uint64_t(inputCStride), "pixel stride");
    job.linebuf.owStepBytes = CheckedU32(uint64_t(input.strideW) * uint64_t(inputCStride), "output-W step");
    job.linebuf.ohStepBytes = CheckedU32(uint64_t(input.strideH) * inputRowStride64, "output-H step");
    job.linebuf.kernelH = CheckedU16(input.kernelH, "kernel height");
    job.linebuf.kernelW = CheckedU16(input.kernelW, "kernel width");
    job.linebuf.cBase = 0;
    job.linebuf.laneBase = CheckedU16(groupBase % C32, "lane base");
    job.linebuf.coalesce = coalesce ? 1 : 0;
    job.linebuf.kgen = kgen ? 1 : 0;
    job.linebuf.c32Fast = c32Fast ? 1 : 0;
    job.linebuf.depthwise = input.isDepthwise ? 1 : 0;
    job.linebuf.c32GroupStationary = groupStationary ? 1 : 0;
    job.linebuf.blockValidBytes = CheckedU16(valid, "valid channel bytes");
    job.linebuf.kTiles = CheckedU32(kTiles64, "K-tile count");
    job.linebuf.spatialM = CheckedU32(uint64_t(tileOh) * uint64_t(tileOw), "spatial M");
    job.linebuf.channelAddrOffset = CheckedU32(groupStationary ? groupPlaneBytes : inputGroupOffset,
        "channel address offset");
    job.linebuf.coalesceKBytes = CheckedU32(uint64_t(input.kernelH) * uint64_t(input.kernelW) * uint64_t(valid),
        "coalesced K bytes");

    job.gemm.weightAddr = CheckedAddress(uint64_t(input.weightBase));
    job.gemm.psumAddr = CheckedAddress(uint64_t(input.psumBase));
    job.gemm.ofmAddr = CheckedAddress(outputAddress);
    job.gemm.dimM = CheckedU32(uint64_t(tileOh) * uint64_t(tileOw), "GEMM M");
    job.gemm.accumEn = input.accumMode != 0 ? uint32_t(input.accumMode) :
                       (inputGroup > 0 ? 1u : 0u);
    job.gemm.ofmRowStrideBytes = CheckedU32(uint64_t(ofmW) * C32, "OFM row stride");
    job.gemm.ofmTileCols = CheckedU32(uint64_t(tileOw), "OFM tile columns");
    job.gemm.psumRowStrideBytes = CheckedU32(uint64_t(tileOw) * C32 * 4, "PSum row stride");
    job.rows = job.gemm.dimM;
    job.kTiles = job.linebuf.kTiles;
    return job;
}

}  // namespace

std::vector<LinebufferJob> LinebufferPlanner::Plan(const LinebufferPlannerInput &input) const
{
    if ( input.logicalIfm.Batch() != 1 || input.logicalOfm.Batch() != 1 || input.kernelH <= 0 ||
         input.kernelW <= 0 || input.kernelH > 5 || input.kernelW > 5 || input.strideH < 1 ||
         input.strideW < 1 || input.strideH > 2 || input.strideW > 2 || input.ic <= 0 ||
         input.oc <= 0 || input.logicalIfm.Height() <= 0 || input.logicalIfm.Width() <= 0 ||
         input.logicalOfm.Height() <= 0 || input.logicalOfm.Width() <= 0 || input.maxM <= 0 ||
         input.groupIndex < 0 || input.inputGroupIndex < 0 || input.outputGroupIndex < 0 ||
         input.accumMode < 0 || input.accumMode > 3 || input.validLaneCount <= 0 ||
         input.validLaneCount > C32 )
        throw std::invalid_argument("Invalid Neural-AI linebuffer planner input");

    const int outputH = input.logicalOfm.Height();
    const int outputW = input.logicalOfm.Width();
    const int maxTileW = std::min(outputW, 640);
    int tileH = std::min(outputH, std::max(1, input.maxM / std::max(1, maxTileW)));
    if ( tileH == 0 ) tileH = 1;
    int tileW = std::min(maxTileW, std::max(1, input.maxM / tileH));
    if ( tileW == 0 ) tileW = 1;
    if ( tileH * tileW > input.maxM ) tileH = std::max(1, input.maxM / tileW);

    const uint64_t ifmBytes = uint64_t(input.logicalIfm.Height()) * uint64_t(input.logicalIfm.Width()) *
        uint64_t(std::max(input.ic, C32));
    const uint64_t ofmBytes = uint64_t(tileH) * uint64_t(tileW) * C32;
    /* Linebuffer weights are staged as complete 32x32 systolic tiles.  The
       previous estimate counted only one K lane and could accept a candidate
       that overflowed TCDM once the encoded tile was loaded. */
    const uint64_t weightBytes = uint64_t(input.kernelH) * uint64_t(input.kernelW) *
        uint64_t(input.isDepthwise ? C32 : C32 * C32);
    const uint64_t psumBytes = input.accumMode != 0 ? ofmBytes * sizeof(int32_t) : 0u;
    const uint64_t required = uint64_t(std::max(0, input.alreadyLiveBytes)) + ifmBytes +
        ofmBytes + weightBytes + psumBytes;
    if ( input.tcdmBudget > 0 && required > uint64_t(input.tcdmBudget) )
        throw std::runtime_error("Neural-AI linebuffer tile exceeds TCDM budget");

    std::vector<LinebufferJob> jobs;
    for ( int oh = 0; oh < outputH; oh += tileH )
    {
        for ( int ow = 0; ow < outputW; ow += tileW )
        {
            jobs.push_back(MakeJob(input, oh, ow, std::min(tileH, outputH - oh),
                std::min(tileW, outputW - ow)));
        }
    }
    return jobs;
}

}  // namespace regor::neuralai
