//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_command_ir.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>

namespace regor::neuralai
{
namespace
{

constexpr uint32_t DMAQueueDepth = 16;
constexpr uint16_t TCDMWordBytes = 32;
constexpr uint16_t TCDMBanks = 16;

uint16_t Read16(const uint8_t *data)
{
    return uint16_t(data[0]) | uint16_t(uint16_t(data[1]) << 8);
}

uint32_t Read32(const uint8_t *data)
{
    return uint32_t(data[0]) | (uint32_t(data[1]) << 8) | (uint32_t(data[2]) << 16) |
           (uint32_t(data[3]) << 24);
}

uint64_t SaturatingMultiply(uint64_t lhs, uint64_t rhs)
{
    if ( lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs )
        return std::numeric_limits<uint64_t>::max();
    return lhs * rhs;
}

uint64_t SaturatingAdd(uint64_t lhs, uint64_t rhs)
{
    if ( rhs > std::numeric_limits<uint64_t>::max() - lhs )
        return std::numeric_limits<uint64_t>::max();
    return lhs + rhs;
}

int64_t CycleCount(uint64_t value)
{
    return int64_t(std::min<uint64_t>(
        std::max<uint64_t>(1, value), uint64_t(std::numeric_limits<int64_t>::max())));
}

uint64_t DivRoundUp(uint64_t value, uint64_t divisor)
{
    return value / divisor + (value % divisor != 0);
}

uint64_t StorageKey(uint16_t region, uint16_t index)
{
    return (uint64_t(region) << 32) | index;
}

uint64_t InitialContentIdentity(uint16_t region, uint16_t index)
{
    return 0x8000000000000000ULL | StorageKey(region, index);
}

uint64_t CommandContentIdentity(uint32_t command, uint32_t access)
{
    return (uint64_t(command + 1) << 32) | uint64_t(access + 1);
}

uint64_t StateKey(SemanticState state, uint32_t instance)
{
    return (uint64_t(uint8_t(state)) << 32) | instance;
}

uint64_t HashWords(const uint8_t *command, int offset, int words)
{
    uint64_t hash = 1469598103934665603ULL;
    for ( int word = 0; word < words; ++word )
    {
        hash ^= Read32(command + offset + word * 4);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool SameStorage(const SemanticMemoryAccess &lhs, const SemanticMemoryAccess &rhs)
{
    return lhs.region == rhs.region && lhs.index == rhs.index;
}

bool Overlaps(const SemanticMemoryAccess &lhs, const SemanticMemoryAccess &rhs)
{
    return SameStorage(lhs, rhs) && lhs.begin < rhs.end && rhs.begin < lhs.end;
}

bool IsWrite(SemanticAccessMode mode)
{
    return mode == SemanticAccessMode::Write;
}

bool IsDMA(CommandType type)
{
    return type == CommandType::DMA1D || type == CommandType::DMA2D ||
           type == CommandType::DMA3D || type == CommandType::DMASubmit1D ||
           type == CommandType::DMASubmit2D || type == CommandType::DMASubmit3D;
}

bool IsDMASubmit(CommandType type)
{
    return type == CommandType::DMASubmit1D || type == CommandType::DMASubmit2D ||
           type == CommandType::DMASubmit3D;
}

bool IsSystolic(CommandType type)
{
    return type == CommandType::Gemm32 || type == CommandType::Gemm32Accum ||
           type == CommandType::Gemm32Requant || type == CommandType::PointwiseC32 ||
           type == CommandType::DepthwiseC32 || type == CommandType::LineBufferJob ||
           type == CommandType::LineBufferSubmit || type == CommandType::LineBufferBinary ||
           type == CommandType::LineBufferBinarySubmit || type == CommandType::MaxPool ||
           type == CommandType::SystolicWait;
}

bool IsSystolicSubmit(CommandType type)
{
    return type == CommandType::LineBufferSubmit ||
           type == CommandType::LineBufferBinarySubmit;
}

uint16_t MinimumCommandSize(CommandType type)
{
    switch ( type )
    {
        case CommandType::End:
        case CommandType::Barrier:
        case CommandType::RollingReset:
        case CommandType::RollingProduce:
        case CommandType::RollingConsumeRelease:
        case CommandType::SpatzRequant:
        case CommandType::SpatzMul: return sizeof(CommandHeaderV2);
        case CommandType::RQLoad: return sizeof(CommandRQLoadV2);
        case CommandType::DMA1D:
        case CommandType::DMASubmit1D: return sizeof(CommandDMA1DV2);
        case CommandType::DMA2D:
        case CommandType::DMASubmit2D: return sizeof(CommandDMA2DV2);
        case CommandType::DMA3D:
        case CommandType::DMASubmit3D: return sizeof(CommandDMA3DV2);
        case CommandType::DMAWait: return sizeof(CommandDMAWaitV2);
        case CommandType::SystolicWait: return sizeof(CommandHeaderV2);
        case CommandType::Gemm32:
        case CommandType::Gemm32Accum:
        case CommandType::Gemm32Requant: return sizeof(CommandGemm32V2);
        case CommandType::PointwiseC32: return sizeof(CommandPointwiseC32V2);
        case CommandType::DepthwiseC32: return sizeof(CommandDepthwiseC32V2);
        case CommandType::AFULut: return sizeof(CommandAFULutV2);
        case CommandType::AFUBinary: return sizeof(CommandAFUBinaryV2);
        case CommandType::SpatzAdd: return sizeof(CommandSpatzAddV2);
        case CommandType::AFUGlobalAvgPool: return sizeof(CommandAFUGlobalAvgPoolV2);
        case CommandType::UpsampleNearest: return sizeof(CommandUpsampleNearestV2);
        case CommandType::MaxPool: return sizeof(CommandMaxPoolV2);
        case CommandType::LineBufferJob:
        case CommandType::LineBufferSubmit: return sizeof(CommandLineBufferJobV2);
        case CommandType::LineBufferBinary:
        case CommandType::LineBufferBinarySubmit: return sizeof(CommandLineBufferBinaryV2);
        case CommandType::CopyLayout: return sizeof(CommandCopyLayoutV2);
        case CommandType::AFUDFL16: return sizeof(CommandAFUDFL16V2);
        default: return 0;
    }
}

void AddResource(SemanticCommand &command, SemanticResource resource)
{
    command.resources |= ResourceMask(resource);
}

void AddState(SemanticCommand &command, SemanticState state, uint32_t instance,
    SemanticAccessMode mode, uint64_t identity = 0)
{
    command.stateAccesses.push_back({state, instance, mode, identity, 0});
}

void AddReadWriteState(SemanticCommand &command, SemanticState state,
    uint32_t instance, uint64_t identity = 0)
{
    AddState(command, state, instance, SemanticAccessMode::Read);
    AddState(command, state, instance, SemanticAccessMode::Write, identity);
}

bool AddAccess(SemanticCommand &command, const uint8_t *encoded,
    int referenceOffset, uint64_t bytes, SemanticAccessMode mode)
{
    const uint64_t begin = Read32(encoded + referenceOffset + 4);
    if ( bytes == 0 || begin > std::numeric_limits<uint32_t>::max() ||
         bytes > uint64_t(std::numeric_limits<uint32_t>::max()) + 1 - begin )
        return false;
    const uint16_t region = Read16(encoded + referenceOffset);
    SemanticMemoryAccess access{region, Read16(encoded + referenceOffset + 2),
        begin, begin + bytes, mode, UINT16_MAX, {}};
    if ( region == uint16_t(Region::TCDMScratch) )
        access.bankPhase = uint16_t((begin / TCDMWordBytes) % TCDMBanks);
    command.memoryAccesses.push_back(std::move(access));
    return true;
}

bool AddLocalAccess(SemanticCommand &command, uint64_t begin, uint64_t bytes,
    SemanticAccessMode mode)
{
    if ( bytes == 0 || begin > std::numeric_limits<uint32_t>::max() ||
         bytes > uint64_t(std::numeric_limits<uint32_t>::max()) + 1 - begin )
        return false;
    SemanticMemoryAccess access{uint16_t(Region::TCDMScratch), 0, begin,
        begin + bytes, mode, uint16_t((begin / TCDMWordBytes) % TCDMBanks), {}};
    command.memoryAccesses.push_back(std::move(access));
    return true;
}

uint64_t StridedSpan(uint64_t rows, uint64_t rowBytes,
    uint64_t rowStride, uint64_t tileColumns)
{
    if ( rows == 0 ) return 0;
    if ( rowStride == 0 || tileColumns == 0 ) return SaturatingMultiply(rows, rowBytes);
    return SaturatingAdd(
        SaturatingMultiply((rows - 1) / tileColumns, rowStride),
        SaturatingMultiply(std::min(rows, tileColumns), rowBytes));
}

bool DecodeDMA(SemanticCommand &command, const uint8_t *encoded)
{
    const uint64_t length = Read32(encoded + 32);
    uint64_t sourceSpan = length;
    uint64_t destinationSpan = length;
    uint64_t transfers = 1;
    int directionOffset = 36;
    if ( command.type == CommandType::DMA2D || command.type == CommandType::DMASubmit2D )
    {
        const uint64_t repetitions = Read32(encoded + 44);
        if ( repetitions == 0 ) return false;
        sourceSpan = SaturatingAdd(
            sourceSpan, SaturatingMultiply(Read32(encoded + 36), repetitions - 1));
        destinationSpan = SaturatingAdd(
            destinationSpan, SaturatingMultiply(Read32(encoded + 40), repetitions - 1));
        transfers = repetitions;
        directionOffset = 48;
    }
    else if ( command.type == CommandType::DMA3D || command.type == CommandType::DMASubmit3D )
    {
        const uint64_t repetitions2 = Read32(encoded + 44);
        const uint64_t repetitions3 = Read32(encoded + 56);
        if ( repetitions2 == 0 || repetitions3 == 0 ) return false;
        sourceSpan = SaturatingAdd(sourceSpan,
            SaturatingAdd(SaturatingMultiply(Read32(encoded + 36), repetitions2 - 1),
                SaturatingMultiply(Read32(encoded + 48), repetitions3 - 1)));
        destinationSpan = SaturatingAdd(destinationSpan,
            SaturatingAdd(SaturatingMultiply(Read32(encoded + 40), repetitions2 - 1),
                SaturatingMultiply(Read32(encoded + 52), repetitions3 - 1)));
        transfers = SaturatingMultiply(repetitions2, repetitions3);
        directionOffset = 60;
    }
    if ( length == 0 || sourceSpan > std::numeric_limits<uint32_t>::max() ||
         destinationSpan > std::numeric_limits<uint32_t>::max() )
        return false;
    if ( !AddAccess(command, encoded, 16, sourceSpan, SemanticAccessMode::Read) ||
         !AddAccess(command, encoded, 24, destinationSpan, SemanticAccessMode::Write) )
        return false;

    const uint32_t direction = Read32(encoded + directionOffset);
    command.queueId = direction;
    if ( direction == uint32_t(DMADirection::ExternalToLocal) )
        AddResource(command, SemanticResource::DMAExternalToLocal);
    else if ( direction == uint32_t(DMADirection::LocalToExternal) )
        AddResource(command, SemanticResource::DMALocalToExternal);
    else if ( direction == uint32_t(DMADirection::LocalToLocal) )
        AddResource(command, SemanticResource::Spatz);
    else
        return false;

    command.asyncCapable = direction <= uint32_t(DMADirection::LocalToExternal);
    command.asynchronous = IsDMASubmit(command.type);
    command.queueCapacity = command.asyncCapable ? DMAQueueDepth : 0;
    command.estimatedCycles = CycleCount(DivRoundUp(SaturatingMultiply(length, transfers), 32));
    if ( command.asyncCapable )
        AddReadWriteState(command, SemanticState::DMAQueue, direction);
    else
        AddReadWriteState(command, SemanticState::SpatzEngine, 0);
    return true;
}

bool DecodeGemm(SemanticCommand &command, const uint8_t *encoded)
{
    const uint64_t rows = Read32(encoded + 48);
    if ( rows == 0 ||
         !AddAccess(command, encoded, 16, 32 * 32, SemanticAccessMode::Read) ||
         !AddAccess(command, encoded, 24, SaturatingMultiply(rows, 32), SemanticAccessMode::Read) )
        return false;
    const uint64_t partialBytes = SaturatingMultiply(rows, 32 * 4);
    if ( command.type != CommandType::Gemm32 &&
         !AddAccess(command, encoded, 32, partialBytes, SemanticAccessMode::Read) )
        return false;
    if ( command.type == CommandType::Gemm32Requant )
    {
        if ( !AddAccess(command, encoded, 40, SaturatingMultiply(rows, 32),
                 SemanticAccessMode::Write) )
            return false;
    }
    else if ( !AddAccess(command, encoded, 32, partialBytes, SemanticAccessMode::Write) )
        return false;
    command.estimatedCycles = CycleCount(rows);
    return true;
}

bool DecodePointwise(SemanticCommand &command, const uint8_t *encoded)
{
    const uint64_t rows = Read32(encoded + 48);
    const uint64_t inputGroups = Read32(encoded + 52);
    const uint64_t outputGroups = Read32(encoded + 56);
    const uint64_t inputStride = Read32(encoded + 64);
    const uint64_t outputStride = Read32(encoded + 68);
    if ( rows == 0 || inputGroups == 0 || outputGroups == 0 ) return false;
    const uint64_t groups = SaturatingMultiply(inputGroups, outputGroups);
    const uint64_t weightBytes = SaturatingMultiply(groups, 32 * 32);
    const uint64_t inputBytes = SaturatingAdd(
        SaturatingMultiply(inputGroups - 1, inputStride), SaturatingMultiply(rows, 32));
    const uint64_t outputBytes = SaturatingAdd(
        SaturatingMultiply(outputGroups - 1, outputStride), SaturatingMultiply(rows, 32));
    if ( !AddAccess(command, encoded, 16, weightBytes, SemanticAccessMode::Read) ||
         !AddAccess(command, encoded, 24, inputBytes, SemanticAccessMode::Read) )
        return false;
    if ( inputGroups > 1 )
    {
        const uint64_t partialBytes = SaturatingMultiply(rows, 32 * 4);
        if ( !AddAccess(command, encoded, 32, partialBytes, SemanticAccessMode::Read) ||
             !AddAccess(command, encoded, 32, partialBytes, SemanticAccessMode::Write) )
            return false;
    }
    if ( !AddAccess(command, encoded, 40, outputBytes, SemanticAccessMode::Write) ) return false;
    command.estimatedCycles = CycleCount(SaturatingMultiply(rows, groups));
    return true;
}

bool DecodeLineBuffer(SemanticCommand &command, const uint8_t *encoded)
{
    constexpr int job = offsetof(CommandLineBufferJobV2, job);
    constexpr int cfg = job + offsetof(LinebufJobWireV1, linebuf);
    constexpr int gemm = job + offsetof(LinebufJobWireV1, gemm);
    const uint64_t inputRows = Read16(encoded + cfg + offsetof(SystolicLinebufCfg, inputH));
    const uint64_t inputStride = Read32(encoded + cfg + offsetof(SystolicLinebufCfg, rowStrideBytes));
    const uint64_t rows = Read32(encoded + gemm + offsetof(SystolicGemm32Req, dimM));
    const uint64_t accum = Read32(encoded + gemm + offsetof(SystolicGemm32Req, accumEn));
    const uint64_t outputStride = Read32(encoded + gemm + offsetof(SystolicGemm32Req, ofmRowStrideBytes));
    const uint64_t tileColumns = Read32(encoded + gemm + offsetof(SystolicGemm32Req, ofmTileCols));
    const uint64_t psumStride = Read32(encoded + gemm + offsetof(SystolicGemm32Req, psumRowStrideBytes));
    const uint64_t kTiles = Read32(encoded + job + offsetof(LinebufJobWireV1, kTiles));
    if ( inputRows == 0 || inputStride == 0 || rows == 0 || kTiles == 0 || accum > 3 ) return false;
    if ( !AddLocalAccess(command,
            Read32(encoded + cfg + offsetof(SystolicLinebufCfg, inputBase)),
            SaturatingMultiply(inputRows, inputStride), SemanticAccessMode::Read) ||
         !AddLocalAccess(command,
            Read32(encoded + gemm + offsetof(SystolicGemm32Req, weightAddr)),
            SaturatingMultiply(kTiles, 32 * 32), SemanticAccessMode::Read) )
        return false;
    const uint64_t psumBytes = StridedSpan(rows, 32 * 4, psumStride, tileColumns);
    const uint64_t outputBytes = StridedSpan(rows, 32, outputStride, tileColumns);
    const uint64_t psumAddress = Read32(encoded + gemm + offsetof(SystolicGemm32Req, psumAddr));
    const uint64_t outputAddress = Read32(encoded + gemm + offsetof(SystolicGemm32Req, ofmAddr));
    if ( (accum == 2 || accum == 3) &&
         !AddLocalAccess(command, psumAddress, psumBytes, SemanticAccessMode::Read) )
        return false;
    if ( command.type == CommandType::LineBufferBinary ||
         command.type == CommandType::LineBufferBinarySubmit )
    {
        constexpr int binary = offsetof(CommandLineBufferBinaryV2, binary);
        const uint64_t rhsStride = Read32(encoded + binary + offsetof(SystolicBinaryCfg, rhsRowStrideBytes));
        const uint64_t rhsColumns = Read32(encoded + binary + offsetof(SystolicBinaryCfg, rhsTileCols));
        if ( !AddLocalAccess(command,
                Read32(encoded + binary + offsetof(SystolicBinaryCfg, rhsAddr)),
                StridedSpan(rows, 32, rhsStride, rhsColumns), SemanticAccessMode::Read) )
            return false;
    }
    if ( accum == 1 || accum == 3 )
    {
        if ( !AddLocalAccess(command, psumAddress, psumBytes, SemanticAccessMode::Write) ) return false;
    }
    else if ( !AddLocalAccess(command, outputAddress, outputBytes, SemanticAccessMode::Write) )
        return false;
    command.asyncCapable = true;
    command.asynchronous = IsSystolicSubmit(command.type);
    command.queueCapacity = 1;
    command.queueId = 0;
    command.estimatedCycles = CycleCount(SaturatingMultiply(rows, kTiles));
    return true;
}

bool DecodeCopyLayout(SemanticCommand &command, const uint8_t *encoded)
{
    const uint64_t pixels = SaturatingMultiply(
        SaturatingMultiply(Read32(encoded + 40), Read32(encoded + 44)), Read32(encoded + 48));
    const uint64_t channels = Read32(encoded + 56);
    const uint64_t elementBytes = Read16(encoded + 38) == uint16_t(DataType::Int32) ? 4 : 1;
    const auto storageBytes = [&](uint16_t layout)
    {
        const uint64_t depth = layout == uint16_t(TensorLayout::NHWC) ? channels :
            DivRoundUp(channels, 32) * 32;
        return SaturatingMultiply(SaturatingMultiply(pixels, depth), elementBytes);
    };
    const uint64_t sourceBytes = storageBytes(Read16(encoded + 34));
    const uint64_t destinationBytes = storageBytes(Read16(encoded + 36));
    if ( !AddAccess(command, encoded, 16, sourceBytes, SemanticAccessMode::Read) ||
         !AddAccess(command, encoded, 24, destinationBytes, SemanticAccessMode::Write) )
        return false;
    command.estimatedCycles = CycleCount(DivRoundUp(std::max(sourceBytes, destinationBytes), 16));
    return true;
}

bool DecodeKnownCommand(SemanticCommand &command)
{
    const uint8_t *encoded = command.encoding.data();
    if ( IsDMA(command.type) ) return DecodeDMA(command, encoded);

    if ( IsSystolic(command.type) )
    {
        AddResource(command, SemanticResource::Systolic);
        if ( command.type == CommandType::SystolicWait )
        {
            AddReadWriteState(command, SemanticState::SystolicSlot, 0);
            return true;
        }
        AddState(command, SemanticState::Quantization, 0, SemanticAccessMode::Read);
        AddReadWriteState(command, SemanticState::SystolicSlot, 0);
    }

    switch ( command.type )
    {
        case CommandType::End:
        case CommandType::Barrier:
            AddResource(command, SemanticResource::Control);
            command.controlFence = true;
            return true;
        case CommandType::RollingReset:
        case CommandType::RollingProduce:
        case CommandType::RollingConsumeRelease:
            AddResource(command, SemanticResource::Control);
            command.controlFence = true;
            return true;
        case CommandType::SpatzRequant:
        case CommandType::SpatzMul:
            AddResource(command, SemanticResource::Spatz);
            AddReadWriteState(command, SemanticState::SpatzEngine, 0);
            // These reserved ABI values have no public payload contract yet.
            // Preserve their order conservatively if encountered.
            command.controlFence = true;
            return true;
        case CommandType::RQLoad:
        {
            const uint64_t identity = HashWords(encoded, 16, 3);
            AddState(command, SemanticState::Quantization, 0,
                SemanticAccessMode::Write, identity);
            command.estimatedCycles = CycleCount(Read32(encoded + 20));
            return Read32(encoded + 20) != 0;
        }
        case CommandType::DMAWait:
        {
            const uint32_t direction = Read32(encoded + 16);
            if ( direction > uint32_t(DMADirection::LocalToExternal) ) return false;
            command.queueId = direction;
            command.queueCapacity = DMAQueueDepth;
            AddReadWriteState(command, SemanticState::DMAQueue, direction);
            return true;
        }
        case CommandType::Gemm32:
        case CommandType::Gemm32Accum:
        case CommandType::Gemm32Requant: return DecodeGemm(command, encoded);
        case CommandType::PointwiseC32: return DecodePointwise(command, encoded);
        case CommandType::DepthwiseC32:
        {
            const uint64_t inputPixels = SaturatingMultiply(Read32(encoded + 40), Read32(encoded + 44));
            const uint64_t outputPixels = SaturatingMultiply(Read32(encoded + 48), Read32(encoded + 52));
            if ( !AddAccess(command, encoded, 16, 3 * 3 * 32, SemanticAccessMode::Read) ||
                 !AddAccess(command, encoded, 24, SaturatingMultiply(inputPixels, 32), SemanticAccessMode::Read) ||
                 !AddAccess(command, encoded, 32, SaturatingMultiply(outputPixels, 32), SemanticAccessMode::Write) )
                return false;
            command.estimatedCycles = CycleCount(SaturatingMultiply(outputPixels, 9));
            return true;
        }
        case CommandType::LineBufferJob:
        case CommandType::LineBufferSubmit:
        case CommandType::LineBufferBinary:
        case CommandType::LineBufferBinarySubmit: return DecodeLineBuffer(command, encoded);
        case CommandType::AFULut:
        {
            const uint64_t bytes = Read32(encoded + 40);
            AddResource(command, SemanticResource::AFU);
            AddReadWriteState(command, SemanticState::AFUEngine, 0);
            AddState(command, SemanticState::AFULut, 0, SemanticAccessMode::Write,
                HashWords(encoded, 32, 2));
            command.estimatedCycles = CycleCount(DivRoundUp(bytes, 4));
            return AddAccess(command, encoded, 16, bytes, SemanticAccessMode::Read) &&
                   AddAccess(command, encoded, 24, bytes, SemanticAccessMode::Write) &&
                   AddAccess(command, encoded, 32, 256, SemanticAccessMode::Read);
        }
        case CommandType::AFUBinary:
        case CommandType::SpatzAdd:
        {
            const uint64_t bytes = Read32(encoded + 40);
            const bool afu = command.type == CommandType::AFUBinary;
            AddResource(command, afu ? SemanticResource::AFU : SemanticResource::Spatz);
            AddReadWriteState(command, afu ? SemanticState::AFUEngine :
                SemanticState::SpatzEngine, 0);
            command.estimatedCycles = CycleCount(DivRoundUp(bytes, afu ? 32 : 16));
            return AddAccess(command, encoded, 16, bytes, SemanticAccessMode::Read) &&
                   AddAccess(command, encoded, 24, bytes, SemanticAccessMode::Read) &&
                   AddAccess(command, encoded, 32, bytes, SemanticAccessMode::Write);
        }
        case CommandType::AFUGlobalAvgPool:
        {
            const uint64_t pixels = SaturatingMultiply(Read32(encoded + 32), Read32(encoded + 36));
            const uint64_t channels = Read32(encoded + 40);
            AddResource(command, SemanticResource::AFU);
            AddReadWriteState(command, SemanticState::AFUEngine, 0);
            command.estimatedCycles = CycleCount(SaturatingMultiply(pixels, DivRoundUp(channels, 32)));
            return AddAccess(command, encoded, 16, SaturatingMultiply(pixels, channels),
                       SemanticAccessMode::Read) &&
                   AddAccess(command, encoded, 24, channels, SemanticAccessMode::Write);
        }
        case CommandType::CopyLayout:
            AddResource(command, SemanticResource::Spatz);
            AddReadWriteState(command, SemanticState::SpatzEngine, 0);
            return DecodeCopyLayout(command, encoded);
        case CommandType::MaxPool:
        {
            const uint64_t inputH = Read32(encoded + 32);
            const uint64_t inputW = Read32(encoded + 36);
            const uint64_t channels = Read32(encoded + 40);
            const uint64_t groups = DivRoundUp(channels, 32);
            const uint64_t kernelH = Read32(encoded + 44);
            const uint64_t kernelW = Read32(encoded + 48);
            const uint64_t strideH = Read32(encoded + 52);
            const uint64_t strideW = Read32(encoded + 56);
            const uint64_t padH = Read32(encoded + 60);
            const uint64_t padW = Read32(encoded + 64);
            const uint64_t paddedH = SaturatingAdd(inputH, SaturatingMultiply(2, padH));
            const uint64_t paddedW = SaturatingAdd(inputW, SaturatingMultiply(2, padW));
            if ( strideH == 0 || strideW == 0 || paddedH < kernelH ||
                 paddedW < kernelW )
                return false;
            const uint64_t outputH = (paddedH - kernelH) / strideH + 1;
            const uint64_t outputW = (paddedW - kernelW) / strideW + 1;
            const uint64_t groupBytes = SaturatingMultiply(groups, 32);
            const uint64_t inputBytes = SaturatingMultiply(
                SaturatingMultiply(inputH, inputW), groupBytes);
            const uint64_t outputPixels = SaturatingMultiply(outputH, outputW);
            const uint64_t outputBytes = SaturatingMultiply(outputPixels, groupBytes);
            command.estimatedCycles = CycleCount(SaturatingMultiply(
                SaturatingMultiply(outputPixels, groups), SaturatingMultiply(kernelH, kernelW)));
            return AddAccess(command, encoded, 16, inputBytes, SemanticAccessMode::Read) &&
                   AddAccess(command, encoded, 24, outputBytes, SemanticAccessMode::Write);
        }
        case CommandType::UpsampleNearest:
        {
            const uint64_t pixels = SaturatingMultiply(Read32(encoded + 32), Read32(encoded + 36));
            const uint64_t channels = Read32(encoded + 40);
            const uint64_t scale = SaturatingMultiply(Read32(encoded + 44), Read32(encoded + 48));
            const uint64_t inputBytes = SaturatingMultiply(pixels, channels);
            const uint64_t outputBytes = SaturatingMultiply(inputBytes, scale);
            AddResource(command, SemanticResource::Spatz);
            AddReadWriteState(command, SemanticState::SpatzEngine, 0);
            command.estimatedCycles = CycleCount(DivRoundUp(outputBytes, 32));
            return AddAccess(command, encoded, 16, inputBytes, SemanticAccessMode::Read) &&
                   AddAccess(command, encoded, 24, outputBytes, SemanticAccessMode::Write);
        }
        case CommandType::AFUDFL16:
        {
            const uint64_t locations = Read32(encoded + 56);
            AddResource(command, SemanticResource::AFU);
            AddResource(command, SemanticResource::Spatz);
            AddReadWriteState(command, SemanticState::AFUEngine, 0);
            AddReadWriteState(command, SemanticState::SpatzEngine, 0);
            AddState(command, SemanticState::AFULut, 0, SemanticAccessMode::Write,
                HashWords(encoded, 40, 4));
            command.estimatedCycles = CycleCount(SaturatingMultiply(locations, 16));
            return AddAccess(command, encoded, 16, SaturatingMultiply(locations, 64), SemanticAccessMode::Read) &&
                   AddAccess(command, encoded, 32, 4096, SemanticAccessMode::Write) &&
                   AddAccess(command, encoded, 40, 256 * 4, SemanticAccessMode::Read) &&
                   AddAccess(command, encoded, 48, 256 * 4, SemanticAccessMode::Read) &&
                   AddAccess(command, encoded, 24, SaturatingMultiply(locations, 4), SemanticAccessMode::Write);
        }
        default: return false;
    }
}

DependencyKind MemoryDependencyKind(
    SemanticAccessMode previous, SemanticAccessMode current)
{
    if ( IsWrite(previous) && !IsWrite(current) ) return DependencyKind::MemoryRAW;
    if ( !IsWrite(previous) && IsWrite(current) ) return DependencyKind::MemoryWAR;
    return DependencyKind::MemoryWAW;
}

struct StoredContent
{
    uint64_t begin = 0;
    uint64_t end = 0;
    uint64_t identity = 0;
    uint32_t generation = 0;
};

void AddContentSlice(std::vector<SemanticContentSlice> &content,
    uint64_t begin, uint64_t end, uint64_t identity, uint32_t generation)
{
    if ( begin == end ) return;
    if ( !content.empty() && content.back().end == begin &&
         content.back().identity == identity && content.back().generation == generation )
    {
        content.back().end = end;
        return;
    }
    content.push_back({begin, end, identity, generation});
}

void AnnotateRead(SemanticMemoryAccess &access,
    const std::vector<StoredContent> &stored)
{
    uint64_t cursor = access.begin;
    const uint64_t initial = InitialContentIdentity(access.region, access.index);
    for ( const StoredContent &span : stored )
    {
        if ( span.end <= cursor ) continue;
        if ( span.begin >= access.end ) break;
        if ( span.begin > cursor )
            AddContentSlice(access.content, cursor, std::min(span.begin, access.end), initial, 0);
        const uint64_t begin = std::max(cursor, span.begin);
        const uint64_t end = std::min(access.end, span.end);
        AddContentSlice(access.content, begin, end, span.identity, span.generation);
        cursor = end;
        if ( cursor == access.end ) break;
    }
    if ( cursor < access.end ) AddContentSlice(access.content, cursor, access.end, initial, 0);
}

void ReplaceContent(std::vector<StoredContent> &stored, const StoredContent &replacement)
{
    std::vector<StoredContent> updated;
    updated.reserve(stored.size() + 1);
    bool inserted = false;
    for ( const StoredContent &span : stored )
    {
        if ( span.end <= replacement.begin || span.begin >= replacement.end )
        {
            if ( !inserted && span.begin >= replacement.end )
            {
                updated.push_back(replacement);
                inserted = true;
            }
            updated.push_back(span);
            continue;
        }
        if ( span.begin < replacement.begin )
            updated.push_back({span.begin, replacement.begin, span.identity, span.generation});
        if ( !inserted )
        {
            updated.push_back(replacement);
            inserted = true;
        }
        if ( span.end > replacement.end )
            updated.push_back({replacement.end, span.end, span.identity, span.generation});
    }
    if ( !inserted ) updated.push_back(replacement);
    stored = std::move(updated);
}

void AnnotateContent(std::vector<SemanticCommand> &commands)
{
    std::unordered_map<uint64_t, std::vector<StoredContent>> contentByStorage;
    std::unordered_map<uint64_t, uint32_t> generationByStorage;
    for ( int commandIndex = 0; commandIndex < int(commands.size()); ++commandIndex )
    {
        SemanticCommand &command = commands[commandIndex];
        for ( int accessIndex = 0; accessIndex < int(command.memoryAccesses.size()); ++accessIndex )
        {
            SemanticMemoryAccess &access = command.memoryAccesses[accessIndex];
            auto &stored = contentByStorage[StorageKey(access.region, access.index)];
            if ( !IsWrite(access.mode) )
            {
                AnnotateRead(access, stored);
                continue;
            }
            const uint32_t generation = ++generationByStorage[StorageKey(access.region, access.index)];
            uint64_t identity = CommandContentIdentity(uint32_t(commandIndex), uint32_t(accessIndex));
            // A contiguous DMA copy preserves logical content identity. Strided
            // copies retain a fresh identity until the future tile IR represents
            // every repetition explicitly.
            if ( IsDMA(command.type) && accessIndex == 1 &&
                 command.memoryAccesses[0].end - command.memoryAccesses[0].begin ==
                     access.end - access.begin &&
                 command.memoryAccesses[0].content.size() == 1 )
                identity = command.memoryAccesses[0].content[0].identity;
            access.content.push_back({access.begin, access.end, identity, generation});
            ReplaceContent(stored, {access.begin, access.end, identity, generation});
        }
    }
}

void AnnotateState(std::vector<SemanticCommand> &commands)
{
    struct StateVersion
    {
        uint64_t identity = 0;
        uint32_t generation = 0;
    };
    std::unordered_map<uint64_t, StateVersion> versions;
    for ( int commandIndex = 0; commandIndex < int(commands.size()); ++commandIndex )
    {
        SemanticCommand &command = commands[commandIndex];
        for ( int accessIndex = 0; accessIndex < int(command.stateAccesses.size()); ++accessIndex )
        {
            SemanticStateAccess &access = command.stateAccesses[accessIndex];
            StateVersion &version = versions[StateKey(access.state, access.instance)];
            if ( IsWrite(access.mode) )
            {
                ++version.generation;
                if ( access.valueIdentity != 0 ) version.identity = access.valueIdentity;
                else version.identity = CommandContentIdentity(
                    uint32_t(commandIndex), uint32_t(accessIndex));
            }
            access.generation = version.generation;
            access.valueIdentity = version.identity;
        }
    }
}

}  // namespace

bool DecodeSemanticCommandStream(const std::vector<uint8_t> &bytes,
    std::vector<SemanticCommand> &commands, std::string &error)
{
    commands.clear();
    error.clear();
    int offset = 0;
    while ( offset < int(bytes.size()) )
    {
        if ( int(bytes.size()) - offset < int(sizeof(CommandHeaderV2)) )
        {
            error = "Neural-AI semantic IR encountered a truncated command header";
            commands.clear();
            return false;
        }
        const uint8_t *encoded = bytes.data() + offset;
        const uint16_t rawType = Read16(encoded);
        if ( rawType > uint16_t(CommandType::LineBufferBinarySubmit) )
        {
            error = "Neural-AI semantic IR encountered an unknown command type";
            commands.clear();
            return false;
        }
        const CommandType type = CommandType(rawType);
        const uint16_t size = Read16(encoded + 2);
        const uint16_t minimum = MinimumCommandSize(type);
        if ( minimum == 0 || size < minimum || (size & 15u) != 0 || offset + size > int(bytes.size()) )
        {
            error = "Neural-AI semantic IR encountered invalid size " + std::to_string(size) +
                " for command " + std::to_string(commands.size()) + " type " +
                std::to_string(rawType) + " (minimum " + std::to_string(minimum) + ")";
            commands.clear();
            return false;
        }
        SemanticCommand command;
        command.type = type;
        command.flags = Read32(encoded + 4);
        command.layerId = Read32(encoded + 8);
        command.tileId = Read32(encoded + 12);
        command.sourceCommandIndex = uint32_t(commands.size());
        command.encoding.assign(encoded, encoded + size);
        if ( !DecodeKnownCommand(command) )
        {
            error = "Neural-AI semantic IR could not decode command " +
                std::to_string(commands.size()) + " type " + std::to_string(rawType);
            commands.clear();
            return false;
        }
        commands.push_back(std::move(command));
        offset += size;
    }
    return true;
}

bool BuildCommandDependencyGraph(std::vector<SemanticCommand> &commands,
    CommandDependencyGraph &graph, std::string &error)
{
    struct StateHazards
    {
        int lastWriter = -1;
        std::vector<uint32_t> readers;
    };
    graph = {};
    error.clear();
    std::set<std::tuple<uint32_t, uint32_t, uint8_t>> uniqueEdges;
    std::unordered_map<uint64_t, StateHazards> stateHazards;
    int lastFence = -1;
    for ( int current = 0; current < int(commands.size()); ++current )
    {
        const SemanticCommand &command = commands[current];
        if ( lastFence >= 0 )
            uniqueEdges.emplace(uint32_t(lastFence), uint32_t(current),
                uint8_t(DependencyKind::Control));
        if ( command.controlFence )
        {
            for ( int previous = 0; previous < current; ++previous )
                uniqueEdges.emplace(uint32_t(previous), uint32_t(current),
                    uint8_t(DependencyKind::Control));
            lastFence = current;
        }
        for ( int previous = 0; previous < current; ++previous )
        {
            for ( const SemanticMemoryAccess &lhs : commands[previous].memoryAccesses )
            {
                for ( const SemanticMemoryAccess &rhs : command.memoryAccesses )
                {
                    if ( !Overlaps(lhs, rhs) || (!IsWrite(lhs.mode) && !IsWrite(rhs.mode)) ) continue;
                    uniqueEdges.emplace(uint32_t(previous), uint32_t(current),
                        uint8_t(MemoryDependencyKind(lhs.mode, rhs.mode)));
                }
            }
        }
        for ( const SemanticStateAccess &access : command.stateAccesses )
        {
            StateHazards &hazards = stateHazards[StateKey(access.state, access.instance)];
            if ( !IsWrite(access.mode) )
            {
                if ( hazards.lastWriter >= 0 && hazards.lastWriter != current )
                    uniqueEdges.emplace(uint32_t(hazards.lastWriter), uint32_t(current),
                        uint8_t(DependencyKind::StateRAW));
                if ( std::find(hazards.readers.begin(), hazards.readers.end(),
                         uint32_t(current)) == hazards.readers.end() )
                    hazards.readers.push_back(uint32_t(current));
                continue;
            }
            if ( hazards.lastWriter >= 0 && hazards.lastWriter != current )
                uniqueEdges.emplace(uint32_t(hazards.lastWriter), uint32_t(current),
                    uint8_t(DependencyKind::StateWAW));
            for ( uint32_t reader : hazards.readers )
            {
                if ( reader != uint32_t(current) )
                    uniqueEdges.emplace(reader, uint32_t(current),
                        uint8_t(DependencyKind::StateWAR));
            }
            hazards.readers.clear();
            hazards.lastWriter = current;
        }
    }

    graph.predecessors.resize(commands.size());
    graph.successors.resize(commands.size());
    graph.edges.reserve(uniqueEdges.size());
    for ( const auto &[predecessor, successor, rawKind] : uniqueEdges )
    {
        const auto kind = DependencyKind(rawKind);
        graph.edges.push_back({predecessor, successor, kind});
        graph.predecessors[successor].push_back(predecessor);
        graph.successors[predecessor].push_back(successor);
    }
    for ( auto &adjacent : graph.predecessors )
    {
        std::sort(adjacent.begin(), adjacent.end());
        adjacent.erase(std::unique(adjacent.begin(), adjacent.end()), adjacent.end());
    }
    for ( auto &adjacent : graph.successors )
    {
        std::sort(adjacent.begin(), adjacent.end());
        adjacent.erase(std::unique(adjacent.begin(), adjacent.end()), adjacent.end());
    }
    AnnotateContent(commands);
    AnnotateState(commands);
    return ValidateCommandDependencyGraph(commands, graph, error);
}

bool ValidateCommandDependencyGraph(const std::vector<SemanticCommand> &commands,
    const CommandDependencyGraph &graph, std::string &error)
{
    error.clear();
    if ( graph.predecessors.size() != commands.size() || graph.successors.size() != commands.size() )
    {
        error = "Neural-AI dependency graph adjacency size does not match its command stream";
        return false;
    }
    std::set<std::tuple<uint32_t, uint32_t, uint8_t>> unique;
    for ( const CommandDependency &edge : graph.edges )
    {
        if ( edge.predecessor >= commands.size() || edge.successor >= commands.size() ||
             edge.predecessor >= edge.successor ||
             !unique.emplace(edge.predecessor, edge.successor, uint8_t(edge.kind)).second )
        {
            error = "Neural-AI dependency graph contains an invalid or duplicate edge";
            return false;
        }
        if ( std::find(graph.predecessors[edge.successor].begin(),
                 graph.predecessors[edge.successor].end(), edge.predecessor) ==
                 graph.predecessors[edge.successor].end() ||
             std::find(graph.successors[edge.predecessor].begin(),
                 graph.successors[edge.predecessor].end(), edge.successor) ==
                 graph.successors[edge.predecessor].end() )
        {
            error = "Neural-AI dependency graph edge is missing from its adjacency lists";
            return false;
        }
    }
    for ( int commandIndex = 0; commandIndex < int(commands.size()); ++commandIndex )
    {
        const SemanticCommand &command = commands[commandIndex];
        if ( command.sourceCommandIndex != uint32_t(commandIndex) || command.encoding.size() < 16 ||
             Read16(command.encoding.data()) != uint16_t(command.type) ||
             Read16(command.encoding.data() + 2) != command.encoding.size() )
        {
            error = "Neural-AI semantic command metadata does not match its encoding";
            return false;
        }
        for ( const SemanticMemoryAccess &access : command.memoryAccesses )
        {
            if ( access.begin >= access.end || access.content.empty() ||
                 access.content.front().begin != access.begin || access.content.back().end != access.end )
            {
                error = "Neural-AI semantic memory access has incomplete content coverage";
                return false;
            }
            uint64_t cursor = access.begin;
            for ( const SemanticContentSlice &content : access.content )
            {
                if ( content.begin != cursor || content.begin >= content.end )
                {
                    error = "Neural-AI semantic content slices are not contiguous";
                    return false;
                }
                cursor = content.end;
            }
            if ( access.region == uint16_t(Region::TCDMScratch) )
            {
                if ( access.bankPhase != uint16_t((access.begin / TCDMWordBytes) % TCDMBanks) )
                {
                    error = "Neural-AI semantic TCDM bank phase is invalid";
                    return false;
                }
            }
            else if ( access.bankPhase != UINT16_MAX )
            {
                error = "Neural-AI non-TCDM access has a TCDM bank phase";
                return false;
            }
        }
    }
    return true;
}

std::vector<uint8_t> SerializeSemanticCommandStream(
    const std::vector<SemanticCommand> &commands)
{
    uint64_t total = 0;
    for ( const SemanticCommand &command : commands ) total += command.encoding.size();
    std::vector<uint8_t> result;
    if ( total <= std::numeric_limits<uint32_t>::max() ) result.reserve(uint32_t(total));
    for ( const SemanticCommand &command : commands )
        result.insert(result.end(), command.encoding.begin(), command.encoding.end());
    return result;
}

}  // namespace regor::neuralai
