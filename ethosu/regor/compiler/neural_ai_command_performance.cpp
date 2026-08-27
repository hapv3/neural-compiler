//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_command_performance.hpp"

#include "architecture/architecture.hpp"
#include "architecture/neuralai/neural_ai_abi.hpp"
#include "compiler/database.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace regor
{
namespace
{

using neuralai::CommandType;
using neuralai::RefV1;
using neuralai::Region;

uint16_t Read16(const uint8_t *data)
{
    return uint16_t(data[0]) | uint16_t(uint16_t(data[1]) << 8);
}

uint32_t Read32(const uint8_t *data)
{
    return uint32_t(data[0]) | (uint32_t(data[1]) << 8) | (uint32_t(data[2]) << 16) |
           (uint32_t(data[3]) << 24);
}

RefV1 ReadRef(const uint8_t *data)
{
    return {Read16(data), Read16(data + 2), Read32(data + 4)};
}

int64_t SaturatingMultiply(int64_t lhs, int64_t rhs)
{
    if ( lhs <= 0 || rhs <= 0 ) return 0;
    if ( lhs > std::numeric_limits<int64_t>::max() / rhs ) return std::numeric_limits<int64_t>::max();
    return lhs * rhs;
}

int64_t DivRoundUp64(int64_t value, int64_t divisor)
{
    return value <= 0 ? 0 : 1 + (value - 1) / divisor;
}

const char *CommandName(CommandType type)
{
    switch ( type )
    {
        case CommandType::End: return "END";
        case CommandType::Barrier: return "BARRIER";
        case CommandType::DMA1D: return "DMA1D";
        case CommandType::DMA2D: return "DMA2D";
        case CommandType::DMA3D: return "DMA3D";
        case CommandType::RQLoad: return "RQ_LOAD";
        case CommandType::Gemm32: return "GEMM32";
        case CommandType::Gemm32Accum: return "GEMM32_ACCUM";
        case CommandType::Gemm32Requant: return "GEMM32_REQUANT";
        case CommandType::LineBufferJob: return "LINE_BUFFER_JOB";
        case CommandType::PointwiseC32: return "POINTWISE_C32";
        case CommandType::DepthwiseC32: return "DEPTHWISE_C32";
        case CommandType::AFULut: return "AFU_LUT";
        case CommandType::AFUBinary: return "AFU_BINARY";
        case CommandType::AFUGlobalAvgPool: return "AFU_GLOBAL_AVG_POOL";
        case CommandType::SpatzRequant: return "SPATZ_REQUANT";
        case CommandType::SpatzAdd: return "SPATZ_ADD";
        case CommandType::SpatzMul: return "SPATZ_MUL";
        case CommandType::CopyLayout: return "COPY_LAYOUT";
        case CommandType::MaxPool: return "MAX_POOL";
        case CommandType::UpsampleNearest: return "UPSAMPLE_NEAREST";
        case CommandType::RollingReset: return "ROLLING_RESET";
        case CommandType::RollingProduce: return "ROLLING_PRODUCE";
        case CommandType::RollingConsumeRelease: return "ROLLING_CONSUME_RELEASE";
        case CommandType::DMASubmit1D: return "DMA_SUBMIT_1D";
        case CommandType::DMASubmit2D: return "DMA_SUBMIT_2D";
        case CommandType::DMASubmit3D: return "DMA_SUBMIT_3D";
        case CommandType::DMAWait: return "DMA_WAIT";
        case CommandType::AFUDFL16: return "AFU_DFL16";
        default: return "UNKNOWN";
    }
}

ArchitectureMemory *MemoryForRef(Architecture *architecture, const RefV1 &ref)
{
    switch ( Region(ref.region) )
    {
        case Region::ModelConstants: return architecture->ReadonlyMemory().memory;
        case Region::InputBinding:
        case Region::OutputBinding:
        case Region::L2TemporaryBinding: return architecture->FeatureMapMemory().memory;
        case Region::TCDMScratch: return architecture->StagingMemory().memory;
        default: return nullptr;
    }
}

int64_t TransferCycles(const ArchitectureMemory *memory, int64_t bytes, bool write)
{
    if ( memory == nullptr || bytes <= 0 ) return 0;
    const float bandwidth = std::max(memory->BurstBandwidth(write), 0.0001f);
    return int64_t(std::ceil(double(bytes) / bandwidth)) +
           (write ? memory->WriteLatency() : memory->ReadLatency());
}

class CommandMeasurement
{
public:
    CommandMeasurement(Architecture *arch, PerformanceResult &perf) :
            architecture(arch), performance(perf)
    {
    }

    Architecture *architecture;
    PerformanceResult &performance;
    NeuralAICommandPerformance row;

    void AddAccess(const RefV1 &ref, int64_t bytes, bool write, AccessType accessType)
    {
        if ( bytes <= 0 ) return;
        ArchitectureMemory *memory = MemoryForRef(architecture, ref);
        if ( memory == nullptr ) return;
        auto &memoryAccesses = performance.memory[memory];
        auto &access = memoryAccesses.access[accessType];
        if ( write ) access.bytesWritten += bytes;
        else access.bytesRead += bytes;
        const int64_t cycles = Region(ref.region) == Region::TCDMScratch ?
            DivRoundUp64(bytes, std::max<int64_t>(1, int64_t(memory->Bandwidth()))) :
            TransferCycles(memory, bytes, write);
        access.accessCycles += cycles;
        memoryAccesses.accessCycles += cycles;

        switch ( Region(ref.region) )
        {
            case Region::ModelConstants:
                if ( !write ) row.modelReadBytes += bytes;
                break;
            case Region::InputBinding:
            case Region::OutputBinding:
            case Region::L2TemporaryBinding:
                if ( write ) row.l2WriteBytes += bytes;
                else row.l2ReadBytes += bytes;
                break;
            case Region::TCDMScratch:
                if ( write ) row.tcdmWriteBytes += bytes;
                else row.tcdmReadBytes += bytes;
                break;
            default: break;
        }
    }

    int64_t MemoryCycles() const
    {
        const auto *model = architecture->ReadonlyMemory().memory;
        const auto *l2 = architecture->FeatureMapMemory().memory;
        const auto *tcdm = architecture->StagingMemory().memory;
        const int64_t dramRead = row.modelReadBytes + row.l2ReadBytes;
        const int64_t dramCycles = std::max(
            TransferCycles(model, dramRead, false), TransferCycles(l2, row.l2WriteBytes, true));
        const int64_t tcdmRaw = DivRoundUp64(row.tcdmReadBytes + row.tcdmWriteBytes,
            std::max<int64_t>(1, int64_t(tcdm->Bandwidth())));
        return std::max(dramCycles, tcdmRaw);
    }
};

void AddDMA(CommandMeasurement &measurement, const uint8_t *command, int64_t bytes)
{
    const RefV1 source = ReadRef(command + 16);
    const RefV1 destination = ReadRef(command + 24);
    measurement.AddAccess(source, bytes, false,
        Region(source.region) == Region::ModelConstants ? AccessType::Weights : AccessType::FeatureMap);
    measurement.AddAccess(destination, bytes, true, AccessType::FeatureMap);
}

void AddTCDM(CommandMeasurement &measurement, int64_t readBytes, int64_t writeBytes,
    AccessType readType = AccessType::FeatureMap)
{
    const RefV1 tcdm{uint16_t(Region::TCDMScratch), 0, 0};
    measurement.AddAccess(tcdm, readBytes, false, readType);
    measurement.AddAccess(tcdm, writeBytes, true, AccessType::FeatureMap);
}

}  // namespace

NeuralAICommandPerformanceResult MeasureNeuralAICommandPerformance(
    const CompiledNeuralAIArtifact &artifact, Architecture *architecture, Database *db)
{
    NeuralAICommandPerformanceResult result;
    if ( architecture == nullptr ) return result;

    int table = 0;
    if ( db != nullptr )
    {
        table = db->AddTable("nai_command_perf", false);
        db->AddColumns(table, {"command_index", "command", "type", "layer_id", "tile_id",
            "compute_cycles", "memory_cycles", "cycles", "model_read_bytes", "l2_read_bytes",
            "l2_write_bytes", "tcdm_read_bytes", "tcdm_write_bytes"});
    }

    int offset = 0;
    int index = 0;
    std::array<int64_t, 2> pendingDMACycles = {-1, -1};
    std::array<int64_t, 2> hiddenDMACycles = {0, 0};
    while ( offset + int(sizeof(neuralai::CommandHeaderV2)) <= int(artifact.commands.size()) )
    {
        const uint8_t *command = artifact.commands.data() + offset;
        const CommandType type = CommandType(Read16(command));
        const uint16_t size = Read16(command + 2);
        if ( size < sizeof(neuralai::CommandHeaderV2) || offset + size > int(artifact.commands.size()) ) break;

        CommandMeasurement measurement{architecture, result.performance};
        measurement.row.index = index;
        measurement.row.type = uint16_t(type);
        measurement.row.layerId = Read32(command + 8);
        measurement.row.tileId = Read32(command + 12);
        measurement.row.name = CommandName(type);
        int asyncDirection = -1;
        const bool dmaSubmit = type == CommandType::DMASubmit1D ||
            type == CommandType::DMASubmit2D || type == CommandType::DMASubmit3D;
        const bool dmaWait = type == CommandType::DMAWait;

        switch ( type )
        {
            case CommandType::DMA1D:
            case CommandType::DMASubmit1D:
                if ( size >= sizeof(neuralai::CommandDMA1DV2) ) AddDMA(measurement, command, Read32(command + 32));
                if ( dmaSubmit ) asyncDirection = int(Read32(command + 36));
                break;
            case CommandType::DMA2D:
            case CommandType::DMASubmit2D:
                if ( size >= sizeof(neuralai::CommandDMA2DV2) )
                    AddDMA(measurement, command, SaturatingMultiply(Read32(command + 32), Read32(command + 44)));
                if ( dmaSubmit ) asyncDirection = int(Read32(command + 48));
                break;
            case CommandType::DMA3D:
            case CommandType::DMASubmit3D:
                if ( size >= sizeof(neuralai::CommandDMA3DV2) )
                    AddDMA(measurement, command, SaturatingMultiply(
                        SaturatingMultiply(Read32(command + 32), Read32(command + 44)), Read32(command + 56)));
                if ( dmaSubmit ) asyncDirection = int(Read32(command + 60));
                break;
            case CommandType::DMAWait:
                if ( size >= sizeof(neuralai::CommandDMAWaitV2) ) asyncDirection = int(Read32(command + 16));
                break;
            case CommandType::RQLoad:
                if ( size >= sizeof(neuralai::CommandRQLoadV2) )
                {
                    const int64_t bytes = SaturatingMultiply(Read32(command + 20), sizeof(neuralai::QParamV1));
                    const RefV1 model{uint16_t(Region::ModelConstants), 0, 0};
                    measurement.AddAccess(model, bytes, false, AccessType::Scales);
                }
                break;
            case CommandType::Gemm32:
            case CommandType::Gemm32Accum:
            case CommandType::Gemm32Requant:
                if ( size >= sizeof(neuralai::CommandGemm32V2) )
                {
                    const int64_t rows = Read32(command + 48);
                    const int64_t ifmBytes = rows * 32;
                    const int64_t partialBytes = rows * 32 * 4;
                    int64_t writes = 0;
                    if ( type == CommandType::Gemm32 ) writes = partialBytes;
                    else if ( type == CommandType::Gemm32Accum ) writes = partialBytes;
                    else writes = ifmBytes;
                    measurement.AddAccess(ReadRef(command + 16), 32 * 32, false, AccessType::Weights);
                    measurement.AddAccess(ReadRef(command + 24), ifmBytes, false, AccessType::FeatureMap);
                    if ( type != CommandType::Gemm32 )
                        measurement.AddAccess(ReadRef(command + 32), partialBytes, false, AccessType::FeatureMap);
                    measurement.AddAccess(type == CommandType::Gemm32Requant ? ReadRef(command + 40) :
                        ReadRef(command + 32), writes, true, AccessType::FeatureMap);
                    measurement.row.computeCycles = std::max<int64_t>(1, rows);
                }
                break;
            case CommandType::PointwiseC32:
                if ( size >= sizeof(neuralai::CommandPointwiseC32V2) )
                {
                    const int64_t rows = Read32(command + 48);
                    const int64_t inputGroups = Read32(command + 52);
                    const int64_t outputGroups = Read32(command + 56);
                    const int64_t groups = SaturatingMultiply(inputGroups, outputGroups);
                    const int64_t weightBytes = SaturatingMultiply(groups, 32 * 32);
                    const int64_t ifmBytes = rows * groups * 32;
                    const int64_t partialBytes = rows * outputGroups * 32 * 4;
                    const int64_t outputBytes = rows * outputGroups * 32;
                    measurement.AddAccess(ReadRef(command + 16), weightBytes, false, AccessType::Weights);
                    measurement.AddAccess(ReadRef(command + 24), ifmBytes, false, AccessType::FeatureMap);
                    if ( inputGroups > 1 )
                    {
                        measurement.AddAccess(ReadRef(command + 32), partialBytes, false, AccessType::FeatureMap);
                        measurement.AddAccess(ReadRef(command + 32), partialBytes, true, AccessType::FeatureMap);
                    }
                    measurement.AddAccess(ReadRef(command + 40), outputBytes, true, AccessType::FeatureMap);
                    measurement.row.computeCycles = std::max<int64_t>(1, SaturatingMultiply(rows, groups));
                }
                break;
            case CommandType::LineBufferJob:
                if ( size >= sizeof(neuralai::CommandLineBufferJobV2) )
                {
                    constexpr size_t jobOffset = offsetof(neuralai::CommandLineBufferJobV2, job);
                    constexpr size_t rowsOffset = jobOffset + offsetof(neuralai::LinebufJobWireV1, rows);
                    constexpr size_t kTilesOffset = jobOffset + offsetof(neuralai::LinebufJobWireV1, kTiles);
                    const int64_t rows = Read32(command + rowsOffset);
                    const int64_t kTiles = Read32(command + kTilesOffset);
                    const int64_t work = SaturatingMultiply(rows, kTiles);
                    AddTCDM(measurement, kTiles * 32 * 32 + work * 32, rows * 32);
                    measurement.row.computeCycles = std::max<int64_t>(1, work);
                }
                break;
            case CommandType::DepthwiseC32:
                if ( size >= sizeof(neuralai::CommandDepthwiseC32V2) )
                {
                    const int64_t inputPixels = SaturatingMultiply(Read32(command + 40), Read32(command + 44));
                    const int64_t outputPixels = SaturatingMultiply(Read32(command + 48), Read32(command + 52));
                    constexpr int64_t kernel = 3 * 3;
                    measurement.AddAccess(ReadRef(command + 16), kernel * 32, false, AccessType::Weights);
                    measurement.AddAccess(ReadRef(command + 24), inputPixels * 32, false, AccessType::FeatureMap);
                    measurement.AddAccess(ReadRef(command + 32), outputPixels * 32, true, AccessType::FeatureMap);
                    measurement.row.computeCycles = std::max<int64_t>(1, outputPixels * kernel);
                }
                break;
            case CommandType::AFULut:
                if ( size >= sizeof(neuralai::CommandAFULutV2) )
                {
                    const int64_t length = Read32(command + 40);
                    measurement.AddAccess(ReadRef(command + 16), length, false, AccessType::FeatureMap);
                    measurement.AddAccess(ReadRef(command + 24), length, true, AccessType::FeatureMap);
                    measurement.AddAccess(ReadRef(command + 32), 256, false, AccessType::Lut);
                    measurement.row.computeCycles = std::max<int64_t>(1, DivRoundUp64(length, 32));
                }
                break;
            case CommandType::AFUBinary:
            case CommandType::SpatzAdd:
                if ( (type == CommandType::AFUBinary && size >= sizeof(neuralai::CommandAFUBinaryV2)) ||
                     (type == CommandType::SpatzAdd && size >= sizeof(neuralai::CommandSpatzAddV2)) )
                {
                    const int64_t length = Read32(command + 40);
                    measurement.AddAccess(ReadRef(command + 16), length, false, AccessType::FeatureMap);
                    measurement.AddAccess(ReadRef(command + 24), length, false, AccessType::FeatureMap);
                    measurement.AddAccess(ReadRef(command + 32), length, true, AccessType::FeatureMap);
                    const int throughput = type == CommandType::AFUBinary ? 32 : 16;
                    measurement.row.computeCycles = std::max<int64_t>(1, DivRoundUp64(length, throughput));
                }
                break;
            case CommandType::AFUGlobalAvgPool:
                if ( size >= sizeof(neuralai::CommandAFUGlobalAvgPoolV2) )
                {
                    const int64_t pixels = SaturatingMultiply(Read32(command + 32), Read32(command + 36));
                    const int64_t channels = Read32(command + 40);
                    const int64_t bytes = SaturatingMultiply(pixels, channels);
                    measurement.AddAccess(ReadRef(command + 16), bytes, false, AccessType::FeatureMap);
                    measurement.AddAccess(ReadRef(command + 24), channels, true, AccessType::FeatureMap);
                    measurement.row.computeCycles = std::max<int64_t>(1, pixels * DivRoundUp64(channels, 32));
                }
                break;
            case CommandType::CopyLayout:
                if ( size >= sizeof(neuralai::CommandCopyLayoutV2) )
                {
                    const int64_t pixels = SaturatingMultiply(SaturatingMultiply(Read32(command + 40),
                        Read32(command + 44)), Read32(command + 48));
                    const int64_t channels = Read32(command + 56);
                    const int64_t elementBytes = Read16(command + 38) == uint16_t(neuralai::DataType::Int32) ? 4 : 1;
                    const auto storageBytes = [&](uint16_t layout)
                    {
                        const int64_t depth = layout == uint16_t(neuralai::TensorLayout::NHWC) ? channels :
                            DivRoundUp64(channels, 32) * 32;
                        return SaturatingMultiply(SaturatingMultiply(pixels, depth), elementBytes);
                    };
                    const int64_t sourceBytes = storageBytes(Read16(command + 34));
                    const int64_t destinationBytes = storageBytes(Read16(command + 36));
                    measurement.AddAccess(ReadRef(command + 16), sourceBytes, false, AccessType::FeatureMap);
                    measurement.AddAccess(ReadRef(command + 24), destinationBytes, true, AccessType::FeatureMap);
                    measurement.row.computeCycles = std::max<int64_t>(1,
                        DivRoundUp64(std::max(sourceBytes, destinationBytes), 16));
                }
                break;
            case CommandType::MaxPool:
                if ( size >= sizeof(neuralai::CommandMaxPoolV2) )
                {
                    const int64_t pixels = SaturatingMultiply(Read32(command + 32), Read32(command + 36));
                    const int64_t groups = DivRoundUp64(Read32(command + 40), 32);
                    const int64_t kernel = SaturatingMultiply(Read32(command + 44), Read32(command + 48));
                    const int64_t bytes = pixels * groups * 32;
                    measurement.AddAccess(ReadRef(command + 16), bytes, false, AccessType::FeatureMap);
                    measurement.AddAccess(ReadRef(command + 24), bytes, true, AccessType::FeatureMap);
                    measurement.row.computeCycles = std::max<int64_t>(1, pixels * groups * kernel);
                }
                break;
            case CommandType::UpsampleNearest:
                if ( size >= sizeof(neuralai::CommandUpsampleNearestV2) )
                {
                    const int64_t inputPixels = SaturatingMultiply(Read32(command + 32), Read32(command + 36));
                    const int64_t channels = Read32(command + 40);
                    const int64_t scale = SaturatingMultiply(Read32(command + 44), Read32(command + 48));
                    const int64_t inputBytes = SaturatingMultiply(inputPixels, channels);
                    const int64_t outputBytes = SaturatingMultiply(inputBytes, scale);
                    measurement.AddAccess(ReadRef(command + 16), inputBytes, false, AccessType::FeatureMap);
                    measurement.AddAccess(ReadRef(command + 24), outputBytes, true, AccessType::FeatureMap);
                    measurement.row.computeCycles = std::max<int64_t>(1, DivRoundUp64(outputBytes, 32));
                }
                break;
            case CommandType::AFUDFL16:
                if ( size >= sizeof(neuralai::CommandAFUDFL16V2) )
                {
                    const int64_t locations = Read32(command + 56);
                    measurement.AddAccess(ReadRef(command + 16), locations * 64, false, AccessType::FeatureMap);
                    measurement.AddAccess(ReadRef(command + 24), locations * 4, true, AccessType::FeatureMap);
                    measurement.AddAccess(ReadRef(command + 40), 256 * 4, false, AccessType::Lut);
                    measurement.AddAccess(ReadRef(command + 48), 256 * 4, false, AccessType::Lut);
                    measurement.row.computeCycles = std::max<int64_t>(1, locations * 16);
                }
                break;
            default: break;
        }

        measurement.row.memoryCycles = measurement.MemoryCycles();
        measurement.row.totalCycles = std::max<int64_t>(1,
            std::max(measurement.row.computeCycles, measurement.row.memoryCycles));
        if ( dmaSubmit && asyncDirection >= 0 && asyncDirection < int(pendingDMACycles.size()) )
        {
            pendingDMACycles[asyncDirection] = measurement.row.memoryCycles;
            hiddenDMACycles[asyncDirection] = 0;
            measurement.row.totalCycles = 1;
        }
        else if ( dmaWait && asyncDirection >= 0 && asyncDirection < int(pendingDMACycles.size()) )
        {
            const int64_t remaining = pendingDMACycles[asyncDirection] < 0 ? 0 :
                std::max<int64_t>(0, pendingDMACycles[asyncDirection] - hiddenDMACycles[asyncDirection]);
            measurement.row.memoryCycles = remaining;
            measurement.row.totalCycles = std::max<int64_t>(1, remaining);
            pendingDMACycles[asyncDirection] = -1;
            hiddenDMACycles[asyncDirection] = 0;
        }
        for ( int direction = 0; direction < int(pendingDMACycles.size()); ++direction )
        {
            if ( pendingDMACycles[direction] < 0 ||
                 (dmaSubmit && direction == asyncDirection) ) continue;
            hiddenDMACycles[direction] += measurement.row.totalCycles;
        }
        result.performance.npuCycles += measurement.row.totalCycles;
        result.performance.totalCycles += measurement.row.totalCycles;
        result.commands.push_back(measurement.row);

        if ( db != nullptr )
        {
            const auto &row = measurement.row;
            db->AddRow(table, index, {std::to_string(row.index), row.name, std::to_string(row.type),
                std::to_string(row.layerId), std::to_string(row.tileId), std::to_string(row.computeCycles),
                std::to_string(row.memoryCycles), std::to_string(row.totalCycles),
                std::to_string(row.modelReadBytes), std::to_string(row.l2ReadBytes),
                std::to_string(row.l2WriteBytes), std::to_string(row.tcdmReadBytes),
                std::to_string(row.tcdmWriteBytes)});
        }

        offset += size;
        ++index;
        if ( type == CommandType::End ) break;
    }
    return result;
}

}  // namespace regor
