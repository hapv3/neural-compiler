//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_performance.hpp"

#include "common/common.hpp"

#include <algorithm>
#include <cmath>

namespace regor
{
namespace
{

int64_t TransferCycles(const ArchitectureMemory *memory, int64_t bytes, bool write)
{
    if ( memory == nullptr || bytes <= 0 ) return 0;
    const int64_t transfer = int64_t(std::ceil(double(bytes) / memory->BurstBandwidth(write)));
    return transfer + (write ? memory->WriteLatency() : memory->ReadLatency());
}

int64_t ElementBytes(int64_t elements, DataType type)
{
    return (elements * DataTypeSizeBits(type) + 7) / 8;
}

struct DomainTransfers
{
    int64_t fmRead = 0;
    int64_t fmWrite = 0;
    int64_t weightRead = 0;
    int64_t scaleRead = 0;
};

int64_t GroupCycles(const ArchitectureMemory *memory, int64_t readBytes, int64_t writeBytes)
{
    const int64_t rawCycles = int64_t(std::ceil(double(readBytes + writeBytes) / memory->Bandwidth()));
    const int64_t readCycles = int64_t(std::ceil(double(readBytes) / memory->BurstBandwidth(false)));
    const int64_t writeCycles = int64_t(std::ceil(double(writeBytes) / memory->BurstBandwidth(true)));
    return std::max(rawCycles, std::max(readCycles, writeCycles));
}

}  // namespace

CycleCost NeuralAIPerformance::MeasureCycleCost(const PerformanceQuery &query)
{
    CycleCost cost;
    if ( query.type == OpType::MemoryCopy ) return cost;

    const int64_t depthK = query.ifm[0].shape.Depth();
    if ( query.type == OpType::AvgPool )
    {
        const int64_t spatial = depthK > 0 ? query.ifm[0].shape.Elements64() / depthK : 0;
        const int64_t groups = depthK > 0 ? (depthK + 31) / 32 : 0;
        cost.opCycles = std::max<int64_t>(1, spatial * groups);
        return cost;
    }
    const int64_t depthN = query.ofm.shape.Depth();
    const int64_t rows = depthN > 0 ? query.ofm.shape.Elements64() / depthN : 0;
    cost.macs = rows * depthK * depthN;
    cost.opCycles = std::max<int64_t>(1, (cost.macs + 32 * 32 - 1) / (32 * 32));
    return cost;
}

int64_t NeuralAIPerformance::MemToMemCycles(
    const ArchitectureMemory *dest, const ArchitectureMemory *source, int sizeBytes)
{
    return std::max(TransferCycles(source, sizeBytes, false), TransferCycles(dest, sizeBytes, true));
}

ElementAccess NeuralAIPerformance::MeasureElementAccess(const PerformanceQuery &query)
{
    ElementAccess access;
    access.ifmRead[0] = query.ifm[0].shape.Elements64();
    if ( query.ifm[1].shape ) access.ifmRead[1] = query.ifm[1].shape.Elements64();
    access.ofmWrite = query.ofm.shape.Elements64();
    if ( query.type != OpType::MemoryCopy )
    {
        access.constRead[0] = query.encodedWeightSize;
        access.constRead[1] = query.encodedScaleSize;
        access.weightsRefetch = 1;
    }
    return access;
}

ElementAccess NeuralAIPerformance::ElementTransferToBytes(
    const PerformanceQuery &query, const ElementAccess &access)
{
    ElementAccess bytes = access;
    bytes.ifmRead[0] = ElementBytes(access.ifmRead[0], query.ifm[0].type);
    if ( query.ifm[1].shape ) bytes.ifmRead[1] = ElementBytes(access.ifmRead[1], query.ifm[1].type);
    bytes.ofmWrite = ElementBytes(access.ofmWrite, query.ofm.type);
    bytes.constRead[0] = query.encodedWeightSize;
    bytes.constRead[1] = query.encodedScaleSize;
    return bytes;
}

int64_t NeuralAIPerformance::WeightDecodeCycles(const PerformanceQuery &, const WeightStats &weights,
    Flags<WeightFormat>, ArchitectureMemory *weightsMemory)
{
    return TransferCycles(weightsMemory, int64_t(weights.encodedSize), false);
}

void NeuralAIPerformance::InitDatabase(Database *)
{
}

void NeuralAIPerformance::RecordToDB(int)
{
}

int64_t NeuralAIPerformance::MinReadCycles(
    const ArchitectureMemory *mem, int64_t size, TensorUsage, OpType, bool)
{
    return TransferCycles(mem, size, false);
}

int64_t NeuralAIPerformance::MinWriteCycles(const ArchitectureMemory *mem, int64_t size)
{
    return TransferCycles(mem, size, true);
}

std::unordered_map<const ArchitectureMemory *, AccessCycles>
NeuralAIPerformance::MeasureAccessCycles(const PerformanceQuery &query, const ElementAccess &byteAccess)
{
    const auto domain = [this](const ArchitectureMemory *memory)
    {
        return memory == _modelMemory ? _l2Memory : memory;
    };
    std::unordered_map<const ArchitectureMemory *, DomainTransfers> transfers;
    std::unordered_map<const ArchitectureMemory *, AccessCycles> result;
    transfers[domain(query.ifm[0].memory)].fmRead += byteAccess.ifmRead[0];
    if ( query.ifm[1].shape ) transfers[domain(query.ifm[1].memory)].fmRead += byteAccess.ifmRead[1];
    transfers[domain(query.ofm.memory)].fmWrite += byteAccess.ofmWrite;
    if ( query.constMemory )
    {
        transfers[domain(query.constMemory)].weightRead += byteAccess.constRead[0];
        transfers[domain(query.constMemory)].scaleRead += byteAccess.constRead[1];
    }
    if ( query.tmpMemory )
    {
        transfers[domain(query.tmpMemory)].fmRead += byteAccess.tmpRead;
        transfers[domain(query.tmpMemory)].fmWrite += byteAccess.tmpWrite;
    }
    for ( const auto &[memory, transfer] : transfers )
    {
        if ( memory == nullptr ) continue;
        AccessCycles &cycles = result[memory];
        cycles.fmAccessCycles = GroupCycles(memory, transfer.fmRead, transfer.fmWrite);
        cycles.weightsAccessCycles = GroupCycles(memory, transfer.weightRead, 0);
        cycles.scalesAccessCycles = GroupCycles(memory, transfer.scaleRead, 0);
        cycles.totalAccessCycles = GroupCycles(memory,
            transfer.fmRead + transfer.weightRead + transfer.scaleRead, transfer.fmWrite);
    }
    return result;
}

}  // namespace regor
