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
    const int64_t transfer = int64_t(std::ceil(double(bytes) / memory->Bandwidth()));
    return transfer + (write ? memory->WriteLatency() : memory->ReadLatency());
}

int64_t ElementBytes(int64_t elements, DataType type)
{
    return (elements * DataTypeSizeBits(type) + 7) / 8;
}

void AddTransfer(std::unordered_map<const ArchitectureMemory *, AccessCycles> &result,
    const ArchitectureMemory *memory, int64_t fmBytes, int64_t weightBytes, int64_t scaleBytes)
{
    if ( memory == nullptr ) return;
    auto &cycles = result[memory];
    cycles.fmAccessCycles += TransferCycles(memory, fmBytes, false);
    cycles.weightsAccessCycles += TransferCycles(memory, weightBytes, false);
    cycles.scalesAccessCycles += TransferCycles(memory, scaleBytes, false);
    cycles.totalAccessCycles += TransferCycles(memory, fmBytes + weightBytes + scaleBytes, false);
}

}  // namespace

CycleCost NeuralAIPerformance::MeasureCycleCost(const PerformanceQuery &query)
{
    CycleCost cost;
    if ( query.type == OpType::MemoryCopy ) return cost;

    const int64_t depthK = query.ifm[0].shape.Depth();
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
    std::unordered_map<const ArchitectureMemory *, AccessCycles> result;
    AddTransfer(result, query.ifm[0].memory, byteAccess.ifmRead[0], 0, 0);
    if ( query.ifm[1].shape ) AddTransfer(result, query.ifm[1].memory, byteAccess.ifmRead[1], 0, 0);
    AddTransfer(result, query.ofm.memory, byteAccess.ofmWrite, 0, 0);
    AddTransfer(result, query.constMemory, 0, byteAccess.constRead[0], byteAccess.constRead[1]);
    if ( query.tmpMemory )
    {
        AddTransfer(result, query.tmpMemory, byteAccess.tmpRead + byteAccess.tmpWrite, 0, 0);
    }
    return result;
}

}  // namespace regor
