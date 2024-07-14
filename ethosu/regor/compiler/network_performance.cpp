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

#include "network_performance.hpp"

#include "common/common.hpp"

#include "database.hpp"
#include "graph_optimiser.hpp"

#include <unordered_map>

BEGIN_ENUM_TABLE(regor::AccessType)
    ADD_ENUM_NAME(Lut)
    ADD_ENUM_NAME(FeatureMap)
    ADD_ENUM_NAME(Weights)
    ADD_ENUM_NAME(Scales)
END_ENUM_TABLE()

namespace regor
{
NetworkPerformance::NetworkPerformance(Architecture *arch, const std::vector<std::unique_ptr<SchedulerOperation>> &ops) :
        _arch(arch), _ops(ops)
{
    assert(arch);
}

PerformanceResult NetworkPerformance::Measure(Schedule *schedule, OptimiserDatabase *optDb)
{
    SchedulerOperation *prevOp = nullptr;
    SchedulerOpInfo *prevCost = nullptr;
    PerformanceResult performance;
    Database *db = nullptr;
    std::unordered_set<ArchitectureMemory *> memories({_arch->ReadonlyMemory().memory, _arch->FeatureMapMemory().memory,
        _arch->LUTMemory().memory, _arch->StagingMemory().memory});
    std::unordered_set<MemArea, MemArea::hash> regions(
        {_arch->ReadonlyMemory(), _arch->FeatureMapMemory(), _arch->LUTMemory(), _arch->StagingMemory()});
    int opTable = 0;
    int opTableColumnCount = 0;
    std::unordered_set<UniqueId> tensorUids;

    if ( optDb )
    {
        db = optDb->Get();
        opTable = db->AddTable("perf");
        std::vector<std::string> columns = {
            "source_id",
            "optimised_id",
            "operator",
            "name",
            "staging_usage",
            "op_cycles",
            "npu_cycles",
            "mac_count",
        };
        for ( const auto &mem : memories )
        {
            std::string label = mem->Name() + "_ac";
            columns.push_back(label);
        }
        db->AddColumns(opTable, columns);
        opTableColumnCount = int(columns.size());
    }

    // TODO MLBEDSW-7954 handle sub-operations
    for ( auto const &schedOp : _ops )
    {
        SchedulerOpInfo *cost = schedule->Cost(schedOp.get());
        PerformanceResult perf = {};
        if ( schedOp->IsNpuOp() )
        {
            perf = EstimateFullOpPerformance(schedOp.get(), cost, prevOp, prevCost);
            perf.npuOps = 1;
            perf.memory[_arch->StagingMemory().memory].peakUsage = schedule->MemoryUsageAt(cost->timeIndex);

            // Calculate total original and encoded weights
            // Weight statistics is not set on a per-operation level as some operations share weight tensors
            SchedulerConnection *weightConn = schedOp->TryInput(TensorUsage::Weights);
            if ( weightConn && cost->npuWeightsTensor )
            {
                // check if the weight tensor has already been accounted for in total weights
                auto pos = tensorUids.find(weightConn->tensor->uid);
                if ( pos == std::end(tensorUids) )
                {
                    tensorUids.insert(weightConn->tensor->uid);
                    performance.originalWeights += weightConn->tensor->AllocationSizeBytes();
                    performance.encodedWeights += cost->npuWeightsTensor->totalWeightBytes;
                }
            }
        }
        else
        {
            perf.cpuCycles = 1;  // TODO: model CPU cycle counts
            perf.cpuOps = 1;
        }
        // Insert any missing memories
        for ( ArchitectureMemory *a : memories )
        {
            perf.memory.emplace(a, PerformanceResult::MemoryAccesses{});
        }

        if ( optDb != nullptr )
        {
            AddToDatabase(perf, schedOp, opTable, opTableColumnCount, memories, optDb);
        }

        performance += perf;
        prevOp = schedOp.get();
        prevCost = cost;
    }
    // TODO: Remove this line and separate memory allocation from usage.
    performance.memory[_arch->StagingMemory().memory].peakUsage = 0;

    for ( auto &region : regions )
    {
        // RHS is not peak usage, but peak allocation.
        performance.memory[region.memory].peakUsage += schedule->memoryUsage[region];
    }

    performance.cascades = schedule->cascades.size();

    return performance;
}

void NetworkPerformance::AddToDatabase(const PerformanceResult &perf, const std::unique_ptr<SchedulerOperation> &schedOp,
    int opTable, int /*opTableColumnCount*/, const std::unordered_set<ArchitectureMemory *> &memories, OptimiserDatabase *optDb)
{
    // Per-layer calculations
    assert(optDb != nullptr);
    std::vector<std::string> row;
    std::string opName = "N/A";
    Database *db = optDb->Get();

    const auto *conn = schedOp->TryOFM();
    if ( conn != nullptr && conn->tensor != nullptr && conn->tensor->srcTensor != nullptr )
    {
        opName = conn->tensor->srcTensor->Name();
    }

    int sourceId = optDb->SourceId(schedOp->_srcKey);
    int optId = optDb->OptimisedId(schedOp->_srcKey);
    row = {
        std::to_string(sourceId),
        std::to_string(optId),
        OpTypeToString(schedOp->Type()),
        std::move(opName),
        std::to_string(perf.memory.at(_arch->StagingMemory().memory).peakUsage),
        std::to_string(perf.totalCycles),
        std::to_string(perf.npuCycles),
        std::to_string(perf.macCount),
    };

    for ( const auto mem : memories )
    {
        row.push_back(std::to_string(perf.memory.at(mem).AccessCycles()));
    }

    db->AddRow(opTable, schedOp->Index(), std::move(row));
}


PerformanceResult NetworkPerformance::EstimateFullOpPerformance(
    SchedulerOperation *schedOp, SchedulerOpInfo *cost, SchedulerOperation *prevOp, SchedulerOpInfo *prevCost)
{
    UNUSED(prevOp);
    auto wgtFormat = cost->npuWeightsTensor ? cost->npuWeightsTensor->config->Format() : Flags<WeightFormat>(WeightFormat::Default);
    PerformanceQuery query = Scheduler::InitPerfQuery(schedOp, cost->Config(), -1, wgtFormat);
    std::vector<FusionQuery> fused = Scheduler::InitFusionQuery(schedOp);

    CycleCost cycles = _arch->Performance()->MeasureCycleCost(query, fused);

    PerformanceResult result;
    result.npuCycles = cycles.opCycles;
    result.macCount = cycles.macs;

    if ( cost->cascade != 0 )
    {
        result.cascadedOps = 1;
    }

    ElementAccess access = _arch->Performance()->MeasureElementAccess(query);
    ElementAccess byteAccess = _arch->Performance()->ElementTransferToBytes(query, access);

    // How many NPU cycles are available under the previously executing
    // operator for performing buffered DMA transfers
    int64_t slackCycles = (prevCost != nullptr) ? prevCost->slackBufferingCycles : 0;

    // LUT transfer stats
    auto lut = schedOp->TryInput(TensorUsage::LUT);
    int64_t lutTransferCycles = 0;

    if ( lut )
    {
        auto srcMemory = lut->tensor->memArea.memory;
        auto dstMemory = _arch->LUTMemory().memory;
        assert(srcMemory);

        if ( (srcMemory != nullptr) && (dstMemory != srcMemory) )
        {
            int copySize = lut->PartialAllocationSizeBytes();
            lutTransferCycles = _arch->Performance()->MemToMemCycles(dstMemory, srcMemory, copySize);

            result.memory[srcMemory].access[AccessType::Lut].bytesRead += copySize;
            result.memory[dstMemory].access[AccessType::Lut].bytesWritten += copySize;
        }
    }

    // Memory that NPU will source weights from for operations
    ArchitectureMemory *weightsMemory = cost->npuWeightsTensor ? cost->npuWeightsTensor->memArea.memory : nullptr;

    if ( weightsMemory && cost->bufferedWeightTensor.tensor )
    {
        // DMA Weight Transfer
        int initialSize = 0;

        // Get the size of the first DMA
        for ( int streamIndex = 0; streamIndex < cost->npuWeightsTensor->subStreams; streamIndex++ )
        {
            auto pos = cost->npuWeightsTensor->encodedRanges.find(streamIndex);
            if ( pos != cost->npuWeightsTensor->encodedRanges.end() )
            {
                initialSize += pos->second.TotalBytes();
            }
        }

        auto srcWeightMem = weightsMemory;
        auto dstWeightMem = cost->bufferedWeightTensor.tensor->memArea.memory;
        assert(srcWeightMem != dstWeightMem);

        weightsMemory = dstWeightMem;  // Update source to use buffered weight memory

        // Calculate initial weight transfer cycles
        int64_t weightCycles = _arch->Performance()->MemToMemCycles(dstWeightMem, srcWeightMem, initialSize);
        weightCycles = std::max(weightCycles - slackCycles, int64_t(0));

        int weightsSize = cost->npuWeightsTensor->AllocationSizeBytes();
        result.memory[srcWeightMem].access[AccessType::Weights].bytesRead += weightsSize;
        result.memory[dstWeightMem].access[AccessType::Weights].bytesWritten += weightsSize;

        // Add cycles for Weight + Scale Transfer
        result.npuCycles = std::max(cost->fullWeightTransferCycles - slackCycles + cost->slackBufferingCycles, cycles.opCycles + weightCycles);
    }
    else
    {
        // Calculate non-hidden LUT transfer cycles
        lutTransferCycles = std::max(lutTransferCycles - slackCycles, int64_t(0));
    }

    // Add cycles for LUT Transfer
    result.npuCycles += lutTransferCycles;

    // OFM write
    auto ofm = schedOp->OFM();
    result.memory[ofm->tensor->memArea.memory].access[AccessType::FeatureMap].bytesWritten += byteAccess.ofmWrite;

    // IFM1 read
    auto ifm = schedOp->IFM(0);
    result.memory[ifm->tensor->memArea.memory].access[AccessType::FeatureMap].bytesRead += byteAccess.ifmRead[0];

    // IFM2 read
    auto ifm2 = schedOp->TryIFM(1);
    if ( ifm2 )
    {
        result.memory[ifm2->tensor->memArea.memory].access[AccessType::FeatureMap].bytesRead += byteAccess.ifmRead[1];
    }

    // Weight read
    if ( cost->npuWeightsTensor && access.constRead[0] > 0 )
    {
        int encodedWeightsSize = cost->npuWeightsTensor->totalWeightBytes;
        result.memory[weightsMemory].access[AccessType::Weights].bytesRead += int64_t(encodedWeightsSize) * access.weightsRefetch;
    }

    // Scale read
    if ( cost->npuWeightsTensor && access.constRead[1] > 0 )
    {
        int encodedScaleSize = cost->npuWeightsTensor->AllocationSizeBytes() - cost->npuWeightsTensor->totalWeightBytes;
        result.memory[weightsMemory].access[AccessType::Scales].bytesRead += int64_t(encodedScaleSize) * access.weightsRefetch;
    }

    // Update memory-access cycles and find the maximum memory read cycle time
    int64_t maxMemCycles = 0;
    for ( auto &[mem, stats] : result.memory )
    {
        float bandwidth = mem->Bandwidth();
        int64_t memBytes = 0;
        for ( auto &[accType, acc] : stats.access )
        {
            // compute cycles per accessType
            int64_t bytes = acc.bytesRead + acc.bytesWritten;
            memBytes += bytes;
            int64_t accCycles = int64_t(float(bytes) / bandwidth);
            acc.accessCycles = accCycles;
        }
        // get maximum cycles per memory
        int64_t memCycles = int64_t(float(memBytes) / bandwidth);
        maxMemCycles = std::max(maxMemCycles, memCycles);
    }

    result.totalCycles = std::max(result.npuCycles, maxMemCycles);
    return result;
}

}  // namespace regor
