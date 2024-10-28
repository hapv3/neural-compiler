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

#include "compiler/shape_util.hpp"
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
    std::unordered_set<UniqueId> tensorUids;
    int opTable = 0;
    int perfDebugTable = 0;

    if ( optDb )
    {
        db = optDb->Get();
        _arch->Performance()->InitDatabase(db);
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

        perfDebugTable = db->AddTable("perf_debug");

        columns = {};
        const std::vector<std::string> shapeColumns = {
            "ifm_shape",
            "ifm2_shape",
            "ofm_shape",
            "ifm_slice",
            "ifm2_slice",
            "ofm_slice",
            "ifm_stripe",
            "ifm2_stripe",
            "ofm_stripe",
        };

        for ( auto &shape : shapeColumns )
        {
            columns.push_back(shape + "_n");
            columns.push_back(shape + "_h");
            columns.push_back(shape + "_w");
            columns.push_back(shape + "_c");
        }

        columns.insert(columns.end(),
            {
                "ifm_memory",
                "ifm2_memory",
                "ofm_memory",
                "ifm_format",
                "ifm2_format",
                "ofm_format",
                "ifm_dtype",
                "ifm2_dtype",
                "ofm_dtype",
                "ifm_pre_buffering",
                "ifm2_pre_buffering",
                "ifm_buffering",
                "ifm2_buffering",
                "reverse_type",
                "transpose_type",
                "time_index",
                "cascade",
                "weight_format",
                "weight_dtype",
                "weight_total_bytes",
                "weight_max_range_bytes",
                "weight_sub_streams",
                "weight_distinct",
                "weight_zero",
                "scales_dtype",
                "scales_total_bytes",
                "scales_max_range_bytes",
                "ofm_depth_slices",
                "weight_pre_buffer",
                "weight_buffering",
                "weight_transfer_cycles",
                "kernel_depth_multiplier",
            });

        columns.emplace_back("kernel_padding_T");
        columns.emplace_back("kernel_padding_B");
        columns.emplace_back("kernel_padding_L");
        columns.emplace_back("kernel_padding_R");
        columns.emplace_back("kernel_padding_N");
        columns.emplace_back("kernel_padding_F");

        const std::vector<std::string> xyzColumns = {
            "kernel_size",
            "kernel_dilation",
            "kernel_stride",
        };
        for ( auto &xyzCol : xyzColumns )
        {
            columns.push_back(xyzCol + "_x");
            columns.push_back(xyzCol + "_y");
            columns.push_back(xyzCol + "_z");
        }

        for ( const auto &mem : memories )
        {
            columns.push_back(mem->Name() + "_read_efficiency");
            columns.push_back(mem->Name() + "_write_efficiency");
            columns.push_back(mem->Name() + EnumToString(AccessType::Lut) + "_ac");
            columns.push_back(mem->Name() + EnumToString(AccessType::Lut) + "_read");
            columns.push_back(mem->Name() + EnumToString(AccessType::Lut) + "_write");
            columns.push_back(mem->Name() + EnumToString(AccessType::FeatureMap) + "_ac");
            columns.push_back(mem->Name() + EnumToString(AccessType::FeatureMap) + "_read");
            columns.push_back(mem->Name() + EnumToString(AccessType::FeatureMap) + "_write");
            columns.push_back(mem->Name() + EnumToString(AccessType::Weights) + "_ac");
            columns.push_back(mem->Name() + EnumToString(AccessType::Weights) + "_read");
            columns.push_back(mem->Name() + EnumToString(AccessType::Weights) + "_write");
            columns.push_back(mem->Name() + EnumToString(AccessType::Scales) + "_ac");
            columns.push_back(mem->Name() + EnumToString(AccessType::Scales) + "_read");
            columns.push_back(mem->Name() + EnumToString(AccessType::Scales) + "_write");
        }

        db->AddColumns(perfDebugTable, std::move(columns));
    }

    for ( auto const &schedOp : _ops )
    {
        SchedulerOpInfo *cost = schedule->Cost(schedOp.get());
        PerformanceResult perf = ProcessOpPerformance(schedOp.get(), cost, schedule, prevOp, prevCost, memories);
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
        if ( optDb != nullptr )
        {
            AddToDatabase(perf, schedOp.get(), cost, opTable, perfDebugTable, memories, optDb);
        }
        performance += perf;
        prevOp = schedOp.get();
        prevCost = cost;

        for ( auto const &subOp : schedOp->_subOps )
        {
            cost = schedule->Cost(subOp.get());
            perf = ProcessOpPerformance(subOp.get(), cost, schedule, prevOp, prevCost, memories);
            if ( optDb != nullptr )
            {
                AddToDatabase(perf, subOp.get(), cost, opTable, perfDebugTable, memories, optDb);
            }
            performance += perf;
            prevOp = subOp.get();
            prevCost = cost;
        }
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


PerformanceResult NetworkPerformance::ProcessOpPerformance(SchedulerOperation *schedOp, SchedulerOpInfo *cost, Schedule *schedule,
    SchedulerOperation *prevOp, SchedulerOpInfo *prevCost, const std::unordered_set<ArchitectureMemory *> &memories)
{
    PerformanceResult perf = {};
    if ( schedOp->IsNpuOp() )
    {
        perf = EstimateFullOpPerformance(schedOp, cost, prevOp, prevCost);
        perf.npuOps = 1;
        perf.memory[_arch->StagingMemory().memory].peakUsage = schedule->MemoryUsageAt(cost->timeIndex);
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
    return perf;
}


void NetworkPerformance::AddToDatabase(const PerformanceResult &perf, SchedulerOperation *schedOp, SchedulerOpInfo *cost,
    int opTable, int perfDebugTable, const std::unordered_set<ArchitectureMemory *> &memories, OptimiserDatabase *optDb)
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

    db->AddRow(opTable, schedOp->Uid(), std::move(row));

    row = {};
    auto shapeToStrings = [&row](const std::vector<int> &shape)
    {
        std::transform(shape.begin(), shape.end(), std::back_inserter(row),
            [](int n) -> std::string { return n ? std::to_string(n) : ""; });
    };
    // clang-format off
    // FM shapes
    shapeToStrings(ReshapeToNHWC(schedOp->IFM(0)->shape).ToList<int>());
    shapeToStrings(ReshapeToNHWC(schedOp->TryIFM(1) ? schedOp->IFM(1)->shape : Shape()).ToList<int>());
    shapeToStrings(ReshapeToNHWC(schedOp->OFM()->shape).ToList<int>());
    // Slice shapes
    shapeToStrings(ReshapeToNHWC(schedOp->IFM(0)->slice.shape).ToList<int>());
    shapeToStrings(ReshapeToNHWC(schedOp->TryIFM(1) ? schedOp->IFM(1)->slice.shape : Shape()).ToList<int>());
    shapeToStrings(ReshapeToNHWC(schedOp->OFM()->slice.shape).ToList<int>());
    // Stripe shapes
    shapeToStrings(ReshapeToNHWC(cost->stripeInput[0]).ToList<int>());
    shapeToStrings(ReshapeToNHWC(schedOp->TryIFM(1) ? cost->stripeInput[1] : Shape()).ToList<int>());
    shapeToStrings(ReshapeToNHWC(cost->stripe).ToList<int>());

    row.insert(row.end(), {
        // FM Memory
        fmt::format("{}", schedOp->IFM(0)->tensor->memArea.memory->Name()),
        fmt::format("{}", schedOp->TryIFM(1) ? schedOp->IFM(1)->tensor->memArea.memory->Name() : ""),
        fmt::format("{}", schedOp->OFM()->tensor->memArea.memory->Name()),
        // Formats
        fmt::format("{}", EnumToString(schedOp->IFM(0)->tensor->format)),
        fmt::format("{}", schedOp->TryIFM(1) ? EnumToString(schedOp->IFM(1)->tensor->format) : ""),
        fmt::format("{}", EnumToString(schedOp->OFM()->tensor->format)),
        // Data types
        fmt::format("{}", EnumToString(schedOp->IFM(0)->tensor->dataType)),
        fmt::format("{}", schedOp->TryIFM(1) ? EnumToString(schedOp->IFM(1)->tensor->dataType) : ""),
        fmt::format("{}", EnumToString(schedOp->OFM()->tensor->dataType)),
        // IFM Buffering
        std::to_string(schedOp->IFM(0)->preBuffer),
        schedOp->TryIFM(1) ? std::to_string(schedOp->IFM(1)->preBuffer) : "",
        EnumToString(schedOp->IFM(0)->buffering),
        schedOp->TryIFM(1) ? EnumToString(schedOp->IFM(1)->buffering) : "",
        // Transpose and Reverse Types
        EnumToString(schedOp->OFM()->transpose),
        EnumToString(schedOp->OFM()->reverse),
        // Timeindex
        std::to_string(cost->timeIndex),
        // Cascade
        std::to_string(cost->cascade),
        // Weights
        cost->npuWeightsTensor ? cost->npuWeightsTensor->config->Format().ToString() : "",
        cost->npuWeightsTensor ? EnumToString(cost->npuWeightsTensor->dataType) : "",
        cost->npuWeightsTensor ? std::to_string(cost->npuWeightsTensor->totalWeightBytes) : "",
        cost->npuWeightsTensor ? std::to_string(cost->npuWeightsTensor->maxRangeBytes) : "",
        cost->npuWeightsTensor ? std::to_string(cost->npuWeightsTensor->subStreams) : "",
        cost->npuWeightsTensor ? std::to_string(cost->npuWeightsTensor->distinctWeights) : "",
        cost->npuWeightsTensor ? std::to_string(cost->npuWeightsTensor->zeroCount) : "",
        // Scales
        cost->npuScalesTensor ? EnumToString(cost->npuScalesTensor->dataType) : "",
        cost->npuScalesTensor ? std::to_string(cost->npuScalesTensor->totalWeightBytes) : "",
        cost->npuScalesTensor ? std::to_string(cost->npuScalesTensor->maxRangeBytes) : "",
        // Weight Buffering
        fmt::format("{}", fmt::join(cost->ofmDepthSlices, "|")),
        cost->bufferedWeightTensor.tensor ? std::to_string(cost->bufferedWeightTensor.preBuffer) : "",
        cost->bufferedWeightTensor.tensor ? EnumToString(cost->bufferedWeightTensor.buffering) : "",
        cost->bufferedWeightTensor.tensor ? std::to_string(cost->fullWeightTransferCycles) : "",
        // Kernel
        std::to_string(schedOp->Kernel()->DepthMultiplier()),
        std::to_string(schedOp->Kernel()->Padding().Top()),
        std::to_string(schedOp->Kernel()->Padding().Bottom()),
        std::to_string(schedOp->Kernel()->Padding().Left()),
        std::to_string(schedOp->Kernel()->Padding().Right()),
        std::to_string(schedOp->Kernel()->Padding().Near()),
        std::to_string(schedOp->Kernel()->Padding().Far()),
        std::to_string(schedOp->Kernel()->Size3D().x),
        std::to_string(schedOp->Kernel()->Size3D().y),
        std::to_string(schedOp->Kernel()->Size3D().z),
        std::to_string(schedOp->Kernel()->Dilation3D().x),
        std::to_string(schedOp->Kernel()->Dilation3D().y),
        std::to_string(schedOp->Kernel()->Dilation3D().z),
        std::to_string(schedOp->Kernel()->Stride3D().x),
        std::to_string(schedOp->Kernel()->Stride3D().y),
        std::to_string(schedOp->Kernel()->Stride3D().z),
    });
    // clang-format on
    for ( const auto mem : memories )
    {
        // Add read/write transferEfficiencies for all memories
        row.push_back(std::to_string(perf.memory.at(mem).readTransferEff));
        row.push_back(std::to_string(perf.memory.at(mem).writeTransferEff));
        // For all usages, add access read and access write:
        for ( int i = 0; i < int(AccessType::Last); i++ )
        {
            if ( perf.memory.at(mem).access.find(static_cast<AccessType>(i)) != perf.memory.at(mem).access.end() )
            {
                row.push_back(std::to_string(perf.memory.at(mem).access.at(static_cast<AccessType>(i)).accessCycles));
                row.push_back(std::to_string(perf.memory.at(mem).access.at(static_cast<AccessType>(i)).bytesRead));
                row.push_back(std::to_string(perf.memory.at(mem).access.at(static_cast<AccessType>(i)).bytesWritten));
            }
            else
            {
                row.emplace_back("");
                row.emplace_back("");
                row.emplace_back("");
            }
        }
    }

    db->AddRow(perfDebugTable, schedOp->Uid(), std::move(row));
}


PerformanceResult NetworkPerformance::EstimateFullOpPerformance(
    SchedulerOperation *schedOp, SchedulerOpInfo *cost, SchedulerOperation *prevOp, SchedulerOpInfo *prevCost)
{
    UNUSED(prevOp);
    auto wgtFormat = cost->npuWeightsTensor ? cost->npuWeightsTensor->config->Format() : Flags<WeightFormat>(WeightFormat::Default);
    PerformanceQuery query = Scheduler::InitPerfQuery(schedOp, cost->Config(), -1, wgtFormat);
    std::vector<FusionQuery> fused = Scheduler::InitFusionQuery(schedOp);

    // Memory that NPU will source weights from for operations
    ArchitectureMemory *weightsMemory = cost->npuWeightsTensor ? cost->npuWeightsTensor->memArea.memory : nullptr;

    _arch->Performance()->RecordToDB(schedOp->Uid());
    CycleCost cycles = _arch->Performance()->MeasureCycleCost(query, fused);

    if ( cost->npuWeightsTensor )
    {
        WeightStats weightStats;
        weightStats.size = cost->npuWeightsTensor->totalSourceBytes;
        weightStats.encodedSize = cost->npuWeightsTensor->totalWeightBytes;
        weightStats.zeroCount = cost->npuWeightsTensor->zeroCount;
        weightStats.distinctWeights = cost->npuWeightsTensor->distinctWeights;
        _arch->Performance()->RecordToDB(schedOp->Uid());
        _arch->Performance()->WeightDecodeCycles(query, weightStats, query.weightFormat, weightsMemory);
    }

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
    result.memory[ofm->tensor->memArea.memory]
        .writeTransferOverhead += byteAccess.ofmWrite - DataTypeSizeBits(ofm->tensor->dataType) / 8 * access.ofmWrite;

    // IFM1 read
    auto ifm = schedOp->IFM(0);
    result.memory[ifm->tensor->memArea.memory].access[AccessType::FeatureMap].bytesRead += byteAccess.ifmRead[0];
    result.memory[ifm->tensor->memArea.memory]
        .readTransferOverhead += byteAccess.ifmRead[0] - DataTypeSizeBits(ifm->tensor->dataType) / 8 * access.ifmRead[0];

    // IFM2 read
    auto ifm2 = schedOp->TryIFM(1);
    if ( ifm2 )
    {
        result.memory[ifm2->tensor->memArea.memory].access[AccessType::FeatureMap].bytesRead += byteAccess.ifmRead[1];
        result.memory[ifm2->tensor->memArea.memory]
            .readTransferOverhead += byteAccess.ifmRead[1] - DataTypeSizeBits(ifm2->tensor->dataType) / 8 * access.ifmRead[1];
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
        int64_t totalReadBytes = 0;
        int64_t totalWriteBytes = 0;
        float bandwidth = mem->Bandwidth();
        int64_t memBytes = 0;
        for ( auto &[accType, acc] : stats.access )
        {
            // compute cycles per accessType
            int64_t bytes = acc.bytesRead + acc.bytesWritten;
            memBytes += bytes;
            int64_t accCycles = int64_t(float(bytes) / bandwidth);
            acc.accessCycles = accCycles;
            totalReadBytes += acc.bytesRead;
            totalWriteBytes += acc.bytesWritten;
        }
        // get maximum cycles per memory
        int64_t memCycles = int64_t(float(memBytes) / bandwidth);
        maxMemCycles = std::max(maxMemCycles, memCycles);

        if ( totalReadBytes > 0 )
        {
            stats.readTransferEff = float(totalReadBytes - stats.readTransferOverhead) / totalReadBytes;
        }
        if ( totalWriteBytes > 0 )
        {
            stats.writeTransferEff = float(totalWriteBytes - stats.writeTransferOverhead) / totalWriteBytes;
        }
    }

    result.totalCycles = std::max(result.npuCycles, maxMemCycles);
    return result;
}

}  // namespace regor
