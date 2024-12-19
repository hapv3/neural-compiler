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

#pragma once

#include "common/common.hpp"

#include "architecture/architecture.hpp"

namespace regor
{

class ArchEthosU55;

struct EthosU55PerfInfo
{
    float outputCycles[8];
    float activationCycles[3];
};

struct EthosU55Cycles
{
    int64_t cycles;
    int64_t macCycles;
    int64_t aoCycles;
    int64_t cmdCycles;
};

struct EthosU55ElementCycles
{
    float cycles;
    float aoCycles;
    float cmdCycles;
};

/// <summary>
/// Profiles performance analysis for Ethos-U55
/// </summary>
class EthosU55Performance : public ArchitecturePerformance
{
protected:
    ArchEthosU55 *_arch;
    const EthosU55PerfInfo *_perfInfo;
    Database *_db = nullptr;
    int _nextId = -1;
    int _mainTable = -1;
    int _wdTable = -1;

public:
    EthosU55Performance(ArchEthosU55 *arch, const EthosU55PerfInfo *perfInfo);

public:
    CycleCost MeasureCycleCost(const PerformanceQuery &query, const std::vector<FusionQuery> &fused) override;
    CycleCost MeasureCycleCostForSparsity(const PerformanceQuery &query, const std::vector<FusionQuery> &fused) override;
    int64_t MemToMemCycles(const ArchitectureMemory *dest, const ArchitectureMemory *source, int sizeBytes) override;
    ElementAccess MeasureElementAccess(const PerformanceQuery &query) override;
    ElementAccess ElementTransferToBytes(const PerformanceQuery &query, const ElementAccess &access) override;
    int64_t WeightDecodeCycles(const PerformanceQuery &query, const WeightStats &weights, Flags<WeightFormat> format,
        ArchitectureMemory *weightsMemory) override;
    float ChannelBW(const ArchitectureMemory *mem, MemChannel channel) override;
    void InitDatabase(Database *optDB) override;
    void RecordToDB(int opId) override;

private:
    EthosU55Cycles EstimateConvCycles(const PerformanceQuery &query, const std::vector<FusionQuery> &fused);
    EthosU55ElementCycles EstimateOutputCyclesPerElement(const PerformanceQuery &query, const std::vector<FusionQuery> &fused);
    int64_t EstimateMinimumMemoryCycles(const PerformanceQuery &query);
};

}  // namespace regor
