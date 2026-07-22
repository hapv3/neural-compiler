//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "architecture/architecture.hpp"

namespace regor
{

class NeuralAIPerformance final : public ArchitecturePerformance
{
public:
    CycleCost MeasureCycleCost(const PerformanceQuery &query) override;
    int64_t MemToMemCycles(
        const ArchitectureMemory *dest, const ArchitectureMemory *source, int sizeBytes) override;
    ElementAccess MeasureElementAccess(const PerformanceQuery &query) override;
    ElementAccess ElementTransferToBytes(const PerformanceQuery &query, const ElementAccess &access) override;
    int64_t WeightDecodeCycles(const PerformanceQuery &query, const WeightStats &weights,
        Flags<WeightFormat> format, ArchitectureMemory *weightsMemory) override;
    void InitDatabase(Database *db) override;
    void RecordToDB(int opId) override;
    int64_t MinReadCycles(const ArchitectureMemory *mem, int64_t size, TensorUsage usage,
        OpType type, bool fastWeights) override;
    int64_t MinWriteCycles(const ArchitectureMemory *mem, int64_t size) override;
    std::unordered_map<const ArchitectureMemory *, AccessCycles>
    MeasureAccessCycles(const PerformanceQuery &query, const ElementAccess &byteAccess) override;
};

}  // namespace regor
