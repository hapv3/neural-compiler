//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "architecture/architecture.hpp"

namespace regor
{

class NeuralAIOpConfig final : public ArchitectureOpConfig
{
private:
    int _maxRows;

public:
    explicit NeuralAIOpConfig(int maxRows = 256) : _maxRows(maxRows) {}

    std::unique_ptr<ArchitectureOpConfig> Clone() override;
    int MaxIFMBuffering() override { return _maxRows * 32; }
    Point2i OptimalStripeGranule() override { return Point2i(32, 1); }
    Point2i MinimalStripeGranule() override { return Point2i(1, 1); }
    int OptimalDepthGranule() override { return 32; }
    int MinimumDepthGranule() override { return 32; }
    std::string ToString(bool full) override;
};

class NeuralAIOpGroup final : public ArchitectureOpGroup
{
private:
    bool _hasOp = false;

public:
    int Add(const ArchitectureOpGroupQuery &op, const std::vector<int> &dependsOn = {}) override;
    bool NeedsAllocation(UniqueId) override { return true; }
    Flags<Requirement> Requirements() override { return Requirement::None; }
};

}  // namespace regor
