//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "architecture/architecture.hpp"

namespace regor
{

enum class NeuralAIOpMode
{
    Unsupported,
    FullyConnectedRow32,
    MatMulRow32,
    Conv2DRgbLinebufRequant,
    Conv2DPointwiseC32Requant,
    Conv2DLinebufC32S1Requant,
    Conv2DLinebufC32S2Requant,
    Conv2DLinebufC32TailRequant,
    DepthwiseC32S1Requant,
    DepthwiseC32S2Requant,
};

const char *NeuralAIOpModeName(NeuralAIOpMode mode);

class NeuralAIOpConfig final : public ArchitectureOpConfig
{
private:
    int _maxRows;
    NeuralAIOpMode _mode;
    bool _directNhwcInput;
    bool _groupStationary;

public:
    explicit NeuralAIOpConfig(int maxRows = 256, NeuralAIOpMode mode = NeuralAIOpMode::Unsupported,
        bool directNhwcInput = false, bool groupStationary = false) :
            _maxRows(maxRows), _mode(mode), _directNhwcInput(directNhwcInput),
            _groupStationary(groupStationary)
    {
    }

    std::unique_ptr<ArchitectureOpConfig> Clone() override;
    NeuralAIOpMode Mode() const { return _mode; }
    bool DirectNhwcInput() const { return _directNhwcInput; }
    bool GroupStationary() const { return _groupStationary; }
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
