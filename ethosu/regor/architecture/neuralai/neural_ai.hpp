//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "architecture/architecture.hpp"

namespace regor
{

class ArchNeuralAI final : public Architecture
{
public:
    static constexpr int Clusters = 1;
    static constexpr int ArrayDimension = 32;
    static constexpr int DMAAlignment = 32;
    static constexpr int TCDMSizeBytes = 512 * 1024;
    static constexpr int CommandStagingBytes = 4 * 1024;
    static constexpr int AllocatableTCDMBytes = TCDMSizeBytes - CommandStagingBytes;
    static constexpr int LineBufferMaxKernel = 5;
    static constexpr int LineBufferMaxStride = 2;
    static constexpr int LineBufferMaxInputWidth = 640;
    static constexpr int RequantShiftMax = 31;

private:
    ArchitectureMemory *_modelMemory = nullptr;
    ArchitectureMemory *_tcdmMemory = nullptr;

public:
    ArchNeuralAI();

    bool ParseConfig(IniReader *reader) override;
    bool CheckConfiguration(std::string &error) override;
    std::unique_ptr<ArchitectureOpConfig> GetOpConfig(OpType opType, const ArchitectureConfigQuery &query) override;
    std::unique_ptr<ArchitectureOpGroup> CreateOpGroup(const ArchitectureOpGroupQuery &op) override;
    class WeightEncoder *WeightEncoder() override { return nullptr; }
    ArchitecturePerformance *Performance() override { return nullptr; }
    IRegisterCommandStreamGenerator *RegisterCommandStreamGenerator() override { return nullptr; }
    IArchitectureConstraints *Constraints() override { return nullptr; }
    TensorFormat IdealBufferingFormat() override { return TensorFormat::Row32; }
    int AllocationQuantum() const override { return DMAAlignment; }
    int TensorAlignment(TensorUsage usage, TensorFormat format) const override;
    TensorFormat ModelBindingFormat(TensorUsage usage) const override;
    TensorFormat DefaultInternalTensorFormat(TensorUsage usage, bool linearRequired) const override;
    Shape StorageShape(const Shape &logicalShape, TensorFormat format) const override;
    Shape TensorStrides(const Shape &logicalShape, TensorFormat format, DataType dataType) const override;
    bool CanAliasDepthOffset(TensorFormat format, int depthOffset) const override;
    Shape RollingBufferShape(const Shape &producerShape, const Shape &consumerShape,
        TensorFormat format) const override;
    Address MaxAddress() override { return INT64_C(1) << 32; }
    std::vector<uint32_t> ConfigRegisters() override { return {}; }
    uint32_t Version() override;
    int UpscaleAndRounding(ArchResampling resampling, int &rounding) override;
    AxisMask CanSubdivide(OpType opType, TransposeType transpose, ReverseType reverse) override;
    bool SupportsScalar(OpType opType, DataType dataType, TensorUsage usage) override;
    Flags<WeightFormat> SupportedWeightFormat(OpType op) override;
    void Call(std::function<void(const std::string &)> callBack) override;
};

}  // namespace regor
