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

#include "architecture/architecture.hpp"
#include "architecture/architecture_constraints.hpp"
#include "architecture/ethos_u_scaling.hpp"
#include "architecture/register_command_stream_generator.hpp"
#include "architecture/weight_encoder.hpp"
#include "common/bit_flags.hpp"
#include "common/shape.hpp"
#include "ethos_u55_performance.hpp"

#include <string>
#include <unordered_set>

namespace regor
{

enum EthosU55SHRamElements
{
    SHRAM_IFM8 = 0,
    SHRAM_IFM16 = 1,
    SHRAM_IFM8_Elementwise = 2,
    SHRAM_IFM16_Elementwise = 3,
    SHRAM_IFM32 = 4,
    SHRAM_Acc16 = 5,
    SHRAM_Acc32 = 6,
    SHRAM_Acc40 = 7,
    SHRAM_Last = SHRAM_Acc40
};

enum class EthosUTraversal
{
    DepthFirst = 0,
    PartKernel = 1,
    Depthwise = 2,
};

class ArchEthosU55;

enum class EthosU55NpuOp
{
    None = 0,
    Convolution,
    Depthwise,
    VectorProduct,
    Pooling,
    ReduceSum,
    Elementwise,
    Dma,
    Compound,
    Last = Compound,
};

/// <summary>
/// Per-operator architecture configuration
/// </summary>
class EthosU55OpConfig : public ArchitectureOpConfig
{
    friend class ArchEthosU55;
    friend class EthosU55RCSGenerator;

public:
    struct SHRAMLayout
    {
        int ibStart = 0;
        int ibEnd = 0;
        int ibStart2 = 0;
        int abStart = 0;
        int lutStart = 0;
    };

private:
    SHRAMLayout _layout;
    Shape _ifmBlock;
    Shape _ofmBlock;
    EthosU55SHRamElements _accumulatorType = SHRAM_Acc32;
    EthosUTraversal _traversal = EthosUTraversal::DepthFirst;
    int _bankSize = 0;
    int _ifmDepthBufScaling = 0;

public:
    EthosUTraversal Traversal() const { return _traversal; }
    const Shape &IfmBlock() const { return _ifmBlock; }
    const Shape &OfmBlock() const { return _ofmBlock; }
    EthosU55SHRamElements Acc() const { return _accumulatorType; }

    std::unique_ptr<ArchitectureOpConfig> Clone() override;
    int MaxIFMBuffering() override;
    Point2i OptimalStripeGranule() override;
    int OptimalDepthGranule() override;
    std::string ToString(bool full) override;
};

/// <summary>
/// Group of ops that can be fused and/or chained
/// </summary>
class EthosU55OpGroup : public ArchitectureOpGroup
{
    friend class ArchEthosU55;

    using OpInfo = ArchitectureOpGroupQuery;

    struct InternalOpInfo
    {
        std::vector<int> dependsOn;
    };

private:
    std::array<OpInfo, 2> _ops;
    std::array<InternalOpInfo, 2> _opsInternal;
    int _opsCount = 0;
    std::unordered_set<UniqueId> _fusedTensors;

public:
    int Add(const ArchitectureOpGroupQuery &op, const std::vector<int> &dependsOn = {}) override;
    bool NeedsAllocation(UniqueId TensorUID) override;

protected:
    bool CanRunOnNPU(const ArchitectureOpGroupQuery &op) override;
};

/// <summary>
/// EthosU55 specialisation
/// </summary>
class ArchEthosU55 : public Architecture
{
    friend class EthosU55WeightEncoder;
    friend class EthosU55Performance;
    friend class EthosU55RCSGenerator;
    friend class EthosU65RCSGenerator;
    friend class EthosU55OpGroup;
    friend class EthosU55Constraints;

public:
    struct AcceleratorConfig
    {
        int macs;
        int cores;
        Shape ofmUBlock;
        Shape ifmUblock;
        int shramBanks;
        int8_t shramGranules[8];
        int elemUnits;
        const EthosU55PerfInfo *perfInfo;
    };

private:
    std::unique_ptr<ArchitectureMemory> _shramMemory;
    Shape _subkernelMax;
    Shape _ofmBlockMax;
    int _cores = 0;
    int _macs = 0;
    Shape _ofmUBlock;
    Shape _ifmUBlock;
    const int8_t *_shramGranules = nullptr;
    int _ifmBankGranules[4] = {0};
    int _ifmEWBankGranules[4] = {0};

    struct
    {
        int reservedOutputBanks = 0;
        int bankSizeBytes = 0;
        int totalBanks = 0;
        int reservedEndBanks = 0;
        int lutBanks = 2;
        int lutSlotSize = 256;
    } _shram;

protected:
    std::unique_ptr<class WeightEncoder> _weightEncoder;
    std::unique_ptr<ArchitecturePerformance> _performance;
    std::unique_ptr<IRegisterCommandStreamGenerator> _rcsGenerator;
    std::unique_ptr<IArchitectureConstraints> _constraints;

public:
    ArchEthosU55();

public:
    bool ParseConfig(IniReader *reader) override;

    std::unique_ptr<ArchitectureOpConfig> GetOpConfig(OpType opType, const ArchitectureConfigQuery &query) override;
    std::unique_ptr<ArchitectureOpGroup> CreateOpGroup(const ArchitectureOpGroupQuery &op) override;
    class WeightEncoder *WeightEncoder() override { return _weightEncoder.get(); }
    IRegisterCommandStreamGenerator *RegisterCommandStreamGenerator() override { return _rcsGenerator.get(); }
    IArchitectureConstraints *Constraints() override { return _constraints.get(); }
    ArchitecturePerformance *Performance() override { return _performance.get(); }
    TensorFormat IdealBufferingFormat() override { return TensorFormat::NHCWB16; }
    Address MaxAddress() override { return 1LL << 32; }
    std::vector<uint32_t> ConfigRegisters() override;
    int UpscaleAndRounding(ArchResampling resampling, int &rounding) override;
    AxisMask CanSubdivide(OpType opType, TransposeType transpose, ReverseType reverse) override;
    bool SupportsScalar(OpType opType, DataType dataType, TensorUsage usage) override;
    Flags<WeightFormat> SupportedWeightFormat(OpType op) override;
    uint32_t Version() override;

protected:
    Shape OfmUBlock() { return _ofmUBlock; }
    void ApplyConfig(const AcceleratorConfig *cfg);

    std::unique_ptr<ArchitectureOpConfig> FindBlockConfig(OpType opType, const ArchitectureConfigQuery &query);

    bool TryBlockConfig(EthosU55OpConfig::SHRAMLayout &layout, int ewUsage, const Shape &ofmBlock, const Shape &ifmBlock,
        int ifmBits, int ifmGranule, int accBits, int accGranule, int lutBanks, int ifmDepthBufScaling);

    Shape GetStorageRounding(TensorFormat format);

    uint32_t ConfigRegister(int product);

    bool IsU55_32() const { return (_macs == 32) && (_cores == 1); }

    // Checks if the operation is to be mapped on AvgPool
    static bool UseAvgPoolNop(OpType type);
    static EthosU55NpuOp GetHWOp(OpType type);

private:
    int MaxOutstandingKernelOps() { return 2; }
    virtual int MaxOutstandingDMAOps() { return 1; }
    int MaxBlockdep() { return 3; }
};

}  // namespace regor
