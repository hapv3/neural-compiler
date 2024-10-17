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
#include "ethos_u85_performance.hpp"

#include <string>
#include <unordered_set>

namespace regor
{

enum class EthosU85Accumulator
{
    Acc32 = 0,
    Acc48 = 1,
    Acc_Last = Acc48
};

enum class EthosU85Traversal
{
    DepthFirst = 0,
    PartKernel = 1,
    Depthwise = 2,
};

class ArchEthosU85;

enum class EthosU85NpuOp
{
    None = 0,
    Convolution,
    Depthwise,
    VectorProduct,
    Pooling,
    ReduceSum,
    ReduceMinMax,
    ArgMax,
    Elementwise,
    Resize,
    Dma,
};

/// <summary>
/// Per-operator architecture configuration
/// </summary>
class EthosU85OpConfig : public ArchitectureOpConfig
{
    friend class ArchEthosU85;
    friend class EthosU85RCSGenerator;

private:
    Shape _ifmBlock;
    Shape _ofmBlock;
    Shape _ofmUBlock;
    EthosU85Accumulator _accumulatorType = EthosU85Accumulator::Acc32;
    ArchAccumulatorSource _accumulatorSource = ArchAccumulatorSource::Reset;
    bool _accumulatorOutputEnabled = true;
    EthosU85Traversal _traversal = EthosU85Traversal::DepthFirst;
    int _ifmRamSizeBytes = 0;

public:
    EthosU85Traversal Traversal() const { return _traversal; }
    const Shape &IfmBlock() const { return _ifmBlock; }
    const Shape &OfmBlock() const { return _ofmBlock; }
    const Shape &OfmUBlock() const { return _ofmUBlock; }
    EthosU85Accumulator Acc() const { return _accumulatorType; }
    ArchAccumulatorSource AccSource() const { return _accumulatorSource; }
    bool AccOutputEnabled() const { return _accumulatorOutputEnabled; }
    std::unique_ptr<ArchitectureOpConfig> Clone() override;
    int MaxIFMBuffering() override;
    Point2i OptimalStripeGranule() override;
    int OptimalDepthGranule() override;
    std::string ToString(bool full) override;
};

/// <summary>
/// Group of ops that can be fused and/or chained
/// </summary>
class EthosU85OpGroup : public ArchitectureOpGroup
{
    friend class ArchEthosU85;
    friend class EthosU85RCSGenerator;

    using OpInfo = ArchitectureOpGroupQuery;

    struct InternalOpInfo
    {
        std::vector<int> dependsOn;
    };

private:
    ArchEthosU85 *_arch;
    std::array<OpInfo, 8> _ops;
    std::array<InternalOpInfo, 8> _opsInternal;
    std::unordered_map<UniqueId, int> _tensorCbMap;
    std::unordered_set<UniqueId> _fusedTensors;
    const int _maxChainLength = 4;
    const int _maxExternalIfms = 3;  // non-chained IFMs for a chain
    int _opsCount = 0;
    int _chainLength = 0;
    int _externalIfms = 0;
    int _chainIdx = 0;
    bool _supportsChaining = false;
    bool _hasFusedActivation = false;
    bool _hasFusedTranspose = false;
    bool _hasFusedReverse = false;

public:
    EthosU85OpGroup(ArchEthosU85 *arch) : _arch(arch){};
    int Add(const ArchitectureOpGroupQuery &op, const std::vector<int> &dependsOn = {}) override;
    bool NeedsAllocation(UniqueId tensorUID) override;

protected:
    bool CanRunOnNPU(const ArchitectureOpGroupQuery &op) override;
    int ChainingBuffer(UniqueId tensorUID);
    bool IsChained(UniqueId tensorUID);
    bool IsFused(UniqueId tensorUID);

private:
    int KeyToOpIndex(int key);
    int ExternalIfms(const ArchitectureOpGroupQuery &op);
    bool CanStartChain(const ArchitectureOpGroupQuery &op);
    bool Chain(const ArchitectureOpGroupQuery &op, const std::vector<int> &dependsOn, int externalInputs);
    bool Fuse(const ArchitectureOpGroupQuery &op, const std::vector<int> &dependsOn);
};

/// <summary>
/// EthosU85 specialisation
/// </summary>
class ArchEthosU85 : public Architecture
{
    friend class EthosU85WeightEncoder;
    friend class EthosU85Performance;
    friend class EthosU85RCSGenerator;
    friend class EthosU85Constraints;
    friend class EthosU85OpGroup;

public:
    struct AcceleratorConfig
    {
        int macs;
        int cores;
        std::array<Shape, 3> ofmUBlocks;
        Shape ifmUBlock;
        int nOfmUBlocks;
        int ifmRamSizeBytes;
        int accRamSizeBytes;
        int obRamSizeBytes;
        int cbRamSizeBytes;
        uint8_t numAxiSramLog2;
        uint8_t numAxiExtLog2;
        const std::array<int, 5> channelRBs;
        const EthosU85PerfInfo *perfInfo;
    };

private:
    static constexpr int LUT_SLOT_SIZE = 256;
    std::unique_ptr<ArchitectureMemory> _lutRam;
    Shape _subkernelMax;
    Shape _ofmBlockMax;
    int _cores = 0;
    int _macs = 0;
    std::array<Shape, 3> _ofmUBlocks;
    int _nOfmUBlocks = 1;
    // maps ofm microblock and ifmbits to a bitmask of supported operations
    std::array<std::array<unsigned, 3>, 3> _uBlockToOpTable{};
    // maps ofm microblock to supported IFM allocation unit
    std::array<std::array<Shape, 3>, 3> _uBlockToIfmAuTable{};
    Shape _ifmUBlock;
    int _ifmRamSizeBytes = 0;
    int _cbRamSizeBytes = 0;
    int _obRamSizeBytes = 0;
    int _accRamSizeBytes = 0;
    int _numAxiSramLog2 = 0;
    int _numAxiExtLog2 = 0;
    const std::array<int, 5> *_channelRBs{};

protected:
    std::unique_ptr<class WeightEncoder> _weightEncoder;
    std::unique_ptr<ArchitecturePerformance> _performance;
    std::unique_ptr<IRegisterCommandStreamGenerator> _rcsGenerator;
    std::unique_ptr<IArchitectureConstraints> _constraints;

public:
    ArchEthosU85();

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
    void ApplyConfig(const AcceleratorConfig *cfg);

    struct FindConfigCommon
    {
        Shape ofmBlockMax;
        Shape granule;
        Shape ublock;
        int accBits;
        int ifmBlockDepth;
        int ifmBits;
        bool isPooling;
    };

    Shape AreaFit(const FindConfigCommon &common, const Shape &ofmShape, const Shape &ofmBlockLimit,
        const Shape &ifmShape, const Kernel *kernel);
    Shape FindElementwiseConfig(const ArchitectureConfigQuery &query, const FindConfigCommon &common);
    Shape FindDepthwiseConfig(const ArchitectureConfigQuery &query, const FindConfigCommon &common, Shape &ifmBlock);
    std::unique_ptr<ArchitectureOpConfig> FindBlockConfig(OpType opType, const ArchitectureConfigQuery &query);

    bool TryBlockConfig(EthosU85NpuOp npuOp, const Shape &ofmBlock, const Shape &ifmBlock, const Shape &ifmShape,
        int ifmBits, int accBits, int ifmSpace, int accSpace, int ifmAuDepth, int numBlocksInRam, bool isEqualDepthOp);

    Shape GetStorageRounding(TensorFormat format);

    uint32_t ConfigRegister(int product);
    // Checks if the operation is to be mapped on AvgPool
    static bool UseAvgPoolNop(OpType type);
    // Checks if the operation is to be mapped to a NullPool
    static bool UseNullPool(OpType opType, int bits);
    static EthosU85NpuOp GetHWOp(OpType type);

private:
    int MaxOutstandingKernelOps() { return 2; }
    int MaxOutstandingDMAOps() { return 4; }
    int MaxBlockdep() { return 7; }
    bool IsUBlockValid(const OpType opType, int ifmBits, const Shape &ofmUBlock, bool hasIfm2, bool depthFirst1x1);
    Shape FindUBlock(OpType opType, const ArchitectureConfigQuery &query, bool partKernel);
    Shape CalcIfmAUSize(int ifmBlkDepth, int ifmBits, const Shape &ofmUBlk);
    int CalcResizeMaxOfmBlockWidth(int ifmBits, int scaleN, int scaleD);
    int IndexForOfmUBlock(const Shape &ofmUBlock);
    void SetupOfmUBlockToOpTable();
    void SetupOfmUBlockToIfmAuTable();
};

}  // namespace regor
