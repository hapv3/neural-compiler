//
// SPDX-FileCopyrightText: Copyright 2021-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/bit_flags.hpp"
#include "common/data_type.hpp"
#include "common/ini_reader.hpp"
#include "common/numeric_util.hpp"
#include "common/reverse_type.hpp"
#include "common/scaling.hpp"
#include "common/shape.hpp"
#include "common/transpose_type.hpp"
#include "compiler/database.hpp"
#include "compiler/kernel.hpp"
#include "compiler/op_type.hpp"
#include "compiler/tensor_properties.hpp"
#include "mlw_encode.hpp"

#include <memory>
#include <string>
#include <unordered_map>
namespace regor
{

class WeightEncoder;
class IRegisterCommandStreamGenerator;
class IArchitectureConstraints;

using Address = int64_t;

enum class TensorFormat : uint16_t
{
    Unknown = 0,
    NHWC = 1,
    NHCWB16 = 2,
    WeightsEncoded = 3,
};

/// <summary>
/// Architecture connected-memory definition
/// </summary>
struct ArchitectureMemory
{
protected:
    std::string _name;
    Address _sizeBytes = 0;
    float _bandwidthPerCycle = 1;
    int _readLatencyCycles = 0;
    int _writeLatencyCycles = 0;
    int _maxBurstLengthBytes = 0;
    int _portsUsed = 1;
    int _maxReads = 0;
    int _maxWrites = 0;

public:
    ArchitectureMemory(const std::string &name, Address sizeBytes) : _name(name), _sizeBytes(sizeBytes) {}

public:
    void SetParameters(float bandwidth, int readLatency, int writeLatency, int maxBurstLengthBytes, int portsUsed, int maxReads, int maxWrites)
    {
        _bandwidthPerCycle = bandwidth;
        _readLatencyCycles = readLatency;
        _writeLatencyCycles = writeLatency;
        _maxBurstLengthBytes = maxBurstLengthBytes;
        _portsUsed = portsUsed;
        _maxReads = maxReads;
        _maxWrites = maxWrites;
    }

    float Bandwidth() const { return _bandwidthPerCycle; }
    int ReadLatency() const { return _readLatencyCycles; }
    int WriteLatency() const { return _writeLatencyCycles; }
    Address SizeBytes() const { return _sizeBytes; }
    int MaxBurstLength() const { return _maxBurstLengthBytes; }
    int PortsUsed() const { return _portsUsed; }
    int MaxReads() const { return _maxReads; }
    int MaxWrites() const { return _maxWrites; }
    std::string Name() const { return _name; }
};

enum class MemUsage : uint16_t
{
    None = 0,
    ReadOnly = 0x1,
    FeatureMap = 0x2,
    LUT = 0x4,
    Staging = 0x8,
};

struct MemArea
{
    ArchitectureMemory *memory = nullptr;  // The physical memory area
    Flags<MemUsage> usage;                 // Usage partition within the memory area

    MemArea() = default;
    MemArea(ArchitectureMemory *architectureMemory, MemUsage memUsage) : memory(architectureMemory), usage(memUsage) {}

    bool operator==(const MemArea &other) const { return memory == other.memory && usage == other.usage; }
    bool operator!=(const MemArea &other) const { return !operator==(other); }
    explicit operator bool() const { return (memory != nullptr) && (usage != MemUsage::None); }

    struct hash
    {
        size_t operator()(const MemArea &memArea) const { return size_t(memArea.memory) | unsigned(memArea.usage); }
    };
};

/// <summary>
/// Per-operator architecture configuration base
/// </summary>
class ArchitectureOpConfig
{
public:
    virtual ~ArchitectureOpConfig() = default;
    virtual std::unique_ptr<ArchitectureOpConfig> Clone() = 0;
    virtual int MaxIFMBuffering() = 0;
    virtual Point2i OptimalStripeGranule() = 0;
    virtual int OptimalDepthGranule() = 0;
    virtual std::string ToString(bool full) = 0;
};

enum class ArchResampling : uint8_t
{
    None = 0,
    Nearest = 1,
    Zeros = 2,
};

enum class ArchResizeMode : uint8_t
{
    Nearest = 0,
    Bilinear = 1,
    Replicate = 2,
};

/// <summary>
/// Description of a candidate op to add to a ArchitectureOpGroup
/// </summary>
struct ArchitectureOpGroupQuery
{
    struct TensorInfo
    {
        UniqueId key;
        DataType type;
        Shape shape;
        TransposeType transpose;
        ReverseType reverse;
        bool isConst;
    };

    OpType type;
    const Kernel *kernel;
    std::array<TensorInfo, 2> ifm;
    TensorInfo ofm;
    int inputs;
};

/// <summary>
/// Group of ops that can be fused and/or chained
/// </summary>
class ArchitectureOpGroup
{
public:
    virtual ~ArchitectureOpGroup() = default;
    virtual int Add(const ArchitectureOpGroupQuery &op, const std::vector<int> &dependsOn = {}) = 0;
    virtual bool NeedsAllocation(UniqueId tensorUID) = 0;

protected:
    virtual bool CanRunOnNPU(const ArchitectureOpGroupQuery &op) = 0;
};

enum class ArchAccumulatorSource : uint8_t
{
    Reset = 0,
    Acc = 1,
    Ifm2 = 2
};

/// <summary>
/// Query Information to retrieve HW specific operator config
/// </summary>
struct ArchitectureConfigQuery
{
    Shape ofmShape;
    Shape ifmShape[2];
    int ifmBits;
    int ofmBits;
    Kernel *kernel;
    int lutBytes;
    bool scaled;
    ArchResampling ifmResampling;
    TransposeType transpose;
    ReverseType reverse;
    TensorFormat ofmFormat;
    WeightFormat weightFormat;
    ArchAccumulatorSource accSource = ArchAccumulatorSource::Reset;
    bool accOutputEnabled = true;
    struct Rescale
    {
        Fraction<int> scaleY{1, 1};
        Fraction<int> scaleX{1, 1};
    } rescaling;
};

/// <summary>
/// Information for querying operation performance
/// </summary>
struct PerformanceQuery
{
    OpType type;
    Kernel *kernel;
    ArchitectureOpConfig *config;
    Shape ifmShape[2];
    ArchitectureMemory *ifmMemory[2];
    DataType ifmType[2];
    TensorFormat ifmFormat[2];
    Shape ofmShape;
    ArchitectureMemory *ofmMemory;
    DataType ofmType;
    TensorFormat ofmFormat;
    Shape constShape;
    ArchitectureMemory *constMemory;
    WeightFormat weightFormat;
};

struct WeightStats
{
    size_t size;
    size_t encodedSize;
    size_t zeroCount;
    int distinctWeights;
};

/// <summary>
/// Information for querying performance for HW fused operations
/// </summary>
struct FusionQuery
{
    OpType type;
    Kernel *kernel = nullptr;
    Shape ifm2Shape;
    ArchitectureMemory *ifm2Memory = nullptr;
    DataType ifm2Type;
    TensorFormat ifm2Format;
};

/// <summary>
/// Cycle cost of performing an operation
/// </summary>
struct CycleCost
{
    int64_t opCycles = 0;
    int64_t macs = 0;
};

/// <summary>
/// How elements are accessed during an operation
/// </summary>
struct ElementAccess
{
    int ifmRead[2] = {0, 0};
    int ofmWrite = 0;
    int weightsRefetch = 0;
    int constRead[2] = {0, 0};
};

enum class MemChannel
{
    Mem2Mem = 0,
    IFMStream = 1,
    Weight = 2,
    FastWeight = 3,
    IFM = 4,
    OFM = 5,
};

/// <summary>
/// Architecture performance interface
/// </summary>
class ArchitecturePerformance
{
public:
    virtual ~ArchitecturePerformance() = default;
    virtual CycleCost MeasureCycleCost(const PerformanceQuery &query, const std::vector<FusionQuery> &fused) = 0;
    virtual CycleCost MeasureCycleCostForSparsity(const PerformanceQuery &query, const std::vector<FusionQuery> &fused) = 0;
    virtual int64_t MemToMemCycles(const ArchitectureMemory *dest, const ArchitectureMemory *source, int sizeBytes) = 0;
    virtual ElementAccess MeasureElementAccess(const PerformanceQuery &query) = 0;
    virtual ElementAccess ElementTransferToBytes(const PerformanceQuery &query, const ElementAccess &access) = 0;
    virtual int64_t WeightDecodeCycles(const PerformanceQuery &query, const WeightStats &weights,
        Flags<WeightFormat> format, ArchitectureMemory *weightsMemory) = 0;
    virtual float ChannelBW(const ArchitectureMemory *mem, MemChannel channel) = 0;
    virtual void InitDatabase(Database *db) = 0;
    virtual void RecordToDB(int opId) = 0;
};

enum class IniParseResult
{
    Unknown = 0,
    Done,
    Error,
};

enum class AxisMask
{
    None = 0,
    AxisX = 1,
    AxisY = 2,
};

/// <summary>
/// ArchitectureFeatures base
/// </summary>
class Architecture
{
protected:
    std::unordered_map<std::string, std::unique_ptr<ArchitectureMemory>> _memories;
    ArchitectureMemory *_readonlyMemory = nullptr;
    ArchitectureMemory *_featuremapMemory = nullptr;
    ArchitectureMemory *_lutMemory = nullptr;
    ArchitectureMemory *_stagingMemory = nullptr;

public:
    virtual ~Architecture() = default;
    virtual bool ParseConfig(IniReader *reader) = 0;
    virtual bool CheckConfiguration(std::string &error);
    virtual std::unique_ptr<ArchitectureOpConfig> GetOpConfig(OpType opType, const ArchitectureConfigQuery &query) = 0;
    virtual std::unique_ptr<ArchitectureOpGroup> CreateOpGroup(const ArchitectureOpGroupQuery &op) = 0;
    virtual class WeightEncoder *WeightEncoder() = 0;
    virtual ArchitecturePerformance *Performance() = 0;
    virtual IRegisterCommandStreamGenerator *RegisterCommandStreamGenerator() = 0;
    virtual IArchitectureConstraints *Constraints() = 0;
    virtual TensorFormat IdealBufferingFormat() { return TensorFormat::Unknown; }
    virtual Address MaxAddress() = 0;
    virtual std::vector<uint32_t> ConfigRegisters() = 0;
    virtual uint32_t Version() = 0;
    virtual int UpscaleAndRounding(ArchResampling resampling, int &rounding) = 0;
    virtual AxisMask CanSubdivide(OpType opType, TransposeType transpose, ReverseType reverse) = 0;
    virtual bool SupportsScalar(OpType opType, DataType dataType, TensorUsage usage) = 0;
    virtual Flags<WeightFormat> SupportedWeightFormat(OpType op) = 0;

    MemArea ReadonlyMemory()
    {
        assert(_readonlyMemory);
        return MemArea(_readonlyMemory, MemUsage::ReadOnly);
    }
    MemArea FeatureMapMemory()
    {
        assert(_featuremapMemory);
        Flags<MemUsage> usage = MemUsage::FeatureMap;
        if ( _featuremapMemory == _stagingMemory )
        {
            usage |= MemUsage::Staging;
        }
        return MemArea(_featuremapMemory, usage);
    }
    MemArea LUTMemory()
    {
        assert(_lutMemory);
        return MemArea(_lutMemory, MemUsage::LUT);
    }
    MemArea StagingMemory()
    {
        assert(_stagingMemory);
        Flags<MemUsage> usage = MemUsage::Staging;
        if ( _featuremapMemory == _stagingMemory )
        {
            usage |= MemUsage::FeatureMap;
        }
        return MemArea(_stagingMemory, usage);
    }

    IniParseResult ParseSection(const std::string &section, IniReader *reader);
    // Select named memories
    void SetReadonlyMemory(const std::string &name) { _readonlyMemory = _memories.at(name).get(); }
    void SetFeatureMapMemory(const std::string &name) { _featuremapMemory = _memories.at(name).get(); }
    void SetStagingMemory(const std::string &name) { _stagingMemory = _memories.at(name).get(); }

private:
    bool ParseMemory(IniReader *reader, const std::string &section);
};

}  // namespace regor
