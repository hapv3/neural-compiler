//
// SPDX-FileCopyrightText: Copyright 2021-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "architecture/weight_encoder.hpp"
#include "cascade_builder.hpp"
#include "common/shape.hpp"
#include "graph.hpp"
#include "live_range.hpp"
#include "quantization.hpp"
#include "scheduler_operation.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace regor
{

class IncrementalLinearAllocator;

enum class OptimizationStrategy
{
    Size,
    Performance,
};

enum class TensorAllocator
{
    // Allocator that does not reuse memory
    LinearAlloc = 0,
    // Search based allocator
    HillClimb = 1,
    Last,
};

enum class SchedulerFeature : uint16_t
{
    WeightBuffering = 1 << 0,
    Cascading = 1 << 1,
    Grouping = 1 << 2,
    FWD = 1 << 3,
    Sparsity = 1 << 4,
    FMStaging = 1 << 5,
    ReuseIFM = 1 << 6,
};

/// <summary>
/// Scheduling options
/// </summary>
struct SchedulerOptions
{
    OptimizationStrategy optimizationStrategy = OptimizationStrategy::Size;
    Address optimizationStagingLimit = 0;
    bool verboseSchedule = false;
    bool verboseAllocation = false;
    Flags<SchedulerFeature> disabled;
    bool separateIORegions = false;
    int cpuTensorAlignment = 16;
    TensorAllocator tensorAllocator = TensorAllocator::HillClimb;
};

struct WeightScaleEncoding
{
    std::unique_ptr<ArchitectureOpConfig> blockConfig;
    WeightScaleTensors weightScales;
    // Keep track of op cycles - used in ChooseBestWeightFormat
    CycleCost cycleCost;
};

struct SchedulerBufferTensor
{
    std::shared_ptr<SchedulerTensor> tensor[2];
    unsigned parts = 0;
    bool preBuffer = false;
    Buffering buffering = Buffering::None;

    Address AllocatedSize() const
    {
        Address size = tensor[0] ? tensor[0]->AllocationSizeBytes() : 0;
        if ( parts > 1 && tensor[1] ) size += tensor[1]->AllocationSizeBytes();
        return size;
    }
};


struct EstimatedPerf
{
    int64_t weightReadBytes;   // Total number of weight bytes read
    int64_t ifmReadBytes;      // Total number of ifm bytes read
    int64_t opRunCycles;       // Op run cycles without buffering
    int64_t fullRunCycles;     // Op run cycles with buffering
    int64_t ifmReadCycles;     // Total cycles spent doing required ifm reads
    int64_t weightReadCycles;  // Total cycles spent doing required weights reads
    int64_t lastSliceCycles;   // Cycles taken when executing the last slice
    int visibleBufferCycles;   // Leftover visible buffering cycles
    int ifmSizeBytes;          // Size of the primary IFM in bytes
    bool OpDominated() const { return opRunCycles > weightReadCycles && opRunCycles > ifmReadCycles; }
    double IfmRatio() const { return weightReadCycles ? double(ifmReadCycles) / weightReadCycles : 1.0; }
};

enum class StagingPref
{
    None = 0x00,
    IFM = 0x01,
    Weights = 0x02,
    OFM = 0x04,
};

/// <summary>
/// Metadata for each scheduled operation (unique per schedule)
/// </summary>
class SchedulerOpInfo
{
private:
    std::unique_ptr<ArchitectureOpConfig> _config;

public:
    // Uses primary-IFM order: [0] is the stripe for IFM(PrimaryIfmIndex()), [1] is the stripe for any other IFM.
    Shape stripeInput[2];
    Shape stripe;
    int cascade = 0;
    // Setting this to the UID of the IFM signals reuse.
    UniqueId ofmEquivalenceId = INVALID_UID;
    bool firstInCascade = false;
    int timeIndex = -1;
    std::vector<int> ofmDepthSlices;
    int64_t slackBufferingCycles = 0;
    int slackBufferingMemory = 0;
    int64_t fullWeightTransferCycles = 0;
    // Encoded weights in readonly memory
    std::shared_ptr<NpuWeightTensor> npuWeightsTensor;
    // Encoded scales in readonly memory
    std::shared_ptr<NpuWeightTensor> npuScalesTensor;
    // Buffered weights/scales in fast storage
    SchedulerBufferTensor bufferedWeightTensor;
    CycleCost cycles;
    ElementAccess elementAccess;
    EstimatedPerf perf{};
    Flags<StagingPref> stagingPreference;

public:
    SchedulerOpInfo(std::unique_ptr<ArchitectureOpConfig> opConfig, const Shape &stripeInput1, const Shape &stripeInput2, const Shape &stripe_)
    {
        this->_config = std::move(opConfig);
        this->stripeInput[0] = stripeInput1;
        this->stripeInput[1] = stripeInput2;
        this->stripe = stripe_;

        // Create default single depth slice that covers the whole stripe depth. It may be updated later.
        this->ofmDepthSlices = {0, (stripe_.Size() > 0 ? stripe.Depth() : 0)};
    }

    SchedulerOpInfo(const SchedulerOpInfo &other) { Copy(other); }

    const SchedulerOpInfo &operator=(const SchedulerOpInfo &other)
    {
        Copy(other);
        return *this;
    }

    void SetWeightScaleTensors(const std::shared_ptr<NpuWeightTensor> &weights, const std::shared_ptr<NpuWeightTensor> &scales)
    {
        npuWeightsTensor = weights;
        npuScalesTensor = scales;
    }

    bool SourcesCascadeBuffer() const { return this->cascade != 0 && !this->firstInCascade; }

    ArchitectureOpConfig *Config() const { return _config.get(); }

    std::string ToString() const
    {
        std::string temp = fmt::format("\t\tTime index = {0}\n", this->timeIndex);
        temp += fmt::format(
            "\t\tOperator Config = {0}\n"
            "\t\tIFM Stripe   = [{1}]\n"
            "\t\tIFM2 Stripe  = [{2}]\n"
            "\t\tOFM Stripe   = [{3}]\n",
            _config ? _config->ToString(false) : "no config", stripeInput[0].ToString(), stripeInput[1].ToString(),
            stripe.ToString());

        temp += fmt::format("\t\tAssigned Cascade = {0}", this->cascade);

        if ( npuWeightsTensor )
        {
            // TODO: Finish formatting;
            temp += fmt::format(
                "\n\t\tEncoded Weights = {0} bytes\n"
                "\t\tWeight buffer = {1} bytes ({3})\n"
                "\t\tDepth slices = [{2}]",
                npuWeightsTensor->AllocationSizeBytes(), bufferedWeightTensor.parts ? bufferedWeightTensor.AllocatedSize() : 0,
                fmt::join(ofmDepthSlices, ", "), EnumToString(bufferedWeightTensor.buffering));
        }

        return temp;
    }

private:
    void Copy(const SchedulerOpInfo &other)
    {
        if ( other._config )
        {
            // Must duplicate (can't be auto-generated)
            _config = other._config->Clone();
        }

        // Potentially generatable
        stripeInput[0] = other.stripeInput[0];
        stripeInput[1] = other.stripeInput[1];
        stripe = other.stripe;
        cascade = other.cascade;
        ofmEquivalenceId = other.ofmEquivalenceId;
        timeIndex = other.timeIndex;
        firstInCascade = other.firstInCascade;
        ofmDepthSlices = other.ofmDepthSlices;
        slackBufferingCycles = other.slackBufferingCycles;
        slackBufferingMemory = other.slackBufferingMemory;
        fullWeightTransferCycles = other.fullWeightTransferCycles;
        npuWeightsTensor = other.npuWeightsTensor;
        npuScalesTensor = other.npuScalesTensor;
        bufferedWeightTensor = other.bufferedWeightTensor;
        cycles = other.cycles;
        elementAccess = other.elementAccess;
        perf = other.perf;
    }
};


using SchedulerCostMap = std::unordered_map<UniqueId, std::unique_ptr<SchedulerOpInfo>>;
using SchedulerOpConfigMap = std::unordered_map<UniqueId, std::shared_ptr<ArchitectureOpConfig>>;

/// <summary>
/// Individual schedule
/// </summary>
class Schedule
{
private:
    std::string _name;
    SchedulerCostMap _costMap;
    int _subScheduleStart = 0;  // Start index in external ops array
    int _subScheduleEnd = 0;    // Exclusive end index in external ops array

public:
    std::unordered_map<int, CascadeInfo> cascades;
    MemorySnapshot memorySnapshot;
    int fastStoragePeakUsage = 0;
    std::unordered_map<MemArea, int, MemArea::hash> memoryUsage;
    std::unique_ptr<LiveRangeGraph> featureMapLRGraph;
    std::unique_ptr<LiveRangeGraph> stagingLRGraph;

public:
    Schedule(const std::string &name, int startIndex, unsigned endIndex) :
            _name(name), _subScheduleStart(startIndex), _subScheduleEnd(int(endIndex))
    {
        assert(_subScheduleStart >= 0 && _subScheduleEnd >= _subScheduleStart);
    }

    const std::string &Name() const { return _name; }
    int Start() const { return _subScheduleStart; }
    int End() const { return _subScheduleEnd; }

    void SetCost(UniqueId id, std::unique_ptr<SchedulerOpInfo> opInfo) { _costMap[id] = std::move(opInfo); }

    SchedulerOpInfo *Cost(const SchedulerOperation *op) const { return op ? Cost(*op) : nullptr; }
    SchedulerOpInfo *Cost(UniqueId id) const
    {
        auto pos = _costMap.find(id);
        return (pos != _costMap.end()) ? pos->second.get() : nullptr;
    }

    const SchedulerCostMap &Costs() const { return _costMap; }

    const LRMemory &MemoryAt(int timeIndex) const;

    int MemoryUsageAt(int timeIndex) const
    {
        return (timeIndex >= 0 && timeIndex < int(memorySnapshot.size())) ? memorySnapshot[timeIndex].Used() : 0;
    }

    void DetachCosts(SchedulerCostMap &costs) { costs = std::move(_costMap); }

    void UpdateCosts(SchedulerCostMap &costs)
    {
        for ( auto &pos : costs )
        {
            _costMap[pos.first] = std::move(pos.second);
        }
    }

    void UpdateCascades(const std::unordered_map<int, CascadeInfo> &other)
    {
        cascades.insert(other.begin(), other.end());
    }

    const CascadeInfo *Cascade(int cascade) const
    {
        auto it = cascades.find(cascade);
        return it == cascades.end() ? nullptr : &it->second;
    }
};


/// <summary>
/// Executable scheduling implementation
/// </summary>
class Scheduler
{
    struct TensorCacheKey
    {
    private:
        IWeightEncodingConfig *_config;  // must persist as map entry
        uint32_t _hash;
        UniqueId _uid;
        int _weightDepthBase;

    public:
        TensorCacheKey(IWeightEncodingConfig *config, int weightDepthBase, const std::vector<int> &depthOffsets, const BufferView &view, UniqueId uid) :
                _config(config), _uid(uid), _weightDepthBase(weightDepthBase)
        {
            _hash = SimpleHash32(weightDepthBase, HashVector32(depthOffsets), view.BaseOffset(), view.StrideBytes(),
                view.ViewShape(), _config->Hash(), uid);
        }

        bool operator==(const TensorCacheKey &other) const
        {
            return _config->Equals(other._config) && (_uid == other._uid) && (_hash == other._hash) &&
                   (_weightDepthBase == other._weightDepthBase);
        }
        uint32_t Hash() const { return _hash; }
    };

    struct TensorCacheHash
    {
        std::size_t operator()(const TensorCacheKey &key) const { return key.Hash(); }
    };

private:
    Architecture *_arch = nullptr;
    SchedulerOptions _options;
    std::string _name;
    std::vector<std::unique_ptr<SchedulerOperation>> &_ops;
    std::shared_ptr<Schedule> _maxSchedule;
    int _minMemoryRequired = 0;
    bool _spilling = false;
    std::unordered_map<TensorCacheKey, WeightScaleTensors, TensorCacheHash> _tensorCache;
    std::unordered_map<Hash128, UniqueId> _equivalenceIdMap;
    const SchedulerOpConfigMap &_opConfigCompatablility;

public:
    Scheduler(Architecture *arch, const SchedulerOptions &options, const std::string &name,
        std::vector<std::unique_ptr<SchedulerOperation>> &ops, const SchedulerOpConfigMap &opConfigCompatablility);

public:
    std::shared_ptr<Schedule> Process();

    static std::unique_ptr<Graph> ToGraph(std::vector<std::unique_ptr<SchedulerOperation>> &ops,
        TensorAddressMap &tensorAddressMap, const Graph *srcGraph);

    void AllocateReadOnlyAddresses(Schedule *schedule, IncrementalLinearAllocator &readOnlyAllocator);

    void AllocateIOAddresses(Schedule *schedule, const std::vector<std::unique_ptr<SchedulerOperation>> &ops);

    static PerformanceQuery InitPerfQuery(const SchedulerOperation *op, ArchitectureOpConfig *config, int ofm_depth = -1,
        WeightFormat wgtFormat = WeightFormat::Default, const SchedulerOpInfo *cost = nullptr, Schedule *schedule = nullptr);

private:
    Shape AlignStripe(const SchedulerOperation *schedOp, const Shape &stripe);

    int UpdateSchedulerTensor(TensorUsage usage, SchedulerConnection *conn, std::unordered_set<UniqueId> &visited);

    Address CreateSchedulerRepresentation();

    Point2i GetStripeInputRequirement(const Shape &ofmShape, const Kernel *kernel, const Point2i &ifmStep, ArchResampling resampling);

    std::unique_ptr<SchedulerOpInfo> CreateSchedulerOpInfo(SchedulerOperation *op, const Shape &ofmStripeShape,
        const std::unique_ptr<SchedulerOpInfo> &parentInfo = nullptr);

    std::unique_ptr<Schedule> CreateInitialSchedule();

    void MoveConstantData(Schedule *refSchedule);

    bool AllocateAddresses(Schedule *schedule);

    void UpdateOpMemorySnapshot(Schedule *schedule, LiveRangeGraph *liveRanges = nullptr);

    void PopulateLiveRanges(const LiveRangeGraph &lrGraph, std::unordered_map<UniqueId, LiveRangeSummary> *liveRanges) const;

    int ComputeLocalMemUsage(const SchedulerOperation &schedOp, const SchedulerOpInfo &cost,
        const LiveRangeGraph &lrGraph, const MemArea &stagingMemory) const;

    std::unordered_map<UniqueId, int> ComputeNonLocalUsage(Schedule *schedule,
        std::unordered_map<UniqueId, LiveRangeSummary> *liveRanges = nullptr, std::unordered_map<UniqueId, int> *opLocalMemUsage = nullptr);

    std::shared_ptr<Schedule> ProposeScheduleBuffering(Schedule *refSchedule, Address stagingLimitBytes);

    void ProposeOperatorBuffering(SchedulerOperation *schedOp, SchedulerOperation *prevOp, Schedule *bufferedSchedule,
        Schedule *refSchedule, int stagingLimitBytes);

    Buffering ProposeWeightBuffering(SchedulerConnection *weights, SchedulerConnection *scales, SchedulerOperation *schedOp,
        SchedulerOperation *prevOp, Schedule *bufferedSchedule, Schedule *refSchedule, int bufferLimitBytes);

    bool ProposeSlicedWeightBuffering(SchedulerConnection *weights, SchedulerConnection *scales, SchedulerOperation *schedOp,
        SchedulerOperation *prevOp, Schedule *bufferedSchedule, Schedule *refSchedule, int bufferLimitBytes, int untransposedFullDepth);

    std::shared_ptr<Schedule> ProposeMinimalSchedule();

    std::shared_ptr<Schedule> OptimizeSchedule(Schedule *schedule, const std::shared_ptr<Schedule> &maxSchedule, Address stagingLimitBytes);

    std::shared_ptr<Schedule> ProposeScheduleStriping(const Shape &finalStripe, const std::string &label, Schedule *refSchedule);

    Address EstimateScheduleMemoryUsage(Schedule *schedule, const std::unordered_map<UniqueId, int> &nonLocalMem);

    std::shared_ptr<Schedule> OptimizeSubSchedule(const CascadeInfo &cascadeInfo, Schedule *refSchedule, Address stagingLimitBytes);

    void ApplySchedule(Schedule *schedule);

    void CoalesceWeightBufferTensors(Schedule *schedule);

    CycleCost EstimateOpPerformance(SchedulerOperation *op, ArchitectureOpConfig *config, int ofm_depth,
        WeightFormat wgtFormat = WeightFormat::Default, ArchitectureMemory *wgtStaging = nullptr,
        OpScheduling scheduling = OpScheduling::Single);

    EstimatedPerf EstimateSlicedOpPerformance(
        SchedulerOperation *schedOp, SchedulerOpInfo *cost, const Point2i stripe, int slackCycles, Buffering buffering);

    EstimatedPerf EstimateSlicedOpPerformance(SchedulerOperation *schedOp, const std::vector<int> &depthSlices, ArchitectureOpConfig *opConfig,
        NpuWeightTensor *weights, Flags<StagingPref> stageFlags, const Point2i stripe, int slackCycles, Buffering buffering);

    void PrintSchedule(Schedule *schedule);

    WeightScaleTensors EncodeQuantizationScaleTensor(OpType forOp, std::unique_ptr<IWeightEncodingConfig> encodingParams, int weightDepthBase,
        const std::vector<int> &depthOffsets, const Quantization &ofmQuantization, const SchedulerTensor *scales = nullptr);

    WeightScaleTensors EncodeWeightAndScaleTensor(OpType forOp, std::unique_ptr<IWeightEncodingConfig> encodingParams,
        int weightDepthBase, const std::vector<int> &depthOffsets, const SchedulerTensor *weightTens,
        const SchedulerTensor *scaleTens, const Quantization &weightQuantization, const Quantization &ofmQuantization);

    WeightScaleTensors TryEncodeWeightAndScaleTensor(OpType forOp, IWeightEncodingConfig *encodingParams, int weightDepthBase,
        const std::vector<int> &depthOffsets, const SchedulerTensor *weightTens, const SchedulerTensor *scaleTens,
        const Quantization &weightQuantization, const Quantization &ofmQuantization, bool doWeights, bool doScales);

    std::unique_ptr<ArchitectureOpConfig> GetOpConfig(SchedulerOperation *op, const Shape &ifmShape,
        const Shape &ifm2Shape, const Shape &ofmShape, WeightFormat wgtFormat);

    WeightScaleEncoding ChooseBestWeightFormat(SchedulerOperation *op, OptimizationStrategy optimizationStrategy,
        std::vector<WeightScaleEncoding> &encodingResults);

    bool UseFastDecoder(SchedulerOperation *op, OptimizationStrategy optimizationStrategy, NpuWeightTensor *weightTensor);

    std::unique_ptr<ArchitectureOpConfig> MaybeGetSparsityConfig(SchedulerOperation *op, Shape &ifmShape,
        Shape &ifm2Shape, Shape &ofmShape, Flags<WeightFormat> supportedFormat);

    WeightScaleEncoding EncodeBestWeightFormat(SchedulerOperation *op, Shape &ifmShape, Shape &ifm2Shape,
        Shape &ofmShape, Flags<WeightFormat> supportedFormats);
};

bool ParseSchedulerOptions(SchedulerOptions &opt, IniReader &reader, const Architecture *arch);

}  // namespace regor
