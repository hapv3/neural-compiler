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

#include "architecture/architecture.hpp"
#include "architecture/weight_encoder.hpp"
#include "cascade_builder.hpp"
#include "common/shape.hpp"
#include "graph.hpp"
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

enum class SchedulerFeature : uint16_t
{
    WeightBuffering = 1 << 0,
    Cascading = 1 << 1,
    Grouping = 1 << 2,
    FWD = 1 << 3,
    Sparsity = 1 << 4,
    FMStaging = 1 << 5,
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
};

struct WeightScaleEncoding
{
    std::unique_ptr<ArchitectureOpConfig> blockConfig;
    WeightScaleTensors weightScales;
};

struct SchedulerBufferTensor
{
    std::shared_ptr<SchedulerTensor> tensor;
    bool preBuffer = false;
    Buffering buffering = Buffering::None;
};

/// <summary>
/// Metadata for each scheduled operation (unique per schedule)
/// </summary>
class SchedulerOpInfo
{
private:
    std::unique_ptr<ArchitectureOpConfig> _config;

public:
    Shape stripeInput[2];
    Shape stripe;
    int cascade = 0;
    int timeIndex = -1;
    int weightSize = 0;
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

public:
    SchedulerOpInfo(std::unique_ptr<ArchitectureOpConfig> opConfig, const Shape &stripeInput1, const Shape &stripeInput2, const Shape &stripe_)
    {
        this->_config = std::move(opConfig);
        this->stripeInput[0] = stripeInput1;
        this->stripeInput[1] = stripeInput2;
        this->stripe = stripe_;
        this->ofmDepthSlices = {0, stripe_.Size() > 0 ? stripe.Depth() : 0};
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
                "\t\tWeight buffer = {1} bytes\n"
                "\t\tDepth slices = [{2}]",
                npuWeightsTensor->AllocationSizeBytes(),
                bufferedWeightTensor.tensor ? bufferedWeightTensor.tensor->AllocationSizeBytes() : 0, fmt::join(ofmDepthSlices, ", "));
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
        timeIndex = other.timeIndex;
        weightSize = other.weightSize;
        ofmDepthSlices = other.ofmDepthSlices;
        slackBufferingCycles = other.slackBufferingCycles;
        slackBufferingMemory = other.slackBufferingMemory;
        fullWeightTransferCycles = other.fullWeightTransferCycles;
        npuWeightsTensor = other.npuWeightsTensor;
        npuScalesTensor = other.npuScalesTensor;
        bufferedWeightTensor = other.bufferedWeightTensor;
        cycles = other.cycles;
        elementAccess = other.elementAccess;
    }
};


using SchedulerCostMap = std::unordered_map<UniqueId, std::unique_ptr<SchedulerOpInfo>>;

/// <summary>
/// Individual schedule
/// </summary>
class Schedule
{
private:
    std::string _name;
    SchedulerCostMap _costMap;

public:
    std::unordered_map<int, CascadeInfo> cascades;
    std::vector<int> memorySnapshot;
    int fastStoragePeakUsage = 0;
    std::unordered_map<MemArea, int, MemArea::hash> memoryUsage;

public:
    Schedule(const std::string &name) : _name(name) {}

    const std::string &Name() const { return _name; }

    void SetCost(UniqueId id, std::unique_ptr<SchedulerOpInfo> opInfo) { _costMap[id] = std::move(opInfo); }

    SchedulerOpInfo *Cost(const SchedulerOperation *op) const { return op ? Cost(*op) : nullptr; }
    SchedulerOpInfo *Cost(UniqueId id) const
    {
        auto pos = _costMap.find(id);
        return (pos != _costMap.end()) ? pos->second.get() : nullptr;
    }

    const SchedulerCostMap &Costs() const { return _costMap; }

    int MemoryUsageAt(int timeIndex) const
    {
        return (timeIndex >= 0 && timeIndex < int(memorySnapshot.size())) ? memorySnapshot[timeIndex] : 0;
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
    public:
        IWeightEncodingConfig *_config;  // must persist as map entry
        UniqueId _uid;

    public:
        TensorCacheKey(IWeightEncodingConfig *config, UniqueId uid) : _config(config), _uid(uid) {}

        bool operator==(const TensorCacheKey &other) const
        {
            return _config->Equals(other._config) && _uid == other._uid;
        }
    };

    struct TensorCacheHash
    {
        std::size_t operator()(const TensorCacheKey &key) const
        {
            return key._config->Hash() + 37 * std::uintptr_t(key._uid);
        }
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

public:
    Scheduler(Architecture *arch, const SchedulerOptions &options, const std::string &name,
        std::vector<std::unique_ptr<SchedulerOperation>> &ops);

public:
    std::shared_ptr<Schedule> Process();

    static std::unique_ptr<Graph> ToGraph(std::vector<std::unique_ptr<SchedulerOperation>> &ops,
        std::unordered_map<const Tensor *, Address> &tensorAddressMap, const Graph *srcGraph);

    void AllocateReadOnlyAddresses(Schedule *schedule, IncrementalLinearAllocator &readOnlyAllocator);

    static PerformanceQuery InitPerfQuery(SchedulerOperation *op, ArchitectureOpConfig *config, int ofm_depth = -1,
        WeightFormat wgtFormat = WeightFormat::Default);
    static std::vector<FusionQuery> InitFusionQuery(SchedulerOperation *op);

private:
    Address CreateSchedulerRepresentation();

    Point2i GetStripeInputRequirement(const Shape &ofmShape, Kernel *kernel, const Point2i &ifmStep, ArchResampling resampling);

    std::unique_ptr<SchedulerOpInfo> CreateSchedulerOpInfo(SchedulerOperation *op, const Shape &ofmStripeShape,
        const std::unique_ptr<SchedulerOpInfo> &parentInfo = nullptr);

    std::unique_ptr<Schedule> CreateInitialSchedule();

    void MoveConstantData(Schedule *refSchedule);

    bool AllocateAddresses(Schedule *schedule);

    void UpdateOpMemorySnapshot(Schedule *schedule);

    std::shared_ptr<Schedule> ProposeScheduleBuffering(Schedule *refSchedule, Address stagingLimitBytes);

    void ProposeOperatorBuffering(SchedulerOperation *schedOp, SchedulerOperation *prevOp, Schedule *bufferedSchedule,
        Schedule *refSchedule, int stagingLimitBytes);

    void ProposeWeightBuffering(SchedulerConnection *weights, SchedulerConnection *scales, SchedulerOperation *schedOp,
        SchedulerOperation *prevOp, Schedule *bufferedSchedule, Schedule *refSchedule, int bufferLimitBytes);

    std::shared_ptr<Schedule> ProposeMinimalSchedule();

    std::shared_ptr<Schedule> OptimizeSchedule(Schedule *schedule, const std::shared_ptr<Schedule> &maxSchedule);

    std::shared_ptr<Schedule> ProposeScheduleStriping(const Shape &finalStripe, const std::string &label, Schedule *refSchedule);

    Address EstimateScheduleMemoryUsage(Schedule *schedule, const std::unordered_map<UniqueId, int> &nonLocalMem);

    std::shared_ptr<Schedule> OptimizeSubSchedule(const CascadeInfo &cascadeInfo, Schedule *refSchedule, Address stagingLimitBytes);

    void ApplySchedule(Schedule *schedule);

    void CoalesceWeightBufferTensors(Schedule *schedule);

    CycleCost EstimateOpPerformance(SchedulerOperation *op, ArchitectureOpConfig *config, int ofm_depth);

    CycleCost EstimateOpPerformanceForSparsity(SchedulerOperation *op, ArchitectureOpConfig *config, int ofm_depth);

    ElementAccess EstimateOpElementAccess(SchedulerOperation *op, ArchitectureOpConfig *config, int ofm_depth);

    void PrintSchedule(Schedule *schedule);

    WeightScaleTensors EncodeQuantizationScaleTensor(std::unique_ptr<IWeightEncodingConfig> encodingParams,
        const Quantization &ofmQuantization, const SchedulerTensor *scales = nullptr);

    WeightScaleTensors EncodeWeightAndScaleTensor(std::unique_ptr<IWeightEncodingConfig> encodingParams, const SchedulerTensor *weightTens,
        const SchedulerTensor *scaleTens, const Quantization &weightQuantization, const Quantization &ofmQuantization);

    WeightScaleTensors TryEncodeWeightAndScaleTensor(IWeightEncodingConfig *encodingParams,
        const SchedulerTensor *weightTens, const SchedulerTensor *scaleTens, const Quantization &weightQuantization,
        const Quantization &ofmQuantization, bool doWeights, bool doScales);

    WeightScaleEncoding EncodeBestWeightFormat(SchedulerOperation *op, Shape &ifmShape, Shape &ifm2Shape,
        Shape &ofmShape, Flags<WeightFormat> supportedFormats);
};

void ParseSchedulerOptions(SchedulerOptions &opt, IniReader &reader);

}  // namespace regor
