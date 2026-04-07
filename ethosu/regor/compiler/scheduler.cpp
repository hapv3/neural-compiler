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

#include "scheduler.hpp"

#include "common/logging.hpp"

#include "architecture/architecture_constraints.hpp"
#include "architecture/weight_encoder.hpp"
#include "cascade_builder.hpp"
#include "common/data_type.hpp"
#include "common/scaling.hpp"
#include "common/vector_span.hpp"
#include "faststorage_allocator.hpp"
#include "live_range.hpp"
#include "scheduler_decompose.hpp"
#include "tensor_allocator.hpp"

#include <cassert>
#include <memory>
#include <optional>
#include <vector>


BEGIN_ENUM_TABLE(regor::SchedulerFeature)
    ADD_ENUM_NAME(WeightBuffering)
    ADD_ENUM_NAME(Cascading)
    ADD_ENUM_NAME(Grouping)
    ADD_ENUM_NAME(FWD)
    ADD_ENUM_NAME(Sparsity)
    ADD_ENUM_NAME(FMStaging)
    ADD_ENUM_NAME(ReuseIFM)
END_ENUM_TABLE()

namespace regor
{

constexpr int AllocationQuantum = 16;
constexpr int NPUTensorAlignment = 16;

static Shape GetShapeForFormat(const Shape &shape, TensorFormat format)
{
    if ( format == TensorFormat::NHCWB16 )
    {
        return shape.With(-1, RoundAway(shape.Depth(), 16));
    }
    return shape;
}

int TensorAllocationBytes(const Shape &shape, TensorFormat format, DataType dtype)
{
    if ( !shape ) return 0;
    Shape storageShape = GetShapeForFormat(shape, format);
    return RoundAway(DataTypeStorageSizeBytes(dtype, storageShape.Elements()), AllocationQuantum);
}

const LRMemory &Schedule::MemoryAt(int timeIndex) const
{
    static LRMemory s_empty;
    return (timeIndex >= 0 && timeIndex < int(memorySnapshot.size())) ? memorySnapshot[timeIndex] : s_empty;
}



Scheduler::Scheduler(Architecture *arch, const SchedulerOptions &options, const std::string &name,
    std::vector<std::unique_ptr<SchedulerOperation>> &ops, const SchedulerOpConfigMap &opConfigCompatablility) :
        _ops(ops),
        _opConfigCompatablility(opConfigCompatablility)
{
    assert(arch != nullptr);
    _arch = arch;
    _options = options;
    _name = name;
    _spilling = _arch->StagingMemory() != _arch->FeatureMapMemory();
}

std::shared_ptr<Schedule> Scheduler::Process()
{
    Address peakMemoryUsage = CreateSchedulerRepresentation();

    // Create the Max schedule template
    _maxSchedule = CreateInitialSchedule();

    // TODO: Disabled until fully implemented
    // MoveConstantData( _maxSchedule.get() );

    // Create the optimised Max schedule
    UpdateOpMemorySnapshot(_maxSchedule.get());
    auto optMaxSchedule = ProposeScheduleBuffering(_maxSchedule.get(), std::numeric_limits<int>::max());
    UpdateOpMemorySnapshot(optMaxSchedule.get());

    // Create Min schedule
    auto minSchedule = ProposeMinimalSchedule();
    Address initialStagingLimit = _options.optimizationStagingLimit;
    if ( _options.optimizationStrategy == OptimizationStrategy::Size )
    {
        auto &memorySnapShot = _maxSchedule->memorySnapshot;
        assert(!memorySnapShot.memory.empty());
        initialStagingLimit = std::min_element(memorySnapShot.memory.begin(), memorySnapShot.memory.end())->Used();
    }

    std::shared_ptr<Schedule> chosenSchedule = _maxSchedule;

    if ( !_options.disabled.All(SchedulerFeature::Cascading) )
    {
        // Build cascades from min schedule
        std::unordered_map<UniqueId, LiveRangeSummary> liveRanges;
        std::unordered_map<UniqueId, int> opLocalMemUsage;
        auto nonLocal = ComputeNonLocalUsage(minSchedule.get(), &liveRanges, &opLocalMemUsage);

        CascadeBuilder cascadeBuilder(_ops, nonLocal, opLocalMemUsage, liveRanges, _spilling);
        cascadeBuilder.BuildCascades(minSchedule.get(), _maxSchedule.get(), initialStagingLimit);
        UpdateOpMemorySnapshot(minSchedule.get());

        chosenSchedule = minSchedule;

        Address stagingLimit = _options.optimizationStagingLimit;

        if ( _options.optimizationStrategy == OptimizationStrategy::Performance )
        {
            // Create an optimized schedule
            auto optSchedule = OptimizeSchedule(minSchedule.get(), optMaxSchedule, _options.optimizationStagingLimit);
            chosenSchedule = std::move(optSchedule);
        }
        else
        {
            // Run optimizer with a cap derived from the current peak usage
            auto &memorySnapShot = chosenSchedule->memorySnapshot;
            stagingLimit = memorySnapShot.maxMemory != 0 ? memorySnapShot.maxMemory : _options.optimizationStagingLimit;
            auto optSchedule = OptimizeSchedule(minSchedule.get(), optMaxSchedule, stagingLimit);
            chosenSchedule = std::move(optSchedule);
        }
    }

    if ( !_options.disabled.All(SchedulerFeature::WeightBuffering) )
    {
        CoalesceWeightBufferTensors(chosenSchedule.get());
    }

    UpdateOpMemorySnapshot(chosenSchedule.get());

    ApplySchedule(chosenSchedule.get());

    UpdateOpMemorySnapshot(chosenSchedule.get());

    if ( _spilling && !_options.disabled.All(SchedulerFeature::FMStaging) )
    {
        // Use fast storage for feature maps
        FastStorageAllocator allocator;
        const auto reuseIfms = !_options.disabled.All(SchedulerFeature::ReuseIFM);
        allocator.AllocateFeatureMaps(_ops, chosenSchedule.get(), _arch->StagingMemory(), _options.optimizationStagingLimit, reuseIfms);
    }

    UpdateOpMemorySnapshot(chosenSchedule.get());

    if ( _options.verboseSchedule )
    {
        PrintSchedule(chosenSchedule.get());
    }

    if ( !AllocateAddresses(chosenSchedule.get()) )
    {
        int used = chosenSchedule->memoryUsage[_arch->StagingMemory()];
        throw std::runtime_error(fmt::format("Failed to allocate tensors. Memory used {}, limit {})\n", used, _options.optimizationStagingLimit));
    }

    return chosenSchedule;
}

Point2i Scheduler::GetStripeInputRequirement(const Shape &ofmShape, const Kernel *kernel, const Point2i &ifmStep, ArchResampling resampling)
{
    int rounding;
    int upscale = _arch->UpscaleAndRounding(resampling, rounding);
    auto stride = kernel->Stride() * ifmStep;
    int h = RequiredInputSize(ofmShape.Height(), stride.y, kernel->DilatedWH().y, upscale, rounding);
    int w = RequiredInputSize(ofmShape.Width(), stride.x, kernel->DilatedWH().x, upscale, rounding);
    return Point2i(w, h);
}

// Returns true if NHWC format must be used for the given tensor
static bool CheckLinearFormatForConcatSplit(SchedulerTensor *tensor)
{
    for ( const auto &prod : tensor->producers )
    {
        // If axis corresponds to C-dimension, NHCWB16 can only be used in the output if all the concat_start's
        // are a multiple of 16. This as, it is only then the address offset for the ofm, for all operations,
        // will be 16 byte aligned. For other values of axis the address offsets will be 16 byte aligned, as they
        // are all based on c = 0 and those addresses are always 16 byte aligned due to the NHCWB16 format.
        for ( auto &conn : prod->outputs )
        {
            if ( conn.tensor.get() == tensor && conn.slice.offset.Size() > 0 && (conn.slice.offset.Depth() & 15) != 0 )
            {
                return true;
            }
        }
    }
    for ( const auto &cons : tensor->consumers )
    {
        // If read offset is not a multiple of 16 in the C-dimension, NHCWB16 need to be avoided in the input.
        for ( auto &conn : cons->inputs )
        {
            if ( conn.tensor.get() == tensor && conn.slice.offset.Size() > 0 && (conn.slice.offset.Depth() & 15) != 0 )
            {
                return true;
            }
        }
    }
    return false;
}

// Align stripes so that height is a multiple of the upscaling-factor
Shape Scheduler::AlignStripe(const SchedulerOperation *schedOp, const Shape &stripe)
{
    auto ifm = schedOp->TryIFM(0);
    auto ofm = schedOp->TryOFM();
    if ( !ofm || !ifm ) return stripe;
    if ( stripe != ofm->SliceShape() )
    {
        // striped operation, align stripe-height with upscaling factor
        int rounding = 0;
        int upscale = _arch->UpscaleAndRounding(ifm->resamplingMode, rounding);
        return stripe.WithHeight(RoundAway(stripe.Height(), upscale));
    }
    return stripe;
}

int Scheduler::UpdateSchedulerTensor(TensorUsage usage, SchedulerConnection *conn, std::unordered_set<UniqueId> &visited)
{
    auto tensor = conn->tensor.get();
    if ( visited.insert(tensor->uid).second )
    {
        // Force linear format if number of elements overflows in brick format
        if ( tensor->storageShape &&
             Shape::RoundAway(tensor->storageShape, Shape(1, 1, 1, 16)).Elements64() > std::numeric_limits<int>::max() )
        {
            tensor->needsLinearFormat = true;
        }

        // Force linear format for read only or persistent tensors
        if ( tensor->IsConstant() || tensor->isPersistent )
        {
            tensor->needsLinearFormat = true;
        }
        if ( CheckLinearFormatForConcatSplit(tensor) )
        {
            tensor->needsLinearFormat = true;
        }

        std::unordered_set<Point2i, Point2Hash<int>> ifmShapes;
        bool isAnyConsumerReduceSum = false;

        for ( auto producer : tensor->producers )  // Can be refactored into check tensor once.
        {
            if ( producer->IsNpuOp() )
            {
                tensor->hasNPUWriters = true;
            }
            else
            {
                tensor->hasCPUWriters = true;
            }

            // TODO: Gather doesn't support brick format yet (MLBEDSW-8410)
            if ( producer->Type() == OpType::Scatter || producer->Type() == OpType::Gather )
            {
                tensor->needsLinearFormat = true;
                continue;
            }
            else if ( IsControlFlow(producer->Type()) )
            {
                tensor->needsLinearFormat = true;
                continue;
            }
            else
            {
                ArchRequirements req;
                ArchOperatorQuery query;
                Set(query.ifm[0], producer->TryIFM(0));
                Set(query.ifm[1], producer->TryIFM(1));
                Set(query.ofm, producer->OFM());
                query.transposeMask = producer->OFM()->transpose;
                if ( _arch->Constraints()->OperatorQuery(producer->Type(), &query, &req).Any(QueryResult::Native) )
                {
                    if ( req.req % ArchRequirement::Tensor )
                    {
                        auto *tr = Get(&req.tensor, TensorUsage::OFM);
                        if ( tr && tr->format == TensorFormat::NHWC )
                        {
                            tensor->needsLinearFormat = true;
                            continue;
                        }
                    }
                }
            }
        }

        for ( auto consumer : tensor->consumers )
        {
            if ( consumer->IsNpuOp() )
            {
                tensor->hasNPUReaders = true;
            }
            else
            {
                tensor->hasCPUReaders = true;
            }
            // Int32 ReduceSum requires linear format
            if ( consumer->Type() == OpType::ReduceSum )
            {
                isAnyConsumerReduceSum = true;
            }
            for ( const auto [tensorUsage, connection] : consumer->inputs.pairs() )
            {
                if ( connection.tensor.get() == tensor && IsIFM(tensorUsage) )
                {
                    ifmShapes.insert(connection.SliceShape().WC(1));
                }
            }

            // TODO: Gather doesn't support brick format yet (MLBEDSW-8410)
            if ( consumer->Type() == OpType::Scatter || consumer->Type() == OpType::Gather )
            {
                tensor->needsLinearFormat = true;
                continue;
            }
            else if ( IsControlFlow(consumer->Type()) )
            {
                tensor->needsLinearFormat = true;
                continue;
            }
        }
        // Check if consumer shape requires linear format
        // Brick format can only be used if both shapes have equal W and C
        // Need to check full shape on connection since tensor might have many producers (concat)
        for ( auto producer : tensor->producers )
        {
            if ( tensor->needsLinearFormat ) break;
            for ( const auto [_, connection] : producer->outputs.pairs() )
            {
                if ( connection.tensor.get() == tensor )
                {
                    if ( ifmShapes.count(connection.shape.WC<int>(1)) != ifmShapes.size() )
                    {
                        tensor->needsLinearFormat = true;
                        break;
                    }
                    else if ( isAnyConsumerReduceSum && connection.Type() == DataType::Int32 )
                    {
                        tensor->needsLinearFormat = true;
                        break;
                    }
                }
            }
        }
        for ( auto consumer : tensor->consumers )
        {
            if ( tensor->needsLinearFormat ) break;

            ArchRequirements req;
            ArchOperatorQuery query;
            Set(query.ifm[0], consumer->TryIFM(0));
            Set(query.ifm[1], consumer->TryIFM(1));
            Set(query.ofm, consumer->OFM());
            query.transposeMask = consumer->OFM()->transpose;
            for ( const auto [consumerUsage, connection] : consumer->inputs.pairs() )
            {
                if ( connection.tensor.get() == tensor )
                {
                    if ( ifmShapes.count(connection.shape.WC<int>(1)) != ifmShapes.size() )
                    {
                        tensor->needsLinearFormat = true;
                        break;
                    }
                    else if ( isAnyConsumerReduceSum && connection.Type() == DataType::Int32 )
                    {
                        tensor->needsLinearFormat = true;
                        break;
                    }
                    else if ( _arch->Constraints()->OperatorQuery(consumer->Type(), &query, &req).Any(QueryResult::Native) )
                    {
                        if ( req.req % ArchRequirement::Tensor )
                        {
                            auto *tr = Get(&req.tensor, consumerUsage);
                            if ( tr && tr->format == TensorFormat::NHWC )
                            {
                                tensor->needsLinearFormat = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    // Multiple consumers/producers require the full tensor present
    if ( tensor->producers.size() > 1 || tensor->consumers.size() > 1 || (IsOFM(usage) && conn->slice.offset.Size() > 0) ||  // Concat
         tensor->isGraphInput || tensor->isGraphOutput )
    {
        conn->requireFullTensor = true;
    }
    // Force linear output from Reverse for C dimension because brick output from Reverse has special requirements
    if ( IsOFM(usage) && conn->reverse == ReverseType::C )
    {
        tensor->needsLinearFormat = true;
    }
    // Force linear format for any reversal using negative striding
    if ( _arch->Constraints()->SupportsNegativeStrides() && conn->reverse != ReverseType::None )
    {
        tensor->needsLinearFormat = true;
    }
    // Force linear format for strided access in the width dimension
    if ( conn->stepXY.x != 1 )
    {
        tensor->needsLinearFormat = true;
    }

    // Figure out if a tensor has dynamic shape (any negative axis), i.e. the shape is unknown at compile time
    const bool unknownShape = tensor->storageShape ? (tensor->storageShape.LessMask(tensor->storageShape.WithZeros()) != 0) : true;

    // Initial criteria (may change)
    const bool cpuTensor =
        tensor->hasCPUWriters || tensor->hasCPUReaders || tensor->isGraphInput || tensor->isGraphOutput || tensor->isPersistent;
    conn->requireFullTensor = conn->requireFullTensor || cpuTensor;
    tensor->needsLinearFormat = tensor->needsLinearFormat || cpuTensor;

    if ( (_options.separateIORegions || tensor->IsConstant() || unknownShape) && cpuTensor && !tensor->hasNPUWriters &&
         !tensor->hasNPUReaders )
    {
        tensor->memArea = _arch->CPUMemory();
    }
    else if ( _options.separateIORegions && !tensor->IsConstant() && cpuTensor )
    {
        tensor->memArea = tensor->hasNPUWriters ? _arch->OutputFeatureMapMemory() : _arch->InputFeatureMapMemory();
    }

    // Set tensor format to NHCWB16 for FeatureMaps, if possible
    if ( IsIFM(usage) || IsOFM(usage) )
    {
        tensor->format = tensor->needsLinearFormat ? TensorFormat::NHWC : TensorFormat::NHCWB16;
    }

    return tensor->IsConstant() ? 0 : tensor->AllocationSizeBytes();
}


Address Scheduler::CreateSchedulerRepresentation()
{
    int minMemoryRequired = 0;
    std::unordered_set<UniqueId> visited;

    for ( auto const &schedOp : _ops )
    {
        int opMemoryRequired = 0;

        for ( auto pos : schedOp->outputs.pairs() )
        {
            assert(!pos.second.tensor->producers.empty());
            opMemoryRequired += UpdateSchedulerTensor(pos.first, &pos.second, visited);
        }

        for ( auto pos : schedOp->inputs.pairs() )
        {
            assert(!pos.second.tensor->consumers.empty());
            opMemoryRequired += UpdateSchedulerTensor(pos.first, &pos.second, visited);
        }

        for ( auto const &subOp : schedOp->SubOps() )
        {
            for ( auto pos : subOp->outputs.pairs() )
            {
                opMemoryRequired += UpdateSchedulerTensor(pos.first, &pos.second, visited);
            }

            for ( auto pos : subOp->inputs.pairs() )
            {
                opMemoryRequired += UpdateSchedulerTensor(pos.first, &pos.second, visited);
            }
        }
        minMemoryRequired = std::max(minMemoryRequired, opMemoryRequired);
    }

    return minMemoryRequired;
}


namespace
{


ArchAccumulatorSource GetArchAccumulatorSource(const AccumulatorControl &ac)
{
    switch ( ac.source )
    {
        case AccumulatorSource::Reset:
            return ArchAccumulatorSource::Reset;
        case AccumulatorSource::Acc:
            return ArchAccumulatorSource::Acc;
        case AccumulatorSource::Ifm2:
            return ArchAccumulatorSource::Ifm2;
        default:
            return ArchAccumulatorSource::Reset;
    }
}


}  // namespace


std::unique_ptr<ArchitectureOpConfig> Scheduler::GetOpConfig(SchedulerOperation *op, const Shape &ifmShape,
    const Shape &ifm2Shape, const Shape &ofmShape, WeightFormat wgtFormat)
{
    assert(op->IsNpuOp());
    using OpGroupReq = ArchitectureOpGroup::Requirement;

    SchedulerConnection *ifm = op->IFM(0);
    SchedulerConnection *ifm2 = op->TryIFM(1);
    SchedulerConnection *ofm = op->OFM();

    ArchitectureConfigQuery query{};
    query.compatibleWithConfig = _opConfigCompatablility.count(*op) ? _opConfigCompatablility.at(*op).get() : nullptr;
    query.ofmShape = Shape::PadAxes(ofmShape, 3, 1);
    query.ifmShape[0] = ifmShape;
    query.ifmShape[1] = ifm2Shape;
    query.ifmBits = DataTypeSizeBits(ifm->Type());
    query.ofmBits = DataTypeSizeBits(ofm->Type());
    query.kernel = op->Kernel();
    query.lutBytes = op->OpGroup()->Requirements().Any(OpGroupReq::UsesLUT) ? 2048 : 0;
    query.scaled = op->HasScaling();
    query.ifmResampling = ifm->resamplingMode;
    query.ofmShape = query.ofmShape.Unpermute(uint32_t(ofm->transpose));
    query.transpose = ofm->transpose;
    query.reverse = ofm->reverse;
    query.ofmFormat = ofm->tensor->format;
    const auto &accMode = op->AccumulatorMode();
    query.accSource = GetArchAccumulatorSource(accMode);
    query.accOutputEnabled = accMode.outputEnabled;
    query.weightFormat = wgtFormat;
    if ( op->Type() == OpType::Resize )
    {
        const auto *attr = op->Attribute<resize_attr_t>();
        query.rescaling.scaleX = attr->scaleX;
        query.rescaling.scaleY = attr->scaleY;
    }

    return _arch->GetOpConfig(op->Type(), query);
}

WeightScaleEncoding Scheduler::ChooseBestWeightFormat(SchedulerOperation *op, OptimizationStrategy optimizationStrategy,
    std::vector<WeightScaleEncoding> &encodingResults)
{
    WeightScaleEncoding *bestResult = nullptr;

    if ( optimizationStrategy == OptimizationStrategy::Size )
    {
        auto compare = [](const WeightScaleEncoding &a, const WeightScaleEncoding &b) {
            return a.weightScales.npuWeightsTensor->totalWeightBytes < b.weightScales.npuWeightsTensor->totalWeightBytes;
        };
        bestResult = &*std::min_element(encodingResults.begin(), encodingResults.end(), compare);
    }
    else
    {
        auto minCycles = std::numeric_limits<int64_t>::max();
        for ( auto &encodingResult : encodingResults )
        {
            WeightStats weightStats;
            auto weightTensor = encodingResult.weightScales.npuWeightsTensor;
            weightStats.size = weightTensor->totalSourceBytes;
            weightStats.encodedSize = weightTensor->totalWeightBytes;
            weightStats.zeroCount = weightTensor->zeroCount;
            weightStats.distinctWeights = weightTensor->distinctWeights;
            auto query = Scheduler::InitPerfQuery(op, nullptr);
            auto totalCycles =
                _arch->Performance()->WeightDecodeCycles(
                    query, weightStats, weightTensor->config->Format(), weightTensor->memArea.memory) +
                encodingResult.cycleCost.opCycles;
            if ( totalCycles < minCycles )
            {
                bestResult = &encodingResult;
                minCycles = totalCycles;
            }
        }
    }
    return std::move(*bestResult);
}

bool Scheduler::UseFastDecoder(SchedulerOperation *op, OptimizationStrategy optimizationStrategy, NpuWeightTensor *weightTensor)
{
    int fastSizeDivisor = 1;
    if ( weightTensor->distinctWeights > 0 && weightTensor->distinctWeights <= 16 )
    {
        fastSizeDivisor = weightTensor->distinctWeights <= 4 ? 4 : 2;
    }
    int fastWeightSize = 32 + weightTensor->totalSourceBytes / fastSizeDivisor;
    if ( optimizationStrategy == OptimizationStrategy::Size )
    {
        return fastWeightSize < weightTensor->totalWeightBytes;
    }
    WeightStats weightStats;
    weightStats.size = weightTensor->totalSourceBytes;
    weightStats.encodedSize = weightTensor->totalWeightBytes;
    weightStats.zeroCount = weightTensor->zeroCount;
    weightStats.distinctWeights = weightTensor->distinctWeights;
    auto query = Scheduler::InitPerfQuery(op, nullptr);
    auto defaultCycles = _arch->Performance()->WeightDecodeCycles(
        query, weightStats, WeightFormat::Default, weightTensor->memArea.memory);
    weightStats.encodedSize = fastWeightSize;
    auto fastCycles = _arch->Performance()->WeightDecodeCycles(
        query, weightStats, WeightFormat::Fast, weightTensor->memArea.memory);
    return fastCycles < defaultCycles;
}

std::unique_ptr<ArchitectureOpConfig> Scheduler::MaybeGetSparsityConfig(
    SchedulerOperation *op, Shape &ifmShape, Shape &ifm2Shape, Shape &ofmShape, Flags<WeightFormat> supportedFormat)
{
    using WF = Flags<WeightFormat>;
    std::unique_ptr<ArchitectureOpConfig> blockConfigSparse;
    if ( supportedFormat % WeightFormat::Sparse2_4 )
    {
        blockConfigSparse = GetOpConfig(op, ifmShape, ifm2Shape, ofmShape, WF(WeightFormat::Default, WeightFormat::Sparse2_4));
    }
    return blockConfigSparse;
}

WeightScaleEncoding Scheduler::EncodeBestWeightFormat(
    SchedulerOperation *op, Shape &ifmShape, Shape &ifm2Shape, Shape &ofmShape, Flags<WeightFormat> supportedFormats)
{
    using WF = Flags<WeightFormat>;
    // We assume that block config depends only on the sparsity bit in the weight format.
    std::unique_ptr<ArchitectureOpConfig> blockConfigDefault = GetOpConfig(op, ifmShape, ifm2Shape, ofmShape, WF(WeightFormat::Default));
    std::unique_ptr<ArchitectureOpConfig> blockConfigSparse = MaybeGetSparsityConfig(op, ifmShape, ifm2Shape, ofmShape, supportedFormats);

    CycleCost defaultCycleCost;
    CycleCost sparseCycleCost;
    if ( blockConfigSparse )
    {
        defaultCycleCost = EstimateOpPerformance(op, blockConfigDefault.get(), op->OFM()->SliceShape().Depth());
        sparseCycleCost = EstimateOpPerformance(op, blockConfigSparse.get(), op->OFM()->SliceShape().Depth(), WeightFormat::Sparse2_4);
        if ( sparseCycleCost.opCycles > defaultCycleCost.opCycles )
        {
            supportedFormats.Unset(WeightFormat::Sparse2_4);
        }
    }
    else if ( supportedFormats % WeightFormat::Sparse2_4 )
    {  // No block config available for sparse 2_4, so disable.
        supportedFormats.Unset(WeightFormat::Sparse2_4);
    }

    std::vector<WeightScaleEncoding> encodingResults;
    auto weights = op->Input(TensorUsage::Weights);
    auto scales = op->Input(TensorUsage::Scales);
    auto ifm = op->IFM(op->PrimaryIfmIndex());
    auto ifmType = ifm->Type();

    std::vector<int> ofmDepthOffsets{0, ofmShape.Unpermute(uint32_t(op->OFM()->transpose)).Depth()};

    const std::array<WF, 4> formatList = {WF(WeightFormat::Default, WeightFormat::Sparse2_4), WF(WeightFormat::Default),
        WF(WeightFormat::Fast, WeightFormat::Sparse2_4), WF(WeightFormat::Fast)};

    for ( auto weightFormat : formatList )
    {
        if ( (weightFormat & supportedFormats) != weightFormat ) continue;
        bool checkFastDecoder = !(weightFormat % WeightFormat::Fast) && (supportedFormats % WeightFormat::Fast);

        auto *blockConfig = (weightFormat % WeightFormat::Sparse2_4) ? blockConfigSparse.get() : blockConfigDefault.get();
        if ( !blockConfig )
        {
            throw std::runtime_error("Failed to find block configuration\n");
        }
        auto encodingParams = _arch->WeightEncoder()->GetEncodingConfig(blockConfig, op->Kernel(), ifmType, weightFormat);

        try
        {
            WeightScaleEncoding encoding;
            encoding.weightScales = EncodeWeightAndScaleTensor(op->Type(), std::move(encodingParams), op->weightDepthOffset,
                ofmDepthOffsets, weights->tensor.get(), scales->tensor.get(), weights->quantization, op->OFM()->quantization);

            if ( checkFastDecoder &&
                 !UseFastDecoder(op, _options.optimizationStrategy, encoding.weightScales.npuWeightsTensor.get()) )
            {
                supportedFormats.Unset(WeightFormat::Fast);
            }
            // Sparse2_4 affects opCycles and must be accounted for when selecting the best weight format
            encoding.cycleCost = (weightFormat % WeightFormat::Sparse2_4) ? sparseCycleCost : defaultCycleCost;
            encodingResults.emplace_back(std::move(encoding));
        }
        catch ( const WeightEncodeException & )
        {
            if ( weightFormat % WeightFormat::Sparse2_4 )
            {
                supportedFormats.Unset(WeightFormat::Sparse2_4);
            }
            continue;
        }
    }
    assert(!encodingResults.empty());
    auto bestEncoding = ChooseBestWeightFormat(op, _options.optimizationStrategy, encodingResults);
    bestEncoding.blockConfig =
        (bestEncoding.weightScales.npuWeightsTensor->config->Format() % WeightFormat::Sparse2_4) ? std::move(blockConfigSparse) : std::move(blockConfigDefault);

    return bestEncoding;
}

std::unique_ptr<SchedulerOpInfo> Scheduler::CreateSchedulerOpInfo(
    SchedulerOperation *op, const Shape &ofmStripeShape, const std::unique_ptr<SchedulerOpInfo> &parentInfo)
{
    assert(op->PrimaryIfmIndex() >= 0 && op->PrimaryIfmIndex() <= 1);
    SchedulerConnection *ifm = op->IFM(op->PrimaryIfmIndex());
    SchedulerConnection *ifm2 = op->TryIFM(1 - op->PrimaryIfmIndex());
    SchedulerConnection *ofm = op->OFM();

    auto ifmShape = ifm->SliceShape();
    auto ifm2Shape = ifm2 ? ifm2->SliceShape() : Shape();
    auto ofmShape = ofmStripeShape;

    const auto &subOps = op->SubOps();
    const bool isChained =
        op->Parent() != nullptr ||
        subOps.end() !=
            std::find_if(subOps.begin(), subOps.end(), [](const auto &subOp) { return IsElementwise(subOp->Type()); });

    // Operations that cannot be subdivided require full OFM shape
    // TODO MLBEDSW-9143 support cascading for chains..
    Flags<AxisMask> subdivideMask = _arch->CanSubdivide(op->Type(), ofm->transpose, ofm->reverse);
    if ( subdivideMask == AxisMask::None || isChained )
    {
        ofmShape = op->OFM()->SliceShape();
    }

    // Give empty operation info to CPU ops
    if ( !op->IsNpuOp() )
    {
        return std::make_unique<SchedulerOpInfo>(nullptr, ifmShape, ifm2Shape, ofmShape);
    }

    // Determine if striped operation
    if ( ofmShape != op->OFM()->SliceShape() )
    {
        // Striped Op - Need to calculate stripe input volume
        Point2i stripeInput = GetStripeInputRequirement(ofmShape, op->Kernel(), ifm->stepXY, ifm->resamplingMode);

        // Ensure stripe input volume is within the full IFM volume
        stripeInput = Point2i::Min(stripeInput, ifmShape.WH());
        if ( !subdivideMask.Any(AxisMask::AxisX) && (stripeInput.x != ifmShape.Width()) )
        {
            assert(stripeInput.x * ifm->stepXY.x >= (ifmShape.Width() - op->Kernel()->Stride().x) && "Unexpected stripe input width");
            stripeInput.x = ifmShape.Width();
        }

        ifmShape = ifmShape.WithHW(stripeInput.y, stripeInput.x);

        if ( !ifm2Shape.IsEmpty() )
        {
            Point2i stripeInput2 = Point2i::Min(stripeInput, ifm2Shape.WH());
            ifm2Shape = ifm2Shape.WithHW(stripeInput2.y, stripeInput2.x);
        }
    }

    auto weightFormat = _arch->SupportedWeightFormat(op->Type());

    // Disable specific weight formats if requested
    if ( _options.disabled.All(SchedulerFeature::FWD) ) weightFormat.Unset(WeightFormat::Fast);
    if ( _options.disabled.All(SchedulerFeature::Sparsity) ) weightFormat.Unset(WeightFormat::Sparse2_4);

    WeightScaleTensors weightScales;
    auto weights = op->TryInput(TensorUsage::Weights);
    if ( !weights || !weights->tensor->IsConstant() ) weightFormat = WeightFormat::Default;


    std::unique_ptr<ArchitectureOpConfig> blockConfig;
    if ( weights )
    {
        assert(op->OFM()->quantization.type == QuantizationType::EXPLICIT);
        auto weightEncoding = EncodeBestWeightFormat(op, ifmShape, ifm2Shape, ofmShape, weightFormat);
        blockConfig = std::move(weightEncoding.blockConfig);
        weightScales = weightEncoding.weightScales;
    }
    else
    {
        blockConfig = parentInfo ? parentInfo->Config()->Clone() : GetOpConfig(op, ifmShape, ifm2Shape, ofmShape, weightFormat);
    }
    assert((ofmShape == op->OFM()->SliceShape() || ofmShape.Height() % blockConfig->MinimalStripeGranule().y == 0) && "Stripe is not divisible with minimalStripeGranule");
    auto scales = op->TryInput(TensorUsage::Scales);
    if ( !weights && (op->OFM()->quantization.scales.size() > 1 || scales) )
    {
        std::vector<int> untransposedDepthOffsets{0, ofmShape.Unpermute(uint32_t(op->OFM()->transpose)).Depth()};

        // The operation might have been decomposed in depth dimension and have an offset
        auto encodingParams = _arch->WeightEncoder()->GetEncodingConfig(blockConfig.get(), op->Kernel(), ifm->Type(), weightFormat);

        const SchedulerTensor *scaleTensor = scales ? scales->tensor.get() : nullptr;
        weightScales = EncodeQuantizationScaleTensor(op->Type(), std::move(encodingParams), op->weightDepthOffset,
            untransposedDepthOffsets, op->OFM()->quantization, scaleTensor);
    }
    // Finally construct and populate operator information (cost)
    auto opInfo = std::make_unique<SchedulerOpInfo>(std::move(blockConfig), ifmShape, ifm2Shape, ofmShape);
    opInfo->SetWeightScaleTensors(weightScales.npuWeightsTensor, weightScales.npuScalesTensor);

    return opInfo;
}

std::unique_ptr<Schedule> Scheduler::CreateInitialSchedule()
{
    auto schedule = std::make_unique<Schedule>(_name + "_MAX", 0, _ops.size());
    for ( auto &op : _ops )
    {
        const auto ofm = op->OFM();
        const auto &ofmShape = ofm->SliceShape();
        auto cost = CreateSchedulerOpInfo(op.get(), ofmShape);
        for ( auto &subOp : op->SubOps() )
        {
            const auto subOfm = subOp->OFM();
            const auto &subOfmShape = subOfm->SliceShape();
            auto subCost = CreateSchedulerOpInfo(subOp.get(), subOfmShape, cost);
            schedule->SetCost(*subOp, std::move(subCost));
        }

        if ( ofmShape && op->IsNpuOp() )
        {
            auto query = InitPerfQuery(
                op.get(), cost->Config(), ofmShape.Depth(), WeightFormat::Default, cost.get(), schedule.get());
            cost->cycles = _arch->Performance()->MeasureCycleCost(query);
            cost->elementAccess = _arch->Performance()->MeasureElementAccess(query);
            cost->perf = EstimateSlicedOpPerformance(op.get(), cost->ofmDepthSlices, cost->Config(),
                cost->npuWeightsTensor.get(), StagingPref::None, ofmShape.WH(), 0, Buffering::None);
        }

        schedule->SetCost(*op, std::move(cost));
    }
    return schedule;
}


void Scheduler::MoveConstantData(Schedule *refSchedule)
{
    auto permanentStorageMemory = _arch->ReadonlyMemory();
    const bool moveConstantData = permanentStorageMemory != _arch->FeatureMapMemory();

    // Determine if data can be moved from permanent storage to another memory area. A difference in source tensor
    // and target tensor memory area will generate a DMA command in the command stream.
    for ( auto &schedOp : _ops )
    {
        // Ignore CPU ops
        if ( !schedOp->IsNpuOp() )
        {
            continue;
        }

        auto cost = refSchedule->Cost(schedOp.get());
        int maxIfmShramAvail = cost->Config()->MaxIFMBuffering() / 2;
        for ( auto pos : schedOp->inputs.pairs() )
        {
            SchedulerConnection *conn = &pos.second;
            if ( !conn->tensor->IsConstant() )
            {
                continue;
            }

            // Determine whether or not to move data from permanent storage to more suitable
            // storage before use.
            bool moveData = false;
            if ( conn->tensor->memArea == permanentStorageMemory && moveConstantData )
            {
                moveData = std::any_of(conn->tensor->consumers.begin(), conn->tensor->consumers.end(),
                    [](const SchedulerOperation *op) { return op->Type() != OpType::FullyConnected; });

                // Check if broadcast elementwise can be buffered
                if ( IsIFM(pos.first) && IsElementwise(schedOp->Type()) && (conn->shape != schedOp->OFM()->shape) &&
                     conn->tensor->bufferView.Buffer()->Size() > maxIfmShramAvail )
                {
                    moveData = true;
                }
            }

            if ( moveData )
            {
                // Set scheduler tensor to different memory area i.e. move from srcTensor to (scheduler) tensor
                conn->tensor->memArea = _arch->StagingMemory();
            }
        }
    }
}


bool Scheduler::AllocateAddresses(Schedule *schedule)
{
    const auto verbose = _options.verboseAllocation;
    const auto reuseIfms = !_options.disabled.All(SchedulerFeature::ReuseIFM);
    schedule->featureMapLRGraph = std::make_unique<LiveRangeGraph>(reuseIfms);
    // If graph input/outputs tensors are in FeatureMap memory, allocate with user-specified tensor alignment
    AllocateTensors(_ops, schedule, *schedule->featureMapLRGraph, _arch->FeatureMapMemory(), _options.tensorAllocator,
        _options.separateIORegions ? NPUTensorAlignment : _options.cpuTensorAlignment, verbose);
    if ( _spilling )
    {
        const auto limit = _options.optimizationStagingLimit;
        schedule->stagingLRGraph = std::make_unique<LiveRangeGraph>(reuseIfms);
        AllocateTensors(_ops, schedule, *schedule->stagingLRGraph, _arch->StagingMemory(), _options.tensorAllocator,
            NPUTensorAlignment, verbose, limit);
        return schedule->memoryUsage[_arch->StagingMemory()] <= limit;
    }
    return true;
}


void Scheduler::AllocateReadOnlyAddresses(Schedule *schedule, IncrementalLinearAllocator &readOnlyAllocator)
{
    LiveRangeGraph lrGraph{false};
    lrGraph.ExtractLiveRangesFromCascades(_ops, schedule, _arch->ReadonlyMemory(), false);
    auto totalSize = readOnlyAllocator.Allocate(&lrGraph, NPUTensorAlignment, _options.verboseAllocation);
    schedule->memoryUsage[_arch->ReadonlyMemory()] = int(totalSize);
}


void Scheduler::AllocateIOAddresses(Schedule *schedule, const std::vector<std::unique_ptr<SchedulerOperation>> &ops)
{
    const auto verbose = _options.verboseAllocation;
    const auto separateIORegions = _options.separateIORegions;
    const auto reuseIfms = !_options.disabled.All(SchedulerFeature::ReuseIFM);
    if ( separateIORegions )
    {
        assert(_arch->InputFeatureMapMemory() != _arch->OutputFeatureMapMemory());

        LiveRangeGraph inputLRGraph(false);
        AllocateTensors(ops, schedule, inputLRGraph, _arch->InputFeatureMapMemory(), TensorAllocator::LinearAlloc, NPUTensorAlignment, verbose);
        LiveRangeGraph outputLRGraph{false};
        AllocateTensors(ops, schedule, outputLRGraph, _arch->OutputFeatureMapMemory(), TensorAllocator::LinearAlloc,
            NPUTensorAlignment, verbose);
    }
}


void Scheduler::UpdateOpMemorySnapshot(Schedule *schedule, LiveRangeGraph *liveRanges)
{
    const auto fastStorage = _arch->StagingMemory();
    const auto reuseIfms = !_options.disabled.All(SchedulerFeature::ReuseIFM);
    LiveRangeGraph localGraph{reuseIfms};
    LiveRangeGraph &lrGraph = liveRanges ? *liveRanges : localGraph;
    lrGraph.ExtractLiveRangesFromCascades(_ops, schedule, fastStorage, true);
    // Populate time-array with memory used by live ranges
    schedule->memorySnapshot = lrGraph.GetTemporalMemoryUsage();
    schedule->fastStoragePeakUsage = schedule->memorySnapshot.maxMemory;
}

// Summarize live ranges so callers can query per-tensor timing information.
void Scheduler::PopulateLiveRanges(const LiveRangeGraph &lrGraph, std::unordered_map<UniqueId, LiveRangeSummary> *liveRanges) const
{
    if ( !liveRanges )
    {
        return;
    }

    liveRanges->clear();
    liveRanges->reserve(_ops.size());
    for ( const auto &lr : lrGraph.LiveRanges() )
    {
        LiveRangeSummary summary;
        summary.startTime = lr->startTime;
        summary.endTime = lr->endTime;
        summary.size = RoundAway(lr->size, AllocationQuantum);
        UniqueId rangeId = INVALID_UID;
        for ( const auto *tensor : lr->tensors )
        {
            rangeId = std::min(rangeId, tensor->equivalenceId);
        }
        summary.rangeId = rangeId;
        for ( auto *tensor : lr->tensors )
        {
            (*liveRanges)[tensor->equivalenceId] = summary;
        }
    }
}

// Compute the staging-memory footprint local to a single op.
int Scheduler::ComputeLocalMemUsage(const SchedulerOperation &schedOp, const SchedulerOpInfo &cost,
    const LiveRangeGraph &lrGraph, const MemArea &stagingMemory) const
{
    int opMemUsage = 0;
    auto *ifmConn = schedOp.TryIFM(schedOp.PrimaryIfmIndex());
    assert(ifmConn);
    auto *ofmConn = schedOp.SubOps().empty() ? schedOp.OFM() : schedOp.SubOps().back()->OFM();

    if ( ifmConn->tensor->memArea == stagingMemory )
    {
        opMemUsage += ifmConn->tensor->AllocationSizeBytes();
    }

    const bool reuseOfmFlag = lrGraph.AreInSameRange(ifmConn->tensor.get(), ofmConn->tensor.get());
    if ( ofmConn->tensor->memArea == stagingMemory && !reuseOfmFlag )
    {
        opMemUsage += ofmConn->tensor->AllocationSizeBytes();
    }

    opMemUsage += cost.bufferedWeightTensor.AllocatedSize();
    return opMemUsage;
}

std::unordered_map<UniqueId, int> Scheduler::ComputeNonLocalUsage(Schedule *schedule,
    std::unordered_map<UniqueId, LiveRangeSummary> *liveRanges, std::unordered_map<UniqueId, int> *opLocalMemUsage)
{
    std::unordered_map<UniqueId, int> nonLocal;
    nonLocal.reserve(_ops.size());
    const auto reuseIfms = !_options.disabled.All(SchedulerFeature::ReuseIFM);
    const auto stagingMemory = _arch->StagingMemory();
    LiveRangeGraph lrGraph{reuseIfms};
    UpdateOpMemorySnapshot(schedule, &lrGraph);
    PopulateLiveRanges(lrGraph, liveRanges);
    if ( opLocalMemUsage )
    {
        opLocalMemUsage->clear();
        opLocalMemUsage->reserve(_ops.size());
    }
    for ( auto const &schedOp : _ops )
    {
        auto *cost = schedule->Cost(schedOp.get());
        assert(cost);

        int opMemUsage = ComputeLocalMemUsage(*schedOp, *cost, lrGraph, stagingMemory);
        if ( opLocalMemUsage )
        {
            (*opLocalMemUsage)[*schedOp] = opMemUsage;
        }

        int snapshotUsage = schedule->MemoryUsageAt(cost->timeIndex);
        int nonLocalBytes = snapshotUsage - opMemUsage;
        nonLocal[*schedOp] = nonLocalBytes;
    }
    return nonLocal;
}


std::shared_ptr<Schedule> Scheduler::ProposeScheduleBuffering(Schedule *refSchedule, Address stagingLimitBytes)
{
    auto bufferedSchedule = std::make_shared<Schedule>(
        refSchedule->Name() + "_BUFFERED", refSchedule->Start(), refSchedule->End());
    int stagingLimitClamped = int(std::min(INT64_C(1) << 30, stagingLimitBytes));

    bufferedSchedule->memorySnapshot = refSchedule->memorySnapshot;

    SchedulerOperation *prevOp = refSchedule->Start() > 0 ? _ops.at(refSchedule->Start() - 1).get() : nullptr;
    for ( int pos = refSchedule->Start(); pos < refSchedule->End(); pos++ )
    {
        auto *schedOp = _ops.at(pos).get();

        ProposeOperatorBuffering(schedOp, prevOp, bufferedSchedule.get(), refSchedule, stagingLimitClamped);

        // chained sub-operations
        for ( auto const &subOp : schedOp->SubOps() )
        {
            ProposeOperatorBuffering(subOp.get(), prevOp, bufferedSchedule.get(), refSchedule, stagingLimitClamped);
        }
        prevOp = schedOp;
    }

    return bufferedSchedule;
}


void Scheduler::ProposeOperatorBuffering(SchedulerOperation *schedOp, SchedulerOperation *prevOp,
    Schedule *bufferedSchedule, Schedule *refSchedule, int stagingLimitBytes)
{
    // Mild recursion might mean this Op has already been seen
    if ( bufferedSchedule->Cost(schedOp) != nullptr )
    {
        return;
    }

    // Take the reference schedule as default costings for this op
    auto refCost = refSchedule->Cost(schedOp);
    assert(refCost != nullptr);
    auto costCopy = std::make_unique<SchedulerOpInfo>(*refCost);
    auto cost = costCopy.get();
    bufferedSchedule->SetCost(*schedOp, std::move(costCopy));

    // SubOps don't currently have slack or buffering (only the parent)
    if ( schedOp->Parent() )
    {
        cost->slackBufferingMemory = 0;
        cost->slackBufferingCycles = 0;
        return;
    }

    // Don't buffer non NPU operations
    if ( !schedOp->IsNpuOp() )
    {
        return;
    }

    // Snapshot may already contain this op's buffering, which the weight buffering function does not want
    const auto &lrmem = bufferedSchedule->MemoryAt(refCost->timeIndex);
    int used = std::max(lrmem.buffering - int(refCost->bufferedWeightTensor.AllocatedSize()), 0) + lrmem.Unbuffered();
    // TODO: This is an unstable removal because you don't know which buffers are currently contributing.
    // Instead the underlying memory tracking needs changing to use the liverange mechanism directly.
    cost->slackBufferingMemory = std::max(stagingLimitBytes - used, 0);
    cost->slackBufferingCycles = refCost->cycles.opCycles;

    // Attempt weight buffering on anything with a weights tensor
    auto weights = schedOp->TryInput(TensorUsage::Weights);
    if ( weights != nullptr )
    {
        auto scales = schedOp->Input(TensorUsage::Scales);
        Buffering method = ProposeWeightBuffering(weights, scales, schedOp, prevOp, bufferedSchedule, refSchedule, cost->slackBufferingMemory);

        // The cost of buffering needs propagating across the memory snapshot if we're
        // sub schedule buffering.
        if ( method != Buffering::None )
        {
            int partSize0 = cost->bufferedWeightTensor.tensor[0] ? cost->bufferedWeightTensor.tensor[0]->AllocationSizeBytes() : 0;
            int partSize1 = cost->bufferedWeightTensor.tensor[1] ? cost->bufferedWeightTensor.tensor[1]->AllocationSizeBytes() : 0;
            int peakBuffering = partSize0 + partSize1;

            bufferedSchedule->memorySnapshot[cost->timeIndex].buffering += peakBuffering;
            if ( cost->bufferedWeightTensor.preBuffer && cost->timeIndex > 0 )
            {
                bufferedSchedule->memorySnapshot[cost->timeIndex - 1].nonlocal += partSize0;
            }
        }
    }
}

namespace
{

unsigned AdjacentWeightConsumers(const std::vector<std::unique_ptr<SchedulerOperation>> &ops, SchedulerTensor *weights,
    SchedulerOperation *schedOp, SchedulerOpInfo *cost, SchedulerOperation *prevOp, SchedulerOpInfo *prevCost, Schedule *refSchedule)
{
    unsigned adjacentConsumers = 1;
    // TODO: Replace this mechanism with a weight tensor memory copy to a tensor
    // located in staging.
    if ( weights->consumers.size() > 1u )
    {
        // Check for special case where several consecutive ops have the same weight tensor.
        // If the weights can fit entire in bufferLimit and the ops all have the same ofm depth slice
        // only one dma transfer will be needed for the ops, see CoalesceWeightBufferTensors.
        // If this is the case ignore prebuffering and instead force a full depth slice
        auto cmpOp = prevOp;
        auto cmpCost = prevCost;
        SchedulerConnection *cmpWeights = nullptr;

        if ( prevOp == nullptr && schedOp->Index() == 0 && ops.size() > 1 )
        {
            // First op in schedule, so check with next op instead
            cmpOp = ops[1].get();
            cmpCost = refSchedule->Cost(cmpOp);
        }

        if ( cmpOp != nullptr )
        {
            cmpWeights = cmpOp->TryInput(TensorUsage::Weights);
        }

        if ( cmpWeights != nullptr && cmpCost != nullptr )
        {
            if ( cmpWeights->tensor->equivalenceId == weights->equivalenceId &&
                 cmpCost->ofmDepthSlices == cost->ofmDepthSlices && cmpOp->weightDepthOffset == schedOp->weightDepthOffset )
            {
                adjacentConsumers = weights->consumers.size();
            }
        }
    }

    return adjacentConsumers;
}

Buffering TryWeightBuffering(NpuWeightTensor *npuWeightsTensor, int bufferLimitBytes, int partSizes[2])
{
    const int encodedWeightsSize = npuWeightsTensor->AllocationSizeBytes();

    // Determine whether to double buffer or single buffer
    int doubleBufferSize = npuWeightsTensor->doubleBufferSizes[0] + npuWeightsTensor->doubleBufferSizes[1];
    if ( (doubleBufferSize <= bufferLimitBytes) && (npuWeightsTensor->maxRangeBytes < encodedWeightsSize) )
    {
        if ( partSizes )
        {
            partSizes[0] = npuWeightsTensor->doubleBufferSizes[0];
            partSizes[1] = npuWeightsTensor->doubleBufferSizes[1];
        }
        return Buffering::Double;
    }

    // Only buffer weights if there's still space left for the buffer
    int weightBufferSize = std::min(encodedWeightsSize, npuWeightsTensor->maxRangeBytes);
    if ( weightBufferSize <= bufferLimitBytes )
    {
        if ( partSizes )
        {
            partSizes[0] = weightBufferSize;
            partSizes[1] = 0;
        }
        return Buffering::Single;
    }

    // No buffering
    partSizes[0] = partSizes[1] = 0;
    return Buffering::None;
}

bool PreferFutureOpPerformance(SchedulerOperation *schedOp, Schedule *refSchedule, const EstimatedPerf &perf,
    SchedulerOpInfo *curCost, SchedulerOpInfo *prevCost, int bufferingSize)
{
    auto ofm = schedOp->FinalSubOFM();

    // Future op is not op dominated, it has weights, and its IFM is a problem
    SchedulerOperation *consumerOp = ofm->tensor->consumers.empty() ? nullptr : ofm->tensor->consumers.front();

    SchedulerOpInfo *consumerCost = consumerOp ? refSchedule->Cost(consumerOp) : nullptr;
    if ( consumerCost && !consumerCost->perf.OpDominated() && consumerCost->perf.IfmRatio() > 2.0f &&
         (consumerCost->perf.weightReadCycles > 0) && (consumerCost->perf.ifmSizeBytes < curCost->slackBufferingMemory) )
    {
        // This op's OFM is the future op's IFM and it doesn't all fit
        if ( (consumerCost->perf.ifmSizeBytes + perf.ifmSizeBytes + bufferingSize) > prevCost->slackBufferingMemory )
        {
            // Future IFM reads are worse than my IFM reads, abort buffering
            // TODO: Test and choose to keep the correct combination of IFM/WGT/OFM
            if ( curCost->perf.OpDominated() || (consumerCost->perf.ifmReadCycles > perf.ifmReadCycles) )
            {
                return true;
            }
        }
    }
    return false;
}


}  // namespace

EstimatedPerf Scheduler::EstimateSlicedOpPerformance(SchedulerOperation *schedOp, const std::vector<int> &depthSlices, ArchitectureOpConfig *opConfig,
    NpuWeightTensor *weights, Flags<StagingPref> stageFlags, const Point2i stripe, int slackCycles, Buffering buffering)
{
    EstimatedPerf result{};

    auto *ifm = schedOp->IFM(0);
    auto *ofm = schedOp->OFM();
    double stripeRepeats = std::max(1.0, double(ofm->SliceShape().Height()) / stripe.y);
    Flags<WeightFormat> weightFormat;
    ArchitectureMemory *wgtMemory = nullptr;
    if ( weights )
    {
        weightFormat = weights->config->Format();
        wgtMemory = (stageFlags % StagingPref::Weights) ? _arch->StagingMemory().memory : weights->memArea.memory;
    }

    auto query = InitPerfQuery(schedOp, opConfig, 1, weightFormat, nullptr, nullptr);
    query.weightStagingMemory = wgtMemory;  // TODO: Design out - stop passing weights via const[]

    if ( stageFlags % StagingPref::IFM ) query.ifmMemory[0] = _arch->StagingMemory().memory;

    // Striped OFM dimensions
    query.ofmShape = Shape(1, stripe.y, stripe.x, 0);
    auto inputArea = GetStripeInputRequirement(query.ofmShape, schedOp->Kernel(), ifm->stepXY, ifm->resamplingMode);
    inputArea = Point2i::Min(inputArea, ifm->SliceShape().WH());
    query.ifmShape[0] = Shape(1, inputArea.y, inputArea.x, ifm->SliceShape().Depth());

    assert(depthSlices.size() > 1);
    const unsigned slices = depthSlices.size() - 1;
    assert(slices);

    int totalWeightFetches = 0;
    // Calculate total IFM and Weight reads
    {
        int ifmRd = 0;
        ElementAccess elementAccess;

        for ( unsigned i = 0; i < slices; i++ )
        {
            int depth = depthSlices[i + 1] - depthSlices[i];
            // Cache results for same-depth slices
            if ( query.ofmShape[-1] != depth )
            {
                query.ofmShape[-1] = depth;
                elementAccess = _arch->Performance()->MeasureElementAccess(query);
                ifmRd = DataTypeStorageSizeBytes(ifm->Type(), elementAccess.ifmRead[0]);
            }
            result.ifmReadBytes += ifmRd;
            if ( weights )
            {
                result.weightReadBytes += weights->rangeSizes[i] * elementAccess.weightsRefetch;
                totalWeightFetches += elementAccess.weightsRefetch;
            }
        }

        result.ifmSizeBytes = DataTypeStorageSizeBytes(ifm->Type(), query.ifmShape[0].Elements());
    }

    // Calculate the full, sliced, operator runtime taking into account
    // any visible delay time spent buffering each slice.
    int64_t visibleBufferCycles = 0;
    int64_t runCycles = 0;
    int64_t transferCycles = 0;
    int64_t sliceCycles = 0;
    OpScheduling sched = (slices == 1) ? OpScheduling::Single : OpScheduling::First;
    bool transferRequired = weights && (weights->memArea.memory != query.weightStagingMemory);
    for ( unsigned i = 0; i < slices; i++ )
    {
        int depth = depthSlices[i + 1] - depthSlices[i];
        if ( i == slices - 1 && i != 0 ) sched = OpScheduling::Last;

        query.ofmShape[-1] = depth;
        query.scheduling = sched;

        sliceCycles = _arch->Performance()->MeasureCycleCost(query).opCycles;

        if ( transferRequired )
        {
            transferCycles = _arch->Performance()->MemToMemCycles(wgtMemory, weights->memArea.memory, weights->rangeSizes[i]);
            // Track non-hidden transfer cycles under the previous slice
            if ( transferCycles > slackCycles ) visibleBufferCycles += transferCycles - slackCycles;
        }

        runCycles += sliceCycles;
        slackCycles = (buffering == Buffering::Double) ? sliceCycles : 0;
        sched = OpScheduling::BackToBack;
    }

    result.lastSliceCycles = sliceCycles;
    result.opRunCycles = int64_t(stripeRepeats * runCycles);
    result.fullRunCycles = result.opRunCycles + visibleBufferCycles;
    result.visibleBufferCycles = int(visibleBufferCycles);
    if ( wgtMemory )
    {
        assert(totalWeightFetches > 0);
        result.weightReadCycles =
            _arch->Performance()->MinReadCycles(wgtMemory, result.weightReadBytes / totalWeightFetches,
                TensorUsage::Weights, schedOp->Type(), weightFormat % WeightFormat::Fast) *
            totalWeightFetches;
    }

    result.ifmReadCycles = _arch->Performance()->MinReadCycles(
        query.ifmMemory[0], result.ifmReadBytes, TensorUsage::IFM, schedOp->Type(), false);

    result.weightReadCycles = int64_t(stripeRepeats * result.weightReadCycles);
    result.weightReadBytes = int64_t(stripeRepeats * result.weightReadBytes);
    result.ifmReadCycles = int64_t(stripeRepeats * result.ifmReadCycles);

    return result;
}

Buffering Scheduler::ProposeWeightBuffering(SchedulerConnection *weights, SchedulerConnection *scales,
    SchedulerOperation *schedOp, SchedulerOperation *prevOp, Schedule *bufferedSchedule, Schedule *refSchedule, int bufferLimitBytes)
{
    auto cost = bufferedSchedule->Cost(schedOp);
    auto refCost = refSchedule->Cost(schedOp);
    auto ofm = schedOp->OFM();

    assert(cost && refCost);
    assert(cost->npuWeightsTensor);

    // Only unstriped operators can be transposed in depth.
    assert((ofm->transpose & TransposeType::MaskC) == TransposeType::C || refCost->stripe == ofm->shape);
    const int fullDepthBeforeTransposition =
        (refCost->stripe == ofm->shape) ? refCost->stripe.Unpermute(uint32_t(ofm->transpose)).Depth() : refCost->stripe.Depth();

    // Default full-weight encoding
    auto encodingParams = _arch->WeightEncoder()->GetEncodingConfig(
        cost->Config(), schedOp->Kernel(), schedOp->IFM(0)->tensor->dataType, cost->npuWeightsTensor->config->Format());

    std::vector<int> ofmFullDepthSlicesBeforeTransposition{0, fullDepthBeforeTransposition};

    auto fullWeightScales = EncodeWeightAndScaleTensor(schedOp->Type(), std::move(encodingParams), schedOp->weightDepthOffset,
        ofmFullDepthSlicesBeforeTransposition, weights->tensor.get(), scales->tensor.get(), weights->quantization, ofm->quantization);

    cost->ofmDepthSlices = {0, refCost->stripe.Depth()};
    cost->SetWeightScaleTensors(fullWeightScales.npuWeightsTensor, fullWeightScales.npuScalesTensor);

    // Attempt different buffering strategies
    if ( !_options.disabled.Any(SchedulerFeature::WeightBuffering) &&
         !ProposeSlicedWeightBuffering(weights, scales, schedOp, prevOp, bufferedSchedule, refSchedule, bufferLimitBytes, fullDepthBeforeTransposition) )
    {
        // Don't slice or buffer - use the whole depth from persistent storage
        cost->bufferedWeightTensor.buffering = Buffering::None;
        cost->bufferedWeightTensor.parts = 0;
        cost->bufferedWeightTensor.tensor[0] = nullptr;
        cost->bufferedWeightTensor.tensor[1] = nullptr;
        cost->bufferedWeightTensor.preBuffer = false;
    }

    return cost->bufferedWeightTensor.buffering;
}

bool Scheduler::ProposeSlicedWeightBuffering(SchedulerConnection *weights, SchedulerConnection *scales, SchedulerOperation *schedOp,
    SchedulerOperation *prevOp, Schedule *bufferedSchedule, Schedule *refSchedule, int bufferLimitBytes, int untransposedFullDepth)
{
    constexpr int OFM_SPLIT_DEPTH = 16;
    auto cost = bufferedSchedule->Cost(schedOp);
    auto refCost = refSchedule->Cost(schedOp);
    auto prevCost = bufferedSchedule->Cost(prevOp);
    if ( !prevCost ) prevCost = refSchedule->Cost(prevOp);
    auto ifm = schedOp->IFM(0);
    auto ofm = schedOp->OFM();

    assert(cost && refCost);

    LOG_TRACE1("#{}: Op '{}' OFM={}  kernel={}  config={}\n", schedOp->Index(), OpTypeToString(schedOp->Type()),
        schedOp->OFM()->SliceShape().ToString(), schedOp->Kernel()->ToString(), cost->Config()->ToString(true));

    enum class BufferingChoice
    {
        None,
        ForceFullDepth,
        Sliced,
        SingleSlice,
    } method = BufferingChoice::None;

    // Weights start in permanent storage. When permanent storage differs from feature map storage,
    // there is a point in moving the data
    auto weightTens = weights->tensor.get();
    auto scaleTens = scales->tensor.get();

    // No need to move the weights if they are already in the same memory as the staging area
    // or weight buffering is disabled.
    auto *stagingMemory = _arch->StagingMemory().memory;
    const int fullWeightsBytes = cost->npuWeightsTensor->AllocationSizeBytes();

    // Estimate the buffering cycle time for the full set of weights
    int64_t fullTransferCycles = _arch->Performance()->MemToMemCycles(stagingMemory, weightTens->memArea.memory, fullWeightsBytes);
    cost->fullWeightTransferCycles = fullTransferCycles;

    // Buffering needs DMA if staging is different to storage
    if ( weightTens->memArea.memory != stagingMemory )
    {
        // Cascade stripe count increases the weight refetch
        const int stripeRepeats =
            cost->cascade != 0 && cost->stripe.Height() ? DivRoundUp(ofm->SliceShape().Height(), cost->stripe.Height()) : 1;

        const auto weightFormat = cost->npuWeightsTensor->config->Format();
        const int refetches = refCost->elementAccess.weightsRefetch * stripeRepeats;

        // Cycles taken to 'see' all the weights when left unbuffered
        int64_t cyclesUnmoved =
            _arch->Performance()->MinReadCycles(weightTens->memArea.memory, fullWeightsBytes, TensorUsage::Weights,
                schedOp->Type(), weightFormat % WeightFormat::Fast) *
            refetches;

        // Cycles taken to 'see' all the weights when buffered
        int64_t cyclesFromBuffer =
            _arch->Performance()->MinReadCycles(stagingMemory, fullWeightsBytes, TensorUsage::Weights, schedOp->Type(), weightFormat % WeightFormat::Fast) * refetches;

        // We don't expect reads from staging to be slower
        if ( cyclesFromBuffer >= cyclesUnmoved )
        {
            return false;  // No buffering
        }

        // See if we can buffer in depth-slices
        const bool transposedInDepth = (ofm->transpose & TransposeType::MaskC) != TransposeType::C;
        bool canSlice =
            (_arch->CanSubdivide(schedOp->Type(), ofm->transpose, ofm->reverse) != AxisMask::None) &&
            (ofm->reverse != ReverseType::C) && !transposedInDepth;

        // Cascades shouldn't be depth sliced at this time
        canSlice = canSlice && (cost->cascade == 0);

        const int fullConvDepth =
            (transposedInDepth && (refCost->stripe == ofm->shape)) ? untransposedFullDepth : refCost->stripe.Depth();

        // How many NPU cycles are available under the previously executing
        // operator for performing buffered DMA transfers
        int64_t slackCycles = (prevCost != nullptr) ? prevCost->slackBufferingCycles : 0;
        const int slackMemory = (prevCost != nullptr) ? prevCost->slackBufferingMemory : bufferLimitBytes;
        const int64_t smallestTransferTime = int64_t(fullTransferCycles * double(std::min(fullConvDepth, OFM_SPLIT_DEPTH)) / fullConvDepth);

        // Partial check for prebuffering (clarified later) - we cannot prebuffer during cascades or at start of network
        const bool allowPreBuffer = (schedOp->Index() != 0) && (cost->cascade == 0 || cost->firstInCascade);

        // Allow initial buffering to occur at the start of a schedule only if it looks promising (tweakable)
        const int STARTUP_DELAY_CYCLES = 1000;  // TODO: Affects small networks, model better in performance estimator
        // If no preceding op, a transfer cost of 10% of the unmoved weight read time is enough to try buffering
        const bool forceInitialBuffer =
            (!prevOp || !prevOp->IsNpuOp()) && (refetches > 1) &&
            ((smallestTransferTime < cyclesUnmoved / 10) || (smallestTransferTime <= STARTUP_DELAY_CYCLES));

        // Does it take longer to run the op than it does to see all the unbuffered weights
        const bool opcycleDominated = refCost->perf.opRunCycles > cyclesUnmoved;

        // Pre-select a buffering method to shortcut wasteful weight encoding attempts.
        {
            const bool canFullDepthBuffer = (fullWeightsBytes <= bufferLimitBytes);
            const unsigned weightConsumers = AdjacentWeightConsumers(_ops, weightTens, schedOp, cost, prevOp, prevCost, refSchedule);
            const int64_t visibleFullTransferCycles = std::max<int64_t>(fullTransferCycles - slackCycles, 0);
            const int64_t cyclesFullBuffered = cyclesFromBuffer + visibleFullTransferCycles;

            if ( canSlice )
            {
                int expectedSlices = fullConvDepth / cost->Config()->OptimalDepthGranule();
                // Try slicing if buffering is better than the op runtime, or if there's enough time to
                // hide some buffering either before or inbetween slices.
                if ( forceInitialBuffer || (cyclesFullBuffered < cyclesUnmoved) || (expectedSlices > 2) ||
                     (allowPreBuffer && (smallestTransferTime < slackCycles && (refetches > 1))) )
                {
                    method = BufferingChoice::Sliced;
                }
            }

            // Full depth buffer if we have to or if full buffering is sufficiently hidden (~95% covered,
            // or leaves less than 1/20th of the transfer cycles visible)
            int64_t staticTestCycles = DivRoundUp<int64_t>(visibleFullTransferCycles, stripeRepeats);
            bool fullBufferingHidden = (visibleFullTransferCycles < (fullTransferCycles / 20)) || (staticTestCycles < STARTUP_DELAY_CYCLES);
            if ( canFullDepthBuffer && (weightConsumers > 1 || (!canSlice && (fullBufferingHidden || forceInitialBuffer))) )
            {
                if ( (cyclesFullBuffered < cyclesUnmoved) || !opcycleDominated )
                {
                    // Must attempt full depth for the weight coalescing
                    method = (weightConsumers > 1) ? BufferingChoice::ForceFullDepth : BufferingChoice::SingleSlice;
                }
            }
        }

        LOG_TRACE1("\t\tTry Method={}\n",
            method == BufferingChoice::None ?
                "None" :
            (method == BufferingChoice::Sliced) ?
                "sliced" :
                "full");

        if ( method == BufferingChoice::None )
        {
            return false;  // No buffering
        }

        // Calculate the amount of pre-buffering necessary (or what is possible with a limited
        // double-buffer buffer size)
        const int halfBufferLimit = bufferLimitBytes / 2;
        double prebufferRatio = 1.0;  // start at full depth

        if ( method == BufferingChoice::Sliced )
        {
            const double minSplitRatio = std::min(double(OFM_SPLIT_DEPTH) / fullConvDepth, 1.0);
            const double halfSplitRatio = std::min(double(halfBufferLimit) / fullWeightsBytes, 1.0);
            const double transferRatio = std::min(double(slackCycles) / fullTransferCycles, 1.0);

            // Align and discard a prebuffer that leaves a singular small tail piece
            double aligned = std::floor(transferRatio / minSplitRatio) * minSplitRatio;
            prebufferRatio = (1.0 - aligned) < (minSplitRatio / 2) ? 1.0 : transferRatio;

            if ( opcycleDominated && (prebufferRatio > 0.5) && (prebufferRatio < 1.0) )
            {
                // Round 50% up to nearest split depth
                prebufferRatio = RoundAway(std::floor(fullConvDepth * 0.5), OFM_SPLIT_DEPTH) / fullConvDepth;
                prebufferRatio = std::clamp(prebufferRatio, 0.5, 1.0);
            }

            prebufferRatio = std::min(std::max(prebufferRatio, minSplitRatio), halfSplitRatio);
        }

        std::vector<int> untransposedDepthSlices{0, fullConvDepth};
        WeightScaleTensors encodedWeightScales{cost->npuWeightsTensor, 0, cost->npuScalesTensor};

        const int configBlockDepth = cost->Config()->OptimalDepthGranule();
        const int configMinDepth = cost->Config()->MinimumDepthGranule();

        // Have to split the weights if the initial buffering can't store
        // all of the compressed weights
        if ( prebufferRatio < 1.0 )
        {
            int bufferingDepth = 0;

            // Choose initial pre-buffering depth (already buffer clamped)
            int prebufferDepth = int(fullConvDepth * prebufferRatio);
            prebufferDepth = std::max(RoundZero(prebufferDepth, configMinDepth), OFM_SPLIT_DEPTH);
            if ( prebufferDepth < fullConvDepth )
            {
                // Calculate cycles executed during the pre-buffer
                assert(prebufferDepth > 0);
                auto preOpCycles = EstimateOpPerformance(
                    schedOp, cost->Config(), prebufferDepth, weightFormat, stagingMemory, OpScheduling::First);

                bufferingDepth = int((fullConvDepth * preOpCycles.opCycles) / fullTransferCycles);
                bufferingDepth = RoundAway(std::max(bufferingDepth, OFM_SPLIT_DEPTH), configBlockDepth);

                // Choose initial buffering depth and clamp to the double buffering limit
                int bufferingBytes = int(double(bufferingDepth) / fullConvDepth * fullWeightsBytes);
                if ( bufferingBytes > halfBufferLimit )
                {
                    bufferingDepth = int(double(halfBufferLimit) / fullWeightsBytes * fullConvDepth);
                }
            }

            while ( bufferingDepth != 0 )
            {
                // Attempt to buffer whole blocks
                bufferingDepth = RoundZero(bufferingDepth, (bufferingDepth > configBlockDepth) ? configBlockDepth : configMinDepth);
                bufferingDepth = std::max(bufferingDepth, configMinDepth);

                // Create list of depth slices
                untransposedDepthSlices.clear();
                untransposedDepthSlices.push_back(0);
                for ( int depth = prebufferDepth; depth < fullConvDepth; depth += bufferingDepth )
                {
                    assert(prebufferDepth != 0);
                    untransposedDepthSlices.push_back(depth);
                }
                untransposedDepthSlices.push_back(fullConvDepth);

                auto encodingParams = _arch->WeightEncoder()->GetEncodingConfig(
                    cost->Config(), schedOp->Kernel(), ifm->tensor->dataType, weightFormat);

                encodedWeightScales = EncodeWeightAndScaleTensor(schedOp->Type(), std::move(encodingParams), schedOp->weightDepthOffset,
                    untransposedDepthSlices, weightTens, scaleTens, weights->quantization, ofm->quantization);

                // Chosen buffering might not fit at all, iterate until it does
                // or until the minimum usable slice size is reached
                if ( (encodedWeightScales.npuWeightsTensor->maxRangeBytes <= halfBufferLimit) ||
                     (prebufferDepth == OFM_SPLIT_DEPTH && bufferingDepth == configMinDepth) )
                {
                    break;
                }

                // Failed to choose buffer sizes above, reduce them and try again
                if ( bufferingDepth > prebufferDepth )
                {
                    bufferingDepth = RoundAway(bufferingDepth / 2, configMinDepth);
                }
                else
                {
                    prebufferDepth = RoundAway(prebufferDepth / 2, OFM_SPLIT_DEPTH);
                }
            }

            // Utilisation is only checked for intermediate slices (not
            // prebuffer and tail slices as they are allowed to be smaller)
            if ( untransposedDepthSlices.size() > 3 )
            {
                assert(bufferingDepth);
                float blockUtilisation = float(bufferingDepth) / RoundAway(bufferingDepth, configBlockDepth);
                // Less than 30% of the block config depth is considered underutilised
                if ( blockUtilisation < 0.3f )
                {
                    return false;  // No buffering
                }
            }
        }

        assert(untransposedDepthSlices.size() >= 2);

        // Weight buffering can be either single or multi-slice as a result of above choices
        int partSizes[2] = {};
        auto buffering = TryWeightBuffering(encodedWeightScales.npuWeightsTensor.get(), bufferLimitBytes, partSizes);
        if ( buffering == Buffering::None )
        {
            return false;  // No buffering
        }

        int usableSlack = (allowPreBuffer && partSizes[0] < slackMemory) ? slackCycles : (forceInitialBuffer ? STARTUP_DELAY_CYCLES : 0);
        cost->stagingPreference = StagingPref::None;

        Flags<StagingPref> expected(StagingPref::Weights, cost->SourcesCascadeBuffer() ? StagingPref::IFM : StagingPref::None);
        EstimatedPerf perf = EstimateSlicedOpPerformance(schedOp, untransposedDepthSlices, cost->Config(),
            encodedWeightScales.npuWeightsTensor.get(), expected, cost->stripe.WH(), usableSlack, buffering);

        // Evaluate sliced buffering performance
        if ( method == BufferingChoice::Sliced || method == BufferingChoice::SingleSlice )
        {
            int64_t ifmReadCycles = perf.ifmReadCycles;

            // Compare to IFM only if IFM can be moved.
            if ( !ifm->tensor->producers.empty() && !cost->SourcesCascadeBuffer() &&
                 !IsDataLayout(ifm->tensor->producers[0]->Type()) )
            {
                bool bothFit = (perf.ifmSizeBytes + partSizes[0] + partSizes[1]) <= bufferLimitBytes;

                // Contrast just staging the old IFM against staging the weight slices
                int64_t ifmOldStagedReadCycles = _arch->Performance()->MinReadCycles(_arch->StagingMemory().memory,
                    refCost->perf.ifmReadBytes, TensorUsage::IFM, schedOp->Type(), false);
                int64_t newPerfIfm = std::max({refCost->perf.opRunCycles, refCost->perf.weightReadCycles, ifmOldStagedReadCycles});
                int64_t newPerfWgt = std::max({perf.fullRunCycles, perf.weightReadCycles, perf.ifmReadCycles});

                if ( !bothFit && (perf.ifmSizeBytes < prevCost->slackBufferingMemory) && (newPerfIfm < newPerfWgt) &&
                     (newPerfIfm < perf.ifmReadCycles) )
                {
                    cost->stagingPreference |= StagingPref::IFM;
                    return false;  // No buffering
                }

                // Check future op
                if ( PreferFutureOpPerformance(schedOp, refSchedule, perf, cost, prevCost, partSizes[0] + partSizes[1]) )
                {
                    return false;  // No buffering
                }

                if ( bothFit )
                {
                    cost->stagingPreference |= StagingPref::IFM;
                    ifmReadCycles = _arch->Performance()->MinReadCycles(
                        _arch->StagingMemory().memory, perf.ifmReadBytes, TensorUsage::IFM, schedOp->Type(), false);
                }
            }

            // Check that performance is improved and the choices don't underutilise the system (technically
            // multiplied by stripeRepeats, but not needed for relative comparison). This guesses where the IFM
            // will be staged (same for both choices).
            int64_t originalPerf = std::max({refCost->perf.opRunCycles, refCost->perf.weightReadCycles, ifmReadCycles});
            int64_t newPerf = std::max({perf.fullRunCycles, perf.weightReadCycles, ifmReadCycles});

            // POOR PERFORMANCE ESTIMATION COMPENSATION: Remove scheduling noise from
            // op-dominated estimates. Because the estimator always measures true op-
            // dominated ops as being the same, regardless of slicing, it's hard to
            // tell if the startup delay is relevant or not.
            unsigned slices = untransposedDepthSlices.size() - 1;
            if ( opcycleDominated )
            {
                int visiblePerSlice = perf.visibleBufferCycles / slices;
                if ( visiblePerSlice < STARTUP_DELAY_CYCLES )
                {
                    newPerf -= perf.visibleBufferCycles;
                }
                // If we were op dominated and we are now sliced and still op dominated then remove
                // scheduling 'noise' from the slicing. This observes that cutting an OFM into pieces
                // allows some scheduling overlap that the estimator isn't measuring.
                if ( (slices > 1) && (perf.fullRunCycles == newPerf) && forceInitialBuffer )
                {
                    newPerf -= slices * (STARTUP_DELAY_CYCLES / 2);
                }
            }

            // Take the buffering option if it looks the SAME or BETTER
            if ( (originalPerf < newPerf) || (refetches == 1 && (fullTransferCycles + perf.weightReadCycles > originalPerf)) )
            {
                return false;  // No buffering
            }
        }

        // Commit to the encoded weights based on these depth slices
        if ( !transposedInDepth )
        {
            cost->ofmDepthSlices = std::move(untransposedDepthSlices);
        }
        else
        {
            assert(untransposedDepthSlices.size() == 2);
            cost->ofmDepthSlices = {0, refCost->stripe.Depth()};
        }
        cost->perf = perf;

        // Create a new tensor in fast storage to use as weights buffer
        cost->bufferedWeightTensor.tensor[0] = std::make_shared<SchedulerTensor>(DataType::UInt8, Shape(partSizes[0]));
        cost->bufferedWeightTensor.tensor[0]->SetInternalName(fmt::format(
            "WB_{}_{}_op{}", buffering == Buffering::Double ? "double" : "single", partSizes[0], schedOp->Index()));
        cost->bufferedWeightTensor.tensor[0]->SetAllocatedSize(partSizes[0]);
        cost->bufferedWeightTensor.tensor[0]->memArea = _arch->StagingMemory();
        cost->bufferedWeightTensor.parts = 1;

        if ( buffering == Buffering::Double )
        {
            cost->bufferedWeightTensor.tensor[1] = std::make_shared<SchedulerTensor>(DataType::UInt8, Shape(partSizes[1]));
            cost->bufferedWeightTensor.tensor[1]->SetInternalName(
                fmt::format("WB_double2_{}_op{}", partSizes[1], schedOp->Index()));
            cost->bufferedWeightTensor.tensor[1]->SetAllocatedSize(partSizes[1]);
            cost->bufferedWeightTensor.tensor[1]->memArea = _arch->StagingMemory();
            cost->bufferedWeightTensor.parts = 2;
        }

        cost->bufferedWeightTensor.buffering = buffering;
        cost->bufferedWeightTensor.preBuffer = allowPreBuffer && (partSizes[0] < slackMemory);
        cost->slackBufferingCycles = perf.lastSliceCycles;
        cost->stagingPreference |= StagingPref::Weights;
        const unsigned lastSlicePart = (cost->ofmDepthSlices.size() - 2) % cost->bufferedWeightTensor.parts;
        cost->slackBufferingMemory -= partSizes[lastSlicePart];
        cost->SetWeightScaleTensors(encodedWeightScales.npuWeightsTensor, encodedWeightScales.npuScalesTensor);
    }

    LOG_TRACE1("\t\tChose Method={}\n",
        method == BufferingChoice::None ?
            "None" :
        (method == BufferingChoice::Sliced && cost->ofmDepthSlices.size() > 2) ?
            "sliced" :
            "full");

    return (method != BufferingChoice::None);
}


std::shared_ptr<Schedule> Scheduler::ProposeMinimalSchedule()
{
    // Proposes scheduling parameters where every operator is subdivided into the smallest stripe that
    // satisfies the next operators stride
    auto minSchedule = std::make_shared<Schedule>(_name + "_MIN", 0, _ops.size());

    // Keep track of the previous Op - which consumes the current Op's OFM
    SchedulerOperation *prevOp = nullptr;

    // Work backwards up the schedule setting the minimum stripe height
    for ( auto pos = _ops.rbegin(); pos != _ops.rend(); pos++ )
    {
        auto const &schedOp = *pos;
        const auto ofm = schedOp->OFM();
        const auto &ofmShape = ofm->SliceShape();
        int minStripeHeight = 1;
        if ( prevOp )
        {
            // Accumulator keep decomposed ops have special shape requirements and can't currently be striped
            const bool isNextAccKeep = prevOp->AccumulatorMode().source == AccumulatorSource::Acc;
            const bool isAccKeep = schedOp->AccumulatorMode().source == AccumulatorSource::Acc;
            minStripeHeight = isNextAccKeep || isAccKeep ? ofmShape.Height() : prevOp->Kernel()->Stride().y;
        }
        Shape minStripe = Shape::PadAxes(ofmShape, 3, 1).WithHeight(minStripeHeight);
        // Align stripe based on resamplingMode
        minStripe = AlignStripe(schedOp.get(), minStripe);
        auto cost = CreateSchedulerOpInfo(schedOp.get(), minStripe);
        for ( auto &subOp : schedOp->SubOps() )
        {
            const auto subOfm = subOp->OFM();
            const auto &subOfmShape = subOfm->SliceShape();
            auto subCost = CreateSchedulerOpInfo(subOp.get(), minStripe, cost);
            minSchedule->SetCost(*subOp, std::move(subCost));
        }

        if ( ofmShape && schedOp->IsNpuOp() )
        {
            auto query = InitPerfQuery(
                schedOp.get(), cost->Config(), ofmShape.Depth(), WeightFormat::Default, cost.get(), minSchedule.get());
            cost->cycles = _arch->Performance()->MeasureCycleCost(query);
            cost->elementAccess = _arch->Performance()->MeasureElementAccess(query);
            cost->perf = EstimateSlicedOpPerformance(schedOp.get(), cost->ofmDepthSlices, cost->Config(),
                cost->npuWeightsTensor.get(), StagingPref::None, cost->stripe.WH(), 0, Buffering::None);
        }

        minSchedule->SetCost(*schedOp, std::move(cost));
        prevOp = schedOp.get();
    }

    return minSchedule;
}


std::shared_ptr<Schedule> Scheduler::OptimizeSchedule(Schedule *schedule, const std::shared_ptr<Schedule> &maxSchedule, Address stagingLimitBytes)
{
    // Extracts sub-schedules based on the cascades and optimizes them and applies them to the final schedule
    if ( maxSchedule->fastStoragePeakUsage < stagingLimitBytes && !_spilling )
    {
        return maxSchedule;
    }

    // Optimize cascades separately
    // Iterate over a copy of the cascades since they may change during the loop
    auto cascades = schedule->cascades;
    for ( const auto &pos : cascades )
    {
        const CascadeInfo &cascadeInfo = pos.second;

        auto optSubSchedule = OptimizeSubSchedule(cascadeInfo, schedule, stagingLimitBytes);
        if ( optSubSchedule != nullptr )
        {
            // Remove the existing cascade
            schedule->cascades.erase(pos.first);
            // Move subschedule costs/cascades back into the schedule
            SchedulerCostMap costs;
            optSubSchedule->DetachCosts(costs);
            schedule->UpdateCosts(costs);
            schedule->UpdateCascades(optSubSchedule->cascades);
        }
    }

    // Update memory snapshot
    UpdateOpMemorySnapshot(schedule);

    // Propose schedule buffering to the optimized schedule
    auto optSchedule = ProposeScheduleBuffering(schedule, stagingLimitBytes);
    optSchedule->cascades = std::move(schedule->cascades);  // TODO: Check this is okay
    // Copy the cascade's metadata from the unbuffered schedule
    return optSchedule;
}


std::shared_ptr<Schedule> Scheduler::ProposeScheduleStriping(const Shape &finalStripe, const std::string &label, Schedule *refSchedule)
{
    // Proposes new striping for a schedule. The stripe is derived from the ifm requirements of the next Op down
    auto stripedSchedule = std::make_shared<Schedule>(label, refSchedule->Start(), refSchedule->End());

    Shape stripe = finalStripe;
    for ( int pos = refSchedule->End() - 1; pos >= refSchedule->Start(); pos-- )
    {
        auto *schedOp = _ops.at(pos).get();
        auto refCost = refSchedule->Cost(schedOp);
        if ( !schedOp->IsNpuOp() || refCost == nullptr )
        {
            // sched_op is not part of the sub-schedule - skip
            continue;
        }
        // align stripe based on resampling mode
        stripe = AlignStripe(schedOp, stripe);
        // Create a cost entry with the new stripe
        auto cost = CreateSchedulerOpInfo(schedOp, stripe);
        cost->timeIndex = refCost->timeIndex;
        for ( auto &subOp : schedOp->SubOps() )
        {
            const auto subOfm = subOp->OFM();
            const auto &subOfmShape = subOfm->SliceShape();
            auto subCost = CreateSchedulerOpInfo(subOp.get(), stripe, cost);
            auto subRefCost = refSchedule->Cost(subOp.get());
            if ( subRefCost )
            {
                subCost->timeIndex = subRefCost->timeIndex;
            }
            stripedSchedule->SetCost(*subOp, std::move(subCost));
        }

        // Cascades cannot currently use sliced buffering so take only single buffering choices from the reference
        // schedule.
        // TODO: Replace with in-loop buffering
        if ( (refCost->bufferedWeightTensor.buffering == Buffering::Single) && refCost->bufferedWeightTensor.tensor[0] )
        {
            assert(cost->npuWeightsTensor);
            auto bufferingTensor = std::make_shared<SchedulerTensor>(DataType::UInt8, Shape(cost->npuWeightsTensor->AllocationSizeBytes()));
            bufferingTensor->SetAllocatedSize(cost->npuWeightsTensor->AllocationSizeBytes());
            bufferingTensor->memArea = refCost->bufferedWeightTensor.tensor[0]->memArea;
            cost->bufferedWeightTensor.buffering = Buffering::Single;  // Stripes are currently single-buffered
            cost->bufferedWeightTensor.preBuffer = false;
            cost->bufferedWeightTensor.tensor[0] = std::move(bufferingTensor);
            cost->bufferedWeightTensor.parts = 1;
        }

        // Estimate performance
        auto query = InitPerfQuery(schedOp, cost->Config(), schedOp->OFM()->SliceShape().Depth(), WeightFormat::Default,
            cost.get(), stripedSchedule.get());
        cost->cycles = _arch->Performance()->MeasureCycleCost(query);
        cost->elementAccess = _arch->Performance()->MeasureElementAccess(query);
        cost->perf = EstimateSlicedOpPerformance(schedOp, cost->ofmDepthSlices, cost->Config(),
            cost->npuWeightsTensor.get(), StagingPref::None, cost->stripe.WH(), 0, Buffering::None);

        stripedSchedule->SetCost(*schedOp, std::move(cost));

        // Calculate the preceeding Op's stripe
        int upscaling = 1;
        if ( auto ifm = schedOp->TryIFM(0) )
        {
            int rounding;
            upscaling = _arch->UpscaleAndRounding(ifm->resamplingMode, rounding);
        }
        assert((stripe.Height() % upscaling == 0) && "stripe height is not aligned with the upscaling-factor");
        stripe = schedOp->IFM(schedOp->PrimaryIfmIndex())
                     ->shape.With(-3, (stripe.Height() / upscaling) * schedOp->Kernel()->Stride().y);
    }
    return stripedSchedule;
}


Address Scheduler::EstimateScheduleMemoryUsage(Schedule *schedule, const std::unordered_map<UniqueId, int> &nonLocalMem)
{
    // Estimates the memory usage of a schedule AS IF it was all placed
    // in fastest staging storage.
    int peakMemUsage = 0;
    for ( int i = schedule->Start(); i < schedule->End(); i++ )
    {
        const auto *schedOp = _ops.at(i).get();
        auto cost = schedule->Cost(schedOp);
        if ( cost == nullptr )
        {
            continue;
        }

        if ( cost->cascade != 0 )
        {
            // This Op is part of a cascade - use the cascade's memory usage
            auto const &cascadeInfo = schedule->cascades.at(cost->cascade);
            // Non-local memory usage is already included in the cascade_info
            peakMemUsage = std::max(cascadeInfo.memUsage, peakMemUsage);
        }
        else
        {
            // This Op is not part of a cascade - calculate the memory usage
            // (TODO: lacks detail about IFM reuse improve or replace)
            int opMemUsage = cost->bufferedWeightTensor.AllocatedSize();

            opMemUsage += schedOp->IFM(0)->PartialAllocationSizeBytes() + schedOp->OFM()->PartialAllocationSizeBytes();

            if ( nonLocalMem.find(*schedOp) != nonLocalMem.end() )
            {
                opMemUsage += nonLocalMem.at(*schedOp);
            }

            auto ifm1 = schedOp->TryIFM(1);
            if ( ifm1 )
            {
                opMemUsage += ifm1->PartialAllocationSizeBytes();
            }

            peakMemUsage = std::max(opMemUsage, peakMemUsage);
        }
    }
    return peakMemUsage;
}


std::shared_ptr<Schedule> Scheduler::OptimizeSubSchedule(const CascadeInfo &cascadeInfo, Schedule *refSchedule, Address stagingLimitBytes)
{
    // Extracts the Ops covered by the given cascade and creates a sub-schedule. The sub-schedule is optimized by
    // proposing weight buffering and then continuously proposing new stripe sizes

    // Extract the costs for the ops that are part of this sub-schedule and one preceding op (to pass the cost to the
    // subschedule)
    auto subSchedule = std::make_shared<Schedule>(
        _name + fmt::format("SUB_{}_{}", cascadeInfo.start, cascadeInfo.end), cascadeInfo.start, cascadeInfo.end + 1);

    for ( int i = std::max(subSchedule->Start() - 1, 0); i < subSchedule->End(); i++ )
    {
        const auto &op = _ops.at(i);
        // NOTE: Copies the cost objects, consider optimising this
        auto costCopy = std::make_unique<SchedulerOpInfo>(*refSchedule->Cost(op.get()));
        int cascadeFreeMemory = stagingLimitBytes - refSchedule->MemoryAt(costCopy->timeIndex).Used();
        costCopy->slackBufferingMemory = cascadeFreeMemory;
        costCopy->slackBufferingCycles = costCopy->cycles.opCycles;  // Potentially: costCopy->perf.lastSliceCycles
        subSchedule->SetCost(*op, std::move(costCopy));

        // chained sub-operations
        for ( auto &subOp : op->SubOps() )
        {
            costCopy = std::make_unique<SchedulerOpInfo>(*refSchedule->Cost(subOp.get()));
            costCopy->slackBufferingMemory = cascadeFreeMemory;
            subSchedule->SetCost(*subOp, std::move(costCopy));
        }
    }

    // Set correct processing range
    vector_span<std::unique_ptr<SchedulerOperation>> subOps(_ops, cascadeInfo.start, cascadeInfo.end + 1);

    // Update subschedule cascade list
    subSchedule->cascades[cascadeInfo.end] = cascadeInfo;

    // Use the memory snapshot from the reference schedule (takes a copy)
    subSchedule->memorySnapshot = refSchedule->memorySnapshot;

    SchedulerOperation *firstOp = subOps.front().get();

    // Calculate memory usage that is live during the sub-schedule but not part of it
    int timeForCascade = refSchedule->Cost(firstOp)->timeIndex;

    int memUsageParallelToSubSchedule = refSchedule->MemoryUsageAt(timeForCascade) - cascadeInfo.localMemUsage;
    memUsageParallelToSubSchedule = std::max(0, memUsageParallelToSubSchedule);

    // If the first Op's IFM has other consumers it has to live throughout the whole sub-schedule whether it's
    // included in a cascade or not. Not valid if spilling enabled
    int persistentInitialIFM = 0;
    auto firstOpIfm = firstOp->IFM(firstOp->PrimaryIfmIndex());
    if ( !_spilling && firstOpIfm->tensor->consumers.size() > 1 )
    {
        persistentInitialIFM = firstOpIfm->tensor->AllocationSizeBytes();
    }

    // Calculate non-local-mem-usage per Operator
    std::unordered_map<UniqueId, int> nonLocalMemUsage;
    nonLocalMemUsage[*firstOp] = memUsageParallelToSubSchedule;
    for ( int i = 1; i < subOps.size(); i++ )
    {
        nonLocalMemUsage[*subOps[i]] = memUsageParallelToSubSchedule + persistentInitialIFM;
    }

    std::unordered_map<UniqueId, LiveRangeSummary> liveRanges;
    std::unordered_map<UniqueId, int> opLocalMemUsage;
    ComputeNonLocalUsage(refSchedule, &liveRanges, &opLocalMemUsage);
    CascadeBuilder cascadeBuilder(subOps, nonLocalMemUsage, opLocalMemUsage, liveRanges, _spilling);

    // Start by adding buffering
    auto bufferedSubSchedule = ProposeScheduleBuffering(subSchedule.get(), stagingLimitBytes);

    // Copy the cascades over from the unbuffered-schedule
    bufferedSubSchedule->cascades = subSchedule->cascades;

    // Generate the possible stripings for the final Op in the sub-schedule
    Shape finalOFMShape = subOps.back()->OFM()->SliceShape();
    const int maxStripeHeight = (finalOFMShape.Height() + 1) / 2;

    // Skip testing the min stripe used in the MIN schedule since that will be used
    // anyway if no new cascades are created below
    SchedulerOpInfo *minCost = refSchedule->Cost(subOps.back().get());
    const int minStripeHeight = minCost->stripe.Height() + 1;
    const int minStripeHeightStep = minCost->Config()->MinimalStripeGranule().y;

    std::vector<Shape> possibleStripes;
    possibleStripes.reserve(maxStripeHeight / minStripeHeightStep);
    for ( int h = RoundAway(minStripeHeight, minStripeHeightStep); h <= maxStripeHeight; h += minStripeHeightStep )
    {
        possibleStripes.push_back(finalOFMShape.With(-3, h));
    }

    // Propose different striping - the possible stripes are proposed similarly to a binary search
    std::shared_ptr<Schedule> bestSchedule;

#if LOG_PRINT_ON
    LOG_INDENT(Logging::Out);
#endif

    int maxCascadeSize = 0;
    for ( auto &proposedStripe : possibleStripes )
    {
        auto proposedSchedule = ProposeScheduleStriping(
            proposedStripe, fmt::format("_OPT_{}", proposedStripe.Height()), bufferedSubSchedule.get());

        cascadeBuilder.BuildCascades(proposedSchedule.get(), _maxSchedule.get(), stagingLimitBytes);

        int cascadeSize = proposedSchedule->cascades.size();
        if ( maxCascadeSize == 0 )
        {
            // First iteration - used as limit to prevent splitting up the cascades
            // Long cascades are better in order to reduce IFM/OFM dram bandwidth
            maxCascadeSize = cascadeSize;
        }

        // Check if proposal fits
        Address proposedMemUsage = EstimateScheduleMemoryUsage(proposedSchedule.get(), nonLocalMemUsage);

        if ( proposedMemUsage <= stagingLimitBytes && cascadeSize <= maxCascadeSize )
        {
            bestSchedule = proposedSchedule;
            // No cascading required - early exit
            if ( proposedSchedule->cascades.empty() )
            {
                break;
            }
        }
        else
        {
            break;
        }
    }

    return bestSchedule;
}


void Scheduler::ApplySchedule(Schedule *schedule)
{
    const auto idealFormat = _arch->IdealBufferingFormat();

    // Applies the given schedule as the end result
    for ( auto &schedOp : _ops )
    {
        if ( !schedOp->IsNpuOp() )
        {
            continue;
        }

        auto cost = schedule->Cost(schedOp.get());
        if ( cost->cascade > 0 )
        {
            const CascadeInfo &cascadeInfo = schedule->cascades.at(cost->cascade);
            auto pos = cascadeInfo.buffers.find(*schedOp);
            if ( pos != cascadeInfo.buffers.end() )
            {
                auto ifmConn = schedOp->IFM(schedOp->PrimaryIfmIndex());
                auto bufferTensor = ifmConn->tensor.get();
                // Apply memory area
                bufferTensor->memArea = _arch->StagingMemory();
                // Apply rolling buffer dimensions
                Shape bufferShape = pos->second.shape;
                assert(!bufferTensor->needsLinearFormat);
                bufferTensor->format = idealFormat;
                assert(bufferShape.Width() == ifmConn->shape.Width() && "Only y-striping implemented");
                bufferTensor->storageShape = ifmConn->shape.WithHW(bufferShape.Height(), bufferShape.Width());
            }
        }
    }
}


// Coalesce repeated weight buffer tensors
void Scheduler::CoalesceWeightBufferTensors(Schedule *schedule)
{
    SchedulerOpInfo *prevCost = nullptr;
    SchedulerOperation *prevOp = nullptr;

    for ( auto &schedOp : _ops )
    {
        if ( !schedOp->IsNpuOp() )
        {
            continue;
        }

        auto cost = schedule->Cost(schedOp.get());
        if ( prevCost && cost && prevOp )
        {
            auto &prevBufTensor = prevCost->bufferedWeightTensor.tensor;
            auto &bufTensor = cost->bufferedWeightTensor.tensor;
            if ( prevCost->bufferedWeightTensor.parts == cost->bufferedWeightTensor.parts )
            {
                UniqueId prevWeightsTensorId = prevCost->npuWeightsTensor ? prevCost->npuWeightsTensor->equivalenceId : -1;
                UniqueId weightsTensorId = cost->npuWeightsTensor ? cost->npuWeightsTensor->equivalenceId : -2;
                if ( prevWeightsTensorId == weightsTensorId &&
                     prevCost->bufferedWeightTensor.AllocatedSize() == cost->bufferedWeightTensor.AllocatedSize() &&
                     prevCost->ofmDepthSlices.size() == 2 && prevCost->ofmDepthSlices == cost->ofmDepthSlices &&
                     schedOp->weightDepthOffset == prevOp->weightDepthOffset )
                {
                    // Reuse previous weight buffer tensor if both current and previous op use 1 depth slice
                    // This will extend the life range weight buffer tensor
                    bufTensor[0] = prevBufTensor[0];
                    bufTensor[1] = prevBufTensor[1];
                }
            }
        }

        prevCost = cost;
        prevOp = schedOp.get();
    }
}


PerformanceQuery Scheduler::InitPerfQuery(const SchedulerOperation *op, ArchitectureOpConfig *config, int ofmDepth,
    WeightFormat wgtFormat, const SchedulerOpInfo *cost, Schedule *schedule)
{
    PerformanceQuery query = {};
    query.type = op->Type();
    query.kernel = op->Kernel();
    query.config = config;

    const SchedulerConnection *ifm0 = op->IFM(0);
    query.ifmShape[0] = ifm0->SliceShape();
    query.ifmMemory[0] = ifm0->tensor->memArea.memory;
    query.ifmType[0] = ifm0->Type();
    query.ifmFormat[0] = ifm0->tensor->format;

    const SchedulerConnection *ifm1 = op->TryIFM(1);
    if ( ifm1 )
    {
        query.ifmShape[1] = ifm1->SliceShape();
        query.ifmMemory[1] = ifm1->tensor->memArea.memory;
        query.ifmType[1] = ifm1->Type();
        query.ifmFormat[1] = ifm1->tensor->format;
    }

    const SchedulerConnection *ofm = op->OFM();
    ofmDepth = (ofmDepth >= 0) ? ofmDepth : ofm->SliceShape().Depth();
    query.ofmShape = ofm->SliceShape().WithDepth(ofmDepth);
    query.ofmMemory = ofm->tensor->memArea.memory;
    query.ofmType = ofm->Type();
    query.ofmFormat = ofm->tensor->format;

    const SchedulerConnection *scratch = op->TryInput(TensorUsage::Scratch);
    if ( scratch )
    {
        query.tmpMemory = scratch->tensor->memArea.memory;
    }

    const SchedulerConnection *scales = op->TryInput(TensorUsage::Scales);
    if ( scales )
    {
        query.constShape = Shape(1, 1, 1, query.ofmShape.Depth());
        query.constMemory = scales->tensor->memArea.memory;
    }

    // If post-schedule cost is available, update with encoded sizes
    if ( cost && cost->npuWeightsTensor )
    {
        float ratio = float(ofmDepth) / ofm->SliceShape().Depth();
        unsigned weightBytes = cost->npuWeightsTensor->totalWeightBytes;
        unsigned scaleBytes = cost->npuWeightsTensor->AllocationSizeBytes() - weightBytes;

        // Encoded weight and scale sizes, estimated as a proportion if sliced.
        query.encodedWeightSize = unsigned(weightBytes * ratio);
        query.encodedScaleSize = unsigned(scaleBytes * ratio);
        query.constMemory = cost->npuWeightsTensor->memArea.memory;
        if ( cost->bufferedWeightTensor.tensor[0] )
        {
            query.weightStagingMemory = cost->bufferedWeightTensor.tensor[0]->memArea.memory;
            if ( cost->bufferedWeightTensor.preBuffer )
            {
                query.firstWeightDMASize = cost->npuWeightsTensor->rangeSizes.front();
            }
        }
    }

    // Record information regarding external feature maps for sub-operations
    query.weightFormat = wgtFormat;
    auto opGroup = op->OpGroup();
    query.opGroup = opGroup;
    for ( auto &subOp : op->SubOps() )
    {
        assert(opGroup);
        UniqueId subOpUId = subOp->Uid();

        SchedulerConnection *subIfm0 = subOp->IFM(0);
        if ( opGroup->NeedsAllocation(subIfm0->tensor->uid) )
        {
            FeatureMapRecord ifm0Record = {};
            ifm0Record.opId = subOpUId;
            ifm0Record.usage = TensorUsage::IFM0;
            ifm0Record.shape = subIfm0->SliceShape();
            ifm0Record.memory = subIfm0->tensor->memArea.memory;
            ifm0Record.format = subIfm0->tensor->format;
            if ( schedule )
            {
                // Attach a pointer to the sub-ops element access which will be updated by the performance estimator
                auto subCost = schedule->Cost(subOp.get());
                ifm0Record.access = &subCost->elementAccess;
            }
            query.featureMapRecords.push_back(ifm0Record);
        }

        SchedulerConnection *subIfm1 = subOp->TryIFM(1);
        if ( subIfm1 && opGroup->NeedsAllocation(subIfm1->tensor->uid) )
        {
            FeatureMapRecord ifm1Record = {};
            ifm1Record.opId = subOpUId;
            ifm1Record.usage = TensorUsage::IFM1;
            ifm1Record.shape = subIfm1->SliceShape();
            ifm1Record.memory = subIfm1->tensor->memArea.memory;
            ifm1Record.format = subIfm1->tensor->format;
            if ( schedule )
            {
                // Attach a pointer to the sub-ops element access which will be updated by the performance estimator
                auto subCost = schedule->Cost(subOp.get());
                ifm1Record.access = &subCost->elementAccess;
            }
            query.featureMapRecords.push_back(ifm1Record);
        }

        SchedulerConnection *subOfm = subOp->OFM();
        if ( opGroup->NeedsAllocation(subOfm->tensor->uid) )
        {
            FeatureMapRecord ofmRecord = {};
            ofmRecord.opId = subOpUId;
            ofmRecord.usage = TensorUsage::OFM;
            ofmRecord.shape = subOfm->SliceShape();
            ofmRecord.memory = subOfm->tensor->memArea.memory;
            ofmRecord.format = subOfm->tensor->format;
            if ( schedule )
            {
                // Attach a pointer to the sub-ops element access which will be updated by the performance estimator
                auto subCost = schedule->Cost(subOp.get());
                ofmRecord.access = &subCost->elementAccess;
            }
            query.featureMapRecords.push_back(ofmRecord);
        }
    }

    return query;
}


CycleCost Scheduler::EstimateOpPerformance(SchedulerOperation *op, ArchitectureOpConfig *config, int ofm_depth,
    WeightFormat wgtFormat, ArchitectureMemory *wgtStaging, OpScheduling scheduling)
{
    CycleCost cycleCost;
    if ( !op->IsNpuOp() )
    {
        LOG_WARN("CPU performance estimation for \"{}\" not implemented\n", OpTypeToString(op->Type()));
        return cycleCost;
    }

    PerformanceQuery query = InitPerfQuery(op, config, ofm_depth, wgtFormat);
    query.scheduling = scheduling;
    if ( !query.weightStagingMemory && wgtStaging )
    {
        query.weightStagingMemory = wgtStaging;
    }
    cycleCost = _arch->Performance()->MeasureCycleCost(query);
    return cycleCost;
}


void Scheduler::PrintSchedule(Schedule *schedule)
{
    LOG_PRINT("Schedule: '{}'\n", schedule->Name());
    for ( auto const &schedOp : _ops )
    {
        auto cost = schedule->Cost(schedOp.get());
        if ( cost == nullptr )
        {
            continue;
        }
        const SchedulerConnection *ofmConn = schedOp->OFM();
        LOG_PRINT("\t{0}: Operation {1}  - OFM {2} (in {3})\n", schedOp->Index(), OpTypeToString(schedOp->Type()),
            ofmConn->shape.ToString(), ofmConn->tensor->memArea.memory->Name());
        LOG_PRINT("\t\tKernel: {0}\n", schedOp->Kernel()->ToString());

        if ( !schedOp->IsNpuOp() )
        {
            LOG_PRINT("\t\tCPU Operation\n");
        }
        else
        {
            LOG_PRINT("{0}\n", cost->ToString());
        }
        if ( schedOp->SubOps().size() )
        {
            LOG_PRINT("\t\tsub-operations: [ ");
            for ( auto &subOp : schedOp->SubOps() )
            {
                LOG_PRINT("{} ", OpTypeToString(subOp->Type()));
            }
            LOG_PRINT("]\n");
        }
        else
        {
            LOG_PRINT("\t\tsub-operations: -\n");
        }

        LRMemory mem;
        if ( cost->timeIndex >= 0 && cost->timeIndex < int(schedule->memorySnapshot.size()) )
        {
            mem = schedule->memorySnapshot[cost->timeIndex];
        }

        LOG_PRINT("\t\tEstimated Perf: Macs={0} Cycles={1}\n", cost->cycles.macs, cost->cycles.opCycles);
        LOG_PRINT("\t\tMemory Used: {0} bytes (op={1}, buf={2}, ccd={3}, nl={4})\n", mem.Used(), mem.op, mem.buffering,
            mem.cascade, mem.nonlocal);
    }

    LOG_PRINT("\tCascades:\n");
    auto const &cascades = schedule->cascades;

    // Sort cascade contents by id and start time
    std::vector<int> keys;
    for ( auto const &pos : cascades )
    {
        keys.push_back(pos.first | (pos.second.start << 16));
    }
    std::sort(keys.begin(), keys.end());

    // Print sorted cascade indices
    for ( auto key : keys )
    {
        auto const &cascade = cascades.at(key & 0xFFFF);
        int cascadeMemUsage = 0;
        if ( cascade.start >= 0 && cascade.start < int(_ops.size()) )
        {
            auto *cost = schedule->Cost(_ops[cascade.start].get());
            cascadeMemUsage = cost ? schedule->MemoryUsageAt(cost->timeIndex) : 0;
        }
        LOG_PRINT("\t\t{0}: {1} -> {2}, size: {3}\n", key & 0xFFFF, cascade.start, cascade.end, cascadeMemUsage);
    }
}


bool ParseSchedulerOptions(SchedulerOptions &opt, IniReader &reader)
{
    // Parse debug settings
    std::string key;
    while ( reader.Begin(key) )
    {
        if ( key == "optimize" )
        {
            std::string value;
            if ( reader.Read(value) )
            {
                if ( _strnicmp(value.data(), "size", 5) == 0 )
                {
                    opt.optimizationStrategy = OptimizationStrategy::Size;
                }
                else if ( _strnicmp(value.data(), "performance", 12) == 0 )
                {
                    opt.optimizationStrategy = OptimizationStrategy::Performance;
                }
                else
                {
                    LOG_WARN("Unrecognised optimize value {}\n", value);
                }
            }
        }
        else if ( key == "verbose" )
        {
            opt.verboseSchedule = reader.Get<bool>();
        }
        else if ( key == "verbose_allocation" )
        {
            opt.verboseAllocation = reader.Get<bool>();
        }
        else if ( key == "arena_size_limit" )
        {
            opt.optimizationStagingLimit = reader.Get<int64_t>();
            std::string suffix;
            if ( reader.Read(suffix) )
            {
                if ( suffix == "kb" )
                {
                    opt.optimizationStagingLimit *= 1024;
                }
                else if ( suffix == "mb" )
                {
                    opt.optimizationStagingLimit *= 1024 * 1024;
                }
            }
        }
        else if ( key == "disable_feature" )
        {
            std::string value = reader.Get<std::string>();
            if ( !opt.disabled.Parse(value) )
            {
                LOG_WARN("Unrecognised disable_feature not in [{}]\n", AllFlagsToString<SchedulerFeature>());
            }
        }
        else if ( key == "separate_io_regions" )
        {
            opt.separateIORegions = reader.Get<bool>();
        }
        else if ( key == "cpu_tensor_alignment" )
        {
            opt.cpuTensorAlignment = reader.Get<int>();
        }
        else if ( key == "tensor_allocator" )
        {
            std::string value;
            if ( reader.Read(value) )
            {
                if ( _strnicmp(value.data(), "linearalloc", 11) == 0 )
                {
                    opt.tensorAllocator = TensorAllocator::LinearAlloc;
                }
                else if ( _strnicmp(value.data(), "hillclimb", 9) == 0 )
                {
                    opt.tensorAllocator = TensorAllocator::HillClimb;
                }
                else
                {
                    LOG_WARN("Unrecognised allocator value {}\n", value);
                }
            }
        }

        reader.End();
    }

    if ( opt.cpuTensorAlignment <= 0 || opt.cpuTensorAlignment % NPUTensorAlignment != 0 )
    {
        LOG_ERROR("CPU tensor alignment ({}) must be a multiple of {}\n", opt.cpuTensorAlignment, NPUTensorAlignment);
        return false;
    }

    return true;
}


struct SchedulerTransformParam : public WeightTransformParam
{
    const int64_t *zeroPoints;
    int zeroCount;
};

static int ApplyZeroPointAxisO(const WeightTransformParam *param, int value)
{
    const SchedulerTransformParam *p = static_cast<const SchedulerTransformParam *>(param);
    value = (value - int(p->zeroPoints[p->o % p->zeroCount]));
    assert(value >= -255 && value <= 255);
    return value;
}

static int ApplyZeroPointAxisI(const WeightTransformParam *param, int value)
{
    const SchedulerTransformParam *p = static_cast<const SchedulerTransformParam *>(param);
    value = (value - int(p->zeroPoints[p->i % p->zeroCount]));
    assert(value >= -255 && value <= 255);
    return value;
}

WeightScaleTensors Scheduler::EncodeQuantizationScaleTensor(OpType forOp, std::unique_ptr<IWeightEncodingConfig> encodingParams,
    int weightDepthBase, const std::vector<int> &depthOffsets, const Quantization &ofmQuantization, const SchedulerTensor *scales)
{
    SchedulerTensor scaleTens(DataType::UInt8, {0});
    scaleTens.dataType = DataType::Int32;
    if ( scales == nullptr ) scales = &scaleTens;
    return TryEncodeWeightAndScaleTensor(
        forOp, encodingParams.get(), weightDepthBase, depthOffsets, nullptr, scales, {}, ofmQuantization, false, true);
}

WeightScaleTensors Scheduler::EncodeWeightAndScaleTensor(OpType forOp, std::unique_ptr<IWeightEncodingConfig> encodingParams,
    int weightDepthBase, const std::vector<int> &depthOffsets, const SchedulerTensor *weightTens,
    const SchedulerTensor *scaleTens, const Quantization &weightQuantization, const Quantization &ofmQuantization)
{
    bool doWeights = true;
    bool doScales = true;

    // Check cache for weight tensors already encoded with this configuration.
    auto cacheKey = TensorCacheKey(
        encodingParams.get(), weightDepthBase, depthOffsets, weightTens->bufferView, weightTens->equivalenceId);
    auto pos = _tensorCache.find(cacheKey);
    std::shared_ptr<NpuWeightTensor> cachedWeightsTensor;
    if ( pos != _tensorCache.end() )
    {
        const WeightScaleTensors &cached = pos->second;
        assert(ofmQuantization.type == QuantizationType::EXPLICIT);
        uint32_t scaleHash = HashVector32(ofmQuantization.scales);
        // If scale tensor hashes match, return this combined weights tensor.
        if ( cached.scaleHash == scaleHash )
        {
            return cached;
        }
        // Already cached weights, but scales differ so perform scale encoding
        cachedWeightsTensor = cached.npuWeightsTensor;
        doWeights = false;
    }

    // Attempt the encode (may fail)
    WeightScaleTensors result = TryEncodeWeightAndScaleTensor(forOp, encodingParams.get(), weightDepthBase,
        depthOffsets, weightTens, scaleTens, weightQuantization, ofmQuantization, doWeights, doScales);

    if ( doWeights )
    {
        // Weights and scales now encoded together
        _tensorCache.emplace(cacheKey, result);
        result.npuWeightsTensor->config = std::move(encodingParams);
    }
    else
    {
        // Going to reuse a cached tensor for weights (must alias if memory areas don't match).
        if ( cachedWeightsTensor->memArea.Compatible(weightTens->memArea) )
        {
            result.npuWeightsTensor = std::move(cachedWeightsTensor);
        }
        else
        {
            // TODO: Clone tensor (but share buffer) if mem area assignment conflicts.
            //       Or cache encoded buffers and always wrap in a new tensor.
            assert(false);
            throw WeightEncodeException{};
        }
    }

    return result;
}

WeightScaleTensors Scheduler::TryEncodeWeightAndScaleTensor(OpType forOp, IWeightEncodingConfig *encodingParams,
    int weightDepthBase, const std::vector<int> &depthOffsets, const SchedulerTensor *weightTens, const SchedulerTensor *scaleTens,
    const Quantization &weightQuantization, const Quantization &ofmQuantization, bool doWeights, bool doScales)
{
    assert(doWeights || doScales);

    // Create tensor to hold encoded output
    auto npuTensor = std::make_shared<NpuWeightTensor>();
    npuTensor->uid = GenerateUniqueId();
    int rangeIndex = 0;
    int maxBufferLen[2] = {};
    std::vector<uint8_t> encodedStream;
    Shape ohwiShape;
    int channels;

    SchedulerTransformParam param;
    const uint8_t *weightsData = nullptr;
    std::unique_ptr<IVolumeWeightSource> weightSource;
    std::unique_ptr<IVolumeScaleSource> scaleSource;

    if ( weightTens )
    {
        npuTensor->memArea = weightTens->memArea;
        weightsData = weightTens->bufferView.Buffer()->Data<uint8_t>();
        ohwiShape = weightTens->bufferView.ViewShape();

        channels = (forOp == OpType::DepthwiseConv2D) ? ohwiShape.Depth() : ohwiShape.Batch();

        // Set up weight source
        param.zeroPoints = weightQuantization.zeroPoints.data();
        param.zeroCount = int(weightQuantization.zeroPoints.size());

        auto zeroOffsetFunc = (forOp == OpType::DepthwiseConv2D) ? ApplyZeroPointAxisO : ApplyZeroPointAxisI;
        weightSource = _arch->WeightEncoder()->GetWeightSource(encodingParams, weightTens->dataType, zeroOffsetFunc, &param);
        npuTensor->SetInternalName(weightTens->Name());
    }
    else
    {
        npuTensor->memArea = _arch->ReadonlyMemory();
        channels = ofmQuantization.scales.size();
        ohwiShape = Shape{channels};
    }

    if ( doScales )
    {
        scaleSource = _arch->WeightEncoder()->GetScaleSource(encodingParams, scaleTens->dataType, ofmQuantization);
        if ( !doWeights )
        {
            assert(scaleTens);
            npuTensor->SetInternalName(scaleTens->Name());
        }
    }

    int totalSourceBytes = 0;
    int totalWeightBytes = 0;
    int subStreams = 1;
    int scaleStreamsRequired = 1;
    int streamsRequired = _arch->WeightEncoder()->StreamsRequired(encodingParams, ohwiShape, scaleStreamsRequired);
    std::bitset<64> distinctWeights[8];
    std::vector<int> rangeSizes;

    if ( weightTens == nullptr ) streamsRequired = scaleStreamsRequired;

    // Note: in case of multiple cores, each core's weights are interleaved in O-dimension
    const int depthOffsetSize = int(depthOffsets.size());
    assert(depthOffsetSize > 1);
    rangeSizes.reserve(depthOffsetSize);
    for ( int idx = 0; idx < depthOffsetSize - 1; ++idx )
    {
        int depthOffset = depthOffsets[idx];
        int weightDepthOffset = depthOffset + weightDepthBase;

        // Do not generate for offsets outside the OFM
        assert(depthOffset >= 0 && depthOffset < channels);
        int depthLength = depthOffsets[idx + 1] - depthOffset;

        size_t bufferStartOffset = encodedStream.size();

        // For each stream, deinterleave weights/scales from the larger volume
        // and generate separate compressed streams.
        for ( int stream = 0; stream < streamsRequired; ++stream )
        {
            int key = WeightKey(stream, depthOffset);
            WeightRange range;
            range.offset = int(encodedStream.size());
            range.index = rangeIndex++;

            if ( doScales && stream < scaleStreamsRequired )
            {
                // Encode Scales and biases
                const uint8_t *biases = scaleTens->bufferView.HasBuffer() ? scaleTens->bufferView.RawData<uint8_t>() : nullptr;
                int biasCount = scaleTens->bufferView.HasBuffer() ? scaleTens->bufferView.ViewShape().Depth() : weightDepthOffset + depthLength;
                scaleSource->SetSource(biases, biasCount, weightDepthOffset, depthLength, stream);
                if ( scaleSource->Elements() == 0 )
                {
                    // No more elements left to encode
                    continue;
                }
                range.scaleBytes = _arch->WeightEncoder()->EncodeScales(encodingParams, scaleSource.get(), encodedStream, false);

                // Align to 16 for start of next substream
                while ( encodedStream.size() % 16 != 0 )
                {
                    encodedStream.push_back(0);
                }
            }

            if ( doWeights )
            {
                range.weightOffset = int(encodedStream.size()) - range.offset;

                // Encode Weights
                ohwiShape = (forOp == OpType::DepthwiseConv2D) ? ohwiShape.WithDepth(depthLength) : ohwiShape.WithBatch(depthLength);
                weightSource->SetSource(weightsData, weightDepthOffset, ohwiShape, weightTens->bufferView.StrideBytes(), stream);
                auto weightInfo = _arch->WeightEncoder()->EncodeWeights(encodingParams, weightSource.get(), encodedStream);
                range.weightBytes = weightInfo.encodedSize;
                totalWeightBytes += weightInfo.encodedSize;
                totalSourceBytes += weightInfo.sourceSize;
                int popcount = 0;

                // Stop counting when we know 4-bit palette mode can't be used,
                // no need to have exact popcount.
                for ( int i = 0; i < 8 && popcount <= 16; i++ )
                {
                    distinctWeights[i] |= weightInfo.weightsUsed[i];
                    popcount += distinctWeights[i].count();
                }
                npuTensor->distinctWeights = popcount;
                npuTensor->zeroCount += weightInfo.zeroCount;
            }

            assert(encodedStream.size() % 16 == 0);
            npuTensor->encodedRanges[key] = range;
            subStreams = std::max(stream + 1, subStreams);
        }

        // Remember maximum encoded length for DoubleBuffering
        int rangeSize = int(encodedStream.size() - bufferStartOffset);
        maxBufferLen[idx % 2] = std::max(maxBufferLen[idx % 2], rangeSize);
        rangeSizes.push_back(rangeSize);
    }

    // Reduce stored memory usage as much as possible
    encodedStream.shrink_to_fit();

    int streamSize = int(encodedStream.size());
    auto buf = std::make_shared<Buffer>(std::move(encodedStream));
    Shape storageShape(1, 1, 1, streamSize);
    npuTensor->bufferView = BufferView(buf, 0, 8, storageShape, Shape());
    npuTensor->dataType = DataType::UInt8;
    npuTensor->rangeSizes = std::move(rangeSizes);
    npuTensor->maxRangeBytes = std::max(maxBufferLen[0], maxBufferLen[1]);
    npuTensor->doubleBufferSizes[0] = maxBufferLen[0];
    npuTensor->doubleBufferSizes[1] = maxBufferLen[1];
    npuTensor->totalSourceBytes = totalSourceBytes;
    npuTensor->totalWeightBytes = totalWeightBytes;
    npuTensor->subStreams = subStreams;
    npuTensor->storageShape = storageShape;
    npuTensor->SetAllocatedSize(buf->Size());

    // Insert encoded weights hash and equivalenceId into map
    auto entry = _equivalenceIdMap.emplace(buf->Hash(), npuTensor->equivalenceId);
    if ( !entry.second )
    {
        // Encoded weights hash was already in the map, reuse stored equivalenceId
        npuTensor->equivalenceId = entry.first->second;
    }

    WeightScaleTensors result;
    result.scaleHash = HashVector32(ofmQuantization.scales);

    if ( doWeights )
    {
        result.npuWeightsTensor = std::move(npuTensor);
    }
    else
    {
        result.npuScalesTensor = std::move(npuTensor);
    }

    // Trace encoded tensor details to check uniqueness (duplicate values here should be cached)
    if ( weightTens )
    {
        LOG_TRACE2("Cache Tensor: {},  EncSize: {},  Hashed: {:08X}{:08X}{:08X}{:08X}, params={:08X}\n",
            weightTens->storageShape.ToString(), streamSize, buf->Hash().v32[0], buf->Hash().v32[1], buf->Hash().v32[2],
            buf->Hash().v32[3], encodingParams->Hash());
    }

    return result;
}

}  // namespace regor
