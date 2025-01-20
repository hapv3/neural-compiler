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
END_ENUM_TABLE()

namespace regor
{

constexpr int AllocationQuantum = 16;
constexpr int AlignmentQuantum = 16;

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

Scheduler::Scheduler(Architecture *arch, const SchedulerOptions &options, const std::string &name,
    std::vector<std::unique_ptr<SchedulerOperation>> &ops) :
        _ops(ops)
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
        initialStagingLimit = peakMemoryUsage;
    }

    std::shared_ptr<Schedule> chosenSchedule = _maxSchedule;

    if ( !_options.disabled.All(SchedulerFeature::Cascading) )
    {
        // Build cascades from min schedule
        std::unordered_map<UniqueId, int> nonLocal;
        CascadeBuilder cascadeBuilder(_ops, nonLocal, _spilling);
        cascadeBuilder.BuildCascades(minSchedule.get(), _maxSchedule.get(), initialStagingLimit);
        UpdateOpMemorySnapshot(minSchedule.get());

        chosenSchedule = minSchedule;

        if ( _options.optimizationStrategy == OptimizationStrategy::Performance )
        {
            // Create an optimized schedule
            auto optSchedule = OptimizeSchedule(minSchedule.get(), optMaxSchedule);
            chosenSchedule = std::move(optSchedule);
        }
    }

    if ( !_options.disabled.All(SchedulerFeature::WeightBuffering) )
    {
        CoalesceWeightBufferTensors(chosenSchedule.get());
    }

    UpdateOpMemorySnapshot(chosenSchedule.get());

    ApplySchedule(chosenSchedule.get());

    if ( _spilling && !_options.disabled.All(SchedulerFeature::FMStaging) )
    {
        // Use fast storage for feature maps
        FastStorageAllocator allocator;
        allocator.AllocateFeatureMaps(_ops, chosenSchedule.get(), _arch->StagingMemory(), _options.optimizationStagingLimit);
    }

    UpdateOpMemorySnapshot(chosenSchedule.get());

    if ( _options.verboseSchedule )
    {
        PrintSchedule(chosenSchedule.get());
    }

    if ( !AllocateAddresses(chosenSchedule.get()) )
    {
        throw std::runtime_error("Failed to allocate tensors\n");
    }

    return chosenSchedule;
}

Point2i Scheduler::GetStripeInputRequirement(const Shape &ofmShape, Kernel *kernel, const Point2i &ifmStep, ArchResampling resampling)
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


static int UpdateSchedulerTensor(Architecture *arch, TensorUsage usage, SchedulerConnection *conn)
{
    auto tensor = conn->tensor.get();

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
    if ( arch->Constraints()->SupportsNegativeStrides() && conn->reverse != ReverseType::None )
    {
        tensor->needsLinearFormat = true;
    }

    for ( auto producer : tensor->producers )
    {
        // TODO: Gather doesn't support brick format yet (MLBEDSW-8410)
        if ( producer->Type() == OpType::Scatter || producer->Type() == OpType::Gather )
        {
            tensor->needsLinearFormat = true;
        }
        // TODO: Tile doesn't support brick format yet (MLBEDSW-9485)
        else if ( producer->Type() == OpType::Tile )
        {
            tensor->needsLinearFormat = true;
        }
        else if ( producer->Type() == OpType::Transpose )
        {
            ArchRequirements req;
            ArchOperatorQuery query;
            query.transposeMask = producer->OFM()->transpose;
            if ( arch->Constraints()->OperatorQuery(OpType::Transpose, &query, &req).Any(QueryResult::Native) )
            {
                if ( req.ofmFormat == TensorFormat::NHWC )
                {
                    tensor->needsLinearFormat = true;
                }
            }
        }

        if ( !producer->IsNpuOp() )
        {
            tensor->hasCPUWriters = true;
        }
    }

    for ( auto consumer : tensor->consumers )
    {
        if ( !consumer->IsNpuOp() )
        {
            tensor->hasCPUReaders = true;
        }
        if ( tensor->needsLinearFormat ) continue;
        // TODO: Gather doesn't support brick format yet (MLBEDSW-8410)
        if ( consumer->Type() == OpType::Scatter || consumer->Type() == OpType::Gather )
        {
            tensor->needsLinearFormat = true;
            continue;
        }
        // TODO: Tile doesn't support brick format yet (MLBEDSW-9485)
        else if ( consumer->Type() == OpType::Tile )
        {
            tensor->needsLinearFormat = true;
            continue;
        }
        // Int32 ReduceSum requires linear format
        else if ( consumer->Type() == OpType::ReduceSum && tensor->dataType == DataType::Int32 )
        {
            tensor->needsLinearFormat = true;
            continue;
        }
        else if ( consumer->Type() == OpType::Transpose )
        {
            ArchRequirements req;
            ArchOperatorQuery query;
            query.transposeMask = consumer->OFM()->transpose;
            if ( arch->Constraints()->OperatorQuery(OpType::Transpose, &query, &req).Any(QueryResult::Native) )
            {
                if ( req.ofmFormat == TensorFormat::NHWC )
                {
                    tensor->needsLinearFormat = true;
                }
            }
        }

        // Check if consumer shape requires linear format
        // Brick format can only be used if both shapes have equal W and C
        // Need to check full shape on connection since tensor might have many producers (concat)
        auto ifm0 = consumer->TryIFM(0);
        auto ifm1 = consumer->TryIFM(1);
        auto ifm2 = consumer->TryIFM(2);
        if ( (ifm0 && ifm0->tensor.get() == tensor && ifm0->SliceShape() && conn->shape &&
                 Shape::PadAxes(ifm0->SliceShape(), 2, 1).WC<int>() != Shape::PadAxes(conn->shape, 2, 1).WC<int>()) ||
             (ifm1 && ifm1->tensor.get() == tensor && ifm1->SliceShape() && conn->shape &&
                 Shape::PadAxes(ifm1->SliceShape(), 2, 1).WC<int>() != Shape::PadAxes(conn->shape, 2, 1).WC<int>()) ||
             (ifm2 && ifm2->tensor.get() == tensor && ifm2->SliceShape() && conn->shape &&
                 Shape::PadAxes(ifm2->SliceShape(), 2, 1).WC<int>() != Shape::PadAxes(conn->shape, 2, 1).WC<int>()) )
        {
            tensor->needsLinearFormat = true;
        }
    }

    // Initial criteria (may change)
    bool cpuTensor = tensor->hasCPUWriters || tensor->hasCPUReaders || tensor->isGraphInput || tensor->isGraphOutput;
    conn->requireFullTensor = conn->requireFullTensor || cpuTensor;
    tensor->needsLinearFormat = tensor->needsLinearFormat || cpuTensor || CheckLinearFormatForConcatSplit(tensor);

    // Set tensor format to NHCWB16 for output FeatureMaps, if possible
    if ( IsOFM(usage) )
    {
        tensor->format = tensor->needsLinearFormat ? TensorFormat::NHWC : TensorFormat::NHCWB16;
    }

    return tensor->IsConstant() ? 0 : tensor->AllocationSizeBytes();
}


Address Scheduler::CreateSchedulerRepresentation()
{
    int minMemoryRequired = 0;

    for ( auto const &schedOp : _ops )
    {
        int opMemoryRequired = 0;

        for ( auto pos : schedOp->outputs.pairs() )
        {
            assert(!pos.second.tensor->producers.empty());
            opMemoryRequired += UpdateSchedulerTensor(_arch, pos.first, &pos.second);
        }

        for ( auto pos : schedOp->inputs.pairs() )
        {
            assert(!pos.second.tensor->consumers.empty());
            opMemoryRequired += UpdateSchedulerTensor(_arch, pos.first, &pos.second);
        }

        for ( auto const &subOp : schedOp->SubOps() )
        {
            for ( auto pos : subOp->outputs.pairs() )
            {
                opMemoryRequired += UpdateSchedulerTensor(_arch, pos.first, &pos.second);
            }

            for ( auto pos : subOp->inputs.pairs() )
            {
                opMemoryRequired += UpdateSchedulerTensor(_arch, pos.first, &pos.second);
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

std::unique_ptr<ArchitectureOpConfig> GetOpConfig(Architecture *arch, SchedulerOperation *op, const Shape &ifmShape,
    const Shape &ifm2Shape, const Shape &ofmShape, WeightFormat wgtFormat)
{
    assert(op->IsNpuOp());

    SchedulerConnection *ifm = op->IFM(0);
    SchedulerConnection *ifm2 = op->TryIFM(1);
    SchedulerConnection *ofm = op->OFM();

    ArchitectureConfigQuery query;
    query.ofmShape = Shape::PadAxes(ofmShape, 3, 1);
    query.ifmShape[0] = ifmShape;
    query.ifmShape[1] = ifm2Shape;
    query.ifmBits = DataTypeSizeBits(ifm->tensor->dataType);
    query.ofmBits = DataTypeSizeBits(ofm->tensor->dataType);
    query.kernel = op->Kernel();
    query.lutBytes = op->TryInput(TensorUsage::LUT) ? 2048 : 0;
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

    return arch->GetOpConfig(op->Type(), query);
}

Shape GetOhwiShape(const SchedulerTensor *weightTens)
{
    const auto &weightView = weightTens->bufferView;
    Shape ohwiShape = weightView.ViewShape();
    if ( weightTens->srcTensor->AxisOrder() == AxisOrder::IHWO )
    {
        ohwiShape = ohwiShape.Extract(3, 1, 2, 0);
    }
    else if ( weightTens->srcTensor->AxisOrder() == AxisOrder::HWCM )
    {
        ohwiShape = ohwiShape.Extract(2, 0, 1, 3);
    }
    return ohwiShape;
}

Shape GetOhwiStrides(const SchedulerTensor *weightTens)
{
    const auto &weightView = weightTens->bufferView;
    Shape ohwiStrides = weightView.StrideBytes() * 8 / DataTypeSizeBits(weightTens->dataType);
    if ( weightTens->srcTensor->AxisOrder() == AxisOrder::IHWO )
    {
        ohwiStrides = ohwiStrides.Extract(3, 1, 2, 0);
    }
    else if ( weightTens->srcTensor->AxisOrder() == AxisOrder::HWCM )
    {
        ohwiStrides = ohwiStrides.Extract(2, 0, 1, 3);
    }
    return ohwiStrides;
}

WeightScaleEncoding ChooseBestWeightFormat(Architecture *arch, SchedulerOperation *op,
    OptimizationStrategy optimizationStrategy, std::vector<WeightScaleEncoding> &encodingResults)
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
            auto cycles = arch->Performance()->WeightDecodeCycles(
                query, weightStats, weightTensor->config->Format(), weightTensor->memArea.memory);
            if ( cycles < minCycles )
            {
                bestResult = &encodingResult;
                minCycles = cycles;
            }
        }
    }
    return std::move(*bestResult);
}

bool UseFastDecoder(regor::Architecture *arch, SchedulerOperation *op, OptimizationStrategy optimizationStrategy, NpuWeightTensor *weightTensor)
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
    auto defaultCycles = arch->Performance()->WeightDecodeCycles(
        query, weightStats, WeightFormat::Default, weightTensor->memArea.memory);
    weightStats.encodedSize = fastWeightSize;
    auto fastCycles = arch->Performance()->WeightDecodeCycles(
        query, weightStats, WeightFormat::Fast, weightTensor->memArea.memory);
    return fastCycles < defaultCycles;
}

std::unique_ptr<ArchitectureOpConfig> MaybeGetSparsityConfig(regor::Architecture *arch, SchedulerOperation *op,
    Shape &ifmShape, Shape &ifm2Shape, Shape &ofmShape, Flags<WeightFormat> supportedFormat)
{
    using WF = Flags<WeightFormat>;
    std::unique_ptr<ArchitectureOpConfig> blockConfigSparse;
    if ( supportedFormat % WeightFormat::Sparse2_4 )
    {
        blockConfigSparse = GetOpConfig(arch, op, ifmShape, ifm2Shape, ofmShape, WF(WeightFormat::Default, WeightFormat::Sparse2_4));
    }
    return blockConfigSparse;
}
}  // namespace


WeightScaleEncoding Scheduler::EncodeBestWeightFormat(
    SchedulerOperation *op, Shape &ifmShape, Shape &ifm2Shape, Shape &ofmShape, Flags<WeightFormat> supportedFormats)
{
    using WF = Flags<WeightFormat>;
    // We assume that block config depends only on the sparsity bit in the weight format.
    std::unique_ptr<ArchitectureOpConfig> blockConfigDefault = GetOpConfig(
        _arch, op, ifmShape, ifm2Shape, ofmShape, WF(WeightFormat::Default));
    std::unique_ptr<ArchitectureOpConfig> blockConfigSparse = MaybeGetSparsityConfig(_arch, op, ifmShape, ifm2Shape, ofmShape, supportedFormats);

    if ( blockConfigSparse )
    {
        auto perfDefault = EstimateOpPerformanceForSparsity(op, blockConfigDefault.get(), op->OFM()->SliceShape().Depth());
        auto perfSparse = EstimateOpPerformanceForSparsity(op, blockConfigSparse.get(), op->OFM()->SliceShape().Depth());
        if ( perfSparse.opCycles > perfDefault.opCycles )
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
    WeightsRef weightsRef = {&weights->tensor->bufferView, weights->tensor->srcTensor->AxisOrder(), weights->tensor->dataType};
    auto ifm = op->IFM(op->PrimaryIfmIndex());
    auto ifmType = ifm->tensor->dataType;
    std::vector<int> depthOffsets{0, ofmShape.Unpermute(uint32_t(op->OFM()->transpose)).Depth()};

    std::vector<WF> formatList = {WF(WeightFormat::Default, WeightFormat::Sparse2_4), WF(WeightFormat::Default),
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
        auto encodingParams = _arch->WeightEncoder()->GetEncodingConfig(
            blockConfig, weightsRef, op->Kernel(), ifmType, depthOffsets, weightFormat);

        try
        {
            WeightScaleEncoding encoding;
            encoding.weightScales = EncodeWeightAndScaleTensor(std::move(encodingParams), weights->tensor.get(),
                scales->tensor.get(), weights->quantization, op->OFM()->quantization);

            if ( checkFastDecoder &&
                 !UseFastDecoder(_arch, op, _options.optimizationStrategy, encoding.weightScales.npuWeightsTensor.get()) )
            {
                supportedFormats.Unset(WeightFormat::Fast);
            }
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
    auto bestEncoding = ChooseBestWeightFormat(_arch, op, _options.optimizationStrategy, encodingResults);
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
        stripeInput = Point2i::Min(stripeInput, ifmShape.WH<int>());
        if ( !subdivideMask.Any(AxisMask::AxisX) && (stripeInput.x != ifmShape.Width()) )
        {
            assert(stripeInput.x * ifm->stepXY.x >= (ifmShape.Width() - op->Kernel()->Stride().x) && "Unexpected stripe input width");
            stripeInput.x = ifmShape.Width();
        }

        ifmShape = ifmShape.WithHW(stripeInput.y, stripeInput.x);

        if ( !ifm2Shape.IsEmpty() )
        {
            Point2i stripeInput2 = Point2i::Min(stripeInput, ifm2Shape.WH<int>());
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
        if ( op->OFM()->quantization.type != QuantizationType::EXPLICIT )
        {
            auto scales = op->Input(TensorUsage::Scales);
            auto temp = _arch->WeightEncoder()->MakeExplicit(ifm->quantization, weights->quantization,
                op->OFM()->quantization, scales->tensor->dataType, ifm->tensor->dataType, op->Type());
            op->OFM()->quantization = std::move(temp);
            assert(op->OFM()->quantization.type == QuantizationType::EXPLICIT);
        }
        auto weightEncoding = EncodeBestWeightFormat(op, ifmShape, ifm2Shape, ofmShape, weightFormat);
        blockConfig = std::move(weightEncoding.blockConfig);
        weightScales = weightEncoding.weightScales;
    }
    else
    {
        blockConfig = parentInfo ? parentInfo->Config()->Clone() : GetOpConfig(_arch, op, ifmShape, ifm2Shape, ofmShape, weightFormat);
    }
    auto scales = op->TryInput(TensorUsage::Scales);
    if ( !weights && (op->OFM()->quantization.scales.size() > 1 || scales) )
    {
        WeightsRef weightsRef;
        weightsRef.isScales = true;

        std::vector<int> depthOffsets{0, ofmShape.Unpermute(uint32_t(op->OFM()->transpose)).Depth()};

        auto encodingParams = _arch->WeightEncoder()->GetEncodingConfig(
            blockConfig.get(), weightsRef, op->Kernel(), ifm->tensor->dataType, depthOffsets, weightFormat);

        const SchedulerTensor *scaleTensor = scales ? scales->tensor.get() : nullptr;
        weightScales = EncodeQuantizationScaleTensor(std::move(encodingParams), op->OFM()->quantization, scaleTensor);
    }
    // Finally construct and populate operator information (cost)
    auto opInfo = std::make_unique<SchedulerOpInfo>(std::move(blockConfig), ifmShape, ifm2Shape, ofmShape);
    opInfo->SetWeightScaleTensors(weightScales.npuWeightsTensor, weightScales.npuScalesTensor);

    return opInfo;
}


std::unique_ptr<Schedule> Scheduler::CreateInitialSchedule()
{
    auto schedule = std::make_unique<Schedule>(_name + "_MAX");

    for ( auto &op : _ops )
    {
        const auto ofm = op->OFM();
        const auto &ofmShape = ofm->SliceShape();
        auto cost = CreateSchedulerOpInfo(op.get(), ofmShape);
        if ( ofmShape )
        {
            cost->cycles = EstimateOpPerformance(op.get(), cost->Config(), ofmShape.Depth());
            cost->elementAccess = EstimateOpElementAccess(op.get(), cost->Config(), ofmShape.Depth());
        }
        // sub-operations
        for ( auto &subOp : op->SubOps() )
        {
            const auto subOfm = subOp->OFM();
            const auto &subOfmShape = subOfm->SliceShape();
            auto subCost = CreateSchedulerOpInfo(subOp.get(), subOfmShape, cost);
            if ( subOfmShape )
            {
                subCost->cycles = EstimateOpPerformance(subOp.get(), subCost->Config(), subOfmShape.Depth());
                subCost->elementAccess = EstimateOpElementAccess(subOp.get(), subCost->Config(), subOfmShape.Depth());
            }
            schedule->SetCost(*subOp, std::move(subCost));
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
                     conn->tensor->srcTensor->View().Buffer()->Size() > maxIfmShramAvail )
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
    AllocateTensors(_ops, schedule, _arch->FeatureMapMemory(), TensorAllocator::HillClimb, AlignmentQuantum, verbose);
    if ( _spilling )
    {
        const auto limit = _options.optimizationStagingLimit;
        AllocateTensors(_ops, schedule, _arch->StagingMemory(), TensorAllocator::HillClimb, AlignmentQuantum, verbose, limit);

        return schedule->memoryUsage[_arch->StagingMemory()] <= limit;
    }
    return true;
}


/// @brief Specialised LiveRangeGraph for read only (flash) memory which ignores scalars if possible
class ReadOnlyLiveRangeGraph : public LiveRangeGraph
{
private:
    Architecture *_arch;

public:
    ReadOnlyLiveRangeGraph(Architecture *arch) : _arch(arch) {}
    bool ShouldBeIgnored(SchedulerTensor *tens, const MemArea &targetMemory) override
    {
        // First do the regular check for matching memory type
        if ( LiveRangeGraph::ShouldBeIgnored(tens, targetMemory) )
        {
            return true;
        }
        // Memory type correct, check if this tensor is a scalar that can be encoded
        // in the command stream for this architecture
        auto srcTens = tens->srcTensor;
        if ( srcTens && srcTens->StorageShape().Elements() == 1 && srcTens->IsConstant() )
        {
            // All consumers must accept this scalar if we are to ignore it
            for ( auto op : tens->consumers )
            {
                // Find usage of the tensor for this consumer op
                TensorUsage usage(TensorUsage::None);
                for ( auto input : op->inputs.pairs() )
                {
                    if ( input.second.tensor.get() == tens )
                    {
                        usage = input.first;
                    }
                }
                if ( !_arch->SupportsScalar(op->Type(), tens->dataType, usage) )
                {  // This scalar cannot be ignored and must be handled
                    return false;
                }
            }
            // At this point we have determined that the tensor can be encoded
            // as a scalar in the command stream and can safely be ignored
            return true;
        }
        // Not a scalar - cannot be ignored
        return false;
    }
};


void Scheduler::AllocateReadOnlyAddresses(Schedule *schedule, IncrementalLinearAllocator &readOnlyAllocator)
{
    auto lrGraph = ReadOnlyLiveRangeGraph(_arch);
    lrGraph.ExtractLiveRangesFromCascades(_ops, schedule, _arch->ReadonlyMemory(), false);
    auto totalSize = readOnlyAllocator.Allocate(&lrGraph, AlignmentQuantum, _options.verboseAllocation);
    schedule->memoryUsage[_arch->ReadonlyMemory()] = int(totalSize);
}


void Scheduler::UpdateOpMemorySnapshot(Schedule *schedule)
{
    const auto fastStorage = _arch->StagingMemory();
    auto lrGraph = LiveRangeGraph();
    lrGraph.ExtractLiveRangesFromCascades(_ops, schedule, fastStorage, true);
    // Populate time-array with memory used by live ranges
    std::vector<int> temporalUsage = lrGraph.GetTemporalMemoryUsage(schedule->fastStoragePeakUsage);
    schedule->memorySnapshot = std::move(temporalUsage);
}


std::shared_ptr<Schedule> Scheduler::ProposeScheduleBuffering(Schedule *refSchedule, Address stagingLimitBytes)
{
    auto bufferedSchedule = std::make_shared<Schedule>(refSchedule->Name() + "_BUFFERED");
    int stagingLimitClamped = int(std::min(INT64_C(1) << 30, stagingLimitBytes));

    SchedulerOperation *prevOp = nullptr;
    for ( auto const &schedOp : _ops )
    {
        SchedulerOpInfo *cost = refSchedule->Cost(schedOp.get());
        // schedOp is not part of this sub-schedule - skip
        if ( cost == nullptr )
        {
            continue;
        }

        ProposeOperatorBuffering(schedOp.get(), prevOp, bufferedSchedule.get(), refSchedule, stagingLimitClamped);

        // chained sub-operations
        for ( auto const &subOp : schedOp->SubOps() )
        {
            ProposeOperatorBuffering(subOp.get(), prevOp, bufferedSchedule.get(), refSchedule, stagingLimitClamped);
        }
        prevOp = schedOp.get();
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

    // Take the reference schedule as default costings for this schedule
    auto refCost = refSchedule->Cost(schedOp);
    assert(refCost != nullptr);
    auto costCopy = std::make_unique<SchedulerOpInfo>(*refCost);
    auto cost = costCopy.get();
    bufferedSchedule->SetCost(*schedOp, std::move(costCopy));

    // Don't buffer non NPU operations
    if ( !schedOp->IsNpuOp() )
    {
        return;
    }
    // Snapshot may already contain buffering, which the weight buffering function does not expect
    int unwantedExistingBuffering = refCost->bufferedWeightTensor.tensor ? refCost->bufferedWeightTensor.tensor->AllocationSizeBytes() : 0;
    int slackBufferingMemory = stagingLimitBytes - (refSchedule->MemoryUsageAt(refCost->timeIndex) - unwantedExistingBuffering);

    cost->slackBufferingMemory = slackBufferingMemory;
    cost->slackBufferingCycles = refCost->cycles.opCycles;

    // Attempt weight buffering on anything with a weights tensor
    auto weights = schedOp->TryInput(TensorUsage::Weights);
    if ( weights != nullptr )
    {
        auto scales = schedOp->Input(TensorUsage::Scales);
        ProposeWeightBuffering(weights, scales, schedOp, prevOp, bufferedSchedule, refSchedule, slackBufferingMemory);
    }
}

static bool FulldepthWeightBuffering(const std::vector<std::unique_ptr<SchedulerOperation>> &ops, SchedulerTensor *weights,
    SchedulerOperation *schedOp, SchedulerOpInfo *cost, SchedulerOperation *prevOp, SchedulerOpInfo *prevCost, Schedule *refSchedule)
{
    bool forceFullDepthSlice = false;
    if ( weights->srcTensor->Readers().size() > 1 )
    {
        // Check for special case where several consecutive ops have the same weight tensor.
        // If the weigths can fit entire in bufferLimit and the ops all have the same ofm depth slice
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

        if ( cmpWeights != nullptr )
        {
            UniqueId weightsTensorId = weights->equivalenceId;
            UniqueId cmpWeightsTensorId = cmpWeights->tensor->equivalenceId;

            if ( cmpWeightsTensorId == weightsTensorId && cmpCost->ofmDepthSlices == cost->ofmDepthSlices )
            {
                forceFullDepthSlice = true;
            }
        }
    }

    return forceFullDepthSlice;
}

void Scheduler::ProposeWeightBuffering(SchedulerConnection *weights, SchedulerConnection *scales, SchedulerOperation *schedOp,
    SchedulerOperation *prevOp, Schedule *bufferedSchedule, Schedule *refSchedule, int bufferLimitBytes)
{
    constexpr int OFMSplitDepth = 16;
    auto cost = bufferedSchedule->Cost(schedOp);
    auto prevCost = bufferedSchedule->Cost(prevOp);
    auto refCost = refSchedule->Cost(schedOp);
    auto ifm = schedOp->IFM(0);
    auto ofm = schedOp->OFM();

    assert(cost && refCost);

    // Weights are in permanent storage. When permanent storage differs from feature map storage,
    // there is a point moving the data
    auto weightTens = weights->tensor.get();
    auto scaleTens = scales->tensor.get();
    // No need to move the weights if they are already in the same memory as the staging area
    bool needsDMA = weightTens->memArea.memory != _arch->StagingMemory().memory;

    const auto &subOps = schedOp->SubOps();
    const bool isChained =
        schedOp->Parent() != nullptr ||
        subOps.end() !=
            std::find_if(subOps.begin(), subOps.end(), [](const auto &subOp) { return IsElementwise(subOp->Type()); });

    assert(ofm->transpose == TransposeType::None || (ofm->transpose & TransposeType::MaskC) == TransposeType::C || refCost->stripe == ofm->shape);
    auto fullDepthBeforeTransposition =
        (refCost->stripe == ofm->shape) ? refCost->stripe.Unpermute(uint32_t(ofm->transpose)).Depth() : refCost->stripe.Depth();
    auto fullDepthAfterTransposition = refCost->stripe.Depth();
    std::vector<int> ofmFullDepthSlicesBeforeTransposition = {0, fullDepthBeforeTransposition};
    std::vector<int> ofmFullDepthSlicesAfterTransposition = {0, fullDepthAfterTransposition};

    WeightsRef weightsRef = {&weightTens->bufferView, weightTens->srcTensor->AxisOrder(), weightTens->dataType};

    auto weightFormat = cost->npuWeightsTensor->config->Format();

    auto encodingParams = _arch->WeightEncoder()->GetEncodingConfig(cost->Config(), weightsRef, schedOp->Kernel(),
        ifm->tensor->dataType, ofmFullDepthSlicesBeforeTransposition, weightFormat);

    auto fullWeightScales = EncodeWeightAndScaleTensor(
        std::move(encodingParams), weightTens, scaleTens, weights->quantization, ofm->quantization);

    int fullWeightsBytes = fullWeightScales.npuWeightsTensor->AllocationSizeBytes();

    bool forceFullDepthSlice = false;
    if ( fullWeightsBytes <= bufferLimitBytes )
    {
        forceFullDepthSlice = FulldepthWeightBuffering(_ops, weightTens, schedOp, cost, prevOp, prevCost, refSchedule);
    }

    if ( _spilling && !forceFullDepthSlice )
    {
        // To be refined and architecture specific depending on mem2mem characteristics and prebuffering
        float bwRatio = std::round(
            _arch->Performance()->ChannelBW(weightTens->memArea.memory, MemChannel::Weight) /
            _arch->Performance()->ChannelBW(weightTens->memArea.memory, MemChannel::Mem2Mem));
        needsDMA = (cost->elementAccess.weightsRefetch > 2) || (cost->elementAccess.weightsRefetch == 2 && bwRatio < 2);
    }

    // No buffering required - take all the weights from permanent storage
    if ( (schedOp->Type() == OpType::FullyConnected && cost->elementAccess.weightsRefetch == 1) || !needsDMA ||
         _arch->CanSubdivide(schedOp->Type(), ofm->transpose, ofm->reverse) == AxisMask::None || ofm->reverse == ReverseType::C ||
         (ofm->transpose != TransposeType::None && (ofm->transpose & TransposeType::MaskC) != TransposeType::C) ||
         _options.disabled.All(SchedulerFeature::WeightBuffering) )
    {
        cost->ofmDepthSlices = std::move(ofmFullDepthSlicesAfterTransposition);
        // Make sure any former buffering cost is cleared
        cost->bufferedWeightTensor.buffering = Buffering::None;
        cost->bufferedWeightTensor.tensor = nullptr;
        cost->bufferedWeightTensor.preBuffer = false;
        cost->SetWeightScaleTensors(fullWeightScales.npuWeightsTensor, fullWeightScales.npuScalesTensor);
        return;
    }

    auto encodedWeightScales = fullWeightScales;

    // How many NPU cycles are available under the previously executing
    // operator for performing buffered DMA transfers
    int64_t slackCycles = (prevCost != nullptr) ? prevCost->slackBufferingCycles : 0;
    int slackMemory = (prevCost != nullptr) ? prevCost->slackBufferingMemory : 0;

    int weightBufferSize = 0;

    // Force full depth for cascaded and chained ops
    // TODO MLBEDSW-9145: support depth-slicing for chains
    if ( cost->cascade != 0 || isChained || forceFullDepthSlice )
    {
        weightBufferSize = fullWeightsBytes;
        // Update the memory snapshot to reflect the added size of the weights
        refSchedule->memorySnapshot[cost->timeIndex] += weightBufferSize;
    }
    else
    {
        // Estimate the buffering cycle time for the full set of weights
        int64_t fullTransferCycles = _arch->Performance()->MemToMemCycles(
            _arch->StagingMemory().memory, weightTens->memArea.memory, fullWeightsBytes);
        cost->fullWeightTransferCycles = fullTransferCycles;

        // Calculate the amount of pre-buffering necessary (or what is possible with limited
        // double buffer buffer size)
        double prebufferRatio = 0;
        const int halfBufferLimit = bufferLimitBytes / 2;
        int prebufferBytes = std::min(fullWeightsBytes, halfBufferLimit);
        if ( fullTransferCycles > slackCycles )
        {
            prebufferRatio = double(slackCycles) / double(fullTransferCycles);
            prebufferBytes = std::min(int(prebufferRatio * fullWeightsBytes), halfBufferLimit);
        }

        prebufferRatio = double(prebufferBytes) / fullWeightsBytes;

        // Have to split the weights if the initial buffering can't store
        // all of the compressed weights
        if ( prebufferBytes < fullWeightsBytes )
        {
            int blockDepth = cost->Config()->OptimalDepthGranule();

            // Choose initial pre-buffering depth (already buffer clamped)
            int prebufferDepth = int(refCost->stripe.Depth() * prebufferRatio);
            prebufferDepth = int(std::max(16, RoundZero(prebufferDepth, OFMSplitDepth)));

            // Calculate cycles executed during the pre-buffer
            auto preOpCycles = EstimateOpPerformance(schedOp, cost->Config(), prebufferDepth);
            int bufferingDepth = int((refCost->stripe.Depth() * preOpCycles.opCycles) / fullTransferCycles);

            // Choose initial buffering depth and clamp to the double buffering limit
            bufferingDepth = RoundAway(bufferingDepth, blockDepth);
            int bufferingBytes = (bufferingDepth / refCost->stripe.Depth()) * fullWeightsBytes;
            if ( bufferingBytes > halfBufferLimit )
            {
                bufferingDepth = (halfBufferLimit * refCost->stripe.Depth()) / fullWeightsBytes;
            }

            while ( true )
            {
                // Attempt to buffer whole blocks
                if ( bufferingDepth > blockDepth )
                {
                    bufferingDepth = RoundZero(bufferingDepth, blockDepth);
                }
                else
                {
                    bufferingDepth = RoundZero(bufferingDepth, OFMSplitDepth);
                }

                bufferingDepth = int(std::max(bufferingDepth, OFMSplitDepth));

                // Create list of depth slices
                std::vector<int> depthSlices = {0};

                for ( int depth = prebufferDepth; depth < refCost->stripe.Depth(); depth += bufferingDepth )
                {
                    depthSlices.push_back(depth);
                }
                depthSlices.push_back(refCost->stripe.Depth());

                // Encode weights based depth slices
                cost->ofmDepthSlices = std::move(depthSlices);

                weightsRef = {&weightTens->bufferView, weightTens->srcTensor->AxisOrder(), weightTens->dataType, false};

                encodingParams = _arch->WeightEncoder()->GetEncodingConfig(cost->Config(), weightsRef,
                    schedOp->Kernel(), ifm->tensor->dataType, cost->ofmDepthSlices, weightFormat);

                encodedWeightScales = EncodeWeightAndScaleTensor(
                    std::move(encodingParams), weightTens, scaleTens, weights->quantization, ofm->quantization);

                // Chosen buffering might not fit at all, iterate until it does
                // or until the minimum usable slice size is reached
                if ( encodedWeightScales.npuWeightsTensor->maxRangeBytes <= halfBufferLimit ||
                     (prebufferDepth == OFMSplitDepth && bufferingDepth == OFMSplitDepth) )
                {
                    break;
                }

                // Failed to choose buffer sizes above, reduce them and try again
                if ( bufferingDepth > prebufferDepth )
                {
                    bufferingDepth = RoundAway(bufferingDepth / 2, OFMSplitDepth);
                }
                else
                {
                    prebufferDepth = RoundAway(prebufferDepth / 2, OFMSplitDepth);
                }
            }

            // Calculate cycles required to run the last op for use as future slack
            assert(cost->ofmDepthSlices.size() >= 2);
            int lastDepth = cost->ofmDepthSlices.back();
            lastDepth -= *(cost->ofmDepthSlices.rbegin() + 1);
            auto tailCycles = EstimateOpPerformance(schedOp, cost->Config(), lastDepth);
            cost->slackBufferingCycles = tailCycles.opCycles;
        }
    }

    // Determine whether the weights need to be double buffered
    int encodedWeightsSize = encodedWeightScales.npuWeightsTensor->AllocationSizeBytes();
    weightBufferSize = std::min(encodedWeightsSize, encodedWeightScales.npuWeightsTensor->maxRangeBytes);
    int doubleBufferSize = encodedWeightScales.npuWeightsTensor->doubleBufferSize;

    // Only buffer weights if there's still space left for the buffer
    if ( weightBufferSize <= bufferLimitBytes )
    {
        assert(weightBufferSize % 16 == 0);  // NOTE: vague check, leave validation until later?

        // Determine whether to double buffer or single buffer
        Buffering buffering = Buffering::Single;
        if ( (doubleBufferSize <= bufferLimitBytes) && (weightBufferSize < encodedWeightsSize) )
        {
            weightBufferSize = doubleBufferSize;
            buffering = Buffering::Double;
        }

        // Create a new tensor in fast storage to use as weights buffer
        cost->bufferedWeightTensor.tensor = std::make_shared<SchedulerTensor>();
        cost->bufferedWeightTensor.tensor->srcTensor = encodedWeightScales.npuWeightsTensor->srcTensor;
        cost->bufferedWeightTensor.tensor->SetAllocatedSize(weightBufferSize);
        cost->bufferedWeightTensor.tensor->memArea = _arch->StagingMemory();
        cost->bufferedWeightTensor.buffering = buffering;

        if ( cost->cascade == 0 )
        {
            // Determine if the lifetime can be extended and pre-buffer weights under the previous operation
            cost->bufferedWeightTensor.preBuffer = (weightBufferSize < slackMemory);
        }

        cost->slackBufferingMemory -= weightBufferSize;
    }
    else
    {
        // Don't slice or buffer - use the whole depth from persistent storage
        cost->ofmDepthSlices = std::move(ofmFullDepthSlicesAfterTransposition);
        cost->bufferedWeightTensor.buffering = Buffering::None;
        cost->bufferedWeightTensor.tensor = nullptr;
        cost->bufferedWeightTensor.preBuffer = false;
        encodedWeightScales = std::move(fullWeightScales);
    }
    cost->SetWeightScaleTensors(encodedWeightScales.npuWeightsTensor, encodedWeightScales.npuScalesTensor);
}


std::shared_ptr<Schedule> Scheduler::ProposeMinimalSchedule()
{
    // Proposes scheduling parameters where every operator is subdivided into the smallest stripe that
    // satisfies the next operators stride
    auto minSchedule = std::make_shared<Schedule>(_name + "_MIN");

    // Keep track of the previous Op - which consumes the current Op's OFM
    SchedulerOperation *prevOp = nullptr;

    // Work backwards up the schedule setting the minimum stripe height
    for ( auto pos = _ops.rbegin(); pos != _ops.rend(); pos++ )
    {
        auto const &schedOp = *pos;
        int minStripeHeight = (prevOp != nullptr) ? prevOp->Kernel()->Stride().y : 1;
        const auto ofm = schedOp->OFM();
        const auto &ofmShape = ofm->SliceShape();
        Shape minStripe = Shape::PadAxes(ofmShape, 3, 1).WithHeight(minStripeHeight);
        auto cost = CreateSchedulerOpInfo(schedOp.get(), minStripe);
        if ( ofmShape )
        {
            cost->cycles = EstimateOpPerformance(schedOp.get(), cost->Config(), ofmShape.Depth());
            cost->elementAccess = EstimateOpElementAccess(schedOp.get(), cost->Config(), ofmShape.Depth());
        }

        // sub-operations use the same stripe as their parent
        for ( auto &subOp : schedOp->SubOps() )
        {
            const auto subOfm = subOp->OFM();
            const auto &subOfmShape = subOfm->SliceShape();
            auto subCost = CreateSchedulerOpInfo(subOp.get(), minStripe, cost);
            if ( subOfmShape )
            {
                subCost->cycles = EstimateOpPerformance(subOp.get(), subCost->Config(), subOfmShape.Depth());
                subCost->elementAccess = EstimateOpElementAccess(subOp.get(), subCost->Config(), subOfmShape.Depth());
            }
            minSchedule->SetCost(*subOp, std::move(subCost));
        }
        minSchedule->SetCost(*schedOp, std::move(cost));
        prevOp = schedOp.get();
    }

    return minSchedule;
}


std::shared_ptr<Schedule> Scheduler::OptimizeSchedule(Schedule *schedule, const std::shared_ptr<Schedule> &maxSchedule)
{
    // Extracts sub-schedules based on the cascades and optimizes them and applies them to the final schedule
    if ( maxSchedule->fastStoragePeakUsage < _options.optimizationStagingLimit && !_spilling )
    {
        return maxSchedule;
    }

    // Optimize cascades separately
    // Iterate over a copy of the cascades since they may change during the loop
    auto cascades = schedule->cascades;
    for ( const auto &pos : cascades )
    {
        const CascadeInfo &cascadeInfo = pos.second;

        auto optSubSchedule = OptimizeSubSchedule(cascadeInfo, schedule, _options.optimizationStagingLimit);
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
    auto optSchedule = ProposeScheduleBuffering(schedule, _options.optimizationStagingLimit);
    optSchedule->cascades = std::move(schedule->cascades);  // TODO: Check this is okay
    // Copy the cascade's metadata from the unbuffered schedule
    return optSchedule;
}


std::shared_ptr<Schedule> Scheduler::ProposeScheduleStriping(const Shape &finalStripe, const std::string &label, Schedule *refSchedule)
{
    // Proposes new striping for a schedule. The stripe is derived from the ifm requirements of the next Op down
    auto stripedSchedule = std::make_shared<Schedule>(label);

    Shape stripe = finalStripe;
    for ( auto pos = _ops.rbegin(); pos != _ops.rend(); pos++ )
    {
        auto schedOp = pos->get();
        auto refCost = refSchedule->Cost(schedOp);
        if ( !schedOp->IsNpuOp() || refCost == nullptr )
        {
            // sched_op is not part of the sub-schedule - skip
            continue;
        }

        // Create a cost entry with the new stripe
        auto cost = CreateSchedulerOpInfo(schedOp, stripe);

        // Take buffering choice from the reference schedule for this striping proposal.
        // TODO: Replace with in-loop buffering
        if ( refCost->bufferedWeightTensor.tensor )
        {
            auto bufferingTensor = std::make_shared<SchedulerTensor>();
            bufferingTensor->srcTensor = cost->npuWeightsTensor->srcTensor;
            bufferingTensor->SetAllocatedSize(cost->npuWeightsTensor->AllocationSizeBytes());
            bufferingTensor->memArea = refCost->bufferedWeightTensor.tensor->memArea;
            cost->bufferedWeightTensor.buffering = Buffering::Single;  // Stripes are currently single-buffered
            cost->bufferedWeightTensor.preBuffer = false;
            cost->bufferedWeightTensor.tensor = std::move(bufferingTensor);
        }

        // Estimate performance
        cost->cycles = EstimateOpPerformance(schedOp, cost->Config(), schedOp->OFM()->SliceShape().Depth());
        cost->elementAccess = EstimateOpElementAccess(schedOp, cost->Config(), schedOp->OFM()->SliceShape().Depth());

        // sub-operations use the same stripe as their parent
        for ( auto &subOp : schedOp->SubOps() )
        {
            auto subCost = CreateSchedulerOpInfo(subOp.get(), stripe, cost);
            subCost->cycles = EstimateOpPerformance(subOp.get(), subCost->Config(), subOp->OFM()->SliceShape().Depth());
            subCost->elementAccess = EstimateOpElementAccess(subOp.get(), subCost->Config(), subOp->OFM()->SliceShape().Depth());
            stripedSchedule->SetCost(*subOp, std::move(subCost));
        }

        stripedSchedule->SetCost(*schedOp, std::move(cost));
        // Calculate the preceeding Op's stripe
        stripe = schedOp->IFM(schedOp->PrimaryIfmIndex())->shape.With(-3, stripe.Height() * schedOp->Kernel()->Stride().y);
    }
    return stripedSchedule;
}


Address Scheduler::EstimateScheduleMemoryUsage(Schedule *schedule, const std::unordered_map<UniqueId, int> &nonLocalMem)
{
    // Estimates the memory usage of a schedule
    // cascades = schedule.cascades;
    int peakMemUsage = 0;
    for ( auto const &schedOp : _ops )
    {
        auto cost = schedule->Cost(schedOp.get());
        if ( cost == nullptr )
        {
            // sched_op is not part of the sub-schedule - skip
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
            int opWeightBuffer = 0;
            if ( cost->bufferedWeightTensor.tensor )
            {
                opWeightBuffer = cost->bufferedWeightTensor.tensor->AllocationSizeBytes();
            }

            int opMemUsage = schedOp->IFM(0)->PartialAllocationSizeBytes() + schedOp->OFM()->PartialAllocationSizeBytes() + opWeightBuffer;
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

    // Extract the ops that are part of this sub-schedule
    vector_span<std::unique_ptr<SchedulerOperation>> subOps(_ops, cascadeInfo.start, cascadeInfo.end + 1);

    // Create a sub-schedule that contains only the costs for the Ops that are part of the sub-schedule
    auto subSchedule = std::make_shared<Schedule>(_name + fmt::format("SUB_{}_{}", cascadeInfo.start, cascadeInfo.end));
    for ( auto &op : subOps )
    {
        // NOTE: Copies the cost objects, consider optimising this
        auto costCopy = std::make_unique<SchedulerOpInfo>(*refSchedule->Cost(op.get()));
        subSchedule->SetCost(*op, std::move(costCopy));

        // chained sub-operations
        for ( auto &subOp : op->SubOps() )
        {
            costCopy = std::make_unique<SchedulerOpInfo>(*refSchedule->Cost(subOp.get()));
            subSchedule->SetCost(*subOp, std::move(costCopy));
        }
    }

    // Update subschedule cascade list
    subSchedule->cascades[cascadeInfo.end] = cascadeInfo;

    // Use the memory snapshot from the reference schedule (takes a copy)
    subSchedule->memorySnapshot = refSchedule->memorySnapshot;

    SchedulerOperation *firstOp = subOps.front().get();

    // Calculate memory usage that is live during the sub-schedule but not part of it
    int timeForCascade = refSchedule->Cost(firstOp)->timeIndex;

    int memUsageParallelToSubSchedule = refSchedule->MemoryUsageAt(timeForCascade) - cascadeInfo.memUsage;

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

    CascadeBuilder cascadeBuilder(subOps, nonLocalMemUsage, _spilling);

    // Start by adding buffering
    auto bufferedSubSchedule = ProposeScheduleBuffering(subSchedule.get(), _options.optimizationStagingLimit);

    // Copy the cascades over from the unbuffered-schedule
    bufferedSubSchedule->cascades = subSchedule->cascades;

    // Generate the possible stripings for the final Op in the sub-schedule
    Shape finalOFMShape = subOps.back()->OFM()->shape;
    const int maxStripeHeight = (finalOFMShape.Height() + 1) / 2;

    // Skip testing the min stripe used in the MIN schedule since that will be used
    // anyway if no new cascades are created below
    SchedulerOpInfo *minCost = refSchedule->Cost(subOps.back().get());
    const int minStripeHeight = minCost->stripe.Height() + 1;

    std::vector<Shape> possibleStripes;
    possibleStripes.reserve(maxStripeHeight);
    for ( int h = minStripeHeight; h <= maxStripeHeight; h++ )
    {
        possibleStripes.push_back(finalOFMShape.With(-3, h));
    }

    // Propose different striping - the possible stripes are proposed similarly to a binary search
    std::shared_ptr<Schedule> bestSchedule;

#if LOG_TRACE1_ON
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
                auto bufferTensor = schedOp->IFM(schedOp->PrimaryIfmIndex())->tensor.get();
                // Apply memory area
                bufferTensor->memArea = _arch->StagingMemory();
                // Apply rolling buffer dimensions
                Shape bufferShape = pos->second.shape;
                assert(!bufferTensor->needsLinearFormat);
                bufferTensor->format = idealFormat;
                assert(bufferShape.Width() == bufferTensor->storageShape.Width() && "Only y-striping implemented");
                bufferTensor->storageShape = bufferTensor->storageShape.WithHW(bufferShape.Height(), bufferShape.Width());
            }
        }

        // Check buffering tensors are meaningfully defined
        assert(!cost->bufferedWeightTensor.tensor || (cost->bufferedWeightTensor.tensor->srcTensor != nullptr));
    }
}


// Coalesce repeated weight buffer tensors
void Scheduler::CoalesceWeightBufferTensors(Schedule *schedule)
{
    SchedulerOpInfo *prevCost = nullptr;

    for ( auto &schedOp : _ops )
    {
        if ( !schedOp->IsNpuOp() )
        {
            continue;
        }

        auto cost = schedule->Cost(schedOp.get());
        if ( prevCost && cost )
        {
            auto &prevBufTensor = prevCost->bufferedWeightTensor.tensor;
            auto &bufTensor = cost->bufferedWeightTensor.tensor;
            if ( prevBufTensor && bufTensor )
            {
                UniqueId prevWeightsTensorId = prevCost->npuWeightsTensor ? prevCost->npuWeightsTensor->equivalenceId : -1;
                UniqueId weightsTensorId = cost->npuWeightsTensor ? cost->npuWeightsTensor->equivalenceId : -2;
                if ( prevWeightsTensorId == weightsTensorId && prevBufTensor->AllocationSizeBytes() == bufTensor->AllocationSizeBytes() &&
                     prevCost->ofmDepthSlices.size() == 2 && cost->ofmDepthSlices.size() == 2 && prevCost->ofmDepthSlices == cost->ofmDepthSlices )
                {
                    // Reuse previous weight buffer tensor if both current and previous op use 1 depth slice
                    // This will extend the life range weight buffer tensor
                    bufTensor = prevBufTensor;
                }
            }
        }

        prevCost = cost;
    }
}


PerformanceQuery Scheduler::InitPerfQuery(SchedulerOperation *op, ArchitectureOpConfig *config, int ofmDepth, WeightFormat wgtFormat)
{
    PerformanceQuery query = {};
    query.type = op->Type();
    query.kernel = op->Kernel();
    query.config = config;

    SchedulerConnection *ifm0 = op->IFM(0);
    query.ifmShape[0] = ifm0->SliceShape();
    query.ifmMemory[0] = ifm0->tensor->memArea.memory;
    query.ifmType[0] = ifm0->tensor->dataType;
    query.ifmFormat[0] = ifm0->tensor->format;

    SchedulerConnection *ifm1 = op->TryIFM(1);
    if ( ifm1 )
    {
        query.ifmShape[1] = ifm1->SliceShape();
        query.ifmMemory[1] = ifm1->tensor->memArea.memory;
        query.ifmType[1] = ifm1->tensor->dataType;
        query.ifmFormat[1] = ifm1->tensor->format;
    }

    SchedulerConnection *ofm = op->OFM();
    query.ofmShape = (ofmDepth >= 0) ? ofm->SliceShape().WithDepth(ofmDepth) : ofm->SliceShape();
    query.ofmMemory = ofm->tensor->memArea.memory;
    query.ofmType = ofm->tensor->dataType;
    query.ofmFormat = ofm->tensor->format;

    SchedulerConnection *scales = op->TryInput(TensorUsage::Scales);
    if ( scales )
    {
        query.constShape = Shape(1, 1, 1, query.ofmShape.Depth());
        query.constMemory = scales->tensor->memArea.memory;
    }

    query.weightFormat = wgtFormat;

    return query;
}


std::vector<FusionQuery> Scheduler::InitFusionQuery(SchedulerOperation *op)
{
    std::vector<FusionQuery> fused;
    if ( op->SubOps().size() && IsActivation(op->SubOps().front()->Type()) )
    {
        auto &subOp = op->SubOps().front();
        fused.emplace_back();

        FusionQuery &fusedOp = fused.back();
        fusedOp.type = subOp->Type();
        fusedOp.kernel = subOp->Kernel();
        auto ifm2 = subOp->TryIFM(1);
        if ( ifm2 )
        {
            fusedOp.ifm2Shape = ifm2->shape;
            fusedOp.ifm2Memory = ifm2->tensor->memArea.memory;
            fusedOp.ifm2Type = ifm2->tensor->dataType;
            fusedOp.ifm2Format = ifm2->tensor->format;
        }
    }

    return fused;
}


CycleCost Scheduler::EstimateOpPerformance(SchedulerOperation *op, ArchitectureOpConfig *config, int ofm_depth)
{
    CycleCost cycleCost;
    if ( !op->IsNpuOp() )
    {
        LOG_WARN("CPU performance estimation for \"{}\" not implemented\n", OpTypeToString(op->Type()));
        return cycleCost;
    }

    PerformanceQuery query = InitPerfQuery(op, config, ofm_depth);
    std::vector<FusionQuery> fused = InitFusionQuery(op);
    cycleCost = _arch->Performance()->MeasureCycleCost(query, fused);
    return cycleCost;
}

CycleCost Scheduler::EstimateOpPerformanceForSparsity(SchedulerOperation *op, ArchitectureOpConfig *config, int ofm_depth)
{
    CycleCost cycleCost;
    assert(op->IsNpuOp());

    PerformanceQuery query = InitPerfQuery(op, config, ofm_depth);
    std::vector<FusionQuery> fused = InitFusionQuery(op);
    cycleCost = _arch->Performance()->MeasureCycleCostForSparsity(query, fused);
    return cycleCost;
}

ElementAccess Scheduler::EstimateOpElementAccess(SchedulerOperation *op, ArchitectureOpConfig *config, int ofm_depth)
{
    // TODO MLBEDSW-7954: Account for chaining in performance estimation
    ElementAccess access;
    if ( !op->IsNpuOp() )
    {
        LOG_WARN("CPU performance estimation for \"{}\" not implemented\n", OpTypeToString(op->Type()));
        return access;
    }
    PerformanceQuery query = InitPerfQuery(op, config, ofm_depth);
    access = _arch->Performance()->MeasureElementAccess(query);
    return access;
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

        LOG_PRINT("\t{0}: Operation {1}  - OFM {2}\n", schedOp->Index(), OpTypeToString(schedOp->Type()),
            schedOp->OFM()->shape.ToString());
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

        int mem_usage = 0;
        if ( cost->timeIndex >= 0 && cost->timeIndex < int(schedule->memorySnapshot.size()) )
        {
            mem_usage = schedule->memorySnapshot[cost->timeIndex];
        }

        LOG_PRINT("\t\tEstimated Perf: Macs={0} Cycles={1}\n", cost->cycles.macs, cost->cycles.opCycles);
        LOG_PRINT("\t\tMemory Used: {0} bytes\n", mem_usage);
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
        LOG_PRINT("\t\t{0}: {1} -> {2}, size: {3}\n", key & 0xFFFF, cascade.start, cascade.end, cascade.memUsage);
    }
}


void ParseSchedulerOptions(SchedulerOptions &opt, IniReader &reader)
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

        reader.End();
    }
}


struct SchedulerTransformParam : public WeightTransformParam
{
    const int64_t *zeroPoints;
    int zeroCount;
};


static int ApplyZeroPointIHWO(const WeightTransformParam *param, int value)
{
    const SchedulerTransformParam *p = static_cast<const SchedulerTransformParam *>(param);
    value = (value - int(p->zeroPoints[p->o % p->zeroCount]));
    assert(value >= -255 && value <= 255);
    return value;
}


static int ApplyZeroPointOHWI(const WeightTransformParam *param, int value)
{
    const SchedulerTransformParam *p = static_cast<const SchedulerTransformParam *>(param);
    value = (value - int(p->zeroPoints[p->i % p->zeroCount]));
    assert(value >= -255 && value <= 255);
    return value;
}

WeightScaleTensors Scheduler::EncodeQuantizationScaleTensor(std::unique_ptr<IWeightEncodingConfig> encodingParams,
    const Quantization &ofmQuantization, const SchedulerTensor *scales)
{
    SchedulerTensor scaleTens;
    scaleTens.dataType = DataType::Int32;
    if ( scales == nullptr ) scales = &scaleTens;
    return TryEncodeWeightAndScaleTensor(encodingParams.get(), nullptr, scales, {}, ofmQuantization, false, true);
}

WeightScaleTensors Scheduler::EncodeWeightAndScaleTensor(std::unique_ptr<IWeightEncodingConfig> encodingParams, const SchedulerTensor *weightTens,
    const SchedulerTensor *scaleTens, const Quantization &weightQuantization, const Quantization &ofmQuantization)
{
    bool doWeights = true;
    bool doScales = true;

    // Check cache for weight tensors already encoded with this configuration.
    auto cacheKey = TensorCacheKey(encodingParams.get(), weightTens->equivalenceId);
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
    WeightScaleTensors result = TryEncodeWeightAndScaleTensor(
        encodingParams.get(), weightTens, scaleTens, weightQuantization, ofmQuantization, doWeights, doScales);

    if ( doWeights )
    {
        // Weights and scales now encoded together
        _tensorCache.emplace(cacheKey, result);
        result.npuWeightsTensor->config = std::move(encodingParams);
    }
    else
    {
        // Going to reuse a cached tensor for weights (must alias if memory areas don't match).
        if ( cachedWeightsTensor->memArea == weightTens->memArea )
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

WeightScaleTensors Scheduler::TryEncodeWeightAndScaleTensor(IWeightEncodingConfig *encodingParams,
    const SchedulerTensor *weightTens, const SchedulerTensor *scaleTens, const Quantization &weightQuantization,
    const Quantization &ofmQuantization, bool doWeights, bool doScales)
{
    assert(doWeights || doScales);

    // Create tensor to hold encoded output
    auto npuTensor = std::make_shared<NpuWeightTensor>();
    int rangeIndex = 0;
    int maxBufferLen[2] = {};
    std::vector<uint8_t> encodedStream;
    Shape ohwiShape;
    int channels;

    SchedulerTransformParam param;
    const uint8_t *weightsData = nullptr;
    Shape ohwiStrides;
    std::unique_ptr<IVolumeWeightSource> weightSource;
    std::unique_ptr<IVolumeScaleSource> scaleSource;

    if ( weightTens )
    {
        npuTensor->memArea = weightTens->memArea;
        ohwiStrides = GetOhwiStrides(weightTens);
        ohwiShape = GetOhwiShape(weightTens);

        const auto &weightView = weightTens->bufferView;
        weightsData = weightView.Buffer()->Data<uint8_t>();

        channels = ohwiShape[0];

        // Set up weight source
        param.zeroPoints = weightQuantization.zeroPoints.data();
        param.zeroCount = int(weightQuantization.zeroPoints.size());

        auto zeroOffsetFunc = (weightTens->srcTensor->AxisOrder() == AxisOrder::OHWI) ? ApplyZeroPointOHWI : ApplyZeroPointIHWO;
        weightSource = _arch->WeightEncoder()->GetWeightSource(encodingParams, weightTens->dataType, zeroOffsetFunc, &param);
    }
    else
    {
        npuTensor->memArea = _arch->ReadonlyMemory();
        channels = ofmQuantization.scales.size();
        ohwiShape = Shape{channels};
    }

    if ( doScales )
        scaleSource = _arch->WeightEncoder()->GetScaleSource(encodingParams, scaleTens->dataType, ofmQuantization);

    int totalSourceBytes = 0;
    int totalWeightBytes = 0;
    int subStreams = 1;
    int scaleStreamsRequired = 1;
    int streamsRequired = _arch->WeightEncoder()->StreamsRequired(encodingParams, ohwiShape, scaleStreamsRequired);
    std::bitset<64> distinctWeights[8];

    if ( weightTens == nullptr ) streamsRequired = scaleStreamsRequired;

    // Note: in case of multiple cores, each core's weights are interleaved in O-dimension
    auto const &depthOffsets = encodingParams->DepthOffsets();
    const int nrDepthOffsets = int(depthOffsets.size());
    for ( int idx = 0; idx < nrDepthOffsets - 1; ++idx )
    {
        int depthOffset = depthOffsets[idx];

        // Do not generate for offsets outside the OFM
        assert(depthOffset >= 0 && depthOffset < channels);
        int depthLength = depthOffsets[idx + 1] - depthOffset;

        int bufferStartOffset = int(encodedStream.size());

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
                int biasCount = scaleTens->bufferView.HasBuffer() ? scaleTens->bufferView.ViewShape().Depth() : depthOffset + depthLength;
                scaleSource->SetSource(biases, biasCount, depthOffset, depthLength, stream);
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
                ohwiShape[0] = depthLength;
                weightSource->SetSource(weightsData, depthOffset, ohwiShape, ohwiStrides, stream);
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
        maxBufferLen[idx % 2] = std::max(maxBufferLen[idx % 2], int(encodedStream.size()) - bufferStartOffset);
    }

    // Reduce stored memory usage as much as possible
    encodedStream.shrink_to_fit();

    auto encodedTensor = std::make_shared<Tensor>(
        doWeights ?
            weightTens->Name() :
        scaleTens->bufferView.HasBuffer() ?
            scaleTens->Name() :
            "Scales",
        DataType::UInt8);
    int streamSize = int(encodedStream.size());
    auto buf = std::make_shared<Buffer>(std::move(encodedStream));
    encodedTensor->SetStorageShape(Shape(1, 1, 1, streamSize));
    encodedTensor->SetBuffer(buf);

    npuTensor->srcTensor = encodedTensor;
    npuTensor->maxRangeBytes = std::max(maxBufferLen[0], maxBufferLen[1]);
    npuTensor->doubleBufferSize = maxBufferLen[0] + maxBufferLen[1];
    npuTensor->doubleBufferOffset = maxBufferLen[0];
    npuTensor->totalSourceBytes = totalSourceBytes;
    npuTensor->totalWeightBytes = totalWeightBytes;
    npuTensor->subStreams = subStreams;
    npuTensor->storageShape = encodedTensor->StorageShape();
    npuTensor->SetAllocatedSize(encodedTensor->View().Buffer()->Size());

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

    return result;
}

}  // namespace regor
