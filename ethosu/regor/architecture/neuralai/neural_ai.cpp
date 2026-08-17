//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai.hpp"

#include "common/logging.hpp"

#include "architecture/neuralai/neural_ai_abi.hpp"
#include "architecture/neuralai/neural_ai_constraints.hpp"
#include "architecture/neuralai/neural_ai_op_config.hpp"
#include "architecture/neuralai/neural_ai_performance.hpp"
#include "architecture/neuralai/neural_ai_weight_encoder.hpp"
#include "include/regor.h"

namespace regor
{
namespace
{

bool ReadFixedConfigValue(IniReader *reader, const std::string &key, int expected)
{
    const int value = reader->Get<int>();
    if ( value != expected )
    {
        LOG_ERROR("Neural-AI architecture option '{}' must be {}, got {}\n", key, expected, value);
        return false;
    }
    return true;
}

}  // namespace

ArchNeuralAI::ArchNeuralAI()
{
    _constraints = std::make_unique<NeuralAIConstraints>();
    _weightEncoder = std::make_unique<NeuralAIWeightEncoder>();
    _performance = std::make_unique<NeuralAIPerformance>();
    auto modelMemory = std::make_unique<ArchitectureMemory>("model", MaxAddress());
    modelMemory->SetParameters(DMAAlignment, 1, 1, DMAAlignment, 1, 1, 1);
    _modelMemory = modelMemory.get();
    _memories.emplace("model", std::move(modelMemory));

    auto tcdmMemory = std::make_unique<ArchitectureMemory>("tcdm", AllocatableTCDMBytes);
    tcdmMemory->SetParameters(DMAAlignment, 1, 1, DMAAlignment, 1, 1, 1);
    _tcdmMemory = tcdmMemory.get();
    _memories.emplace("tcdm", std::move(tcdmMemory));

    _readonlyMemory = _modelMemory;
    _featuremapMemory = _tcdmMemory;
    _stagingMemory = _tcdmMemory;
    _lutMemory = _tcdmMemory;
}

ArchNeuralAI::~ArchNeuralAI() = default;

bool ArchNeuralAI::ParseConfig(IniReader *reader)
{
    bool valid = true;
    std::string key;
    while ( reader->Begin(key) )
    {
        if ( key == "clusters" ) valid &= ReadFixedConfigValue(reader, key, Clusters);
        else if ( key == "array_dimension" ) valid &= ReadFixedConfigValue(reader, key, ArrayDimension);
        else if ( key == "dma_alignment" ) valid &= ReadFixedConfigValue(reader, key, DMAAlignment);
        else if ( key == "tcdm_size" ) valid &= ReadFixedConfigValue(reader, key, TCDMSizeBytes);
        else if ( key == "command_staging_size" ) valid &= ReadFixedConfigValue(reader, key, CommandStagingBytes);
        else if ( key == "linebuffer_max_kernel" ) valid &= ReadFixedConfigValue(reader, key, LineBufferMaxKernel);
        else if ( key == "linebuffer_max_stride" ) valid &= ReadFixedConfigValue(reader, key, LineBufferMaxStride);
        else if ( key == "linebuffer_max_input_width" )
            valid &= ReadFixedConfigValue(reader, key, LineBufferMaxInputWidth);
        else if ( key == "requant_shift_max" ) valid &= ReadFixedConfigValue(reader, key, RequantShiftMax);
        else LOG_WARN("Skipping unrecognised Neural-AI architecture option '{}'.\n", key);
        reader->End();
    }
    return valid;
}

bool ArchNeuralAI::CheckConfiguration(std::string &error)
{
    if ( !Architecture::CheckConfiguration(error) ) return false;
    if ( _readonlyMemory != _modelMemory || _featuremapMemory != _tcdmMemory || _stagingMemory != _tcdmMemory ||
         _lutMemory != _tcdmMemory )
    {
        error = "Neural-AI memory roles must use the fixed model and TCDM arenas";
        return false;
    }
    if ( _tcdmMemory->SizeBytes() != AllocatableTCDMBytes )
    {
        error = "Neural-AI allocatable TCDM size conflicts with the hardware contract";
        return false;
    }
    return true;
}

std::unique_ptr<ArchitectureOpConfig> ArchNeuralAI::GetOpConfig(OpType opType, const ArchitectureConfigQuery &query)
{
    if ( opType != OpType::FullyConnected && opType != OpType::MatMul && opType != OpType::Conv2D &&
         opType != OpType::DepthwiseConv2D &&
         opType != OpType::LUT &&
         opType != OpType::Add &&
         opType != OpType::AvgPool &&
         opType != OpType::MaxPool &&
         opType != OpType::Concat &&
         opType != OpType::Transpose &&
         opType != OpType::Dfl && opType != OpType::MemoryCopy ) return nullptr;
    const bool headPack = opType == OpType::Transpose && query.transpose == TransposeType::NCHW &&
                          query.ifmShape[0].Size() == 4 && query.ofmShape.Size() == 4 &&
                          query.ifmShape[0].Batch() == 1 && query.ifmShape[0].Depth() == 144 &&
                          (query.ifmShape[0].Height() == 10 || query.ifmShape[0].Height() == 20 ||
                              query.ifmShape[0].Height() == 40) &&
                          query.ifmShape[0].Width() == query.ifmShape[0].Height() &&
                          (query.ofmShape == query.ifmShape[0] ||
                              query.ofmShape == Shape(1, 144, query.ifmShape[0].Height(),
                                                    query.ifmShape[0].Width()));
    if ( query.ifmBits != 8 || query.ofmBits != 8 ||
         (query.transpose != TransposeType::None && !headPack) ||
         query.reverse != ReverseType::None )
    {
        return nullptr;
    }
    NeuralAIOpMode mode = NeuralAIOpMode::Unsupported;
    bool directNhwcInput = false;
    if ( opType == OpType::FullyConnected ) mode = NeuralAIOpMode::FullyConnectedRow32;
    else if ( opType == OpType::MatMul ) mode = NeuralAIOpMode::MatMulRow32;
    else if ( opType == OpType::LUT ) mode = NeuralAIOpMode::AFULutI8;
    else if ( opType == OpType::Add ) mode = NeuralAIOpMode::AddI8;
    else if ( opType == OpType::AvgPool )
        mode = query.ifmResampling == ArchResampling::Nearest ?
            NeuralAIOpMode::UpsampleNearestC32 : NeuralAIOpMode::AFUGlobalAvgPoolC32;
    else if ( opType == OpType::MaxPool ) mode = NeuralAIOpMode::MaxPoolK5S1P2C32;
    else if ( opType == OpType::Concat ) mode = NeuralAIOpMode::ConcatC32;
    else if ( opType == OpType::Dfl ) mode = NeuralAIOpMode::Dfl16;
    else if ( headPack ) mode = NeuralAIOpMode::HeadPackC32ToCHW;
    else if ( opType == OpType::Conv2D && query.kernel != nullptr )
    {
        const auto &kernel = *query.kernel;
        const bool sameSpatial = kernel.Stride().x > 0 && kernel.Stride().y > 0 &&
            query.ofmShape.Height() ==
                (query.ifmShape[0].Height() + kernel.Stride().y - 1) / kernel.Stride().y &&
            query.ofmShape.Width() ==
                (query.ifmShape[0].Width() + kernel.Stride().x - 1) / kernel.Stride().x;
        const auto validStripeDimension = [](int ifm, int ofm, int stride, int kernelSize)
        {
            if ( ifm <= 0 || ofm <= 0 || stride <= 0 || kernelSize <= 0 ) return false;
            const int64_t first = int64_t(ofm - 1) * stride + 1;
            const int64_t last = int64_t(ofm - 1) * stride + kernelSize;
            return int64_t(ifm) >= first && int64_t(ifm) <= last;
        };
        const bool validStripeSpatial =
            validStripeDimension(query.ifmShape[0].Height(), query.ofmShape.Height(),
                kernel.Stride().y, kernel.Size().y) &&
            validStripeDimension(query.ifmShape[0].Width(), query.ofmShape.Width(),
                kernel.Stride().x, kernel.Size().x);
        if ( kernel.Size() == Point2i(1, 1) && kernel.Stride() == Point2i(1, 1) &&
             kernel.Dilation() == Point2i(1, 1) && kernel.Padding().IsZero() )
            mode = NeuralAIOpMode::Conv2DPointwiseC32Requant;
        else if ( kernel.Size() == Point2i(3, 3) && kernel.Stride() == Point2i(2, 2) &&
                  kernel.Dilation() == Point2i(1, 1) &&
                  (sameSpatial || validStripeSpatial || kernel.Padding().IsZero()) &&
                  query.ifmShape[0].Depth() == 3 && query.ofmShape.Depth() > 0 &&
                  query.ofmShape.Depth() <= 32 &&
                  kernel.Padding().Top() >= 0 && kernel.Padding().Top() <= 1 &&
                  kernel.Padding().Left() >= 0 && kernel.Padding().Left() <= 1 &&
                  kernel.Padding().Bottom() >= 0 && kernel.Padding().Bottom() <= 1 &&
                  kernel.Padding().Right() >= 0 && kernel.Padding().Right() <= 1 )
        {
            mode = NeuralAIOpMode::Conv2DRgbLinebufRequant;
            directNhwcInput = true;
        }
        else if ( kernel.Size() == Point2i(3, 3) &&
                  (kernel.Stride() == Point2i(1, 1) || kernel.Stride() == Point2i(2, 2)) &&
                  kernel.Dilation() == Point2i(1, 1) &&
                  (sameSpatial || validStripeSpatial || kernel.Padding().IsZero()) &&
                  kernel.Padding().Top() >= 0 && kernel.Padding().Top() <= 1 &&
                  kernel.Padding().Left() >= 0 && kernel.Padding().Left() <= 1 &&
                  kernel.Padding().Bottom() >= 0 && kernel.Padding().Bottom() <= 1 &&
                  kernel.Padding().Right() >= 0 && kernel.Padding().Right() <= 1 )
        {
            const int ifmTail = query.ifmShape[0].Depth() % ArrayDimension;
            const int ofmTail = query.ofmShape.Depth() % ArrayDimension;
            // A non-final C32 input group may be represented by the existing
            // block-valid-lane contract.  Admit only a 16-lane remainder;
            // the linebuffer compiler emits full C32 jobs followed by one
            // masked tail job and keeps all other arbitrary tails rejected.
            const bool supportedC32Tails = (ifmTail == 0 || ifmTail == 16) &&
                                           (ofmTail == 0 || ofmTail == 16);
            if ( supportedC32Tails )
                mode = kernel.Stride() == Point2i(1, 1) ? NeuralAIOpMode::Conv2DLinebufC32S1Requant :
                                                         NeuralAIOpMode::Conv2DLinebufC32S2Requant;
        }
    }
    else if ( opType == OpType::DepthwiseConv2D && query.kernel != nullptr &&
              query.kernel->Size() == Point2i(3, 3) &&
              (query.kernel->Stride() == Point2i(1, 1) ||
               query.kernel->Stride() == Point2i(2, 2)) &&
              query.kernel->Dilation() == Point2i(1, 1) &&
              (query.kernel->Padding().IsZero() ||
                  (query.ofmShape.Height() ==
                          (query.ifmShape[0].Height() + query.kernel->Stride().y - 1) /
                              query.kernel->Stride().y &&
                      query.ofmShape.Width() ==
                          (query.ifmShape[0].Width() + query.kernel->Stride().x - 1) /
                              query.kernel->Stride().x)) &&
              query.kernel->Padding().Top() >= 0 && query.kernel->Padding().Top() <= 1 &&
              query.kernel->Padding().Left() >= 0 && query.kernel->Padding().Left() <= 1 &&
              query.kernel->Padding().Bottom() >= 0 && query.kernel->Padding().Bottom() <= 1 &&
              query.kernel->Padding().Right() >= 0 && query.kernel->Padding().Right() <= 1 &&
              query.ifmShape[0].Depth() == query.ofmShape.Depth() )
    {
        mode = query.kernel->Stride() == Point2i(2, 2) ? NeuralAIOpMode::DepthwiseC32S2Requant :
                                                         NeuralAIOpMode::DepthwiseC32S1Requant;
    }
    const bool groupStationary = (mode == NeuralAIOpMode::Conv2DLinebufC32S1Requant ||
                                  mode == NeuralAIOpMode::Conv2DLinebufC32S2Requant) &&
                                 query.ifmShape[0].Depth() > ArrayDimension &&
                                 query.ifmShape[0].Depth() % ArrayDimension == 0 &&
                                 query.ofmShape.Depth() % ArrayDimension == 0;
    return std::make_unique<NeuralAIOpConfig>(256, mode, directNhwcInput, groupStationary);
}

std::unique_ptr<ArchitectureOpGroup> ArchNeuralAI::CreateOpGroup(const ArchitectureOpGroupQuery &op)
{
    auto group = std::make_unique<NeuralAIOpGroup>();
    return group->Add(op) ? std::move(group) : nullptr;
}

int ArchNeuralAI::TensorAlignment(TensorUsage, TensorFormat format) const
{
    // Compact public NHWC (and scalar/layout-only temporaries represented in
    // that format) may start at any byte address.  Native ROW32/C32 and
    // encoded weights are consumed by 32-byte engines and retain the DMA
    // alignment requirement.
    return format == TensorFormat::NHWC || format == TensorFormat::CompactNHWC ? 1 : DMAAlignment;
}

TensorFormat ArchNeuralAI::ModelBindingFormat(TensorUsage) const
{
    return TensorFormat::NHWC;
}

TensorFormat ArchNeuralAI::DefaultInternalTensorFormat(TensorUsage usage, bool linearRequired) const
{
    return IsIFM(usage) || IsOFM(usage) ?
        (linearRequired ? TensorFormat::NHWC : TensorFormat::Row32) : TensorFormat::Unknown;
}

Shape ArchNeuralAI::StorageShape(const Shape &logicalShape, TensorFormat format) const
{
    if ( format == TensorFormat::Row32 || format == TensorFormat::C32Blocked )
    {
        return logicalShape.WithDepth(RoundAway(logicalShape.Depth(), ArrayDimension));
    }
    return logicalShape;
}

Shape ArchNeuralAI::TensorStrides(const Shape &logicalShape, TensorFormat format, DataType dataType) const
{
    if ( format == TensorFormat::Row32 )
    {
        const int elementBytes = DataTypeSizeBits(dataType) / 8;
        const Shape storageShape = StorageShape(logicalShape, format);
        const int strideC = elementBytes;
        const int strideX = storageShape.Depth() * strideC;
        const int strideY = logicalShape.Width() * strideX;
        const int strideN = logicalShape.Height() * strideY;
        return Shape(strideN, strideY, strideX, strideC);
    }
    if ( format == TensorFormat::C32Blocked )
    {
        // C32 channel addressing is non-affine and is handled by the Neural-AI command generator.
        return Shape();
    }
    return Architecture::TensorStrides(logicalShape, format, dataType);
}

bool ArchNeuralAI::CanAliasDepthOffset(TensorFormat format, int depthOffset) const
{
    if ( format == TensorFormat::Row32 || format == TensorFormat::C32Blocked )
    {
        return depthOffset % ArrayDimension == 0;
    }
    return Architecture::CanAliasDepthOffset(format, depthOffset);
}

Shape ArchNeuralAI::RollingBufferShape(const Shape &producerShape, const Shape &consumerShape,
    TensorFormat format) const
{
    const bool structuralCspConcatRow = format == TensorFormat::C32Blocked &&
        producerShape.Size() == 4 && producerShape.Batch() == 1 &&
        producerShape.Height() == 1 && producerShape.Width() > 0 &&
        producerShape.Depth() == 48 && consumerShape == producerShape;
    const int bufferHeight = structuralCspConcatRow ? 1 :
        RoundAway(producerShape.Height() + consumerShape.Height(), consumerShape.Height());
    return StorageShape(consumerShape.With(-3, bufferHeight).WithDepth(producerShape.Depth()), format);
}

uint32_t ArchNeuralAI::Version()
{
    return (uint32_t(neuralai::AbiMajor) << 16) | neuralai::AbiMinor;
}

int ArchNeuralAI::UpscaleAndRounding(ArchResampling, int &rounding)
{
    rounding = 0;
    return 1;
}

AxisMask ArchNeuralAI::CanSubdivide(OpType opType, TransposeType transpose, ReverseType reverse)
{
    if ( transpose == TransposeType::None && reverse == ReverseType::None &&
         (opType == OpType::Conv2D || opType == OpType::DepthwiseConv2D || opType == OpType::LUT ||
             opType == OpType::Concat) )
    {
        // Neural-AI linebuffer and AFU commands preserve complete rows.  Width
        // subdivision would require a second address-wrap contract, while Y
        // stripes map directly to the target's rolling row buffers.
        return AxisMask::AxisY;
    }
    return AxisMask::None;
}

bool ArchNeuralAI::SupportsScalar(OpType, DataType, TensorUsage)
{
    return false;
}

Flags<WeightFormat> ArchNeuralAI::SupportedWeightFormat(OpType)
{
    return Flags<WeightFormat>(WeightFormat::Default);
}

void ArchNeuralAI::Call(std::function<void(const std::string &)> callBack)
{
    callBack(REGOR_ARCH_NEURALAI);
}

}  // namespace regor
