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
    if ( opType != OpType::FullyConnected && opType != OpType::MatMul && opType != OpType::MemoryCopy ) return nullptr;
    if ( query.ifmBits != 8 || query.ofmBits != 8 || query.transpose != TransposeType::None ||
         query.reverse != ReverseType::None )
    {
        return nullptr;
    }
    return std::make_unique<NeuralAIOpConfig>();
}

std::unique_ptr<ArchitectureOpGroup> ArchNeuralAI::CreateOpGroup(const ArchitectureOpGroupQuery &op)
{
    auto group = std::make_unique<NeuralAIOpGroup>();
    return group->Add(op) ? std::move(group) : nullptr;
}

int ArchNeuralAI::TensorAlignment(TensorUsage, TensorFormat) const
{
    return DMAAlignment;
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
    const int bufferHeight = RoundAway(producerShape.Height() + consumerShape.Height(), consumerShape.Height());
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

AxisMask ArchNeuralAI::CanSubdivide(OpType, TransposeType, ReverseType)
{
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
