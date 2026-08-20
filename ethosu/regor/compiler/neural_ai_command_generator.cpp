//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_command_generator.hpp"

#include "common/numeric_util.hpp"

#include "architecture/neuralai/neural_ai.hpp"
#include "architecture/neuralai/neural_ai_op_config.hpp"
#include "architecture/neuralai/neural_ai_quantization.hpp"
#include "compiler/high_level_command_stream_generator.hpp"
#include "compiler/shape_util.hpp"
#include "tflite/tflite_schema_generated.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace regor
{
namespace
{

using neuralai::AFUBinaryMode;
using neuralai::BindingDirection;
using neuralai::CommandType;
using neuralai::CopyLayoutMode;
using neuralai::DataType;
using neuralai::DMADirection;
using neuralai::RefV1;
using neuralai::Region;
using neuralai::SpatzBinaryMode;
using neuralai::TensorLayout;

constexpr int MaxDirectLinebufferM = 1024;
constexpr int MaxExternalPsumLinebufferM = 256;

bool IsLinebufferConvMode(NeuralAIOpMode mode)
{
    return mode == NeuralAIOpMode::Conv2DRgbLinebufRequant ||
           mode == NeuralAIOpMode::Conv2DLinebufC32S1Requant ||
           mode == NeuralAIOpMode::Conv2DLinebufC32S2Requant;
}

bool IsDepthwiseMode(NeuralAIOpMode mode)
{
    return mode == NeuralAIOpMode::DepthwiseC32S1Requant ||
           mode == NeuralAIOpMode::DepthwiseC32S2Requant;
}

bool IsFullTensorConnection(const SchedulerConnection *connection)
{
    const Shape &shape = connection->shape;
    return connection->SliceShape() == shape &&
           (!connection->slice.offset || connection->slice.offset == shape.WithZeros()) &&
           (!connection->slice.stride || connection->slice.stride == shape.WithOnes());
}

bool IsCompactNHWC(TensorFormat format)
{
    return format == TensorFormat::NHWC || format == TensorFormat::CompactNHWC;
}

int64_t TensorStorageBytes(const SchedulerTensor *tensor)
{
    return tensor->architecture->StorageBytes(tensor->storageShape, tensor->format, tensor->dataType);
}

bool C32SliceByteOffset(const SchedulerConnection *connection, uint32_t &offset,
    std::string &error)
{
    offset = 0;
    if ( !connection->slice ) return true;
    const Shape fullShape = ReshapeToNHWC(connection->shape);
    const Shape sliceShape = ReshapeToNHWC(connection->SliceShape());
    const Shape sliceOffset = ReshapeToNHWC(connection->slice.offset);
    const Shape sliceStride = connection->slice.stride ?
        ReshapeToNHWC(connection->slice.stride) : fullShape.WithOnes();
    const bool c32Aligned = sliceOffset.Depth() >= 0 && sliceOffset.Depth() % 32 == 0 &&
        sliceShape.Depth() > 0 && sliceShape.Depth() % 32 == 0;
    const bool lowC16 = fullShape.Depth() == 32 && sliceOffset.Depth() == 0 &&
        sliceShape.Depth() == 16;
    if ( connection->tensor->format != TensorFormat::C32Blocked ||
         sliceShape.WithDepth(1) != fullShape.WithDepth(1) ||
         sliceOffset.WithDepth(0) != fullShape.WithZeros() ||
         sliceStride != sliceStride.WithOnes() || (!c32Aligned && !lowC16) ||
         sliceOffset.Depth() + sliceShape.Depth() > fullShape.Depth() )
    {
        error = "Neural-AI native tensor slice requires a full-spatial C32-aligned depth view";
        return false;
    }
    const uint64_t groupPlaneBytes = uint64_t(fullShape.Height()) *
        uint64_t(fullShape.Width()) * 32u;
    const uint64_t byteOffset = uint64_t(sliceOffset.Depth() / 32) * groupPlaneBytes;
    if ( byteOffset > std::numeric_limits<uint32_t>::max() )
    {
        error = "Neural-AI native tensor slice offset overflows the ABI";
        return false;
    }
    offset = uint32_t(byteOffset);
    return true;
}

bool CompactPlaneSliceByteOffset(const SchedulerConnection *connection, uint32_t &offset,
    std::string &error)
{
    const Shape fullShape = ReshapeToNHWC(connection->shape);
    const Shape sliceShape = ReshapeToNHWC(connection->SliceShape());
    const Shape sliceOffset = connection->slice.offset ?
        ReshapeToNHWC(connection->slice.offset) : fullShape.WithZeros();
    const Shape sliceStride = connection->slice.stride ?
        ReshapeToNHWC(connection->slice.stride) : fullShape.WithOnes();
    if ( !IsCompactNHWC(connection->tensor->format) || fullShape.Batch() != 1 ||
         fullShape.Height() != 1 || fullShape.Width() <= 0 || fullShape.Depth() <= 0 ||
         sliceShape.Batch() != 1 || sliceShape.Height() != 1 || sliceShape.Width() <= 0 ||
         sliceShape.Depth() != fullShape.Depth() || sliceOffset.Batch() != 0 ||
         sliceOffset.Height() != 0 || sliceOffset.Width() < 0 || sliceOffset.Depth() != 0 ||
         sliceOffset.Width() + sliceShape.Width() > fullShape.Width() ||
         sliceStride != fullShape.WithOnes() )
    {
        error = "Neural-AI compact plane view must retain the full innermost dimension";
        return false;
    }
    const uint64_t byteOffset = uint64_t(sliceOffset.Width()) * uint64_t(fullShape.Depth());
    if ( byteOffset > std::numeric_limits<uint32_t>::max() )
    {
        error = "Neural-AI compact plane view offset overflows the ABI";
        return false;
    }
    offset = uint32_t(byteOffset);
    return true;
}

bool IsLocalDMARegion(uint16_t region)
{
    return region == uint16_t(Region::TCDMScratch) ||
           region == uint16_t(Region::DTCMRuntime);
}

bool IsExternalDMARegion(uint16_t region)
{
    return region == uint16_t(Region::ModelConstants) ||
           region == uint16_t(Region::ModelCommands) ||
           region == uint16_t(Region::InputBinding) ||
           region == uint16_t(Region::OutputBinding) ||
           region == uint16_t(Region::L2TemporaryBinding);
}

bool ResolveDMADirection(const RefV1 &source, const RefV1 &destination, DMADirection &direction)
{
    const bool sourceLocal = IsLocalDMARegion(source.region);
    const bool destinationLocal = IsLocalDMARegion(destination.region);
    if ( (!sourceLocal && !IsExternalDMARegion(source.region)) ||
         (!destinationLocal && !IsExternalDMARegion(destination.region)) )
        return false;
    if ( !sourceLocal && !destinationLocal ) return false;
    direction = sourceLocal ?
        (destinationLocal ? DMADirection::LocalToLocal : DMADirection::LocalToExternal) :
        DMADirection::ExternalToLocal;
    return true;
}

void Append16(std::vector<uint8_t> &output, uint16_t value)
{
    output.push_back(uint8_t(value));
    output.push_back(uint8_t(value >> 8));
}

void Append32(std::vector<uint8_t> &output, uint32_t value)
{
    output.push_back(uint8_t(value));
    output.push_back(uint8_t(value >> 8));
    output.push_back(uint8_t(value >> 16));
    output.push_back(uint8_t(value >> 24));
}

uint32_t Read32(const uint8_t *data)
{
    return uint32_t(data[0]) | (uint32_t(data[1]) << 8) | (uint32_t(data[2]) << 16) |
           (uint32_t(data[3]) << 24);
}

void AppendRef(std::vector<uint8_t> &output, const RefV1 &reference)
{
    Append16(output, reference.region);
    Append16(output, reference.index);
    Append32(output, reference.offset);
}

void AppendHeader(std::vector<uint8_t> &output, CommandType type, uint16_t size,
    uint32_t layerId, uint32_t tileId)
{
    Append16(output, uint16_t(type));
    Append16(output, size);
    Append32(output, 0);
    Append32(output, layerId);
    Append32(output, tileId);
}

void AppendZeros(std::vector<uint8_t> &output, int words)
{
    for ( int index = 0; index < words; ++index ) Append32(output, 0);
}

uint16_t ABIDataType(regor::DataType type)
{
    if ( type == regor::DataType::Int8 ) return uint16_t(DataType::Int8);
    if ( type == regor::DataType::Int32 ) return uint16_t(DataType::Int32);
    return 0;
}

uint16_t ABILayout(TensorFormat format)
{
    if ( format == TensorFormat::NHWC || format == TensorFormat::CompactNHWC )
        return uint16_t(TensorLayout::NHWC);
    if ( format == TensorFormat::Row32 ) return uint16_t(TensorLayout::Row32);
    if ( format == TensorFormat::C32Blocked ) return uint16_t(TensorLayout::C32Blocked);
    return 0;
}

std::array<uint32_t, 4> Dimensions(const Shape &shape)
{
    const Shape nhwc = ReshapeToNHWC(shape);
    return {uint32_t(nhwc.Batch()), uint32_t(nhwc.Height()), uint32_t(nhwc.Width()), uint32_t(nhwc.Depth())};
}

uint32_t FloatBits(float value)
{
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void BindingQuantization(const Tensor *tensor, const SchedulerConnection *connection,
    uint32_t &scaleBits, int32_t &zeroPoint)
{
    const auto *tfliteTensor = tensor->Passthrough() ?
        static_cast<const tflite::Tensor *>(tensor->Passthrough()) : nullptr;
    const auto *quantization = tfliteTensor ? tfliteTensor->quantization() : nullptr;
    if ( quantization != nullptr && quantization->scale() != nullptr &&
         quantization->scale()->size() == 1 && quantization->zero_point() != nullptr &&
         quantization->zero_point()->size() == 1 )
    {
        scaleBits = FloatBits((*quantization->scale())[0]);
        zeroPoint = int32_t((*quantization->zero_point())[0]);
        return;
    }
    scaleBits = FloatBits(float(connection->quantization.Scale().Dequantize()));
    zeroPoint = connection->quantization.zeroPoints.empty() ? 0 :
        int32_t(connection->quantization.zeroPoints[0]);
}

struct GeneratorContext
{
    struct MatrixData
    {
        uint32_t qparamBase = 0;
        uint32_t weightBase = 0;
        bool linebufferWeightsStaged = false;
    };

    const Graph *graph;
    const Schedule *schedule;
    CompiledNeuralAIArtifact *artifact;
    std::unordered_map<UniqueId, uint16_t> inputBindings;
    std::unordered_map<UniqueId, uint16_t> outputBindings;
    std::unordered_map<UniqueId, uint32_t> constantOffsets;
    std::unordered_map<UniqueId, RefV1> lutReferences;
    std::unordered_map<int, RefV1> fillReferences;
    std::unordered_map<UniqueId, MatrixData> matrixData;
    uint32_t nextTensorId = 0;
    uint32_t scratchEnd = 0;
    uint32_t stageOffset = 0;
    uint32_t weightStageOffset = 0;
    uint32_t partialOffset = 0;
    uint32_t stripeStageOffset = 0;
    uint32_t stageBytes = 0;
    uint32_t partialBytes = 0;
    uint32_t stripeStageBytes = 0;

    GeneratorContext(const Graph *sourceGraph, const Schedule *sourceSchedule,
        CompiledNeuralAIArtifact *output) :
            graph(sourceGraph), schedule(sourceSchedule), artifact(output)
    {
    }

    bool SetError(std::string &error, const std::string &message)
    {
        error = message;
        return false;
    }

    RefV1 TensorRef(const SchedulerTensor *tensor, uint32_t offset, std::string &error)
    {
        RefV1 reference{};
        if ( tensor->isGraphInput )
        {
            auto position = inputBindings.find(tensor->srcTensor->Uid());
            if ( position == inputBindings.end() )
            {
                error = "Neural-AI input tensor has no public binding";
                return reference;
            }
            reference.region = uint16_t(Region::InputBinding);
            reference.index = position->second;
        }
        else if ( tensor->isGraphOutput )
        {
            auto position = outputBindings.find(tensor->srcTensor->Uid());
            if ( position == outputBindings.end() )
            {
                error = "Neural-AI output tensor has no public binding";
                return reference;
            }
            reference.region = uint16_t(Region::OutputBinding);
            reference.index = position->second;
        }
        else if ( tensor->IsConstant() )
        {
            auto position = constantOffsets.find(tensor->uid);
            if ( position == constantOffsets.end() )
            {
                const uint32_t aligned = uint32_t(RoundAway(
                    int(artifact->constants.size()), int(neuralai::Alignment)));
                artifact->constants.resize(aligned, 0);
                const uint32_t constantOffset = uint32_t(artifact->constants.size());
                const int elementBytes = tensor->dataType == regor::DataType::Int8 ? 1 :
                    tensor->dataType == regor::DataType::Int32 ? 4 : 0;
                if ( elementBytes == 0 )
                {
                    error = "Neural-AI constant DMA supports only INT8 and INT32";
                    return reference;
                }
                const Shape expectedStrides = Shape::GetStridesForShape(
                    tensor->bufferView.ViewShape(), elementBytes);
                if ( tensor->bufferView.StrideBytes() != expectedStrides )
                {
                    error = "Neural-AI constant DMA requires a contiguous tensor view";
                    return reference;
                }
                const int64_t byteCount = int64_t(tensor->bufferView.Elements()) * elementBytes;
                const int64_t baseByte = int64_t(tensor->bufferView.BaseOffset()) * elementBytes;
                if ( byteCount <= 0 || baseByte < 0 ||
                     baseByte + byteCount > int64_t(tensor->bufferView.Buffer()->Size()) )
                {
                    error = "Neural-AI constant DMA view is outside its backing buffer";
                    return reference;
                }
                const uint8_t *source = tensor->bufferView.Buffer()->Data<uint8_t>() + baseByte;
                artifact->constants.insert(artifact->constants.end(), source, source + byteCount);
                position = constantOffsets.emplace(tensor->uid, constantOffset).first;
            }
            reference.region = uint16_t(Region::ModelConstants);
            reference.offset = position->second;
        }
        else
        {
            if ( tensor->AllocatedAddress() < 0 || tensor->AllocatedAddress() > std::numeric_limits<uint32_t>::max() )
            {
                error = "Neural-AI scratch tensor has no valid allocated address";
                return reference;
            }
            reference.region = uint16_t(Region::TCDMScratch);
            reference.offset = uint32_t(tensor->AllocatedAddress());
        }
        if ( offset > std::numeric_limits<uint32_t>::max() - reference.offset )
        {
            error = "Neural-AI tensor reference offset overflows the ABI";
            return {};
        }
        reference.offset += offset;
        return reference;
    }

    void AppendControl(CommandType type, uint32_t layerId, uint32_t tileId)
    {
        AppendHeader(artifact->commands, type, 32, layerId, tileId);
        AppendZeros(artifact->commands, 4);
    }

    void AppendRQLoad(uint32_t qparamIndex, uint32_t qparamBlock, uint32_t layerId, uint32_t tileId)
    {
        AppendHeader(artifact->commands, CommandType::RQLoad, 32, layerId, tileId);
        Append32(artifact->commands, qparamIndex);
        Append32(artifact->commands, 32);
        Append32(artifact->commands, qparamBlock);
        Append32(artifact->commands, 0);
        ++artifact->commandCount;
    }

    bool AppendDMA2D(const RefV1 &source, const RefV1 &destination, uint32_t length,
        uint32_t sourceStride, uint32_t destinationStride, uint32_t repetitions,
        uint32_t layerId, uint32_t tileId, std::string &error)
    {
        DMADirection direction;
        if ( !ResolveDMADirection(source, destination, direction) )
            return SetError(error, "Neural-AI DMA2D requires a local source or destination");
        AppendHeader(artifact->commands, CommandType::DMA2D, 64, layerId, tileId);
        AppendRef(artifact->commands, source);
        AppendRef(artifact->commands, destination);
        Append32(artifact->commands, length);
        Append32(artifact->commands, sourceStride);
        Append32(artifact->commands, destinationStride);
        Append32(artifact->commands, repetitions);
        Append32(artifact->commands, uint32_t(direction));
        AppendZeros(artifact->commands, 3);
        ++artifact->commandCount;
        return true;
    }

    bool AppendDMA3D(const RefV1 &source, const RefV1 &destination, uint32_t length,
        uint32_t sourceStride2, uint32_t destinationStride2, uint32_t repetitions2,
        uint32_t sourceStride3, uint32_t destinationStride3, uint32_t repetitions3,
        uint32_t layerId, uint32_t tileId, std::string &error)
    {
        DMADirection direction;
        if ( !ResolveDMADirection(source, destination, direction) )
            return SetError(error, "Neural-AI DMA3D requires a local source or destination");
        AppendHeader(artifact->commands, CommandType::DMA3D, 64, layerId, tileId);
        AppendRef(artifact->commands, source);
        AppendRef(artifact->commands, destination);
        Append32(artifact->commands, length);
        Append32(artifact->commands, sourceStride2);
        Append32(artifact->commands, destinationStride2);
        Append32(artifact->commands, repetitions2);
        Append32(artifact->commands, sourceStride3);
        Append32(artifact->commands, destinationStride3);
        Append32(artifact->commands, repetitions3);
        Append32(artifact->commands, uint32_t(direction));
        ++artifact->commandCount;
        return true;
    }

    bool AppendDMA1D(const RefV1 &source, const RefV1 &destination, uint32_t length,
        uint32_t layerId, uint32_t tileId, std::string &error)
    {
        DMADirection direction;
        if ( !ResolveDMADirection(source, destination, direction) )
            return SetError(error, "Neural-AI DMA1D requires a local source or destination");
        AppendHeader(artifact->commands, CommandType::DMA1D, 64, layerId, tileId);
        AppendRef(artifact->commands, source);
        AppendRef(artifact->commands, destination);
        Append32(artifact->commands, length);
        Append32(artifact->commands, uint32_t(direction));
        AppendZeros(artifact->commands, 6);
        ++artifact->commandCount;
        return true;
    }

    void AppendGEMM(CommandType type, const RefV1 &weights, const RefV1 &ifm,
        const RefV1 &partial, const RefV1 &ofm, uint32_t dimM, uint32_t ofmStride,
        uint32_t qparamBlock, uint32_t layerId, uint32_t tileId)
    {
        AppendHeader(artifact->commands, type, 96, layerId, tileId);
        AppendRef(artifact->commands, weights);
        AppendRef(artifact->commands, ifm);
        AppendRef(artifact->commands, partial);
        AppendRef(artifact->commands, ofm);
        Append32(artifact->commands, dimM);
        Append32(artifact->commands, ofmStride);
        Append32(artifact->commands, 32 * 4);
        Append32(artifact->commands, qparamBlock);
        AppendZeros(artifact->commands, 8);
        ++artifact->commandCount;
    }

    void AppendPointwiseC32(const RefV1 &weights, const RefV1 &ifm,
        const RefV1 &partial, const RefV1 &ofm, uint32_t rows,
        uint32_t inputGroups, uint32_t outputGroups, uint32_t qparamBlock,
        uint32_t inputGroupStride, uint32_t outputGroupStride,
        uint32_t layerId, uint32_t tileId)
    {
        AppendHeader(artifact->commands, CommandType::PointwiseC32, 96, layerId, tileId);
        AppendRef(artifact->commands, weights);
        AppendRef(artifact->commands, ifm);
        AppendRef(artifact->commands, partial);
        AppendRef(artifact->commands, ofm);
        Append32(artifact->commands, rows);
        Append32(artifact->commands, inputGroups);
        Append32(artifact->commands, outputGroups);
        Append32(artifact->commands, qparamBlock);
        Append32(artifact->commands, inputGroupStride);
        Append32(artifact->commands, outputGroupStride);
        AppendZeros(artifact->commands, 6);
        ++artifact->commandCount;
    }

    void AppendLineBufferJob(const neuralai::LinebufferJob &job, uint32_t layerId, uint32_t tileId)
    {
        const auto &cfg = job.linebuf;
        const auto &gemm = job.gemm;
        AppendHeader(artifact->commands, CommandType::LineBufferJob, 160, layerId, tileId);
        Append32(artifact->commands, cfg.inputBase);
        Append16(artifact->commands, cfg.inputH);
        Append16(artifact->commands, cfg.inputW);
        Append16(artifact->commands, cfg.inputC);
        Append16(artifact->commands, cfg.outputW);
        Append16(artifact->commands, cfg.strideH);
        Append16(artifact->commands, cfg.strideW);
        Append16(artifact->commands, cfg.padH);
        Append16(artifact->commands, cfg.padW);
        Append32(artifact->commands, cfg.rowStrideBytes);
        Append32(artifact->commands, cfg.pixelStrideBytes);
        Append32(artifact->commands, cfg.owStepBytes);
        Append32(artifact->commands, cfg.ohStepBytes);
        Append16(artifact->commands, cfg.kernelH);
        Append16(artifact->commands, cfg.kernelW);
        Append16(artifact->commands, cfg.cBase);
        Append16(artifact->commands, cfg.laneBase);
        Append16(artifact->commands, cfg.coalesce);
        Append16(artifact->commands, cfg.kgen);
        Append16(artifact->commands, cfg.pool);
        Append16(artifact->commands, cfg.c32Fast);
        Append16(artifact->commands, cfg.depthwise);
        Append16(artifact->commands, cfg.c32GroupStationary);
        Append16(artifact->commands, cfg.blockValidBytes);
        Append16(artifact->commands, cfg.kSeedKh);
        Append16(artifact->commands, cfg.kSeedKw);
        Append16(artifact->commands, cfg.kSeedIc);
        Append32(artifact->commands, cfg.kTiles);
        Append32(artifact->commands, cfg.spatialM);
        Append32(artifact->commands, cfg.channelAddrOffset);
        Append32(artifact->commands, cfg.coalesceKBytes);
        Append32(artifact->commands, gemm.weightAddr);
        Append32(artifact->commands, gemm.ifmAddr);
        Append32(artifact->commands, gemm.psumAddr);
        Append32(artifact->commands, gemm.ofmAddr);
        Append32(artifact->commands, gemm.dimM);
        Append32(artifact->commands, gemm.accumEn);
        Append32(artifact->commands, gemm.ofmRowStrideBytes);
        Append32(artifact->commands, gemm.ofmTileCols);
        Append32(artifact->commands, gemm.psumRowStrideBytes);
        Append32(artifact->commands, job.rows);
        Append32(artifact->commands, job.kTiles);
        AppendZeros(artifact->commands, 5);
        ++artifact->commandCount;
    }

    bool AppendSlicedNHWCCopy(const SchedulerOperation *operation, std::string &error)
    {
        const SchedulerConnection *ifm = operation->IFM(0);
        const SchedulerConnection *ofm = operation->OFM();
        if ( ifm->Type() != ofm->Type() ||
             (ifm->Type() != regor::DataType::Int8 && ifm->Type() != regor::DataType::Int32) ||
             ifm->SliceShape().Elements64() != ofm->SliceShape().Elements64() )
            return SetError(error, "Neural-AI sliced NHWC copy requires equal INT8/INT32 slices");

        const Shape copyShape = ReshapeToNHWC(ofm->SliceShape());
        if ( copyShape.Batch() != 1 || copyShape.Height() <= 0 || copyShape.Width() <= 0 ||
             copyShape.Depth() <= 0 )
            return SetError(error, "Neural-AI sliced NHWC copy requires a positive batch-one rectangle");
        const uint32_t elementBytes = ofm->Type() == regor::DataType::Int32 ? 4u : 1u;
        const int64_t rowBytes64 = copyShape.Width() * int64_t(copyShape.Depth()) * elementBytes;
        if ( rowBytes64 <= 0 || rowBytes64 > std::numeric_limits<uint32_t>::max() )
            return SetError(error, "Neural-AI sliced NHWC row size is invalid");

        const auto resolve = [&](const SchedulerConnection *connection, uint32_t &offset,
                                 uint32_t &rowStride) -> bool
        {
            const Shape storageShape = connection->tensor->storageShape;
            if ( storageShape.Size() == 1 || connection->tensor->IsConstant() )
            {
                if ( connection->slice.offset && connection->slice.offset != connection->shape.WithZeros() )
                    return SetError(error, "Neural-AI linear sliced-copy source requires zero offset");
                offset = 0;
                rowStride = uint32_t(rowBytes64);
                return true;
            }
            const Shape storage = ReshapeToNHWC(storageShape);
            const Shape slice = ReshapeToNHWC(connection->SliceShape());
            const Shape sliceOffset = connection->slice.offset ?
                ReshapeToNHWC(connection->slice.offset) : storage.WithZeros();
            if ( slice.Batch() != 1 || slice.Depth() != storage.Depth() || sliceOffset.Batch() != 0 ||
                 sliceOffset.Depth() != 0 || sliceOffset.Height() < 0 || sliceOffset.Width() < 0 ||
                 sliceOffset.Height() + slice.Height() > storage.Height() ||
                 sliceOffset.Width() + slice.Width() > storage.Width() ||
                 (connection->slice.stride && connection->slice.stride != connection->shape.WithOnes()) )
                return SetError(error, fmt::format(
                    "Neural-AI sliced NHWC copy requires an unstrided full-depth rectangle "
                    "('{}' storage [{}], slice [{}], offset [{}], stride [{}])",
                    connection->tensor->Name(), storage.ToString(), slice.ToString(),
                    sliceOffset.ToString(), connection->slice.stride.ToString()));
            const int64_t stride64 = storage.Width() * int64_t(storage.Depth()) * elementBytes;
            const int64_t offset64 =
                (sliceOffset.Height() * int64_t(storage.Width()) + sliceOffset.Width()) *
                storage.Depth() * elementBytes;
            if ( stride64 <= 0 || stride64 > std::numeric_limits<uint32_t>::max() || offset64 < 0 ||
                 offset64 > std::numeric_limits<uint32_t>::max() )
                return SetError(error, "Neural-AI sliced NHWC address is outside the ABI range");
            offset = uint32_t(offset64);
            rowStride = uint32_t(stride64);
            return true;
        };

        uint32_t sourceOffset = 0;
        uint32_t destinationOffset = 0;
        uint32_t sourceStride = 0;
        uint32_t destinationStride = 0;
        if ( !resolve(ifm, sourceOffset, sourceStride) ||
             !resolve(ofm, destinationOffset, destinationStride) ) return false;
        RefV1 source = TensorRef(ifm->tensor.get(), sourceOffset, error);
        if ( !error.empty() ) return false;
        RefV1 destination = TensorRef(ofm->tensor.get(), destinationOffset, error);
        if ( !error.empty() ) return false;
        const uint32_t rows = uint32_t(copyShape.Height());
        const uint32_t rowBytes = uint32_t(rowBytes64);
        const uint32_t layerId = uint32_t(operation->Index());
        if ( rows == 1 || (sourceStride == rowBytes && destinationStride == rowBytes) )
        {
            if ( rows > std::numeric_limits<uint32_t>::max() / rowBytes )
                return SetError(error, "Neural-AI sliced NHWC copy size overflows the ABI");
            return AppendDMA1D(source, destination, rows * rowBytes, layerId, 0, error);
        }
        return AppendDMA2D(source, destination, rowBytes, sourceStride,
            destinationStride, rows, layerId, 0, error);
    }

    bool AppendSlicedC32Copy(const SchedulerOperation *operation, std::string &error)
    {
        const SchedulerConnection *ifm = operation->IFM(0);
        const SchedulerConnection *ofm = operation->OFM();
        const bool toC32 = ofm->tensor->format == TensorFormat::C32Blocked &&
            (ifm->tensor->format == TensorFormat::C32Blocked ||
                IsCompactNHWC(ifm->tensor->format) || ifm->tensor->IsConstant());
        const bool fromC32 = ifm->tensor->format == TensorFormat::C32Blocked &&
            (ofm->tensor->format == TensorFormat::C32Blocked ||
                IsCompactNHWC(ofm->tensor->format));
        if ( ifm->Type() != regor::DataType::Int8 || ofm->Type() != regor::DataType::Int8 ||
             ifm->SliceShape().Elements64() != ofm->SliceShape().Elements64() ||
             (!toC32 && !fromC32) )
            return SetError(error,
                "Neural-AI sliced C32 copy requires equal INT8 C32/NHWC slices or a constant fill source");

        const Shape copyShape = ReshapeToNHWC(ofm->SliceShape());
        if ( copyShape.Batch() != 1 || copyShape.Height() <= 0 || copyShape.Width() <= 0 ||
             copyShape.Depth() <= 0 || (copyShape.Depth() % 32 != 0 && copyShape.Depth() != 16) )
            return SetError(error,
                "Neural-AI sliced C32 copy requires a positive batch-one rectangle with C%32=0 or C=16");

        const auto checkedU32 = [&](uint64_t value, const char *message, uint32_t &result) -> bool
        {
            if ( value > std::numeric_limits<uint32_t>::max() )
                return SetError(error, message);
            result = uint32_t(value);
            return true;
        };
        const auto checkedMulU32 = [&](uint64_t lhs, uint64_t rhs, const char *message,
                                       uint32_t &result) -> bool
        {
            if ( lhs != 0 && rhs > std::numeric_limits<uint32_t>::max() / lhs )
                return SetError(error, message);
            result = uint32_t(lhs * rhs);
            return true;
        };

        if ( copyShape.Depth() == 16 )
        {
            struct TailAddressing
            {
                uint32_t base = 0;
                uint32_t pixelStride = 0;
                uint32_t rowStride = 0;
            };
            const auto resolveTail = [&](const SchedulerConnection *connection,
                                         TailAddressing &addressing) -> bool
            {
                const Shape storage = ReshapeToNHWC(connection->tensor->storageShape);
                const Shape slice = ReshapeToNHWC(connection->SliceShape());
                const Shape sliceOffset = connection->slice.offset ?
                    ReshapeToNHWC(connection->slice.offset) : storage.WithZeros();
                const bool compact = IsCompactNHWC(connection->tensor->format) ||
                    (connection->tensor->IsConstant() &&
                        connection->tensor->format != TensorFormat::C32Blocked);
                const int pixelStride = compact ? 16 : 32;
                const bool linearConstant = connection->tensor->IsConstant() &&
                    connection->tensor->format != TensorFormat::C32Blocked;
                if ( linearConstant )
                {
                    if ( (connection->slice.offset &&
                             connection->slice.offset != connection->shape.WithZeros()) ||
                         connection->slice.stride || storage.Elements64() != slice.Elements64() )
                        return SetError(error,
                            "Neural-AI linear C16 padding source requires one complete unstrided slice");
                    addressing.pixelStride = 16;
                    addressing.rowStride = uint32_t(slice.Width() * 16);
                    return true;
                }
                const bool compactC16 = storage.Depth() == 16 && sliceOffset.Depth() == 0;
                const bool c16FromC32 = connection->tensor->format == TensorFormat::C32Blocked &&
                    storage.Depth() == 32 &&
                    (sliceOffset.Depth() == 0 || sliceOffset.Depth() == 16);
                if ( storage.Batch() != 1 || (!compactC16 && !c16FromC32) ||
                     slice.Batch() != 1 || slice.Depth() != 16 ||
                     sliceOffset.Batch() != 0 ||
                     sliceOffset.Height() < 0 || sliceOffset.Width() < 0 ||
                     sliceOffset.Height() + slice.Height() > storage.Height() ||
                     sliceOffset.Width() + slice.Width() > storage.Width() ||
                     (connection->slice.stride && connection->slice.stride != connection->shape.WithOnes()) )
                    return SetError(error, fmt::format(
                        "Neural-AI sliced C16-to-C32 copy requires an unstrided full-depth rectangle "
                        "('{}' {} storage [{}], slice [{}], offset [{}], stride [{}])",
                        connection->tensor->Name(), EnumToString(connection->tensor->format),
                        storage.ToString(), slice.ToString(), sliceOffset.ToString(),
                        connection->slice.stride.ToString()));
                const int64_t rowStride64 = storage.Width() * int64_t(pixelStride);
                const int64_t base64 =
                    (sliceOffset.Height() * int64_t(storage.Width()) + sliceOffset.Width()) * pixelStride +
                    sliceOffset.Depth();
                if ( rowStride64 <= 0 || rowStride64 > std::numeric_limits<uint32_t>::max() ||
                     base64 < 0 || base64 > std::numeric_limits<uint32_t>::max() )
                    return SetError(error, "Neural-AI sliced C16 address is outside the ABI range");
                addressing.base = uint32_t(base64);
                addressing.pixelStride = uint32_t(pixelStride);
                addressing.rowStride = uint32_t(rowStride64);
                return true;
            };

            TailAddressing sourceAddressing;
            TailAddressing destinationAddressing;
            if ( !resolveTail(ifm, sourceAddressing) || !resolveTail(ofm, destinationAddressing) ) return false;
            RefV1 source = TensorRef(ifm->tensor.get(), sourceAddressing.base, error);
            if ( !error.empty() ) return false;
            RefV1 destination = TensorRef(ofm->tensor.get(), destinationAddressing.base, error);
            if ( !error.empty() ) return false;
            return AppendDMA3D(source, destination, 16,
                sourceAddressing.pixelStride, destinationAddressing.pixelStride,
                uint32_t(copyShape.Width()), sourceAddressing.rowStride,
                destinationAddressing.rowStride, uint32_t(copyShape.Height()),
                uint32_t(operation->Index()), 0, error);
        }

        struct Addressing
        {
            uint32_t base = 0;
            uint32_t rowStride = 0;
            uint32_t groupStride = 0;
            uint32_t pixelStride = 0;
        };
        const auto resolve = [&](const SchedulerConnection *connection, Addressing &addressing) -> bool
        {
            const bool linearConstant = connection->tensor->IsConstant() &&
                connection->tensor->format != TensorFormat::C32Blocked;
            if ( linearConstant )
            {
                if ( connection->slice.offset && connection->slice.offset != connection->shape.WithZeros() )
                    return SetError(error,
                        "Neural-AI linear sliced C32 fill source requires zero offset");
                if ( !checkedMulU32(uint64_t(copyShape.Width()), 32,
                        "Neural-AI linear C32 row stride is outside the ABI range",
                        addressing.rowStride) ||
                     !checkedMulU32(uint64_t(copyShape.Height()), addressing.rowStride,
                        "Neural-AI linear C32 group stride is outside the ABI range",
                        addressing.groupStride) )
                    return false;
                return true;
            }

            const Shape storage = ReshapeToNHWC(connection->tensor->storageShape);
            const Shape slice = ReshapeToNHWC(connection->SliceShape());
            const Shape sliceOffset = connection->slice.offset ?
                ReshapeToNHWC(connection->slice.offset) : storage.WithZeros();
            if ( IsCompactNHWC(connection->tensor->format) )
            {
                if ( storage.Batch() != 1 || storage.Depth() < 32 || storage.Depth() % 32 != 0 ||
                     slice.Batch() != 1 || slice.Depth() <= 0 || slice.Depth() % 32 != 0 ||
                     sliceOffset.Batch() != 0 || sliceOffset.Depth() < 0 ||
                     sliceOffset.Depth() % 32 != 0 ||
                     int64_t(sliceOffset.Depth()) + slice.Depth() > storage.Depth() ||
                     sliceOffset.Height() < 0 || sliceOffset.Width() < 0 ||
                     int64_t(sliceOffset.Height()) + slice.Height() > storage.Height() ||
                     int64_t(sliceOffset.Width()) + slice.Width() > storage.Width() ||
                     (connection->slice.stride && connection->slice.stride != connection->shape.WithOnes()) )
                    return SetError(error, fmt::format(
                        "Neural-AI sliced NHWC-to-C32 copy requires an unstrided C32-aligned rectangle "
                        "('{}' storage [{}], slice [{}], offset [{}], stride [{}])",
                        connection->tensor->Name(), storage.ToString(), slice.ToString(),
                        sliceOffset.ToString(), connection->slice.stride.ToString()));
                if ( !checkedMulU32(uint64_t(storage.Width()), uint64_t(storage.Depth()),
                        "Neural-AI sliced NHWC row stride is outside the ABI range",
                        addressing.rowStride) )
                    return false;
                const uint64_t pixelIndex = uint64_t(sliceOffset.Height()) *
                    uint64_t(storage.Width()) + uint64_t(sliceOffset.Width());
                if ( pixelIndex > std::numeric_limits<uint32_t>::max() /
                        uint64_t(storage.Depth()) )
                    return SetError(error, "Neural-AI sliced NHWC address is outside the ABI range");
                const uint64_t base64 = pixelIndex * uint64_t(storage.Depth()) +
                    uint64_t(sliceOffset.Depth());
                if ( !checkedU32(base64, "Neural-AI sliced NHWC address is outside the ABI range",
                        addressing.base) )
                    return false;
                // Keep the existing DMA2D path for a single C32 public tensor.
                // For wider compact tensors, one C32 group is a strided
                // rectangle in NHWC storage: channels are contiguous within a
                // pixel, while pixels and rows have independent strides.  The
                // command generator emits one DMA3D per group below so both
                // strides are represented by the ABI without staging a row.
                if ( storage.Depth() == 32 )
                {
                    if ( !checkedMulU32(uint64_t(storage.Height()), addressing.rowStride,
                            "Neural-AI sliced NHWC group stride is outside the ABI range",
                            addressing.groupStride) )
                        return false;
                }
                else
                {
                    addressing.groupStride = 32;
                    addressing.pixelStride = uint32_t(storage.Depth());
                }
                return true;
            }
            if ( connection->tensor->format != TensorFormat::C32Blocked || storage.Batch() != 1 ||
                 storage.Depth() % 32 != 0 || slice.Batch() != 1 ||
                 slice.Depth() <= 0 || slice.Depth() % 32 != 0 ||
                 sliceOffset.Batch() != 0 || sliceOffset.Depth() < 0 ||
                 int64_t(sliceOffset.Depth()) + slice.Depth() > storage.Depth() ||
                 sliceOffset.Depth() % 32 != 0 || sliceOffset.Height() < 0 || sliceOffset.Width() < 0 ||
                 int64_t(sliceOffset.Height()) + slice.Height() > storage.Height() ||
                 int64_t(sliceOffset.Width()) + slice.Width() > storage.Width() ||
                 (connection->slice.stride && connection->slice.stride != connection->shape.WithOnes()) )
                return SetError(error, fmt::format(
                    "Neural-AI sliced C32 copy requires an unstrided C32-aligned rectangle "
                    "('{}' storage [{}], slice [{}], offset [{}], stride [{}])",
                    connection->tensor->Name(), storage.ToString(), slice.ToString(),
                    sliceOffset.ToString(), connection->slice.stride.ToString()));
            if ( !checkedMulU32(uint64_t(storage.Width()), 32,
                    "Neural-AI sliced C32 row stride is outside the ABI range",
                    addressing.rowStride) ||
                 !checkedMulU32(uint64_t(storage.Height()), addressing.rowStride,
                    "Neural-AI sliced C32 group stride is outside the ABI range",
                    addressing.groupStride) )
                return false;
            const uint64_t pixelIndex = uint64_t(sliceOffset.Height()) *
                uint64_t(storage.Width()) + uint64_t(sliceOffset.Width());
            if ( pixelIndex > std::numeric_limits<uint32_t>::max() / 32u )
                return SetError(error, "Neural-AI sliced C32 address is outside the ABI range");
            const uint64_t basePixel = pixelIndex * 32u;
            const uint64_t groupIndex = uint64_t(sliceOffset.Depth() / 32);
            if ( groupIndex > std::numeric_limits<uint32_t>::max() /
                    uint64_t(addressing.groupStride) )
                return SetError(error, "Neural-AI sliced C32 address is outside the ABI range");
            const uint64_t base64 = basePixel + groupIndex * addressing.groupStride;
            if ( !checkedU32(base64, "Neural-AI sliced C32 address is outside the ABI range",
                    addressing.base) )
                return false;
            return true;
        };

        Addressing sourceAddressing;
        Addressing destinationAddressing;
        if ( !resolve(ifm, sourceAddressing) || !resolve(ofm, destinationAddressing) ) return false;
        const uint32_t rows = uint32_t(copyShape.Height());
        uint32_t rowBytes = 0;
        if ( !checkedMulU32(uint64_t(copyShape.Width()), 32,
                "Neural-AI sliced C32 row size is outside the ABI range", rowBytes) )
            return false;
        const uint32_t groups = uint32_t(copyShape.Depth() / 32);
        const uint32_t layerId = uint32_t(operation->Index());
        for ( uint32_t group = 0; group < groups; ++group )
        {
            const uint64_t sourceGroupOffset = uint64_t(sourceAddressing.base) +
                uint64_t(group) * sourceAddressing.groupStride;
            const uint64_t destinationGroupOffset = uint64_t(destinationAddressing.base) +
                uint64_t(group) * destinationAddressing.groupStride;
            uint32_t sourceGroupBase = 0;
            uint32_t destinationGroupBase = 0;
            if ( !checkedU32(sourceGroupOffset,
                    "Neural-AI sliced C32 source address is outside the ABI range", sourceGroupBase) ||
                 !checkedU32(destinationGroupOffset,
                    "Neural-AI sliced C32 destination address is outside the ABI range",
                    destinationGroupBase) )
                return false;
            RefV1 source = TensorRef(ifm->tensor.get(),
                sourceGroupBase, error);
            if ( !error.empty() ) return false;
            RefV1 destination = TensorRef(ofm->tensor.get(),
                destinationGroupBase, error);
            if ( !error.empty() ) return false;
            if ( sourceAddressing.pixelStride != 0 || destinationAddressing.pixelStride != 0 )
            {
                if ( !AppendDMA3D(source, destination, 32,
                        sourceAddressing.pixelStride != 0 ? sourceAddressing.pixelStride : 32,
                        destinationAddressing.pixelStride != 0 ? destinationAddressing.pixelStride : 32,
                        uint32_t(copyShape.Width()),
                        sourceAddressing.rowStride, destinationAddressing.rowStride, rows,
                        layerId, group, error) )
                    return false;
            }
            else if ( rows == 1 || (sourceAddressing.rowStride == rowBytes &&
                                  destinationAddressing.rowStride == rowBytes) )
            {
                uint32_t transferBytes = 0;
                if ( !checkedMulU32(rows, rowBytes,
                        "Neural-AI sliced C32 copy size overflows the ABI", transferBytes) )
                    return false;
                if ( !AppendDMA1D(source, destination, transferBytes, layerId, group, error) )
                    return false;
            }
            else if ( !AppendDMA2D(source, destination, rowBytes, sourceAddressing.rowStride,
                          destinationAddressing.rowStride, rows, layerId, group, error) )
                return false;
        }
        return true;
    }

    bool AppendCopy(const SchedulerOperation *operation, std::string &error)
    {
        const SchedulerConnection *ifm = operation->IFM(0);
        const SchedulerConnection *ofm = operation->OFM();
        if ( IsCompactNHWC(ifm->tensor->format) && IsCompactNHWC(ofm->tensor->format) )
        {
            if ( !IsFullTensorConnection(ifm) || !IsFullTensorConnection(ofm) )
                return AppendSlicedNHWCCopy(operation, error);
            if ( ifm->Type() != ofm->Type() ||
                 ifm->shape.Elements64() != ofm->shape.Elements64() )
                return SetError(error, "Neural-AI compact copy requires equal element counts and types");
            const uint16_t dataType = ABIDataType(ofm->Type());
            if ( dataType == 0 )
                return SetError(error, "Neural-AI compact copy supports only INT8 and INT32");
            const uint32_t elementBytes = dataType == uint16_t(DataType::Int32) ? 4u : 1u;
            const int64_t bytes = ofm->shape.Elements64() * elementBytes;
            if ( bytes <= 0 || bytes > std::numeric_limits<uint32_t>::max() )
                return SetError(error, "Neural-AI compact copy size is invalid");
            RefV1 source = TensorRef(ifm->tensor.get(), 0, error);
            if ( !error.empty() ) return false;
            RefV1 destination = TensorRef(ofm->tensor.get(), 0, error);
            if ( !error.empty() ) return false;
            const uint32_t layerId = uint32_t(operation->Index());
            if ( IsLocalDMARegion(source.region) || IsLocalDMARegion(destination.region) )
                return AppendDMA1D(source, destination, uint32_t(bytes), layerId, 0, error);
            RefV1 bounce{};
            bounce.region = uint16_t(Region::TCDMScratch);
            bounce.offset = stageOffset;
            if ( !AppendDMA1D(source, bounce, uint32_t(bytes), layerId, 0, error) ||
                 !AppendDMA1D(bounce, destination, uint32_t(bytes), layerId, 1, error) )
                return false;
            return true;
        }
        const bool slicedC32Boundary =
            (ofm->tensor->format == TensorFormat::C32Blocked &&
                (ifm->tensor->format == TensorFormat::C32Blocked ||
                    IsCompactNHWC(ifm->tensor->format) || ifm->tensor->IsConstant())) ||
            (ifm->tensor->format == TensorFormat::C32Blocked &&
                IsCompactNHWC(ofm->tensor->format));
        if ( (!IsFullTensorConnection(ifm) || !IsFullTensorConnection(ofm)) && slicedC32Boundary )
            return AppendSlicedC32Copy(operation, error);
        if ( !IsFullTensorConnection(ifm) || !IsFullTensorConnection(ofm) )
            return SetError(error, fmt::format(
                "Neural-AI sliced MemoryCopy requires compact NHWC or supported C32 tensors "
                "('{}' {} -> '{}' {})",
                ifm->tensor->Name(), EnumToString(ifm->tensor->format),
                ofm->tensor->Name(), EnumToString(ofm->tensor->format)));
        CopyLayoutMode mode;
        if ( IsCompactNHWC(ifm->tensor->format) && ofm->tensor->format == TensorFormat::Row32 )
            mode = CopyLayoutMode::NHWCToRow32;
        else if ( ifm->tensor->format == TensorFormat::Row32 && IsCompactNHWC(ofm->tensor->format) )
            mode = CopyLayoutMode::Row32ToNHWC;
        else if ( IsCompactNHWC(ifm->tensor->format) && ofm->tensor->format == TensorFormat::C32Blocked )
            mode = CopyLayoutMode::NHWCToC32;
        else if ( ifm->tensor->format == TensorFormat::C32Blocked && IsCompactNHWC(ofm->tensor->format) )
            mode = CopyLayoutMode::C32ToNHWC;
        else return SetError(error, "Neural-AI MemoryCopy requires an NHWC/ROW32/C32 boundary");

        const auto dimensions = Dimensions(ofm->shape);
        const uint16_t dataType = ABIDataType(ofm->Type());
        if ( dataType == 0 ) return SetError(error, "Neural-AI COPY_LAYOUT supports only INT8 and INT32");
        RefV1 source = TensorRef(ifm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        RefV1 destination = TensorRef(ofm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;

        AppendHeader(artifact->commands, CommandType::CopyLayout, 96, uint32_t(operation->Index()), 0);
        AppendRef(artifact->commands, source);
        AppendRef(artifact->commands, destination);
        Append16(artifact->commands, uint16_t(mode));
        Append16(artifact->commands, ABILayout(ifm->tensor->format));
        Append16(artifact->commands, ABILayout(ofm->tensor->format));
        Append16(artifact->commands, dataType);
        for ( uint32_t dimension : dimensions ) Append32(artifact->commands, dimension);
        Append32(artifact->commands, dimensions[3]);
        const uint32_t elementBytes = dataType == uint16_t(DataType::Int32) ? 4 : 1;
        const uint32_t compactStride = dimensions[3] * elementBytes;
        const uint32_t nativeStride = uint32_t(RoundAway(int(dimensions[3]), 32)) * elementBytes;
        const bool toNative = mode == CopyLayoutMode::NHWCToRow32 || mode == CopyLayoutMode::NHWCToC32;
        Append32(artifact->commands, toNative ? compactStride : nativeStride);
        Append32(artifact->commands, toNative ? nativeStride : compactStride);
        AppendZeros(artifact->commands, 7);
        ++artifact->commandCount;
        return true;
    }

    bool AppendDepthwise(const SchedulerOperation *operation, std::string &error)
    {
        const SchedulerOpInfo *cost = schedule->Cost(operation);
        if ( cost == nullptr || cost->Config() == nullptr || cost->npuWeightsTensor == nullptr )
            return SetError(error, "Neural-AI depthwise operation has no encoded constant tensor");
        const auto *config = static_cast<const NeuralAIOpConfig *>(cost->Config());
        if ( !IsDepthwiseMode(config->Mode()) )
            return SetError(error, "Neural-AI depthwise operation has no validated depthwise mode");
        const NpuWeightTensor *encoded = cost->npuWeightsTensor.get();
        if ( encoded->encodedRanges.size() != 1 || !encoded->bufferView.HasBuffer() )
            return SetError(error, "Neural-AI depthwise operation requires one encoded constant range");
        const WeightRange &range = encoded->encodedRanges.begin()->second;
        const SchedulerConnection *ifm = operation->IFM(0);
        const SchedulerConnection *ofm = operation->OFM();
        const uint32_t channels = uint32_t(ifm->shape.Depth());
        const uint32_t groups = (channels + 31u) / 32u;
        const uint32_t paddedChannels = groups * 32u;
        if ( channels == 0 || ifm->shape.Height() == 0 || ifm->shape.Width() == 0 ||
             ofm->shape.Height() == 0 || ofm->shape.Width() == 0 ||
             range.scaleBytes != int(paddedChannels * sizeof(neuralai::QParamV1)) ||
             range.weightBytes != int(groups * 3u * 3u * 32u) )
            return SetError(error, "Neural-AI depthwise constants do not match C32 group dimensions");
        const Kernel *kernel = operation->Kernel();
        if ( kernel == nullptr || kernel->Size() != Point2i(3, 3) || kernel->Dilation() != Point2i(1, 1) ||
             kernel->Padding().Top() < 0 || kernel->Padding().Top() > 1 ||
             kernel->Padding().Left() < 0 || kernel->Padding().Left() > 1 ||
             kernel->Padding().Bottom() < 0 || kernel->Padding().Bottom() > 1 ||
             kernel->Padding().Right() < 0 || kernel->Padding().Right() > 1 ||
             (kernel->Stride() != Point2i(1, 1) && kernel->Stride() != Point2i(2, 2)) )
            return SetError(
                error, "Neural-AI depthwise requires K3 SAME with each padding side P0/P1, S1/S2, D1");

        const uint8_t *data = encoded->bufferView.RawData<uint8_t>() + range.offset;
        const uint32_t qparamBase = uint32_t(artifact->qparams.size());
        for ( uint32_t channel = 0; channel < paddedChannels; ++channel )
        {
            const uint8_t *source = data + channel * sizeof(neuralai::QParamV1);
            neuralai::QParamV1 qparam{};
            qparam.bias = int32_t(Read32(source));
            qparam.multiplier = int32_t(Read32(source + 4));
            qparam.shift = Read32(source + 8);
            qparam.zeroPoint = int32_t(Read32(source + 12));
            qparam.clampMin = int32_t(Read32(source + 16));
            qparam.clampMax = int32_t(Read32(source + 20));
            artifact->qparams.push_back(qparam);
        }
        const uint32_t weightBase = uint32_t(artifact->constants.size());
        const uint8_t *weightData = data + range.weightOffset;
        artifact->constants.insert(artifact->constants.end(), weightData, weightData + range.weightBytes);

        const uint32_t inputPixels = uint32_t(ifm->shape.Height() * ifm->shape.Width());
        const uint32_t outputPixels = uint32_t(ofm->shape.Height() * ofm->shape.Width());
        for ( uint32_t group = 0; group < groups; ++group )
        {
            const uint32_t validChannels = std::min(32u, channels - group * 32u);
            AppendRQLoad(qparamBase + group * 32u, group, uint32_t(operation->Index()), group * 2u);
            RefV1 weights{};
            weights.region = uint16_t(Region::ModelConstants);
            weights.offset = weightBase + group * 3u * 3u * 32u;
            RefV1 ifmRef = TensorRef(ifm->tensor.get(), group * inputPixels * 32u, error);
            if ( !error.empty() ) return false;
            RefV1 ofmRef = TensorRef(ofm->tensor.get(), group * outputPixels * 32u, error);
            if ( !error.empty() ) return false;
            AppendHeader(artifact->commands, CommandType::DepthwiseC32, 96,
                uint32_t(operation->Index()), group * 2u + 1u);
            AppendRef(artifact->commands, weights);
            AppendRef(artifact->commands, ifmRef);
            AppendRef(artifact->commands, ofmRef);
            Append32(artifact->commands, uint32_t(ifm->shape.Height()));
            Append32(artifact->commands, uint32_t(ifm->shape.Width()));
            Append32(artifact->commands, uint32_t(ofm->shape.Height()));
            Append32(artifact->commands, uint32_t(ofm->shape.Width()));
            Append32(artifact->commands, validChannels);
            Append32(artifact->commands, uint32_t(kernel->Stride().y));
            Append32(artifact->commands, uint32_t(kernel->Stride().x));
            Append32(artifact->commands, uint32_t(kernel->Padding().Top()));
            Append32(artifact->commands, uint32_t(kernel->Padding().Left()));
            Append32(artifact->commands, group);
            AppendZeros(artifact->commands, 4);
            ++artifact->commandCount;
        }
        return true;
    }

    bool AppendAdd(const SchedulerOperation *operation, std::string &error)
    {
        const bool subtract = operation->Type() == OpType::Sub;
        const SchedulerOpInfo *cost = schedule->Cost(operation);
        if ( cost == nullptr || cost->Config() == nullptr )
            return SetError(error, "Neural-AI Add operation has no validated target mode");
        const auto *config = static_cast<const NeuralAIOpConfig *>(cost->Config());
        if ( config->Mode() != NeuralAIOpMode::AddI8 )
            return SetError(error, "Neural-AI Add operation has no validated INT8 mode");
        const SchedulerConnection *lhs = operation->IFM(0);
        const SchedulerConnection *rhs = operation->IFM(1);
        const SchedulerConnection *ofm = operation->OFM();
        const Shape lhsShape = lhs != nullptr ? lhs->SliceShape() : Shape();
        const Shape rhsShape = rhs != nullptr ? rhs->SliceShape() : Shape();
        const Shape ofmShape = ofm != nullptr ? ofm->SliceShape() : Shape();
        const bool c32 = lhs != nullptr && rhs != nullptr && ofm != nullptr &&
            lhs->tensor->format == TensorFormat::C32Blocked &&
            rhs->tensor->format == TensorFormat::C32Blocked &&
            ofm->tensor->format == TensorFormat::C32Blocked;
        const bool structuralCompactCsp = lhs != nullptr && rhs != nullptr && ofm != nullptr &&
            lhsShape.Size() == 4 && lhsShape.Batch() == 1 &&
            lhsShape.Height() > 0 && lhsShape.Width() > 0 && lhsShape.Depth() == 16 &&
            rhsShape == lhsShape && ofmShape == lhsShape &&
            IsCompactNHWC(lhs->tensor->format) &&
            IsCompactNHWC(rhs->tensor->format) && IsCompactNHWC(ofm->tensor->format);
        const bool compactPlane = lhs != nullptr && rhs != nullptr && ofm != nullptr &&
            ((lhsShape.Size() == 3 && lhsShape[0] == 1 &&
                 lhsShape[1] > 0 && lhsShape[1] <= 4 && lhsShape[2] > 0) ||
                (lhsShape.Size() == 4 && lhsShape.Batch() == 1 && lhsShape.Height() == 1 &&
                    lhsShape.Width() > 0 && lhsShape.Width() <= 4 && lhsShape.Depth() > 0)) &&
            rhsShape == lhsShape && ofmShape == lhsShape &&
            IsCompactNHWC(lhs->tensor->format) &&
            IsCompactNHWC(rhs->tensor->format) && IsCompactNHWC(ofm->tensor->format);
        if ( lhs == nullptr || rhs == nullptr || ofm == nullptr ||
             (!c32 && !compactPlane && (!IsFullTensorConnection(lhs) || !IsFullTensorConnection(rhs) ||
                 !IsFullTensorConnection(ofm))) || lhs->Type() != regor::DataType::Int8 ||
             rhs->Type() != regor::DataType::Int8 || ofm->Type() != regor::DataType::Int8 ||
             (!structuralCompactCsp && !compactPlane && !c32) ||
             lhsShape != rhsShape || lhsShape != ofmShape ||
             lhsShape.Depth() <= 0 || lhsShape.Elements64() <= 0 )
            return SetError(error, fmt::format(
                "Neural-AI AFU Add operation {} requires equal C32 or selected compact tensors "
                "(lhs [{}] {}, rhs [{}] {}, ofm [{}] {})",
                operation->Index(), lhs == nullptr ? "missing" : lhs->shape.ToString(),
                lhs == nullptr ? "missing" : EnumToString(lhs->tensor->format),
                rhs == nullptr ? "missing" : rhs->shape.ToString(),
                rhs == nullptr ? "missing" : EnumToString(rhs->tensor->format),
                ofm == nullptr ? "missing" : ofm->shape.ToString(),
                ofm == nullptr ? "missing" : EnumToString(ofm->tensor->format)));
        const int64_t pixels = lhsShape.Elements64() / lhsShape.Depth();
        const int64_t lhsBytes = structuralCompactCsp || compactPlane ? lhsShape.Elements64() :
            pixels * RoundAway(lhsShape.Depth(), 32);
        uint32_t lhsOffset = 0;
        uint32_t rhsOffset = 0;
        uint32_t ofmOffset = 0;
        if ( compactPlane && (!CompactPlaneSliceByteOffset(lhs, lhsOffset, error) ||
                                !CompactPlaneSliceByteOffset(rhs, rhsOffset, error) ||
                                !CompactPlaneSliceByteOffset(ofm, ofmOffset, error)) )
            return false;
        if ( c32 && (!C32SliceByteOffset(lhs, lhsOffset, error) ||
                       !C32SliceByteOffset(rhs, rhsOffset, error) ||
                       !C32SliceByteOffset(ofm, ofmOffset, error)) )
            return false;
        if ( lhsBytes <= 0 || lhsBytes > std::numeric_limits<uint32_t>::max() ||
             uint64_t(lhsOffset) + uint64_t(lhsBytes) > uint64_t(TensorStorageBytes(lhs->tensor.get())) ||
             uint64_t(rhsOffset) + uint64_t(lhsBytes) > uint64_t(TensorStorageBytes(rhs->tensor.get())) ||
             uint64_t(ofmOffset) + uint64_t(lhsBytes) > uint64_t(TensorStorageBytes(ofm->tensor.get())) )
            return SetError(error, "Neural-AI AFU Add tensor storage size is invalid");
        RefV1 lhsRef = TensorRef(lhs->tensor.get(), lhsOffset, error);
        if ( !error.empty() ) return false;
        RefV1 rhsRef = TensorRef(rhs->tensor.get(), rhsOffset, error);
        if ( !error.empty() ) return false;
        RefV1 ofmRef = TensorRef(ofm->tensor.get(), ofmOffset, error);
        if ( !error.empty() ) return false;
        if ( lhsRef.region != uint16_t(Region::TCDMScratch) ||
             rhsRef.region != uint16_t(Region::TCDMScratch) ||
             ofmRef.region != uint16_t(Region::TCDMScratch) )
            return SetError(error, "Neural-AI Add requires internal TCDM tensors");
        const auto overlaps = [lhsBytes](const RefV1 &first, const RefV1 &second)
        {
            if ( first.region != second.region || first.index != second.index ) return false;
            return uint64_t(first.offset) < uint64_t(second.offset) + uint64_t(lhsBytes) &&
                   uint64_t(second.offset) < uint64_t(first.offset) + uint64_t(lhsBytes);
        };
        if ( overlaps(lhsRef, ofmRef) || overlaps(rhsRef, ofmRef) )
            return SetError(error, "Neural-AI Add requires out-of-place output storage");

        const auto scalarZeroPoint = [](const Quantization &quantization)
        {
            return quantization.zeroPoints.empty() ? 0 : int(quantization.zeroPoints[0]);
        };
        if ( lhs->quantization.scales.size() != 1 || rhs->quantization.scales.size() != 1 ||
             ofm->quantization.scales.size() != 1 || lhs->quantization.zeroPoints.size() > 1 ||
             rhs->quantization.zeroPoints.size() > 1 || ofm->quantization.zeroPoints.size() > 1 )
            return SetError(error, "Neural-AI Add requires scalar quantization");
        const QuantizedScale lhsScale = lhs->quantization.scales[0];
        const QuantizedScale rhsScale = rhs->quantization.scales[0];
        const QuantizedScale outputScale = ofm->quantization.scales[0];
        const int lhsZeroPoint = scalarZeroPoint(lhs->quantization);
        const int rhsZeroPoint = scalarZeroPoint(rhs->quantization);
        const int outputZeroPoint = scalarZeroPoint(ofm->quantization);
        const QuantizedScale inputIdentity(32768.0);
        const QuantizedScale outputIdentity(1.0 / 32768.0);
        const bool rawSafe = !subtract && lhsScale == inputIdentity && rhsScale == inputIdentity &&
            outputScale == outputIdentity &&
            int64_t(lhsZeroPoint) + int64_t(rhsZeroPoint) == int64_t(outputZeroPoint) &&
            (ofm->quantization.quantMin.empty() ||
                ofm->quantization.quantMin[0] <= -128) &&
            (ofm->quantization.quantMax.empty() || ofm->quantization.quantMax[0] >= 127);
        if ( rawSafe )
        {
            AppendHeader(artifact->commands, CommandType::AFUBinary, 64,
                uint32_t(operation->Index()), 0);
            AppendRef(artifact->commands, lhsRef);
            AppendRef(artifact->commands, rhsRef);
            AppendRef(artifact->commands, ofmRef);
            Append32(artifact->commands, uint32_t(lhsBytes));
            Append32(artifact->commands, uint32_t(AFUBinaryMode::AddI8));
            AppendZeros(artifact->commands, 4);
        }
        else
        {
            const auto *round = operation->Attribute<double_round_shift_attr_t>();
            const int clampMin = ofm->quantization.quantMin.empty() ? -128 :
                int(ofm->quantization.quantMin[0]);
            const int clampMax = ofm->quantization.quantMax.empty() ? 127 :
                int(ofm->quantization.quantMax[0]);
            if ( round == nullptr || (round->shift != 0 && round->shift != 20) ||
                 ofm->rounding != RoundMode::DBL || lhsScale.scale <= 0 || rhsScale.scale <= 0 ||
                 outputScale.scale <= 0 || lhsScale.shift < 0 || lhsScale.shift > 63 ||
                 rhsScale.shift < 0 || rhsScale.shift > 63 || outputScale.shift < 0 ||
                 outputScale.shift > 63 || clampMin < -128 || clampMax > 127 || clampMin > clampMax )
                return SetError(error, "Neural-AI SPATZ_ADD quantization is outside the INT8 contract");
            AppendHeader(artifact->commands, CommandType::SpatzAdd, 96,
                uint32_t(operation->Index()), 0);
            AppendRef(artifact->commands, lhsRef);
            AppendRef(artifact->commands, rhsRef);
            AppendRef(artifact->commands, ofmRef);
            Append32(artifact->commands, uint32_t(lhsBytes));
            Append32(artifact->commands, uint32_t(lhsScale.scale));
            Append32(artifact->commands, uint32_t(lhsScale.shift));
            Append32(artifact->commands, uint32_t(rhsScale.scale));
            Append32(artifact->commands, uint32_t(rhsScale.shift));
            Append32(artifact->commands, uint32_t(outputScale.scale));
            Append32(artifact->commands, uint32_t(outputScale.shift));
            Append32(artifact->commands, uint32_t(lhsZeroPoint));
            Append32(artifact->commands, uint32_t(rhsZeroPoint));
            Append32(artifact->commands, uint32_t(outputZeroPoint));
            Append32(artifact->commands, uint32_t(clampMin));
            Append32(artifact->commands, uint32_t(clampMax));
            Append32(artifact->commands, uint32_t(round->shift));
            Append32(artifact->commands, uint32_t(subtract ? SpatzBinaryMode::Subtract :
                                                        SpatzBinaryMode::Add));
        }
        ++artifact->commandCount;
        return true;
    }

    bool AppendPassthroughConcat(const SchedulerOperation *operation, std::string &error)
    {
        const SchedulerConnection *ofm = operation->OFM();
        std::vector<const SchedulerConnection *> inputs;
        for ( const auto &[usage, connection] : operation->inputs.pairs() )
        {
            if ( IsIFM(usage) ) inputs.push_back(&connection);
        }
        if ( ofm == nullptr || inputs.size() < 3 || !IsFullTensorConnection(ofm) ||
             ofm->Type() != regor::DataType::Int8 ||
             ofm->tensor->format != TensorFormat::C32Blocked )
            return SetError(error, fmt::format(
                "Neural-AI Passthrough requires a selected aligned multi-input Concat "
                "(inputs={}, output='{}' [{}] {} full={})",
                inputs.size(), ofm == nullptr ? "missing" : ofm->tensor->Name(),
                ofm == nullptr ? "missing" : ofm->SliceShape().ToString(),
                ofm == nullptr ? "missing" : EnumToString(ofm->tensor->format),
                ofm != nullptr && IsFullTensorConnection(ofm)));
        const Shape inputShape = inputs.front()->SliceShape();
        const Shape outputShape = ofm->SliceShape();
        if ( inputShape.Size() != 4 || inputShape.Batch() != 1 ||
             inputShape.Height() <= 0 || inputShape.Width() <= 0 || inputShape.Depth() <= 0 ||
             inputShape.Depth() % 32 != 0 ||
             outputShape != inputShape.WithDepth(int(inputs.size()) * inputShape.Depth()) )
            return SetError(error, fmt::format(
                "Neural-AI Passthrough Concat requires equal C32 groups on the final axis "
                "(inputs={}, first=[{}], output=[{}])",
                inputs.size(), inputShape.ToString(), outputShape.ToString()));
        const int64_t inputBytes64 = inputShape.Elements64();
        const int64_t outputBytes64 = outputShape.Elements64();
        if ( inputBytes64 <= 0 || outputBytes64 != inputBytes64 * int64_t(inputs.size()) ||
             outputBytes64 > std::numeric_limits<uint32_t>::max() ||
             TensorStorageBytes(ofm->tensor.get()) != outputBytes64 )
            return SetError(error, "Neural-AI Passthrough Concat storage size is invalid");
        RefV1 destination = TensorRef(ofm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        if ( destination.region != uint16_t(Region::TCDMScratch) )
            return SetError(error, "Neural-AI Passthrough Concat requires an internal output");
        const uint32_t inputBytes = uint32_t(inputBytes64);
        const uint32_t outputBytes = uint32_t(outputBytes64);
        const uint32_t groups = uint32_t(inputShape.Depth() / 32);
        const uint32_t groupBytes = inputBytes / groups;
        const uint32_t sourceRowStride = uint32_t(
            int64_t(inputShape.Width()) * inputShape.Depth());
        const uint32_t destinationRowStride = uint32_t(
            int64_t(inputShape.Width()) * 32);
        for ( int index = 0; index < int(inputs.size()); ++index )
        {
            const SchedulerConnection *input = inputs[index];
            if ( !IsFullTensorConnection(input) || input->Type() != regor::DataType::Int8 ||
                 input->SliceShape() != inputShape ||
                 (input->tensor->format != TensorFormat::C32Blocked &&
                     !IsCompactNHWC(input->tensor->format)) ||
                 TensorStorageBytes(input->tensor.get()) != inputBytes64 )
                return SetError(error,
                    "Neural-AI Passthrough Concat inputs must be complete C32 groups");
            RefV1 source = TensorRef(input->tensor.get(), 0, error);
            if ( !error.empty() ) return false;
            RefV1 partDestination = destination;
            partDestination.offset += uint32_t(index) * inputBytes;
            const bool overlaps = source.region == destination.region &&
                source.index == destination.index &&
                uint64_t(source.offset) < uint64_t(destination.offset) + outputBytes &&
                uint64_t(destination.offset) < uint64_t(source.offset) + inputBytes;
            const bool alreadyPlaced = input->tensor->format == TensorFormat::C32Blocked &&
                source.region == partDestination.region && source.index == partDestination.index &&
                source.offset == partDestination.offset;
            if ( overlaps && !alreadyPlaced )
                return SetError(error,
                    "Neural-AI Passthrough Concat does not allow overlapping gather buffers");
            if ( alreadyPlaced ) continue;
            for ( uint32_t group = 0; group < groups; ++group )
            {
                RefV1 groupSource = source;
                RefV1 groupDestination = partDestination;
                groupDestination.offset += group * groupBytes;
                if ( input->tensor->format == TensorFormat::C32Blocked || groups == 1 )
                {
                    groupSource.offset += group * groupBytes;
                    if ( !AppendDMA1D(groupSource, groupDestination, groupBytes,
                             uint32_t(operation->Index()), uint32_t(index) * groups + group, error) )
                        return false;
                }
                else
                {
                    groupSource.offset += group * 32;
                    if ( !AppendDMA3D(groupSource, groupDestination, 32,
                             uint32_t(inputShape.Depth()), 32, uint32_t(inputShape.Width()),
                             sourceRowStride, destinationRowStride, uint32_t(inputShape.Height()),
                             uint32_t(operation->Index()), uint32_t(index) * groups + group, error) )
                        return false;
                }
            }
        }
        return true;
    }

    RefV1 LutRef(const SchedulerConnection *lut)
    {
        auto position = lutReferences.find(lut->tensor->uid);
        if ( position != lutReferences.end() ) return position->second;
        while ( artifact->constants.size() % ArchNeuralAI::DMAAlignment != 0 )
            artifact->constants.push_back(0);
        RefV1 reference{};
        reference.region = uint16_t(Region::ModelConstants);
        reference.offset = uint32_t(artifact->constants.size());
        const uint8_t *lutBytes = lut->tensor->bufferView.RawData<uint8_t>();
        // Regor LUT tensors are ordered by signed INT8 value (-128..127),
        // whereas the AFU indexes the table with the input's raw byte.
        for ( uint32_t raw = 0; raw < 256; ++raw )
        {
            const int32_t signedValue = raw < 128 ? int32_t(raw) : int32_t(raw) - 256;
            artifact->constants.push_back(lutBytes[signedValue + 128]);
        }
        lutReferences.emplace(lut->tensor->uid, reference);
        return reference;
    }

    bool AppendAFULut(const SchedulerOperation *operation, std::string &error)
    {
        const SchedulerOpInfo *cost = schedule->Cost(operation);
        if ( cost == nullptr || cost->Config() == nullptr )
            return SetError(error, "Neural-AI LUT operation has no validated target mode");
        const auto *config = static_cast<const NeuralAIOpConfig *>(cost->Config());
        if ( config->Mode() != NeuralAIOpMode::AFULutI8 )
            return SetError(error, "Neural-AI LUT operation has no validated AFU mode");
        const SchedulerConnection *ifm = operation->IFM(0);
        const SchedulerConnection *ofm = operation->OFM();
        const SchedulerConnection *lut = operation->TryInput(TensorUsage::LUT);
        const bool c32 = ifm != nullptr && ofm != nullptr &&
            ifm->tensor->format == TensorFormat::C32Blocked &&
            ofm->tensor->format == TensorFormat::C32Blocked;
        const bool compact = ifm != nullptr && ofm != nullptr &&
            IsCompactNHWC(ifm->tensor->format) && IsCompactNHWC(ofm->tensor->format);
        if ( ifm == nullptr || ofm == nullptr || lut == nullptr ||
             !IsFullTensorConnection(ofm) ||
             ifm->Type() != regor::DataType::Int8 || ofm->Type() != regor::DataType::Int8 ||
             ifm->SliceShape() != ofm->shape ||
             (!c32 && !compact) || (c32 && !IsFullTensorConnection(ifm)) ||
             !lut->tensor->IsConstant() || lut->tensor->dataType != regor::DataType::Int8 ||
             lut->tensor->bufferView.Elements() != 256 )
            return SetError(error,
                "Neural-AI AFU LUT requires equal C32 or compact INT8 views and a 256-byte LUT");
        const int64_t bytes = compact ?
            ifm->SliceShape().Elements64() : ifm->tensor->AllocationSizeBytes();
        const int64_t outputBytes = TensorStorageBytes(ofm->tensor.get());
        if ( bytes <= 0 || outputBytes < bytes || (!compact && bytes != outputBytes) ||
             bytes > std::numeric_limits<uint32_t>::max() )
            return SetError(error, fmt::format(
                "Neural-AI AFU LUT tensor storage size is invalid (view={}, output={})",
                bytes, outputBytes));
        uint32_t ifmOffset = 0;
        if ( !IsFullTensorConnection(ifm) )
        {
            if ( !IsCompactNHWC(ifm->tensor->format) )
                return SetError(error, "Neural-AI sliced AFU LUT input must use compact storage");
            const Shape storage = ReshapeToNHWC(ifm->tensor->storageShape);
            const Shape slice = ReshapeToNHWC(ifm->SliceShape());
            const Shape offset = ifm->slice.offset ?
                ReshapeToNHWC(ifm->slice.offset) : storage.WithZeros();
            const Shape stride = ifm->slice.stride ?
                ReshapeToNHWC(ifm->slice.stride) : storage.WithOnes();
            const bool contiguous = slice.Height() == 1 ||
                (slice.Width() == storage.Width() && offset.Width() == 0);
            if ( storage.Batch() != 1 || slice.Batch() != 1 ||
                 slice.Depth() != storage.Depth() || offset.Batch() != 0 ||
                 offset.Depth() != 0 || offset.Height() < 0 || offset.Width() < 0 ||
                 offset.Height() + slice.Height() > storage.Height() ||
                 offset.Width() + slice.Width() > storage.Width() ||
                 stride != stride.WithOnes() || !contiguous )
                return SetError(error,
                    "Neural-AI sliced AFU LUT requires one contiguous compact rectangle");
            const uint64_t offset64 =
                (uint64_t(offset.Height()) * uint64_t(storage.Width()) +
                    uint64_t(offset.Width())) * uint64_t(storage.Depth());
            if ( offset64 > std::numeric_limits<uint32_t>::max() )
                return SetError(error, "Neural-AI sliced AFU LUT offset overflows the ABI");
            ifmOffset = uint32_t(offset64);
        }
        RefV1 ifmRef = TensorRef(ifm->tensor.get(), ifmOffset, error);
        if ( !error.empty() ) return false;
        RefV1 ofmRef = TensorRef(ofm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        if ( ifmRef.region != uint16_t(Region::TCDMScratch) ||
             ofmRef.region != uint16_t(Region::TCDMScratch) )
            return SetError(error, "Neural-AI AFU LUT requires internal TCDM tensors");
        if ( ifmRef.offset < ofmRef.offset + uint32_t(bytes) &&
             ofmRef.offset < ifmRef.offset + uint32_t(bytes) )
            return SetError(error, "Neural-AI AFU LUT requires out-of-place output storage");

        const RefV1 lutRef = LutRef(lut);

        AppendHeader(artifact->commands, CommandType::AFULut, 64,
            uint32_t(operation->Index()), 0);
        AppendRef(artifact->commands, ifmRef);
        AppendRef(artifact->commands, ofmRef);
        AppendRef(artifact->commands, lutRef);
        Append32(artifact->commands, uint32_t(bytes));
        AppendZeros(artifact->commands, 5);
        ++artifact->commandCount;
        return true;
    }

    bool AppendAFULutStripe(const SchedulerOperation *operation, const HLCStripe *stripe,
        uint32_t &tileId, std::string &error)
    {
        const SchedulerOpInfo *cost = schedule->Cost(operation);
        const auto *config = cost != nullptr && cost->Config() != nullptr ?
            static_cast<const NeuralAIOpConfig *>(cost->Config()) : nullptr;
        const SchedulerConnection *ifm = operation->IFM(0);
        const SchedulerConnection *ofm = operation->OFM();
        const SchedulerConnection *lut = operation->TryInput(TensorUsage::LUT);
        const bool c32ToCompactC16 = ifm != nullptr && ofm != nullptr &&
            ifm->tensor->format == TensorFormat::C32Blocked &&
            IsCompactNHWC(ofm->tensor->format) && ifm->shape.Depth() == 16 &&
            ofm->shape.Depth() == 16;
        if ( config == nullptr || config->Mode() != NeuralAIOpMode::AFULutI8 ||
             stripe == nullptr || stripe->stripeAreas.size() != 1 || ifm == nullptr || ofm == nullptr ||
             lut == nullptr || ifm->Type() != regor::DataType::Int8 || ofm->Type() != regor::DataType::Int8 ||
             ifm->tensor->format != TensorFormat::C32Blocked ||
             (ofm->tensor->format != TensorFormat::C32Blocked && !c32ToCompactC16) ||
             !lut->tensor->IsConstant() ||
             lut->tensor->dataType != regor::DataType::Int8 || lut->tensor->bufferView.Elements() != 256 )
            return SetError(error, fmt::format(
                "Neural-AI striped AFU LUT operation {} is invalid (IFM {}, OFM {}, config {})",
                operation->Index(), EnumToString(ifm->tensor->format),
                EnumToString(ofm->tensor->format),
                config == nullptr ? "none" : NeuralAIOpModeName(config->Mode())));
        const StripeArea &area = stripe->stripeAreas.front();
        if ( area.ifmCount != 1 || !(area.ifmAreas[0] == area.ofmArea) )
            return SetError(error, "Neural-AI striped AFU LUT requires equal IFM and OFM areas");
        const Shape start = ReshapeToNHWC(area.ofmArea.Start());
        const Shape shape = ReshapeToNHWC(area.ofmArea.SizeShape());
        const Shape ifmStorage = ReshapeToNHWC(ifm->tensor->storageShape);
        const Shape ofmStorage = ReshapeToNHWC(ofm->tensor->storageShape);
        if ( start.Batch() != 0 || start.Width() != 0 || start.Depth() != 0 || shape.Batch() != 1 ||
             shape.Width() != ifm->shape.Width() || shape.Width() != ofm->shape.Width() ||
             shape.Depth() != ifm->shape.Depth() || shape.Depth() != ofm->shape.Depth() ||
             ifmStorage.Width() != shape.Width() || ofmStorage.Width() != shape.Width() ||
             ifmStorage.Height() <= 0 || ofmStorage.Height() <= 0 )
            return SetError(error, "Neural-AI striped AFU LUT requires a full-width, full-depth C32 area");

        RefV1 ifmBase = TensorRef(ifm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        RefV1 ofmBase = TensorRef(ofm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        if ( ifmBase.region != uint16_t(Region::TCDMScratch) ||
             ofmBase.region != uint16_t(Region::TCDMScratch) )
            return SetError(error, "Neural-AI striped AFU LUT requires internal TCDM tensors");
        const uint32_t ifmBytes = uint32_t(TensorStorageBytes(ifm->tensor.get()));
        const uint32_t ofmBytes = uint32_t(TensorStorageBytes(ofm->tensor.get()));
        const bool buffersOverlap = ifmBase.offset < ofmBase.offset + ofmBytes &&
            ofmBase.offset < ifmBase.offset + ifmBytes;
        if ( buffersOverlap && (!c32ToCompactC16 || ifmBase.offset != ofmBase.offset) )
            return SetError(error,
                "Neural-AI striped AFU LUT requires separate buffers or exact C16 compaction reuse");

        const RefV1 lutRef = LutRef(lut);
        const uint32_t rowBytes = uint32_t(shape.Width()) * 32u;
        const uint32_t ifmPlaneBytes = uint32_t(ifmStorage.Height()) * rowBytes;
        const uint32_t ofmRowBytes = uint32_t(shape.Width()) *
            (c32ToCompactC16 ? 16u : 32u);
        const uint32_t ofmPlaneBytes = uint32_t(ofmStorage.Height()) * ofmRowBytes;
        if ( c32ToCompactC16 )
        {
            int rowsDone = 0;
            while ( rowsDone < shape.Height() )
            {
                const int logicalY = start.Height() + rowsDone;
                const int ifmY = logicalY % ifmStorage.Height();
                const int ofmY = logicalY % ofmStorage.Height();
                const int rows = std::min({shape.Height() - rowsDone,
                    ifmStorage.Height() - ifmY, ofmStorage.Height() - ofmY});
                RefV1 ifmRef = ifmBase;
                RefV1 ofmRef = ofmBase;
                ifmRef.offset += uint32_t(ifmY) * rowBytes;
                ofmRef.offset += uint32_t(ofmY) * ofmRowBytes;
                if ( !AppendDMA2D(ifmRef, ofmRef, 16, 32, 16,
                         uint32_t(rows * shape.Width()), uint32_t(operation->Index()), tileId++, error) )
                    return false;
                AppendHeader(artifact->commands, CommandType::AFULut, 64,
                    uint32_t(operation->Index()), tileId++);
                AppendRef(artifact->commands, ofmRef);
                AppendRef(artifact->commands, ofmRef);
                AppendRef(artifact->commands, lutRef);
                Append32(artifact->commands, uint32_t(rows) * ofmRowBytes);
                AppendZeros(artifact->commands, 5);
                ++artifact->commandCount;
                rowsDone += rows;
            }
            return true;
        }
        const int groups = DivRoundUp(shape.Depth(), 32);
        for ( int group = 0; group < groups; ++group )
        {
            int rowsDone = 0;
            while ( rowsDone < shape.Height() )
            {
                const int logicalY = start.Height() + rowsDone;
                const int ifmY = logicalY % ifmStorage.Height();
                const int ofmY = logicalY % ofmStorage.Height();
                const int rows = std::min({shape.Height() - rowsDone,
                    ifmStorage.Height() - ifmY, ofmStorage.Height() - ofmY});
                RefV1 ifmRef = ifmBase;
                RefV1 ofmRef = ofmBase;
                ifmRef.offset += uint32_t(group) * ifmPlaneBytes + uint32_t(ifmY) * rowBytes;
                ofmRef.offset += uint32_t(group) * ofmPlaneBytes + uint32_t(ofmY) * rowBytes;
                AppendHeader(artifact->commands, CommandType::AFULut, 64,
                    uint32_t(operation->Index()), tileId++);
                AppendRef(artifact->commands, ifmRef);
                AppendRef(artifact->commands, ofmRef);
                AppendRef(artifact->commands, lutRef);
                Append32(artifact->commands, uint32_t(rows) * rowBytes);
                AppendZeros(artifact->commands, 5);
                ++artifact->commandCount;
                rowsDone += rows;
            }
        }
        return true;
    }

    bool AppendAFUGlobalAvgPool(const SchedulerOperation *operation, std::string &error)
    {
        const SchedulerOpInfo *cost = schedule->Cost(operation);
        if ( cost == nullptr || cost->Config() == nullptr )
            return SetError(error, "Neural-AI AvgPool operation has no validated target mode");
        const auto *config = static_cast<const NeuralAIOpConfig *>(cost->Config());
        if ( config->Mode() != NeuralAIOpMode::AFUGlobalAvgPoolC32 )
            return SetError(error, "Neural-AI AvgPool operation has no validated AFU global mode");
        const SchedulerConnection *ifm = operation->IFM(0);
        const SchedulerConnection *ofm = operation->OFM();
        const Kernel *kernel = operation->Kernel();
        if ( ifm == nullptr || ofm == nullptr || kernel == nullptr ||
             !IsFullTensorConnection(ifm) || !IsFullTensorConnection(ofm) ||
             ifm->tensor->format != TensorFormat::C32Blocked ||
             ofm->tensor->format != TensorFormat::C32Blocked )
            return SetError(error, "Neural-AI global AvgPool requires full C32-blocked tensors");
        const Shape inputShape = ReshapeToNHWC(ifm->shape);
        const Shape outputShape = ReshapeToNHWC(ofm->shape);
        if ( inputShape.Batch() != 1 || inputShape.Height() <= 0 || inputShape.Width() <= 0 ||
             inputShape.Depth() <= 0 || outputShape != Shape(1, 1, 1, inputShape.Depth()) ||
             kernel->Size() != Point2i(inputShape.Width(), inputShape.Height()) ||
             kernel->Stride() != Point2i(1, 1) || kernel->Dilation() != Point2i(1, 1) ||
             !kernel->Padding().IsZero() )
            return SetError(error, "Neural-AI AFU AvgPool requires a full-spatial 1x1 reduction");
        RefV1 ifmRef = TensorRef(ifm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        RefV1 ofmRef = TensorRef(ofm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        if ( ifmRef.region != uint16_t(Region::TCDMScratch) ||
             ofmRef.region != uint16_t(Region::TCDMScratch) )
            return SetError(error, "Neural-AI AFU AvgPool requires internal TCDM tensors");

        AppendHeader(artifact->commands, CommandType::AFUGlobalAvgPool, 64,
            uint32_t(operation->Index()), 0);
        AppendRef(artifact->commands, ifmRef);
        AppendRef(artifact->commands, ofmRef);
        Append32(artifact->commands, uint32_t(inputShape.Height()));
        Append32(artifact->commands, uint32_t(inputShape.Width()));
        Append32(artifact->commands, uint32_t(inputShape.Depth()));
        AppendZeros(artifact->commands, 5);
        ++artifact->commandCount;
        return true;
    }

    bool AppendUpsampleNearest(const SchedulerOperation *operation, std::string &error)
    {
        const SchedulerOpInfo *cost = schedule->Cost(operation);
        if ( cost == nullptr || cost->Config() == nullptr )
            return SetError(error, "Neural-AI upsample operation has no validated target mode");
        const auto *config = static_cast<const NeuralAIOpConfig *>(cost->Config());
        if ( config->Mode() != NeuralAIOpMode::UpsampleNearestC32 )
            return SetError(error, "Neural-AI upsample operation has no validated nearest mode");
        const SchedulerConnection *ifm = operation->IFM(0);
        const SchedulerConnection *ofm = operation->OFM();
        if ( ifm == nullptr || ofm == nullptr )
            return SetError(error, "Neural-AI upsample is missing an IFM or OFM connection");
        if ( !IsFullTensorConnection(ifm) || !IsFullTensorConnection(ofm) )
            return SetError(error, "Neural-AI upsample requires full tensor connections");
        if ( ifm->Type() != regor::DataType::Int8 || ofm->Type() != regor::DataType::Int8 )
            return SetError(error, "Neural-AI upsample requires INT8 tensors");
        if ( ifm->tensor->format != TensorFormat::C32Blocked ||
             ofm->tensor->format != TensorFormat::C32Blocked )
            return SetError(error, "Neural-AI upsample requires C32-blocked tensors");
        if ( ifm->resamplingMode != ArchResampling::Nearest )
            return SetError(error, "Neural-AI upsample requires nearest resampling metadata");
        const bool supportedDepth =
            ifm->shape.Depth() == 32 || ifm->shape.Depth() == 128 || ifm->shape.Depth() == 256;
        if ( !supportedDepth || ofm->shape.Depth() != ifm->shape.Depth() )
            return SetError(error, "Neural-AI upsample requires a selected C32-grouped depth");
        if ( ofm->shape.Height() % 2 != 0 ||
             ofm->shape.Height() / 2 != ifm->shape.Height() ||
             ofm->shape.Width() % 2 != 0 ||
             ofm->shape.Width() / 2 != ifm->shape.Width() )
            return SetError(error, "Neural-AI upsample requires an exact 2x spatial shape");
        const int64_t inputBytes = ifm->tensor->AllocationSizeBytes();
        const int64_t outputBytes = ofm->tensor->AllocationSizeBytes();
        if ( inputBytes <= 0 ||
             inputBytes > std::numeric_limits<uint32_t>::max() / 4 ||
             outputBytes != inputBytes * 4 || outputBytes > std::numeric_limits<uint32_t>::max() )
            return SetError(error, "Neural-AI upsample tensor storage size is invalid");
        RefV1 ifmRef = TensorRef(ifm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        RefV1 ofmRef = TensorRef(ofm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        if ( ifmRef.region != uint16_t(Region::TCDMScratch) ||
             ofmRef.region != uint16_t(Region::TCDMScratch) )
            return SetError(error, "Neural-AI upsample requires internal TCDM tensors");
        if ( ifmRef.offset < ofmRef.offset + uint32_t(outputBytes) &&
             ofmRef.offset < ifmRef.offset + uint32_t(inputBytes) )
            return SetError(error, "Neural-AI upsample requires out-of-place output storage");

        AppendHeader(artifact->commands, CommandType::UpsampleNearest, 64,
            uint32_t(operation->Index()), 0);
        AppendRef(artifact->commands, ifmRef);
        AppendRef(artifact->commands, ofmRef);
        Append32(artifact->commands, uint32_t(ifm->shape.Height()));
        Append32(artifact->commands, uint32_t(ifm->shape.Width()));
        Append32(artifact->commands, uint32_t(ifm->shape.Depth()));
        Append32(artifact->commands, 2);
        Append32(artifact->commands, 2);
        AppendZeros(artifact->commands, 3);
        ++artifact->commandCount;
        return true;
    }

    bool AppendMaxPool(const SchedulerOperation *operation, std::string &error)
    {
        const SchedulerOpInfo *cost = schedule->Cost(operation);
        if ( cost == nullptr || cost->Config() == nullptr )
            return SetError(error, "Neural-AI MaxPool operation has no validated target mode");
        const auto *config = static_cast<const NeuralAIOpConfig *>(cost->Config());
        if ( config->Mode() != NeuralAIOpMode::MaxPoolK5S1P2C32 )
            return SetError(error, "Neural-AI MaxPool operation has no validated C32 mode");
        const SchedulerConnection *ifm = operation->IFM(0);
        const SchedulerConnection *ofm = operation->OFM();
        const Kernel *kernel = operation->Kernel();
        if ( ifm == nullptr || ofm == nullptr || kernel == nullptr ||
             !IsFullTensorConnection(ifm) || !IsFullTensorConnection(ofm) ||
             ifm->Type() != regor::DataType::Int8 || ofm->Type() != regor::DataType::Int8 ||
             ifm->tensor->format != TensorFormat::C32Blocked ||
             ofm->tensor->format != TensorFormat::C32Blocked )
            return SetError(error, "Neural-AI MaxPool requires full C32-blocked INT8 tensors");
        const Shape inputShape = ReshapeToNHWC(ifm->shape);
        const Shape outputShape = ReshapeToNHWC(ofm->shape);
        const bool supportedDepth = inputShape.Depth() == 32 || inputShape.Depth() == 128;
        if ( inputShape.Batch() != 1 || !supportedDepth || outputShape != inputShape ||
             kernel->Size() != Point2i(5, 5) || kernel->Stride() != Point2i(1, 1) ||
             kernel->Dilation() != Point2i(1, 1) ||
             kernel->Padding().Top() != 2 || kernel->Padding().Bottom() != 2 ||
             kernel->Padding().Left() != 2 || kernel->Padding().Right() != 2 )
            return SetError(error, "Neural-AI MaxPool requires batch-1 K5/S1/P2 C32-grouped shape");
        const int64_t bytes = ifm->tensor->AllocationSizeBytes();
        if ( bytes <= 0 || bytes != ofm->tensor->AllocationSizeBytes() ||
             bytes > std::numeric_limits<uint32_t>::max() )
            return SetError(error, "Neural-AI MaxPool tensor storage size is invalid");
        RefV1 ifmRef = TensorRef(ifm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        RefV1 ofmRef = TensorRef(ofm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        if ( ifmRef.region != uint16_t(Region::TCDMScratch) ||
             ofmRef.region != uint16_t(Region::TCDMScratch) )
            return SetError(error, "Neural-AI MaxPool requires internal TCDM tensors");
        if ( ifmRef.offset < ofmRef.offset + uint32_t(bytes) &&
             ofmRef.offset < ifmRef.offset + uint32_t(bytes) )
            return SetError(error, "Neural-AI MaxPool requires out-of-place output storage");

        AppendHeader(artifact->commands, CommandType::MaxPool, 96,
            uint32_t(operation->Index()), 0);
        AppendRef(artifact->commands, ifmRef);
        AppendRef(artifact->commands, ofmRef);
        Append32(artifact->commands, uint32_t(inputShape.Height()));
        Append32(artifact->commands, uint32_t(inputShape.Width()));
        Append32(artifact->commands, uint32_t(inputShape.Depth()));
        Append32(artifact->commands, 5);
        Append32(artifact->commands, 5);
        Append32(artifact->commands, 1);
        Append32(artifact->commands, 1);
        Append32(artifact->commands, 2);
        Append32(artifact->commands, 2);
        AppendZeros(artifact->commands, 7);
        ++artifact->commandCount;
        return true;
    }

    bool AppendConcat(const SchedulerOperation *operation, std::string &error)
    {
        const SchedulerOpInfo *cost = schedule->Cost(operation);
        if ( cost == nullptr || cost->Config() == nullptr )
            return SetError(error, "Neural-AI Concat operation has no validated target mode");
        const auto *config = static_cast<const NeuralAIOpConfig *>(cost->Config());
        if ( config->Mode() != NeuralAIOpMode::ConcatC32 )
            return SetError(error, "Neural-AI Concat operation has no validated C32 mode");
        const SchedulerConnection *lhs = operation->IFM(0);
        const SchedulerConnection *rhs = operation->IFM(1);
        const SchedulerConnection *tail = operation->TryIFM(2);
        const SchedulerConnection *ofm = operation->OFM();
        const Shape lhsSlice = lhs != nullptr ? ReshapeToNHWC(lhs->SliceShape()) : Shape();
        const Shape rhsSlice = rhs != nullptr ? ReshapeToNHWC(rhs->SliceShape()) : Shape();
        const Shape tailSlice = tail != nullptr ? ReshapeToNHWC(tail->SliceShape()) : Shape();
        const Shape ofmSlice = ofm != nullptr ? ReshapeToNHWC(ofm->SliceShape()) : Shape();
        const bool structuralCsp = lhs != nullptr && rhs != nullptr && tail != nullptr && ofm != nullptr &&
            lhsSlice.Depth() == 16 && rhsSlice == lhsSlice && tailSlice == lhsSlice &&
            lhsSlice.Batch() == 1 && lhsSlice.Height() > 0 && lhsSlice.Width() > 0 &&
            ofmSlice == lhsSlice.WithDepth(48) && IsCompactNHWC(lhs->tensor->format) &&
            IsCompactNHWC(rhs->tensor->format) && IsCompactNHWC(tail->tensor->format) &&
            ofm->tensor->format == TensorFormat::C32Blocked;
        const bool compactHead = lhs != nullptr && rhs != nullptr && tail != nullptr && ofm != nullptr &&
            lhsSlice.Batch() == 1 && lhsSlice.Height() == 1 &&
            (lhsSlice.Width() == 4 || lhsSlice.Width() == 80 || lhsSlice.Width() == 144) &&
            lhsSlice.Depth() > 0 && rhsSlice.Depth() > 0 && tailSlice.Depth() > 0 &&
            lhsSlice.WithDepth(1) == rhsSlice.WithDepth(1) &&
            lhsSlice.WithDepth(1) == tailSlice.WithDepth(1) &&
            ofmSlice == lhsSlice.WithDepth(
                lhsSlice.Depth() + rhsSlice.Depth() + tailSlice.Depth()) &&
            IsCompactNHWC(lhs->tensor->format) && IsCompactNHWC(rhs->tensor->format) &&
            IsCompactNHWC(tail->tensor->format) && IsCompactNHWC(ofm->tensor->format) &&
            IsFullTensorConnection(lhs) && IsFullTensorConnection(rhs) &&
            IsFullTensorConnection(tail) && IsFullTensorConnection(ofm);
        if ( compactHead )
        {
            const std::array<const SchedulerConnection *, 3> inputs{lhs, rhs, tail};
            const uint32_t outputDepth = uint32_t(ofmSlice.Depth());
            const uint64_t outputRowBytes64 = uint64_t(ofmSlice.Width()) * outputDepth;
            const uint64_t outputBytes = uint64_t(ofmSlice.Elements64());
            if ( outputBytes > uint64_t(TensorStorageBytes(ofm->tensor.get())) ||
                 outputBytes > std::numeric_limits<uint32_t>::max() ||
                 outputRowBytes64 > std::numeric_limits<uint32_t>::max() )
                return SetError(error, "Neural-AI compact head Concat output storage is invalid");
            const uint32_t outputRowBytes = uint32_t(outputRowBytes64);
            RefV1 output = TensorRef(ofm->tensor.get(), 0, error);
            if ( !error.empty() ) return false;
            uint32_t destinationDepth = 0;
            for ( int index = 0; index < int(inputs.size()); ++index )
            {
                const SchedulerConnection *input = inputs[index];
                const Shape inputShape = ReshapeToNHWC(input->SliceShape());
                const uint32_t inputDepth = uint32_t(inputShape.Depth());
                const uint64_t inputRowBytes64 = uint64_t(inputShape.Width()) * inputDepth;
                const uint64_t inputBytes = uint64_t(inputShape.Elements64());
                if ( inputBytes > uint64_t(TensorStorageBytes(input->tensor.get())) ||
                     inputBytes > std::numeric_limits<uint32_t>::max() ||
                     inputRowBytes64 > std::numeric_limits<uint32_t>::max() )
                    return SetError(error, "Neural-AI compact head Concat input storage is invalid");
                const uint32_t inputRowBytes = uint32_t(inputRowBytes64);
                RefV1 source = TensorRef(input->tensor.get(), 0, error);
                if ( !error.empty() ) return false;
                RefV1 destination = output;
                destination.offset += destinationDepth;
                if ( source.region != uint16_t(Region::TCDMScratch) ||
                     destination.region != uint16_t(Region::TCDMScratch) )
                    return SetError(error, "Neural-AI compact head Concat requires internal TCDM tensors");
                if ( uint64_t(source.offset) < uint64_t(output.offset) + outputBytes &&
                     uint64_t(output.offset) < uint64_t(source.offset) + inputBytes )
                    return SetError(error, "Neural-AI compact head Concat requires out-of-place storage");
                if ( !AppendDMA3D(source, destination, inputDepth, inputDepth, outputDepth,
                         uint32_t(inputShape.Width()), inputRowBytes, outputRowBytes,
                         uint32_t(inputShape.Height()), uint32_t(operation->Index()),
                         uint32_t(index), error) )
                    return false;
                destinationDepth += inputDepth;
            }
            return true;
        }
        if ( structuralCsp )
        {
            const Shape outputStorage = ReshapeToNHWC(ofm->tensor->storageShape);
            const Shape outputOffset = ofm->slice.offset ?
                ReshapeToNHWC(ofm->slice.offset) : ofmSlice.WithZeros();
            if ( outputStorage.Batch() != 1 || outputStorage.Height() <= 0 ||
                 outputStorage.Width() != lhsSlice.Width() || outputStorage.Depth() < 48 ||
                 outputOffset.Batch() != 0 || outputOffset.Width() != 0 ||
                 outputOffset.Depth() != 0 || ofmSlice.Height() > outputStorage.Height() )
                return SetError(error, "Neural-AI selected CSP Concat has invalid rolling output storage");
            const uint64_t outputRowBytes64 = uint64_t(outputStorage.Width()) * 32u;
            const uint64_t outputGroupBytes64 =
                uint64_t(outputStorage.Height()) * outputRowBytes64;
            if ( outputGroupBytes64 > std::numeric_limits<uint32_t>::max() )
                return SetError(error, "Neural-AI selected CSP Concat output storage overflows the ABI");

            const std::array<const SchedulerConnection *, 3> inputs{lhs, rhs, tail};
            for ( int index = 0; index < int(inputs.size()); ++index )
            {
                const SchedulerConnection *input = inputs[index];
                const Shape inputStorage = ReshapeToNHWC(input->tensor->storageShape);
                const Shape inputOffset = input->slice.offset ?
                    ReshapeToNHWC(input->slice.offset) : lhsSlice.WithZeros();
                if ( inputStorage.Batch() != 1 || inputStorage.Width() != lhsSlice.Width() ||
                     inputStorage.Depth() != 16 || inputOffset.Batch() != 0 ||
                     inputOffset.Width() != 0 || inputOffset.Depth() != 0 ||
                     inputOffset.Height() < 0 ||
                     inputOffset.Height() + lhsSlice.Height() > inputStorage.Height() )
                    return SetError(error, "Neural-AI selected CSP Concat has invalid compact input storage");
                const uint64_t inputRowBytes64 = uint64_t(inputStorage.Width()) * 16u;
                const uint64_t sourceBase64 = uint64_t(inputOffset.Height()) * inputRowBytes64;
                const uint64_t outputRow = uint64_t(outputOffset.Height() % outputStorage.Height());
                const uint64_t destinationBase64 = outputRow * outputRowBytes64 +
                    (index == 2 ? outputGroupBytes64 : 0u) + (index == 1 ? 16u : 0u);
                if ( sourceBase64 > std::numeric_limits<uint32_t>::max() ||
                     destinationBase64 > std::numeric_limits<uint32_t>::max() ||
                     inputRowBytes64 > std::numeric_limits<uint32_t>::max() ||
                     outputRowBytes64 > std::numeric_limits<uint32_t>::max() )
                    return SetError(error, "Neural-AI selected CSP Concat reference overflows the ABI");
                RefV1 source = TensorRef(input->tensor.get(), uint32_t(sourceBase64), error);
                if ( !error.empty() ) return false;
                RefV1 destination = TensorRef(ofm->tensor.get(), uint32_t(destinationBase64), error);
                if ( !error.empty() ) return false;
                if ( !AppendDMA3D(source, destination, 16, 16, 32,
                         uint32_t(lhsSlice.Width()), uint32_t(inputRowBytes64),
                         uint32_t(outputRowBytes64), uint32_t(lhsSlice.Height()),
                         uint32_t(operation->Index()), uint32_t(index), error) )
                    return false;
            }
            return true;
        }
        if ( lhs == nullptr || rhs == nullptr || ofm == nullptr ||
             !IsFullTensorConnection(lhs) || !IsFullTensorConnection(rhs) ||
             !IsFullTensorConnection(ofm) || lhs->Type() != regor::DataType::Int8 ||
             rhs->Type() != regor::DataType::Int8 || ofm->Type() != regor::DataType::Int8 ||
             lhs->tensor->format != TensorFormat::C32Blocked ||
             rhs->tensor->format != TensorFormat::C32Blocked ||
             ofm->tensor->format != TensorFormat::C32Blocked )
            return SetError(error, fmt::format(
                "Neural-AI Concat {} requires full C32-blocked INT8 tensors "
                "(lhs={} shape={} format={} full={}, rhs={} shape={} format={} full={}, "
                "ofm={} shape={} format={} full={})",
                operation->Index(), lhs == nullptr ? "missing" : lhs->tensor->Name(),
                lhs == nullptr ? "missing" : lhs->shape.ToString(),
                lhs == nullptr ? "missing" : EnumToString(lhs->tensor->format),
                lhs != nullptr && IsFullTensorConnection(lhs),
                rhs == nullptr ? "missing" : rhs->tensor->Name(),
                rhs == nullptr ? "missing" : rhs->shape.ToString(),
                rhs == nullptr ? "missing" : EnumToString(rhs->tensor->format),
                rhs != nullptr && IsFullTensorConnection(rhs),
                ofm == nullptr ? "missing" : ofm->tensor->Name(),
                ofm == nullptr ? "missing" : ofm->shape.ToString(),
                ofm == nullptr ? "missing" : EnumToString(ofm->tensor->format),
                ofm != nullptr && IsFullTensorConnection(ofm)));
        const Shape lhsShape = ReshapeToNHWC(lhs->shape);
        const Shape rhsShape = ReshapeToNHWC(rhs->shape);
        const Shape ofmShape = ReshapeToNHWC(ofm->shape);
        if ( lhsShape.Batch() != 1 || lhsShape.WithDepth(1) != rhsShape.WithDepth(1) ||
             lhsShape.WithDepth(1) != ofmShape.WithDepth(1) ||
             lhsShape.Depth() % 32 != 0 ||
             (rhsShape.Depth() % 32 != 0 &&
                 !(lhsShape.Depth() == 64 && rhsShape.Depth() == 80 &&
                   (lhsShape.Height() == 10 || lhsShape.Height() == 20 || lhsShape.Height() == 40) &&
                   lhsShape.Width() == lhsShape.Height())) ||
             ofmShape.Depth() != lhsShape.Depth() + rhsShape.Depth() )
            return SetError(error, "Neural-AI Concat requires aligned or selected C64+C80 channel-axis inputs");
        const int64_t lhsBytes = lhs->tensor->AllocationSizeBytes();
        const int64_t rhsBytes = rhs->tensor->AllocationSizeBytes();
        const int64_t ofmBytes = ofm->tensor->AllocationSizeBytes();
        if ( lhsBytes <= 0 || rhsBytes <= 0 || ofmBytes != lhsBytes + rhsBytes ||
             ofmBytes > std::numeric_limits<uint32_t>::max() )
            return SetError(error, "Neural-AI Concat tensor storage size is invalid");
        RefV1 lhsRef = TensorRef(lhs->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        RefV1 rhsRef = TensorRef(rhs->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        RefV1 ofmLhsRef = TensorRef(ofm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        RefV1 ofmRhsRef = TensorRef(ofm->tensor.get(), uint32_t(lhsBytes), error);
        if ( !error.empty() ) return false;
        if ( lhsRef.region != uint16_t(Region::TCDMScratch) ||
             rhsRef.region != uint16_t(Region::TCDMScratch) ||
             ofmLhsRef.region != uint16_t(Region::TCDMScratch) )
            return SetError(error, "Neural-AI Concat requires internal TCDM tensors");
        const auto overlaps = [](const RefV1 &source, uint32_t sourceBytes,
                                  const RefV1 &destination, uint32_t destinationBytes)
        {
            return source.offset < destination.offset + destinationBytes &&
                   destination.offset < source.offset + sourceBytes;
        };
        if ( overlaps(lhsRef, uint32_t(lhsBytes), ofmLhsRef, uint32_t(ofmBytes)) ||
             overlaps(rhsRef, uint32_t(rhsBytes), ofmLhsRef, uint32_t(ofmBytes)) )
            return SetError(error, "Neural-AI Concat requires out-of-place output storage");
        const uint32_t layerId = uint32_t(operation->Index());
        return AppendDMA1D(lhsRef, ofmLhsRef, uint32_t(lhsBytes), layerId, 0, error) &&
               AppendDMA1D(rhsRef, ofmRhsRef, uint32_t(rhsBytes), layerId, 1, error);
    }

    bool AppendConcatStripe(const SchedulerOperation *operation, const HLCStripe *stripe,
        uint32_t &tileId, std::string &error)
    {
        const SchedulerConnection *ofm = operation->OFM();
        if ( stripe == nullptr || stripe->stripeAreas.size() != 1 || ofm == nullptr ||
             ofm->tensor->format != TensorFormat::C32Blocked )
            return SetError(error, "Neural-AI striped Concat has an invalid scheduled operation");
        const Shape start = ReshapeToNHWC(stripe->stripeAreas.front().ofmArea.Start());
        const Shape shape = ReshapeToNHWC(stripe->stripeAreas.front().ofmArea.SizeShape());
        const Shape outputStorage = ReshapeToNHWC(ofm->tensor->storageShape);
        if ( start.Batch() != 0 || start.Width() != 0 || start.Depth() != 0 ||
             shape.Batch() != 1 || shape.Height() <= 0 ||
             shape.Width() != ofm->shape.Width() || shape.Depth() != ofm->shape.Depth() ||
             outputStorage.Batch() != 1 || outputStorage.Height() <= 0 ||
             outputStorage.Width() != shape.Width() )
            return SetError(error, "Neural-AI striped Concat requires a full-width, full-depth Y stripe");

        std::vector<const SchedulerConnection *> inputs;
        for ( int index = 0; ; ++index )
        {
            const SchedulerConnection *input = operation->TryIFM(index);
            if ( input == nullptr ) break;
            inputs.push_back(input);
        }
        const bool compactC16x3 = inputs.size() == 3 && shape.Depth() == 48 &&
            std::all_of(inputs.begin(), inputs.end(), [&](const SchedulerConnection *input)
            {
                return input->shape == ofm->shape.WithDepth(16) &&
                       IsCompactNHWC(input->tensor->format);
            });
        const bool c32Inputs = inputs.size() == 2 &&
            std::all_of(inputs.begin(), inputs.end(), [&](const SchedulerConnection *input)
            {
                return input->shape.WithDepth(1) == ofm->shape.WithDepth(1) &&
                       input->tensor->format == TensorFormat::C32Blocked;
            });
        if ( !compactC16x3 && !c32Inputs )
            return SetError(error,
                "Neural-AI striped Concat requires structural compact C16x3 or native C32 inputs");
        if ( c32Inputs &&
             DivRoundUp(inputs[0]->shape.Depth(), 32) +
                     DivRoundUp(inputs[1]->shape.Depth(), 32) !=
                 DivRoundUp(ofm->shape.Depth(), 32) )
            return SetError(error, "Neural-AI striped Concat physical C32 groups are inconsistent");

        RefV1 outputBase = TensorRef(ofm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        const uint32_t rowBytes = uint32_t(shape.Width()) * 32u;
        const uint32_t outputPlaneBytes = uint32_t(outputStorage.Height()) * rowBytes;
        int outputGroup = 0;
        for ( int inputIndex = 0; inputIndex < int(inputs.size()); ++inputIndex )
        {
            const SchedulerConnection *input = inputs[inputIndex];
            const Shape inputStorage = ReshapeToNHWC(input->tensor->storageShape);
            if ( inputStorage.Batch() != 1 || inputStorage.Height() <= 0 ||
                 inputStorage.Width() != shape.Width() )
                return SetError(error, "Neural-AI striped Concat input storage is invalid");
            RefV1 inputBase = TensorRef(input->tensor.get(), 0, error);
            if ( !error.empty() ) return false;
            const uint32_t inputPixelBytes = compactC16x3 ? 16u : 32u;
            const uint32_t inputRowBytes = uint32_t(shape.Width()) * inputPixelBytes;
            const uint32_t inputPlaneBytes = uint32_t(inputStorage.Height()) * inputRowBytes;
            const int groups = compactC16x3 ? 1 : DivRoundUp(input->shape.Depth(), 32);
            for ( int group = 0; group < groups; ++group )
            {
                int rowsDone = 0;
                while ( rowsDone < shape.Height() )
                {
                    const int logicalY = start.Height() + rowsDone;
                    const int inputY = logicalY % inputStorage.Height();
                    const int outputY = logicalY % outputStorage.Height();
                    const int rows = std::min({shape.Height() - rowsDone,
                        inputStorage.Height() - inputY, outputStorage.Height() - outputY});
                    RefV1 source = inputBase;
                    RefV1 destination = outputBase;
                    source.offset += uint32_t(group) * inputPlaneBytes +
                        uint32_t(inputY) * inputRowBytes;
                    if ( compactC16x3 )
                    {
                        destination.offset += uint32_t(inputIndex / 2) * outputPlaneBytes +
                            uint32_t(outputY) * rowBytes + uint32_t(inputIndex % 2) * 16u;
                    }
                    else
                    {
                        destination.offset += uint32_t(outputGroup + group) * outputPlaneBytes +
                            uint32_t(outputY) * rowBytes;
                    }
                    if ( !AppendDMA3D(source, destination, compactC16x3 ? 16u : 32u,
                             inputPixelBytes, 32, uint32_t(shape.Width()), inputRowBytes,
                             rowBytes, uint32_t(rows), uint32_t(operation->Index()), tileId++, error) )
                        return false;
                    rowsDone += rows;
                }
            }
            if ( !compactC16x3 ) outputGroup += groups;
        }
        return true;
    }

    bool AppendHeadPack(const SchedulerOperation *operation, std::string &error)
    {
        const SchedulerOpInfo *cost = schedule->Cost(operation);
        if ( cost == nullptr || cost->Config() == nullptr )
            return SetError(error, fmt::format(
                "Neural-AI Transpose operation has no validated target mode (IFM0 {}, IFM1 {}, OFM {})",
                operation->IFM(0)->shape.ToString(),
                operation->TryIFM(1) == nullptr ? "none" : operation->TryIFM(1)->shape.ToString(),
                operation->OFM()->shape.ToString()));
        const auto *config = static_cast<const NeuralAIOpConfig *>(cost->Config());
        if ( config->Mode() != NeuralAIOpMode::HeadPackC32ToCHW )
            return SetError(error, "Neural-AI Transpose operation has no validated head-pack mode");
        const SchedulerConnection *ifm = operation->IFM(0);
        const SchedulerConnection *ifm1 = operation->TryIFM(1);
        const SchedulerConnection *ofm = operation->OFM();
        if ( ifm == nullptr || ofm == nullptr || !IsFullTensorConnection(ifm) ||
             !IsFullTensorConnection(ofm) || ifm->Type() != regor::DataType::Int8 ||
             ofm->Type() != regor::DataType::Int8 ||
             ifm->tensor->format != TensorFormat::C32Blocked ||
             (ifm1 != nullptr && (!IsFullTensorConnection(ifm1) ||
                 ifm1->Type() != regor::DataType::Int8 ||
                 ifm1->tensor->format != TensorFormat::C32Blocked)) ||
             ofm->tensor->format != TensorFormat::NHWC )
            return SetError(error, "Neural-AI head pack requires full C32-to-compact INT8 tensors");
        const Shape inputShape = ifm->shape;
        const Shape outputShape = ofm->shape;
        const bool selectedSpatial = inputShape.Height() == 10 || inputShape.Height() == 20 ||
                                     inputShape.Height() == 40;
        const bool splitHeadPack = ifm1 != nullptr && inputShape.Depth() == 64 &&
                                   ifm1->shape == inputShape.WithDepth(80);
        if ( inputShape.Size() != 4 || outputShape.Size() != 4 || inputShape.Batch() != 1 ||
             (inputShape.Depth() != 144 && !splitHeadPack) || !selectedSpatial ||
             inputShape.Width() != inputShape.Height() ||
             outputShape != Shape(1, 144, inputShape.Height(), inputShape.Width()) )
            return SetError(error, "Neural-AI head pack requires selected square C144 NCHW transpose");
        const int64_t pixels64 = int64_t(inputShape.Height()) * inputShape.Width();
        const int64_t destinationBytes64 = pixels64 * 144;
        const int64_t sourceBytes64 = pixels64 * ((inputShape.Depth() + 31) & ~31);
        const int64_t source1Bytes64 = splitHeadPack ? pixels64 * 96 : 0;
        if ( pixels64 <= 0 || sourceBytes64 != ifm->tensor->AllocationSizeBytes() ||
             (splitHeadPack && source1Bytes64 != ifm1->tensor->AllocationSizeBytes()) ||
             destinationBytes64 != ofm->tensor->AllocationSizeBytes() ||
             sourceBytes64 > std::numeric_limits<uint32_t>::max() ||
             source1Bytes64 > std::numeric_limits<uint32_t>::max() ||
             destinationBytes64 > std::numeric_limits<uint32_t>::max() )
            return SetError(error, "Neural-AI head-pack tensor storage size is invalid");
        RefV1 source = TensorRef(ifm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        RefV1 source1{};
        if ( splitHeadPack )
        {
            source1 = TensorRef(ifm1->tensor.get(), 0, error);
            if ( !error.empty() ) return false;
        }
        RefV1 destination = TensorRef(ofm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        if ( source.region != uint16_t(Region::TCDMScratch) ||
             (splitHeadPack && source1.region != uint16_t(Region::TCDMScratch)) ||
             destination.region != uint16_t(Region::TCDMScratch) )
            return SetError(error, "Neural-AI head pack requires internal TCDM tensors");
        const uint32_t destinationBytes = uint32_t(destinationBytes64);
        const auto appendPart = [&](RefV1 partSource, uint32_t sourceBytes,
                                    uint32_t channelOffset, uint32_t channels, uint32_t part)
        {
            RefV1 partDestination = destination;
            partDestination.offset += channelOffset * uint32_t(pixels64);
            if ( partSource.offset < destination.offset + destinationBytes &&
                 destination.offset < partSource.offset + sourceBytes )
                return SetError(error, "Neural-AI head pack requires out-of-place output storage");
            AppendHeader(artifact->commands, CommandType::CopyLayout, 96,
                uint32_t(operation->Index()), part);
            AppendRef(artifact->commands, partSource);
            AppendRef(artifact->commands, partDestination);
            Append16(artifact->commands, uint16_t(CopyLayoutMode::C32ToCHW));
            Append16(artifact->commands, ABILayout(TensorFormat::C32Blocked));
            Append16(artifact->commands, ABILayout(TensorFormat::NHWC));
            Append16(artifact->commands, uint16_t(DataType::Int8));
            for ( uint32_t dimension : Dimensions(inputShape.WithDepth(int(channels))) )
                Append32(artifact->commands, dimension);
            Append32(artifact->commands, channels);
            Append32(artifact->commands, uint32_t(pixels64 * 32));
            Append32(artifact->commands, uint32_t(pixels64));
            AppendZeros(artifact->commands, 7);
            ++artifact->commandCount;
            return true;
        };
        if ( splitHeadPack )
        {
            if ( !appendPart(source, uint32_t(sourceBytes64), 0, 64, 0) ||
                 !appendPart(source1, uint32_t(source1Bytes64), 64, 80, 1) ) return false;
        }
        else if ( !appendPart(source, uint32_t(sourceBytes64), 0, 144, 0) ) return false;
        return true;
    }

    bool AppendDFL16(const SchedulerOperation *operation, std::string &error)
    {
        const SchedulerOpInfo *cost = schedule->Cost(operation);
        const auto *config = cost != nullptr && cost->Config() != nullptr ?
            static_cast<const NeuralAIOpConfig *>(cost->Config()) : nullptr;
        if ( config == nullptr || config->Mode() != NeuralAIOpMode::Dfl16 )
            return SetError(error, "Neural-AI DFL operation has no validated DFL16 mode");
        const SchedulerConnection *ifm = operation->IFM(0);
        const SchedulerConnection *ifm1 = operation->TryIFM(1);
        const SchedulerConnection *ofm = operation->OFM();
        const SchedulerConnection *classOfm = operation->outputs.try_ref(MakeTensorUsage(TensorUsage::OFM, 1));
        const SchedulerConnection *classLut = operation->TryInput(TensorUsage::LUT);
        const SchedulerConnection *scratch = operation->TryInput(TensorUsage::Scratch);
        const SchedulerConnection *boxScale = operation->TryInput(TensorUsage::Params);
        const bool splitHead = ifm != nullptr && ifm1 != nullptr && ifm->shape.Size() == 4 &&
                               ifm->shape.Batch() == 1 && ifm->shape.Depth() == 64 &&
                               ifm1->shape == ifm->shape.WithDepth(80);
        const int64_t locations64 = splitHead ? int64_t(ifm->shape.Height()) * ifm->shape.Width() :
            (ifm != nullptr && ifm->shape.Size() == 4 ? ifm->shape.Depth() : 0);
        const int locations = locations64 > 0 && locations64 <= std::numeric_limits<int>::max() ?
            int(locations64) : 0;
        if ( ifm == nullptr || ofm == nullptr || scratch == nullptr || !IsFullTensorConnection(ifm) ||
             !IsFullTensorConnection(ofm) || ifm->Type() != regor::DataType::Int8 || ofm->Type() != regor::DataType::Int8 ||
             (splitHead && (!IsFullTensorConnection(ifm1) || ifm1->Type() != regor::DataType::Int8)) ||
             locations <= 0 || (!splitHead && ifm->shape != Shape(1, 1, 144, locations)) ||
             ofm->shape != Shape(1, 1, 4, locations) ||
             (splitHead ? (ifm->tensor->format != TensorFormat::C32Blocked ||
                              ifm1->tensor->format != TensorFormat::C32Blocked) :
                          !IsCompactNHWC(ifm->tensor->format)) ||
             !IsCompactNHWC(ofm->tensor->format) ||
             scratch->tensor->format != TensorFormat::Row32 || scratch->tensor->AllocationSizeBytes() < 1088 )
            return SetError(error,
                "Neural-AI DFL16 requires compact or split-C32 logits, compact output, and tiled ROW32 scratch");
        if ( (classOfm == nullptr) != (classLut == nullptr) )
            return SetError(error, "Neural-AI fused DFL16 class output requires a LUT and output together");
        if ( classOfm != nullptr &&
             (!IsFullTensorConnection(classOfm) || classOfm->Type() != regor::DataType::Int8 ||
              classOfm->shape != Shape(1, 1, 80, locations) ||
              !IsCompactNHWC(classOfm->tensor->format) || !classLut->tensor->IsConstant() ||
              classLut->tensor->dataType != regor::DataType::Int8 ||
              classLut->tensor->bufferView.Elements() != 256) )
            return SetError(error,
                "Neural-AI fused DFL16 class output requires compact INT8 classes and a 256-byte LUT");
        if ( ofm->quantization.scales.size() != 1 || ofm->quantization.zeroPoints.size() != 1 ||
             ofm->quantization.zeroPoints[0] != -128 )
            return SetError(error, "Neural-AI DFL16 requires the selected scalar output quantization");
        double boxScaleValue = 1.0;
        if ( boxScale != nullptr )
        {
            if ( !boxScale->tensor->IsConstant() || boxScale->Type() != regor::DataType::Int8 ||
                 boxScale->tensor->bufferView.Elements() != 1 ||
                 boxScale->quantization.scales.size() != 1 ||
                 boxScale->quantization.zeroPoints.size() != 1 )
                return SetError(error, "Neural-AI DFL16 box scale must be one scalar INT8 constant");
            const int rawScale = boxScale->tensor->bufferView.Values<int>(regor::DataType::Int8)[0];
            boxScaleValue = (double(rawScale) - double(boxScale->quantization.zeroPoints[0])) *
                boxScale->quantization.scales[0].Dequantize();
        }
        const double outputScale = ofm->quantization.scales[0].Dequantize();
        int32_t outputMultiplier = 0;
        int32_t outputShift = 0;
        const int clampMin = ofm->quantization.quantMin.empty() ? -128 :
            int(ofm->quantization.quantMin[0]);
        const int clampMax = ofm->quantization.quantMax.empty() ? 127 :
            int(ofm->quantization.quantMax[0]);
        if ( boxScaleValue <= 0 || outputScale <= 0 ||
             !neuralai::CalculateQuantizedMultiplier(
                 boxScaleValue / (256.0 * outputScale), outputMultiplier, outputShift, 65535) ||
             outputShift < 0 ||
             clampMin < -128 || clampMax > 127 || clampMin > clampMax )
            return SetError(error, fmt::format(
                "Neural-AI DFL16 output requantization is outside the INT8 contract "
                "(box_scale={}, output_scale={}, multiplier={}, shift={}, clamp=[{},{}])",
                boxScaleValue, outputScale, outputMultiplier, outputShift, clampMin, clampMax));
        if ( splitHead &&
             (ifm->tensor->AllocationSizeBytes() != int64_t(locations) * 64 ||
                 ifm1->tensor->AllocationSizeBytes() != int64_t(locations) * 96) )
            return SetError(error, "Neural-AI split DFL16 source storage size is invalid");

        RefV1 source = TensorRef(ifm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        RefV1 destination = TensorRef(ofm->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        RefV1 scratchRef = TensorRef(scratch->tensor.get(), 0, error);
        if ( !error.empty() ) return false;
        if ( source.region != uint16_t(Region::TCDMScratch) ||
             destination.region != uint16_t(Region::TCDMScratch) ||
             scratchRef.region != uint16_t(Region::TCDMScratch) )
            return SetError(error, "Neural-AI DFL16 requires internal TCDM tensors");

        const auto appendLut = [this](bool reciprocal)
        {
            artifact->constants.resize(RoundAway(int(artifact->constants.size()),
                int(ArchNeuralAI::DMAAlignment)), 0);
            RefV1 reference{};
            reference.region = uint16_t(Region::ModelConstants);
            reference.offset = uint32_t(artifact->constants.size());
            for ( uint32_t index = 0; index < 256; ++index )
            {
                uint32_t value;
                if ( reciprocal )
                {
                    const uint32_t midpointQ9 = 513 + index * 2;
                    value = uint32_t(((uint64_t(1) << 37) + (midpointQ9 >> 1)) / midpointQ9);
                }
                else if ( index == 0 )
                {
                    value = 32768;
                }
                else
                {
                    const uint32_t negDelta = 256 - index;
                    const uint32_t shift = negDelta >> 4;
                    const uint32_t fraction = negDelta & 15;
                    const uint32_t base = shift >= 15 ? 1 : 32768 >> shift;
                    const uint32_t next = base > 1 ? base >> 1 : 1;
                    value = ((base * (16 - fraction)) + (next * fraction) + 8) >> 4;
                    if ( value == 0 ) value = 1;
                }
                Append32(artifact->constants, value);
            }
            return reference;
        };
        const RefV1 expLut = appendLut(false);
        const RefV1 recipLut = appendLut(true);
        AppendHeader(artifact->commands, CommandType::AFUDFL16, 96, uint32_t(operation->Index()), 0);
        AppendRef(artifact->commands, source);
        AppendRef(artifact->commands, destination);
        AppendRef(artifact->commands, scratchRef);
        AppendRef(artifact->commands, expLut);
        AppendRef(artifact->commands, recipLut);
        Append32(artifact->commands, uint32_t(locations));
        Append32(artifact->commands, splitHead ? 1u : 0u);
        Append32(artifact->commands, uint32_t(outputMultiplier));
        Append32(artifact->commands, uint32_t(outputShift));
        Append32(artifact->commands, uint32_t(ofm->quantization.zeroPoints[0]));
        Append32(artifact->commands, uint32_t(clampMin));
        Append32(artifact->commands, uint32_t(clampMax));
        AppendZeros(artifact->commands, 3);
        ++artifact->commandCount;
        if ( classOfm != nullptr )
        {
            const uint64_t classOffset64 = uint64_t(64) * uint64_t(locations);
            const uint64_t classBytes64 = uint64_t(80) * uint64_t(locations);
            if ( classOffset64 > std::numeric_limits<uint32_t>::max() ||
                 classBytes64 > std::numeric_limits<uint32_t>::max() )
                return SetError(error, "Neural-AI fused DFL16 class view overflows the ABI");
            RefV1 classSource = splitHead ? TensorRef(ifm1->tensor.get(), 0, error) :
                TensorRef(ifm->tensor.get(), uint32_t(classOffset64), error);
            if ( !error.empty() ) return false;
            RefV1 classDestination = TensorRef(classOfm->tensor.get(), 0, error);
            if ( !error.empty() ) return false;
            if ( classSource.region != uint16_t(Region::TCDMScratch) ||
                 classDestination.region != uint16_t(Region::TCDMScratch) )
                return SetError(error, "Neural-AI fused DFL16 class LUT requires internal TCDM tensors");
            const uint32_t classBytes = uint32_t(classBytes64);
            const uint32_t classSourceBytes = splitHead ? uint32_t(locations) * 96u : classBytes;
            if ( uint64_t(classSource.offset) < uint64_t(classDestination.offset) + classBytes &&
                 uint64_t(classDestination.offset) < uint64_t(classSource.offset) + classSourceBytes )
                return SetError(error, "Neural-AI fused DFL16 class LUT requires out-of-place storage");

            uint32_t lutTile = 1;
            if ( splitHead )
            {
                AppendHeader(artifact->commands, CommandType::CopyLayout, 96,
                    uint32_t(operation->Index()), lutTile++);
                AppendRef(artifact->commands, classSource);
                AppendRef(artifact->commands, classDestination);
                Append16(artifact->commands, uint16_t(CopyLayoutMode::C32ToCHW));
                Append16(artifact->commands, ABILayout(TensorFormat::C32Blocked));
                Append16(artifact->commands, ABILayout(TensorFormat::NHWC));
                Append16(artifact->commands, uint16_t(DataType::Int8));
                for ( uint32_t dimension : Dimensions(ifm1->shape) )
                    Append32(artifact->commands, dimension);
                Append32(artifact->commands, 80);
                Append32(artifact->commands, uint32_t(locations) * 32u);
                Append32(artifact->commands, uint32_t(locations));
                AppendZeros(artifact->commands, 7);
                ++artifact->commandCount;
                classSource = classDestination;
            }

            AppendHeader(artifact->commands, CommandType::AFULut, 64,
                uint32_t(operation->Index()), lutTile);
            AppendRef(artifact->commands, classSource);
            AppendRef(artifact->commands, classDestination);
            AppendRef(artifact->commands, LutRef(classLut));
            Append32(artifact->commands, classBytes);
            AppendZeros(artifact->commands, 5);
            ++artifact->commandCount;
        }
        return true;
    }

    MatrixData &MaterializeMatrixData(const SchedulerOperation *operation, uint32_t paddedN,
        const WeightRange &range, const uint8_t *data)
    {
        auto [position, inserted] = matrixData.emplace(operation->Uid(), MatrixData{});
        MatrixData &materialized = position->second;
        if ( inserted )
        {
            materialized.qparamBase = uint32_t(artifact->qparams.size());
            for ( uint32_t channel = 0; channel < paddedN; ++channel )
            {
                const uint8_t *source = data + channel * sizeof(neuralai::QParamV1);
                neuralai::QParamV1 qparam{};
                qparam.bias = int32_t(Read32(source));
                qparam.multiplier = int32_t(Read32(source + 4));
                qparam.shift = Read32(source + 8);
                qparam.zeroPoint = int32_t(Read32(source + 12));
                qparam.clampMin = int32_t(Read32(source + 16));
                qparam.clampMax = int32_t(Read32(source + 20));
                artifact->qparams.push_back(qparam);
            }
            materialized.weightBase = uint32_t(artifact->constants.size());
            const uint8_t *weightData = data + range.weightOffset;
            artifact->constants.insert(artifact->constants.end(), weightData, weightData + range.weightBytes);
        }
        return materialized;
    }

    RefV1 FillPatternRef(int8_t value)
    {
        auto position = fillReferences.find(int(value));
        if ( position != fillReferences.end() ) return position->second;
        while ( artifact->constants.size() % ArchNeuralAI::DMAAlignment != 0 )
            artifact->constants.push_back(0);
        RefV1 reference{};
        reference.region = uint16_t(Region::ModelConstants);
        reference.offset = uint32_t(artifact->constants.size());
        artifact->constants.insert(artifact->constants.end(), 32, uint8_t(value));
        fillReferences.emplace(int(value), reference);
        return reference;
    }

    bool AppendStripeStaging(const SchedulerConnection *ifm, const neuralai::StripeStagingPlan &plan,
        int8_t paddingValue, uint32_t layerId, uint32_t &tileId, bool fill, std::string &error)
    {
        if ( fill )
        {
            const RefV1 pattern = FillPatternRef(paddingValue);
            RefV1 destination{};
            destination.region = uint16_t(Region::TCDMScratch);
            destination.offset = stripeStageOffset;
            const uint32_t blocks = plan.bytes / 32u;
            if ( blocks != 0 && !AppendDMA2D(pattern, destination, 32, 0, 32, blocks,
                                   layerId, tileId++, error) )
                return false;
            const uint32_t tail = plan.bytes % 32u;
            if ( tail != 0 )
            {
                destination.offset += blocks * 32u;
                if ( !AppendDMA2D(pattern, destination, tail, 0, tail, 1,
                         layerId, tileId++, error) )
                    return false;
            }
        }
        for ( const auto &copy : plan.copies )
        {
            RefV1 source = TensorRef(ifm->tensor.get(), copy.source, error);
            if ( !error.empty() ) return false;
            RefV1 destination{};
            destination.region = uint16_t(Region::TCDMScratch);
            destination.offset = copy.destination;
            if ( copy.repetitions3 != 0 )
            {
                if ( !AppendDMA3D(source, destination, copy.length, copy.sourceStride,
                         copy.destinationStride, copy.repetitions, copy.sourceStride3,
                         copy.destinationStride3, copy.repetitions3, layerId, tileId++, error) )
                    return false;
            }
            else if ( !AppendDMA2D(source, destination, copy.length, copy.sourceStride,
                          copy.destinationStride, copy.repetitions, layerId, tileId++, error) )
                return false;
        }
        return true;
    }

    bool AppendC32MatrixStripe(const SchedulerOperation *operation, const HLCStripe *stripe,
        uint32_t &tileId, std::string &error)
    {
        const SchedulerOpInfo *cost = schedule->Cost(operation);
        const auto *config = cost != nullptr && cost->Config() != nullptr ?
            static_cast<const NeuralAIOpConfig *>(cost->Config()) : nullptr;
        if ( config != nullptr && config->Mode() == NeuralAIOpMode::Conv2DPointwiseC32Requant )
        {
            if ( cost->npuWeightsTensor == nullptr || stripe == nullptr ||
                 stripe->stripeAreas.size() != 1 )
                return SetError(error, "Neural-AI striped pointwise Conv has no scheduled constants or stripe");
            const SchedulerConnection *ifm = operation->IFM(0);
            const SchedulerConnection *ofm = operation->OFM();
            const StripeArea &area = stripe->stripeAreas.front();
            if ( area.ifmCount != 1 || ifm->Type() != regor::DataType::Int8 ||
                 ofm->Type() != regor::DataType::Int8 ||
                 ifm->tensor->format != TensorFormat::C32Blocked ||
                 ofm->tensor->format != TensorFormat::C32Blocked )
                return SetError(error, "Neural-AI striped pointwise Conv requires C32 feature maps");

            const Shape ifmAreaStart = ReshapeToNHWC(area.ifmAreas[0].Start());
            const Shape ifmAreaShape = ReshapeToNHWC(area.ifmAreas[0].SizeShape());
            const Shape ofmAreaStart = ReshapeToNHWC(area.ofmArea.Start());
            const Shape ofmAreaShape = ReshapeToNHWC(area.ofmArea.SizeShape());
            const Shape ifmStorage = ReshapeToNHWC(ifm->tensor->storageShape);
            const Shape ofmStorage = ReshapeToNHWC(ofm->tensor->storageShape);
            const uint32_t channelK = uint32_t(ifm->shape.Depth());
            const uint32_t depthN = uint32_t(ofm->shape.Depth());
            const uint32_t paddedK = uint32_t(RoundAway(int(channelK), 32));
            const uint32_t paddedN = uint32_t(RoundAway(int(depthN), 32));
            const uint32_t kGroups = paddedK / 32u;
            const uint32_t nGroups = paddedN / 32u;
            if ( ifmAreaStart.Batch() != 0 || ofmAreaStart.Batch() != 0 ||
                 ifmAreaStart.Width() != 0 || ofmAreaStart.Width() != 0 ||
                 ifmAreaStart.Depth() != 0 || ofmAreaStart.Depth() != 0 ||
                 ifmAreaShape != ofmAreaShape.WithDepth(int(channelK)) ||
                 ofmAreaShape.Batch() != 1 || ofmAreaShape.Height() <= 0 ||
                 ofmAreaShape.Width() != ofm->shape.Width() ||
                 ofmAreaShape.Depth() != int(depthN) || ifmStorage.Height() <= 0 ||
                 ofmStorage.Height() <= 0 || ifmStorage.Width() != ofmAreaShape.Width() ||
                 ofmStorage.Width() != ofmAreaShape.Width() || kGroups == 0 || nGroups == 0 )
                return SetError(error, "Neural-AI striped pointwise Conv has an invalid full-width Y stripe");

            const NpuWeightTensor *encoded = cost->npuWeightsTensor.get();
            if ( encoded->encodedRanges.size() != 1 || !encoded->bufferView.HasBuffer() )
                return SetError(error, "Neural-AI striped pointwise Conv requires one encoded constant range");
            const WeightRange &range = encoded->encodedRanges.begin()->second;
            if ( range.scaleBytes != int(paddedN * sizeof(neuralai::QParamV1)) ||
                 range.weightBytes != int(kGroups * nGroups * 32u * 32u) )
                return SetError(error, "Neural-AI striped pointwise Conv constants have invalid dimensions");
            const uint8_t *data = encoded->bufferView.RawData<uint8_t>() + range.offset;
            MatrixData &materialized = MaterializeMatrixData(operation, paddedN, range, data);
            const uint32_t inputGroupStride = uint32_t(ifmStorage.Height()) *
                uint32_t(ifmStorage.Width()) * 32u;
            const uint32_t outputGroupStride = uint32_t(ofmStorage.Height()) *
                uint32_t(ofmStorage.Width()) * 32u;
            uint32_t ifmSliceOffset = 0;
            if ( !C32SliceByteOffset(ifm, ifmSliceOffset, error) ) return false;
            RefV1 ifmBase = TensorRef(ifm->tensor.get(), ifmSliceOffset, error);
            if ( !error.empty() ) return false;
            RefV1 ofmBase = TensorRef(ofm->tensor.get(), 0, error);
            if ( !error.empty() ) return false;

            int rowsDone = 0;
            while ( rowsDone < ofmAreaShape.Height() )
            {
                const int inputY = (ifmAreaStart.Height() + rowsDone) % ifmStorage.Height();
                const int outputY = (ofmAreaStart.Height() + rowsDone) % ofmStorage.Height();
                const int rows = std::min({ofmAreaShape.Height() - rowsDone,
                    ifmStorage.Height() - inputY, ofmStorage.Height() - outputY});
                const uint32_t matrixRows = uint32_t(rows * ofmAreaShape.Width());
                if ( kGroups > 1u && uint64_t(matrixRows) * 32u * 4u > partialBytes )
                    return SetError(error, "Neural-AI striped pointwise Conv exceeds its partial buffer");
                for ( uint32_t nGroup = 0; nGroup < nGroups; ++nGroup )
                {
                    AppendRQLoad(materialized.qparamBase + nGroup * 32u, nGroup,
                        uint32_t(operation->Index()), tileId++);
                    RefV1 weights{};
                    weights.region = uint16_t(Region::ModelConstants);
                    weights.offset = materialized.weightBase + nGroup * kGroups * 32u * 32u;
                    RefV1 input = ifmBase;
                    input.offset += uint32_t(inputY * ifmStorage.Width()) * 32u;
                    RefV1 output = ofmBase;
                    output.offset += nGroup * outputGroupStride +
                        uint32_t(outputY * ofmStorage.Width()) * 32u;
                    RefV1 partial{};
                    if ( kGroups > 1u )
                    {
                        partial.region = uint16_t(Region::TCDMScratch);
                        partial.offset = partialOffset;
                    }
                    AppendPointwiseC32(weights, input, partial, output, matrixRows,
                        kGroups, 1, nGroup, inputGroupStride, outputGroupStride,
                        uint32_t(operation->Index()), tileId++);
                }
                rowsDone += rows;
            }
            return true;
        }
        if ( config == nullptr || !IsLinebufferConvMode(config->Mode()) || cost->npuWeightsTensor == nullptr ||
             stripe == nullptr || stripe->stripeAreas.size() != 1 )
            return SetError(error, fmt::format(
                "Neural-AI striped matrix operation {} requires linebuffer Conv2D, got {}",
                operation->Index(), config == nullptr ? "no config" : NeuralAIOpModeName(config->Mode())));
        const SchedulerConnection *ifm = operation->IFM(0);
        const SchedulerConnection *ofm = operation->OFM();
        const StripeArea &area = stripe->stripeAreas.front();
        if ( area.ifmCount != 1 || ifm->Type() != regor::DataType::Int8 ||
             ofm->Type() != regor::DataType::Int8 || ofm->tensor->format != TensorFormat::C32Blocked )
            return SetError(error, "Neural-AI direct RGB stripe has invalid feature maps");
        const Shape ifmShape = ReshapeToNHWC(ifm->shape);
        const Shape ofmAreaShape = ReshapeToNHWC(area.ofmArea.SizeShape());
        const Shape ofmAreaStart = ReshapeToNHWC(area.ofmArea.Start());
        const Shape ofmStorage = ReshapeToNHWC(ofm->tensor->storageShape);
        const uint32_t channelK = uint32_t(ifmShape.Depth());
        const uint32_t depthN = uint32_t(ofmAreaShape.Depth());
        if ( channelK == 0 || channelK > 32 || depthN <= 0 || depthN > 32 || ofmAreaStart.Width() != 0 ||
             ofmAreaStart.Depth() != 0 || ofmAreaShape.Width() != ofm->shape.Width() ||
             ofmStorage.Width() != ofmAreaShape.Width() || ofmStorage.Height() <= 0 ||
             ofmAreaShape.Height() > ofmStorage.Height() ||
             ofmAreaStart.Height() % ofmStorage.Height() + ofmAreaShape.Height() > ofmStorage.Height() )
            return SetError(error, "Neural-AI Conv stripe exceeds its C32 rolling output");

        neuralai::StripeStagingInput stagingInput{};
        stagingInput.logicalIfm = ifmShape;
        stagingInput.storageIfm = ReshapeToNHWC(ifm->tensor->storageShape);
        stagingInput.validArea = area.ifmAreas[0];
        uint32_t ifmSliceOffset = 0;
        if ( !config->DirectNhwcInput() )
        {
            if ( ifm->tensor->format == TensorFormat::C32Blocked )
            {
                if ( !C32SliceByteOffset(ifm, ifmSliceOffset, error) ) return false;
            }
            else if ( !IsCompactNHWC(ifm->tensor->format) || !IsFullTensorConnection(ifm) )
                return SetError(error,
                    "Neural-AI compact stripe staging requires a full tensor connection");
        }
        stagingInput.sourceBase = ifmSliceOffset;
        stagingInput.stagingBase = stripeStageOffset;
        stagingInput.padTop = stripe->padding.top;
        stagingInput.padLeft = stripe->padding.left;
        stagingInput.padBottom = stripe->padding.bottom;
        stagingInput.padRight = stripe->padding.right;
        stagingInput.channels = int(channelK);
        stagingInput.directNhwc = config->DirectNhwcInput();
        const auto staging = neuralai::LinebufferPlanner().PlanStripeStaging(stagingInput);
        if ( staging.bytes > stripeStageBytes )
            return SetError(error, "Neural-AI Conv stripe exceeds its planned staging workspace");
        const bool compactTail = !config->DirectNhwcInput() &&
            stagingInput.storageIfm.Depth() == int(channelK) && channelK % 32 != 0;
        const bool needsFill = compactTail || stripe->padding.top != 0 || stripe->padding.left != 0 ||
            stripe->padding.bottom != 0 || stripe->padding.right != 0;
        int8_t paddingValue = 0;
        if ( const SchedulerConnection *padding = operation->TryInput(TensorUsage::Params1) )
        {
            if ( !padding->tensor->IsConstant() || padding->Type() != regor::DataType::Int8 ||
                 padding->tensor->bufferView.Elements() != 1 )
                return SetError(error, "Neural-AI Conv stripe padding value is invalid");
            paddingValue = padding->tensor->bufferView.Values<int8_t>()[0];
        }
        if ( !AppendStripeStaging(ifm, staging, paddingValue, uint32_t(operation->Index()),
                 tileId, needsFill, error) )
            return false;

        const NpuWeightTensor *encoded = cost->npuWeightsTensor.get();
        if ( encoded->encodedRanges.size() != 1 || !encoded->bufferView.HasBuffer() )
            return SetError(error, "Neural-AI Conv stripe requires one encoded weight range");
        const WeightRange &range = encoded->encodedRanges.begin()->second;
        const uint32_t paddedN = uint32_t(RoundAway(int(depthN), 32));
        const uint32_t weightGroups = config->DirectNhwcInput() ? 1u : 9u;
        if ( range.scaleBytes != int(paddedN * sizeof(neuralai::QParamV1)) ||
             range.weightBytes != int(weightGroups * 32u * 32u) )
            return SetError(error, "Neural-AI Conv stripe encoded constants have invalid dimensions");
        const uint8_t *data = encoded->bufferView.RawData<uint8_t>() + range.offset;
        MatrixData &materialized = MaterializeMatrixData(operation, paddedN, range, data);
        if ( !materialized.linebufferWeightsStaged )
        {
            RefV1 source{};
            source.region = uint16_t(Region::ModelConstants);
            source.offset = materialized.weightBase;
            RefV1 destination{};
            destination.region = uint16_t(Region::TCDMScratch);
            destination.offset = weightStageOffset;
            if ( !AppendDMA2D(source, destination, 32, 32, 32, uint32_t(range.weightBytes) / 32u,
                     uint32_t(operation->Index()), tileId++, error) )
                return false;
            materialized.linebufferWeightsStaged = true;
        }

        AppendRQLoad(materialized.qparamBase, 0, uint32_t(operation->Index()), tileId++);
        neuralai::LinebufferPlannerInput plannerInput{};
        plannerInput.logicalIfm = staging.stagedIfm;
        plannerInput.logicalOfm = ofmAreaShape;
        plannerInput.ifmBase = stripeStageOffset;
        plannerInput.ofmBase = uint32_t(ofm->tensor->AllocatedAddress()) +
            uint32_t(ofmAreaStart.Height() % ofmStorage.Height()) *
                uint32_t(ofmStorage.Width()) * 32u;
        plannerInput.weightBase = weightStageOffset;
        plannerInput.psumBase = partialOffset;
        plannerInput.kernelH = operation->Kernel()->Size().y;
        plannerInput.kernelW = operation->Kernel()->Size().x;
        plannerInput.strideH = operation->Kernel()->Stride().y;
        plannerInput.strideW = operation->Kernel()->Stride().x;
        plannerInput.ic = config->DirectNhwcInput() ? int(channelK) : 32;
        plannerInput.oc = 32;
        plannerInput.validLaneCount = int(channelK);
        plannerInput.ifmPixelStride = config->DirectNhwcInput() ? int(channelK) : 32;
        plannerInput.maxM = MaxDirectLinebufferM;
        plannerInput.tcdmBudget = ArchNeuralAI::AllocatableTCDMBytes;
        for ( const auto &job : neuralai::LinebufferPlanner().Plan(plannerInput) )
            AppendLineBufferJob(job, uint32_t(operation->Index()), tileId++);
        return true;
    }

    bool AppendMatrix(const SchedulerOperation *operation, std::string &error)
    {
        const SchedulerOpInfo *cost = schedule->Cost(operation);
        if ( cost == nullptr || cost->Config() == nullptr || cost->npuWeightsTensor == nullptr )
            return SetError(error, "Neural-AI matrix operation has no encoded constant tensor");
        const auto *config = static_cast<const NeuralAIOpConfig *>(cost->Config());
        const NeuralAIOpMode mode = config->Mode();
        if ( mode != NeuralAIOpMode::FullyConnectedRow32 &&
             mode != NeuralAIOpMode::MatMulRow32 &&
             mode != NeuralAIOpMode::Conv2DPointwiseC32Requant &&
             !IsLinebufferConvMode(mode) )
            return SetError(error, "Neural-AI matrix operation has no validated matrix or Conv mode");
        const NpuWeightTensor *encoded = cost->npuWeightsTensor.get();
        if ( encoded->encodedRanges.size() != 1 || !encoded->bufferView.HasBuffer() )
            return SetError(error, "Neural-AI matrix operation requires one encoded constant range");
        const WeightRange &range = encoded->encodedRanges.begin()->second;
        const uint8_t *data = encoded->bufferView.RawData<uint8_t>() + range.offset;
        if ( range.scaleBytes <= 0 || range.scaleBytes % int(sizeof(neuralai::QParamV1)) != 0 ||
             range.weightBytes <= 0 || range.weightBytes % (32 * 32) != 0 )
            return SetError(error, "Neural-AI encoded matrix constants do not match GEMM32 tiles");

        const SchedulerConnection *ifm = operation->IFM(0);
        const SchedulerConnection *ofm = operation->OFM();
        const Shape ifmShape = ifm->SliceShape();
        const Shape ofmShape = ofm->SliceShape();
        const uint32_t channelK = uint32_t(ifmShape.Depth());
        const bool isK3Conv = IsLinebufferConvMode(mode);
        const uint32_t depthK = channelK * (isK3Conv ? 9u : 1u);
        const uint32_t depthN = uint32_t(ofmShape.Depth());
        const int64_t rows64 = depthN != 0 ? ofmShape.Elements64() / depthN : 0;
        if ( rows64 <= 0 || rows64 > std::numeric_limits<uint32_t>::max() )
            return SetError(error, "Neural-AI matrix row count is outside the ABI range");
        const uint32_t rows = uint32_t(rows64);
        const uint32_t paddedK = uint32_t(RoundAway(int(depthK), 32));
        const uint32_t paddedN = uint32_t(RoundAway(int(depthN), 32));
        const uint32_t kGroups = paddedK / 32;
        const uint32_t nGroups = paddedN / 32;
        const bool linebufferK3 = isK3Conv;
        const uint32_t linebufferKGroups = linebufferK3 && !config->DirectNhwcInput() ?
            9u * uint32_t(RoundAway(int(channelK), 32)) / 32u : kGroups;
        if ( rows == 0 || range.scaleBytes != int(paddedN * sizeof(neuralai::QParamV1)) ||
             range.weightBytes != int(linebufferKGroups * nGroups * 32 * 32) )
            return SetError(error, "Neural-AI encoded matrix dimensions do not match the scheduled operation");

        MatrixData &materialized = MaterializeMatrixData(operation, paddedN, range, data);
        const uint32_t qparamBase = materialized.qparamBase;
        const uint32_t weightBase = materialized.weightBase;

        if ( isK3Conv )
        {
            const bool directRgb = config->DirectNhwcInput();
            if ( directRgb && ifm->tensor->isGraphInput )
                return SetError(error,
                    "Neural-AI direct RGB binding requires a striped stem cascade");
            if ( (!directRgb && ifm->tensor->format != TensorFormat::C32Blocked) ||
                 ofm->tensor->format != TensorFormat::C32Blocked )
                return SetError(error, "Neural-AI linebuffer Conv requires C32-blocked input and output");
            const uint32_t weightBytes = uint32_t(range.weightBytes);
            if ( weightBytes == 0 || (weightBytes % 32u) != 0u )
                return SetError(error, "Neural-AI linebuffer weights are not 32-byte tiled");
            RefV1 modelWeights{};
            modelWeights.region = uint16_t(Region::ModelConstants);
            modelWeights.offset = weightBase;
            RefV1 stagedWeights{};
            stagedWeights.region = uint16_t(Region::TCDMScratch);
            stagedWeights.offset = weightStageOffset;
            if ( !materialized.linebufferWeightsStaged )
            {
                if ( !AppendDMA2D(modelWeights, stagedWeights, 32, 32, 32, weightBytes / 32u,
                         uint32_t(operation->Index()), 1, error) )
                    return false;
                materialized.linebufferWeightsStaged = true;
            }

            uint32_t ifmSliceOffset = 0;
            if ( !directRgb && !C32SliceByteOffset(ifm, ifmSliceOffset, error) ) return false;
            auto logicalIfm = ReshapeToNHWC(ifmShape);
            const auto logicalOfm = ReshapeToNHWC(ofmShape);
            uint32_t linebufferIfmBase = uint32_t(ifm->tensor->AllocatedAddress()) + ifmSliceOffset;
            Margin linebufferPadding = operation->Kernel()->Padding();
            if ( const SchedulerConnection *padding = operation->TryInput(TensorUsage::Params1) )
            {
                if ( !padding->tensor->IsConstant() || padding->Type() != regor::DataType::Int8 ||
                     padding->tensor->bufferView.Elements() != 1 )
                    return SetError(error, "Neural-AI Conv padding value is invalid");
                neuralai::StripeStagingInput stagingInput{};
                stagingInput.logicalIfm = logicalIfm;
                stagingInput.storageIfm = ReshapeToNHWC(ifm->tensor->storageShape);
                stagingInput.validArea = Box(logicalIfm);
                stagingInput.stagingBase = stripeStageOffset;
                stagingInput.padTop = linebufferPadding.Top();
                stagingInput.padLeft = linebufferPadding.Left();
                stagingInput.padBottom = linebufferPadding.Bottom();
                stagingInput.padRight = linebufferPadding.Right();
                stagingInput.channels = int(channelK);
                stagingInput.directNhwc = directRgb;
                const auto staging = neuralai::LinebufferPlanner().PlanStripeStaging(stagingInput);
                if ( staging.bytes > stripeStageBytes )
                    return SetError(error, "Neural-AI Conv padding exceeds its planned staging workspace");
                uint32_t stagingTile = 2;
                const int8_t paddingValue = padding->tensor->bufferView.Values<int8_t>()[0];
                if ( !AppendStripeStaging(ifm, staging, paddingValue, uint32_t(operation->Index()),
                         stagingTile, true, error) )
                    return false;
                logicalIfm = staging.stagedIfm;
                linebufferIfmBase = stripeStageOffset;
                linebufferPadding = {};
            }
            const uint32_t paddedInputChannels = uint32_t(RoundAway(int(channelK), 32));
            const uint32_t inputGroups = directRgb ? 1u : paddedInputChannels / 32u;
            const uint32_t outputGroups = uint32_t(RoundAway(int(depthN), 32)) / 32u;
            const uint32_t kernelTilesPerInputGroup =
                uint32_t(operation->Kernel()->Size().x * operation->Kernel()->Size().y);
            uint32_t tileId = 2;
            for ( uint32_t outputGroup = 0; outputGroup < outputGroups; ++outputGroup )
            {
                AppendRQLoad(qparamBase + outputGroup * 32u, outputGroup,
                    uint32_t(operation->Index()), tileId++);
                std::vector<std::vector<neuralai::LinebufferJob>> groupJobs(inputGroups);
                std::vector<std::vector<std::vector<neuralai::LinebufferJob>>> tailTapJobs(inputGroups);
                std::vector<bool> decomposeTail(inputGroups, false);
                for ( uint32_t inputGroup = 0; inputGroup < inputGroups; ++inputGroup )
                {
                    const uint32_t validLanes = directRgb ? channelK :
                        std::min(32u, channelK - inputGroup * 32u);
                    const bool decompose = !directRgb && validLanes == 16u;
                    decomposeTail[inputGroup] = decompose;
                    if ( decompose )
                    {
                        auto &tapJobs = tailTapJobs[inputGroup];
                        tapJobs.reserve(kernelTilesPerInputGroup);
                        for ( uint32_t tap = 0; tap < kernelTilesPerInputGroup; ++tap )
                        {
                            const uint32_t tapH = tap / uint32_t(operation->Kernel()->Size().x);
                            const uint32_t tapW = tap % uint32_t(operation->Kernel()->Size().x);
                            neuralai::LinebufferPlannerInput plannerInput{};
                            plannerInput.logicalIfm = logicalIfm;
                            plannerInput.logicalOfm = logicalOfm;
                            const uint32_t groupPlaneBytes = uint32_t(logicalIfm.Height()) *
                                uint32_t(logicalIfm.Width()) * 32u;
                            plannerInput.ifmBase = linebufferIfmBase +
                                inputGroup * groupPlaneBytes +
                                tapH * uint32_t(logicalIfm.Width()) * 32u + tapW * 32u;
                            plannerInput.ofmBase = uint32_t(ofm->tensor->AllocatedAddress());
                            plannerInput.weightBase = uint32_t(weightStageOffset +
                                (outputGroup * inputGroups + inputGroup) *
                                    kernelTilesPerInputGroup * 32u * 32u + tap * 32u * 32u);
                            plannerInput.psumBase = uint32_t(partialOffset);
                            plannerInput.kernelH = 1;
                            plannerInput.kernelW = 1;
                            plannerInput.strideH = operation->Kernel()->Stride().y;
                            plannerInput.strideW = operation->Kernel()->Stride().x;
                            plannerInput.ic = 16;
                            plannerInput.oc = 32;
                            plannerInput.groupIndex = 0;
                            plannerInput.inputGroupIndex = 0;
                            plannerInput.outputGroupIndex = int(outputGroup);
                            plannerInput.validLaneCount = 16;
                            plannerInput.ifmPixelStride = 32;
                            plannerInput.maxM = MaxExternalPsumLinebufferM;
                            plannerInput.tcdmBudget = ArchNeuralAI::AllocatableTCDMBytes;
                            const uint32_t firstAccumMode = inputGroups == 1u || inputGroup == 0u ? 1u : 3u;
                            plannerInput.accumMode = tap == 0u ? firstAccumMode :
                                (tap + 1u == kernelTilesPerInputGroup ? 2 : 3);
                            tapJobs.push_back(neuralai::LinebufferPlanner().Plan(plannerInput));
                        }
                        continue;
                    }
                    neuralai::LinebufferPlannerInput plannerInput{};
                    plannerInput.logicalIfm = logicalIfm;
                    plannerInput.logicalOfm = logicalOfm;
                    plannerInput.ifmBase = linebufferIfmBase;
                    plannerInput.ofmBase = uint32_t(ofm->tensor->AllocatedAddress());
                    plannerInput.weightBase = uint32_t(weightStageOffset +
                        (outputGroup * inputGroups + inputGroup) * kernelTilesPerInputGroup * 32u * 32u);
                    plannerInput.psumBase = uint32_t(partialOffset);
                    plannerInput.kernelH = operation->Kernel()->Size().y;
                    plannerInput.kernelW = operation->Kernel()->Size().x;
                    plannerInput.strideH = operation->Kernel()->Stride().y;
                    plannerInput.strideW = operation->Kernel()->Stride().x;
                    plannerInput.padTop = linebufferPadding.Top();
                    plannerInput.padLeft = linebufferPadding.Left();
                    plannerInput.padBottom = linebufferPadding.Bottom();
                    plannerInput.padRight = linebufferPadding.Right();
                    plannerInput.ic = directRgb ? int(channelK) : 32;
                    plannerInput.oc = 32;
                    plannerInput.groupIndex = int(inputGroup);
                    plannerInput.inputGroupIndex = int(inputGroup);
                    plannerInput.outputGroupIndex = int(outputGroup);
                    plannerInput.validLaneCount = int(validLanes);
                    plannerInput.ifmPixelStride = directRgb ? 3 : 32;
                    plannerInput.maxM = inputGroups == 1u ?
                        MaxDirectLinebufferM : MaxExternalPsumLinebufferM;
                    plannerInput.tcdmBudget = ArchNeuralAI::AllocatableTCDMBytes;
                    plannerInput.accumMode = inputGroups == 1u ? 0 :
                        (inputGroup == 0u ? 1 : (inputGroup + 1u == inputGroups ? 2 : 3));
                    groupJobs[inputGroup] = neuralai::LinebufferPlanner().Plan(plannerInput);
                }
                int spatialJobs = -1;
                for ( uint32_t inputGroup = 0; inputGroup < inputGroups; ++inputGroup )
                {
                    const int groupSpatialJobs = decomposeTail[inputGroup] ?
                        int(tailTapJobs[inputGroup].front().size()) : int(groupJobs[inputGroup].size());
                    if ( spatialJobs < 0 ) spatialJobs = groupSpatialJobs;
                    if ( groupSpatialJobs != spatialJobs )
                        return SetError(error, "Neural-AI linebuffer input groups have inconsistent tiling");
                    if ( decomposeTail[inputGroup] )
                    {
                        for ( const auto &jobs : tailTapJobs[inputGroup] )
                        {
                            if ( int(jobs.size()) != spatialJobs )
                                return SetError(error,
                                    "Neural-AI C16 tail taps have inconsistent spatial tiling");
                        }
                    }
                }
                for ( int jobIndex = 0; jobIndex < spatialJobs; ++jobIndex )
                {
                    for ( uint32_t inputGroup = 0; inputGroup < inputGroups; ++inputGroup )
                    {
                        if ( decomposeTail[inputGroup] )
                        {
                            for ( const auto &jobs : tailTapJobs[inputGroup] )
                                AppendLineBufferJob(jobs[jobIndex],
                                    uint32_t(operation->Index()), tileId++);
                        }
                        else
                        {
                            AppendLineBufferJob(groupJobs[inputGroup][jobIndex],
                                uint32_t(operation->Index()), tileId++);
                        }
                    }
                }
            }
            return true;
        }

        if ( mode == NeuralAIOpMode::Conv2DPointwiseC32Requant )
        {
            if ( ifm->tensor->format != TensorFormat::C32Blocked ||
                 ofm->tensor->format != TensorFormat::C32Blocked )
                return SetError(error, "Neural-AI pointwise Conv requires C32-blocked input and output");
            const uint64_t groupStride64 = uint64_t(rows) * 32u;
            if ( groupStride64 > std::numeric_limits<uint32_t>::max() )
                return SetError(error, "Neural-AI pointwise Conv row stride overflows the ABI");
            const uint32_t groupStride = uint32_t(groupStride64);
            uint32_t ifmSliceOffset = 0;
            if ( !C32SliceByteOffset(ifm, ifmSliceOffset, error) ) return false;
            uint32_t tileId = 0;
            for ( uint32_t nGroup = 0; nGroup < nGroups; ++nGroup )
            {
                AppendRQLoad(qparamBase + nGroup * 32, nGroup,
                    uint32_t(operation->Index()), tileId++);
                const uint64_t weightOffset64 = uint64_t(weightBase) +
                    uint64_t(nGroup) * kGroups * 32u * 32u;
                const uint64_t outputOffset64 = uint64_t(nGroup) * groupStride64;
                if ( weightOffset64 > std::numeric_limits<uint32_t>::max() ||
                     outputOffset64 > std::numeric_limits<uint32_t>::max() )
                    return SetError(error, "Neural-AI pointwise Conv tensor reference overflows the ABI");
                RefV1 ifmRef = TensorRef(ifm->tensor.get(), ifmSliceOffset, error);
                if ( !error.empty() ) return false;
                RefV1 weights{};
                weights.region = uint16_t(Region::ModelConstants);
                weights.offset = uint32_t(weightOffset64);
                RefV1 partial{};
                if ( kGroups > 1 )
                {
                    partial.region = uint16_t(Region::TCDMScratch);
                    partial.offset = partialOffset;
                }
                RefV1 output = TensorRef(ofm->tensor.get(), uint32_t(outputOffset64), error);
                if ( !error.empty() ) return false;
                AppendPointwiseC32(weights, ifmRef, partial, output, rows,
                    kGroups, 1, nGroup, groupStride, groupStride,
                    uint32_t(operation->Index()), tileId++);
            }
            return true;
        }

        if ( ifm->slice )
            return SetError(error, "Neural-AI FC and MatMul do not support sliced inputs");

        uint32_t tileId = 0;
        for ( uint32_t nGroup = 0; nGroup < nGroups; ++nGroup )
        {
            AppendRQLoad(qparamBase + nGroup * 32, nGroup,
                uint32_t(operation->Index()), tileId++);
            for ( uint32_t rowBase = 0; rowBase < rows; rowBase += 256 )
            {
                const uint32_t dimM = std::min<uint32_t>(256, rows - rowBase);
                for ( uint32_t kGroup = 0; kGroup < kGroups; ++kGroup )
                {
                    RefV1 ifmRef = TensorRef(ifm->tensor.get(), rowBase * paddedK + kGroup * 32, error);
                    if ( !error.empty() ) return false;
                    if ( paddedK != 32 )
                    {
                        RefV1 staged{};
                        staged.region = uint16_t(Region::TCDMScratch);
                        staged.offset = stageOffset;
                        if ( !AppendDMA2D(ifmRef, staged, 32, paddedK, 32, dimM,
                                 uint32_t(operation->Index()), tileId++, error) )
                            return false;
                        ifmRef = staged;
                    }
                    RefV1 weights{};
                    weights.region = uint16_t(Region::ModelConstants);
                    weights.offset = weightBase + (nGroup * kGroups + kGroup) * 32 * 32;
                    RefV1 partial{};
                    partial.region = uint16_t(Region::TCDMScratch);
                    partial.offset = partialOffset;
                    RefV1 output = partial;
                    CommandType type = CommandType::Gemm32;
                    uint32_t outputStride = 32 * 4;
                    if ( kGroup != 0 ) type = CommandType::Gemm32Accum;
                    if ( kGroup + 1 == kGroups )
                    {
                        type = CommandType::Gemm32Requant;
                        output = TensorRef(ofm->tensor.get(), rowBase * paddedN + nGroup * 32, error);
                        if ( !error.empty() ) return false;
                        outputStride = paddedN;
                        if ( kGroup == 0 ) partial = {};
                    }
                    AppendGEMM(type, weights, ifmRef, partial, output, dimM, outputStride,
                        nGroup, uint32_t(operation->Index()), tileId++);
                }
            }
        }
        return true;
    }

    bool BuildBindings(const std::vector<std::unique_ptr<SchedulerOperation>> &operations, std::string &error)
    {
        std::unordered_map<UniqueId, const SchedulerConnection *> connections;
        std::unordered_map<UniqueId, const SchedulerTensor *> tensors;
        for ( const auto &operation : operations )
        {
            for ( const auto &[usage, connection] : operation->inputs.pairs() )
            {
                UNUSED(usage);
                if ( connection.tensor->srcTensor )
                    connections.emplace(connection.tensor->srcTensor->Uid(), &connection);
                tensors.emplace(connection.tensor->uid, connection.tensor.get());
            }
            for ( const auto &[usage, connection] : operation->outputs.pairs() )
            {
                UNUSED(usage);
                if ( connection.tensor->srcTensor )
                    connections.emplace(connection.tensor->srcTensor->Uid(), &connection);
                tensors.emplace(connection.tensor->uid, connection.tensor.get());
            }
        }

        auto addBindings = [&](const auto &graphTensors, BindingDirection direction,
                               std::unordered_map<UniqueId, uint16_t> &indices) -> bool
        {
            for ( int index = 0; index < int(graphTensors.size()); ++index )
            {
                const auto &tensor = graphTensors[index];
                auto position = connections.find(tensor->Uid());
                if ( position == connections.end() ) return SetError(error, "Neural-AI graph binding is unscheduled");
                const SchedulerConnection *connection = position->second;
                const auto dimensions = Dimensions(tensor->StorageShape());
                const uint16_t dataType = ABIDataType(tensor->Type());
                if ( dataType == 0 ) return SetError(error, "Neural-AI public binding has an unsupported data type");
                neuralai::BindingV1 binding{};
                binding.direction = uint16_t(direction);
                binding.index = uint16_t(index);
                binding.dataType = dataType;
                binding.layout = uint16_t(TensorLayout::NHWC);
                binding.rank = 4;
                binding.tensorId = nextTensorId++;
                std::copy(dimensions.begin(), dimensions.end(), binding.dimensions);
                const uint32_t elementBytes = dataType == uint16_t(DataType::Int32) ? 4 : 1;
                binding.byteSize = uint32_t(tensor->StorageShape().Elements64()) * elementBytes;
                BindingQuantization(tensor.get(), connection, binding.scaleBits, binding.zeroPoint);
                artifact->bindings.push_back(binding);
                indices.emplace(tensor->Uid(), uint16_t(index));

                neuralai::TensorV1 description{};
                description.tensorId = binding.tensorId;
                description.dataType = binding.dataType;
                description.layout = binding.layout;
                description.rank = binding.rank;
                std::copy(dimensions.begin(), dimensions.end(), description.dimensions);
                description.byteSize = binding.byteSize;
                description.alignment = 1u;  // Public bindings are compact NHWC.
                artifact->tensors.push_back(description);
            }
            return true;
        };

        if ( !addBindings(graph->Inputs(), BindingDirection::Input, inputBindings) ) return false;
        if ( !addBindings(graph->Outputs(), BindingDirection::Output, outputBindings) ) return false;

        std::vector<std::pair<UniqueId, const SchedulerTensor *>> orderedTensors(tensors.begin(), tensors.end());
        std::sort(orderedTensors.begin(), orderedTensors.end(),
            [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
        for ( const auto &[uid, tensor] : orderedTensors )
        {
            UNUSED(uid);
            if ( tensor->isGraphInput || tensor->isGraphOutput || tensor->IsConstant() ) continue;
            const uint16_t dataType = ABIDataType(tensor->dataType);
            const uint16_t layout = ABILayout(tensor->format);
            if ( dataType == 0 || layout == 0 )
                return SetError(error, "Neural-AI internal tensor metadata is unsupported");
            neuralai::TensorV1 description{};
            description.tensorId = nextTensorId++;
            description.dataType = dataType;
            description.layout = layout;
            description.rank = 4;
            const auto dimensions = Dimensions(tensor->storageShape);
            std::copy(dimensions.begin(), dimensions.end(), description.dimensions);
            description.byteSize = uint32_t(TensorStorageBytes(tensor));
            description.alignment = tensor->format == TensorFormat::NHWC ? 1u : ArchNeuralAI::DMAAlignment;
            description.scratchOffset = uint32_t(tensor->AllocatedAddress());
            artifact->tensors.push_back(description);
        }
        return true;
    }
};

}  // namespace

bool NeuralAICommandGenerator::Generate(const Graph *graph,
    const std::vector<std::unique_ptr<SchedulerOperation>> &operations, const Schedule *schedule,
    CompiledNeuralAIArtifact &artifact, std::string &error)
{
    if ( graph == nullptr || schedule == nullptr )
    {
        error = "Neural-AI command generation requires a graph and schedule";
        return false;
    }
    artifact = {};
    GeneratorContext context(graph, schedule, &artifact);
    if ( !context.BuildBindings(operations, error) ) return false;
    HLCStreamGenerator hlcGenerator(0, false);
    HLCStream hlcCommands = hlcGenerator.GenerateCommandStream(
        vector_span<std::unique_ptr<SchedulerOperation>>(operations, 0, int(operations.size())),
        schedule, true);
    std::unordered_map<UniqueId, const SchedulerOperation *> operationsByUid;
    for ( const auto &operation : operations ) operationsByUid.emplace(operation->Uid(), operation.get());

    uint32_t scratchBytes = 0;
    for ( const auto &operation : operations )
    {
        for ( const auto &[usage, connection] : operation->inputs.pairs() )
        {
            UNUSED(usage);
            if ( !connection.tensor->isGraphInput && !connection.tensor->isGraphOutput &&
                 !connection.tensor->IsConstant() )
                scratchBytes = std::max(scratchBytes, uint32_t(connection.tensor->AllocatedAddress() +
                    TensorStorageBytes(connection.tensor.get())));
        }
        for ( const auto &[usage, connection] : operation->outputs.pairs() )
        {
            UNUSED(usage);
            if ( !connection.tensor->isGraphInput && !connection.tensor->isGraphOutput )
                scratchBytes = std::max(scratchBytes, uint32_t(connection.tensor->AllocatedAddress() +
                    TensorStorageBytes(connection.tensor.get())));
        }
    }
    context.scratchEnd = uint32_t(RoundAway(int(scratchBytes), ArchNeuralAI::DMAAlignment));
    uint32_t linebufferWeightBytes = 0;
    for ( const auto &operation : operations )
    {
        if ( operation->Type() == OpType::MemoryCopy &&
             operation->IFM(0)->tensor->format == TensorFormat::NHWC &&
             operation->OFM()->tensor->format == TensorFormat::NHWC )
        {
            const uint32_t elementBytes = operation->OFM()->Type() == regor::DataType::Int32 ? 4u : 1u;
            const int64_t bytes = operation->OFM()->shape.Elements64() * elementBytes;
            if ( bytes <= 0 || bytes > std::numeric_limits<uint32_t>::max() )
            {
                error = "Neural-AI compact copy staging size is invalid";
                return false;
            }
            context.stageBytes = std::max(context.stageBytes, uint32_t(bytes));
        }
        if ( operation->Type() != OpType::FullyConnected && operation->Type() != OpType::MatMul &&
             operation->Type() != OpType::Conv2D && operation->Type() != OpType::DepthwiseConv2D ) continue;
        const SchedulerOpInfo *cost = schedule->Cost(operation.get());
        if ( cost == nullptr || cost->Config() == nullptr )
        {
            error = "Neural-AI scheduled operation has no validated target mode";
            return false;
        }
        const auto *config = static_cast<const NeuralAIOpConfig *>(cost->Config());
        const NeuralAIOpMode mode = config->Mode();
        const Shape ifmShape = operation->IFM(0)->SliceShape();
        const Shape ofmShape = operation->OFM()->SliceShape();
        const uint32_t rows = uint32_t(ofmShape.Elements64() / ofmShape.Depth());
        const bool isK3Conv = IsLinebufferConvMode(mode);
        const uint32_t logicalK = uint32_t(ifmShape.Depth()) * (isK3Conv ? 9u : 1u);
        const uint32_t paddedK = uint32_t(RoundAway(int(logicalK), 32));
        const uint32_t stripeRows = std::min<uint32_t>(rows, 256);
        if ( paddedK != 32 ) context.stageBytes = std::max(context.stageBytes, stripeRows * 32);
        if ( paddedK > 32 ) context.partialBytes = std::max(context.partialBytes, stripeRows * 32 * 4);
        if ( isK3Conv )
        {
            const bool directRgb = config->DirectNhwcInput();
            if ( directRgb || paddedK > 0 )
            {
                const uint32_t paddedN = uint32_t(RoundAway(int(ofmShape.Depth()), 32));
                const uint32_t kGroups = directRgb ? 1u :
                    9u * uint32_t(RoundAway(int(ifmShape.Depth()), 32)) / 32u;
                linebufferWeightBytes = std::max(linebufferWeightBytes,
                    uint32_t(kGroups * (paddedN / 32u) * 32u * 32u));
            }
            context.partialBytes = std::max(context.partialBytes, stripeRows * 32 * 4);
        }
    }
    context.stageBytes = std::max(context.stageBytes, linebufferWeightBytes);
    context.stageOffset = context.scratchEnd;
    context.weightStageOffset = context.stageOffset;
    context.partialOffset = uint32_t(RoundAway(
        int(context.stageOffset + context.stageBytes), ArchNeuralAI::DMAAlignment));
    context.stripeStageOffset = uint32_t(RoundAway(
        int(context.partialOffset + context.partialBytes), ArchNeuralAI::DMAAlignment));
    for ( const auto &command : hlcCommands )
    {
        if ( command->CommandType() != HighLevelCommandType::STRIPE ) continue;
        const auto *stripe = static_cast<const HLCStripe *>(command.get());
        auto operationPosition = operationsByUid.find(stripe->operation->srcId);
        if ( operationPosition == operationsByUid.end() ) continue;
        const SchedulerOperation *operation = operationPosition->second;
        const SchedulerOpInfo *cost = schedule->Cost(operation);
        const auto *config = cost != nullptr && cost->Config() != nullptr ?
            static_cast<const NeuralAIOpConfig *>(cost->Config()) : nullptr;
        if ( cost == nullptr || cost->cascade == 0 || operation->Type() != OpType::Conv2D ||
             config == nullptr || !IsLinebufferConvMode(config->Mode()) || operation->IFM(0)->shape.Depth() > 32 )
            continue;
        const StripeArea &area = stripe->stripeAreas.front();
        neuralai::StripeStagingInput stagingInput{};
        stagingInput.logicalIfm = ReshapeToNHWC(operation->IFM(0)->shape);
        stagingInput.storageIfm = ReshapeToNHWC(operation->IFM(0)->tensor->storageShape);
        stagingInput.validArea = area.ifmAreas[0];
        stagingInput.padTop = stripe->padding.top;
        stagingInput.padLeft = stripe->padding.left;
        stagingInput.padBottom = stripe->padding.bottom;
        stagingInput.padRight = stripe->padding.right;
        stagingInput.channels = operation->IFM(0)->shape.Depth();
        stagingInput.directNhwc = config->DirectNhwcInput();
        const auto staging = neuralai::LinebufferPlanner().PlanStripeStaging(stagingInput);
        context.stripeStageBytes = std::max(context.stripeStageBytes, staging.bytes);
    }
    for ( const auto &operation : operations )
    {
        const SchedulerOpInfo *cost = schedule->Cost(operation.get());
        const auto *config = cost != nullptr && cost->Config() != nullptr ?
            static_cast<const NeuralAIOpConfig *>(cost->Config()) : nullptr;
        if ( cost == nullptr || cost->cascade != 0 || operation->Type() != OpType::Conv2D ||
             config == nullptr || !IsLinebufferConvMode(config->Mode()) ||
             operation->TryInput(TensorUsage::Params1) == nullptr )
            continue;
        const SchedulerConnection *ifm = operation->IFM(0);
        const Shape logicalIfm = ReshapeToNHWC(ifm->shape);
        const Margin &padding = operation->Kernel()->Padding();
        neuralai::StripeStagingInput stagingInput{};
        stagingInput.logicalIfm = logicalIfm;
        stagingInput.storageIfm = ReshapeToNHWC(ifm->tensor->storageShape);
        stagingInput.validArea = Box(logicalIfm);
        stagingInput.padTop = padding.Top();
        stagingInput.padLeft = padding.Left();
        stagingInput.padBottom = padding.Bottom();
        stagingInput.padRight = padding.Right();
        stagingInput.channels = logicalIfm.Depth();
        stagingInput.directNhwc = config->DirectNhwcInput();
        const auto staging = neuralai::LinebufferPlanner().PlanStripeStaging(stagingInput);
        context.stripeStageBytes = std::max(context.stripeStageBytes, staging.bytes);
    }

    for ( const auto &operation : operations )
    {
        const SchedulerOpInfo *operationCost = schedule->Cost(operation.get());
        if ( operationCost != nullptr && operationCost->cascade != 0 )
        {
            const CascadeInfo *cascade = schedule->Cascade(operationCost->cascade);
            if ( cascade == nullptr )
            {
                error = "Neural-AI scheduled operation references a missing cascade";
                return false;
            }
            if ( operation->Index() != cascade->start ) continue;
            std::unordered_map<UniqueId, uint32_t> tileIds;
            for ( const auto &command : hlcCommands )
            {
                if ( command->CommandType() != HighLevelCommandType::STRIPE ) continue;
                const auto *stripe = static_cast<const HLCStripe *>(command.get());
                auto operationPosition = operationsByUid.find(stripe->operation->srcId);
                if ( operationPosition == operationsByUid.end() ) continue;
                const SchedulerOperation *stripeOperation = operationPosition->second;
                const SchedulerOpInfo *stripeCost = schedule->Cost(stripeOperation);
                if ( stripeCost == nullptr || stripeCost->cascade != operationCost->cascade ) continue;
                uint32_t &tileId = tileIds[stripeOperation->Uid()];
                if ( stripeOperation->Type() == OpType::Conv2D )
                {
                    if ( !context.AppendC32MatrixStripe(stripeOperation, stripe, tileId, error) )
                        return false;
                }
                else if ( stripeOperation->Type() == OpType::LUT )
                {
                    if ( !context.AppendAFULutStripe(stripeOperation, stripe, tileId, error) ) return false;
                }
                else if ( stripeOperation->Type() == OpType::Concat )
                {
                    if ( !context.AppendConcatStripe(stripeOperation, stripe, tileId, error) ) return false;
                }
                else
                {
                    error = "Neural-AI cascade contains an unsupported striped operation " +
                        OpTypeToString(stripeOperation->Type());
                    return false;
                }
            }
            continue;
        }
        if ( operation->Type() == OpType::MemoryCopy )
        {
            if ( !context.AppendCopy(operation.get(), error) ) return false;
        }
        else if ( operation->Type() == OpType::DepthwiseConv2D )
        {
            if ( !context.AppendDepthwise(operation.get(), error) ) return false;
        }
        else if ( operation->Type() == OpType::Add || operation->Type() == OpType::Sub )
        {
            if ( !context.AppendAdd(operation.get(), error) ) return false;
        }
        else if ( operation->Type() == OpType::LUT )
        {
            if ( !context.AppendAFULut(operation.get(), error) ) return false;
        }
        else if ( operation->Type() == OpType::AvgPool )
        {
            const SchedulerOpInfo *cost = schedule->Cost(operation.get());
            const auto *config = cost != nullptr && cost->Config() != nullptr ?
                static_cast<const NeuralAIOpConfig *>(cost->Config()) : nullptr;
            if ( config != nullptr && config->Mode() == NeuralAIOpMode::UpsampleNearestC32 )
            {
                if ( !context.AppendUpsampleNearest(operation.get(), error) ) return false;
            }
            else if ( !context.AppendAFUGlobalAvgPool(operation.get(), error) ) return false;
        }
        else if ( operation->Type() == OpType::MaxPool )
        {
            if ( !context.AppendMaxPool(operation.get(), error) ) return false;
        }
        else if ( operation->Type() == OpType::Concat )
        {
            if ( !context.AppendConcat(operation.get(), error) ) return false;
        }
        else if ( operation->Type() == OpType::Transpose )
        {
            if ( !context.AppendHeadPack(operation.get(), error) ) return false;
        }
        else if ( operation->Type() == OpType::Dfl )
        {
            if ( !context.AppendDFL16(operation.get(), error) ) return false;
        }
        else if ( operation->Type() == OpType::Passthrough )
        {
            if ( !context.AppendPassthroughConcat(operation.get(), error) ) return false;
        }
        else if ( operation->Type() == OpType::FullyConnected || operation->Type() == OpType::MatMul ||
                  operation->Type() == OpType::Conv2D )
        {
            if ( !context.AppendMatrix(operation.get(), error) ) return false;
        }
        else
        {
            const SchedulerConnection *ofm = operation->OFM();
            std::string inputs;
            for ( const auto &[usage, connection] : operation->inputs.pairs() )
            {
                inputs += fmt::format(" usage{} '{}' [{}] {} addr{};", int(usage),
                    connection.tensor->Name(), connection.SliceShape().ToString(),
                    EnumToString(connection.tensor->format), connection.tensor->AllocatedAddress());
            }
            error = fmt::format(
                "Neural-AI command generation encountered unsupported operation {} {} "
                "(inputs:{} OFM '{}' [{}] {} addr{})",
                OpTypeToString(operation->Type()), operation->Index(),
                inputs, ofm == nullptr ? "missing" : ofm->tensor->Name(),
                ofm == nullptr ? "missing" : ofm->SliceShape().ToString(),
                ofm == nullptr ? "missing" : EnumToString(ofm->tensor->format),
                ofm == nullptr ? -1 : ofm->tensor->AllocatedAddress());
            return false;
        }
    }
    context.AppendControl(CommandType::End, 0, 0);
    artifact.requiredTCDMBytes = uint32_t(RoundAway(
        int(context.stripeStageOffset + context.stripeStageBytes), ArchNeuralAI::DMAAlignment));
    if ( artifact.requiredTCDMBytes > ArchNeuralAI::AllocatableTCDMBytes )
    {
        error = fmt::format("Neural-AI generated command workspace {} exceeds allocatable TCDM {}",
            artifact.requiredTCDMBytes, ArchNeuralAI::AllocatableTCDMBytes);
        return false;
    }
    return true;
}

}  // namespace regor
