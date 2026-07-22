//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "neural_ai_command_generator.hpp"

#include "common/numeric_util.hpp"

#include "architecture/neuralai/neural_ai.hpp"
#include "compiler/shape_util.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace regor
{
namespace
{

using neuralai::BindingDirection;
using neuralai::CommandType;
using neuralai::CopyLayoutMode;
using neuralai::DataType;
using neuralai::RefV1;
using neuralai::Region;
using neuralai::TensorLayout;

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
    if ( format == TensorFormat::NHWC ) return uint16_t(TensorLayout::NHWC);
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

struct GeneratorContext
{
    const Graph *graph;
    const Schedule *schedule;
    CompiledNeuralAIArtifact *artifact;
    std::unordered_map<UniqueId, uint16_t> inputBindings;
    std::unordered_map<UniqueId, uint16_t> outputBindings;
    uint32_t nextTensorId = 0;
    uint32_t scratchEnd = 0;
    uint32_t stageOffset = 0;
    uint32_t partialOffset = 0;
    uint32_t stageBytes = 0;
    uint32_t partialBytes = 0;

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

    void AppendRQLoad(uint32_t qparamIndex, uint32_t layerId, uint32_t tileId)
    {
        AppendHeader(artifact->commands, CommandType::RQLoad, 32, layerId, tileId);
        Append32(artifact->commands, qparamIndex);
        Append32(artifact->commands, 32);
        Append32(artifact->commands, 0);
        Append32(artifact->commands, 0);
        ++artifact->commandCount;
    }

    void AppendDMA2D(const RefV1 &source, const RefV1 &destination, uint32_t sourceStride,
        uint32_t repetitions, uint32_t layerId, uint32_t tileId)
    {
        AppendHeader(artifact->commands, CommandType::DMA2D, 64, layerId, tileId);
        AppendRef(artifact->commands, source);
        AppendRef(artifact->commands, destination);
        Append32(artifact->commands, 32);
        Append32(artifact->commands, sourceStride);
        Append32(artifact->commands, 32);
        Append32(artifact->commands, repetitions);
        Append32(artifact->commands, 2);
        AppendZeros(artifact->commands, 3);
        ++artifact->commandCount;
    }

    void AppendGEMM(CommandType type, const RefV1 &weights, const RefV1 &ifm,
        const RefV1 &partial, const RefV1 &ofm, uint32_t dimM, uint32_t ofmStride,
        uint32_t layerId, uint32_t tileId)
    {
        AppendHeader(artifact->commands, type, 96, layerId, tileId);
        AppendRef(artifact->commands, weights);
        AppendRef(artifact->commands, ifm);
        AppendRef(artifact->commands, partial);
        AppendRef(artifact->commands, ofm);
        Append32(artifact->commands, dimM);
        Append32(artifact->commands, ofmStride);
        Append32(artifact->commands, 32 * 4);
        Append32(artifact->commands, 0);
        AppendZeros(artifact->commands, 8);
        ++artifact->commandCount;
    }

    bool AppendCopy(const SchedulerOperation *operation, std::string &error)
    {
        const SchedulerConnection *ifm = operation->IFM(0);
        const SchedulerConnection *ofm = operation->OFM();
        CopyLayoutMode mode;
        if ( ifm->tensor->format == TensorFormat::NHWC && ofm->tensor->format == TensorFormat::Row32 )
            mode = CopyLayoutMode::NHWCToRow32;
        else if ( ifm->tensor->format == TensorFormat::Row32 && ofm->tensor->format == TensorFormat::NHWC )
            mode = CopyLayoutMode::Row32ToNHWC;
        else return SetError(error, "Neural-AI MemoryCopy requires an NHWC/ROW32 boundary");

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
        Append32(artifact->commands, mode == CopyLayoutMode::NHWCToRow32 ? compactStride : nativeStride);
        Append32(artifact->commands, mode == CopyLayoutMode::NHWCToRow32 ? nativeStride : compactStride);
        AppendZeros(artifact->commands, 7);
        ++artifact->commandCount;
        return true;
    }

    bool AppendMatrix(const SchedulerOperation *operation, std::string &error)
    {
        const SchedulerOpInfo *cost = schedule->Cost(operation);
        if ( cost == nullptr || cost->npuWeightsTensor == nullptr )
            return SetError(error, "Neural-AI matrix operation has no encoded constant tensor");
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
        const uint32_t depthK = uint32_t(ifm->shape.Depth());
        const uint32_t depthN = uint32_t(ofm->shape.Depth());
        const uint32_t rows = uint32_t(ofm->shape.Elements64() / depthN);
        const uint32_t paddedK = uint32_t(RoundAway(int(depthK), 32));
        const uint32_t paddedN = uint32_t(RoundAway(int(depthN), 32));
        const uint32_t kGroups = paddedK / 32;
        const uint32_t nGroups = paddedN / 32;
        if ( rows == 0 || range.scaleBytes != int(paddedN * sizeof(neuralai::QParamV1)) ||
             range.weightBytes != int(kGroups * nGroups * 32 * 32) )
            return SetError(error, "Neural-AI encoded matrix dimensions do not match the scheduled operation");

        const uint32_t qparamBase = uint32_t(artifact->qparams.size());
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
        const uint32_t weightBase = uint32_t(artifact->constants.size());
        const uint8_t *weightData = data + range.weightOffset;
        artifact->constants.insert(artifact->constants.end(), weightData, weightData + range.weightBytes);

        uint32_t tileId = 0;
        for ( uint32_t nGroup = 0; nGroup < nGroups; ++nGroup )
        {
            AppendRQLoad(qparamBase + nGroup * 32, uint32_t(operation->Index()), tileId++);
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
                        AppendDMA2D(ifmRef, staged, paddedK, dimM, uint32_t(operation->Index()), tileId++);
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
                        uint32_t(operation->Index()), tileId++);
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
                const auto dimensions = Dimensions(connection->shape);
                const uint16_t dataType = ABIDataType(connection->Type());
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
                binding.byteSize = uint32_t(connection->shape.Elements64()) * elementBytes;
                binding.scaleBits = FloatBits(float(connection->quantization.Scale().Dequantize()));
                binding.zeroPoint = connection->quantization.zeroPoints.empty() ? 0 :
                    int32_t(connection->quantization.zeroPoints[0]);
                artifact->bindings.push_back(binding);
                indices.emplace(tensor->Uid(), uint16_t(index));

                neuralai::TensorV1 description{};
                description.tensorId = binding.tensorId;
                description.dataType = binding.dataType;
                description.layout = binding.layout;
                description.rank = binding.rank;
                std::copy(dimensions.begin(), dimensions.end(), description.dimensions);
                description.byteSize = binding.byteSize;
                description.alignment = ArchNeuralAI::DMAAlignment;
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
            description.byteSize = uint32_t(tensor->AllocationSizeBytes());
            description.alignment = ArchNeuralAI::DMAAlignment;
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

    uint32_t scratchBytes = 0;
    for ( const auto &operation : operations )
    {
        for ( const auto &[usage, connection] : operation->inputs.pairs() )
        {
            UNUSED(usage);
            if ( !connection.tensor->isGraphInput && !connection.tensor->isGraphOutput &&
                 !connection.tensor->IsConstant() )
                scratchBytes = std::max(scratchBytes, uint32_t(connection.tensor->AllocatedAddress() +
                    connection.tensor->AllocationSizeBytes()));
        }
        for ( const auto &[usage, connection] : operation->outputs.pairs() )
        {
            UNUSED(usage);
            if ( !connection.tensor->isGraphInput && !connection.tensor->isGraphOutput )
                scratchBytes = std::max(scratchBytes, uint32_t(connection.tensor->AllocatedAddress() +
                    connection.tensor->AllocationSizeBytes()));
        }
    }
    context.scratchEnd = uint32_t(RoundAway(int(scratchBytes), ArchNeuralAI::DMAAlignment));
    for ( const auto &operation : operations )
    {
        if ( operation->Type() != OpType::FullyConnected && operation->Type() != OpType::MatMul ) continue;
        const uint32_t rows = uint32_t(operation->OFM()->shape.Elements64() / operation->OFM()->shape.Depth());
        const uint32_t paddedK = uint32_t(RoundAway(operation->IFM(0)->shape.Depth(), 32));
        const uint32_t stripeRows = std::min<uint32_t>(rows, 256);
        if ( paddedK != 32 ) context.stageBytes = std::max(context.stageBytes, stripeRows * 32);
        if ( paddedK > 32 ) context.partialBytes = std::max(context.partialBytes, stripeRows * 32 * 4);
    }
    context.stageOffset = context.scratchEnd;
    context.partialOffset = uint32_t(RoundAway(
        int(context.stageOffset + context.stageBytes), ArchNeuralAI::DMAAlignment));

    for ( const auto &operation : operations )
    {
        if ( operation->Type() == OpType::MemoryCopy )
        {
            if ( !context.AppendCopy(operation.get(), error) ) return false;
        }
        else if ( operation->Type() == OpType::FullyConnected || operation->Type() == OpType::MatMul )
        {
            if ( !context.AppendMatrix(operation.get(), error) ) return false;
        }
        else
        {
            error = "Neural-AI command generation encountered unsupported operation " +
                OpTypeToString(operation->Type());
            return false;
        }
    }
    context.AppendControl(CommandType::End, 0, 0);
    artifact.requiredTCDMBytes = uint32_t(RoundAway(
        int(context.partialOffset + context.partialBytes), ArchNeuralAI::DMAAlignment));
    if ( artifact.requiredTCDMBytes > ArchNeuralAI::AllocatableTCDMBytes )
    {
        error = "Neural-AI generated command workspace exceeds allocatable TCDM";
        return false;
    }
    return true;
}

}  // namespace regor
