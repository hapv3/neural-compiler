//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/buffer_view.hpp"
#include "compiler/scheduler.hpp"
#include "compiler/scheduler_operation.hpp"
#include "compiler/tensor.hpp"

#include <memory>
#include <vector>

namespace regor
{

class DriverActions
{
public:
    DriverActions(const DriverActions &) = delete;  // Never constructed. Static members only.

    static std::unique_ptr<Buffer> CreateDriverPayload(
        const std::vector<uint32_t> &registerCommandStream, uint32_t archConfigWord, uint32_t archVersion)
    {
        if ( registerCommandStream.size() > (1 << 22) )
        {
            double sizeMB = 4.0 * double(registerCommandStream.size()) / (1 << 20);
            throw std::runtime_error(fmt::format(
                "The command stream exceeds the hardware limit of 16 MiB (current size: {:.2f} MiB)", sizeMB));
        }

        std::vector<uint8_t> payload;
        payload.reserve(4 * (registerCommandStream.size() + 8));

        // FourCC
        Emit(payload, 0x31, 0x50, 0x4F, 0x43);  // u8"COP1"

        // Config
        Emit(payload, OptimizerConfigWord(0, 1));  // Optimizer version 0.1
        Emit(payload, archConfigWord);
        Emit(payload, archVersion);  // Architecture version

        // Header
        while ( (payload.size() + 4) % 16 )
        {
            // Insert NOPs to align start of command stream to 16 bytes
            Emit(payload, MakeTag(Command::NOP, 0, 0));
        }
        const auto length_high = registerCommandStream.size() >> 16;
        const auto length_low = registerCommandStream.size() & 0xFFFF;
        Emit(payload, MakeTag(Command::CMDSTRM, uint8_t(length_high), uint16_t(length_low)));

        // Command stream
        for ( const auto &registerCommand : registerCommandStream )
        {
            Emit(payload, registerCommand);
        }

        return std::make_unique<Buffer>(std::move(payload));
    }

private:
    enum class Command : uint8_t
    {
        OPTCFG = 0x01,
        CMDSTRM = 0x02,
        NOP = 0x05,
    };

    static constexpr uint32_t MakeTag(Command id, uint8_t reserved, uint16_t param)
    {
        return unsigned(id) | (reserved << 8) | (param << 16);
    }

    static constexpr uint32_t OptimizerConfigWord(int release, int patch)
    {
        return MakeTag(Command::OPTCFG, 0, uint16_t(release | (patch << 4)));
    }

    static void Emit(std::vector<uint8_t> &data, uint8_t byte3, uint8_t byte2, uint8_t byte1, uint8_t byte0)
    {
        data.insert(data.end(), {byte0, byte1, byte2, byte3});
    }

    static void Emit(std::vector<uint8_t> &data, uint32_t word)
    {
        data.insert(data.end(), {uint8_t(word), uint8_t(word >> 8), uint8_t(word >> 16), uint8_t(word >> 24)});
    }
};

class CustomOperatorBuilder
{
private:
    const Schedule *_schedule;
    uint32_t _archConfigWord;
    uint32_t _archVersion;
    std::shared_ptr<Tensor> _featureMapTensor;
    std::shared_ptr<Tensor> _stagingTensor;
    std::shared_ptr<Tensor> _readOnlyTensor;
    std::shared_ptr<Buffer> _readOnlyBuffer;

public:
    CustomOperatorBuilder(Architecture *architecture, const Schedule *schedule)
    {
        _schedule = schedule;
        _archConfigWord = architecture->ConfigRegisters().front();
        _archVersion = architecture->Version();
        _featureMapTensor = std::make_shared<Tensor>("scratch", DataType::UInt8);
        _readOnlyTensor = std::make_shared<Tensor>("read_only", DataType::UInt8);
        _featureMapTensor->SetStorageShape(Shape(schedule->memoryUsage.at(architecture->FeatureMapMemory())));

        if ( architecture->StagingMemory() == architecture->FeatureMapMemory() )
        {
            _stagingTensor = _featureMapTensor;
        }
        else
        {
            _stagingTensor = std::make_shared<Tensor>("scratch_fast", DataType::UInt8);
            _stagingTensor->SetStorageShape(Shape(schedule->memoryUsage.at(architecture->StagingMemory())));
        }

        const auto readOnlySize = schedule->memoryUsage.at(architecture->ReadonlyMemory());
        std::vector<uint8_t> readOnlyBuffer(readOnlySize);
        _readOnlyBuffer = std::make_shared<Buffer>(std::move(readOnlyBuffer));
        _readOnlyTensor->SetStorageShape(Shape(readOnlySize));
        _readOnlyTensor->SetBuffer(_readOnlyBuffer);
    }

    void AllocateScratchTensors(std::unordered_map<const Tensor *, Address> &tensorAddressMap)
    {
        tensorAddressMap[_featureMapTensor.get()] = 0;
        if ( _featureMapTensor.get() != _stagingTensor.get() )
        {
            tensorAddressMap[_stagingTensor.get()] = 0;
        }
    }

    void Serialise(Operation *operation, const NPUOperation *npuOp, const std::vector<uint32_t> &registerCommandStream)
    {
        operation->ConnectInput(TensorUsage::Params, CreateCommandStreamTensor(registerCommandStream));
        operation->ConnectInput(MakeTensorUsage(TensorUsage::Params, 1), _readOnlyTensor);
        operation->ConnectInput(TensorUsage::State, _featureMapTensor);
        operation->ConnectInput(MakeTensorUsage(TensorUsage::State, 1), _stagingTensor);

        for ( const auto &op : npuOp->Operations() )
        {
            const auto cost = _schedule->Cost(op.get());
            for ( const auto &input : op->inputs.pairs() )
            {
                if ( input.second.tensor->IsConstant() )
                {
                    if ( input.first == TensorUsage::Weights )
                    {
                        AddToReadOnly(cost->npuWeightsTensor.get());
                    }
                    else if ( input.first == TensorUsage::Scales )
                    {
                        AddToReadOnly(cost->npuScalesTensor.get());
                    }
                    else
                    {
                        auto tensor = input.second.tensor.get();
                        if ( tensor->AllocatedAddress() >= 0 )
                        {
                            AddToReadOnly(tensor);
                        }
                    }
                }
            }
            if ( cost->npuScalesTensor && !op->inputs.contains(TensorUsage::Scales) )
            {
                AddToReadOnly(cost->npuScalesTensor.get());
            }
            for ( const auto &subOp : op->SubOps() )
            {
                for ( const auto &input : subOp->inputs.pairs() )
                {
                    if ( input.second.tensor->IsConstant() )
                    {
                        auto tensor = input.second.tensor.get();
                        if ( tensor->AllocatedAddress() >= 0 )
                        {
                            AddToReadOnly(tensor);
                        }
                    }
                }
            }
        }
    }

private:
    std::unique_ptr<Tensor> CreateCommandStreamTensor(const std::vector<uint32_t> &commandStream)
    {
        auto tensor = std::make_unique<Tensor>("ethos_u_command_stream", DataType::UInt8);
        const std::shared_ptr<Buffer> buffer = DriverActions::CreateDriverPayload(commandStream, _archConfigWord, _archVersion);
        tensor->SetStorageShape(Shape(buffer->Size()));
        tensor->SetBuffer(buffer);
        return tensor;
    }

    // TODO: Move into TfLiteWriter and fill directly into the flatbuffer to reduce memory footprint
    void AddToReadOnly(const SchedulerTensor *tensor)
    {
        if ( tensor )
        {
            const auto offset = tensor->AllocatedAddress();
            const auto allocation = tensor->AllocationSizeBytes();
            const auto buffer = tensor->srcTensor ? tensor->srcTensor->View().Buffer() : tensor->bufferView.Buffer();
            const auto size = buffer->Size();

            assert(tensor->memArea.usage % MemUsage::ReadOnly);
            assert((offset >= 0) && (allocation >= 0));                         // Has been allocated
            assert((offset + allocation) <= Address(_readOnlyBuffer->Size()));  // Allocation fits in buffer
            assert(size <= allocation);                                         // Tensor fits in allocation

            std::copy_n(buffer->Data<uint8_t>(), size, _readOnlyBuffer->Data<uint8_t>() + offset);
        }
    }
};

}  // namespace regor
