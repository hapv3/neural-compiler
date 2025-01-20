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
#include "common/ordered_map.hpp"
#include "kernel.hpp"
#include "operation.hpp"
#include "tensor.hpp"

#include <vector>

namespace regor
{

class SchedulerOperation;

int TensorAllocationBytes(const Shape &shape, TensorFormat format, DataType dtype);

/// <summary>
/// Scheduler's metadata for graph tensors.
/// </summary>
struct SchedulerTensor
{
private:
    int allocatedSize = 0;
    Address allocatedAddress = -1;

public:
    std::shared_ptr<Tensor> srcTensor;
    TensorFormat format = TensorFormat::Unknown;
    MemArea memArea;
    Shape storageShape;
    BufferView bufferView;
    DataType dataType = DataType::None;
    bool hasCPUReaders = false;
    bool hasCPUWriters = false;
    bool isGraphInput = false;
    bool isGraphOutput = false;
    bool isPersistent = false;
    bool needsLinearFormat = false;
    // If two tensors have same equivalence id and same memory area, they can be stored on the same address
    UniqueId equivalenceId = GenerateUniqueId();
    UniqueId uid = ~0u;  // Packing must initialise
    std::vector<SchedulerOperation *> producers;
    std::vector<SchedulerOperation *> consumers;

    SchedulerTensor() {}

    SchedulerTensor(DataType type, const Shape &shape, TensorFormat fmt = TensorFormat::Unknown) :
            format(fmt), storageShape(shape), dataType(type)
    {
        this->uid = GenerateUniqueId();
    }

    void RemoveReader(const SchedulerOperation *op)
    {
        auto end = std::remove(consumers.begin(), consumers.end(), op);
        consumers.erase(end, consumers.end());
    }

    void SetAddress(Address address)
    {
        assert(allocatedAddress == -1 && address >= 0);
        allocatedAddress = address;
    }

    Address AllocatedAddress() const { return allocatedAddress; }

    void SetAllocatedSize(int size)
    {
        assert(allocatedSize == 0 && size > 0);
        allocatedSize = size;
    }

    int AllocationSizeBytes() const
    {
        return (allocatedSize > 0) ? allocatedSize : TensorAllocationBytes(storageShape, format, dataType);
    }

    std::string Name() const { return srcTensor.get() == nullptr ? "?" : srcTensor->Name(); }
    bool IsConstant() const { return bufferView.HasBuffer() && bufferView.Buffer()->Size(); }
};


enum class Buffering
{
    None,
    Single,
    Double,
};


/// <summary>
/// Scheduler's metadata for tensor connections. This data is not shared
/// between operators working on the same tensor
/// </summary>
struct SchedulerConnection
{
    std::shared_ptr<SchedulerTensor> tensor;
    Shape shape;
    TensorSlice slice;
    Point2i stepXY{1, 1};
    Quantization quantization;
    ArchResampling resamplingMode = ArchResampling::None;
    TransposeType transpose = TransposeType::None;
    ReverseType reverse = ReverseType::None;
    RoundMode rounding = RoundMode::AUTO;
    bool requireFullTensor = false;
    bool preBuffer = false;
    Buffering buffering = Buffering::None;

    int PartialAllocationSizeBytes() const { return TensorAllocationBytes(shape, tensor->format, tensor->dataType); }
    const Shape &SliceShape() const { return slice.shape.IsEmpty() ? shape : slice.shape; }
};

enum class AccumulatorSource
{
    Reset = 0,
    Acc = 1,
    Ifm2 = 2
};

struct AccumulatorControl
{
    AccumulatorSource source = AccumulatorSource::Reset;
    bool outputEnabled = true;
};

/// <summary>
/// Scheduler's representation of executable operations
/// </summary>
class SchedulerOperation : public Attributable
{
    friend class SchedulerPacking;
    friend class Scheduler;

public:
    OpType _type;
    int _index = -1;  // Execution index
    std::unique_ptr<class Kernel> _kernel;
    bool _npuOp = false;
    bool _hasScaling = false;
    void *_srcKey = nullptr;
    int _primaryIfmIndex = 0;
    AccumulatorControl _accumulatorControl;
    const class SchedulerOperation *_parent = nullptr;
    std::vector<std::unique_ptr<SchedulerOperation>> _subOps;  // activations or Ethos-U85 chained ops
    ordered_map<TensorUsage, SchedulerConnection> inputs;
    ordered_map<TensorUsage, SchedulerConnection> outputs;
    std::unique_ptr<ArchitectureOpGroup> _opGroup;
    int _opGroupKey = 0;

private:
    UniqueId _uid;

public:
    SchedulerOperation(OpType opType) : _type(opType) { _uid = GenerateUniqueId(); }
    ~SchedulerOperation() { Disconnect(); }

    SchedulerOperation &operator=(const SchedulerOperation &) = delete;
    SchedulerOperation &operator=(SchedulerOperation &&) = delete;

public:
    OpType Type() const { return _type; }
    int Index() const { return _index; }

    UniqueId Uid() const { return _uid; }
    operator UniqueId() const { return _uid; }

    bool IsNpuOp() const { return _npuOp; }
    void SetNpuOp(bool npuOp) { _npuOp = npuOp; }

    class Kernel *Kernel() const { return _kernel.get(); }
    void SetKernel(const class Kernel *kernel) { _kernel = std::make_unique<class Kernel>(*kernel); }

    bool HasScaling() const { return _hasScaling; }
    void SetHasScaling(bool hasScaling) { _hasScaling = hasScaling; }

    const AccumulatorControl &AccumulatorMode() const { return _accumulatorControl; }
    void SetAccumulatorMode(const AccumulatorControl &accumulatorControl) { _accumulatorControl = accumulatorControl; }

    const class SchedulerOperation *Parent() const { return _parent; }
    void SetParent(const class SchedulerOperation *parent) { _parent = parent; }

    int PrimaryIfmIndex() const { return _primaryIfmIndex; }
    void SetPrimaryIfmIndex(int index) { _primaryIfmIndex = index; }

    void SetAttributes(const Attributes &attr) { _attr = attr; }

    // Input connections
    SchedulerConnection *AddInput(TensorUsage usage, const std::shared_ptr<SchedulerTensor> &tensor = {})
    {
        auto &conn = inputs[usage];
        if ( tensor )
        {
            conn.tensor = tensor;
            tensor->consumers.push_back(this);
        }
        return &conn;
    }

    const SchedulerConnection *TryInput(TensorUsage usage) const { return inputs.try_ref(usage); }
    SchedulerConnection *TryInput(TensorUsage usage) { return inputs.try_ref(usage); }
    SchedulerConnection *Input(TensorUsage usage) { return &inputs.at(usage); }
    const SchedulerConnection *Input(TensorUsage usage) const { return &inputs.at(usage); }

    SchedulerConnection *TryIFM(int index) { return inputs.try_ref(MakeTensorUsage(TensorUsage::IFM, index)); }
    const SchedulerConnection *TryIFM(int index) const
    {
        return inputs.try_ref(MakeTensorUsage(TensorUsage::IFM, index));
    }
    SchedulerConnection *IFM(int index) { return &inputs.at(MakeTensorUsage(TensorUsage::IFM, index)); }
    const SchedulerConnection *IFM(int index) const { return &inputs.at(MakeTensorUsage(TensorUsage::IFM, index)); }

    // Invalidates all pointers to input connections.
    void RemoveInput(TensorUsage usage)
    {
        auto inputConnection = inputs.try_ref(usage);
        if ( inputConnection )
        {
            if ( inputConnection->tensor ) inputConnection->tensor->RemoveReader(this);
            inputs.erase(usage);
        }
    }

    // Output connections
    SchedulerConnection *AddOutput(TensorUsage usage) { return &outputs[usage]; }

    SchedulerConnection *TryOutput(TensorUsage usage) { return outputs.try_ref(usage); }
    SchedulerConnection *Output(TensorUsage usage) { return &outputs.at(usage); }
    const SchedulerConnection *Output(TensorUsage usage) const { return &outputs.at(usage); }

    SchedulerConnection *TryOFM() { return outputs.try_ref(TensorUsage::OFM); }
    const SchedulerConnection *TryOFM() const { return outputs.try_ref(TensorUsage::OFM); }
    SchedulerConnection *OFM() { return &outputs.at(TensorUsage::OFM); }
    const SchedulerConnection *OFM() const { return &outputs.at(TensorUsage::OFM); }

    void AddSubOp(std::unique_ptr<SchedulerOperation> subOp) { _subOps.push_back(std::move(subOp)); }

    const std::vector<std::unique_ptr<SchedulerOperation>> &SubOps() const { return _subOps; }

    ArchitectureOpGroup *OpGroup() const
    {
        if ( _parent )
        {
            return _parent->OpGroup();
        }
        return _opGroup.get();
    };

    // Returns connections for which live range calculation is needed
    std::vector<std::pair<TensorUsage, SchedulerTensor *>> LiveRangeTensors() const
    {
        std::vector<std::pair<TensorUsage, SchedulerTensor *>> liveTensors;
        for ( const auto *list : {&inputs, &outputs} )
        {
            for ( const auto &item : list->pairs() )
            {
                auto usage = item.first & TensorUsage::TypeMask;
                if ( usage == TensorUsage::IFM || usage == TensorUsage::OFM || usage == TensorUsage::LUT || usage == TensorUsage::Scratch )
                {
                    if ( _opGroup == nullptr || _opGroup->NeedsAllocation(item.second.tensor->uid) )
                    {
                        liveTensors.push_back(std::make_pair(item.first, item.second.tensor.get()));
                    }
                }
            }
        }
        // live-tensors from sub-operations
        for ( const auto &subOp : SubOps() )
        {
            for ( const auto *list : {&subOp->inputs, &subOp->outputs} )
            {
                for ( const auto &item : list->pairs() )
                {
                    auto usage = item.first & TensorUsage::TypeMask;
                    if ( usage == TensorUsage::IFM || usage == TensorUsage::OFM || usage == TensorUsage::LUT || usage == TensorUsage::Scratch )
                    {
                        if ( _opGroup == nullptr || _opGroup->NeedsAllocation(item.second.tensor->uid) )
                        {
                            liveTensors.push_back(std::make_pair(item.first, item.second.tensor.get()));
                        }
                    }
                }
            }
        }
        return liveTensors;
    }

    void SetOpGroup(std::unique_ptr<ArchitectureOpGroup> &&opGroup) { _opGroup = std::move(opGroup); }

    void SetOpGroupKey(int opGroupKey) { _opGroupKey = opGroupKey; }

    bool IsReordering() const
    {
        if ( !IsNone(OFM()->transpose) )
        {
            return true;
        }

        if ( OFM()->reverse != ReverseType::None )
        {
            return true;
        }

        for ( const auto &op : _subOps )
        {
            if ( op->IsReordering() )
            {
                return true;
            }
        }

        return false;
    }

    void Disconnect()
    {
        for ( const auto *list : {&inputs, &outputs} )
        {
            for ( const auto &item : list->pairs() )
            {
                const auto &connection = item.second;
                if ( connection.tensor )
                {
                    auto usage = item.first;
                    auto &vec = IsOFM(usage) ? connection.tensor->producers : connection.tensor->consumers;
                    vec.erase(std::remove(vec.begin(), vec.end(), this), vec.end());
                }
            }
        }
        inputs.clear();
        outputs.clear();
    }

    bool IsDisconnected() const { return inputs.empty() && outputs.empty(); }
};

/// <summary>
/// NPU-Operation
/// </summary>
class NPUOperation
{
private:
    std::vector<std::unique_ptr<SchedulerOperation>> _ops;

public:
    const std::vector<std::unique_ptr<SchedulerOperation>> &Operations() const { return _ops; };
    void AddOperation(std::unique_ptr<SchedulerOperation> op) { _ops.push_back(std::move(op)); }
};

}  // namespace regor
