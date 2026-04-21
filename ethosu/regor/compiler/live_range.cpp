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


#include "live_range.hpp"

#include "scheduler.hpp"
#include "scheduler_operation.hpp"

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

BEGIN_ENUM_TABLE(regor::LRUsage)
    ADD_ENUM_NAME(Unassigned)
    ADD_ENUM_NAME(OpLocal)
    ADD_ENUM_NAME(OpCascade)
    ADD_ENUM_NAME(OpBuffering)
END_ENUM_TABLE()

namespace regor
{

LiveRange::LiveRange(SchedulerTensor *tensor)
{
    size = tensor->AllocationSizeBytes();
    memArea = tensor->memArea;
    name = tensor->Name();
    AddTensor(tensor);
}

void LiveRange::SetAddress(Address address)
{
    for ( auto &tensor : tensors )
    {
        tensor->SetAddress(address);
    }
}

void LiveRange::Adjust(LRMemory &mem, int time, const LiveRange &lr, int amount)
{
    // LR contributes directly to the op if time matches delayed start
    if ( time == (lr.startTime + lr.opDelay) )
    {
        if ( lr.usage == LRUsage::OpLocal ) mem.op += amount;
        else if ( lr.usage == LRUsage::OpCascade ) mem.cascade += amount;
        else if ( lr.usage == LRUsage::OpBuffering ) mem.buffering += amount;
        else
        {
            assert(false && "Memory at op timepoint must be classified");
            mem.nonlocal += amount;
        }
    }
    // LR is not for the op at the current time point
    else
    {
        mem.nonlocal += amount;
    }
    assert(mem.Used() >= 0);
}

MemorySnapshot LiveRangeGraph::GetTemporalMemoryUsage(int granularity)
{
    assert(granularity > 0);
    MemorySnapshot usage(_currentTime + 1);
    usage.maxMemory = 0;
    for ( const auto &lr : _lrs )
    {
        assert(lr->endTime <= _currentTime);
        int maxMemory = AdjustMemoryUsage(usage.memory, *lr, RoundAway(lr->size, granularity));
        usage.maxMemory = std::max(usage.maxMemory, maxMemory);
    }
    return usage;
}

int LiveRangeGraph::AdjustMemoryUsage(std::vector<LRMemory> &usage, const LiveRange &lr, int adjust)
{
    int maxMemory = 0;
    for ( int i = lr.startTime; i <= lr.endTime; i++ )
    {
        LRMemory &mem = usage.at(i);
        LiveRange::Adjust(mem, i, lr, adjust);
        maxMemory = std::max(maxMemory, mem.Used());
    }
    return maxMemory;
}

void LiveRangeGraph::ExtractLiveRangesFromCascades(const std::vector<std::unique_ptr<SchedulerOperation>> &schedOps,
    Schedule *schedule, const MemArea &targetMemory, bool addRollingBuffers)
{
    std::unordered_map<int, int> timeForCascade;
    auto startTime = _currentTime;
    // Live ranges containing graph output
    std::vector<LiveRange *> graphOutputRanges;
    // Live ranges containing persistent tensors
    std::vector<LiveRange *> persistentRanges;
    for ( const auto &schedOp : schedOps )
    {
        SchedulerOpInfo *opInfo = schedule->Cost(schedOp.get());
        int cascade = opInfo->cascade;

        CascadeInfo *cascadeInfo = cascade == 0 ? nullptr : &schedule->cascades[cascade];
        CascadeBuffer *cascadeBuffer = nullptr;

        int timeToSet = _currentTime;
        if ( schedOp->IsNpuOp() )
        {
            auto opGroup = schedOp->OpGroup();
            assert(opGroup != nullptr);

            // Get the ofm of the last operator in the group
            auto opGroupOfm = schedOp->FinalSubOFM();
            if ( opGroup->NeedsAllocation(opGroupOfm->tensor->uid) )
            {
                SchedulerTensor *ifmTens = nullptr;
                auto targetMemoryAllowed = addRollingBuffers || !ShouldBeIgnored(opGroupOfm->tensor.get(), targetMemory);
                if ( targetMemoryAllowed && opInfo->ofmEquivalenceId != INVALID_UID )
                {
                    ifmTens = ReusableRollingBufferIFM(schedOp, opInfo->ofmEquivalenceId);
                }
                if ( !ifmTens && _reuseIfms && cascadeInfo == nullptr )
                {
                    // Check if op have an ifm tensor that can be reused for the ofm
                    ifmTens = ReusableIFM(schedOp, opGroupOfm->tensor.get(), targetMemory);
                }
                if ( ifmTens != nullptr )
                {
                    FuseRanges(ifmTens, opGroupOfm->tensor.get());
                }
            }
            if ( cascadeInfo != nullptr )
            {
                auto entry = cascadeInfo->buffers.find(*schedOp);
                if ( entry != cascadeInfo->buffers.end() )
                {
                    cascadeBuffer = &entry->second;
                }
                auto tfcEntry = timeForCascade.find(cascade);
                if ( tfcEntry != timeForCascade.end() )
                {
                    timeToSet = tfcEntry->second;
                }
                timeForCascade[cascade] = timeToSet;
            }

            // Buffered weight tensor
            const auto &bufferTensor = opInfo->bufferedWeightTensor;
            assert(
                (bufferTensor.buffering == Buffering::None && bufferTensor.parts == 0) ||
                (bufferTensor.buffering == Buffering::Single && bufferTensor.parts == 1) ||
                (bufferTensor.buffering == Buffering::Double && bufferTensor.parts == 2));
            assert(!bufferTensor.preBuffer || bufferTensor.buffering != Buffering::None);
            assert(opInfo->ofmDepthSlices.size() >= bufferTensor.parts);
            for ( unsigned part = 0; part < bufferTensor.parts; part++ )
            {
                const unsigned sliceCount = opInfo->ofmDepthSlices.size() - 1;
                const unsigned lastSlicePart = (sliceCount - 1) % bufferTensor.parts;
                auto *partWeightTensor = bufferTensor.tensor[part].get();
                if ( !ShouldBeIgnored(partWeightTensor, targetMemory) )
                {
                    auto *lr = GetOrCreateRange(partWeightTensor, LRUsage::OpBuffering);
                    int start = timeToSet;
                    int duration = 1;
                    if ( part == 0 && bufferTensor.preBuffer )
                    {
                        // Move start earlier for the part that is used fo pre-buffering
                        lr->opDelay = 1;
                        start--;
                        duration++;
                    }
                    if ( part != lastSlicePart )
                    {
                        // Extend the duration for the part that is used for the last slice
                        duration--;
                    }
                    lr->MarkUsage(start, duration);
                }
            }

            // Read-only weight/scale tensors
            for ( auto tens : {opInfo->npuWeightsTensor, opInfo->npuScalesTensor} )
            {
                if ( !ShouldBeIgnored(tens.get(), targetMemory) )
                {
                    auto lr = GetOrCreateRange(tens.get(), LRUsage::OpLocal);
                    lr->MarkUsage(timeToSet);
                }
            }
        }

        // Set time index for the op and its subops
        opInfo->timeIndex = timeToSet;
        for ( auto &subOp : schedOp->SubOps() )
        {
            SchedulerOpInfo *subOpInfo = schedule->Cost(subOp.get());
            subOpInfo->timeIndex = timeToSet;
        }

        // Mark usage for all relevant tensors related to this operation
        auto liveRangeTensors = schedOp->LiveRangeTensors();
        for ( const auto &liveTensor : liveRangeTensors )
        {
            auto usage = liveTensor.first;
            auto tens = liveTensor.second;

            // This creates rolling-buffer live-range entries only for mid-cascade IFMs
            bool isRollingBuffer = (cascadeBuffer != nullptr) && (usage == MakeTensorUsage(TensorUsage::IFM, schedOp->PrimaryIfmIndex()));
            if ( ShouldBeIgnored(tens, targetMemory) && !(addRollingBuffers && isRollingBuffer) )
            {
                continue;
            }

            // This identifies whether the OFM contributes to the cascade at this time point (classification only)
            bool isOfmRollingBuffer = (usage == TensorUsage::OFM) && (cascade != 0) && (schedOp->Index() < cascadeInfo->end);
            auto lr = GetOrCreateRange(tens, isOfmRollingBuffer ? LRUsage::OpCascade : LRUsage::OpLocal);
            if ( tens->isGraphInput )
            {
                // Graph input must not be overwritten by preceding schedOps
                lr->MarkUsage(startTime);
            }
            if ( tens->isGraphOutput )
            {
                // Graph output must not be overwritten by following schedOps
                graphOutputRanges.push_back(lr);
            }
            if ( tens->isPersistent )
            {
                // Persistent tensors must be alive for the entire inference
                persistentRanges.push_back(lr);
            }
            lr->MarkUsage(timeToSet);
            if ( isRollingBuffer )
            {
                // This tensor is a rolling buffer in a cascade and the size of the LiveRange needs to be modified
                // for enabling temporal memory snapshots without modifying the original Tensor
                lr->size = cascadeBuffer->sizeBytes;
            }
        }
        if ( timeToSet == _currentTime )
        {
            _currentTime += 2;
        }
    }
    for ( auto lr : graphOutputRanges )
    {
        lr->MarkUsage(_currentTime, 1);
    }

    // Persistent tensor live-range is for entire inference
    for ( auto lr : persistentRanges )
    {
        lr->MarkUsage(0, EndTime());
    }
    ++_currentTime;
}

LiveRange *LiveRangeGraph::GetOrCreateRange(SchedulerTensor *tens, LRUsage usage)
{
    // Return the live range of the tensor (or any of its clones)
    const auto entry = _equivalenceIdToLr.find(tens->equivalenceId);
    if ( entry != _equivalenceIdToLr.end() )
    {
        entry->second->AddTensor(tens);
        return entry->second;
    }
    // No live range found for the tensor, create a new one
    auto lr = std::make_unique<LiveRange>(tens);
    auto *plr = lr.get();
    _lrs.push_back(std::move(lr));
    plr->usage = usage;
    _equivalenceIdToLr[tens->equivalenceId] = plr;
    return plr;
}

LiveRange *LiveRangeGraph::FuseRanges(SchedulerTensor *inTens, SchedulerTensor *outTens)
{
    assert(outTens->AllocationSizeBytes() <= inTens->AllocationSizeBytes());
    auto lr = GetOrCreateRange(inTens, LRUsage::OpLocal);  // Assumes we only fuse op local IFM/OFM
    lr->AddTensor(outTens);
    const auto entry = _equivalenceIdToLr.find(outTens->equivalenceId);
    if ( entry != _equivalenceIdToLr.end() )
    {
        // Live range already existed for outTens, move over tensors
        auto &lr2 = entry->second;
        lr->tensors.insert(lr2->tensors.begin(), lr2->tensors.end());
        lr2->tensors.clear();
        lr2->size = 0;
    }
    _equivalenceIdToLr[outTens->equivalenceId] = lr;
    return lr;
}

// Undo FuseRanges by creating a unique live-range for tens
LiveRange *LiveRangeGraph::SplitFromRange(SchedulerTensor *tens)
{
    const auto entry = _equivalenceIdToLr.find(tens->equivalenceId);
    if ( entry != _equivalenceIdToLr.end() )
    {
        auto &lr = entry->second;
        if ( lr->tensors.size() == 1 )
        {
            return lr;
        }
        else
        {
            // If lr contains more than one tensor
            // Split tensor lr and create a new entry
            lr->tensors.erase(tens);
            _equivalenceIdToLr.erase(tens->equivalenceId);
        }
    }
    return GetOrCreateRange(tens, LRUsage::OpLocal);  // Assumes we only fuse op local IFM/OFM
}

bool LiveRangeGraph::CanReuseIFMTensors(const SchedulerTensor *ifmTens, const SchedulerTensor *ofmTens)
{
    assert(ifmTens != nullptr && ofmTens != nullptr && "IFM and OFM tensors can't be nullptr");
    return !ifmTens->isGraphOutput && !ifmTens->isPersistent && !ofmTens->isPersistent &&
           ifmTens->dataType == ofmTens->dataType && ifmTens->consumers.size() == 1 && ofmTens->producers.size() == 1;
}

// Check if any of the IFMs consumed by the first operator in an opgroup can be reused for the OFM
// tensor of the last operator in the opgroup.
// Requires the first operator to be an elementwise operator and is also applicaple to stand-alone
// elementwise operators (which are just opgroups of length 1).
SchedulerTensor *LiveRangeGraph::ReusableIFM(
    const std::unique_ptr<SchedulerOperation> &schedOp, const SchedulerTensor *ofmTens, const MemArea &targetMemory)
{
    SchedulerTensor *reusableIfm = nullptr;
    const auto *ofm = schedOp->Output(TensorUsage::OFM);
    if ( IsOp1To1(schedOp.get()) )
    {
        if ( !ShouldBeIgnored(ofmTens, targetMemory) )
        {
            for ( const auto &[usage, ifmConn] : schedOp->inputs.pairs() )
            {
                const auto ifmTens = ifmConn.tensor.get();
                if ( IsIFM(usage) && ifmTens->storageShape == ofmTens->storageShape && CanReuseIFMTensors(ifmTens, ofmTens) &&
                     ifmTens->format == ofmTens->format && !ShouldBeIgnored(ifmTens, targetMemory) )
                {
                    reusableIfm = ifmTens;
                    break;
                }
            }
        }
    }
    return reusableIfm;
}

SchedulerTensor *LiveRangeGraph::ReusableRollingBufferIFM(const std::unique_ptr<SchedulerOperation> &schedOp, UniqueId ofmEquivalenceId)
{
    for ( const auto &[usage, ifmConn] : schedOp->inputs.pairs() )
    {
        const auto ifmTens = ifmConn.tensor.get();
        if ( IsIFM(usage) && ifmTens->equivalenceId == ofmEquivalenceId )
        {
            return ifmTens;
        }
    }
    return nullptr;
}

bool LiveRangeGraph::AreInSameRange(const SchedulerTensor *lhs, const SchedulerTensor *rhs) const
{
    if ( lhs == nullptr || rhs == nullptr )
    {
        return false;
    }
    const auto lhsIt = _equivalenceIdToLr.find(lhs->equivalenceId);
    if ( lhsIt == _equivalenceIdToLr.end() )
    {
        return false;
    }
    const auto rhsIt = _equivalenceIdToLr.find(rhs->equivalenceId);
    return rhsIt != _equivalenceIdToLr.end() && lhsIt->second == rhsIt->second;
}


bool LiveRangeGraph::ShouldBeIgnored(const SchedulerTensor *tens, const MemArea &targetMemory)
{
    if ( tens == nullptr )
    {
        return true;
    }
    return tens->memArea != targetMemory;
}

// IFM reuse is possible for any operator that maps a single IFM element to a single OFM element
bool LiveRangeGraph::IsOp1To1(const SchedulerOperation *op)
{
    const auto *kernel = op->Kernel();
    const bool unitKernel = kernel && kernel->ElementsWH() == 1;
    const bool noOpReverse = op->Output(TensorUsage::OFM)->reverse == ReverseType::None;
    const bool noneTranspose = IsNone(op->Output(TensorUsage::OFM)->transpose);
    // The caller function ReusableIFM also gates certain conditions. For example, any case where the IFM and OFM type
    // is different, where the IFM is graph output, etc. We therefore do not need to do these checks explicitly here.
    return (DecomposeAsElementwise(op->Type()) || (unitKernel && IsPooling(op->Type())) || op->Type() == OpType::Transpose) && noOpReverse && noneTranspose;
}

}  // namespace regor
