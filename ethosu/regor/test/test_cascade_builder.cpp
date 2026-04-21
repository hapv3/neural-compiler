//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/common.hpp"

#include "architecture/ethosu85/ethos_u85.hpp"
#include "compiler/cascade_builder.hpp"
#include "compiler/live_range.hpp"
#include "compiler/scheduler.hpp"
#include "compiler/tensor_allocator.hpp"
#include "util.hpp"

#include <catch_all.hpp>
#include <memory>
#include <unordered_map>

#include "regor.h"

namespace
{

std::shared_ptr<SchedulerTensor> CreateTestTensor(const std::string &name, const Shape &shape, MemArea memArea)
{
    auto schedTensor = CreateSchedulerTensor(name, shape, DataType::Int8);
    schedTensor->memArea = memArea;
    return schedTensor;
}

ArchitectureOpGroupQuery CreateOpGroupQuery(const SchedulerOperation &schedOp,
    const std::shared_ptr<SchedulerTensor> &ifm, const std::shared_ptr<SchedulerTensor> &ofm)
{
    ArchitectureOpGroupQuery query{};
    query.type = schedOp.Type();
    query.kernel = schedOp.Kernel();
    query.inputs = 1;
    query.ifm[0].key = ifm->uid;
    query.ifm[0].type = ifm->dataType;
    query.ifm[0].shape = ifm->storageShape;
    query.ofm.key = ofm->uid;
    query.ofm.type = ofm->dataType;
    query.ofm.shape = ofm->storageShape;
    query.ofm.transpose = TransposeType::None;
    query.ofm.reverse = ReverseType::None;
    return query;
}

ArchitectureOpGroupQuery CreateOpGroupQuery(const SchedulerOperation &schedOp, const std::shared_ptr<SchedulerTensor> &ifm,
    const std::shared_ptr<SchedulerTensor> &ifm2, const std::shared_ptr<SchedulerTensor> &ofm)
{
    auto query = CreateOpGroupQuery(schedOp, ifm, ofm);
    query.inputs = 2;
    query.ifm[1].key = ifm2->uid;
    query.ifm[1].type = ifm2->dataType;
    query.ifm[1].shape = ifm2->storageShape;
    return query;
}

std::unique_ptr<SchedulerOperation> CreateTestSchedulerOperation(std::unique_ptr<Architecture> &arch, OpType type,
    TensorUsage ifmUsage, std::shared_ptr<SchedulerTensor> &ifm, TensorUsage ofmUsage, std::shared_ptr<SchedulerTensor> &ofm)
{
    auto schedOp = ::CreateSchedulerOperation(type, ifmUsage, ifm, ofmUsage, ofm);
    schedOp->SetNpuOp(true);
    auto query = CreateOpGroupQuery(*schedOp, ifm, ofm);
    auto opGroup = arch->CreateOpGroup(query);
    assert(opGroup);
    schedOp->SetOpGroup(std::move(opGroup));
    return schedOp;
}

std::unique_ptr<SchedulerOperation> CreateTestSchedulerOperation(std::unique_ptr<Architecture> &arch, OpType type,
    TensorUsage ifmUsage, std::shared_ptr<SchedulerTensor> &ifm, TensorUsage ifm2Usage,
    std::shared_ptr<SchedulerTensor> &ifm2, TensorUsage ofmUsage, std::shared_ptr<SchedulerTensor> &ofm)
{
    auto schedOp = ::CreateSchedulerOperation(type, ifmUsage, ifm, ifm2Usage, ifm2, ofmUsage, ofm);
    schedOp->SetNpuOp(true);
    auto query = CreateOpGroupQuery(*schedOp, ifm, ifm2, ofm);
    auto opGroup = arch->CreateOpGroup(query);
    assert(opGroup);
    schedOp->SetOpGroup(std::move(opGroup));
    return schedOp;
}

std::unique_ptr<Schedule> CreateTestSchedule(std::unique_ptr<Architecture> &arch, std::vector<std::unique_ptr<SchedulerOperation>> &schedOps)
{
    auto schedule = std::make_unique<Schedule>("test", 0, schedOps.size());
    for ( auto &op : schedOps )
    {
        auto ifm = op->IFM(op->PrimaryIfmIndex());
        auto ifm2 = op->TryIFM(1 - op->PrimaryIfmIndex());
        auto ofm = op->OFM();
        ArchitectureConfigQuery query{};
        query.kernel = op->Kernel();
        query.ifmBits = DataTypeSizeBits(ifm->tensor->dataType);
        query.ofmBits = DataTypeSizeBits(ofm->tensor->dataType);
        query.ifmShape[0] = ifm->shape;
        query.ifmShape[1] = ifm2 ? ifm2->shape : Shape();
        query.ofmShape = ofm->shape;
        query.transpose = TransposeType::None;
        query.reverse = ReverseType::None;
        auto opConfig = arch->GetOpConfig(op->Type(), query);
        auto schedOpInfo = std::make_unique<SchedulerOpInfo>(std::move(opConfig), ifm->shape, ifm2 ? ifm2->shape : Shape(), ofm->shape);
        schedule->SetCost(*op, std::move(schedOpInfo));
    }
    return schedule;
}

void SetOpIndices(std::vector<std::unique_ptr<SchedulerOperation>> &ops)
{
    for ( int i = 0; i < int(ops.size()); ++i )
    {
        ops[i]->_index = i;
    }
}

void SetScheduleStripes(Schedule *schedule, const std::vector<std::unique_ptr<SchedulerOperation>> &ops, const Shape &stripe)
{
    for ( auto &op : ops )
    {
        auto *cost = schedule->Cost(op.get());
        cost->stripe = stripe;
        cost->stripeInput[0] = stripe;
        cost->stripeInput[1] = op->TryIFM(1 - op->PrimaryIfmIndex()) ? stripe : Shape();
    }
}

}  // namespace

TEST_CASE("Cascade builder requests rolling-buffer reuse for binary op with primary IFM1")
{
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    const MemArea fast = arch->StagingMemory();
    auto tensA = CreateTestTensor("bin_a", Shape(8, 8, 16), fast);
    auto tensB = CreateTestTensor("bin_b", Shape(8, 8, 16), fast);
    auto tensX = CreateTestTensor("bin_x", Shape(8, 8, 16), fast);
    auto tensC = CreateTestTensor("bin_c", Shape(8, 8, 16), fast);
    auto tensD = CreateTestTensor("bin_d", Shape(8, 8, 16), fast);

    std::vector<std::unique_ptr<SchedulerOperation>> ops;
    ops.push_back(CreateTestSchedulerOperation(arch, OpType::Abs, TensorUsage::IFM, tensA, TensorUsage::OFM, tensB));
    ops.push_back(CreateTestSchedulerOperation(
        arch, OpType::Add, TensorUsage::IFM0, tensX, TensorUsage::IFM1, tensB, TensorUsage::OFM, tensC));
    ops.push_back(CreateTestSchedulerOperation(arch, OpType::Abs, TensorUsage::IFM, tensC, TensorUsage::OFM, tensD));
    ops[1]->SetPrimaryIfmIndex(1);
    SetOpIndices(ops);

    auto refSchedule = CreateTestSchedule(arch, ops);
    auto fallbackSchedule = CreateTestSchedule(arch, ops);
    const Shape fullStripe = tensA->storageShape;
    const Shape cascadeStripe(1, fullStripe.Width(), fullStripe.Depth());
    SetScheduleStripes(refSchedule.get(), ops, cascadeStripe);
    SetScheduleStripes(fallbackSchedule.get(), ops, fullStripe);

    std::unordered_map<UniqueId, int> opLocalMemUsage;
    std::unordered_map<UniqueId, int> nonLocalMemUsage;
    std::unordered_map<UniqueId, LiveRangeSummary> liveRanges;

    const int rollingBufferSize = DataTypeStorageSizeBytes(
        tensB->dataType, Shape(2, fullStripe.Width(), fullStripe.Depth()).Elements());
    CascadeBuilder cascadeBuilder(ops, nonLocalMemUsage, opLocalMemUsage, liveRanges, true);
    cascadeBuilder.BuildCascades(refSchedule.get(), fallbackSchedule.get(), rollingBufferSize);

    REQUIRE(refSchedule->Cost(ops[1].get())->ofmEquivalenceId == tensB->equivalenceId);

    LiveRangeGraph lrGraph{false};
    AllocateTensors(ops, refSchedule.get(), lrGraph, fast, TensorAllocator::LinearAlloc, 16, false);

    REQUIRE(tensB->AllocatedAddress() == tensC->AllocatedAddress());
}
