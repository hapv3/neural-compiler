//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "compiler/faststorage_allocator.hpp"
#include "compiler/scheduler.hpp"
#include "util.hpp"

#include <catch_all.hpp>
#include <functional>
#include <memory>

#include "regor.h"

static std::shared_ptr<SchedulerTensor> CreateTensor(std::string name, MemArea memArea)
{
    auto schedTensor = CreateSchedulerTensor(name, Shape(10, 10, 10), DataType::Int8);
    schedTensor->memArea = memArea;
    return schedTensor;
}

static std::unique_ptr<SchedulerOperation> CreateSchedulerOperation(std::unique_ptr<Architecture> &arch, OpType type, bool npu,
    TensorUsage ifmUsage, std::shared_ptr<SchedulerTensor> &ifm, TensorUsage ofmUsage, std::shared_ptr<SchedulerTensor> &ofm)
{
    auto schedOp = CreateSchedulerOperation(type, ifmUsage, ifm, ofmUsage, ofm);
    schedOp->SetNpuOp(npu);
    if ( npu )
    {
        ArchitectureOpGroupQuery query{};
        query.type = schedOp->Type();
        query.kernel = schedOp->Kernel();
        query.inputs = 1;
        query.ifm[0].key = ifm->uid;
        query.ifm[0].type = ifm->dataType;
        query.ifm[0].shape = ifm->storageShape;
        query.ofm.key = ofm->uid;
        query.ofm.type = ofm->dataType;
        query.ofm.shape = ofm->storageShape;
        query.ofm.transpose = TransposeType::None;
        query.ofm.reverse = ReverseType::None;
        auto opGroup = arch->CreateOpGroup(query);
        assert(opGroup);
        schedOp->SetOpGroup(std::move(opGroup));
    }
    else
    {
        ifm->hasCPUReaders = true;
        ofm->hasCPUWriters = true;
    }
    return schedOp;
}

static std::unique_ptr<Schedule> CreateSchedule(std::unique_ptr<Architecture> &arch, std::vector<std::unique_ptr<SchedulerOperation>> &schedOps)
{
    auto schedule = std::make_unique<Schedule>("test", 0, schedOps.size());
    for ( auto &op : schedOps )
    {
        auto ifm = op->IFM(0);
        auto ofm = op->OFM();
        ArchitectureConfigQuery query{};
        query.kernel = op->Kernel();
        query.ifmBits = DataTypeSizeBits(ifm->tensor->dataType);
        query.ofmBits = DataTypeSizeBits(ofm->tensor->dataType);
        query.ifmShape[0] = ifm->shape;
        query.ofmShape = ofm->shape;
        query.transpose = TransposeType::None;
        query.reverse = ReverseType::None;
        auto opConfig = arch->GetOpConfig(op->Type(), query);
        auto schedOpInfo = std::make_unique<SchedulerOpInfo>(std::move(opConfig), ifm->shape, Shape(), ofm->shape);
        schedule->SetCost(*op, std::move(schedOpInfo));
    }
    return schedule;
}

static void ExpectReusableIfmForOp(std::unique_ptr<Architecture> &arch, OpType opType, const MemArea &fast)
{
    auto ifm = CreateTensor("ifm_" + std::to_string(static_cast<int>(opType)), fast);
    auto ofm = CreateTensor("ofm_" + std::to_string(static_cast<int>(opType)), fast);

    std::vector<std::unique_ptr<SchedulerOperation>> ops;
    ops.push_back(CreateSchedulerOperation(arch, opType, true, TensorUsage::IFM, ifm, TensorUsage::OFM, ofm));

    SchedulerOptions opts;
    opts.optimizationStagingLimit = 32 * 1024;
    SchedulerOpConfigMap configMap;
    auto scheduler = Scheduler(arch.get(), opts, "test", ops, configMap);

    auto schedule = scheduler.Process();
    (void)schedule;

    CAPTURE(static_cast<int>(opType));
    // coverity[cert_exp55_cpp_violation]
    REQUIRE(ifm->AllocatedAddress() == ofm->AllocatedAddress());
}

static void ExpectNotReusableIfmForOp(std::unique_ptr<Architecture> &arch, OpType opType, const MemArea &fast, const char *reason,
    const std::function<void(std::shared_ptr<SchedulerTensor> &, std::shared_ptr<SchedulerTensor> &,
        std::vector<std::unique_ptr<SchedulerOperation>> &)> &tweak)
{
    auto ifm = CreateTensor("ifm_neg_" + std::to_string(static_cast<int>(opType)), fast);
    auto ofm = CreateTensor("ofm_neg_" + std::to_string(static_cast<int>(opType)), fast);

    std::vector<std::unique_ptr<SchedulerOperation>> ops;
    ops.push_back(CreateSchedulerOperation(arch, opType, true, TensorUsage::IFM, ifm, TensorUsage::OFM, ofm));

    if ( tweak )
    {
        tweak(ifm, ofm, ops);
    }

    SchedulerOptions opts;
    opts.optimizationStagingLimit = 32 * 1024;
    SchedulerOpConfigMap configMap;
    auto scheduler = Scheduler(arch.get(), opts, "test", ops, configMap);

    auto schedule = scheduler.Process();
    (void)schedule;

    CAPTURE(static_cast<int>(opType));
    // coverity[cert_str30_c_violation]
    CAPTURE(reason);
    // coverity[cert_exp55_cpp_violation]
    REQUIRE(ifm->AllocatedAddress() != ofm->AllocatedAddress());
}

TEST_CASE("test_fast_storage_allocator")
{
    // Create arch
    auto arch = CreateArchDefault<ArchEthosU85>();
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    // Create some memories
    const MemArea fast = arch->StagingMemory();
    const MemArea notFast = arch->FeatureMapMemory();

    // Create some tensors
    auto tens1 = CreateTensor("t1", notFast);
    auto tens2 = CreateTensor("t2", notFast);
    auto tens3 = CreateTensor("t3", notFast);
    auto tens4 = CreateTensor("t4", notFast);
    auto tens5 = CreateTensor("t5", notFast);
    auto tens6 = CreateTensor("t6", notFast);

    SECTION("Sequential network")
    {
        std::vector<std::unique_ptr<SchedulerOperation>> ops;
        ops.push_back(CreateSchedulerOperation(arch, OpType::AvgPool, true, TensorUsage::IFM, tens1, TensorUsage::OFM, tens2));
        ops.push_back(CreateSchedulerOperation(arch, OpType::AvgPool, true, TensorUsage::IFM, tens2, TensorUsage::OFM, tens3));

        auto schedule = CreateSchedule(arch, ops);
        FastStorageAllocator allocator;
        allocator.AllocateFeatureMaps(ops, schedule.get(), fast, 32 * 1024, true);

        REQUIRE(tens1->memArea != fast);  // Because no producers
        REQUIRE(tens2->memArea == fast);
        REQUIRE(tens3->memArea != fast);  // Because no consumers
    }

    SECTION("Mixed NPU/CPU network with a live range covering CPU operation")
    {
        std::vector<std::unique_ptr<SchedulerOperation>> ops;
        ops.push_back(CreateSchedulerOperation(arch, OpType::AvgPool, true, TensorUsage::IFM, tens1, TensorUsage::OFM, tens2));
        ops.push_back(CreateSchedulerOperation(arch, OpType::AvgPool, true, TensorUsage::IFM, tens2, TensorUsage::OFM, tens3));
        ops.push_back(CreateSchedulerOperation(arch, OpType::AvgPool, false, TensorUsage::IFM, tens2, TensorUsage::OFM, tens4));
        ops.push_back(CreateSchedulerOperation(arch, OpType::AvgPool, true, TensorUsage::IFM, tens3, TensorUsage::OFM, tens5));
        ops.push_back(CreateSchedulerOperation(arch, OpType::AvgPool, true, TensorUsage::IFM, tens4, TensorUsage::OFM, tens6));

        auto schedule = CreateSchedule(arch, ops);
        FastStorageAllocator allocator;
        allocator.AllocateFeatureMaps(ops, schedule.get(), fast, 32 * 1024, true);

        REQUIRE(tens1->memArea != fast);  // Because no producers
        REQUIRE(tens2->memArea != fast);  // Because CPU readers
        REQUIRE(tens3->memArea != fast);  // Because live range covers CPU operation
        REQUIRE(tens4->memArea != fast);  // Because CPU writers
        REQUIRE(tens5->memArea != fast);  // Because no consumers
        REQUIRE(tens6->memArea != fast);  // Because no consumers
    }

    SECTION("Network with a live range covering a control flow operation")
    {
        std::vector<std::unique_ptr<SchedulerOperation>> ops;
        ops.push_back(CreateSchedulerOperation(arch, OpType::AvgPool, true, TensorUsage::IFM, tens1, TensorUsage::OFM, tens2));
        ops.push_back(CreateSchedulerOperation(arch, OpType::AvgPool, true, TensorUsage::IFM, tens2, TensorUsage::OFM, tens3));
        ops.push_back(CreateSchedulerOperation(arch, OpType::If, true, TensorUsage::IFM, tens2, TensorUsage::OFM, tens4));
        ops.push_back(CreateSchedulerOperation(arch, OpType::AvgPool, true, TensorUsage::IFM, tens3, TensorUsage::OFM, tens5));
        ops.push_back(CreateSchedulerOperation(arch, OpType::AvgPool, true, TensorUsage::IFM, tens4, TensorUsage::OFM, tens6));

        auto schedule = CreateSchedule(arch, ops);
        FastStorageAllocator allocator;
        allocator.AllocateFeatureMaps(ops, schedule.get(), fast, 32 * 1024, true);

        REQUIRE(tens1->memArea != fast);  // Because no producers
        REQUIRE(tens2->memArea != fast);  // Because consumed by control flow operation
        REQUIRE(tens3->memArea != fast);  // Because live range covers control flow operation
        REQUIRE(tens4->memArea != fast);  // Because produced by control flow operation
        REQUIRE(tens5->memArea != fast);  // Because no consumers
        REQUIRE(tens6->memArea != fast);  // Because no consumers
    }

    SECTION("IFM reuse enabled does reuse for a valid network")
    {
        tens1->memArea = fast;
        tens2->memArea = fast;

        std::vector<std::unique_ptr<SchedulerOperation>> ops;
        ops.push_back(CreateSchedulerOperation(arch, OpType::Abs, true, TensorUsage::IFM, tens1, TensorUsage::OFM, tens2));

        SchedulerOptions opts;
        opts.optimizationStagingLimit = 32 * 1024;
        SchedulerOpConfigMap configMap;
        auto scheduler = Scheduler(arch.get(), opts, "test", ops, configMap);

        auto schedule = scheduler.Process();

        REQUIRE(tens1->AllocatedAddress() == tens2->AllocatedAddress());
    }

    SECTION("IFM reuse disabled skips reuse for an otherwise valid reusable IFM")
    {
        tens1->memArea = fast;
        tens2->memArea = fast;

        std::vector<std::unique_ptr<SchedulerOperation>> ops;
        ops.push_back(CreateSchedulerOperation(arch, OpType::Abs, true, TensorUsage::IFM, tens1, TensorUsage::OFM, tens2));

        SchedulerOptions opts;
        opts.optimizationStagingLimit = 32 * 1024;
        opts.disabled.Set(SchedulerFeature::ReuseIFM);
        SchedulerOpConfigMap configMap;
        auto scheduler = Scheduler(arch.get(), opts, "test", ops, configMap);

        auto schedule = scheduler.Process();

        REQUIRE(tens1->AllocatedAddress() != tens2->AllocatedAddress());
    }

    // coverity[parameter_hidden]
    SECTION("IFM reuse enabled for supported single-input ops")
    {
        // Note: Cast is not an NPU op in this flow, so it's not covered here.
        const std::vector<OpType> opTypes = {
            OpType::Relu,
            OpType::Abs,
            OpType::Rescale,
            OpType::Quantize,
            OpType::MemoryCopy,
            OpType::Transpose,
            OpType::AvgPool,
            OpType::MaxPool,
            OpType::QuantizedAvgPool,
            OpType::QuantizedMaxPool,
        };

        for ( auto opType : opTypes )
        {
            ExpectReusableIfmForOp(arch, opType, fast);
        }
    }

    // coverity[parameter_hidden]
    SECTION("IFM reuse not applied for non-reusable op types")
    {
        const std::vector<OpType> opTypes = {
            OpType::Conv2D,
            OpType::DepthwiseConv2D,
            OpType::FullyConnected,
        };

        for ( auto opType : opTypes )
        {
            ExpectNotReusableIfmForOp(arch, opType, fast, "non-reusable op type", {});
        }
    }
}
