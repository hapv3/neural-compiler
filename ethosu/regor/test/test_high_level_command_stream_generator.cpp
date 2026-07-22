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

#include "common/logging.hpp"

#include "architecture/ethosu85/ethos_u85.hpp"
#include "compiler/high_level_command_stream_generator.hpp"
#include "compiler/kernel.hpp"
#include "compiler/scheduler.hpp"
#include "compiler/scheduler_operation.hpp"
#include "test/util.hpp"

#include <catch_all.hpp>

using namespace regor;

namespace
{

Architecture *TestArchitecture()
{
    static auto s_arch = CreateArchDefault<ArchEthosU85>(1024);
    return s_arch.get();
}

std::unique_ptr<ArchitectureOpConfig> GetOpConfig(std::unique_ptr<SchedulerOperation> &schedOp)
{
    auto *arch = TestArchitecture();
    auto ifm = schedOp->IFM(0);
    auto ofm = schedOp->OFM();
    ArchitectureConfigQuery query{};
    query.ifmBits = DataTypeSizeBits(ifm->tensor->dataType);
    query.ofmBits = DataTypeSizeBits(ofm->tensor->dataType);
    query.ifmResampling = ArchResampling::None;
    query.transpose = ofm->transpose;
    query.reverse = ofm->reverse;
    query.ofmFormat = TensorFormat::NHCWB16;
    OpType type = schedOp->Type();
    query.ofmShape = ofm->shape;
    query.ifmShape[0] = ifm->shape;
    if ( schedOp->TryIFM(1) )
    {
        query.ifmShape[1] = schedOp->IFM(1)->shape;
    }
    query.kernel = schedOp->Kernel();
    query.lutBytes = 0;
    query.scaled = false;
    query.accOutputEnabled = true;
    return arch->GetOpConfig(type, query);
}

Schedule CreateSchedule(std::unique_ptr<SchedulerOperation> &schedOp, Shape stripeShape)
{
    Schedule schedule("Schedule", 0, 1);
    std::unique_ptr<ArchitectureOpConfig> opConfig = GetOpConfig(schedOp);
    Shape ifmShape = schedOp->IFM(0)->SliceShape();
    Shape ifm2Shape;
    std::unique_ptr<SchedulerOpInfo> opInfo = std::make_unique<SchedulerOpInfo>(std::move(opConfig), ifmShape, ifm2Shape, stripeShape);
    opInfo->cascade = 1;
    schedule.SetCost(*schedOp, std::move(opInfo));
    int opIndex = schedOp->Index();
    std::unordered_map<UniqueId, CascadeBuffer> buffers;
    std::unordered_map<int, CascadeInfo> cascadeMap = {
        {1, CascadeInfo(opIndex, opIndex, 0, std::move(buffers))},
    };
    schedule.UpdateCascades(cascadeMap);
    return schedule;
}

NPUOperation CreateNpuOperation(std::unique_ptr<SchedulerOperation> schedOp)
{
    NPUOperation npuOp;
    npuOp.AddOperation(std::move(schedOp));
    return npuOp;
}

std::unique_ptr<SchedulerOperation>
CreateSchedulerOperation(OpType opType, Kernel kernel, Shape ifmShape, Shape ofmShape, Point2i ifmStep, Point2i ofmStep)
{
    auto ifm = CreateSchedulerTensor("ifm", ifmShape, DataType::Int8);
    auto ofm = CreateSchedulerTensor("ofm", ofmShape, DataType::Int8);
    ifm->architecture = TestArchitecture();
    ofm->architecture = TestArchitecture();
    std::unique_ptr<SchedulerOperation> schedOp = ::CreateSchedulerOperation(opType, TensorUsage::IFM0, ifm, TensorUsage::OFM, ofm);
    schedOp->IFM(0)->stepXY = ifmStep;
    schedOp->OFM()->stepXY = ofmStep;
    schedOp->SetKernel(kernel);
    schedOp->_index = 0;
    return schedOp;
}
}  // namespace

TEST_CASE("CalculateIfmStripeAndPadding maps OFM stripes without padding (height-only)")
{
    Kernel kernel(Point2i{1, 3}, Point2i{1, 1}, Point2i{1, 1}, Margin(0, 0, 0, 0));
    Shape ifmShape(1, 12, 1, 3);
    Shape ofmShape(1, 10, 1, 3);
    Point2i ifmStep{1, 1};
    Point2i ofmStep{1, 1};
    Shape stripeShape = ofmShape;

    std::unique_ptr<SchedulerOperation> schedOp = CreateSchedulerOperation(OpType::Conv2D, kernel, ifmShape, ofmShape, ifmStep, ofmStep);
    auto schedule = CreateSchedule(schedOp, stripeShape);
    auto npuOp = CreateNpuOperation(std::move(schedOp));
    HLCStreamGenerator hlcsGen(0, false);
    SubGraphs subGraphs;
    HLCStream cmdStream = hlcsGen.GenerateCommandStream(&npuOp, &schedule, subGraphs);

    REQUIRE(cmdStream.size() == 1);
    auto &command = cmdStream.front();
    REQUIRE(command->CommandType() == HighLevelCommandType::STRIPE);
    HLCStripe *stripe = static_cast<HLCStripe *>(command.get());

    const auto &padding = stripe->padding;
    const auto &ifmBox = stripe->stripeAreas[0].ifmAreas[0];

    CHECK(padding.top == 0);
    CHECK(padding.left == 0);
    CHECK(padding.bottom == 0);
    CHECK(padding.right == 0);

    CHECK(ifmBox.Start().Height() == 0);
    CHECK(ifmBox.Start().Width() == 0);
    CHECK(ifmBox.Start().Depth() == 0);
    CHECK(ifmBox.SizeShape().Height() == 12);
    CHECK(ifmBox.SizeShape().Width() == 1);
    CHECK(ifmBox.SizeShape().Depth() == 3);
}

TEST_CASE("CalculateIfmStripeAndPadding reports padding on boundary stripes (height-only)")
{
    Kernel kernel(Point2i{1, 3}, Point2i{1, 1}, Point2i{1, 1}, Margin(1, 0, 1, 0));
    Shape ifmShape(1, 10, 1, 1);
    Shape ofmShape(1, 10, 1, 1);
    Shape stripeShape(1, 1, 1, 1);
    Point2i ifmStep{1, 1};
    Point2i ofmStep{1, 1};

    std::unique_ptr<SchedulerOperation> schedOp = CreateSchedulerOperation(OpType::Conv2D, kernel, ifmShape, ofmShape, ifmStep, ofmStep);
    auto schedule = CreateSchedule(schedOp, stripeShape);
    auto npuOp = CreateNpuOperation(std::move(schedOp));
    HLCStreamGenerator hlcsGen(0, false);
    SubGraphs subGraphs;
    HLCStream cmdStream = hlcsGen.GenerateCommandStream(&npuOp, &schedule, subGraphs);

    REQUIRE(cmdStream.size() == 10);

    auto &command0 = cmdStream[0];
    REQUIRE(command0->CommandType() == HighLevelCommandType::STRIPE);
    HLCStripe *stripe = static_cast<HLCStripe *>(command0.get());
    auto &padding = stripe->padding;
    auto &ifmBox = stripe->stripeAreas[0].ifmAreas[0];
    CHECK(padding.top == 1);
    CHECK(padding.left == 0);
    CHECK(padding.bottom == 0);
    CHECK(padding.right == 0);
    CHECK(ifmBox.Start().Height() == 0);
    CHECK(ifmBox.Start().Width() == 0);
    CHECK(ifmBox.SizeShape().Height() == 2);
    CHECK(ifmBox.SizeShape().Width() == 1);

    stripe = static_cast<HLCStripe *>(cmdStream[1].get());
    const auto &paddingStripe1 = stripe->padding;
    const auto &ifmBoxStripe1 = stripe->stripeAreas[0].ifmAreas[0];
    CHECK(paddingStripe1.top == 0);
    CHECK(paddingStripe1.left == 0);
    CHECK(paddingStripe1.bottom == 0);
    CHECK(paddingStripe1.right == 0);
    CHECK(ifmBoxStripe1.Start().Height() == 0);
    CHECK(ifmBoxStripe1.Start().Width() == 0);
    CHECK(ifmBoxStripe1.SizeShape().Height() == 3);
    CHECK(ifmBoxStripe1.SizeShape().Width() == 1);

    stripe = static_cast<HLCStripe *>(cmdStream[2].get());
    const auto &paddingStripe2 = stripe->padding;
    const auto &ifmBoxStripe2 = stripe->stripeAreas[0].ifmAreas[0];
    CHECK(paddingStripe2.top == 0);
    CHECK(paddingStripe2.left == 0);
    CHECK(paddingStripe2.bottom == 0);
    CHECK(paddingStripe2.right == 0);
    CHECK(ifmBoxStripe2.Start().Height() == 1);
    CHECK(ifmBoxStripe2.Start().Width() == 0);
    CHECK(ifmBoxStripe2.SizeShape().Height() == 3);
    CHECK(ifmBoxStripe2.SizeShape().Width() == 1);
}

TEST_CASE("CalculateIfmStripeAndPadding reports padding on boundary stripes strided version")
{
    Kernel kernel(Point2i{1, 3}, Point2i{1, 2}, Point2i{1, 1}, Margin(0, 0, 0, 0));
    Shape ifmShape(1, 9, 1, 1);
    Shape ofmShape(1, 4, 1, 1);
    Shape stripeShape(1, 1, 1, 1);
    Point2i ifmStep{1, 1};
    Point2i ofmStep{1, 1};

    std::unique_ptr<SchedulerOperation> schedOp = CreateSchedulerOperation(OpType::Conv2D, kernel, ifmShape, ofmShape, ifmStep, ofmStep);
    auto schedule = CreateSchedule(schedOp, stripeShape);
    auto npuOp = CreateNpuOperation(std::move(schedOp));
    HLCStreamGenerator hlcsGen(0, false);
    SubGraphs subGraphs;
    HLCStream cmdStream = hlcsGen.GenerateCommandStream(&npuOp, &schedule, subGraphs);
    REQUIRE(cmdStream.size() == 4);
    auto &command0 = cmdStream[0];
    REQUIRE(command0->CommandType() == HighLevelCommandType::STRIPE);
    HLCStripe *stripe = static_cast<HLCStripe *>(command0.get());
    auto &padding = stripe->padding;
    auto &ifmBox = stripe->stripeAreas[0].ifmAreas[0];
    CHECK(padding.top == 0);
    CHECK(padding.left == 0);
    CHECK(padding.bottom == 0);
    CHECK(padding.right == 0);
    CHECK(ifmBox.Start().Height() == 0);
    CHECK(ifmBox.Start().Width() == 0);
    CHECK(ifmBox.SizeShape().Height() == 3);
    CHECK(ifmBox.SizeShape().Width() == 1);
}


TEST_CASE("CalculateIfmStripeAndPadding, double stepping")
{
    // Stripe (with step 1) over height 2 in the OFM.
    // Kernel-height 2 and stride 2.
    Kernel kernel(Point2i{1, 2}, Point2i{1, 2}, Point2i{1, 1}, Margin(0, 0, 0, 0));
    Shape ifmShape(1, 4, 1, 1);
    Shape ofmShape(1, 2, 1, 1);
    Shape stripeShape(1, 1, 1, 1);
    Point2i ifmStep{1, 1};
    Point2i ofmStep{1, 1};
    std::unique_ptr<SchedulerOperation> schedOp = CreateSchedulerOperation(OpType::Conv2D, kernel, ifmShape, ofmShape, ifmStep, ofmStep);
    auto schedule = CreateSchedule(schedOp, stripeShape);
    auto npuOp = CreateNpuOperation(std::move(schedOp));
    HLCStreamGenerator hlcsGen(0, false);
    SubGraphs subGraphs;
    HLCStream cmdStream = hlcsGen.GenerateCommandStream(&npuOp, &schedule, subGraphs);
    REQUIRE(cmdStream.size() == 2);

    SECTION("first stripe")
    {
        auto &command0 = cmdStream[0];
        REQUIRE(command0->CommandType() == HighLevelCommandType::STRIPE);
        HLCStripe *stripe = static_cast<HLCStripe *>(command0.get());
        auto &padding = stripe->padding;
        auto &ifmBox = stripe->stripeAreas[0].ifmAreas[0];
        // stripe-padding should be unchanged
        CHECK(padding.top == 0);
        CHECK(padding.left == 0);
        CHECK(padding.bottom == 0);
        CHECK(padding.right == 0);
        // start at IFM start
        CHECK(ifmBox.Start().Height() == 0);
        CHECK(ifmBox.Start().Width() == 0);
        // end at start + kernel size
        CHECK(ifmBox.End().Height() == 2);
        CHECK(ifmBox.End().Width() == 1);
        CHECK(ifmBox.SizeShape().Height() == 2);
        CHECK(ifmBox.SizeShape().Width() == 1);
    }
    SECTION("second stripe")
    {
        HLCStripe *stripe = static_cast<HLCStripe *>(cmdStream[1].get());
        const auto &paddingStripe1 = stripe->padding;
        const auto &ifmBoxStripe1 = stripe->stripeAreas[0].ifmAreas[0];
        // stripe-padding should be unchanged
        CHECK(paddingStripe1.top == 0);
        CHECK(paddingStripe1.left == 0);
        CHECK(paddingStripe1.bottom == 0);
        CHECK(paddingStripe1.right == 0);
        // second stripe starts in 0 + stride (2)
        CHECK(ifmBoxStripe1.Start().Height() == 2);
        CHECK(ifmBoxStripe1.Start().Width() == 0);
        // end at start + kernel size
        CHECK(ifmBoxStripe1.End().Height() == 4);
        CHECK(ifmBoxStripe1.End().Width() == 1);
        CHECK(ifmBoxStripe1.SizeShape().Height() == 2);
        CHECK(ifmBoxStripe1.SizeShape().Width() == 1);
    }
}

TEST_CASE("CalculateIfmStripeAndPadding, ofm step")
{
    Kernel kernel(Point2i{1, 2}, Point2i{1, 2}, Point2i{1, 1}, Margin(0, 0, 0, 0));
    Shape ifmShape(1, 6, 1, 1);
    Shape ofmShape(1, 5, 1, 1);
    Shape stripeShape(1, 1, 1, 1);
    Point2i ifmStep{1, 1};
    Point2i ofmStep{1, 2};
    std::unique_ptr<SchedulerOperation> schedOp = CreateSchedulerOperation(OpType::Conv2D, kernel, ifmShape, ofmShape, ifmStep, ofmStep);
    auto schedule = CreateSchedule(schedOp, stripeShape);
    auto npuOp = CreateNpuOperation(std::move(schedOp));
    HLCStreamGenerator hlcsGen(0, false);
    SubGraphs subGraphs;
    HLCStream cmdStream = hlcsGen.GenerateCommandStream(&npuOp, &schedule, subGraphs);
    REQUIRE(cmdStream.size() == 3);
    // OFM height of 5, striped in stripes of height 1, with OFM-step 2.
    // Should result in 3 stripes with starts: 0, 2, 4
    HLCStripe *stripe = static_cast<HLCStripe *>(cmdStream[0].get());
    const auto &padding = stripe->padding;
    const auto &ifmBox = stripe->stripeAreas[0].ifmAreas[0];
    CHECK(padding.top == 0);
    CHECK(padding.left == 0);
    CHECK(padding.bottom == 0);
    CHECK(padding.right == 0);
    // first stripe starts in 0,0
    CHECK(ifmBox.Start().Height() == 0);
    CHECK(ifmBox.Start().Width() == 0);
    // height should be equal to kernel-size
    // as each stripe has only one OFM-element
    CHECK(ifmBox.End().Height() == 2);
    CHECK(ifmBox.SizeShape().Height() == 2);
    CHECK(ifmBox.SizeShape().Width() == 1);
}


TEST_CASE("CalculateIfmStripeAndPadding handles input stepping for padding (height-only)")
{
    Shape ifmShape(1, 23, 1, 1);
    Shape ofmShape(1, 10, 1, 1);
    Shape stripeShape(1, 1, 1, 1);
    Point2i ifmStep{1, 2};
    Point2i ofmStep{1, 1};
    Kernel kernel(Point2i{1, 5}, Point2i{1, 1}, Point2i{1, 1}, Margin(4, 0, 0, 0));
    // Stepping is not applied in the padded area,
    // so padding is re-calculated based on stepping
    //
    // for the first two stripes:
    //   1. First stripe starts in -4 (original padding)
    //      - first element -4
    //      - second element in -2
    //      - third element in 0
    //      This sums up to a re-calculated padding of 2  (the -4 and -2 elements)
    //      and an IFM-start in 0.
    //   2. Second stripe starts in -2 (original padding)
    //      - first element in -2
    //      - second element in 0
    //      This sums up to a re-calculated padding of 1 (the -2)
    //      and an IFM-start in 0
    std::array<std::tuple<int, int, int>, 10> expectedPadStartEnd = {
        {// Some examples for explaination
         // (stepping is not applied in the padded region)
         // p are padded values that are actually multiplied with the kernel
         // x are IFM-values that we reach when stepping
         // . are stepped-over ifm-values
         // For kernel-size 5, we need the sum of p's and x's to add up to 5.
         // The sum of x's and .'s determine our actual IFM-size.
         // padding, start, end
         // clang-format off
        {2, 0, 5},   // ppx.x.x   (size 5)
        {1, 0, 7},   // px.x.x.x  (size 7)
        {0, 0, 9},   // x.x.x.x.x (size 9)
        {0, 2, 11},  // the rest are size 9, start is just offset by ifm-step
        {0, 4, 13},
        {0, 6, 15},
        {0, 8, 17},
        {0, 10, 19},
        {0, 12, 21},
        {0, 14, 23}}
        // clang-format on
    };

    std::unique_ptr<SchedulerOperation> schedOp = CreateSchedulerOperation(OpType::Conv2D, kernel, ifmShape, ofmShape, ifmStep, ofmStep);
    auto schedule = CreateSchedule(schedOp, stripeShape);
    auto npuOp = CreateNpuOperation(std::move(schedOp));
    HLCStreamGenerator hlcsGen(0, false);
    SubGraphs subGraphs;
    HLCStream cmdStream = hlcsGen.GenerateCommandStream(&npuOp, &schedule, subGraphs);
    REQUIRE(cmdStream.size() == 10);
    for ( size_t i = 0; i < cmdStream.size(); i++ )
    {
        HLCStripe *stripe = static_cast<HLCStripe *>(cmdStream[i].get());
        const auto &padding = stripe->padding;
        const auto &ifmBox = stripe->stripeAreas[0].ifmAreas[0];
        auto [expectedPadding, expectedStart, expectedEnd] = expectedPadStartEnd[i];
        CHECK(padding.top == expectedPadding);
        CHECK(padding.left == 0);
        CHECK(padding.bottom == 0);
        CHECK(padding.right == 0);
        CHECK(ifmBox.Start().Height() == expectedStart);
        CHECK(ifmBox.Start().Width() == 0);
        CHECK(ifmBox.End().Height() == expectedEnd);
        CHECK(ifmBox.End().Width() == 1);
    }
}

TEST_CASE("CalculateIfmStripeAndPadding handles sliced convolutions correctly")
{
    // ifm volumes
    Shape ifmShape(1, 100, 100, 10);
    Shape ifmSliceShape(1, 10, 25, 5);
    Shape ifmSliceOffset(0, 5, 4, 2);
    // ofm volumes
    Shape ofmShape(1, 50, 50, 20);
    Shape ofmSliceShape = ifmSliceShape;
    Shape ofmSliceOffset(0, 1, 3, 4);
    // stripe volume (no striping)
    Shape stripeShape = ofmSliceShape;
    Point2i ifmStep(1, 1);
    Point2i ofmStep(1, 1);
    // unit kernel
    Kernel kernel(Point2i(1, 1), Point2i(1, 1), Point2i(1, 1), Margin(0, 0, 0, 0));

    std::unique_ptr<SchedulerOperation> schedOp = CreateSchedulerOperation(OpType::Conv2D, kernel, ifmShape, ofmShape, ifmStep, ofmStep);
    schedOp->Input(TensorUsage::IFM)->slice.Initialize(ifmSliceOffset, ifmSliceShape);
    schedOp->Output(TensorUsage::OFM)->slice.Initialize(ofmSliceOffset, ofmSliceShape);
    auto schedule = CreateSchedule(schedOp, stripeShape);
    auto npuOp = CreateNpuOperation(std::move(schedOp));
    HLCStreamGenerator hlcsGen(0, false);
    SubGraphs subGraphs;
    HLCStream cmdStream = hlcsGen.GenerateCommandStream(&npuOp, &schedule, subGraphs);
    REQUIRE(cmdStream.size() == 1);

    HLCStripe *stripe = static_cast<HLCStripe *>(cmdStream[0].get());
    const auto &padding = stripe->padding;
    const auto &ifmBox = stripe->stripeAreas[0].ifmAreas[0];
    CHECK(padding.top == 0);
    CHECK(padding.left == 0);
    CHECK(padding.bottom == 0);
    CHECK(padding.right == 0);
    CHECK(ifmBox.Start().Height() == ifmSliceOffset.Height());
    CHECK(ifmBox.Start().Width() == ifmSliceOffset.Width());
    CHECK(ifmBox.Start().Depth() == ifmSliceOffset.Depth());
    CHECK(ifmBox.End().Height() == ifmSliceOffset.Height() + ifmSliceShape.Height());
    CHECK(ifmBox.End().Width() == ifmSliceOffset.Width() + ifmSliceShape.Width());
    CHECK(ifmBox.End().Depth() == ifmSliceOffset.Depth() + ifmSliceShape.Depth());
}
