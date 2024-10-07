//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "compiler/scheduler_decompose.hpp"
#include "util.hpp"

#include <catch_all.hpp>

#include "regor.h"


using namespace regor;

namespace
{

std::unique_ptr<SchedulerOperation> CreateOperation(OpType opType, Shape ifmShape, Shape ifm2Shape, Shape ofmShape)
{
    auto ifm1 = CreateSchedulerTensor("ifm1", ifmShape, DataType::Int8);
    auto ifm2 = CreateSchedulerTensor("ifm2", ifm2Shape, DataType::Int8);
    auto ofm = CreateSchedulerTensor("ofm", ofmShape, DataType::Int8);

    std::unique_ptr<SchedulerOperation> op = CreateSchedulerOperation(
        opType, TensorUsage::IFM0, ifm1, TensorUsage::IFM1, ifm2, TensorUsage::OFM, ofm);

    // set default kernel
    op->_kernel = std::make_unique<class Kernel>(Point2i(1, 1), Point2i(1, 1), Point2i(1, 1));
    return op;
}

std::unique_ptr<SchedulerOperation> CreateOperation(OpType opType, Shape ifmShape, Shape ofmShape)
{
    auto ifm1 = CreateSchedulerTensor("ifm1", ifmShape, DataType::Int8);
    auto ofm = CreateSchedulerTensor("ofm", ofmShape, DataType::Int8);

    std::unique_ptr<SchedulerOperation> op = CreateSchedulerOperation(opType, TensorUsage::IFM0, ifm1, TensorUsage::OFM, ofm);

    // set default kernel
    op->_kernel = std::make_unique<class Kernel>(Point2i(1, 1), Point2i(1, 1), Point2i(1, 1));
    return op;
}

};  // namespace

TEST_CASE("test_scheduler_decompose")
{
    SECTION("Decompose matmul in height dimension")
    {
        Shape ifmShape(1, 100, 3, 2);  // ifm2 is transposed by graphIR optimiser to same shape as ifm1
        Shape ofmShape(1, 100, 3, 3);
        auto op = CreateOperation(OpType::MatMul, ifmShape, ifmShape, ofmShape);
        std::vector<std::unique_ptr<SchedulerOperation>> decomposedOps = DecomposeMatmul(nullptr, std::move(op));
        REQUIRE(decomposedOps.size() == 100);
        for ( size_t i = 0; i < decomposedOps.size(); i++ )
        {
            auto &subOp = decomposedOps[i];
            auto &ifmSlice = subOp->Input(TensorUsage::IFM0)->slice;
            auto &ifm2Slice = subOp->Input(TensorUsage::IFM1)->slice;
            auto &ofmSlice = subOp->Output(TensorUsage::OFM)->slice;
            REQUIRE(ifmSlice.shape == ifmShape.WithHeight(1));
            REQUIRE(ifm2Slice.shape == ifmShape.WithHeight(1));
            REQUIRE(ofmSlice.shape == ofmShape.WithHeight(1));
            REQUIRE(ifmSlice.offset == Shape(0, i, 0, 0));
            REQUIRE(ifm2Slice.offset == Shape(0, i, 0, 0));
            REQUIRE(ofmSlice.offset == Shape(0, i, 0, 0));
        }
    }
    SECTION("Decompose matmul in height dimension with input/output slice")
    {
        Shape ifmShape(1, 100, 3, 2);  // ifm2 is transposed by graphIR optimiser to same shape as ifm1
        Shape ofmShape(1, 100, 3, 3);
        Shape ifmSliceOffset(0, 1, 0, 0);
        Shape ifmSliceShape(1, 98, 3, 2);
        Shape ofmSliceOffset(0, 1, 0, 0);
        Shape ofmSliceShape(1, 98, 3, 3);
        auto op = CreateOperation(OpType::MatMul, ifmShape, ifmShape, ofmShape);
        op->Input(TensorUsage::IFM0)->slice = {ifmSliceOffset, ifmSliceShape};
        op->Input(TensorUsage::IFM1)->slice = {ifmSliceOffset, ifmSliceShape};
        op->Output(TensorUsage::OFM)->slice = {ofmSliceOffset, ofmSliceShape};
        std::vector<std::unique_ptr<SchedulerOperation>> decomposedOps = DecomposeMatmul(nullptr, std::move(op));
        REQUIRE(decomposedOps.size() == 98);
        for ( size_t i = 0; i < decomposedOps.size(); i++ )
        {
            auto &subOp = decomposedOps[i];
            auto &ifmSlice = subOp->Input(TensorUsage::IFM0)->slice;
            auto &ifm2Slice = subOp->Input(TensorUsage::IFM1)->slice;
            auto &ofmSlice = subOp->Output(TensorUsage::OFM)->slice;
            REQUIRE(ifmSlice.shape == ifmSliceShape.WithHeight(1));
            REQUIRE(ifm2Slice.shape == ifmSliceShape.WithHeight(1));
            REQUIRE(ofmSlice.shape == ofmSliceShape.WithHeight(1));
            REQUIRE(ifmSlice.offset == Shape(0, i, 0, 0) + ifmSliceOffset);
            REQUIRE(ifm2Slice.offset == Shape(0, i, 0, 0) + ifmSliceOffset);
            REQUIRE(ofmSlice.offset == Shape(0, i, 0, 0) + ofmSliceOffset);
        }
    }
    SECTION("Decompose matmul in batch dimension")
    {
        Shape ifmShape(100, 1, 3, 2);
        Shape ofmShape(100, 1, 3, 3);
        auto op = CreateOperation(OpType::MatMul, ifmShape, ifmShape, ofmShape);
        std::vector<std::unique_ptr<SchedulerOperation>> decomposedOps = DecomposeMatmul(nullptr, std::move(op));
        REQUIRE(decomposedOps.size() == 100);
        for ( size_t i = 0; i < decomposedOps.size(); i++ )
        {
            auto &subOp = decomposedOps[i];
            auto &ifmSlice = subOp->Input(TensorUsage::IFM0)->slice;
            auto &ifm2Slice = subOp->Input(TensorUsage::IFM1)->slice;
            auto &ofmSlice = subOp->Output(TensorUsage::OFM)->slice;
            REQUIRE(ifmSlice.shape == ifmShape.WithBatch(1));
            REQUIRE(ifm2Slice.shape == ifmShape.WithBatch(1));
            REQUIRE(ofmSlice.shape == ofmShape.WithBatch(1));
            REQUIRE(ifmSlice.offset == Shape(i, 0, 0, 0));
            REQUIRE(ifm2Slice.offset == Shape(i, 0, 0, 0));
            REQUIRE(ofmSlice.offset == Shape(i, 0, 0, 0));
        }
    }
    SECTION("Decompose matmul in batch dimension with input/output slice")
    {
        Shape ifmShape(100, 1, 3, 2);
        Shape ofmShape(100, 1, 3, 3);
        Shape ifmSliceOffset(1, 0, 0, 0);
        Shape ifmSliceShape(98, 1, 3, 2);
        Shape ofmSliceOffset(1, 0, 0, 0);
        Shape ofmSliceShape(98, 1, 3, 3);
        auto op = CreateOperation(OpType::MatMul, ifmShape, ifmShape, ofmShape);
        op->Input(TensorUsage::IFM0)->slice = {ifmSliceOffset, ifmSliceShape};
        op->Input(TensorUsage::IFM1)->slice = {ifmSliceOffset, ifmSliceShape};
        op->Output(TensorUsage::OFM)->slice = {ofmSliceOffset, ofmSliceShape};
        std::vector<std::unique_ptr<SchedulerOperation>> decomposedOps = DecomposeMatmul(nullptr, std::move(op));
        REQUIRE(decomposedOps.size() == 98);
        for ( size_t i = 0; i < decomposedOps.size(); i++ )
        {
            auto &subOp = decomposedOps[i];
            auto &ifmSlice = subOp->Input(TensorUsage::IFM0)->slice;
            auto &ifm2Slice = subOp->Input(TensorUsage::IFM1)->slice;
            auto &ofmSlice = subOp->Output(TensorUsage::OFM)->slice;
            REQUIRE(ifmSlice.shape == ifmSliceShape.WithBatch(1));
            REQUIRE(ifm2Slice.shape == ifmSliceShape.WithBatch(1));
            REQUIRE(ofmSlice.shape == ofmSliceShape.WithBatch(1));
            REQUIRE(ifmSlice.offset == (Shape(i, 0, 0, 0) + ifmSliceOffset));
            REQUIRE(ifm2Slice.offset == Shape(i, 0, 0, 0) + ifmSliceOffset);
            REQUIRE(ofmSlice.offset == Shape(i, 0, 0, 0) + ofmSliceOffset);
        }
    }
    SECTION("Decompose matmul in height and batch")
    {
        Shape ifmShape(10, 10, 3, 2);
        Shape ofmShape(10, 10, 3, 3);
        auto op = CreateOperation(OpType::MatMul, ifmShape, ifmShape, ofmShape);
        std::vector<std::unique_ptr<SchedulerOperation>> decomposedOps = DecomposeMatmul(nullptr, std::move(op));
        REQUIRE(decomposedOps.size() == 100);
        for ( size_t i = 0; i < decomposedOps.size(); i++ )
        {
            auto &subOp = decomposedOps[i];
            auto &ifmSlice = subOp->Input(TensorUsage::IFM0)->slice;
            auto &ifm2Slice = subOp->Input(TensorUsage::IFM1)->slice;
            auto &ofmSlice = subOp->Output(TensorUsage::OFM)->slice;
            REQUIRE(ifmSlice.shape == ifmShape.WithHeight(1).WithBatch(1));
            REQUIRE(ifm2Slice.shape == ifmShape.WithHeight(1).WithBatch(1));
            REQUIRE(ofmSlice.shape == ofmShape.WithHeight(1).WithBatch(1));

            int expectedHeightOffset = i / 10;
            int expectedBatchOffset = i % 10;
            REQUIRE(ifmSlice.offset.Height() == expectedHeightOffset);
            REQUIRE(ifmSlice.offset.Batch() == expectedBatchOffset);
            REQUIRE(ifm2Slice.offset.Height() == expectedHeightOffset);
            REQUIRE(ifm2Slice.offset.Batch() == expectedBatchOffset);
            REQUIRE(ofmSlice.offset.Height() == expectedHeightOffset);
            REQUIRE(ofmSlice.offset.Batch() == expectedBatchOffset);
        }
    }
    SECTION("Decompose matmul in height and batch with input/output slice")
    {
        Shape ifmShape(10, 10, 3, 2);
        Shape ofmShape(10, 10, 3, 3);
        Shape ifmSliceOffset(1, 1, 0, 0);
        Shape ifmSliceShape(8, 8, 3, 2);
        Shape ofmSliceOffset(1, 1, 0, 0);
        Shape ofmSliceShape(8, 8, 3, 3);
        auto op = CreateOperation(OpType::MatMul, ifmShape, ifmShape, ofmShape);
        op->Input(TensorUsage::IFM0)->slice = {ifmSliceOffset, ifmSliceShape};
        op->Input(TensorUsage::IFM1)->slice = {ifmSliceOffset, ifmSliceShape};
        op->Output(TensorUsage::OFM)->slice = {ofmSliceOffset, ofmSliceShape};
        std::vector<std::unique_ptr<SchedulerOperation>> decomposedOps = DecomposeMatmul(nullptr, std::move(op));
        REQUIRE(decomposedOps.size() == 64);
        for ( size_t i = 0; i < decomposedOps.size(); i++ )
        {
            auto &subOp = decomposedOps[i];
            auto &ifmSlice = subOp->Input(TensorUsage::IFM0)->slice;
            auto &ifm2Slice = subOp->Input(TensorUsage::IFM1)->slice;
            auto &ofmSlice = subOp->Output(TensorUsage::OFM)->slice;
            REQUIRE(ifmSlice.shape == ifmSliceShape.WithHeight(1).WithBatch(1));
            REQUIRE(ifm2Slice.shape == ifmSliceShape.WithHeight(1).WithBatch(1));
            REQUIRE(ofmSlice.shape == ofmSliceShape.WithHeight(1).WithBatch(1));

            int expectedHeightOffset = i / 8;
            int expectedBatchOffset = i % 8;
            REQUIRE(ifmSlice.offset.Height() == ifmSliceOffset.Height() + expectedHeightOffset);
            REQUIRE(ifmSlice.offset.Batch() == ifmSliceOffset.Height() + expectedBatchOffset);
            REQUIRE(ifm2Slice.offset.Height() == ifmSliceOffset.Height() + expectedHeightOffset);
            REQUIRE(ifm2Slice.offset.Batch() == ifmSliceOffset.Height() + expectedBatchOffset);
            REQUIRE(ofmSlice.offset.Height() == ofmSliceOffset.Height() + expectedHeightOffset);
            REQUIRE(ofmSlice.offset.Batch() == ofmSliceOffset.Height() + expectedBatchOffset);
        }
    }
    SECTION("Decompose valid matmul")
    {
        // Expect no change when calling DecomposeMatmul
        Shape ifmShape(1, 1, 3, 2);
        Shape ofmShape(1, 1, 3, 3);
        auto op = CreateOperation(OpType::MatMul, ifmShape, ifmShape, ofmShape);
        SchedulerOperation *orig = op.get();
        std::vector<std::unique_ptr<SchedulerOperation>> decomposedOps = DecomposeMatmul(nullptr, std::move(op));
        REQUIRE(decomposedOps.size() == 1);
        REQUIRE(orig == decomposedOps[0].get());
    }
    SECTION("Decompose large axis")
    {
        uint32_t maxSize = (1UL << 16);
        uint32_t shapeSize = maxSize * 10 + 5;
        Shape ifmShape(1, 1, shapeSize, 5);
        Shape ofmShape(1, 1, shapeSize, 5);
        auto op = CreateOperation(OpType::ReduceMax, ifmShape, ofmShape);
        std::vector<std::unique_ptr<SchedulerOperation>> decomposedOps = DecomposeReduce(nullptr, std::move(op));
        REQUIRE(decomposedOps.size() == 11);
        for ( size_t i = 0; i < decomposedOps.size(); i++ )
        {
            auto &subOp = decomposedOps[i];
            auto &ifmSlice = subOp->Input(TensorUsage::IFM0)->slice;
            auto &ofmSlice = subOp->Output(TensorUsage::OFM)->slice;
            int expectedWidth = i < (decomposedOps.size() - 1) ? maxSize : 5;
            REQUIRE(ifmSlice.shape == ifmShape.WithWidth(expectedWidth));
            REQUIRE(ofmSlice.shape == ofmShape.WithWidth(expectedWidth));
        }
    }
    SECTION("Decompose large axis (sliced)")
    {
        uint32_t maxSize = (1UL << 16);
        uint32_t shapeSize = maxSize * 10 + 5;
        Shape ifmShape(1, 1, shapeSize, 50);
        Shape ofmShape(1, 1, shapeSize, 50);
        // slice IFM/OFM into
        // W = 2 * maxSize + 7
        // C = 10
        // expect 3 decomposed ops
        //     two with w = maxSize
        //     one with w = 7
        Shape ifmSliceOffset(0, 0, maxSize, 10);
        Shape ofmSliceOffset(0, 0, 0, 10);
        Shape ifmSliceShape(1, 1, maxSize * 2 + 7, 10);
        Shape ofmSliceShape(1, 1, maxSize * 2 + 7, 10);
        auto op = CreateOperation(OpType::ReduceMax, ifmShape, ofmShape);
        op->Input(TensorUsage::IFM0)->slice = {ifmSliceOffset, ifmSliceShape};
        op->Output(TensorUsage::OFM)->slice = {ofmSliceOffset, ofmSliceShape};
        std::vector<std::unique_ptr<SchedulerOperation>> decomposedOps = DecomposeReduce(nullptr, std::move(op));
        REQUIRE(decomposedOps.size() == 3);
        for ( size_t i = 0; i < decomposedOps.size(); i++ )
        {
            auto &subOp = decomposedOps[i];
            auto &ifmSlice = subOp->Input(TensorUsage::IFM0)->slice;
            auto &ofmSlice = subOp->Output(TensorUsage::OFM)->slice;
            int expectedWidth = i < (decomposedOps.size() - 1) ? maxSize : 7;
            REQUIRE(ifmSlice.shape == ifmSliceShape.WithWidth(expectedWidth));
            REQUIRE(ofmSlice.shape == ofmSliceShape.WithWidth(expectedWidth));
            REQUIRE(ofmSlice.offset == (ofmSliceOffset + Shape(0, 0, i * maxSize, 0)));
            REQUIRE(ifmSlice.offset == (ifmSliceOffset + Shape(0, 0, i * maxSize, 0)));
        }
    }
}
