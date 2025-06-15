//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "common/logging.hpp"

#include "architecture/ethosu55/ethos_u55.hpp"
#include "architecture/ethosu65/ethos_u65.hpp"
#include "architecture/ethosu85/ethos_u85.hpp"
#include "tflite/tflite_supported_operators.hpp"
#include "tflite/tflite_supported_operators_u55.hpp"
#include "tflite/tflite_supported_operators_u85.hpp"
#include "util.hpp"

#include <catch_all.hpp>

#include "include/regor.h"

using namespace regor;

namespace
{

std::shared_ptr<Operation> CreateOperation(OpType opType, Shape ifmShape, DataType ifmType, Shape ifm2Shape,
    DataType ifm2Type, Shape ofmShape, DataType ofmType)
{
    auto ifm = CreateTensor("IFM", ifmShape, ifmType);
    auto ofm = CreateTensor("OFM", ofmShape, ofmType);
    std::shared_ptr<Operation> op = ::CreateOperation(opType, TensorUsage::IFM, ifm, TensorUsage::OFM, ofm);

    if ( ifm2Shape )
    {
        auto ifm2 = CreateTensor("IFM2", ifm2Shape, ifm2Type);
        op->ConnectInput(TensorUsage::IFM1, ifm2).Set(Quantization::Unit());
    }
    if ( opType == OpType::Conv2D )
    {
        auto weights = CreateTensor("weights", Shape(1, 1, 1, 1), DataType::Int8);
        weights->SetAxisOrder(AxisOrder::OHWI);
        op->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
    }
    return op;
};

std::shared_ptr<Operation> CreateOperation(OpType opType, Shape ifmShape, DataType ifmType, Shape ofmShape, DataType ofmType)
{
    return CreateOperation(opType, ifmShape, ifmType, Shape(), DataType::None, ofmShape, ofmType);
};

}  // namespace

TEST_CASE("Supported operators Common")
{
    DisableLogging disableLogging;

    std::shared_ptr<Architecture> arch = CreateArchDefault<ArchEthosU55>(256);
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");
    auto supportedOps = MakeSupportedOpsChecker(REGOR_ARCH_ETHOSU55, arch->Constraints());

    SECTION("ConstraintTensQuantized")
    {
        auto op = CreateOperation(OpType::Conv2D, Shape(1, 8, 8, 1), DataType::Int8, Shape(1, 8, 8, 1), DataType::Int8);
        std::vector<int8_t> values = {1};
        auto weights = CreateTensor("weights", Shape(1, 1, 1, 1), DataType::Int8, std::move(values));
        weights->SetAxisOrder(AxisOrder::OHWI);
        op->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
        // Regular op should pass
        REQUIRE(supportedOps->Check(op.get()) == true);
        auto &quant = op->Output(TensorUsage::OFM)->quantization;
        // Removing scales should fail
        quant.scales.clear();
        REQUIRE(supportedOps->Check(op.get()) == false);
        quant = Quantization::Unit();
        quant.zeroPoints.clear();
        REQUIRE(supportedOps->Check(op.get()) == false);
        op->Disconnect();
    }
    SECTION("ConstraintMustHaveIFM")
    {
        auto op = CreateOperation(OpType::Exp, Shape(1, 8, 8, 1), DataType::Int8, Shape(1, 8, 8, 1), DataType::Int8);
        op->DisconnectInputInvalidatingInputs(TensorUsage::IFM);
        REQUIRE(supportedOps->Check(op.get()) == false);
        op->Disconnect();
    }
    SECTION("ConstraintMustHaveOFM")
    {
        auto op = CreateOperation(OpType::Exp, Shape(1, 8, 8, 1), DataType::Int8, Shape(1, 8, 8, 1), DataType::Int8);
        auto ifm = op->Input(TensorUsage::IFM0)->tensor;
        op->Disconnect();
        op->ConnectInput(TensorUsage::IFM0, ifm);
        REQUIRE(supportedOps->Check(op.get()) == false);
        op->Disconnect();
    }
    SECTION("ConstraintMustHaveShape")
    {
        auto op = CreateOperation(OpType::Add, Shape(1, 8, 8, 1), DataType::Int8, Shape(1, 8, 8, 1), DataType::Int8,
            Shape(1, 8, 8, 1), DataType::Int8);
        op->Output(TensorUsage::OFM)->shape = Shape();
        REQUIRE(supportedOps->Check(op.get()) == false);
        op->Disconnect();
    }
    SECTION("ConstraintFCWeightShape")
    {
        auto op = CreateOperation(OpType::FullyConnected, Shape(1, 2, 2, 1), DataType::Int8, Shape(1, 2, 1, 1), DataType::Int8);
        std::vector<int8_t> values = {1, 1, 1, 1, 1, 1, 1, 1};
        auto weights = CreateTensor("weights", Shape(4, 1, 1, 2), DataType::Int8, std::move(values));
        weights->SetAxisOrder(AxisOrder::OHWI);
        op->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
        REQUIRE(supportedOps->Check(op.get()) == true);
        // reshape and reconnect tensor
        weights->Reshape(Shape(2, 2, 1, 2));
        op->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
        REQUIRE(supportedOps->Check(op.get()) == false);
        op->Disconnect();
    }

    SECTION("ConstraintPerAxisQuant")
    {
        auto op = CreateOperation(OpType::Add, Shape(1, 1, 1, 3), DataType::Int8, Shape(1, 1, 1, 3), DataType::Int8,
            Shape(1, 1, 1, 3), DataType::Int8);
        REQUIRE(supportedOps->Check(op.get()) == true);
        Quantization q = Quantization::Unit();
        q.scales.push_back({8, 2});
        q.zeroPoints.push_back(2);
        op->Input(TensorUsage::IFM)->Set(q);
        REQUIRE(supportedOps->Check(op.get()) == false);
        op->Disconnect();
    }

    SECTION("ConstraintMatchingQuantization")
    {
        auto op = CreateOperation(OpType::Minimum, Shape(1, 1, 1, 1), DataType::Int8, Shape(1, 1, 1, 1), DataType::Int8,
            Shape(1, 1, 1, 1), DataType::Int8);
        REQUIRE(supportedOps->Check(op.get()) == true);
        Quantization q;
        q.scales.push_back(8);
        q.zeroPoints.push_back(2);
        op->Input(TensorUsage::IFM)->Set(q);
        REQUIRE(supportedOps->Check(op.get()) == false);
        op->Disconnect();
    }

    SECTION("ConstraintWeightsPrecision")
    {
        auto op = CreateOperation(OpType::DepthwiseConv2D, Shape(1, 5, 5, 1), DataType::Int8, Shape(1, 5, 5, 1), DataType::Int8);
        {
            std::vector<int8_t> values(1, 1);
            auto weights = CreateTensor("weights", Shape(1, 1, 1, 1), DataType::Int8, std::move(values));
            weights->SetAxisOrder(AxisOrder::IHWO);
            op->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
            REQUIRE(supportedOps->Check(op.get()) == true);
        }
        {
            std::vector<uint8_t> values(1, 1);
            auto weights = CreateTensor("weights", Shape(1, 1, 1, 1), DataType::UInt8, std::move(values));
            weights->SetAxisOrder(AxisOrder::IHWO);
            op->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
            REQUIRE(supportedOps->Check(op.get()) == true);
        }
        {
            std::vector<int16_t> values(1, 1);
            auto weights = CreateTensor("weights", Shape(1, 1, 1, 1), DataType::Int16, std::move(values));
            weights->SetAxisOrder(AxisOrder::IHWO);
            op->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
            REQUIRE(supportedOps->Check(op.get()) == false);
        }
        {
            std::vector<int32_t> values(1, 1);
            auto weights = CreateTensor("weights", Shape(1, 1, 1, 1), DataType::Int32, std::move(values));
            weights->SetAxisOrder(AxisOrder::IHWO);
            op->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
            REQUIRE(supportedOps->Check(op.get()) == false);
        }
        op->Disconnect();
    }

    SECTION("ConstraintWeightSum")
    {
        auto op = CreateOperation(OpType::DepthwiseConv2D, Shape(1, 1, 32768, 2), DataType::Int8, Shape(1, 1, 1, 2), DataType::Int8);
        static const int64_t MAX_SUM = (1 << 16) * 127;
        {
            // Verify supported sum of weights
            std::vector<int8_t> values((1 << 16) * 2, 127);
            auto weights = CreateTensor("weights", Shape(1, 1, (1 << 16), 2), DataType::Int8, std::move(values));
            weights->SetAxisOrder(AxisOrder::IHWO);
            op->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
            REQUIRE(supportedOps->Check(op.get()) == true);
        }
        {
            // Verify unsupported sum of weights
            std::vector<uint8_t> values((1 << 16) * 2, 127);
            auto weights = CreateTensor("weights", Shape(1, 1, (1 << 16), 2), DataType::Int8, std::move(values));
            weights->SetAxisOrder(AxisOrder::OHWI);
            op->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
            REQUIRE(supportedOps->Check(op.get()) == false);
        }
        op->Disconnect();
    }

    SECTION("ConstraintBias")
    {
        auto op = CreateOperation(OpType::DepthwiseConv2D, Shape(1, 5, 5, 2), DataType::Int8, Shape(1, 5, 5, 2), DataType::Int8);
        std::vector<int8_t> wValues(2, 1);
        auto weights = CreateTensor("weights", Shape(1, 1, 1, 2), DataType::Int8, std::move(wValues));
        weights->SetAxisOrder(AxisOrder::IHWO);
        op->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
        REQUIRE(supportedOps->Check(op.get()) == true);
        {
            // Bias must be const
            auto bias = CreateTensor("bias", Shape(1, 1, 1, 2), DataType::Int32);
            op->ConnectInput(TensorUsage::Scales, bias).Set(Quantization::Unit());
            REQUIRE(supportedOps->Check(op.get()) == false);
        }
        {
            // Bias values must be stored in channel-axis
            std::vector<int32_t> values(2, 1);
            auto bias = CreateTensor("bias", Shape(1, 1, 2, 1), DataType::Int32, std::move(values));
            op->ConnectInput(TensorUsage::Scales, bias).Set(Quantization::Unit());
            REQUIRE(supportedOps->Check(op.get()) == false);
        }
        {
            // Bias can't be 8bit
            std::vector<int8_t> values(2, 1);
            auto bias = CreateTensor("bias", Shape(1, 1, 1, 2), DataType::Int8, std::move(values));
            op->ConnectInput(TensorUsage::Scales, bias).Set(Quantization::Unit());
            REQUIRE(supportedOps->Check(op.get()) == false);
        }
        {
            // Bias can't be 16bit
            std::vector<int16_t> values(2, 1);
            auto bias = CreateTensor("bias", Shape(1, 1, 1, 2), DataType::Int16, std::move(values));
            op->ConnectInput(TensorUsage::Scales, bias).Set(Quantization::Unit());
            REQUIRE(supportedOps->Check(op.get()) == false);
        }
        {
            // Bias can be 32 bit
            std::vector<int32_t> values(2, std::numeric_limits<int32_t>::max());
            auto bias = CreateTensor("bias", Shape(1, 1, 1, 2), DataType::Int32, std::move(values));
            op->ConnectInput(TensorUsage::Scales, bias).Set(Quantization::Unit());
            REQUIRE(supportedOps->Check(op.get()) == true);
        }
        {
            // Bias can be 40 bit
            std::vector<int64_t> values(2, (1LL << 40) - 1);
            auto bias = CreateTensor<int64_t>("bias", Shape(1, 1, 1, 2), DataType::Int64, std::move(values));
            op->ConnectInput(TensorUsage::Scales, bias).Set(Quantization::Unit());
            REQUIRE(supportedOps->Check(op.get()) == true);
        }
        {
            // Bias can't be >40 bit
            std::vector<int64_t> values(2, std::numeric_limits<int64_t>::max());
            auto bias = CreateTensor("bias", Shape(1, 1, 1, 2), DataType::Int64, std::move(values));
            op->ConnectInput(TensorUsage::Scales, bias).Set(Quantization::Unit());
            REQUIRE(supportedOps->Check(op.get()) == false);
        }
        op->Disconnect();
    }

    SECTION("ConstrainMaxPoolKernel")
    {
        auto op = CreateOperation(OpType::MaxPool, Shape(1, 1000, 1000, 1), DataType::Int8, Shape(1, 1000, 1000, 1), DataType::Int8);

        auto SetKernel = [&op](int h, int w)
        {
            auto kernel = std::make_unique<Kernel>(Point2i{w, h}, Point2i{1, 1}, Point2i{1, 1}, 1, Margin{0, 0, 0, 0});
            op->SetKernel(std::move(kernel));
            auto ofmConn = op->Output(TensorUsage::OFM);
            auto ifmConn = op->Input(TensorUsage::IFM);
            auto &ofmShape = ofmConn->shape;
            auto &ifmShape = ofmConn->shape;
            ofmShape = ifmShape.WithWidth(ifmShape.Width() - w).WithHeight(ifmShape.Height() - h);
        };
        SetKernel(8, 8);
        REQUIRE(supportedOps->Check(op.get()) == true);
        SetKernel(256, 256);
        REQUIRE(supportedOps->Check(op.get()) == true);
        SetKernel(256, 257);
        REQUIRE(supportedOps->Check(op.get()) == false);
        SetKernel(257, 256);
        REQUIRE(supportedOps->Check(op.get()) == false);
        op->Disconnect();
    }

    SECTION("ConstrainAvgPoolKernel")
    {
        auto op = CreateOperation(OpType::AvgPool, Shape(1, 100, 100, 1), DataType::Int8, Shape(1, 100, 100, 1), DataType::Int8);

        auto SetKernel = [&op](int h, int w, int sh = 1, int sw = 1, int ph = 0, int pw = 0)
        {
            int t = ph / 2;
            int b = ph - t;
            int l = pw / 2;
            int r = pw - l;
            auto kernel = std::make_unique<Kernel>(Point2i{w, h}, Point2i{sw, sh}, Point2i{1, 1}, 1, Margin{t, b, l, r});
            op->SetKernel(std::move(kernel));
            auto ofmConn = op->Output(TensorUsage::OFM);
            auto ifmConn = op->Input(TensorUsage::IFM);
            auto &ofmShape = ofmConn->shape;
            auto &ifmShape = ofmConn->shape;
            ofmShape = ifmShape.WithWidth((ifmShape.Width() - w + pw) / sw).WithHeight((ifmShape.Height() - h + ph) / sh);
        };
        // max size (VALID padding)
        SetKernel(256, 256, 1, 1, 0, 0);
        REQUIRE(supportedOps->Check(op.get()) == true);
        // too large prod (VALID padding)
        SetKernel(256, 257, 1, 1, 0, 0);
        REQUIRE(supportedOps->Check(op.get()) == false);
        // too large height (VALID padding)
        SetKernel(257, 8, 1, 1, 0, 0);
        REQUIRE(supportedOps->Check(op.get()) == false);

        // max size (SAME padding)
        SetKernel(8, 8, 1, 1, 1, 1);
        REQUIRE(supportedOps->Check(op.get()) == true);
        // too large width (SAME padding)
        SetKernel(8, 9, 1, 1, 1, 1);
        REQUIRE(supportedOps->Check(op.get()) == false);
        op->Disconnect();
    }

    SECTION("ConstraintRsqrt")
    {
        // Rsqrt is only supported with int8 or int16 input
        auto op = CreateOperation(OpType::Rsqrt, Shape(1, 10, 10, 1), DataType::Int8, Shape(1, 10, 10, 1), DataType::Int8);
        REQUIRE(supportedOps->Check(op.get()) == true);
        op->Input(TensorUsage::IFM)->tensor->ChangeType(DataType::Int16);
        op->Output(TensorUsage::OFM)->tensor->ChangeType(DataType::Int16);
        REQUIRE(supportedOps->Check(op.get()) == true);
        for ( auto dtype : {DataType::UInt8, DataType::Int32} )
        {
            op->Input(TensorUsage::IFM)->tensor->ChangeType(dtype);
            op->Output(TensorUsage::OFM)->tensor->ChangeType(dtype);
            REQUIRE(supportedOps->Check(op.get()) == false);
        }
        op->Disconnect();
    }

    SECTION("ConstraintConstParams")
    {
        auto op = CreateOperation(OpType::Slice, Shape(1, 10, 10, 1), DataType::Int8, Shape(1, 10, 10, 1), DataType::Int8);
        auto begin = CreateTensor("begin", Shape(4), DataType::Int32);
        auto slice = CreateTensor("slice", Shape(4), DataType::Int32);
        // validate parameter-tensors can't be dynamic
        op->ConnectInput(TensorUsage::Params0, begin);
        op->ConnectInput(TensorUsage::Params1, slice);
        REQUIRE(supportedOps->Check(op.get()) == false);

        // validate parameter-tensors can be const
        begin = CreateTensor("begin", Shape(4), DataType::Int32, std::vector<int>(4, 1));
        slice = CreateTensor("slice", Shape(4), DataType::Int32, std::vector<int>(4, 1));
        op->ConnectInput(TensorUsage::Params0, begin);
        op->ConnectInput(TensorUsage::Params1, slice);
        REQUIRE(supportedOps->Check(op.get()) == true);
        op->Disconnect();
    }

    SECTION("ConstraintMean")
    {
        {
            // Supported mean
            auto op = CreateOperation(OpType::Mean, Shape(1, 10, 10, 1), DataType::Int8, Shape(1, 1, 10, 1), DataType::Int8);
            auto params = CreateTensor("axis", Shape(1), DataType::Int32, std::vector<int>{1});
            op->ConnectInput(TensorUsage::Params, params);
            REQUIRE(supportedOps->Check(op.get()) == true);
            op->Disconnect();
        }
        {
            // Batch > 1 is unsupported
            auto op = CreateOperation(OpType::Mean, Shape(2, 10, 10, 1), DataType::Int8, Shape(2, 1, 10, 1), DataType::Int8);
            auto params = CreateTensor("axis", Shape(1), DataType::Int32, std::vector<int>{1});
            op->ConnectInput(TensorUsage::Params, params);
            REQUIRE(supportedOps->Check(op.get()) == false);
            op->Disconnect();
        }
        {
            // Reduced depth only supported if any of H,W,C is 1
            auto op = CreateOperation(OpType::Mean, Shape(1, 2, 10, 5), DataType::Int8, Shape(1, 2, 10, 1), DataType::Int8);
            auto params = CreateTensor("axis", Shape(1), DataType::Int32, std::vector<int>{3});
            op->ConnectInput(TensorUsage::Params, params);
            REQUIRE(supportedOps->Check(op.get()) == false);
            // change height to 1 and validate pass
            auto ifmConn = op->Input(TensorUsage::IFM);
            auto ofmConn = op->Output(TensorUsage::OFM);
            ifmConn->shape = ifmConn->shape.WithHeight(1);
            ofmConn->shape = ofmConn->shape.WithHeight(1);
            REQUIRE(supportedOps->Check(op.get()) == true);
            op->Disconnect();
        }
        {
            // Kernel_size must not be greater than 64 * 64
            auto op = CreateOperation(OpType::Mean, Shape(1, 64 * 64 + 1, 10, 5), DataType::Int8, Shape(1, 1, 10, 5), DataType::Int8);
            auto params = CreateTensor("axis", Shape(1), DataType::Int32, std::vector<int>{1});
            op->ConnectInput(TensorUsage::Params, params);
            REQUIRE(supportedOps->Check(op.get()) == false);
            // change ifm height to 64*64 and validate pass
            auto ifmConn = op->Input(TensorUsage::IFM);
            ifmConn->shape = ifmConn->shape.WithHeight(64 * 64);
            REQUIRE(supportedOps->Check(op.get()) == true);
            op->Disconnect();
        }
        {
            // max reduced elements uint8 (2²³)
            auto op = CreateOperation(OpType::Mean, Shape(1, 1 << 12, 1 << 11, 1), DataType::UInt8, Shape(1, 1, 1, 1), DataType::UInt8);
            auto params = CreateTensor("axis", Shape(2), DataType::Int32, std::vector<int>{1, 2});
            op->ConnectInput(TensorUsage::Params, params);
            REQUIRE(supportedOps->Check(op.get()) == true);
            auto ifmConn = op->Input(TensorUsage::IFM);
            // increase height and validate failure
            ifmConn->shape = ifmConn->shape.WithHeight(ifmConn->shape.Height() + 1);
            REQUIRE(supportedOps->Check(op.get()) == false);
            op->Disconnect();
        }
        {
            // max reduced elements int16 (2¹⁶)
            auto op = CreateOperation(OpType::Mean, Shape(1, 1 << 11, 1 << 5, 1), DataType::Int16, Shape(1, 1, 1, 1), DataType::Int16);
            auto params = CreateTensor("axis", Shape(2), DataType::Int32, std::vector<int>{1, 2});
            op->ConnectInput(TensorUsage::Params, params);
            REQUIRE(supportedOps->Check(op.get()) == true);
            auto ifmConn = op->Input(TensorUsage::IFM);
            // increase height and validate failure
            ifmConn->shape = ifmConn->shape.WithHeight(ifmConn->shape.Height() + 1);
            REQUIRE(supportedOps->Check(op.get()) == false);
            op->Disconnect();
        }
    }

    SECTION("ConstraintLog")
    {
        // Log is only supported with int8 or int16 input
        auto op = CreateOperation(OpType::Log, Shape(1, 10, 10, 1), DataType::Int8, Shape(1, 10, 10, 1), DataType::Int8);
        REQUIRE(supportedOps->Check(op.get()) == true);
        op->Input(TensorUsage::IFM)->tensor->ChangeType(DataType::Int16);
        op->Output(TensorUsage::OFM)->tensor->ChangeType(DataType::Int16);
        REQUIRE(supportedOps->Check(op.get()) == true);
        for ( auto dtype : {DataType::UInt8, DataType::Int32} )
        {
            op->Input(TensorUsage::IFM)->tensor->ChangeType(dtype);
            op->Output(TensorUsage::OFM)->tensor->ChangeType(dtype);
            REQUIRE(supportedOps->Check(op.get()) == false);
        }
        // IFM and OFM data types must match
        op->Input(TensorUsage::IFM)->tensor->ChangeType(DataType::Int8);
        op->Output(TensorUsage::OFM)->tensor->ChangeType(DataType::Int16);
        REQUIRE(supportedOps->Check(op.get()) == false);
        op->Disconnect();
        // IFM and OFM shape must match
        auto op2 = CreateOperation(OpType::Log, Shape(1, 7, 10, 1), DataType::Int8, Shape(1, 10, 10, 1), DataType::Int8);
        REQUIRE(supportedOps->Check(op2.get()) == false);
        op2->Disconnect();
    }
}

TEST_CASE("Supported operators EthosU55")
{
    DisableLogging disableLogging;

    std::shared_ptr<Architecture> arch = CreateArchDefault<ArchEthosU55>(256);
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    auto supportedOps = MakeSupportedOpsChecker(REGOR_ARCH_ETHOSU55, arch->Constraints());

    SECTION("Test positive")
    {
        // checks are expected to pass
        auto op = CreateOperation(OpType::Add, Shape(1, 8, 8, 1), DataType::Int8, Shape(1, 8, 8, 1), DataType::Int8,
            Shape(1, 8, 8, 1), DataType::Int8);
        REQUIRE(supportedOps->Check(op.get()) == true);
        op->Disconnect();
    }

    SECTION("ConstraintOpType")
    {
        auto op = CreateOperation(OpType::ScatterNd, Shape(1, 8, 8, 1), DataType::Int8, Shape(1, 8, 8, 1), DataType::Int8);
        auto op2 = CreateOperation(OpType::GatherV2, Shape(1, 8, 8, 1), DataType::Int8, Shape(1, 8, 8, 1), DataType::Int8);
        REQUIRE(supportedOps->Check(op.get()) == false);
        REQUIRE(supportedOps->Check(op2.get()) == false);
        op->Disconnect();
        op2->Disconnect();
    }

    SECTION("ConstraintTensDtypes")
    {
        std::set<DataType> unsupported = {
            DataType::Int48,
            DataType::UInt48,
            DataType::UInt64,
            DataType::QInt,
            DataType::QInt,
            DataType::QInt4,
            DataType::QInt8,
            DataType::QInt12,
            DataType::QInt16,
            DataType::QInt32,
            DataType::QUInt,
            DataType::QUInt4,
            DataType::QUInt8,
            DataType::QUInt12,
            DataType::QUInt16,
            DataType::QUInt32,
            DataType::Float,
            DataType::BFloat16,
            DataType::Float16,
            DataType::Float32,
            DataType::Float64,
            DataType::Bool,
            DataType::Bool8,
            DataType::Complex,
            DataType::Complex64,
            DataType::Complex128,
            DataType::VariablySized,
            DataType::String,
            DataType::Resource,
            DataType::Variant,
        };
        std::set<DataType> supported = {
            DataType::UInt8,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
        };
        for ( auto dtype : unsupported )
        {
            auto op = CreateOperation(OpType::Add, Shape(1, 8, 8, 1), dtype, Shape(1, 8, 8, 1), dtype, Shape(1, 8, 8, 1), dtype);
            REQUIRE(supportedOps->Check(op.get()) == false);
            op->Disconnect();
        }
        for ( auto dtype : supported )
        {
            auto op = CreateOperation(OpType::Add, Shape(1, 8, 8, 1), dtype, Shape(1, 8, 8, 1), dtype, Shape(1, 8, 8, 1), dtype);
            REQUIRE(supportedOps->Check(op.get()) == true);
            op->Disconnect();
        }
    }

    SECTION("ConstraintBroadCastShapes")
    {
        auto op = CreateOperation(OpType::Add, Shape(1, 5, 5, 1), DataType::Int8, Shape(1, 2, 2, 1), DataType::Int8,
            Shape(1, 8, 8, 1), DataType::Int8);
        REQUIRE(supportedOps->Check(op.get()) == false);
        op->Disconnect();
    }

    SECTION("ConstraintReverse")
    {
        auto op = CreateOperation(OpType::ReverseV2, Shape(1, 8, 8, 1), DataType::Int8, Shape(1, 8, 8, 1), DataType::Int8);
        // create params
        auto params = CreateTensor("axis", Shape(1, 1, 1, 1), DataType::Int32, 1);
        op->ConnectInput(TensorUsage::Params, params);
        REQUIRE(supportedOps->Check(op.get()) == false);
        op->Disconnect();
    }

    SECTION("Constraint32BitOps")
    {
        auto op = CreateOperation(OpType::Add, Shape(1, 1, 1, 1), DataType::Int32, Shape(1, 1, 1, 1), DataType::Int32,
            Shape(1, 1, 1, 1), DataType::Int32);
        REQUIRE(supportedOps->Check(op.get()) == true);
        auto op2 = CreateOperation(OpType::MaxPool, Shape(1, 1, 1, 1), DataType::Int32, Shape(1, 1, 1, 1), DataType::Int32);
        REQUIRE(supportedOps->Check(op2.get()) == false);
        op->Disconnect();
        op2->Disconnect();
    }

    SECTION("ConstraintStride")
    {
        {
            auto op = CreateOperation(OpType::MaxPool, Shape(1, 10, 10, 1), DataType::Int8, Shape(1, 10, 10, 1), DataType::Int8);
            auto kernel = std::make_unique<Kernel>(Point2i{1, 1}, Point2i{1, 1}, Point2i{1, 1}, 1, Margin{0, 0, 0, 0});
            op->SetKernel(std::move(kernel));
            REQUIRE(supportedOps->Check(op.get()) == true);
            op->Disconnect();
        }
        {
            // stride > 3 is not supported for Add
            auto op = CreateOperation(OpType::Add, Shape(1, 10, 10, 1), DataType::Int8, Shape(1, 10, 10, 1), DataType::Int8);
            auto kernel = std::make_unique<Kernel>(Point2i{1, 1}, Point2i{5, 5}, Point2i{1, 1}, 1, Margin{0, 0, 0, 0});
            op->SetKernel(std::move(kernel));
            REQUIRE(supportedOps->Check(op.get()) == false);
            op->Disconnect();
        }
        {
            // stride > 3 is supported for Conv2D (it's unrolled)
            auto op = CreateOperation(OpType::Conv2D, Shape(1, 10, 10, 1), DataType::Int8, Shape(1, 2, 2, 1), DataType::Int8);
            auto kernel = std::make_unique<Kernel>(Point2i{1, 1}, Point2i{5, 5}, Point2i{1, 1}, 1, Margin{0, 0, 0, 0});
            op->SetKernel(std::move(kernel));
            REQUIRE(supportedOps->Check(op.get()) == true);
            op->Disconnect();
        }
    }
}

TEST_CASE("Supported operators EthosU85")
{
    DisableLogging disableLogging;

    std::shared_ptr<Architecture> arch = CreateArchDefault<ArchEthosU85>(256);
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    auto supportedOps = MakeSupportedOpsChecker(REGOR_ARCH_ETHOSU85, arch->Constraints());

    SECTION("Test positive")
    {
        // Validate that both inputs broadcasted is supported by Ethos-U85
        auto op = CreateOperation(OpType::Add, Shape(1, 5, 5, 1), DataType::Int8, Shape(1, 2, 2, 1), DataType::Int8,
            Shape(1, 8, 8, 1), DataType::Int8);
        REQUIRE(supportedOps->Check(op.get()) == true);
        op->Disconnect();
    }

    SECTION("ConstraintTensDtypes")
    {
        std::set<DataType> unsupported = {
            DataType::UInt48,
            DataType::UInt64,
            DataType::QInt,
            DataType::QInt,
            DataType::QInt4,
            DataType::QInt8,
            DataType::QInt12,
            DataType::QInt16,
            DataType::QInt32,
            DataType::QUInt,
            DataType::QUInt4,
            DataType::QUInt8,
            DataType::QUInt12,
            DataType::QUInt16,
            DataType::QUInt32,
            DataType::Float,
            DataType::BFloat16,
            DataType::Float16,
            DataType::Float32,
            DataType::Float64,
            DataType::Complex,
            DataType::Complex64,
            DataType::Complex128,
            DataType::VariablySized,
            DataType::String,
            DataType::Resource,
            DataType::Variant,
        };
        std::set<DataType> supported = {
            DataType::UInt8,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Bool,
            DataType::Bool8,
            DataType::Int64,
        };
        for ( auto dtype : unsupported )
        {
            auto op = CreateOperation(OpType::Add, Shape(1, 8, 8, 1), dtype, Shape(1, 8, 8, 1), dtype, Shape(1, 8, 8, 1), dtype);
            REQUIRE(supportedOps->Check(op.get()) == false);
            op->Disconnect();
        }
        for ( auto dtype : supported )
        {
            auto op = CreateOperation(OpType::Add, Shape(1, 8, 8, 1), dtype, Shape(1, 8, 8, 1), dtype, Shape(1, 8, 8, 1), dtype);
            REQUIRE(supportedOps->Check(op.get()) == true);
            op->Disconnect();
        }
    }
}
