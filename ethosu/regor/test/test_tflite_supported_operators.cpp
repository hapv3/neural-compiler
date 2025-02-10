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

#include "architecture/ethosu55/ethos_u55.hpp"
#include "architecture/ethosu65/ethos_u65.hpp"
#include "architecture/ethosu85/ethos_u85.hpp"
#include "tflite/tflite_supported_operators.hpp"
#include "tflite/tflite_supported_operators_u55.hpp"
#include "tflite/tflite_supported_operators_u85.hpp"
#include "util.hpp"

#include <catch_all.hpp>

#include "regor.h"

using namespace regor;

namespace
{
std::unique_ptr<TfLiteSupportedOperators> MakeSupportedOpsChecker(std::string target, std::shared_ptr<Architecture> &arch)
{
    if ( target == REGOR_ARCH_ETHOSU85 )
    {
        return std::make_unique<TfLiteSupportedOperatorsU85>(arch->Constraints());
    }
    else
    {
        return std::make_unique<TfLiteSupportedOperatorsU55>(arch->Constraints());
    }
}

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
    std::shared_ptr<Architecture> arch = CreateArchDefault<ArchEthosU55>(256);
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");
    auto supportedOps = MakeSupportedOpsChecker(REGOR_ARCH_ETHOSU55, arch);

    SECTION("ConstraintTensQuantized")
    {
        auto op = CreateOperation(OpType::Conv2D, Shape(1, 8, 8, 1), DataType::Int8, Shape(1, 8, 8, 1), DataType::Int8);
        std::vector<int8_t> values = {1};
        auto weights = CreateTensor("weights", Shape(1, 1, 1, 1), DataType::Int8, std::move(values));
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
        op->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
        REQUIRE(supportedOps->Check(op.get()) == true);
        op->Input(TensorUsage::Weights)->tensor->Reshape(Shape(2, 2, 1, 2));
        REQUIRE(supportedOps->Check(op.get()) == false);
        op->Disconnect();
    }
}

TEST_CASE("Supported operators EthosU55")
{
    std::shared_ptr<Architecture> arch = CreateArchDefault<ArchEthosU55>(256);
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    auto supportedOps = MakeSupportedOpsChecker(REGOR_ARCH_ETHOSU55, arch);

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
            DataType::Int64,
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
}

TEST_CASE("Supported operators EthosU85")
{
    std::shared_ptr<Architecture> arch = CreateArchDefault<ArchEthosU85>(256);
    std::string err = "noerror";
    arch->CheckConfiguration(err);
    REQUIRE(err == "noerror");

    auto supportedOps = MakeSupportedOpsChecker(REGOR_ARCH_ETHOSU85, arch);

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
            DataType::Bool,
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
