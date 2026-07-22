//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "architecture/neuralai/neural_ai.hpp"
#include "architecture/neuralai/neural_ai_constraints.hpp"
#include "architecture/neuralai/neural_ai_op_config.hpp"

#include <catch_all.hpp>

#include "regor.h"

using namespace regor;

TEST_CASE("Neural-AI architecture factory")
{
    regor_context_t context = 0;
    REQUIRE(regor_create(&context, REGOR_ARCH_NEURALAI) == 1);
    REQUIRE(context != 0);
    regor_destroy(context);
}

TEST_CASE("Neural-AI architecture exposes fixed hardware configuration")
{
    ArchNeuralAI arch;
    std::string error;

    REQUIRE(arch.CheckConfiguration(error));
    REQUIRE(error.empty());
    REQUIRE(arch.AllocationQuantum() == 32);
    REQUIRE(arch.TensorAlignment(TensorUsage::IFM, TensorFormat::Row32) == 32);
    REQUIRE(arch.FeatureMapMemory().memory->Name() == "tcdm");
    REQUIRE(arch.FeatureMapMemory().memory->SizeBytes() == ArchNeuralAI::AllocatableTCDMBytes);
    REQUIRE(arch.ReadonlyMemory().memory->Name() == "model");
    REQUIRE(arch.ModelBindingFormat(TensorUsage::IFM) == TensorFormat::NHWC);
    REQUIRE(arch.DefaultInternalTensorFormat(TensorUsage::IFM, false) == TensorFormat::Row32);
    REQUIRE(arch.DefaultInternalTensorFormat(TensorUsage::OFM, true) == TensorFormat::NHWC);

    std::string target;
    arch.Call([&target](const std::string &name) { target = name; });
    REQUIRE(target == REGOR_ARCH_NEURALAI);
}

TEST_CASE("Neural-AI ROW32 storage and alignment")
{
    ArchNeuralAI arch;
    const Shape logical(1, 2, 3, 33);

    REQUIRE(arch.StorageShape(logical, TensorFormat::Row32) == Shape(1, 2, 3, 64));
    REQUIRE(arch.StorageBytes(logical, TensorFormat::Row32, DataType::Int8) == 384);
    REQUIRE(arch.StorageBytes(Shape(31), TensorFormat::Row32, DataType::Int8) == 32);
    REQUIRE(arch.StorageBytes(Shape(33), TensorFormat::Row32, DataType::Int32) == 256);
    REQUIRE(arch.TensorStrides(logical, TensorFormat::Row32, DataType::Int8) == Shape(384, 192, 64, 1));
    REQUIRE(arch.CanAliasDepthOffset(TensorFormat::Row32, 32));
    REQUIRE_FALSE(arch.CanAliasDepthOffset(TensorFormat::Row32, 1));
}

TEST_CASE("Neural-AI architecture rejects conflicting RTL limits")
{
    ArchNeuralAI arch;
    const std::string config = "[architecture]\nclusters=2\n";
    IniReader reader(config.c_str(), config.size());
    std::string section;

    REQUIRE(reader.Begin(section));
    REQUIRE(section == "architecture");
    REQUIRE(arch.ParseSection(section, &reader) == IniParseResult::Error);
}

TEST_CASE("Neural-AI architecture accepts the fixed RTL limits")
{
    ArchNeuralAI arch;
    const std::string config =
        "[architecture]\n"
        "clusters=1\n"
        "array_dimension=32\n"
        "dma_alignment=32\n"
        "tcdm_size=524288\n"
        "command_staging_size=4096\n"
        "linebuffer_max_kernel=5\n"
        "linebuffer_max_stride=2\n"
        "linebuffer_max_input_width=640\n"
        "requant_shift_max=31\n";
    IniReader reader(config.c_str(), config.size());
    std::string section;

    REQUIRE(reader.Begin(section));
    REQUIRE(section == "architecture");
    REQUIRE(arch.ParseSection(section, &reader) == IniParseResult::Done);
}

TEST_CASE("Neural-AI constraints accept only initial INT8 matrix operations")
{
    ArchNeuralAI arch;
    auto *constraints = arch.Constraints();

    REQUIRE(constraints->OperatorQuery(OpType::FullyConnected) == QueryResult::NativeConstrained);
    REQUIRE(constraints->OperatorQuery(OpType::MatMul) == QueryResult::NativeConstrained);
    REQUIRE(constraints->OperatorQuery(OpType::Conv2D) == QueryResult::Unsupported);

    ArchOperatorQuery query;
    query.ifm[0].type = DataType::Int8;
    query.ifm[0].shape = Shape(1, 31);
    query.weights.type = DataType::Int8;
    query.weights.shape = Shape(31, 33);
    query.ofm.type = DataType::Int8;
    query.ofm.shape = Shape(1, 33);
    query.weightFormat = WeightFormat::Default;
    REQUIRE(constraints->OperatorQuery(OpType::MatMul, &query) == QueryResult::Native);

    query.ofm.type = DataType::UInt8;
    REQUIRE(constraints->OperatorQuery(OpType::MatMul, &query) == QueryResult::Unsupported);
    query.ofm.type = DataType::Int8;
    query.weightFormat = WeightFormat::None;
    REQUIRE(constraints->OperatorQuery(OpType::MatMul, &query) == QueryResult::Unsupported);
    query.weightFormat = WeightFormat::Default;
    query.ifm[0].shape = Shape(2, 1, 31);
    REQUIRE(constraints->OperatorQuery(OpType::MatMul, &query) == QueryResult::Unsupported);
}

TEST_CASE("Neural-AI constraints enforce signed matrix zero points")
{
    ArchNeuralAI arch;
    auto *constraints = arch.Constraints();

    REQUIRE(constraints->SupportedZeroPoint(0, TensorUsage::IFM, DataType::Int8, OpType::MatMul));
    REQUIRE_FALSE(constraints->SupportedZeroPoint(1, TensorUsage::IFM, DataType::Int8, OpType::MatMul));
    REQUIRE(constraints->SupportedZeroPoint(-128, TensorUsage::OFM, DataType::Int8, OpType::MatMul));
    REQUIRE(constraints->SupportedZeroPoint(127, TensorUsage::OFM, DataType::Int8, OpType::MatMul));
    REQUIRE_FALSE(constraints->SupportedZeroPoint(128, TensorUsage::OFM, DataType::Int8, OpType::MatMul));
}

TEST_CASE("Neural-AI matrix op configuration exposes GEMM granules")
{
    ArchNeuralAI arch;
    ArchitectureConfigQuery query{};
    query.ifmBits = 8;
    query.ofmBits = 8;
    query.transpose = TransposeType::None;
    query.reverse = ReverseType::None;

    auto config = arch.GetOpConfig(OpType::MatMul, query);
    REQUIRE(config);
    REQUIRE(config->OptimalStripeGranule() == Point2i(32, 1));
    REQUIRE(config->MinimalStripeGranule() == Point2i(1, 1));
    REQUIRE(config->OptimalDepthGranule() == 32);
    REQUIRE(config->MinimumDepthGranule() == 32);
    REQUIRE_FALSE(arch.GetOpConfig(OpType::Conv2D, query));
}

TEST_CASE("Neural-AI op groups do not fuse matrix operations")
{
    ArchNeuralAI arch;
    ArchitectureOpGroupQuery query{};
    query.type = OpType::FullyConnected;

    auto group = arch.CreateOpGroup(query);
    REQUIRE(group);
    REQUIRE(group->NeedsAllocation(1));
    REQUIRE(group->Add(query) == 0);
}
