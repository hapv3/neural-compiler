//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "architecture/neuralai/neural_ai.hpp"
#include "architecture/neuralai/neural_ai_constraints.hpp"
#include "architecture/neuralai/neural_ai_op_config.hpp"
#include "architecture/neuralai/neural_ai_performance.hpp"
#include "architecture/neuralai/neural_ai_weight_encoder.hpp"
#include "compiler/neural_ai_graph_optimiser.hpp"
#include "util.hpp"

#include <catch_all.hpp>

#include "regor.h"

using namespace regor;

namespace
{

uint32_t Read32(const std::vector<uint8_t> &data, size_t offset)
{
    return uint32_t(data[offset]) | (uint32_t(data[offset + 1]) << 8) |
           (uint32_t(data[offset + 2]) << 16) | (uint32_t(data[offset + 3]) << 24);
}

}  // namespace

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

TEST_CASE("Neural-AI constraints accept shape-preserving memory copies")
{
    ArchNeuralAI arch;
    auto *constraints = arch.Constraints();

    REQUIRE(constraints->OperatorQuery(OpType::MemoryCopy) == QueryResult::NativeConstrained);

    ArchOperatorQuery query;
    query.ifm[0].type = DataType::Int8;
    query.ifm[0].shape = Shape(1, 1, 1, 33);
    query.ofm.type = DataType::Int8;
    query.ofm.shape = query.ifm[0].shape;
    REQUIRE(constraints->OperatorQuery(OpType::MemoryCopy, &query) == QueryResult::Native);

    query.ofm.shape = Shape(1, 1, 1, 34);
    REQUIRE(constraints->OperatorQuery(OpType::MemoryCopy, &query) == QueryResult::Unsupported);
}

TEST_CASE("Neural-AI graph optimiser inserts ROW32 boundary conversions")
{
    ArchNeuralAI arch;
    auto input = CreateTensor("input", Shape(1, 1, 1, 33), DataType::Int8);
    auto output = CreateTensor("output", Shape(1, 1, 1, 34), DataType::Int8);
    auto matrix = CreateOperation(
        OpType::FullyConnected, TensorUsage::IFM0, input, TensorUsage::OFM, output);
    std::vector<std::shared_ptr<Operation>> sourceOps = {matrix};
    auto graph = CreateGraph(sourceOps);

    GraphOptimiserOptions options;
    NeuralAIGraphOptimiser optimiser(arch.Constraints(), options, nullptr);
    optimiser.Process(graph.get());

    REQUIRE(graph->IsInput(input.get()));
    REQUIRE(graph->IsOutput(output.get()));
    REQUIRE(matrix->IFM(0) != input.get());
    REQUIRE(matrix->OFM() != output.get());
    REQUIRE(matrix->IFM(0)->Name() == "input/row32");
    REQUIRE(matrix->OFM()->Name() == "output/row32");

    REQUIRE(input->Readers().size() == 1);
    REQUIRE(input->Readers().front()->Type() == OpType::MemoryCopy);
    REQUIRE(input->Readers().front()->OFM() == matrix->IFM(0));
    REQUIRE(output->Writers().size() == 1);
    REQUIRE(output->Writers().front()->Type() == OpType::MemoryCopy);
    REQUIRE(output->Writers().front()->IFM(0) == matrix->OFM());

    std::vector<Operation *> operations;
    graph->GetAllOperations(operations);
    REQUIRE(operations.size() == 3);
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

TEST_CASE("Neural-AI performance model estimates GEMM and DMA costs")
{
    ArchNeuralAI arch;
    PerformanceQuery query{};
    query.type = OpType::FullyConnected;
    query.ifm[0].shape = Shape(1, 1, 1, 33);
    query.ifm[0].type = DataType::Int8;
    query.ifm[0].memory = arch.FeatureMapMemory().memory;
    query.ofm.shape = Shape(1, 1, 1, 34);
    query.ofm.type = DataType::Int8;
    query.ofm.memory = arch.FeatureMapMemory().memory;
    query.constMemory = arch.ReadonlyMemory().memory;
    query.encodedWeightSize = 4096;
    query.encodedScaleSize = 1024;

    auto *performance = arch.Performance();
    REQUIRE(performance != nullptr);
    const CycleCost gemm = performance->MeasureCycleCost(query);
    REQUIRE(gemm.macs == 33 * 34);
    REQUIRE(gemm.opCycles == 2);

    const ElementAccess elements = performance->MeasureElementAccess(query);
    const ElementAccess bytes = performance->ElementTransferToBytes(query, elements);
    REQUIRE(bytes.ifmRead[0] == 33);
    REQUIRE(bytes.ofmWrite == 34);
    REQUIRE(bytes.constRead[0] == 4096);
    REQUIRE(bytes.constRead[1] == 1024);

    query.type = OpType::MemoryCopy;
    REQUIRE(performance->MeasureCycleCost(query).opCycles == 0);
    REQUIRE(performance->MemToMemCycles(
        arch.FeatureMapMemory().memory, arch.ReadonlyMemory().memory, 64) == 3);
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

TEST_CASE("Neural-AI GEMM weight packing follows K-lane N-lane tile order")
{
    constexpr int depthK = 33;
    constexpr int depthN = 34;
    std::vector<int8_t> weights(depthK * depthN);
    for ( int k = 0; k < depthK; ++k )
    {
        for ( int n = 0; n < depthN; ++n )
        {
            weights[k * depthN + n] = int8_t((k * 17 + n * 3) & 0x7f);
        }
    }

    const auto packed = neuralai::PackGEMM32Weights(weights.data(), depthK, depthN);
    REQUIRE(packed.size() == 4 * 32 * 32);
    for ( int nGroup = 0; nGroup < 2; ++nGroup )
    {
        for ( int kGroup = 0; kGroup < 2; ++kGroup )
        {
            for ( int kLane = 0; kLane < 32; ++kLane )
            {
                for ( int nLane = 0; nLane < 32; ++nLane )
                {
                    const int k = kGroup * 32 + kLane;
                    const int n = nGroup * 32 + nLane;
                    const size_t index = size_t(((nGroup * 2 + kGroup) * 32 + kLane) * 32 + nLane);
                    const uint8_t expected = k < depthK && n < depthN ? uint8_t(weights[k * depthN + n]) : 0;
                    REQUIRE(packed[index] == expected);
                }
            }
        }
    }
}

TEST_CASE("Neural-AI Regor weight encoder packs OHWI matrix constants")
{
    constexpr int depthK = 33;
    constexpr int depthN = 34;
    ArchNeuralAI arch;
    NeuralAIOpConfig opConfig;
    auto *encoder = arch.WeightEncoder();
    auto config = encoder->GetEncodingConfig(
        &opConfig, nullptr, DataType::Int8, Flags<WeightFormat>(WeightFormat::Default));
    auto source = encoder->GetWeightSource(config.get(), DataType::Int8, nullptr, nullptr);

    std::vector<int8_t> ohwi(depthN * depthK);
    std::vector<int8_t> matrixKN(depthK * depthN);
    for ( int n = 0; n < depthN; ++n )
    {
        for ( int k = 0; k < depthK; ++k )
        {
            const int8_t value = int8_t(((n * 11 + k * 5) & 0x7f) - 64);
            ohwi[n * depthK + k] = value;
            matrixKN[k * depthN + n] = value;
        }
    }
    source->SetSource(ohwi.data(), 0, Shape(depthN, 1, 1, depthK),
        Shape(depthK, depthK, depthK, 1), 0);

    std::vector<uint8_t> encoded;
    const WeightsInfo info = encoder->EncodeWeights(config.get(), source.get(), encoded);
    REQUIRE(info.sourceSize == depthK * depthN);
    REQUIRE(encoded == neuralai::PackGEMM32Weights(matrixKN.data(), depthK, depthN));
}

TEST_CASE("Neural-AI Regor weight encoder emits ABI qparam lanes")
{
    constexpr int channels = 3;
    ArchNeuralAI arch;
    NeuralAIOpConfig opConfig;
    auto *encoder = arch.WeightEncoder();
    auto config = encoder->GetEncodingConfig(
        &opConfig, nullptr, DataType::Int8, Flags<WeightFormat>(WeightFormat::Default));
    Quantization quantization;
    quantization.scales = {QuantizedScale(101, 3), QuantizedScale(202, 4), QuantizedScale(303, 5)};
    quantization.zeroPoints = {-7, 2, 11};
    quantization.quantMin = {-100};
    quantization.quantMax = {99};
    std::vector<int32_t> biases = {1001, -2002, 3003};
    auto source = encoder->GetScaleSource(config.get(), DataType::Int32, quantization);
    source->SetSource(biases.data(), channels, 0, channels, 0);

    std::vector<uint8_t> encoded;
    REQUIRE(encoder->EncodeScales(config.get(), source.get(), encoded, false) == 32 * 32);
    REQUIRE(encoded.size() == 32 * 32);
    for ( int channel = 0; channel < channels; ++channel )
    {
        const size_t offset = size_t(channel) * 32;
        REQUIRE(int32_t(Read32(encoded, offset)) == biases[channel]);
        REQUIRE(int32_t(Read32(encoded, offset + 4)) == quantization.scales[channel].scale);
        REQUIRE(Read32(encoded, offset + 8) == uint32_t(quantization.scales[channel].shift));
        REQUIRE(int32_t(Read32(encoded, offset + 12)) == quantization.zeroPoints[channel]);
        REQUIRE(int32_t(Read32(encoded, offset + 16)) == -100);
        REQUIRE(int32_t(Read32(encoded, offset + 20)) == 99);
    }
    REQUIRE(Read32(encoded, 3 * 32 + 4) == 0);
}
