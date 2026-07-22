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
#include "compiler/compiler.hpp"
#include "compiler/neural_ai_command_generator.hpp"
#include "compiler/neural_ai_graph_optimiser.hpp"
#include "compiler/neural_ai_writer.hpp"
#include "compiler/scheduler.hpp"
#include "compiler/scheduler_packing.hpp"
#include "tflite/tflite_schema_generated.hpp"
#include "util.hpp"

#include <catch_all.hpp>

#include <cstring>

#include "regor.h"

using namespace regor;

namespace
{

uint32_t Read32(const uint8_t *data)
{
    return uint32_t(data[0]) | (uint32_t(data[1]) << 8) |
           (uint32_t(data[2]) << 16) | (uint32_t(data[3]) << 24);
}

uint32_t Read32(const std::vector<uint8_t> &data, size_t offset)
{
    return Read32(data.data() + offset);
}

uint16_t Read16(const std::vector<uint8_t> &data, size_t offset)
{
    return uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
}

flatbuffers::DetachedBuffer BuildFullyConnectedModel(int depthK, int depthN)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<float> scale = {1.0f};
    const std::vector<int64_t> zeroPoint = {0};
    const auto inputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto weightQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto biasQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);
    const auto outputQuant = tflite::CreateQuantizationParametersDirect(
        builder, nullptr, nullptr, &scale, &zeroPoint);

    std::vector<uint8_t> weightData(depthK * depthN, 1);
    std::vector<int32_t> bias(depthN, 0);
    std::vector<uint8_t> biasData(bias.size() * sizeof(int32_t));
    std::memcpy(biasData.data(), bias.data(), biasData.size());
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
        tflite::CreateBufferDirect(builder),
        tflite::CreateBufferDirect(builder, &weightData),
        tflite::CreateBufferDirect(builder, &biasData),
    };
    const std::vector<int32_t> inputShape = {1, depthK};
    const std::vector<int32_t> weightShape = {depthN, depthK};
    const std::vector<int32_t> biasShape = {depthN};
    const std::vector<int32_t> outputShape = {1, depthN};
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
        tflite::CreateTensorDirect(builder, &inputShape, tflite::TensorType::INT8, 0, "input", inputQuant),
        tflite::CreateTensorDirect(builder, &weightShape, tflite::TensorType::INT8, 1, "weights", weightQuant),
        tflite::CreateTensorDirect(builder, &biasShape, tflite::TensorType::INT32, 2, "bias", biasQuant),
        tflite::CreateTensorDirect(builder, &outputShape, tflite::TensorType::INT8, 0, "output", outputQuant),
    };
    const auto options = tflite::CreateFullyConnectedOptions(builder);
    const std::vector<int32_t> opInputs = {0, 1, 2};
    const std::vector<int32_t> opOutputs = {3};
    const std::vector<flatbuffers::Offset<tflite::Operator>> operations = {
        tflite::CreateOperatorDirect(builder, 0, &opInputs, &opOutputs,
            tflite::BuiltinOptions::FullyConnectedOptions, options.Union()),
    };
    const std::vector<int32_t> graphInputs = {0};
    const std::vector<int32_t> graphOutputs = {3};
    const std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs = {
        tflite::CreateSubGraphDirect(
            builder, &tensors, &graphInputs, &graphOutputs, &operations, "main"),
    };
    const std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes = {
        tflite::CreateOperatorCodeDirect(builder, int8_t(tflite::BuiltinOperator::FULLY_CONNECTED),
            nullptr, 1, tflite::BuiltinOperator::FULLY_CONNECTED),
    };
    const auto model = tflite::CreateModelDirect(builder, 3, &operatorCodes, &subgraphs,
        "Neural-AI FullyConnected test", &buffers);
    tflite::FinishModelBuffer(builder, model);
    return builder.Release();
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

TEST_CASE("Neural-AI scheduler lowers a complete FullyConnected graph")
{
    constexpr int depthK = 33;
    constexpr int depthN = 34;
    ArchNeuralAI arch;
    auto input = CreateTensor("input", Shape(1, 1, 1, depthK), DataType::Int8);
    auto output = CreateTensor("output", Shape(1, 1, 1, depthN), DataType::Int8);
    auto weights = CreateTensor(
        "weights", Shape(depthN, 1, 1, depthK), DataType::Int8,
        std::vector<int8_t>(depthK * depthN, 1));
    auto scales = CreateTensor(
        "scales", Shape(1, 1, 1, depthN), DataType::Int32,
        std::vector<int32_t>(depthN, 0));
    auto matrix = CreateOperation(
        OpType::FullyConnected, TensorUsage::IFM0, input, TensorUsage::OFM, output);
    matrix->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
    matrix->ConnectInput(TensorUsage::Scales, scales).Set(Quantization::Unit());
    std::vector<std::shared_ptr<Operation>> sourceOps = {matrix};
    auto graph = CreateGraph(sourceOps);

    GraphOptimiserOptions graphOptions;
    NeuralAIGraphOptimiser optimiser(arch.Constraints(), graphOptions, nullptr);
    optimiser.Process(graph.get());

    const std::unordered_map<UniqueId, UniqueId> equivalenceIds;
    SchedulerPacking packing(&arch, false, equivalenceIds);
    auto scheduleOps = packing.Process(graph.get());
    REQUIRE(scheduleOps.size() == 3);

    SchedulerOptions schedulerOptions;
    schedulerOptions.disabled.Set(SchedulerFeature::Cascading);
    schedulerOptions.disabled.Set(SchedulerFeature::WeightBuffering);
    Scheduler scheduler(
        &arch, schedulerOptions, "neural-ai", scheduleOps, packing.OpConfigCompatablility());
    auto schedule = scheduler.Process();

    REQUIRE(schedule != nullptr);
    REQUIRE(schedule->Costs().size() == 3);
    REQUIRE(schedule->memoryUsage.at(arch.FeatureMapMemory()) % ArchNeuralAI::DMAAlignment == 0);
    REQUIRE(scheduleOps[0]->OFM()->tensor->format == TensorFormat::Row32);
    REQUIRE(scheduleOps[1]->OFM()->tensor->format == TensorFormat::Row32);
    REQUIRE(scheduleOps[2]->OFM()->tensor->format == TensorFormat::NHWC);
    const SchedulerOpInfo *matrixCost = schedule->Cost(scheduleOps[1].get());
    REQUIRE(matrixCost->npuWeightsTensor != nullptr);
    REQUIRE(matrixCost->npuWeightsTensor->totalWeightBytes == 4 * 32 * 32);
    REQUIRE(matrixCost->npuWeightsTensor->AllocationSizeBytes() == 6 * 32 * 32);
    REQUIRE(matrixCost->npuWeightsTensor->encodedRanges.size() == 1);
    const WeightRange &range = matrixCost->npuWeightsTensor->encodedRanges.begin()->second;
    REQUIRE(range.scaleBytes == 2 * 32 * 32);
    REQUIRE(range.weightOffset == 2 * 32 * 32);
    REQUIRE(range.weightBytes == 4 * 32 * 32);
    REQUIRE(matrixCost->npuScalesTensor == nullptr);

    CompiledNeuralAIArtifact artifact;
    std::string error;
    NeuralAICommandGenerator commandGenerator;
    REQUIRE(commandGenerator.Generate(graph.get(), scheduleOps, schedule.get(), artifact, error));
    REQUIRE(error.empty());
    REQUIRE(artifact.commandCount == 12);
    REQUIRE(artifact.commands.size() == 928);
    REQUIRE(artifact.constants.size() == 4 * 32 * 32);
    REQUIRE(artifact.qparams.size() == 2 * 32);
    REQUIRE(artifact.bindings.size() == 2);
    REQUIRE(artifact.requiredTCDMBytes % ArchNeuralAI::DMAAlignment == 0);

    const std::vector<uint16_t> expectedTypes = {
        uint16_t(neuralai::CommandType::CopyLayout),
        uint16_t(neuralai::CommandType::RQLoad),
        uint16_t(neuralai::CommandType::DMA2D),
        uint16_t(neuralai::CommandType::Gemm32),
        uint16_t(neuralai::CommandType::DMA2D),
        uint16_t(neuralai::CommandType::Gemm32Requant),
        uint16_t(neuralai::CommandType::RQLoad),
        uint16_t(neuralai::CommandType::DMA2D),
        uint16_t(neuralai::CommandType::Gemm32),
        uint16_t(neuralai::CommandType::DMA2D),
        uint16_t(neuralai::CommandType::Gemm32Requant),
        uint16_t(neuralai::CommandType::CopyLayout),
        uint16_t(neuralai::CommandType::End),
    };
    size_t commandOffset = 0;
    for ( uint16_t expected : expectedTypes )
    {
        REQUIRE(Read16(artifact.commands, commandOffset) == expected);
        commandOffset += Read16(artifact.commands, commandOffset + 2);
    }
    REQUIRE(commandOffset == artifact.commands.size());
    REQUIRE(Read32(artifact.commands, 60) == depthK);
    REQUIRE(Read32(artifact.commands, 64) == 64);
    REQUIRE(Read32(artifact.commands, 96 + 16) == 0);
    REQUIRE(Read32(artifact.commands, 352 + 20) == 32 * 32);
    REQUIRE(Read32(artifact.commands, 448 + 16) == 32);
    REQUIRE(Read32(artifact.commands, 704 + 20) == 3 * 32 * 32);
    REQUIRE(Read32(artifact.commands, 800 + 60) == 64);
    REQUIRE(Read32(artifact.commands, 800 + 64) == depthN);

    std::vector<uint8_t> package;
    REQUIRE(WriteNeuralAIModel(artifact, package, error));
    REQUIRE(Read32(package, 0) == neuralai::ModelMagic);
}

TEST_CASE("Neural-AI command generator emits direct single-tile requant")
{
    constexpr int depthK = 31;
    constexpr int depthN = 17;
    ArchNeuralAI arch;
    auto input = CreateTensor("input", Shape(1, 1, 1, depthK), DataType::Int8);
    auto output = CreateTensor("output", Shape(1, 1, 1, depthN), DataType::Int8);
    auto weights = CreateTensor("weights", Shape(depthN, 1, 1, depthK), DataType::Int8,
        std::vector<int8_t>(depthK * depthN, 1));
    auto scales = CreateTensor("scales", Shape(1, 1, 1, depthN), DataType::Int32,
        std::vector<int32_t>(depthN, 0));
    auto matrix = CreateOperation(
        OpType::FullyConnected, TensorUsage::IFM0, input, TensorUsage::OFM, output);
    matrix->ConnectInput(TensorUsage::Weights, weights).Set(Quantization::Unit());
    matrix->ConnectInput(TensorUsage::Scales, scales).Set(Quantization::Unit());
    std::vector<std::shared_ptr<Operation>> sourceOps = {matrix};
    auto graph = CreateGraph(sourceOps);

    GraphOptimiserOptions graphOptions;
    NeuralAIGraphOptimiser optimiser(arch.Constraints(), graphOptions, nullptr);
    optimiser.Process(graph.get());
    const std::unordered_map<UniqueId, UniqueId> equivalenceIds;
    SchedulerPacking packing(&arch, false, equivalenceIds);
    auto scheduleOps = packing.Process(graph.get());
    SchedulerOptions schedulerOptions;
    schedulerOptions.disabled.Set(SchedulerFeature::Cascading);
    schedulerOptions.disabled.Set(SchedulerFeature::WeightBuffering);
    Scheduler scheduler(
        &arch, schedulerOptions, "neural-ai-direct", scheduleOps, packing.OpConfigCompatablility());
    auto schedule = scheduler.Process();

    CompiledNeuralAIArtifact artifact;
    std::string error;
    NeuralAICommandGenerator commandGenerator;
    REQUIRE(commandGenerator.Generate(graph.get(), scheduleOps, schedule.get(), artifact, error));
    REQUIRE(artifact.commandCount == 4);
    REQUIRE(artifact.commands.size() == 352);
    REQUIRE(artifact.constants.size() == 32 * 32);
    REQUIRE(artifact.qparams.size() == 32);
    REQUIRE(Read16(artifact.commands, 0) == uint16_t(neuralai::CommandType::CopyLayout));
    REQUIRE(Read16(artifact.commands, 96) == uint16_t(neuralai::CommandType::RQLoad));
    REQUIRE(Read16(artifact.commands, 128) == uint16_t(neuralai::CommandType::Gemm32Requant));
    REQUIRE(Read16(artifact.commands, 128 + 32) == 0);
    REQUIRE(Read32(artifact.commands, 128 + 52) == 32);
    REQUIRE(Read16(artifact.commands, 224) == uint16_t(neuralai::CommandType::CopyLayout));
    REQUIRE(Read16(artifact.commands, 320) == uint16_t(neuralai::CommandType::End));
}

TEST_CASE("Neural-AI compiler emits a native model package")
{
    constexpr int depthK = 33;
    constexpr int depthN = 34;
    std::unique_ptr<Architecture> architecture = std::make_unique<ArchNeuralAI>();
    Compiler compiler(architecture);
    const std::string options = "[scheduler]\ncpu_tensor_alignment=32\n";
    REQUIRE(compiler.ParseOptions(options.c_str(), options.size()));
    const auto model = BuildFullyConnectedModel(depthK, depthN);
    REQUIRE(compiler.LoadTflite(model.data(), model.size()));

    const bool compiled = compiler.Compile();
    INFO(compiler.LastError());
    REQUIRE(compiled);
    IRegorBlob *blob = compiler.Output();
    REQUIRE(blob != nullptr);
    int64_t size = 0;
    const auto *data = static_cast<const uint8_t *>(blob->Map(size));
    REQUIRE(size > int64_t(sizeof(neuralai::ModelHeaderV1)));
    REQUIRE(Read32(data) == neuralai::ModelMagic);
    blob->Unmap(const_cast<uint8_t *>(data));
    blob->Release();
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
