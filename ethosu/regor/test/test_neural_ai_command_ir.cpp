//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/neural_ai_command_compactor.hpp"
#include "compiler/neural_ai_command_ir.hpp"
#include "compiler/neural_ai_command_scheduler.hpp"

#include <catch_all.hpp>

#include <algorithm>
#include <cstring>
#include <numeric>

using namespace regor::neuralai;

namespace
{

template<typename TYPE>
void Append(std::vector<uint8_t> &bytes, const TYPE &command)
{
    const auto *begin = reinterpret_cast<const uint8_t *>(&command);
    bytes.insert(bytes.end(), begin, begin + sizeof(command));
}

void AppendControl(std::vector<uint8_t> &bytes, CommandType type,
    uint32_t layerId = 0, uint32_t tileId = 0)
{
    CommandHeaderV2 command{};
    command.type = uint16_t(type);
    command.sizeBytes = 32;
    command.layerId = layerId;
    command.tileId = tileId;
    Append(bytes, command);
    bytes.resize(bytes.size() + 16);
}

bool HasEdge(const CommandDependencyGraph &graph, uint32_t predecessor,
    uint32_t successor, DependencyKind kind)
{
    return std::any_of(graph.edges.begin(), graph.edges.end(),
        [&](const CommandDependency &edge)
        {
            return edge.predecessor == predecessor && edge.successor == successor &&
                   edge.kind == kind;
        });
}

const SemanticStateAccess *FindState(const SemanticCommand &command,
    SemanticState state, SemanticAccessMode mode)
{
    auto position = std::find_if(command.stateAccesses.begin(), command.stateAccesses.end(),
        [&](const SemanticStateAccess &access)
        {
            return access.state == state && access.mode == mode;
        });
    return position == command.stateAccesses.end() ? nullptr : &*position;
}

}  // namespace

TEST_CASE("Neural-AI semantic IR round-trips ABI commands and maps hardware resources")
{
    CommandDMA1DV2 load{};
    load.header.type = uint16_t(CommandType::DMASubmit1D);
    load.header.sizeBytes = sizeof(load);
    load.header.layerId = 7;
    load.header.tileId = 3;
    load.source = {uint16_t(Region::ModelConstants), 0, 64};
    load.destination = {uint16_t(Region::TCDMScratch), 0, 32};
    load.length = 96;
    load.direction = uint32_t(DMADirection::ExternalToLocal);

    CommandAFUBinaryV2 add{};
    add.header.type = uint16_t(CommandType::AFUBinary);
    add.header.sizeBytes = sizeof(add);
    add.lhs = load.destination;
    add.rhs = {uint16_t(Region::TCDMScratch), 0, 256};
    add.ofm = {uint16_t(Region::TCDMScratch), 0, 544};
    add.length = 96;
    add.mode = uint32_t(AFUBinaryMode::AddI8);

    std::vector<uint8_t> bytes;
    Append(bytes, load);
    Append(bytes, add);
    AppendControl(bytes, CommandType::End);

    std::vector<SemanticCommand> commands;
    std::string error;
    REQUIRE(DecodeSemanticCommandStream(bytes, commands, error));
    REQUIRE(commands.size() == 3);
    CHECK(SerializeSemanticCommandStream(commands) == bytes);
    CHECK(commands[0].layerId == 7);
    CHECK(commands[0].tileId == 3);
    CHECK(commands[0].asyncCapable);
    CHECK(commands[0].asynchronous);
    CHECK(commands[0].queueCapacity == 16);
    CHECK(commands[0].resources == ResourceMask(SemanticResource::DMAExternalToLocal));
    REQUIRE(commands[0].memoryAccesses.size() == 2);
    CHECK(commands[0].memoryAccesses[1].bankPhase == 1);
    CHECK(commands[1].resources == ResourceMask(SemanticResource::AFU));
    CHECK(commands[1].estimatedCycles == 3);
    CHECK(commands[2].controlFence);
}

TEST_CASE("Neural-AI dependency DAG records RAW WAR WAW and logical content generations")
{
    CommandAFUBinaryV2 producer{};
    producer.header.type = uint16_t(CommandType::AFUBinary);
    producer.header.sizeBytes = sizeof(producer);
    producer.lhs = {uint16_t(Region::TCDMScratch), 0, 512};
    producer.rhs = {uint16_t(Region::TCDMScratch), 0, 768};
    producer.ofm = {uint16_t(Region::TCDMScratch), 0, 0};
    producer.length = 64;

    CommandAFUBinaryV2 consumer = producer;
    consumer.lhs = producer.ofm;
    consumer.rhs.offset = 1024;
    consumer.ofm.offset = 256;

    CommandDMA1DV2 overwrite{};
    overwrite.header.type = uint16_t(CommandType::DMA1D);
    overwrite.header.sizeBytes = sizeof(overwrite);
    overwrite.source = {uint16_t(Region::ModelConstants), 0, 0};
    overwrite.destination = producer.ofm;
    overwrite.length = 64;
    overwrite.direction = uint32_t(DMADirection::ExternalToLocal);

    std::vector<uint8_t> bytes;
    Append(bytes, producer);
    Append(bytes, consumer);
    Append(bytes, overwrite);

    std::vector<SemanticCommand> commands;
    CommandDependencyGraph graph;
    std::string error;
    REQUIRE(DecodeSemanticCommandStream(bytes, commands, error));
    REQUIRE(BuildCommandDependencyGraph(commands, graph, error));
    CHECK(HasEdge(graph, 0, 1, DependencyKind::MemoryRAW));
    CHECK(HasEdge(graph, 0, 2, DependencyKind::MemoryWAW));
    CHECK(HasEdge(graph, 1, 2, DependencyKind::MemoryWAR));

    REQUIRE(commands[0].memoryAccesses[2].content.size() == 1);
    REQUIRE(commands[1].memoryAccesses[0].content.size() == 1);
    const auto &produced = commands[0].memoryAccesses[2].content[0];
    const auto &consumed = commands[1].memoryAccesses[0].content[0];
    CHECK(produced.identity == consumed.identity);
    CHECK(produced.generation == 1);
    CHECK(consumed.generation == produced.generation);
    CHECK(commands[2].memoryAccesses[1].content[0].generation > produced.generation);
}

TEST_CASE("Neural-AI dependency DAG versions quantization systolic and DMA queue state")
{
    CommandRQLoadV2 rq0{};
    rq0.header.type = uint16_t(CommandType::RQLoad);
    rq0.header.sizeBytes = sizeof(rq0);
    rq0.qparamIndex = 32;
    rq0.qparamCount = 32;
    rq0.qparamBlock = 2;

    CommandGemm32V2 gemm{};
    gemm.header.type = uint16_t(CommandType::Gemm32Requant);
    gemm.header.sizeBytes = sizeof(gemm);
    gemm.weights = {uint16_t(Region::TCDMScratch), 0, 0};
    gemm.ifm = {uint16_t(Region::TCDMScratch), 0, 2048};
    gemm.partialSums = {uint16_t(Region::TCDMScratch), 0, 4096};
    gemm.ofm = {uint16_t(Region::TCDMScratch), 0, 8192};
    gemm.dimM = 4;

    CommandRQLoadV2 rq1 = rq0;
    rq1.qparamIndex = 64;
    rq1.qparamBlock = 3;

    CommandDMA1DV2 submit{};
    submit.header.type = uint16_t(CommandType::DMASubmit1D);
    submit.header.sizeBytes = sizeof(submit);
    submit.source = {uint16_t(Region::ModelConstants), 0, 0};
    submit.destination = {uint16_t(Region::TCDMScratch), 0, 16384};
    submit.length = 32;
    submit.direction = uint32_t(DMADirection::ExternalToLocal);

    CommandDMAWaitV2 wait{};
    wait.header.type = uint16_t(CommandType::DMAWait);
    wait.header.sizeBytes = sizeof(wait);
    wait.direction = uint32_t(DMADirection::ExternalToLocal);

    std::vector<uint8_t> bytes;
    Append(bytes, rq0);
    Append(bytes, gemm);
    Append(bytes, rq1);
    Append(bytes, submit);
    Append(bytes, wait);

    std::vector<SemanticCommand> commands;
    CommandDependencyGraph graph;
    std::string error;
    REQUIRE(DecodeSemanticCommandStream(bytes, commands, error));
    REQUIRE(BuildCommandDependencyGraph(commands, graph, error));
    CHECK(HasEdge(graph, 0, 1, DependencyKind::StateRAW));
    CHECK(HasEdge(graph, 1, 2, DependencyKind::StateWAR));
    CHECK(HasEdge(graph, 3, 4, DependencyKind::StateRAW));
    CHECK(HasEdge(graph, 3, 4, DependencyKind::StateWAW));

    const auto *rqWrite = FindState(commands[0], SemanticState::Quantization,
        SemanticAccessMode::Write);
    const auto *rqRead = FindState(commands[1], SemanticState::Quantization,
        SemanticAccessMode::Read);
    const auto *nextWrite = FindState(commands[2], SemanticState::Quantization,
        SemanticAccessMode::Write);
    REQUIRE(rqWrite != nullptr);
    REQUIRE(rqRead != nullptr);
    REQUIRE(nextWrite != nullptr);
    CHECK(rqWrite->generation == 1);
    CHECK(rqRead->generation == rqWrite->generation);
    CHECK(rqRead->valueIdentity == rqWrite->valueIdentity);
    CHECK(nextWrite->generation == 2);
    CHECK(nextWrite->valueIdentity != rqWrite->valueIdentity);
}

TEST_CASE("Neural-AI dependency DAG preserves explicit control fences")
{
    CommandAFUBinaryV2 first{};
    first.header.type = uint16_t(CommandType::AFUBinary);
    first.header.sizeBytes = sizeof(first);
    first.lhs = {uint16_t(Region::TCDMScratch), 0, 0};
    first.rhs = {uint16_t(Region::TCDMScratch), 0, 64};
    first.ofm = {uint16_t(Region::TCDMScratch), 0, 128};
    first.length = 32;
    CommandAFUBinaryV2 last = first;
    last.lhs.offset = 1024;
    last.rhs.offset = 1088;
    last.ofm.offset = 1152;

    std::vector<uint8_t> bytes;
    Append(bytes, first);
    AppendControl(bytes, CommandType::Barrier);
    Append(bytes, last);

    std::vector<SemanticCommand> commands;
    CommandDependencyGraph graph;
    std::string error;
    REQUIRE(DecodeSemanticCommandStream(bytes, commands, error));
    REQUIRE(BuildCommandDependencyGraph(commands, graph, error));
    CHECK(HasEdge(graph, 0, 1, DependencyKind::Control));
    CHECK(HasEdge(graph, 1, 2, DependencyKind::Control));
}

TEST_CASE("Neural-AI global scheduler overlaps engines and waits at first consumers")
{
    CommandDMA1DV2 store{};
    store.header.type = uint16_t(CommandType::DMA1D);
    store.header.sizeBytes = sizeof(store);
    store.header.layerId = 11;
    store.header.tileId = 7;
    store.source = {uint16_t(Region::TCDMScratch), 0, 0x1000};
    store.destination = {uint16_t(Region::L2TemporaryBinding), 0, 0x2000};
    store.length = 256;
    store.direction = uint32_t(DMADirection::LocalToExternal);

    CommandAFUBinaryV2 independent{};
    independent.header.type = uint16_t(CommandType::AFUBinary);
    independent.header.sizeBytes = sizeof(independent);
    independent.lhs = {uint16_t(Region::TCDMScratch), 0, 0x3000};
    independent.rhs = {uint16_t(Region::TCDMScratch), 0, 0x3200};
    independent.ofm = {uint16_t(Region::TCDMScratch), 0, 0x3400};
    independent.length = 256;

    CommandLineBufferJobV2 linebuffer{};
    linebuffer.header.type = uint16_t(CommandType::LineBufferJob);
    linebuffer.header.sizeBytes = sizeof(linebuffer);
    linebuffer.job.linebuf.inputBase = 0x4000;
    linebuffer.job.linebuf.inputH = 4;
    linebuffer.job.linebuf.rowStrideBytes = 256;
    linebuffer.job.gemm.weightAddr = 0x5000;
    linebuffer.job.gemm.ofmAddr = 0x8000;
    linebuffer.job.gemm.dimM = 4;
    linebuffer.job.rows = 4;
    linebuffer.job.kTiles = 9;

    CommandAFUBinaryV2 consumer = independent;
    consumer.lhs.offset = linebuffer.job.gemm.ofmAddr;
    consumer.ofm.offset = store.source.offset;

    std::vector<uint8_t> bytes;
    Append(bytes, store);
    Append(bytes, independent);
    Append(bytes, linebuffer);
    Append(bytes, consumer);

    std::vector<SemanticCommand> commands;
    CommandDependencyGraph dependencies;
    std::string error;
    REQUIRE(DecodeSemanticCommandStream(bytes, commands, error));
    REQUIRE(BuildCommandDependencyGraph(commands, dependencies, error));
    const int64_t blockingCycles = std::accumulate(commands.begin(), commands.end(), int64_t(0),
        [](int64_t total, const SemanticCommand &command)
        { return total + command.estimatedCycles; });

    std::vector<uint8_t> scheduledBytes;
    GlobalCommandScheduleStats stats;
    const bool scheduledOK = ScheduleGlobalCommandStream(
        commands, dependencies, scheduledBytes, stats, error);
    INFO(error);
    REQUIRE(scheduledOK);
    CHECK(stats.asynchronousSubmits == 2);
    CHECK(stats.insertedWaits == 2);
    CHECK(stats.reorderedCommands > 0);
    CHECK(stats.estimatedCycles < blockingCycles);

    std::vector<SemanticCommand> scheduled;
    CommandDependencyGraph scheduledDependencies;
    REQUIRE(DecodeSemanticCommandStream(scheduledBytes, scheduled, error));
    REQUIRE(BuildCommandDependencyGraph(scheduled, scheduledDependencies, error));
    REQUIRE(scheduled.size() == commands.size() + stats.insertedWaits);
    CHECK(scheduled[0].type == CommandType::DMASubmit1D);
    CHECK(scheduled[1].type == CommandType::LineBufferSubmit);
    CHECK(scheduled[2].type == CommandType::AFUBinary);
    CHECK(scheduled[3].type == CommandType::DMAWait);
    CHECK(scheduled[4].type == CommandType::SystolicWait);
    CHECK(scheduled[5].type == CommandType::AFUBinary);
}

TEST_CASE("Neural-AI qparam residency removes only unchanged resident blocks")
{
    CommandRQLoadV2 rq0{};
    rq0.header.type = uint16_t(CommandType::RQLoad);
    rq0.header.sizeBytes = sizeof(rq0);
    rq0.qparamIndex = 32;
    rq0.qparamCount = 32;
    rq0.qparamBlock = 2;
    rq0.header.layerId = 1;

    CommandRQLoadV2 repeated = rq0;
    repeated.header.layerId = 2;
    repeated.header.tileId = 7;
    CommandRQLoadV2 changed = rq0;
    changed.qparamIndex = 64;
    CommandRQLoadV2 restored = rq0;

    std::vector<uint8_t> bytes;
    Append(bytes, rq0);
    Append(bytes, repeated);
    Append(bytes, changed);
    Append(bytes, restored);

    std::vector<SemanticCommand> commands;
    QParamResidencyStats stats;
    std::string error;
    REQUIRE(DecodeSemanticCommandStream(bytes, commands, error));
    REQUIRE(OptimizeQParamResidency(commands, stats, error));
    CHECK(stats.inputLoads == 4);
    CHECK(stats.retainedLoads == 3);
    CHECK(stats.redundantLoads == 1);
    REQUIRE(commands.size() == 3);
    CHECK(commands[0].layerId == 1);
    CHECK(commands[1].stateAccesses[0].valueIdentity !=
          commands[0].stateAccesses[0].valueIdentity);
    CHECK(commands[2].stateAccesses[0].valueIdentity ==
          commands[0].stateAccesses[0].valueIdentity);
    CommandDependencyGraph dependencies;
    REQUIRE(BuildCommandDependencyGraph(commands, dependencies, error));
    CHECK(commands[0].stateAccesses[0].generation == 1);
    CHECK(commands[1].stateAccesses[0].generation == 2);
    CHECK(commands[2].stateAccesses[0].generation == 3);
}

TEST_CASE("Neural-AI AFU LUT residency survives unrelated work and invalidates on shared LUT users")
{
    CommandAFULutV2 first{};
    first.header.type = uint16_t(CommandType::AFULut);
    first.header.sizeBytes = sizeof(first);
    first.ifm = {uint16_t(Region::TCDMScratch), 0, 0};
    first.ofm = {uint16_t(Region::TCDMScratch), 0, 256};
    first.lut = {uint16_t(Region::ModelConstants), 0, 512};
    first.length = 64;

    CommandAFUBinaryV2 unrelated{};
    unrelated.header.type = uint16_t(CommandType::AFUBinary);
    unrelated.header.sizeBytes = sizeof(unrelated);
    unrelated.lhs = first.ofm;
    unrelated.rhs = {uint16_t(Region::TCDMScratch), 0, 1024};
    unrelated.ofm = {uint16_t(Region::TCDMScratch), 0, 1280};
    unrelated.length = 64;

    CommandAFULutV2 resident = first;
    resident.header.layerId = 2;
    resident.ifm.offset = 1536;
    resident.ofm.offset = 1792;

    CommandAFUGlobalAvgPoolV2 invalidator{};
    invalidator.header.type = uint16_t(CommandType::AFUGlobalAvgPool);
    invalidator.header.sizeBytes = sizeof(invalidator);
    invalidator.ifm = {uint16_t(Region::TCDMScratch), 0, 2048};
    invalidator.ofm = {uint16_t(Region::TCDMScratch), 0, 4096};
    invalidator.inputH = 2;
    invalidator.inputW = 2;
    invalidator.channels = 32;

    CommandAFULutV2 reload = resident;
    reload.header.layerId = 3;
    reload.ifm.offset = 4352;
    reload.ofm.offset = 4608;

    CommandAFUDFL16V2 dfl{};
    dfl.header.type = uint16_t(CommandType::AFUDFL16);
    dfl.header.sizeBytes = sizeof(dfl);
    dfl.source = {uint16_t(Region::TCDMScratch), 0, 8192};
    dfl.destination = {uint16_t(Region::TCDMScratch), 0, 12288};
    dfl.scratch = {uint16_t(Region::TCDMScratch), 0, 16384};
    dfl.expLut = {uint16_t(Region::ModelConstants), 0, 2048};
    dfl.recipLut = {uint16_t(Region::ModelConstants), 0, 3072};
    dfl.locations = 1;

    CommandAFULutV2 reloadAfterDFL = resident;
    reloadAfterDFL.header.layerId = 4;
    reloadAfterDFL.ifm.offset = 20480;
    reloadAfterDFL.ofm.offset = 20736;

    std::vector<uint8_t> bytes;
    Append(bytes, first);
    Append(bytes, unrelated);
    Append(bytes, resident);
    Append(bytes, invalidator);
    Append(bytes, reload);
    Append(bytes, dfl);
    Append(bytes, reloadAfterDFL);

    std::vector<SemanticCommand> commands;
    AFULutResidencyStats stats;
    std::string error;
    REQUIRE(DecodeSemanticCommandStream(bytes, commands, error));
    REQUIRE(OptimizeAFULutResidency(commands, stats, error));
    CHECK(stats.commands == 4);
    CHECK(stats.loads == 3);
    CHECK(stats.reused == 1);
    CHECK(stats.invalidations == 2);
    CHECK((commands[0].flags & CommandFlagAFULutReuse) == 0);
    CHECK((commands[2].flags & CommandFlagAFULutReuse) != 0);
    CHECK(commands[2].memoryAccesses.size() == 2);
    CHECK(commands[2].stateAccesses.back().mode == SemanticAccessMode::Read);
    CHECK((commands[4].flags & CommandFlagAFULutReuse) == 0);
    CHECK(commands[4].memoryAccesses.size() == 3);
    CHECK((commands[6].flags & CommandFlagAFULutReuse) == 0);
    CHECK(commands[6].memoryAccesses.size() == 3);

    CommandDependencyGraph dependencies;
    REQUIRE(BuildCommandDependencyGraph(commands, dependencies, error));
    CHECK(HasEdge(dependencies, 0, 2, DependencyKind::StateRAW));

    std::vector<SemanticCommand> roundTrip;
    REQUIRE(DecodeSemanticCommandStream(
        SerializeSemanticCommandStream(commands), roundTrip, error));
    CHECK(roundTrip[2].memoryAccesses.size() == 2);
    CHECK(roundTrip[2].stateAccesses.back().mode == SemanticAccessMode::Read);
}

TEST_CASE("Neural-AI DMA residency removes exact weight and halo reloads until either range changes")
{
    CommandDMA1DV2 weight{};
    weight.header.type = uint16_t(CommandType::DMA1D);
    weight.header.sizeBytes = sizeof(weight);
    weight.source = {uint16_t(Region::ModelConstants), 0, 4096};
    weight.destination = {uint16_t(Region::TCDMScratch), 0, 8192};
    weight.length = 1024;
    weight.direction = uint32_t(DMADirection::ExternalToLocal);

    CommandAFUBinaryV2 readOnly{};
    readOnly.header.type = uint16_t(CommandType::AFUBinary);
    readOnly.header.sizeBytes = sizeof(readOnly);
    readOnly.lhs = weight.destination;
    readOnly.rhs = {uint16_t(Region::TCDMScratch), 0, 12288};
    readOnly.ofm = {uint16_t(Region::TCDMScratch), 0, 16384};
    readOnly.length = 1024;

    CommandDMA1DV2 repeatedWeight = weight;
    repeatedWeight.header.layerId = 2;

    CommandDMA1DV2 halo = weight;
    halo.source = {uint16_t(Region::L2TemporaryBinding), 0, 2048};
    halo.destination.offset = 24576;
    halo.length = 512;
    CommandDMA1DV2 repeatedHalo = halo;
    repeatedHalo.header.tileId = 3;

    CommandAFUBinaryV2 overwrite = readOnly;
    overwrite.lhs.offset = 32768;
    overwrite.rhs.offset = 33792;
    overwrite.ofm = weight.destination;
    overwrite.length = 1024;
    CommandDMA1DV2 reloadAfterWrite = weight;
    reloadAfterWrite.header.layerId = 4;

    std::vector<uint8_t> bytes;
    Append(bytes, weight);
    Append(bytes, readOnly);
    Append(bytes, repeatedWeight);
    Append(bytes, halo);
    Append(bytes, repeatedHalo);
    Append(bytes, overwrite);
    Append(bytes, reloadAfterWrite);

    std::vector<SemanticCommand> commands;
    DMAResidencyStats stats;
    std::string error;
    REQUIRE(DecodeSemanticCommandStream(bytes, commands, error));
    REQUIRE(OptimizeDMAResidency(commands, stats, error));
    CHECK(stats.inputLoads == 5);
    CHECK(stats.retainedLoads == 3);
    CHECK(stats.redundantConstantLoads == 1);
    CHECK(stats.redundantFeatureLoads == 1);
    CHECK(stats.redundantBytes == 1536);
    REQUIRE(commands.size() == 5);
    CHECK(commands[0].type == CommandType::DMA1D);
    CHECK(commands[1].type == CommandType::AFUBinary);
    CHECK(commands[2].memoryAccesses[0].region == uint16_t(Region::L2TemporaryBinding));
    CHECK(commands[3].type == CommandType::AFUBinary);
    CHECK(commands[4].layerId == 4);

    CommandDependencyGraph dependencies;
    REQUIRE(BuildCommandDependencyGraph(commands, dependencies, error));
    CHECK(HasEdge(dependencies, 0, 1, DependencyKind::MemoryRAW));
    CHECK(HasEdge(dependencies, 3, 4, DependencyKind::MemoryWAW));
}

TEST_CASE("Neural-AI affine loops compact and exactly restore repeated command bodies")
{
    std::vector<uint8_t> original;
    for ( uint32_t iteration = 0; iteration < 4; ++iteration )
    {
        CommandDMA2DV2 dma{};
        dma.header.type = uint16_t(CommandType::DMA2D);
        dma.header.sizeBytes = sizeof(dma);
        dma.header.layerId = 7;
        dma.header.tileId = iteration;
        dma.source = {uint16_t(Region::ModelConstants), 0, 4096 + iteration * 64};
        dma.destination = {uint16_t(Region::TCDMScratch), 0, 8192 + iteration * 64};
        dma.length = 32;
        dma.sourceStride2 = 32;
        dma.destinationStride2 = 32;
        dma.repetitions2 = 1;
        dma.direction = uint32_t(DMADirection::ExternalToLocal);
        Append(original, dma);

        CommandRQLoadV2 rq{};
        rq.header.type = uint16_t(CommandType::RQLoad);
        rq.header.sizeBytes = sizeof(rq);
        rq.header.layerId = 7;
        rq.header.tileId = iteration;
        rq.qparamIndex = iteration * 2;
        rq.qparamCount = 1;
        rq.qparamBlock = iteration;
        Append(original, rq);
    }
    AppendControl(original, CommandType::End);

    std::vector<SemanticCommand> commands;
    std::string error;
    REQUIRE(DecodeSemanticCommandStream(original, commands, error));
    std::vector<uint8_t> compacted;
    CommandCompactionStats stats;
    REQUIRE(CompactAffineCommandStream(commands, compacted, stats, error));
    CHECK(stats.logicalCommands == 8);
    CHECK(stats.encodedCommands == 3);
    CHECK(stats.loops == 1);
    CHECK(stats.loopedCommands == 8);
    CHECK(stats.bytesBefore == original.size());
    CHECK(stats.bytesAfter < stats.bytesBefore);
    CHECK((uint16_t(compacted[0]) | uint16_t(uint16_t(compacted[1]) << 8)) ==
          uint16_t(CommandType::AffineLoop));

    std::vector<uint8_t> expanded;
    uint32_t logicalCommands = 0;
    REQUIRE(ExpandAffineCommandStream(compacted, expanded, logicalCommands, error));
    CHECK(logicalCommands == 8);
    CHECK(expanded == original);

    const uint32_t firstPatchOffset = sizeof(CommandAffineLoopV2);
    compacted[firstPatchOffset] = 0;
    compacted[firstPatchOffset + 1] = 0;
    compacted[firstPatchOffset + 2] = 0;
    compacted[firstPatchOffset + 3] = 0;
    CHECK_FALSE(ExpandAffineCommandStream(compacted, expanded, logicalCommands, error));
}

TEST_CASE("Neural-AI scheduler preloads qparams into shadow registers during systolic work")
{
    CommandRQLoadV2 rq0{};
    rq0.header.type = uint16_t(CommandType::RQLoad);
    rq0.header.sizeBytes = sizeof(rq0);
    rq0.qparamCount = 32;

    CommandLineBufferJobV2 first{};
    first.header.type = uint16_t(CommandType::LineBufferJob);
    first.header.sizeBytes = sizeof(first);
    first.job.linebuf.inputBase = 0x1000;
    first.job.linebuf.inputH = 4;
    first.job.linebuf.rowStrideBytes = 256;
    first.job.gemm.weightAddr = 0x2000;
    first.job.gemm.ofmAddr = 0x4000;
    first.job.gemm.dimM = 64;
    first.job.rows = 64;
    first.job.kTiles = 9;

    CommandRQLoadV2 rq1 = rq0;
    rq1.qparamIndex = 32;
    rq1.qparamBlock = 1;
    CommandLineBufferJobV2 second = first;
    second.job.linebuf.inputBase = 0x6000;
    second.job.gemm.weightAddr = 0x7000;
    second.job.gemm.ofmAddr = 0x9000;

    std::vector<uint8_t> bytes;
    Append(bytes, rq0);
    Append(bytes, first);
    Append(bytes, rq1);
    Append(bytes, second);

    std::vector<SemanticCommand> commands;
    CommandDependencyGraph dependencies;
    std::string error;
    REQUIRE(DecodeSemanticCommandStream(bytes, commands, error));
    REQUIRE(BuildCommandDependencyGraph(commands, dependencies, error));
    REQUIRE(HasEdge(dependencies, 1, 2, DependencyKind::StateWAR));

    std::vector<uint8_t> scheduledBytes;
    GlobalCommandScheduleStats stats;
    REQUIRE(ScheduleGlobalCommandStream(
        commands, dependencies, scheduledBytes, stats, error));
    CHECK(stats.qparamPreloads == 1);
    CHECK(stats.asynchronousSubmits == 1);

    std::vector<SemanticCommand> scheduled;
    REQUIRE(DecodeSemanticCommandStream(scheduledBytes, scheduled, error));
    REQUIRE(scheduled.size() == 5);
    CHECK(scheduled[0].type == CommandType::RQLoad);
    CHECK(scheduled[1].type == CommandType::LineBufferSubmit);
    CHECK(scheduled[2].type == CommandType::RQLoad);
    CHECK(scheduled[3].type == CommandType::SystolicWait);
    CHECK(scheduled[4].type == CommandType::LineBufferJob);
}

TEST_CASE("Neural-AI semantic IR rejects malformed commands")
{
    std::vector<uint8_t> bytes(sizeof(CommandHeaderV2));
    bytes[0] = uint8_t(CommandType::DMA1D);
    bytes[2] = uint8_t(sizeof(CommandDMA1DV2));
    std::vector<SemanticCommand> commands;
    std::string error;
    CHECK_FALSE(DecodeSemanticCommandStream(bytes, commands, error));
    CHECK_FALSE(error.empty());
    CHECK(commands.empty());
}
