//
// SPDX-FileCopyrightText: Copyright 2021-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "compiler.hpp"

#include "common/logging.hpp"

#include "architecture/register_command_stream_generator.hpp"
#include "common/bit_flags.hpp"
#include "common/ini_reader.hpp"
#include "graph_optimiser.hpp"
#include "graph_packing.hpp"
#include "graph_validator.hpp"
#include "high_level_command_stream_generator.hpp"
#include "network_performance.hpp"
#include "raw_writer.hpp"
#include "scheduler_packing.hpp"
#include "tensor_allocator.hpp"
#include "tflite/custom_operator_ethosu.hpp"
#include "tflite/tflite_reader.hpp"
#include "tflite/tflite_supported_operators.hpp"
#include "tflite/tflite_writer.hpp"
#include "tosa/tosa_reader.hpp"

#include "include/regor.h"

BEGIN_ENUM_TABLE(regor::OutputFormat)
    ADD_ENUM_NAME(None)
    ADD_ENUM_NAME(TFLite)
    ADD_ENUM_NAME(Raw)
END_ENUM_TABLE()

BEGIN_ENUM_TABLE(regor::COPFormat)
    ADD_ENUM_NAME(COP1)
    ADD_ENUM_NAME(COP2)
END_ENUM_TABLE()

namespace regor
{

Compiler::Compiler(std::unique_ptr<Architecture> &arch)
{
    _architecture = std::move(arch);
}

Compiler::~Compiler()
{
    for ( auto blob : _output )
    {
        blob->Release();
    }
}

bool Compiler::ParseConfig(const char *text, size_t size)
{
    // Get architecture configuration
    IniReader reader(text, size);

    std::string section;
    while ( reader.Begin(section) )
    {
        auto result = _architecture->ParseSection(section, &reader);
        if ( result == IniParseResult::Error )
        {
            SetLastError(fmt::format("Error parsing [{}]", section));
            return false;
        }
        reader.End();
    }

    return true;
}


bool Compiler::ParseOptions(const char *text, size_t size)
{
    Logging::Out.SetFilterMask(1 | 2 | 4);

    // Get compiler info
    IniReader reader(text, size);

    std::string section;
    while ( reader.Begin(section) )
    {
        if ( section == "debug" )
        {
            // Parse debug settings
            std::string key;
            while ( reader.Begin(key) )
            {
                if ( key == "trace" )
                {
                    bool enable = false;
                    if ( reader.Read(enable) )
                    {
                        if ( enable ) Logging::Out.SetFilterMask(Logging::Out.FilterMask() | 8 | 16 | 32);
                    }
                    else LOG_WARN("Bad debug.trace flag\n");
                }
                reader.End();
            }
        }
        else if ( section == "compiler" )
        {
            // Parse compiler options
            std::string key;
            while ( reader.Begin(key) )
            {
                if ( key == "verbose_high_level_command_stream" )
                {
                    _compilerOptions.verboseHighLevelCommandStream = reader.Get<bool>();
                }
                else if ( key == "verbose_register_command_stream" )
                {
                    _compilerOptions.verboseRegisterCommandStream = reader.Get<bool>();
                }
                else if ( key == "enable_db" )
                {
                    _compilerOptions.debugDatabase = reader.Get<bool>();
                }
                else if ( key == "perf_report" )
                {
                    _compilerOptions.perfReport = reader.Get<bool>();
                }
                else if ( key == "output_format" )
                {
                    Flags<OutputFormat> flags;
                    std::string value = reader.Get<std::string>();
                    if ( !flags.Parse(value) )
                    {
                        LOG_WARN("Unrecognised output_format '{}' not in [{}]\n", std::move(value), AllFlagsToString<OutputFormat>());
                    }
                    _compilerOptions.outputFormat = flags;
                }
                else if ( key == "cop_format" )
                {
                    Flags<COPFormat> flags;
                    std::string value = reader.Get<std::string>();
                    if ( !flags.Parse(value) )
                    {
                        LOG_WARN("Unrecognised cop_format '{}' not in [{}]\n", std::move(value), AllFlagsToString<COPFormat>());
                    }
                    _compilerOptions.copFormat = flags;
                }
                else
                {
                    LOG_TRACE0("Unknown 'compiler' configuration key: {}\n", key);
                }
                reader.End();
            }
        }
        else if ( section == "scheduler" )
        {
            if ( !ParseSchedulerOptions(_schedulerOptions, reader) )
            {
                SetLastError(fmt::format("Error parsing [{}]", section));
                return false;
            }
        }
        else if ( section == "graph" )
        {
            GraphOptimiser::ParseGraphOptimiserOptions(_graphOptimiserOptions, reader);
        }
        else
        {
            LOG_WARN("Skipping parsing of unrecognised options section '{}'\n", section);
        }

        reader.End();
    }

    return true;
}


bool Compiler::LoadTosa(const void *input, size_t size)
{
    TosaReader::LoadGraphs(input, size, _builders);
    return !_builders.empty();
}

bool Compiler::LoadTflite(const void *input, size_t size)
{
    assert(input && size > 0);

    // Instantiate debug database if required early for TFLite
    if ( _compilerOptions.debugDatabase != !!_optDb )
        _optDb = _compilerOptions.debugDatabase ? std::make_unique<class OptimiserDatabase>(&_Db) : nullptr;

    TfLiteReader::LoadGraphs(input, size, _graphs, _optDb.get());
    return !_graphs.empty();
}


// Flatbuffer output blob
class RawBlob : public IRegorBlob
{
private:
    int _refCount = 1;
    std::unique_ptr<const uint8_t[]> _buffer;
    int64_t _offset;
    int64_t _size;

public:
    RawBlob(std::unique_ptr<const uint8_t[]> buffer, int64_t offset, int64_t size) :
            _buffer(std::move(buffer)), _offset(offset), _size(size)
    {
    }

    void AddRef() { _refCount++; }

    void Release()
    {
        if ( --_refCount == 0 )
        {
            delete this;
        }
    }

    void *Map(int64_t &size)
    {
        size = _size;
        return const_cast<uint8_t *>(_buffer.get() + _offset);
    }

    void Unmap(void *) {}
};


void Compiler::Store(const std::vector<std::unique_ptr<Graph>> &graphs,
    const std::vector<std::unordered_map<const Tensor *, Address>> &tensorAddressMaps)
{
    if ( _compilerOptions.outputFormat == OutputFormat::Raw )
    {
        RawWriter writer;
        // This will serialise multiple blobs
        auto buffers = writer.Serialise(graphs, tensorAddressMaps);

        for ( auto &[buffer, bufferSize] : buffers )
        {
            RawBlob *output = new RawBlob(std::move(buffer), 0, bufferSize);
            _output.push_back(output);
        }
    }
    else
    {
        TfLiteWriter writer;
        int64_t offset;
        size_t size;

        // This will only serialise one TFLite model
        auto buffer = writer.Serialise(graphs, tensorAddressMaps, offset, size);

        RawBlob *output = new RawBlob(std::move(buffer), offset, int64_t(size));
        _output.push_back(output);
    }
}


bool Compiler::Compile()
{
    // Check that the configuration is okay to start compiling
    std::string error;
    if ( !_architecture->CheckConfiguration(error) )
    {
        SetLastError(error);
        return false;
    }

    // If no graphs defined (nothing already loaded) then create a network from the graph builders
    if ( _graphs.empty() )
    {
        if ( _builders.empty() )
        {
            SetLastError("No networks defined via GraphAPI");
            return false;
        }

        // Instantiate debug database if required for GraphAPI
        if ( _compilerOptions.debugDatabase != !!_optDb )
            _optDb = _compilerOptions.debugDatabase ? std::make_unique<class OptimiserDatabase>(&_Db) : nullptr;

        if ( !BuildNetwork(nullptr) )  // BuildNetworks sets error text
        {
            return false;
        }
    }

    // Select first graph as entrypoint if none selected
    if ( !_entryPoint )
    {
        _entryPoint = _graphs.front().get();
    }

    // Is used to allocate all constant Npu tensors, in permanent storage
    IncrementalLinearAllocator readOnlyAllocator("read-only NPU tensors");

    // Reset network performance report
    _perfResult = {};

    // Compile each graph/subgraph separately
    std::vector<std::unique_ptr<Graph>> newGraphs;
    std::vector<std::unordered_map<const Tensor *, Address>> tensorAddressMaps;
    newGraphs = CompileGraphs(readOnlyAllocator, tensorAddressMaps);
    if ( newGraphs.empty() )
    {
        return false;
    }

    _optDb.reset();
    _builders.clear();

    try
    {
        Store(newGraphs, tensorAddressMaps);
    }
    catch ( const std::invalid_argument &ex )
    {
        SetLastError(fmt::format("Output error: {} \n", ex.what()));
        return false;
    }

    return true;
}


bool Compiler::BuildNetwork(const char *entryGraph)
{
    // Iterate through the builders committing their inputs/outputs to new
    // graph objects. Any un-attached data will be dropped later.
    for ( auto &builder : _builders )
    {
        std::vector<std::shared_ptr<Tensor>> placeholders;
        auto graph = std::make_unique<Graph>(builder.Name(), builder._inputs, builder._outputs, builder._persistent,
            placeholders, GraphNotation::GraphAPI, builder.SyntaxVersion());
        if ( entryGraph && (builder.Name() == entryGraph) )
        {
            assert(!_entryPoint && "Entrypoint already set");
            _entryPoint = graph.get();
        }
        _graphs.push_back(std::move(graph));
        if ( _optDb )
        {
            for ( const auto &op : builder._operations )
            {
                if ( auto it = builder._uidToExt.find(op->Uid()); it != builder._uidToExt.end() )
                {
                    _optDb->SourceOp(op.get(), it->second);
                }
                else
                {
                    _optDb->SourceOp(op.get());
                }
            }
        }
    }

    if ( _graphs.empty() )
    {
        SetLastError("No graphs defined in network");
        return false;
    }

    // Clearing the builders will release anything that the client allocated
    // but didn't use. Unconnected operators will get freed.
    _builders.clear();
    return true;
}

void Compiler::RecordNPUOp(const NPUOperation &npuOp, const CmdRanges &cmdRanges)
{
    assert(_optDb);
    const auto &scheduleOps = npuOp.Operations();
    std::unordered_map<UniqueId, SchedulerOperation *> opMap(scheduleOps.size());

    // Record scheduler operations
    for ( const auto &scheduleOp : scheduleOps )
    {
        opMap.emplace(scheduleOp->Uid(), scheduleOp.get());

        // Add subOps to DB
        for ( auto &subOp : scheduleOp->SubOps() )
        {
            opMap.emplace(subOp->Uid(), subOp.get());
            _optDb->AddSubOp(scheduleOp->Uid(), subOp->Uid());
        }
    }

    // Record command stream op ranges for this NPU op
    int streamId = _optDb->AddStream();
    for ( auto const &cmd : cmdRanges )
    {
        try
        {
            SchedulerOperation *scheduleOp = opMap.at(std::get<0>(cmd));
            assert(scheduleOp);
            auto op = static_cast<Operation *>(scheduleOp->_srcKey);
            _optDb->AddCommand(op->Uid(), streamId, std::get<2>(cmd) - 1, std::get<0>(cmd));
        }
        catch ( const std::out_of_range & )
        {
            _optDb->AddCommand(INVALID_UID, streamId, std::get<2>(cmd) - 1, std::get<0>(cmd));
        }
    }
}

std::vector<std::unique_ptr<Graph>> Compiler::CompileGraphs(IncrementalLinearAllocator &readOnlyAllocator,
    std::vector<std::unordered_map<const Tensor *, Address>> &tensorAddressMaps)
{
    // Generated a graph name -> graph mapping
    std::unordered_map<std::string, Graph *> graphsMap;
    for ( const auto &graph : _graphs )
    {
        graphsMap[graph->Name()] = graph.get();
    }

    // Compile graphs, starting with the entry point
    std::vector<std::unique_ptr<Graph>> newGraphs;
    if ( _entryPoint )
    {
        auto compiledGraphTree = CompileGraph(_entryPoint, graphsMap, readOnlyAllocator);
        if ( !compiledGraphTree )
        {
            return {};
        }

        // Link the compiled graphs
        newGraphs = LinkGraphs(std::move(compiledGraphTree), tensorAddressMaps);
        if ( newGraphs.empty() )
        {
            return {};
        }
    }

    // Iterate the graphs and look for graphs we haven't reached
    for ( auto &graph : _graphs )
    {
        if ( graph.get() == _entryPoint || graph->Notation() != GraphNotation::TFLite ) continue;

        if ( _compiledGraphs.count(graph->Uid()) == 0 )
        {
            std::unordered_map<std::string, Graph *> emptyGraphsMap;
            auto compiled = CompileGraph(graph.get(), emptyGraphsMap, readOnlyAllocator);
            if ( !compiled )
            {
                return {};
            }

            // Link the compiled graphs
            auto linked = LinkGraphs(std::move(compiled), tensorAddressMaps);
            if ( linked.empty() )
            {
                return {};
            }
            newGraphs.insert(newGraphs.end(), std::make_move_iterator(linked.begin()), std::make_move_iterator(linked.end()));
        }
        else
        {
            // Insert a placeholder (empty) graph to preserve the original graph indexing. The ops from this graph were
            // inlined into the main graph, but TFLite CALL/IF/WHILE ops on CPU still reference other graphs by their
            // indices, so the index order must remain intact.
            auto newGraph = std::make_unique<Graph>(graph->Notation());
            newGraph->SetName(graph->Name());
            newGraph->SetPassthrough(graph->Passthrough());
            newGraphs.push_back(std::move(newGraph));

            // Insert a matching empty tensor address map
            tensorAddressMaps.emplace_back();
        }
    }

    return newGraphs;
}

static void mapInputs(const Graph *subgraph, const SchedulerOperation *op, std::unordered_map<UniqueId, UniqueId> &tensorToEqId, int offset)
{
    // Map graph inputs to caller's inputs
    int index = offset;
    for ( const auto &input : subgraph->Inputs() )
    {
        auto ifmConn = op->Input(MakeTensorUsage(TensorUsage::IFM, index++));
        tensorToEqId[input->Uid()] = ifmConn->tensor->equivalenceId;
        LOG_TRACE1("Mapping graph input {} to {} (eqID {})\n", input->Name(), ifmConn->tensor->Name(), ifmConn->tensor->equivalenceId);
    }
}

static void mapOutputs(const Graph *subgraph, const SchedulerOperation *op, std::unordered_map<UniqueId, UniqueId> &tensorToEqId, int offset)
{
    // Map graph outputs to caller's outputs
    int index = offset;
    for ( const auto &output : subgraph->Outputs() )
    {
        auto ofmConn = op->Output(MakeTensorUsage(TensorUsage::OFM, index++));
        tensorToEqId[output->Uid()] = ofmConn->tensor->equivalenceId;
        LOG_TRACE1("Mapping graph output {} to {} (eqID {})\n", output->Name(), ofmConn->tensor->Name(), ofmConn->tensor->equivalenceId);
    }
}

// Compiled a graph and its subgraphs
std::unique_ptr<CompiledGraph> Compiler::CompileGraph(
    Graph *graph, std::unordered_map<std::string, Graph *> &graphs, IncrementalLinearAllocator &readOnlyAllocator)
{
    LOG_TRACE1("Compiling graph {}\n", graph->Name());

    // Validate the input graph semantics
    if ( graph->Notation() == GraphNotation::GraphAPI )
    {
        auto validator = GraphValidator::MakeGraphValidator(graph->Notation(), graph->SyntaxVersion(), this);
        if ( validator == nullptr )
        {
            LOG_WARN("Input graph {0} not validated (required for GraphAPI) syntax={1:X}\n", graph->Name(), graph->SyntaxVersion());
            return nullptr;
        }
        if ( !validator->Validate(graph) )
        {
            SetLastError(validator->GetErrorMsg());
            return nullptr;
        }
    }

    // Remember that we compiled this graph
    _compiledGraphs.insert(graph->Uid());

    try
    {
        // Create both the specific pre-optimiser and the general post-optimiser
        auto graphOptimisers = GraphOptimiser::MakeGraphOptimiser(
            graph->Notation(), _architecture.get(), _graphOptimiserOptions, _optDb.get());

        // Run the graph optimisers
        for ( const auto &optimiser : graphOptimisers )
        {
            optimiser->Process(graph);
        }
    }
    catch ( const std::runtime_error &e )
    {
        SetLastError(e.what());
        return nullptr;
    }

    // Pack/linearise graph Operations into SchedulerOperations
    SchedulerPacking packing(_architecture.get(), _schedulerOptions.disabled.All(SchedulerFeature::Grouping), _tensorToEquivalenceID);
    auto scheduleOps = packing.Process(graph);

    // List of compiled graphs
    std::vector<std::unique_ptr<CompiledGraph>> compiledSubgraphs;

    // Iterate over all schedule ops here and recurse if control flow ops found
    for ( auto &op : scheduleOps )
    {
        if ( !op->IsNpuOp() || !IsControlFlow(op->Type()) ) continue;

        int subScratchSize = 0;
        if ( op->Type() == OpType::If )
        {
            auto *attr = op->Attribute<cond_attr_t>();
            auto *attrInt = op->Attribute<internal_if_attr_t>();

            auto *thenGraph = graphs.at(attr->then_branch);

            // Map subgraph input/output tensor -> eqID
            mapInputs(thenGraph, op.get(), _tensorToEquivalenceID, 1);
            mapOutputs(thenGraph, op.get(), _tensorToEquivalenceID, 0);

            // Compile the "then" branch
            auto thenGraphCompiled = CompileGraph(thenGraph, graphs, readOnlyAllocator);
            if ( !thenGraphCompiled ) return nullptr;
            auto thenPeak = thenGraphCompiled->schedule->memoryUsage.at(_architecture->FeatureMapMemory());
            attrInt->then_graph = thenGraphCompiled.get();
            compiledSubgraphs.push_back(std::move(thenGraphCompiled));

            auto *elseGraph = graphs.at(attr->else_branch);

            // Map subgraph input/output tensor -> eqID
            mapInputs(elseGraph, op.get(), _tensorToEquivalenceID, 1);
            mapOutputs(elseGraph, op.get(), _tensorToEquivalenceID, 0);

            // Compile the "else" branch
            auto elseGraphCompiled = CompileGraph(elseGraph, graphs, readOnlyAllocator);
            if ( !elseGraphCompiled ) return nullptr;
            auto elsePeak = elseGraphCompiled->schedule->memoryUsage.at(_architecture->FeatureMapMemory());
            attrInt->else_graph = elseGraphCompiled.get();
            compiledSubgraphs.push_back(std::move(elseGraphCompiled));

            subScratchSize = std::max(thenPeak, elsePeak);
        }
        else if ( op->Type() == OpType::While )
        {
            auto *attr = op->Attribute<while_attr_t>();
            auto *attrInt = op->Attribute<internal_while_attr_t>();

            auto *condGraph = graphs.at(attr->cond_branch);

            // Map subgraph input tensor -> eqID
            mapInputs(condGraph, op.get(), _tensorToEquivalenceID, 0);

            // Compile the "cond" branch
            auto condGraphCompiled = CompileGraph(condGraph, graphs, readOnlyAllocator);
            if ( !condGraphCompiled ) return nullptr;
            auto condPeak = condGraphCompiled->schedule->memoryUsage.at(_architecture->FeatureMapMemory());
            attrInt->cond_graph = condGraphCompiled.get();
            compiledSubgraphs.push_back(std::move(condGraphCompiled));

            auto *bodyGraph = graphs.at(attr->body_branch);

            // Map subgraph input/output tensor -> eqID
            mapInputs(bodyGraph, op.get(), _tensorToEquivalenceID, 0);
            mapOutputs(bodyGraph, op.get(), _tensorToEquivalenceID, 0);

            // Compile the "body" branch
            auto bodyGraphCompiled = CompileGraph(bodyGraph, graphs, readOnlyAllocator);
            if ( !bodyGraphCompiled ) return nullptr;
            auto bodyPeak = bodyGraphCompiled->schedule->memoryUsage.at(_architecture->FeatureMapMemory());
            attrInt->body_graph = bodyGraphCompiled.get();
            compiledSubgraphs.push_back(std::move(bodyGraphCompiled));

            subScratchSize = std::max(condPeak, bodyPeak);
        }

        // Create a carve-out tensor for the branches
        auto subScratch = std::make_shared<SchedulerTensor>(DataType::Int8, Shape(1, 1, 1, subScratchSize), TensorFormat::NHWC);
        subScratch->SetInternalName(fmt::format("carveout-for-op-{}", op->Uid()));
        subScratch->memArea = _architecture->FeatureMapMemory();
        op->ConnectInput(TensorUsage::Scratch, subScratch);
    }

    // TODO: make sure that we don't include graph input/output tensors in memory alloc for subgraphs
    // Schedule the linearised operation sequence
    auto schedulerOptions = _schedulerOptions;
    // Only use separate IO regions on the main graph
    if ( graph != _entryPoint ) schedulerOptions.separateIORegions = false;
    Scheduler scheduler(_architecture.get(), schedulerOptions, "graph", scheduleOps, packing.OpConfigCompatablility());
    std::shared_ptr<Schedule> schedule;
    try
    {
        schedule = scheduler.Process();
    }
    catch ( const std::runtime_error &e )
    {
        SetLastError(e.what());
        return nullptr;
    }

    scheduler.AllocateReadOnlyAddresses(schedule.get(), readOnlyAllocator);

    // Calculate full network performance
    if ( _compilerOptions.perfReport )
    {
        NetworkPerformance perf(_architecture.get(), scheduleOps);
        _perfResult += perf.Measure(schedule.get(), _optDb.get());
    }

    // Get a new graph and NPU operations from the scheduled operations
    std::vector<std::pair<Operation *, std::unique_ptr<NPUOperation>>> npuOps;
    std::unique_ptr<Graph> newGraph;
    std::unordered_map<const Tensor *, Address> tensorAddressMap;
    try
    {
        newGraph = PackScheduleToGraph(npuOps, scheduleOps, tensorAddressMap, graph);
    }
    catch ( const std::runtime_error &e )
    {
        SetLastError(e.what());
        return nullptr;
    }

    // Work over the NPU ops, generating code
    for ( const auto &pair : npuOps )
    {
        const auto *npuOp = pair.second.get();

        // Allocate addresses for IO tensors with an address space that is local for this sequence of NPU ops
        scheduler.AllocateIOAddresses(schedule.get(), npuOp->Operations());
    }

    return std::make_unique<CompiledGraph>(std::move(schedule), std::move(npuOps), std::move(newGraph),
        std::move(tensorAddressMap), std::move(compiledSubgraphs));
}

std::vector<std::unique_ptr<Graph>> Compiler::LinkGraphs(std::unique_ptr<CompiledGraph> &&result,
    std::vector<std::unordered_map<const Tensor *, Address>> &tensorAddressMaps)
{
    const Schedule *schedule = result->schedule.get();
    const int featureMapSize = schedule->memoryUsage.at(_architecture->FeatureMapMemory());
    const int stagingSize = schedule->memoryUsage.at(_architecture->StagingMemory());
    const int readOnlySize = schedule->memoryUsage.at(_architecture->ReadonlyMemory());
    CustomOperatorBuilder customOperatorBuilder(_architecture.get(), featureMapSize, stagingSize, readOnlySize);
    customOperatorBuilder.AllocateScratchTensors(result->schedule.get(), result->tensorAddressMap);

    // Work over the NPU ops, generating code
    for ( const auto &pair : result->npuOps )
    {
        auto *graphOp = pair.first;
        const auto *npuOp = pair.second.get();

        // Generate HLCS
        HLCStreamGenerator hlcsGenerator(0 /* base */, _compilerOptions.verboseHighLevelCommandStream);
        auto highLevelCommandStream = hlcsGenerator.GenerateCommandStream(npuOp, schedule, result->compiledSubGraphs);

        // Generate LLCS for output
        auto rcsGen = _architecture->RegisterCommandStreamGenerator();
        CmdRanges cmdRanges;
        auto registerCommandStream = rcsGen->GenerateCommandStream(highLevelCommandStream, &cmdRanges, _compilerOptions.verboseRegisterCommandStream);
        if ( registerCommandStream.empty() )
        {
            SetLastError("Failed to generate command stream");
            return {};
        }

        if ( _optDb )
        {
            RecordNPUOp(*npuOp, cmdRanges);
        }

        try
        {
            customOperatorBuilder.Serialise(schedule, graphOp, npuOp, _compilerOptions.copFormat,
                _schedulerOptions.separateIORegions, registerCommandStream);
        }
        catch ( const std::runtime_error &e )
        {
            SetLastError(e.what());
            return {};
        }
    }

    // Collect the packed graphs and return them
    std::vector<std::unique_ptr<Graph>> newGraphs;
    std::function<void(const std::unique_ptr<CompiledGraph> &)> collectNewGraphs;
    collectNewGraphs = [&](const std::unique_ptr<CompiledGraph> &cg) -> void
    {
        if ( result == cg )
        {
            newGraphs.push_back(std::move(cg->newGraph));
            tensorAddressMaps.push_back(std::move(cg->tensorAddressMap));
        }

        for ( const auto &[graphOp, npuOp] : cg->npuOps )
        {
            // Collect read-only tensors of each op
            for ( const auto &op : npuOp->Operations() )
            {
                customOperatorBuilder.SerialiseSchedOp(cg->schedule.get(), op);
            }
        }

        // Recurse to subgraphs
        for ( const auto &subcg : cg->compiledSubGraphs )
        {
            if ( subcg->newGraph ) collectNewGraphs(subcg);
        }
    };
    collectNewGraphs(result);

    return newGraphs;
}

GraphApi::IGraphBuilder *Compiler::CreateGraph(const char *name)
{
    auto pos = std::find_if(_builders.begin(), _builders.end(), [&](auto &b) { return b.Name() == name; });
    if ( pos != _builders.end() )
    {
        return &(*pos);
    }

    _builders.emplace_back(name);
    return &_builders.back();
}

Graph *Compiler::GetGraph(const char *name)
{
    auto pos = std::find_if(_graphs.begin(), _graphs.end(), [&](auto &b) { return b->Name() == name; });
    if ( pos != _graphs.end() )
    {
        return pos->get();
    }
    return nullptr;
}

ordered_map<OpType, std::vector<std::string>> Compiler::GetTFLiteConstraints()
{
    std::unique_ptr<TfLiteSupportedOperators> supportedOps;
    _architecture->Call([&supportedOps](const std::string &target) { supportedOps = MakeSupportedOpsChecker(target); });
    return supportedOps->Documentation();
}

}  // namespace regor
