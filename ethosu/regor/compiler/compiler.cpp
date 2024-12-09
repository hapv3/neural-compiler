//
// SPDX-FileCopyrightText: Copyright 2021-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "tflite/tflite_writer.hpp"
#include "tosa/tosa_reader.hpp"

BEGIN_ENUM_TABLE(regor::OutputFormat)
    ADD_ENUM_NAME(None)
    ADD_ENUM_NAME(TFLite)
    ADD_ENUM_NAME(Raw)
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
                    flags.Parse(reader.Get<std::string>());
                    _compilerOptions.outputFormat = flags;
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
            ParseSchedulerOptions(_schedulerOptions, reader);
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

    TfLiteReader::LoadGraphs(input, size, _graphs, _optDb.get(), _architecture->Constraints());
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

    // Is used to allocate all constant Npu tensors, in permanent storage
    IncrementalLinearAllocator readOnlyAllocator("read-only NPU tensors");

    // Reset network performance report
    _perfResult = {};

    // Compile each graph/subgraph separately
    std::vector<std::unique_ptr<Graph>> newGraphs;
    std::vector<std::unordered_map<const Tensor *, Address>> tensorAddressMaps;
    for ( auto &graph : _graphs )
    {
        std::unordered_map<const Tensor *, Address> tensorAddressMap;
        auto newGraph = CompileGraph(graph, readOnlyAllocator, tensorAddressMap);
        if ( !newGraph )
        {
            return false;
        }
        newGraphs.push_back(std::move(newGraph));
        tensorAddressMaps.push_back(std::move(tensorAddressMap));
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
    }

    if ( _graphs.empty() )
    {
        SetLastError("No graphs defined in network");
        return false;
    }

    // Select first graph as entrypoint if none selected
    if ( !_entryPoint )
    {
        _entryPoint = _graphs.front().get();
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
    ordered_map<UniqueId, SchedulerOperation *> opMap(scheduleOps.size());

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
        SchedulerOperation *scheduleOp;
        if ( opMap.try_get(std::get<0>(cmd), scheduleOp) )
        {
            assert(scheduleOp);
            _optDb->AddCommand(scheduleOp->_srcKey, streamId, std::get<2>(cmd) - 1, std::get<0>(cmd));
        }
        else
        {
            _optDb->AddCommand(nullptr, streamId, std::get<2>(cmd) - 1, std::get<0>(cmd));
        }
    }
}

std::unique_ptr<Graph> Compiler::CompileGraph(std::unique_ptr<Graph> &graph,
    IncrementalLinearAllocator &readOnlyAllocator, std::unordered_map<const Tensor *, Address> &tensorAddressMap)
{
    // Validate the input graph semantics
    if ( graph->Notation() == GraphNotation::GraphAPI )
    {
        auto validator = GraphValidator::MakeGraphValidator(graph->Notation(), graph->SyntaxVersion(), this);
        if ( validator == nullptr )
        {
            LOG_WARN("Input graph {0} not validated (required for GraphAPI) syntax={1:X}\n", graph->Name(), graph->SyntaxVersion());
            return nullptr;
        }
        if ( !validator->Validate(graph.get()) )
        {
            SetLastError(validator->GetErrorMsg());
            return nullptr;
        }
    }

    try
    {
        if ( graph->Notation() == GraphNotation::TFLite )
        {
            // Run GraphNotation::TFLite Preprocess/optimise step
            std::unique_ptr<GraphOptimiser> optimiser = GraphOptimiser::MakeGraphOptimiser(
                GraphNotation::TFLite, _architecture->Constraints(), _graphOptimiserOptions, _optDb.get());
            if ( optimiser )
            {
                optimiser->Process(graph.get());
            }
        }

        // Run GraphNotation::GraphAPI Preprocess/optimise step
        std::unique_ptr<GraphOptimiser> optimiser = GraphOptimiser::MakeGraphOptimiser(
            GraphNotation::GraphAPI, _architecture->Constraints(), _graphOptimiserOptions, _optDb.get());
        if ( optimiser )
        {
            optimiser->Process(graph.get());
        }
    }
    catch ( const std::runtime_error &e )
    {
        SetLastError(e.what());
        return nullptr;
    }

    // Pack/linearise graph Operations into SchedulerOperations
    SchedulerPacking packing(_architecture.get(), _schedulerOptions.disabled.All(SchedulerFeature::Grouping));
    auto scheduleOps = packing.Process(graph.get());

    // Schedule the linearised operation sequence
    Scheduler scheduler(_architecture.get(), _schedulerOptions, "graph", scheduleOps);
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
    try
    {
        newGraph = PackScheduleToGraph(npuOps, scheduleOps, tensorAddressMap, graph.get());
    }
    catch ( const std::runtime_error &e )
    {
        SetLastError(e.what());
        return nullptr;
    }

    auto customOperatorBuilder = CustomOperatorBuilder(_architecture.get(), schedule.get());
    customOperatorBuilder.AllocateScratchTensors(tensorAddressMap);

    // Work over the NPU ops, generating code
    for ( const auto &pair : npuOps )
    {
        auto *graphOp = pair.first;
        const auto *npuOp = pair.second.get();

        // Generate HLCS
        auto hlcsGenerator = HLCStreamGenerator();
        auto highLevelCommandStream = hlcsGenerator.GenerateCommandStream(npuOp, schedule.get(), _compilerOptions.verboseHighLevelCommandStream);

        // Generate LLCS for output
        CmdRanges cmdRanges;
        auto registerCommandStream = _architecture->RegisterCommandStreamGenerator()->GenerateCommandStream(
            highLevelCommandStream, &cmdRanges, _compilerOptions.verboseRegisterCommandStream);

        if ( registerCommandStream.empty() )
        {
            SetLastError("Failed to generate command stream");
            return nullptr;
        }

        if ( _optDb )
        {
            RecordNPUOp(*npuOp, cmdRanges);
        }

        try
        {
            customOperatorBuilder.Serialise(graphOp, npuOp, registerCommandStream);
        }
        catch ( const std::runtime_error &e )
        {
            SetLastError(e.what());
            return nullptr;
        }
    }

    return newGraph;
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

}  // namespace regor
