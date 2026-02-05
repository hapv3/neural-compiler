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

#pragma once

#include "common/common.hpp"

#include "architecture/architecture.hpp"
#include "architecture/register_command_stream_generator.hpp"
#include "common/ordered_map.hpp"
#include "database.hpp"
#include "graph.hpp"
#include "graph_builder.hpp"
#include "graph_optimiser.hpp"
#include "graph_packing.hpp"
#include "include/regor_interface.hpp"
#include "network_performance.hpp"
#include "scheduler.hpp"
#include "tensor_allocator.hpp"

#include <deque>
#include <list>
#include <string>

#include "include/regor.h"

namespace regor
{

enum class OutputFormat : uint16_t
{
    None,
    TFLite,
    Raw,
};

enum class COPFormat : uint16_t
{
    COP1,
    COP2,
};

/// <summary>
/// Compilation options
/// </summary>
struct CompilerOptions
{
    bool verboseHighLevelCommandStream = false;
    bool verboseRegisterCommandStream = false;
    bool debugDatabase = false;
    bool perfReport = true;
    OutputFormat outputFormat = OutputFormat::TFLite;
    COPFormat copFormat = COPFormat::COP1;
};

struct CompiledGraph
{
    std::shared_ptr<Schedule> schedule;
    std::vector<std::pair<Operation *, std::unique_ptr<NPUOperation>>> npuOps;
    std::unique_ptr<Graph> newGraph;
    TensorAddressMap tensorAddressMap;
    std::vector<std::unique_ptr<CompiledGraph>> compiledSubGraphs;

    CompiledGraph() {}

    CompiledGraph(std::shared_ptr<Schedule> &&s, std::vector<std::pair<Operation *, std::unique_ptr<NPUOperation>>> &&o,
        std::unique_ptr<Graph> &&g, TensorAddressMap &&tam, std::vector<std::unique_ptr<CompiledGraph>> &&cc) :
            schedule(std::move(s)),
            npuOps(std::move(o)), newGraph(std::move(g)), tensorAddressMap(std::move(tam)), compiledSubGraphs(std::move(cc))
    {
    }
};

using CompiledGraphs = std::unordered_map<SchedulerOperation *, std::vector<CompiledGraph>>;

/// <summary>
/// Regor top level compiler context (could just become Context)
/// </summary>
class Compiler : public IRegorReporting
{
private:
    SchedulerOptions _schedulerOptions;
    CompilerOptions _compilerOptions;
    GraphOptimiserOptions _graphOptimiserOptions;
    std::unique_ptr<Architecture> _architecture;
    std::string _lastError;
    std::deque<IRegorBlob *> _output;
    PerformanceResult _perfResult;
    class Database _Db;
    std::unique_ptr<class OptimiserDatabase> _optDb;

    Graph *_entryPoint = nullptr;
    std::vector<std::unique_ptr<Graph>> _graphs;
    std::list<GraphBuilder> _builders;

    // Maps tensor UIDs to equivalence ID. This is used to be map WHILE/IF input/output tensors to its subgraphs'
    // input/output tensors so they are allocated on the same address to be able to pass input/output to/from these
    // subgraphs without copying.
    std::unordered_map<UniqueId, UniqueId> _tensorToEquivalenceID;

    // Set of compiled graph UIDs. This is used to remember which graph that has been visited so we can know which
    // graphs should be passed through to CPU.
    std::unordered_set<UniqueId> _compiledGraphs;

public:
    void *userApiArg = nullptr;

public:
    Compiler() = delete;
    Compiler(const Compiler &) = delete;
    Compiler(std::unique_ptr<Architecture> &arch);
    ~Compiler();

public:
    bool ParseConfig(const char *text, size_t size);
    bool ParseOptions(const char *text, size_t size);

    bool LoadTosa(const void *input, size_t size);
    bool LoadTflite(const void *input, size_t size);
    void Store(const std::vector<std::unique_ptr<Graph>> &graphs, const std::vector<TensorAddressMap> &tensorAddressMaps);

    bool Compile();

    [[nodiscard]] IRegorBlob *Output()
    {
        if ( _output.empty() )
        {
            return nullptr;
        }

        auto blob = _output.front();
        _output.pop_front();
        return blob;
    }

    void SetLastError(const char *message) { _lastError = message; }
    void SetLastError(const std::string &message) { _lastError = message; }
    Architecture *Arch() { return _architecture.get(); }
    const std::string &LastError() const { return _lastError; }
    const PerformanceResult &LastPerfResult() const { return _perfResult; }
    // From IRegorReporting
    IDatabase *OptimiserDatabase() { return &_Db; }

    GraphApi::IGraphBuilder *CreateGraph(const char *name);
    Graph *GetGraph(const char *name);

    ordered_map<OpType, std::vector<std::string>> GetTFLiteConstraints();

private:
    bool BuildNetwork(const char *entryGraph);
    void RecordNPUOp(const NPUOperation &npuOp, const CmdRanges &cmdRanges);

    std::vector<std::unique_ptr<Graph>>
    CompileGraphs(IncrementalLinearAllocator &readOnlyAllocator, std::vector<TensorAddressMap> &tensorAddressMaps);
    std::unique_ptr<CompiledGraph> CompileGraph(
        Graph *graph, std::unordered_map<std::string, Graph *> &graphs, IncrementalLinearAllocator &readOnlyAllocator);
    std::vector<std::unique_ptr<Graph>> LinkGraphs(std::unique_ptr<CompiledGraph> &&result, std::vector<TensorAddressMap> &tensorAddressMaps);

    Compiler &operator=(const Compiler &) = delete;
};

}  // namespace regor
