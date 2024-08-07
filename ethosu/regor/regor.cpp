//
// SPDX-FileCopyrightText: Copyright 2021-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "include/regor.h"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "architecture/architecture.hpp"
#include "architecture/ethosu55/ethos_u55.hpp"
#include "architecture/ethosu65/ethos_u65.hpp"
#include "architecture/ethosu85/ethos_u85.hpp"
#include "common/numeric_util.hpp"
#include "common/shape.hpp"
#include "compiler/compiler.hpp"
#include "compiler/network_performance.hpp"
#include "include/regor_interface.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

using namespace regor;

namespace
{

struct Vela
{
    std::string system_config_name;
    std::string memory_mode_name;
};

struct System
{
    std::string const_mem_area;
    std::string arena_mem_area;
    std::string cache_mem_area;
};

struct Memories
{
    struct Memory
    {
        float clock_scale = 1;
        int read_latency = 1;
        int write_latency = 1;
        int burst_length = 64;
        int ports_used = 1;
        int max_reads = 1;
        int max_writes = 1;
    };

    // Maps memory name ("Sram") to a memory struct instance
    std::unordered_map<std::string, Memory> memories;

    // Maps port ("Axi1") to memory name ("Sram")
    std::unordered_map<std::string, std::string> mapping;
};

std::unordered_map<int, std::unique_ptr<Compiler>> s_contextMap;
std::mutex s_contextMutex;

Compiler *GetContext(regor_context_t ctx)
{
    std::lock_guard<std::mutex> lock(s_contextMutex);
    auto pos = s_contextMap.find(ctx);
    if ( pos != s_contextMap.end() )
    {
        return pos->second.get();
    }
    return nullptr;
}

Vela parse_vela_section(IniReader &reader)
{
    Vela vela;

    std::string key;
    while ( reader.Begin(key) )
    {
        if ( key == "system_config_name" )
        {
            vela.system_config_name = reader.Get<std::string>();
        }
        else if ( key == "memory_mode_name" )
        {
            vela.memory_mode_name = reader.Get<std::string>();
        }

        reader.End();
    }

    return vela;
}

Memories parse_vela_system_config_section(IniReader &reader, const std::unordered_map<std::string, Memories> &allMemories)
{
    Memories memories;

    std::string key;
    while ( reader.Begin(key) )
    {
        if ( key == "inherit" )
        {
            // If there's an inherit key set, lookup that section and use it as a base
            std::string inherit = reader.Get<std::string>();
            auto item = allMemories.find(inherit);
            if ( item != allMemories.end() )
            {
                memories = item->second;
            }
            else
            {
                LOG_WARN("inherit= refers to a non-existent section\n");
            }
        }
        else if ( key == "axi0_port" )
        {
            memories.mapping["Axi0"] = reader.Get<std::string>();
        }
        else if ( key == "axi1_port" )
        {
            memories.mapping["Axi1"] = reader.Get<std::string>();
        }
        else
        {
            std::string memoryName(key, 0, key.find('_'));
            std::string memoryParameterName(key, key.find('_') + 1, std::string::npos);

            if ( memoryParameterName == "clock_scale" )
            {
                memories.memories[memoryName].clock_scale = reader.Get<float>();
            }
            else if ( memoryParameterName == "burst_length" )
            {
                memories.memories[memoryName].burst_length = reader.Get<int>();
            }
            else if ( memoryParameterName == "read_latency" )
            {
                memories.memories[memoryName].read_latency = reader.Get<int>();
            }
            else if ( memoryParameterName == "write_latency" )
            {
                memories.memories[memoryName].write_latency = reader.Get<int>();
            }
            else if ( memoryParameterName == "ports_used" )
            {
                memories.memories[memoryName].ports_used = reader.Get<int>();
            }
            else if ( memoryParameterName == "max_reads" )
            {
                memories.memories[memoryName].max_reads = reader.Get<int>();
            }
            else if ( memoryParameterName == "max_writes" )
            {
                memories.memories[memoryName].max_writes = reader.Get<int>();
            }
        }

        reader.End();
    }

    return memories;
}

System parse_vela_memory_mode_section(IniReader &reader, const std::unordered_map<std::string, System> &allSystems)
{
    System system;

    std::string key;
    while ( reader.Begin(key) )
    {
        if ( key == "inherit" )
        {
            // If there's an inherit key set, lookup that section and use it as a base
            std::string inherit = reader.Get<std::string>();
            auto item = allSystems.find(inherit);
            if ( item != allSystems.end() )
            {
                system = item->second;
            }
            else
            {
                LOG_WARN("inherit= refers to a non-existent section\n");
            }
        }
        else if ( key == "const_mem_area" )
        {
            system.const_mem_area = reader.Get<std::string>();
        }
        else if ( key == "arena_mem_area" )
        {
            system.arena_mem_area = reader.Get<std::string>();
        }
        else if ( key == "cache_mem_area" )
        {
            system.cache_mem_area = reader.Get<std::string>();
        }

        reader.End();
    }

    return system;
}

}  // namespace


// Create a new instance of the regor compiler
DLL_EXPORT int regor_create(regor_context_t *ctx, const char *archName)
{
    std::unique_ptr<Architecture> arch;
    if ( strcmp(archName, REGOR_ARCH_ETHOSU55) == 0 )
    {
        arch = std::make_unique<ArchEthosU55>();
    }
    else if ( strcmp(archName, REGOR_ARCH_ETHOSU65) == 0 )
    {
        arch = std::make_unique<ArchEthosU65>();
    }
    else if ( strcmp(archName, REGOR_ARCH_ETHOSU85) == 0 )
    {
        arch = std::make_unique<ArchEthosU85>();
    }

    if ( arch )
    {
        std::lock_guard<std::mutex> lock(s_contextMutex);
        *ctx = regor_context_t(s_contextMap.size() + 1);
        s_contextMap[*ctx] = std::make_unique<Compiler>(arch);
        return 1;
    }

    return 0;
}


// Destroy an instance of the regor compiler
DLL_EXPORT void regor_destroy(regor_context_t ctx)
{
    std::lock_guard<std::mutex> lock(s_contextMutex);
    s_contextMap.erase(ctx);
}

// Set Vela compatible system config
DLL_EXPORT int regor_set_system_config(regor_context_t ctx, const char *config_text, size_t length)
{
    Compiler *compiler = GetContext(ctx);
    if ( !compiler )
    {
        return 0;
    }

    std::string regorIni;
    std::string regorIniUnknown;

    bool foundSystemConfig = false;
    bool foundMemoryMode = false;

    Memories memories;
    System system;
    Vela vela;
    std::unordered_map<std::string, Memories> allMemories;
    std::unordered_map<std::string, System> allSystems;

    IniReader reader(config_text, length);
    ssize_t startOfSection = 0;
    std::string section;
    while ( reader.Begin(section) )
    {
        if ( section == "vela" )
        {
            vela = parse_vela_section(reader);

            reader.End();
        }
        else if ( section.find("System_Config.") == 0 )
        {
            allMemories[section] = parse_vela_system_config_section(reader, allMemories);

            std::string name(section, section.find('.') + 1);
            if ( vela.system_config_name.empty() || name == vela.system_config_name )
            {
                memories = allMemories[section];
                foundSystemConfig = true;
            }

            reader.End();
        }
        else if ( section.find("Memory_Mode.") == 0 )
        {
            allSystems[section] = parse_vela_memory_mode_section(reader, allSystems);

            std::string name(section, section.find('.') + 1);
            if ( vela.memory_mode_name.empty() || name == vela.memory_mode_name )
            {
                system = allSystems[section];
                foundMemoryMode = true;
            }

            reader.End();
        }
        else
        {
            reader.End();

            // If we encounter an unknown section, copy the entire section
            auto sz = reader.Position() - startOfSection;
            assert(sz >= 0);
            regorIniUnknown.append(config_text + startOfSection, size_t(sz));
        }

        startOfSection = reader.Position();
    }
    std::string u55String = "Ethos_U55";
    bool isU55 = vela.system_config_name.substr(0, u55String.size()) == u55String;
    int axiWidthBytes = isU55 ? 8 : 16;

    if ( (foundSystemConfig && foundMemoryMode) || !regorIniUnknown.empty() )
    {
        for ( auto &it : memories.memories )
        {
            regorIni += "[memory." + it.first + "]\n";
            regorIni += "name=" + it.first + "\n";
            regorIni += "bandwidth=" + std::to_string(it.second.clock_scale * axiWidthBytes * it.second.ports_used) + "\n";
            regorIni += "read_latency=" + std::to_string(it.second.read_latency) + "\n";
            regorIni += "write_latency=" + std::to_string(it.second.write_latency) + "\n";
            regorIni += "burst_length=" + std::to_string(it.second.burst_length) + "\n";
            regorIni += "ports_used=" + std::to_string(it.second.ports_used) + "\n";
            regorIni += "max_reads=" + std::to_string(it.second.max_reads) + "\n";
            regorIni += "max_writes=" + std::to_string(it.second.max_writes) + "\n";
        }

        if ( foundSystemConfig )
        {
            regorIni += "[system]\n";
            regorIni += "const=" + memories.mapping[system.const_mem_area] + "\n";
            regorIni += "feature_maps=" + memories.mapping[system.arena_mem_area] + "\n";
            regorIni += "staging=" + memories.mapping[system.cache_mem_area] + "\n";
        }

        regorIni += regorIniUnknown;

        if ( !compiler->ParseConfig(regorIni.c_str(), regorIni.length()) )
        {
            return 0;
        }

        return 1;
    }

    return 0;
}


// Set compiler/scheduler options
DLL_EXPORT int regor_set_compiler_options(regor_context_t ctx, const char *config_text, size_t length)
{
    Compiler *compiler = GetContext(ctx);
    if ( !compiler )
    {
        return 0;
    }

    if ( !compiler->ParseOptions(config_text, length) )
    {
        return 0;
    }

    return 1;
}


DLL_EXPORT int regor_set_callback_arg(regor_context_t ctx, void *userArg)
{
    Compiler *compiler = GetContext(ctx);
    if ( !compiler )
    {
        return 0;
    }
    compiler->userApiArg = userArg;
    return 1;
}


DLL_EXPORT int regor_compile(regor_context_t ctx, regor_format_t fmt, const void *input, size_t in_size, regor_writer_t write_func)
{
    Compiler *compiler = GetContext(ctx);
    if ( !compiler )
    {
        return 0;
    }

    if ( fmt == REGOR_INPUTFORMAT_TFLITE )
    {
        try
        {
            if ( !compiler->LoadTflite(input, in_size) ) throw std::runtime_error("Unable to parse TFLite input data");
        }
        catch ( std::runtime_error &e )
        {
            compiler->SetLastError(e.what());
            return 0;
        }
    }
    else if ( fmt == REGOR_INPUTFORMAT_TOSA )
    {
        try
        {
            if ( !compiler->LoadTosa(input, in_size) ) throw std::runtime_error("Unable to parse tosa input data");
        }
        catch ( std::runtime_error &e )
        {
            compiler->SetLastError(e.what());
            return 0;
        }
    }
    else if ( fmt != REGOR_INPUTFORMAT_GRAPHAPI )
    {
        compiler->SetLastError("Unsupported input format");
        return 0;
    }

    if ( !compiler->Compile() )
    {
        if ( compiler->LastError().empty() )
        {
            compiler->SetLastError("Compile() failed and no error set");
        }
        return 0;
    }

    if ( write_func )
    {
        int64_t size;
        auto blob = compiler->Output();
        if ( blob )
        {
            auto p = blob->Map(size);
            write_func(compiler->userApiArg, p, size);
            blob->Unmap(p);
        }
    }

    return 1;
}

// Retrieve regor data as a binary blob
DLL_EXPORT int regor_get_output(regor_context_t ctx, REGOR_NS IRegorBlob **blob)
{
    Compiler *compiler = GetContext(ctx);
    if ( !compiler )
    {
        return 0;
    }

    assert(blob && "Can't request no result");
    *blob = compiler->Output();
    if ( !*blob ) return 0;
    return 1;
}


// Return last error text (if something returns an error code)
DLL_EXPORT int regor_get_error(regor_context_t ctx, char *text, size_t *length)
{
    Compiler *compiler = GetContext(ctx);
    if ( !compiler || !length )
    {
        return 0;
    }

    size_t lenErrorText = compiler->LastError().size();
    if ( text != nullptr )
    {
        // Return result
        *length = std::min(lenErrorText, *length);
        strncpy(text, compiler->LastError().c_str(), *length);
    }
    else
    {
        // Measure
        *length = lenErrorText;
    }

    return 1;
}


// Free any data allocated by regor
DLL_EXPORT int regor_free_data(regor_context_t ctx, const void *data)
{
    Compiler *compiler = GetContext(ctx);
    if ( !compiler )
    {
        return 0;
    }
    assert(false && "unimplemented");
    UNUSED(data);
    return 0;
}


DLL_EXPORT int regor_set_logging(regor_log_writer_t log_writer, unsigned filter_mask)
{
    Logging::Out.SetWriter(log_writer);
    Logging::Out.SetFilterMask(filter_mask);
    return 1;
}


DLL_EXPORT int regor_get_perf_report(regor_context_t ctx, regor_perf_report_t *report)
{
    Compiler *compiler = GetContext(ctx);
    if ( !compiler )
    {
        return 0;
    }

    const regor::PerformanceResult &result = compiler->LastPerfResult();

    report->npuCycles = result.npuCycles;
    report->cpuCycles = result.cpuCycles;
    report->totalCycles = result.totalCycles;
    report->macCount = result.macCount;
    report->cpuOps = result.cpuOps;
    report->npuOps = result.npuOps;
    report->cascadedOps = result.cascadedOps;
    report->cascades = result.cascades;
    report->originalWeights = result.originalWeights;
    report->encodedWeights = result.encodedWeights;
    report->accessCount = result.Accesses();
    report->numMemories = 0;
    report->stagingMemory = -1;
    int index = 0;

    if ( report->access != nullptr )
    {
        for ( auto const &[archMem, memoryStat] : result.memory )
        {
            // TODO: Clean this up
            if ( report->numMemories >= 0 && report->numMemories < 4 )
            {
                size_t len = archMem->Name().copy(report->peakUsages[report->numMemories].memoryName,
                    std::size(report->peakUsages[report->numMemories].memoryName) - 1);
                report->peakUsages[report->numMemories].memoryName[len] = 0;
                report->peakUsages[report->numMemories].peakUsage = memoryStat.peakUsage;
                if ( archMem == compiler->Arch()->StagingMemory().memory )
                {
                    report->stagingMemory = report->numMemories;
                }
                report->numMemories++;
            }
            for ( auto const &[accType, access] : memoryStat.access )
            {
                size_t len = archMem->Name().copy(report->access[index].memoryName, std::size(report->access[index].memoryName) - 1);
                report->access[index].memoryName[len] = 0;

                len = EnumToString<AccessType>(accType).copy(
                    report->access[index].accessType, std::size(report->access[index].accessType) - 1);
                report->access[index].accessType[len] = 0;

                report->access[index].accessCycles = access.accessCycles;
                report->access[index].bytesRead = access.bytesRead;
                report->access[index].bytesWritten = access.bytesWritten;
                index++;
            }
        }
    }
    return 1;
}


DLL_EXPORT struct REGOR_NS IRegorReporting *regor_get_reporting_interface(regor_context_t ctx)
{
    Compiler *compiler = GetContext(ctx);
    if ( !compiler )
    {
        return nullptr;
    }

    return compiler;
}


DLL_EXPORT struct REGOR_GRAPHAPI_NS IGraphBuilder *regor_get_graph_builder(regor_context_t ctx, const char *graph_name)
{
    Compiler *compiler = GetContext(ctx);
    if ( !compiler )
    {
        return nullptr;
    }

    return compiler->CreateGraph(graph_name);
}
