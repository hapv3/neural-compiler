//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "architecture.hpp"

#include "common/logging.hpp"

#include "common/bit_flags.hpp"

BEGIN_ENUM_TABLE(regor::MemUsage)
    ADD_ENUM_NAME(None)
    ADD_ENUM_NAME(ReadOnly)
    ADD_ENUM_NAME(FeatureMap)
    ADD_ENUM_NAME(LUT)
    ADD_ENUM_NAME(Staging)
END_ENUM_TABLE()

BEGIN_ENUM_TABLE(regor::TensorFormat)
    ADD_ENUM_NAME(Unknown)
    ADD_ENUM_NAME(NHWC)
    ADD_ENUM_NAME(NHCWB16)
    ADD_ENUM_NAME(WeightsEncoded)
END_ENUM_TABLE()

namespace regor
{

IniParseResult Architecture::ParseSection(const std::string &section, IniReader *reader)
{
    // Parse the architecture config (must happen first in INI file ).
    if ( section == "architecture" )
    {
        if ( !ParseConfig(reader) )
        {
            return IniParseResult::Error;
        }
    }
    // Parse memory definitions
    else if ( section == "memory" || section.find("memory.") == 0 )
    {
        if ( !ParseMemory(reader, section) )
        {
            return IniParseResult::Error;
        }
    }
    // Parse system configuration locally
    else if ( section == "system" )
    {
        std::string key;
        std::string tmp;
        while ( reader->Begin(key) )
        {
            if ( key == "const" )
            {
                tmp = reader->Get<std::string>();
                SetReadonlyMemory(tmp);
            }
            else if ( key == "feature_maps" )
            {
                tmp = reader->Get<std::string>();
                SetFeatureMapMemory(tmp);
            }
            else if ( key == "staging" )
            {
                tmp = reader->Get<std::string>();
                SetStagingMemory(tmp);
            }
            else
            {
                LOG_WARN("Skipping parsing of unrecognised configuration option '{}' of section '{}'\n", key, section);
            }
            reader->End();
        }
    }
    else
    {
        LOG_WARN("Skipping parsing of unrecognised configuration section '{}'\n", section);
        return IniParseResult::Unknown;
    }

    return IniParseResult::Done;
}


bool Architecture::ParseMemory(IniReader *reader, const std::string &section)
{
    std::string name;
    Address size = MaxAddress();
    float bandwidth = 1;
    int readLatency = 0;
    int writeLatency = 0;
    int burstLength = 1;
    int ports_used = 0;
    int max_reads = 0;
    int max_writes = 0;
    // Parse memory definition
    std::string key;
    while ( reader->Begin(key) )
    {
        if ( key == "name" )
        {
            name = reader->Get<std::string>();
        }
        else if ( key == "size" )
        {
            size = reader->Get<int>();
            std::string suffix;
            if ( reader->Read(suffix) )
            {
                if ( suffix == "kb" )
                {
                    size *= 1024;
                }
                else if ( suffix == "mb" )
                {
                    size *= 1024 * 1024;
                }
            }
        }
        else if ( key == "bandwidth" )
        {
            bandwidth = std::max(0.0001f, reader->Get<float>());
        }
        else if ( key == "read_latency" )
        {
            readLatency = std::max(0, reader->Get<int>());
        }
        else if ( key == "write_latency" )
        {
            writeLatency = std::max(0, reader->Get<int>());
        }
        else if ( key == "burst_length" )
        {
            burstLength = std::max(1, reader->Get<int>());
        }
        else if ( key == "ports_used" )
        {
            ports_used = std::max(1, reader->Get<int>());
        }
        else if ( key == "max_reads" )
        {
            max_reads = std::max(1, reader->Get<int>());
        }
        else if ( key == "max_writes" )
        {
            max_writes = std::max(1, reader->Get<int>());
        }
        else
        {
            LOG_WARN("Skipping parsing of unrecognised memory configuration option '{}'\n", key);
        }

        reader->End();
    }

    // Add a named, sized, memory to the system memory map
    if ( name.empty() )
    {
        LOG_ERROR("Unable to parse memory configuration. All memories must have a name.\n");
        return false;
    }
    if ( (std::string("memory.") + name) != section )
    {
        LOG_ERROR("Unable to parse memory configuration. All memories must have matching name key and section name.\n");
        return false;
    }
    else if ( _memories.count(name) )
    {
        LOG_ERROR("Unable to parse memory configuration for '{}'. All memories must have a unique name.\n", name);
        return false;
    }
    else if ( size <= 0 )
    {
        LOG_ERROR("Unable to parse memory configuration for '{}' of size {} bytes. All memories must have size > 0.\n", name, size);
        return false;
    }
    else
    {
        auto memory = std::make_unique<ArchitectureMemory>(name, size);
        memory->SetParameters(bandwidth, readLatency, writeLatency, burstLength, ports_used, max_reads, max_writes);
        _memories[name] = std::move(memory);
    }

    return true;
}


bool Architecture::CheckConfiguration(std::string &error)
{
    if ( !_featuremapMemory )
    {
        error = "Feature Map memory not configured";
        return false;
    }
    if ( !_lutMemory )
    {
        error = "LUT memory not configured";
        return false;
    }
    if ( !_stagingMemory )
    {
        error = "Staging memory not configured";
        return false;
    }
    if ( !_readonlyMemory )
    {
        error = "Readonly memory not configured";
        return false;
    }
    for ( auto &mem : _memories )
    {
        if ( mem.second->SizeBytes() > MaxAddress() )
        {
            error = "Configured memory size out of bounds for memory: " + mem.first;
            return false;
        }
    }

    return true;
}

}  // namespace regor
