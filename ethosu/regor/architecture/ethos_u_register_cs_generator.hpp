//
// SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "common/logging.hpp"

#include "register_command_stream_generator.hpp"

namespace regor
{

template<typename BASE>
class EthosURegisterCSGenerator : public IRegisterCommandStreamGenerator
{
public:
    void PrintCommandStream(const std::vector<uint32_t> &stream, std::vector<std::pair<unsigned, std::string>> &debugInfo) override
    {
        LOG_PRINT("Register command stream: {} words\n", stream.size());
        LOG_PRINT("{0:>8}: {1:8}{2:4} {3:4} - {4:30} {5:5}, {6}\n", "  Offset", "Payload", "Param", "Code", "Command", "Param", "Fields");

        size_t debugInfoIdx = 0;
        for ( unsigned i = 0; i < stream.size(); )
        {
            if ( debugInfoIdx < debugInfo.size() && debugInfo[debugInfoIdx].first == i )
            {
                LOG_PRINT("// {}\n", debugInfo[debugInfoIdx++].second);
            }
            const uint32_t *d = &stream[i];
            std::string op;
            std::vector<std::pair<std::string, std::string>> fields;
            int nrWords = static_cast<BASE *>(this)->Disassemble(d, op, fields);
            uint32_t code = *d & 0xffff;
            uint32_t par = *d >> 16;
            uint32_t payload = 0;
            if ( nrWords == 2 && i + 1 < stream.size() )
            {
                payload = stream[i + 1];
            }
            const auto &intr = nrWords == 2 ? fmt::format("{:08x}", payload) : fmt::format("{:8}", "");
            LOG_PRINT("{0:#08x}: {1} {2:04x} {3:04x} - {4:30} {5:5}", i * sizeof(uint32_t), intr, par, code, op, par);
            i += nrWords;
            for ( auto &f : fields )
            {
                LOG_PRINT(", {} = {}", f.first, f.second);
            }
            LOG_PRINT("\n");
        }
    }
};

}  // namespace regor
