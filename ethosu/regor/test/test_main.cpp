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

#define CATCH_CONFIG_RUNNER
#include "randomize.hpp"

#include <catch_all.hpp>
#include <cstdint>
#include <iostream>
#include <stack>
#include <string>

#include "regor.h"

using namespace Catch::Clara;

std::mt19937 default_rnd_generator(0);

// To maximize random number stability we are (potentially) using multiple generators.
// The idea is that each "section" will have its own fresh generator, initialized from Catch's seed.
// This should ensure that the same values are used, even if we only re-run a single section of a test case.
struct RandomGeneratorManager : public Catch::EventListenerBase
{
    using EventListenerBase::EventListenerBase;

    virtual void sectionStarting(Catch::SectionInfo const &sectionInfo) override
    {
        EventListenerBase::sectionStarting(sectionInfo);

        // Preserve the current generator state, and re-initialize a new generator
        generators.push(default_rnd_generator);
        default_rnd_generator = std::mt19937(Catch::rngSeed());
    }

    virtual void sectionEnded(Catch::SectionStats const &sectionStats) override
    {
        EventListenerBase::sectionEnded(sectionStats);

        // Restore the previous generator state
        default_rnd_generator = generators.top();
        generators.pop();
    }

private:
    std::stack<std::mt19937> generators;
};
CATCH_REGISTER_LISTENER(RandomGeneratorManager)

bool OptNightly = false;

extern "C" {

static void LogWriterFunc(const void *data, size_t sizeBytes)
{
    std::cout.write(reinterpret_cast<const char *>(data), sizeBytes);
}
}

int main(int argc, char **argv)
{
    regor_set_logging(LogWriterFunc, ~0u);
    Catch::Session session;

    auto cli = session.cli() | Opt(OptNightly)["--nightly"]("This is a nightly run");
    session.cli(cli);

    auto ret = session.applyCommandLine(argc, argv);
    if ( ret )
    {
        return ret;
    }

    return session.run();
}
