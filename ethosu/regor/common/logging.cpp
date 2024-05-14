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

#include "logging.hpp"

namespace Logging
{

LogContext Out("", ~0u);

static const char g_indentString[] = "\t\t\t\t\t\t\t\t\t\t\t\t";

LogContext::LogContext(const char *prefix, uint32_t filterMask) : _prefix(prefix), _filterMask(filterMask)
{
}

LogContext::~LogContext()
{
}

void LogContext::Indent()
{
    assert(_indent < int(sizeof(g_indentString)));
    _indent++;
}

void LogContext::Unindent()
{
    assert(_indent > 0);
    _indent--;
}

void LogContext::Write(const std::string &s)
{
    assert(_logWriter);

    if ( !_prefix.empty() )
    {
        _logWriter(_prefix.data(), _prefix.size());
    }
    if ( _indent != 0 )
    {
        _logWriter(g_indentString, _indent);
    }

    _logWriter(s.data(), s.size());
}

void LogContext::WriteLn(const std::string &s)
{
    Write(s);
    _logWriter("\n", 1);
}

}  // namespace Logging
