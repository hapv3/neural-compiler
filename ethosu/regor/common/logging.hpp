//
// SPDX-FileCopyrightText: Copyright 2021, 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common.hpp"

#include <cassert>
#include <cctype>
#include <string>

// Log tracing detail settings, set LOG_TRACE_ENABLE to
// TD_0|TD_1|TD_2 to enable those traces before including
// this logging header.
#define TD_0 (1)
#define TD_1 (2)
#define TD_2 (4)

#if !defined LOG_TRACE_ENABLE
#define LOG_TRACE_ENABLE (TD_0)
#endif

#define LOG_TRACE0_ON (((LOG_TRACE_ENABLE)&TD_0) != 0)
#define LOG_TRACE0(...) \
    { \
        if ( LOG_TRACE0_ON ) \
        { \
            Logging::Out(8)(__VA_ARGS__); \
        } \
    }

#define LOG_TRACE1_ON (((LOG_TRACE_ENABLE)&TD_1) != 0)
#define LOG_TRACE1(...) \
    { \
        if ( LOG_TRACE1_ON ) \
        { \
            Logging::Out(16)(__VA_ARGS__); \
        } \
    }

#define LOG_TRACE2_ON (((LOG_TRACE_ENABLE)&TD_2) != 0)
#define LOG_TRACE2(...) \
    { \
        if ( LOG_TRACE2_ON ) \
        { \
            Logging::Out(32)(__VA_ARGS__); \
        } \
    }

#if NDEBUG
#define LOG_DEBUG(...) \
    do \
    { \
    } while ( false )
#else
#define LOG_DEBUG(...) Logging::Out(~0u)(__VA_ARGS__)
#endif

#define LOG_PRINT(...) \
    { \
        Logging::Out(1)(__VA_ARGS__); \
    }
#define LOG_WARN(...) \
    { \
        Logging::Out(1)(__VA_ARGS__); \
    }
#define LOG_ERROR(...) \
    { \
        Logging::Out(2)(__VA_ARGS__); \
    }


extern "C" {
typedef void (*log_writer_t)(const void *data, size_t length);
}

namespace Logging
{

/// <summary>
/// Logging context, currently outputs direct to stdout
/// </summary>
class LogContext
{
private:
    std::string _prefix;
    unsigned _filterMask = ~0u;
    int _indent = 0;
    log_writer_t _logWriter = nullptr;

public:
    struct Filter
    {
    public:
        LogContext *_context;
        unsigned _mask;

    public:
        Filter(LogContext *ctx, unsigned mask) : _context(ctx), _mask(mask) {}

        template<typename... TYPES>
        void operator()(const char *format, TYPES... args) const
        {
            if ( _mask & _context->_filterMask )
            {
                _context->Write(fmt::format(format, std::forward<TYPES>(args)...));
            }
        }

        template<typename... TYPES>
        void operator()(const std::string &format, TYPES... args) const
        {
            if ( _mask & _context->_filterMask )
            {
                _context->Write(fmt::format(format.c_str(), std::forward<TYPES>(args)...));
            }
        }

        template<typename... TYPES>
        void Print(const char *format, TYPES... args) const
        {
            if ( _mask & _context->_filterMask )
            {
                _context->Write(fmt::format(format, std::forward<TYPES>(args)...));
            }
        }

        template<typename... TYPES>
        void Print(const std::string &format, TYPES... args) const
        {
            if ( _mask & _context->_filterMask )
            {
                _context->Write(fmt::format(format, std::forward<TYPES>(args)...));
            }
        }
    };

public:
    LogContext(const char *prefix, unsigned filterMask);
    ~LogContext();

public:
    void SetPrefix(const char *prefix) { _prefix = prefix; }
    void SetFilterMask(unsigned mask) { _filterMask = mask; }
    unsigned FilterMask() const { return _filterMask; }
    void SetWriter(log_writer_t writer)
    {
        _logWriter = writer;
        if ( !_logWriter ) _filterMask = 0;
    }
    void Indent();
    void Unindent();
    void Write(const std::string &s);
    void WriteLn(const std::string &s);

    template<typename... TYPES>
    void Print(const char *format, TYPES... args)
    {
        if ( _filterMask != 0 )
        {
            Write(fmt::format(format, std::forward<TYPES>(args)...));
        }
    }

    template<typename... TYPES>
    void Print(const std::string &format, TYPES... args)
    {
        if ( _filterMask != 0 )
        {
            Write(fmt::format(format, std::forward<TYPES>(args)...));
        }
    }

    Filter operator()(unsigned mask) { return Filter(this, mask); }
};

// Default output stream
extern LogContext Out;

struct LogIndenter
{
    LogContext &_ctx;
    LogIndenter(Logging::LogContext &ctx) : _ctx(ctx) { _ctx.Indent(); }
    ~LogIndenter() { _ctx.Unindent(); }
};

#define LOG_INDENT(ctx) Logging::LogIndenter MACRO_CONCAT(_indent, __LINE__)(ctx)

}  // namespace Logging
