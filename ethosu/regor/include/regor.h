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

#if !defined REGOR_API_H
#define REGOR_API_H

#include <stddef.h>
#include <stdint.h>

// Regor input binary format
typedef enum regor_format_t
{
    REGOR_INPUTFORMAT_GRAPHAPI = 0,
    REGOR_INPUTFORMAT_TFLITE = 1,
    REGOR_INPUTFORMAT_TOSA = 2,
} regor_format_t;

// Regor compiler context
typedef int regor_context_t;

// Regor interfaces
#ifdef __cplusplus
namespace regor
{
#define REGOR_NS regor::
#else
#define REGOR_NS
#endif

struct IRegorReporting;
struct IRegorBlob;

#ifdef __cplusplus
}  // namespace regor
#endif

// GraphAPI interfaces
#ifdef __cplusplus
namespace GraphApi
{
#define REGOR_GRAPHAPI_NS GraphApi::
#else
#define REGOR_GRAPHAPI_NS
#endif

struct IGraphBuilder;

#ifdef __cplusplus
}  // namespace GraphAPI
#endif


typedef struct regor_memory_access_perf_t
{
    char memoryName[32];
    char accessType[32];
    int64_t bytesRead;
    int64_t bytesWritten;
    int64_t accessCycles;
} regor_memory_perf_t;

typedef struct regor_peak_memory_usage_t
{
    char memoryName[32];
    int64_t peakUsage;
} regor_peak_memory_usage_t;

typedef struct regor_perf_report_t
{
    int64_t npuCycles;
    int64_t cpuCycles;
    int64_t totalCycles;
    int64_t macCount;
    int64_t cpuOps;
    int64_t npuOps;
    int64_t cascadedOps;
    int64_t cascades;
    int64_t originalWeights;
    int64_t encodedWeights;
    int accessCount;
    int memory;
    int numMemories;
    int stagingMemory;
    regor_peak_memory_usage_t peakUsages[4];
    regor_memory_access_perf_t *access;
} regor_perf_report_t;

typedef struct regor_raw_tensor_header_t
{
    enum raw_tensor_type_t
    {
        RAW_TENSOR_TYPE_COMMAND_STREAM = 0,
        RAW_TENSOR_TYPE_READ_ONLY = 1,
        RAW_TENSOR_TYPE_SCRATCH = 2,
        RAW_TENSOR_TYPE_SCRATCH_FAST = 3,
        RAW_TENSOR_TYPE_INPUT = 4,
        RAW_TENSOR_TYPE_OUTPUT = 5,
        RAW_TENSOR_TYPE_VARIABLE = 6
    };
    enum raw_tensor_region_t
    {
        RAW_TENSOR_REGION_WEIGHTS = 0,
        RAW_TENSOR_REGION_SCRATCH = 1,
        RAW_TENSOR_REGION_SCRATCH_FAST = 2
    };
    uint8_t type;  // See enum raw_tensor_type
    union
    {
        struct
        {
            uint32_t size;
        } command_stream;
        struct
        {
            uint8_t region;  // See raw_tensor_region
            uint32_t size;
        } read_only;
        struct
        {
            uint8_t region;  // See raw_tensor_region
            uint32_t size;
            uint64_t address;
        } scratch;
        struct
        {
            uint8_t region;  // See raw_tensor_region
            uint32_t size;
            uint64_t address;
        } scratch_fast;
        struct
        {
            uint8_t region;  // See raw_tensor_region
            uint32_t size;
            uint64_t address;
            uint8_t element_size;
            uint32_t shape[4];
        } input;
        struct
        {
            uint8_t region;  // See raw_tensor_region
            uint32_t size;
            uint64_t address;
            uint8_t element_size;
            uint32_t shape[4];
        } output;
        struct
        {
            uint8_t region;  // See raw_tensor_region
            uint32_t size;
            uint64_t address;
            uint8_t element_size;
            uint32_t shape[4];
        } variable;
    } tensor;
} regor_raw_tensor_header_t;

#define REGOR_ARCH_ETHOSU55 "EthosU55"
#define REGOR_ARCH_ETHOSU65 "EthosU65"
#define REGOR_ARCH_ETHOSU85 "EthosU85"

#if defined __cplusplus
extern "C" {
#endif

// Create a new instance of the regor compiler
int regor_create(regor_context_t *ctx, const char *archName);

// Destroy an instance of the regor compiler
void regor_destroy(regor_context_t ctx);

// Set system configuration
int regor_set_system_config(regor_context_t ctx, const char *config_text, size_t length);

// Set compiler/scheduler options
int regor_set_compiler_options(regor_context_t ctx, const char *config_text, size_t length);

// Writer callback function (might be better, actually)
typedef size_t (*regor_writer_t)(void *userArg, const void *data, size_t size_to_write);

int regor_set_callback_arg(regor_context_t ctx, void *userArg);

int regor_compile(regor_context_t ctx, regor_format_t fmt, const void *input, size_t in_size, regor_writer_t write_func);

// Retrieve regor data as a binary blob
int regor_get_output(regor_context_t ctx, REGOR_NS IRegorBlob **blob);

// Return last error text (if something returns an error code), call twice to size buffer
int regor_get_error(regor_context_t ctx, char *text, size_t *length);

// Free any data allocated by regor
int regor_free_data(regor_context_t ctx, const void *data);

// Set logging output
typedef void (*regor_log_writer_t)(const void *data, size_t size_to_write);

int regor_set_logging(regor_log_writer_t log_writer, unsigned filter_mask);

int regor_get_perf_report(regor_context_t ctx, regor_perf_report_t *report);

struct REGOR_NS IRegorReporting *regor_get_reporting_interface(regor_context_t ctx);

struct REGOR_GRAPHAPI_NS IGraphBuilder *regor_get_graph_builder(regor_context_t ctx, const char *graph_name);

#if defined __cplusplus
}  // extern "C"
#endif

#endif  // REGOR_API_H
