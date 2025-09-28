<!--
SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the License); you may
not use this file except in compliance with the License.
You may obtain a copy of the License at

www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an AS IS BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# AGENTS.md

This AGENTS.md file provides guidance to coding agents when working with code in the Vela compiler
with the C++ (Regor) backend. Note, this is not documentation intended for humans.

## Project Overview

This is **Vela**, a compiler for Arm Ethos-U. Vela has a hybrid architecture:

- **Frontend**: Python-based CLI in `ethosu/vela/vela.py` - handles argument parsing, configuration, and orchestration
- **Backend**: C++ implementation in `ethosu/regor/` - contains the core compilation logic and optimisations

### Key components:
- **Input formats**: TOSA (.tosa) and TensorFlow Lite (.tflite) neural network models
- **Target devices**: Ethos-U55, Ethos-U65, Ethos-U85
- **Output**: Optimised command streams and weight data for the Ethos-U NPU

## Development Commands

### Install/Build
```bash
# Default Developer Python package build (also build C++ backend)
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug" CMAKE_BUILD_PARALLEL_LEVEL=10 pip3 install -e ".[dev]"

# Build with specific C++ options
CMAKE_ARGS="-DREGOR_ENABLE_WERROR=ON -DREGOR_ENABLE_CCACHE=ON" pip install -e ".[dev]"
```

### Testing
```bash
# Run C++ unit tests (Catch2)
cmake -S ethosu/regor -B build-unit-tests -DCMAKE_BUILD_TYPE=Debug -DREGOR_SANITIZE=address
cmake --build build-unit-tests -t check
```

### Linting and Formatting
```bash
# Install pre-commit hooks
pre-commit install

# Run all pre-commit checks on all files
pre-commit run --all-files

# Run specific checks
pre-commit run clang-format --all-files
```

### Running Vela
```bash
# Outputs are stored under ./out by default
# Basic usage
vela --config ethos-u-vela/ethosu/config_files/Arm/vela.ini --accelerator-config ethos-u85-2048 --system-config Ethos_U85_SYS_DRAM_High --memory-mode Dedicated_Sram_384KB  mynetwork.tflite
# Ethos-U55 base example ( needs --debug-force-regor ) 
vela --config /ethos-u-vela/ethosu/config_files/Arm/vela.ini --accelerator-config ethos-u55-128 --system-config Ethos_U55_High_End_Embedded --memory-mode Shared_Sram mynetwork.tflite --debug-force-regor
# Flag to optimise for size ( otherwise performance is default ) 
--optimise=Size

# Flags to enable full output (Very verbose for large networks, only use for smaller networks, otherwise use more specific verbose flag)
--verbose-all --enable-debug-db


# List available configurations
vela --list-config-files
```

### Supported Configs summary
- ethos-u55-<32|64|128|256> Memory mode dedicated sram not supported
- ethos-u65-<256|512>
- ethos-u85-<128|256|512|1024|2048>

Memory modes: Sram_Only, Shared_Sram, Dedicated_Sram_<256|384|512|1024>KB
System configs:
- Ethos-U55: Ethos_U55_Deep_Embedded, Ethos_U55_High_End_Embedded
- Ethos-U65: Ethos_U65_Embedded, Ethos_U65_Mid_End, Ethos_U65_High_End, Ethos_U65_Client_Server
- Ethos-U85: Ethos_U85_SYS_Flash_Low, Ethos_U85_SYS_Flash_High, Ethos_U85_SYS_DRAM_Low, Ethos_U85_SYS_DRAM_Mid, Ethos_U85_SYS_DRAM_High

## Architecture

### Frontend (`ethosu/vela/vela.py`)
- **Entry point**: `vela` command installed via pip
- **Key functions**:
  - `process_regor()`: Modern compilation path using C++ backend
  - `process()`: Legacy Python-only compilation path
- **Integration**: Calls into C++ backend via `from ethosu import regor` and `regor.compile()`

### Backend (`ethosu/regor/`)
The C++ backend contains the core compiler implementation with the most important code in `regor/compiler/`:

**Core Compilation Pipeline (`compiler/`):**
- **`compiler.cpp/hpp`**: Main compilation orchestration and configuration parsing
- **`tflite_graph_optimiser.cpp/hpp`** (>1000 lines): TensorFlow Lite specific graph optimisations
- **`tosa_graph_optimiser.cpp/hpp`**: TOSA format specific optimisations
- **`graph_optimiser.cpp/hpp`**: Base graph optimisation framework with rewrite functions
- **`graphir_optimiser.cpp/hpp`** (>1000 lines): Graph-level IR optimisations and transformations
- **`scheduler.cpp/hpp`** (>1000 lines): Operation scheduling, optimisation strategies, and execution planning
- **`scheduler_decompose.cpp/hpp`** (>1000 lines): Complex operation decomposition into hardware-supported primitives
- **`network_performance.cpp/hpp`**: Performance estimation and analysis
- **`high_level_command_stream_generator.cpp/hpp`**: High level HW command stream generation

**Memory Management:**
- **`cascade_builder.cpp/hpp`**: Cascade operation building
- **`hillclimb_allocator.cpp/hpp`**: Advanced memory allocation using hill-climbing algorithms
- **`faststorage_allocator.cpp/hpp`**: Fast storage memory allocation strategies
- **`tensor_allocator.cpp/hpp`**: General tensor memory allocation

**Graph Infrastructure:**
- **`graph_builder.cpp/hpp`**: Graph construction and building
- **`graph_packing.cpp/hpp`**: Operation packing optimisations
- **`graph_validator.cpp/hpp`**: Graph and consistency checking

**Hardware Support:**
- **`architecture/`**: Target-specific implementations for Ethos-U55/65/85
- **`raw_writer.cpp/hpp`**: Hardware-specific output generation

## Key CMake Build Options (C++ Backend)

Set via `CMAKE_ARGS="-DOPTION=VALUE"` when installing:

| Option | Description             | Values                                      |
|--------|-------------------------|---------------------------------------------|
| `REGOR_ENABLE_LTO` | Link Time Optimisation  | ON/OFF                                      |
| `REGOR_ENABLE_WERROR` | Warnings as errors      | ON/OFF                                      |
| `REGOR_ENABLE_CCACHE` | Use ccache if available | ON/OFF                                      |
| `REGOR_ENABLE_TESTING` | Enable C++ unit testing | ON/OFF                                      |
| `REGOR_LOG_TRACE_MASK` | Enable Log Trace Mask   | **0–7 (bitmask of `TD_0 \| TD_1 \| TD_2`)** |

## Logging
- **Source:** `common/logging.hpp`
- Insert `LOG_TRACE` statements where helpful for debugging (they can be disabled at compile time).
- Use the correct macro for the intended output/severity (`LOG_TRACE`, `LOG_WARN`, etc.).

## Coding Conventions

### Naming
- Use `_t`‐suffixed type names in **C** code.
- Prefix **private & protected** members with `_`.
- Avoid using **`get`** for accessors; do use **`set`** for setters.
- Prefix **globally scoped** variables with `g_`.
- Prefix **statically scoped (file-static)** variables with `s_`.
- When counting, avoid using “number” (and its variants).
- Use the term **“last”** to mean **final**, not previous.

### Variable Initialisation
- For numeric and ordinal variables, Regor uses **assignment** (`=`).
- For object initialisation and templated variables, use **construction** `()` or **initialiser-list** `{}` semantics as needed by overload behaviour.
- Don’t force zero-initialisation on objects with default construction.

### Files & File Naming
- Prefer **anonymous namespaces** for file-level statics.
- Pass `std::shared_ptr` by `const&`.
  Return by `const&` **only** when returning a stable class member (never a temporary); the caller must not outlive the owner.
- Avoid using `std::weak_ptr`.
- Don’t do real work in constructors (e.g., allocate memory, open files).
- User exceptions can be used for flow control.
- Don’t use `size_t` for indexing/lengths; use **signed** types (e.g., `int`), except when required by existing functions.
- Return **raw pointers** when exposing a member pointer that isn’t in outer scope.
- Prefer **converting constructors** for numeric casts.


## Compilation Flow

1. **Frontend** (`vela.py`) parses CLI arguments and configuration
2. **Backend Integration**: Calls `regor.compile()` with model data
3. **C++ Processing**:
   - Model reading and graph building
   - Graph optimisations (`tflite_graph_optimiser`, `graphir_optimiser`)
   - Scheduler Packing (`scheduler_packing`) - Linearise graph, operation decomposition and packing (e.g. HW fusing of conv with activation)
     - Operation decomposition (`scheduler_decompose`) - breaking complex ops into NPU primitives
   - Scheduling and memory allocation (`scheduler`, `hillclimb_allocator`)
   - High level command stream generation  (`high_level_command_stream_generator`)
   - Command stream generation for target hardware (e.g. `/architecture/ethosu85/ethos_u85_register_cs_generator)`)
4. **Output Generation**: Results returned to Python for file writing and reporting

## Other Important Files

- **`pyproject.toml`**: Python project config, dependencies, pytest settings
- **`ethosu/regor/CMakeLists.txt`**: C++ build configuration
- **`ethosu/regor/bindings/python/py_regor.cpp`**: Python bindings