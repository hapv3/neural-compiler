#
# SPDX-FileCopyrightText: Copyright 2021, 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.20.2)

#############################################################################
# Policies
#############################################################################

macro(regor_policy num val)
    string(LENGTH "${num}" policy_var)
    math(EXPR policy_var "4 - ${policy_var}")
    string(REPEAT "0" ${policy_var} policy_var)
    set(policy_var "CMP${policy_var}${num}")
    if (POLICY ${policy_var})
        cmake_policy(SET ${policy_var} ${val})
        set(CMAKE_POLICY_DEFAULT_${policy_var} ${val})
    endif()
    unset(policy_var)
endmacro()

regor_policy(63 NEW)
regor_policy(69 NEW)
regor_policy(91 NEW)
regor_policy(92 NEW)
regor_policy(94 NEW)
regor_policy(117 NEW)

#############################################################################
# Project
#############################################################################

if ("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    project(regor
        VERSION 0.1.0
        DESCRIPTION "Regor Ethos-U compiler"
        LANGUAGES CXX)

    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
        message(FATAL_ERROR "${CMAKE_PROJECT_NAME} requires an out of source build.")
    endif()
endif()

#############################################################################
# Folders
#############################################################################

include(GNUInstallDirs)
# Set libdir in a consistent way
math(EXPR arch_bits "${CMAKE_SIZEOF_VOID_P}*8")
if (arch_bits EQUAL 64)
    set(CMAKE_INSTALL_LIBDIR "lib64" CACHE PATH "Libdir path" FORCE)
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation path" FORCE)
endif()

#############################################################################
# Config
#############################################################################

set(DEFAULT_CMAKE_BUILD_TYPE "Debug")
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to ${DEFAULT_CMAKE_BUILD_TYPE}")
    set(CMAKE_BUILD_TYPE "${DEFAULT_CMAKE_BUILD_TYPE}" CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)
endif()

if (CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
endif()

#############################################################################
# Options
#############################################################################

# CCACHE support
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    message(STATUS "Looking for CCACHE support - Success")
    set(ENABLE_CCACHE_DEFAULT ON)
else()
    message(STATUS "Looking for CCACHE support - Not found")
    set(ENABLE_CCACHE_DEFAULT OFF)
endif()

# LTO/IPO support. Only enable in Release mode
include(CheckIPOSupported)
check_ipo_supported(RESULT ENABLE_LTO_DEFAULT)
if (ENABLE_LTO_DEFAULT)
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(ENABLE_LTO_DEFAULT OFF)
        message(STATUS "Looking for IPO support - Success (default disabled)")
    else()
        message(STATUS "Looking for IPO support - Success")
    endif()
else()
    message(STATUS "Looking for IPO support - Not found")
endif()

# Gold linker
find_program(LD_GOLD "ld.gold")
if (LD_GOLD)
    message(STATUS "Looking for ld.gold support - Success (default disabled)")
    set(ENABLE_LDGOLD_DEFAULT OFF)
else()
    message(STATUS "Looking for ld.gold support - Not found")
    set(ENABLE_LDGOLD_DEFAULT OFF)
endif()

# Werror
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(ENABLE_WERROR_DEFAULT OFF)
else()
    set(ENABLE_WERROR_DEFAULT ON)
endif()

# Asserts
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(ENABLE_ASSERT_DEFAULT ON)
else()
    set(ENABLE_ASSERT_DEFAULT OFF)
endif()

option(REGOR_ENABLE_LTO "Link Time Optimization" ${ENABLE_LTO_DEFAULT})
option(REGOR_ENABLE_LDGOLD "Enable Gold linker if available" ${ENABLE_LDGOLD_DEFAULT})
option(REGOR_ENABLE_CCACHE "Enable ccache if available" ${ENABLE_CCACHE_DEFAULT})
option(REGOR_ENABLE_WERROR "Warnings as errors" ${ENABLE_WERROR_DEFAULT})
option(REGOR_ENABLE_STD_STATIC "Link libstdc and libgcc statically" OFF)
option(REGOR_ENABLE_COVERAGE "Enable coverage build" OFF)
option(REGOR_ENABLE_PROFILING "Enable timer based runtime profiling" OFF)
option(REGOR_ENABLE_ASSERT "Enable asserts" ${ENABLE_ASSERT_DEFAULT})
option(REGOR_ENABLE_EXPENSIVE_CHECKS "Enable expensive STL GLICXX asserts" OFF)
option(REGOR_ENABLE_RTTI "Enable RTTI" OFF)
option(REGOR_ENABLE_VALGRIND "Enable valgrind during check target" OFF)
option(REGOR_ENABLE_TESTING "Enable unit testing" ON)
option(REGOR_ENABLE_CPPCHECK "Enable cppcheck" OFF)
set(REGOR_SANITIZE "" CACHE STRING "Sanitizer setting. For example undefined")
set(REGOR_LOG_TRACE_MASK "" CACHE STRING "Log trace enable mask")
set(REGOR_PACKAGE_NAME "${PROJECT_NAME}" CACHE STRING "CPack package name. Will be suffixed with platform tag")
set(REGOR_DEBUG_COMPRESSION "zlib-gnu" CACHE STRING "Debug symbol compression. none, zlib or zlib-gnu")
set(REGOR_PYTHON_BINDINGS_DESTINATION "" CACHE STRING "Python bindings install destination")
set(REGOR_PYEXT_VERSION "${${PROJECT_NAME}_VERSION}" CACHE STRING "Python extension version")

#############################################################################
# General purpose
#############################################################################

# Modules
list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_LIST_DIR}/cmake/
    ${CMAKE_CURRENT_LIST_DIR}/dependencies/thirdparty/Catch2
)

include(utils)

# Python
utils_find_python()

# Export compile commands and flags
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Action CCache
if (REGOR_ENABLE_CCACHE)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

# Action CPPCHECK
if (REGOR_ENABLE_CPPCHECK)
    find_program(CPPCHECK cppcheck)
    if (CPPCHECK)
        message(STATUS "Looking for cppcheck support - Success")
        set(CMAKE_CXX_CPPCHECK ${CMAKE_COMMAND} -E env CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/tools/cppcheck.py
            ${CPPCHECK} --enable=all --inconclusive --quiet
            --suppress=missingIncludeSystem
            --suppress=*:${CMAKE_CURRENT_LIST_DIR}/dependencies/*
        )
    else()
        message(STATUS "Looking for cppcheck support - Not found")
    endif()
endif()

#############################################################################
# Modules
#############################################################################

include(regor_options)
include(regor_lib)
include(regor_dependencies)

#############################################################################
# Top level targets
############################################################################

find_package(Threads REQUIRED)

if (REGOR_PYTHON_BINDINGS_DESTINATION AND NOT TARGET pybind11::module)
    set(PYBIND11_FINDPYTHON ON)
    regor_add_dependency(dependencies/thirdparty/pybind11)
endif()

set(REGOR_LIB_DEPS
    Threads::Threads
    regor::flatbuffers
    regor::fmt
    regor::gemmlowp
    mlw_codec_st
)

set(REGOR_HEADERS
    "include/regor.h"
    "include/regor_database.hpp"
    "include/regor_interface.hpp"
)

regor_lib(
    NAME
        regor-objects
    TYPE OBJECT
    DEFINES
        "$<$<BOOL:${REGOR_LOG_TRACE_MASK}>:LOG_TRACE_ENABLE=${REGOR_LOG_TRACE_MASK}>"
    DEPS
        ${REGOR_LIB_DEPS}
    PUBLIC_HEADERS
        ${REGOR_HEADERS}
    INC_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
    SOURCES
        "regor.cpp"
        "common/common.cpp"
        "common/data_type.cpp"
        "common/hash.cpp"
        "common/logging.cpp"
        "common/scaling.cpp"
        "common/transpose_type.cpp"
        "common/reverse_type.cpp"
        "architecture/architecture.cpp"
        "architecture/ethos_u_scaling.cpp"
        "architecture/mlw_encode.cpp"
        "architecture/ethosu55/ethos_u55.cpp"
        "architecture/ethosu55/ethos_u55_constraints.cpp"
        "architecture/ethosu55/ethos_u55_scaling.cpp"
        "architecture/ethosu55/ethos_u55_weight_encoder.cpp"
        "architecture/ethosu55/ethos_u55_performance.cpp"
        "architecture/ethosu55/ethos_u55_register_cs_generator.cpp"
        "architecture/ethosu65/ethos_u65.cpp"
        "architecture/ethosu65/ethos_u65_register_cs_generator.cpp"
        "architecture/ethosu85/ethos_u85.cpp"
        "architecture/ethosu85/ethos_u85_constraints.cpp"
        "architecture/ethosu85/ethos_u85_register_cs_generator.cpp"
        "architecture/ethosu85/ethos_u85_scaling.cpp"
        "architecture/ethosu85/ethos_u85_weight_encoder.cpp"
        "architecture/ethosu85/ethos_u85_performance.cpp"
        "compiler/attributes.cpp"
        "compiler/compiler.cpp"
        "compiler/faststorage_allocator.cpp"
        "compiler/graph_builder.cpp"
        "compiler/graph_packing.cpp"
        "compiler/high_level_command_stream_generator.cpp"
        "compiler/hillclimb_allocator.cpp"
        "compiler/live_range.cpp"
        "compiler/network_performance.cpp"
        "compiler/operation.cpp"
        "compiler/quantization.cpp"
        "compiler/raw_writer.cpp"
        "compiler/scheduler.cpp"
        "compiler/scheduler_decompose.cpp"
        "compiler/scheduler_packing.cpp"
        "compiler/scheduler_operation.cpp"
        "compiler/softmax.cpp"
        "compiler/tensor.cpp"
        "compiler/tensor_allocator.cpp"
        "compiler/tflite_graph_optimiser.cpp"
        "compiler/tflite_graph_optimiser_tp.cpp"
        "compiler/cascade_builder.cpp"
        "compiler/graph_optimiser.cpp"
        "compiler/graphir_optimiser.cpp"
        "compiler/optimiser_utils.cpp"
        "compiler/graph_validator.cpp"
        "compiler/tosa_graph_validator.cpp"
        "compiler/op_type.cpp"
        "tosa/tosa_validator.cpp"
        "tosa/tosa_argument_checks.cpp"
        "tosa/tosa_error_checks.cpp"
        "tosa/tosa_level_checks.cpp"
        "tosa/tosa_require_checks.cpp"
        "tosa/tosa_validator_version_0_60_0_profile_bi.cpp"
        "tosa/tosa_validator_version_0_80_0_profile_bi.cpp"
        "tosa/tosa_reader.cpp"
        "tosa/tosa_mapping.cpp"
        "tflite/tflite_reader.cpp"
        "tflite/tflite_writer.cpp"
        "tflite/tflite_mapping.cpp"
        "tflite/tflite_model_semantics.cpp"
)

regor_lib(
    NAME
        regor-static
    OUTPUT_NAME
        regor
    COMPONENT regor
    TYPE STATIC
    DEPS
        regor-objects
    PUBLIC_HEADERS
        ${REGOR_HEADERS}
)

if (REGOR_PYTHON_BINDINGS_DESTINATION)
    regor_lib(
        NAME
            PyRegor
        OUTPUT_NAME
            regor
        COMPONENT python-bindings
        INSTALL_LOCATION
            ${REGOR_PYTHON_BINDINGS_DESTINATION}
        TYPE PY_MODULE
        DEFINES
            REGOR_VERSION="${REGOR_PYEXT_VERSION}"
        SOURCES
            "bindings/python/py_regor.cpp"
        DEPS
            regor-objects
        COPTS
            "$<IF:$<CXX_COMPILER_ID:MSVC>,/GR,-frtti>"
    )
endif()

#############################################################################
# Subdirs
############################################################################

if (REGOR_ENABLE_TESTING)
    # Enable testing at the top level to get a top level check target
    include(CTest)
    add_subdirectory(test)
endif()

# CPack last
include(cpack_config)
