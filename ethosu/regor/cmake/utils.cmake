#
# SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

include_guard(GLOBAL)

# Anchor the project root folder
set(REGOR_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/.. CACHE INTERNAL "")

# Flag target interface include paths as SYSTEM to prevent warnings leaking out
function(utils_set_system_include_paths tgt)
    get_target_property(__tgt_type ${tgt} TYPE)
    if (${__tgt_type} STREQUAL "INTERFACE_LIBRARY")
        set(__default_scope INTERFACE)
    else()
        set(__default_scope PUBLIC)
    endif()
    get_target_property(itf_inc_dirs ${tgt} INTERFACE_INCLUDE_DIRECTORIES)
    if (itf_inc_dirs)
        target_include_directories(${tgt} SYSTEM BEFORE ${__default_scope} ${itf_inc_dirs})
    endif()
endfunction()

# Disable warnings for target
function(utils_disable_warnings tgt)
    get_target_property(__tgt_type ${tgt} TYPE)
    if (NOT ${__tgt_type} STREQUAL "INTERFACE_LIBRARY")
        # Remove previous setting to prevent MSVC warning
        get_target_property(copts ${tgt} COMPILE_OPTIONS)
        if (copts)
            set(new_copts)
            foreach (c IN LISTS copts)
                if (c MATCHES "^/W[0-9]")
                    continue()
                endif()
                list(APPEND new_copts "${c}")
            endforeach()
            set_target_properties(${tgt} PROPERTIES COMPILE_OPTIONS "${new_copts}")
        endif()
        target_compile_options(${tgt} PRIVATE "$<IF:$<CXX_COMPILER_ID:MSVC>,,-w>")
    endif()
endfunction()

# Find Python the right way
macro(utils_find_python)
    if (NOT Python3_FOUND)
        if(CMAKE_VERSION VERSION_LESS "3.18.0")
            if (DEFINED ENV{PYTHON_VERSION})
                find_package(Python3 $ENV{PYTHON_VERSION} EXACT COMPONENTS Interpreter Development REQUIRED)
            else()
                find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
            endif()
        else()
            if (DEFINED ENV{PYTHON_VERSION})
                find_package(Python3 $ENV{PYTHON_VERSION} EXACT COMPONENTS Interpreter Development.Module REQUIRED)
            else()
                find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
            endif()
        endif()
    endif()
endmacro()
