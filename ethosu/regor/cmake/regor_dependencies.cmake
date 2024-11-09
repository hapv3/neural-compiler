#
# SPDX-FileCopyrightText: Copyright 2021, 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

include(regor_lib)

############################################################################
# An add_subdirectory wrapper fixing up imported targets
############################################################################

function(regor_add_dependency dir)
    add_subdirectory(${dir} ${ARGN})

    # Get all target dirs
    set(to_visit ${dir})
    set(total_dirs ${dir})
    while (to_visit)
        set(_to_visit)
        foreach (d IN LISTS to_visit)
            get_directory_property(d_dirs DIRECTORY ${d} SUBDIRECTORIES)
            if (d_dirs)
                list(APPEND _to_visit ${d_dirs})
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _to_visit)
        if (_to_visit)
            list(APPEND total_dirs ${_to_visit})
        endif()
        set(to_visit ${_to_visit})
    endwhile()
    list(REMOVE_DUPLICATES total_dirs)

    # Fix all targets
    foreach (d IN LISTS total_dirs)
        get_directory_property(bdir_targets DIRECTORY ${d} BUILDSYSTEM_TARGETS)
        foreach (bdir_target IN LISTS bdir_targets)
            # Skip custom targets
            get_target_property(tp ${bdir_target} TYPE)
            if ("${tp}" STREQUAL "UTILITY")
                continue()
            endif()

            # Add default flags
            regor_add_options(${bdir_target})

            # Set include paths as system
            utils_set_system_include_paths(${bdir_target})

            # Disable diagnostics
            utils_disable_warnings(${bdir_target})

            # Exclude from ALL. This can't be done directly in add_subdirectory
            if (NOT "${tp}" STREQUAL "INTERFACE_LIBRARY")
                set_target_properties(${bdir_target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
            endif()
        endforeach()
    endforeach()
endfunction()

############################################################################
# Add internal and thirdparty dependencies
############################################################################

regor_add_dependency("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/mlw_codec")

set(CATCH2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/thirdparty/Catch2")
regor_add_dependency("${CATCH2_DIR}")
target_include_directories(Catch2 SYSTEM INTERFACE ${CATCH2_DIR}/src/catch2)
list(APPEND CMAKE_MODULE_PATH "${CATCH2_DIR}/CMake")
list(APPEND CMAKE_MODULE_PATH "${CATCH2_DIR}/extras")
include(Catch)
add_library(regor::Catch2 ALIAS Catch2)

include(regor_thirdparty)
