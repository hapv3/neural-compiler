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
# Header only libs
############################################################################

add_library(fmt INTERFACE)
target_compile_definitions(fmt INTERFACE FMT_HEADER_ONLY)
target_include_directories(fmt SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dependencies/thirdparty/fmt/include>
    $<INSTALL_INTERFACE:fmt/include>)
add_library(regor::fmt ALIAS fmt)

add_library(flatbuffers INTERFACE)
target_include_directories(flatbuffers SYSTEM INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/thirdparty/flatbuffers/include")
add_library(regor::flatbuffers ALIAS flatbuffers)

add_library(gemmlowp INTERFACE)
target_include_directories(gemmlowp SYSTEM INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/thirdparty/gemmlowp")
add_library(regor::gemmlowp ALIAS gemmlowp)
