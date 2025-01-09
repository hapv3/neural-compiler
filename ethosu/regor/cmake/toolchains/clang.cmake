#
# SPDX-FileCopyrightText: Copyright 2023,2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

set(CMAKE_TOOLCHAIN_PREFIX llvm-)

find_program(_C_COMPILER "clang" REQUIRED)
set(CMAKE_C_COMPILER ${_C_COMPILER})

find_program(_CXX_COMPILER "clang++" REQUIRED)
set(CMAKE_CXX_COMPILER ${_CXX_COMPILER})

find_program(_CXX_COMPILER_AR "${CMAKE_TOOLCHAIN_PREFIX}ar" REQUIRED)
set(CMAKE_CXX_COMPILER_AR ${_CXX_COMPILER_AR})
set(CMAKE_AR ${_CXX_COMPILER_AR})

find_program(_CXX_COMPILER_RANLIB "${CMAKE_TOOLCHAIN_PREFIX}ranlib" REQUIRED)
set(CMAKE_CXX_COMPILER_RANLIB ${_CXX_COMPILER_RANLIB})
set(CMAKE_RANLIB ${_CXX_COMPILER_RANLIB})
