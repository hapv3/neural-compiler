#
# SPDX-FileCopyrightText: Copyright 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR i386 CACHE STRING "" FORCE)

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

# Find GCC base path and use it
find_program(_GCC_PATH i686-linux-gnu-gcc)
if (NOT _GCC_PATH)
    find_program(_GCC_PATH gcc REQUIRED)
endif()
get_filename_component(_GCC_PATH ${_GCC_PATH} DIRECTORY)
add_compile_options(--gcc-toolchain=${_GCC_PATH}/..)
add_link_options(--gcc-toolchain=${_GCC_PATH}/..)

# Add system headers from GCC
execute_process(COMMAND cpp -m32 -xc++ -Wp,-v /dev/null OUTPUT_QUIET ERROR_VARIABLE cpp_out)
string(REPLACE " " ";" cpp_out "${cpp_out}")
string(REPLACE "\n" ";" cpp_out "${cpp_out}")
foreach (e ${cpp_out})
    if (IS_DIRECTORY "${e}")
        include_directories(SYSTEM ${e})
    endif()
endforeach()

add_compile_options("-m32" "-march=i686" "-msse" "-msse2" "-mfpmath=sse")
add_link_options("-m32")
