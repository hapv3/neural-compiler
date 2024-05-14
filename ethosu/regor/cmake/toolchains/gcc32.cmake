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

# First look for a cross-compilation toolchain
set(CMAKE_TOOLCHAIN_PREFIX i686-linux-gnu-)
set(CMAKE_TOOLCHAIN_SUFFIX "" CACHE STRING "Toolchain suffix e.g. -8 for GCC8") # Can be -9 for GCC9 or -8 for GCC8 etc

find_program(_C_COMPILER "${CMAKE_TOOLCHAIN_PREFIX}gcc${CMAKE_TOOLCHAIN_SUFFIX}")
if (NOT _C_COMPILER)
    # Assume the main toolchain can manage
    set(CMAKE_TOOLCHAIN_PREFIX "")
endif()

# Find compilers
find_program(_C_COMPILER "${CMAKE_TOOLCHAIN_PREFIX}gcc${CMAKE_TOOLCHAIN_SUFFIX}" REQUIRED)
find_program(_CXX_COMPILER "${CMAKE_TOOLCHAIN_PREFIX}g++${CMAKE_TOOLCHAIN_SUFFIX}" REQUIRED)

# Now set prefix for binutils
set(CMAKE_TOOLCHAIN_PREFIX ${CMAKE_TOOLCHAIN_PREFIX}gcc-)

set(CMAKE_C_COMPILER ${_C_COMPILER})
set(CMAKE_CXX_COMPILER ${_CXX_COMPILER})

find_program(_CXX_COMPILER_AR "${CMAKE_TOOLCHAIN_PREFIX}ar${CMAKE_TOOLCHAIN_SUFFIX}" REQUIRED)
set(CMAKE_CXX_COMPILER_AR ${_CXX_COMPILER_AR})
set(CMAKE_AR ${_CXX_COMPILER_AR})

find_program(_CXX_COMPILER_RANLIB "${CMAKE_TOOLCHAIN_PREFIX}ranlib${CMAKE_TOOLCHAIN_SUFFIX}" REQUIRED)
set(CMAKE_CXX_COMPILER_RANLIB ${_CXX_COMPILER_RANLIB})
set(CMAKE_RANLIB ${_CXX_COMPILER_RANLIB})

add_compile_options("-m32" "-march=i686" "-msse" "-msse2" "-mfpmath=sse")
add_link_options("-m32")
