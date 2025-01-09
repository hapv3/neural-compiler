#
# SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

cmake_minimum_required(VERSION 3.15.6)
cmake_policy(SET CMP0063 NEW)
project(mlw_codec VERSION 1.0 DESCRIPTION "MLW Codec" LANGUAGES C CXX)
include(GNUInstallDirs)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(DEBUG_PACKET "Debug packet syntax" OFF)
option(DEBUG_BITSTREAM "Debug bitstream contents" OFF)

if (DEBUG_PACKET)
    list(APPEND MLW_DEFINES "ENABLE_DEBUG_PACKET=1")
endif()

if (DEBUG_BITSTREAM)
    list(APPEND MLW_DEFINES "ENABLE_DEBUG_BITSTREAM=1")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Add a target library made from the above source files
add_library(mlw_codec_st STATIC
    "source/mlw_encode.cpp"
    "source/mlw_encode_fwd.cpp"
    "source/ml_ethosu_encode.cpp"
    "source/mlw_decode.cpp"
)
set_target_properties(mlw_codec_st PROPERTIES OUTPUT_NAME mlw_codec)

# Default compiler settings
set_property(TARGET mlw_codec_st PROPERTY CXX_STANDARD 11)
set_property(TARGET mlw_codec_st PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET mlw_codec_st PROPERTY POSITION_INDEPENDENT_CODE ON)

add_custom_target(mlw_codec DEPENDS mlw_codec_st)

if (MSVC)
    set(MLW_TOOLCHAIN_CXX_OPTIONS
        "/DWIN32_LEAN_AND_MEAN"
        "/DNOMINMAX"
        "/D_USE_MATH_DEFINES"
        "/D_CRT_SECURE_NO_WARNINGS"
        "/D_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES"
        "/bigobj"
        "/experimental:external"
        "/external:W0"
        "/external:anglebrackets"
        # Diagnostics
        "/W3"      # Default warning level (severe + significant + production quality).
        "/wd4200"  # "nonstandard extension used : zero-sized array in struct/union"
        "/wd4018"  # "signed/unsigned mismatch in comparison"
        "/wd4146"  # operator applied to unsigned type, result still unsigned
        "/wd4244"  # possible loss of data
        "/wd4267"  # initializing: possible loss of data
        "/wd4005"  # allow: macro redefinition
        "/wd4065"  # allow: switch statement contains 'default' but no 'case' labels
        "/wd4141"  # allow: inline used more than once
        "/wd4624"  # allow: destructor was implicitly defined as deleted
        "/wd4146"  # operator applied to unsigned type, result still unsigned
        "/wd4244"  # possible loss of data
        "/wd4267"  # initializing: possible loss of data
        "/wd5105"  # allow: macro expansion producing 'defined' has undefined behavior
    )
else()
    set(MLW_TOOLCHAIN_CXX_OPTIONS
        # Enabled compiler options
        -Wall -Wextra -Wsign-compare -Wold-style-cast -Wswitch-default
        -Wformat -Wdouble-promotion -Wredundant-decls -Wlogical-op
        -Wnon-virtual-dtor -Wcast-align -Wshadow
        # Disabled compiler options
        -Wno-format-contains-nul -Wno-format-extra-args
        -Wno-unused-function -Wno-unused-label -Wno-maybe-uninitialized -Wno-unused
        -Wno-unused-local-typedefs -Wno-stringop-truncation -Wno-missing-field-initializers
        # Config specific options
        $<$<CONFIG:Release>:-Werror>
    )
endif()

# Config specific compiler defines
set(MLW_TOOLCHAIN_CXX_DEFINES
    RELEASE=$<BOOL:$<CONFIG:Release>>
    DEBUG=$<BOOL:$<CONFIG:Debug>>)

# Configure target's compilation options
target_compile_definitions(mlw_codec_st PRIVATE ${MLW_DEFINES} $<$<COMPILE_LANGUAGE:CXX>:${MLW_TOOLCHAIN_CXX_DEFINES}>)
target_compile_options(mlw_codec_st PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${MLW_TOOLCHAIN_CXX_OPTIONS}>)

# Properties
file(GLOB MLW_CODEC_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/include/*)

set_target_properties(mlw_codec_st PROPERTIES
    VERSION ${${PROJECT_NAME}_VERSION}
    #PUBLIC_HEADER "${MLW_CODEC_HDRS}"
)

# Include directories (private and public interface)
target_include_directories(mlw_codec_st
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>"
)

# Install
install(TARGETS mlw_codec_st
    EXPORT ${PROJECT_NAME}
    LIBRARY       DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE       DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME       DESTINATION "${CMAKE_INSTALL_BINDIR}"
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}"
)
if(MSVC AND NOT CMAKE_VERSION VERSION_LESS 3.15)
    install(FILES
        "$<TARGET_FILE_DIR:mlw_codec_st>/$<TARGET_FILE_PREFIX:mlw_codec_st>$<TARGET_FILE_BASE_NAME:mlw_codec_st>.pdb"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        OPTIONAL
    )
endif()
install(EXPORT ${PROJECT_NAME}
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
)

# Aliases
add_library(${PROJECT_NAME}::mlw_codec_st ALIAS mlw_codec_st)

# Use a PEP-656 compliant package tag
# The default value for this variable is not useful
if (NOT Python3_FOUND)
    set(Python3_FIND_STRATEGY VERSION)
    set(Python3_FIND_REGISTRY LAST)
    set(Python3_FIND_FRAMEWORK LAST)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_platform())"
    OUTPUT_VARIABLE MLW_CODEC_SYSTEM_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
# Default variables
set(CPACK_PACKAGE_VENDOR "Arm")
set(CPACK_PACKAGE_DESCRIPTION "${PROJECT_DESCRIPTION}")
set(CPACK_PACKAGE_VERSION_MAJOR "${${PROJECT_NAME}_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${${PROJECT_NAME}_VERSION_MINOR}")
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CPACK_STRIP_FILES FALSE)
    set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-dev-${MLW_CODEC_SYSTEM_NAME})
else()
    set(CPACK_STRIP_FILES TRUE)
    set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${MLW_CODEC_SYSTEM_NAME})
endif()
set(CPACK_VERBATIM_VARIABLES TRUE)

# Archive generator setup
set(CPACK_BINARY_TGZ  ON)
set(CPACK_BINARY_STGZ OFF)
set(CPACK_BINARY_TBZ2 OFF)
set(CPACK_BINARY_TXZ  OFF)
set(CPACK_BINARY_TZ   OFF)
set(CPACK_INSTALL_CMAKE_PROJECTS
    "${CMAKE_CURRENT_BINARY_DIR};${CMAKE_PROJECT_NAME};/")

include(CPack)
