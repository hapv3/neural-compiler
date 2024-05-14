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

# only include last at the top level
if (NOT "${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    return()
endif()

include(utils)
include(InstallRequiredSystemLibraries)

# Use a PEP-656 compliant package tag
# The default value for this variable is not useful
if (CMAKE_CROSSCOMPILING)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "i386")
        set(REGOR_SYSTEM_NAME "linux-i586")
    else()
        message(FATAL_ERROR "Unknown cross-compile system")
    endif()
else()
    utils_find_python()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_platform())"
        OUTPUT_VARIABLE REGOR_SYSTEM_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

set(CPACK_PACKAGE_NAME ${REGOR_PACKAGE_NAME})

if (NOT CPACK_PROJECT_NAME)
    # Can be overriden
    set(CPACK_PROJECT_NAME ${CMAKE_PROJECT_NAME})
endif()
if (NOT CPACK_PACKAGE_NAME)
    # Can be overriden
    set(CPACK_PACKAGE_NAME ${CPACK_PROJECT_NAME})
endif()
if (NOT CPACK_COMPONENT_NAME)
    # Can be overriden
    set(CPACK_COMPONENT_NAME ${CPACK_PROJECT_NAME})
endif()

set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${REGOR_SYSTEM_NAME})

# Default variables
set(CPACK_PACKAGE_VENDOR "Arm")
set(CPACK_PACKAGE_DESCRIPTION "${${CPACK_PROJECT_NAME}_DESCRIPTION}")
set(CPACK_PACKAGE_VERSION_MAJOR "${${CPACK_PROJECT_NAME}_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${${CPACK_PROJECT_NAME}_VERSION_MINOR}")
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CPACK_STRIP_FILES FALSE)
else()
    set(CPACK_STRIP_FILES TRUE)
endif()
set(CPACK_VERBATIM_VARIABLES TRUE)

# Archive generator setup
set(CPACK_BINARY_TGZ   ON)
set(CPACK_BINARY_STGZ OFF)
set(CPACK_BINARY_TBZ2 OFF)
set(CPACK_BINARY_TXZ  OFF)
set(CPACK_BINARY_TZ   OFF)

# Collect all exported targets
set(CPACK_INSTALL_CMAKE_PROJECTS
    "${CMAKE_CURRENT_BINARY_DIR};${CPACK_PROJECT_NAME};${CPACK_COMPONENT_NAME};/")

# Include CPack last
include(CPack)
