#
# SPDX-FileCopyrightText: Copyright 2021, 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

# No inclusion guard. We need these set and included in every scope

set(ENABLE_COVERAGE ${REGOR_ENABLE_COVERAGE} CACHE INTERNAL
    "Enable coverage passed to codecov package")

include(regor_lib)
include(utils)
include(CTest)
include(Catch)

find_package(codecov)

if (NOT TARGET check)
    # Target to build and run all unit-tests from the top level
    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} -L unit_test --output-on-failure $<$<BOOL:${REGOR_SANITIZE}>:--verbose>
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# Coverage target
if(REGOR_ENABLE_COVERAGE AND NOT TARGET coverage)
    add_custom_target(coverage)
    if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
        set(CHECK_JOB_CNT --parallel $ENV{CMAKE_BUILD_PARALLEL_LEVEL})
    endif()
    add_custom_command(TARGET coverage POST_BUILD
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} -t check ${CHECK_JOB_CNT}
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} -t lcov-capture
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} -t lcov-genhtml
    )
endif()

function(add_catch_test)
    cmake_parse_arguments(
        _FARG
        ""
        "NAME"
        "SOURCES;DEPS;COPTS;DEFINES;INC_DIRS"
        ${ARGN}
    )
    ### Adds an executable unit test and ammends it to the unit tests top target
    regor_exe(NAME ${_FARG_NAME}
        SOURCES ${_FARG_SOURCES}
        COPTS ${_FARG_COPTS}
        DEFINES ${_FARG_DEFINES}
        INC_DIRS ${_FARG_INC_DIRS}
    )
    # Links the individual tests to the build all target
    add_dependencies(check ${_FARG_NAME})
    # Tests require introspection
    foreach(_dep ${_FARG_DEPS})
        if (NOT TARGET ${_dep})
            continue()
        endif()
        get_target_property(__incs ${_dep} INCLUDE_DIRECTORIES)
        if (__incs)
            target_include_directories(${_FARG_NAME} PRIVATE ${__incs})
        endif()
    endforeach()
    # Link deps
    target_link_libraries(${_FARG_NAME} PRIVATE ${_FARG_DEPS})
    # Now finally enable catch
    target_link_libraries(${_FARG_NAME} PRIVATE regor::Catch2)
    # Valgrind support
    if (REGOR_ENABLE_VALGRIND)
        find_program(VALGRIND_EXECUTABLE valgrind REQUIRED)
        # We hijack CROSSCOMPILING_EMULATOR to get catch_ctest
        # to prepend the valgrind command to all executables
        set_property(TARGET ${_FARG_NAME} PROPERTY
            CROSSCOMPILING_EMULATOR ${VALGRIND_EXECUTABLE} $ENV{VALGRIND_OPTIONS})
    endif()
    catch_discover_tests(${_FARG_NAME} PROPERTIES LABELS unit_test)
    # Coverage
    add_coverage(${_FARG_NAME})
endfunction()

function(add_py_test)
    cmake_parse_arguments(
        _FARG
        ""
        "NAME"
        "SOURCES;DEPS"
        ${ARGN}
    )
    list(TRANSFORM _FARG_SOURCES PREPEND ${CMAKE_CURRENT_LIST_DIR}/)
    add_test(NAME ${_FARG_NAME}
        COMMAND ${Python3_EXECUTABLE} -m pytest ${_FARG_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    foreach(_dep ${_FARG_DEPS})
        list(APPEND _INC_DIRS $<TARGET_FILE_DIR:${_dep}>)
    endforeach()
    if(MSVC)
        set(_path_sep ";")
    else()
        set(_path_sep ":")
    endif()
    string(REPLACE ";" "${_path_sep}" _INC_DIRS "${_INC_DIRS}")
    set_tests_properties(${_FARG_NAME}
        PROPERTIES
            ENVIRONMENT "PYTHONPATH=${_INC_DIRS}${_path_sep}$ENV{PYTHONPATH}"
            LABELS unit_test
    )
endfunction()
