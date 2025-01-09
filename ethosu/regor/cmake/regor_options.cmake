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

# No include guard - we want the below variables to be set in any including scope

set(REGOR_DEFAULT_COPTS)
set(REGOR_DEFAULT_DOPTS)
set(REGOR_DEFAULT_LOPTS)

set(REGOR_C_STANDARD 99)
set(REGOR_C_STANDARD_REQUIRED ON)
set(REGOR_C_EXTENSIONS OFF)
set(REGOR_CXX_STANDARD 17)
set(REGOR_CXX_STANDARD_REQUIRED ON)
set(REGOR_CXX_EXTENSIONS OFF)
set(REGOR_POSITION_INDEPENDENT_CODE ON)
set(REGOR_CXX_VISIBILITY_PRESET hidden)
set(REGOR_VISIBILITY_INLINES_HIDDEN ON)

# Remove default debug flags from C/CXX flags, this is controlled by REGOR_ENABLE_ASSERT instead
string(TOUPPER ${CMAKE_BUILD_TYPE} UPPER_CONFIG)
string( REGEX REPLACE "[/-]D[;]*[N|_]DEBUG" "" CMAKE_CXX_FLAGS_${UPPER_CONFIG} "${CMAKE_CXX_FLAGS_${UPPER_CONFIG}}")
string( REGEX REPLACE "[/-]D[;]*[N|_]DEBUG" "" CMAKE_C_FLAGS_${UPPER_CONFIG} "${CMAKE_C_FLAGS_${UPPER_CONFIG}}")

# Check ASSEMBLER/CXX/LINKER flag together with other_flags
# If it checks they all get added to flag_list
function (checked_flag tool flag flag_list)
    set(other_flags ${ARGN})
    # Hash a var name for the cache
    string(REGEX REPLACE "[ -;=]" "_" var_name "${flag} ${other_flags}")

    # Tool option
    if (MSVC)
        if ("${tool}" STREQUAL "LINKER")
            set(tool_opt "/link")
        endif()
    else()
        if ("${tool}" STREQUAL "LINKER")
            set(tool_opt "-Xlinker")
        elseif ("${tool}" STREQUAL "ASSEMBLER")
            set(tool_opt "-Xassembler")
        endif()
    endif()

    if (${var_name}_set)
        set(flag_not_supported "${${var_name}_val}")
    else()
        string(REPLACE " " ";" __flags "${flag}")
        if (other_flags OR tool_opt)
            list(INSERT __flags 0 ${other_flags} ${tool_opt})
        endif()
        if (MSVC)
            list(APPEND __flags "/link")
            list(APPEND __flags "/out:cxx_check.exe")
        else()
            list(APPEND __flags "-o")
            list(APPEND __flags "cxx_check")
        endif()
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cxx_check.cc "int main(void){ return 0; }")
        string(REPLACE " " ";" user_args "${CMAKE_CXX_COMPILER_ARG1}")
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} cxx_check.cc ${user_args} ${__flags}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            OUTPUT_QUIET ERROR_QUIET
            RESULT_VARIABLE flag_not_supported)
        set(${var_name}_val "${flag_not_supported}" CACHE INTERNAL "")
        set(${var_name}_set TRUE CACHE INTERNAL "")
    endif()
    if (flag_not_supported)
        message(STATUS "Looking for ${tool} flag support (${flag}) - Not found")
    else()
        message(STATUS "Looking for ${tool} flag support (${flag}) - Success")
        string(REPLACE " " ";" flag "${tool_opt} ${flag} ${other_flags}")
        set(${flag_list} "${${flag_list}};${flag}" PARENT_SCOPE)
    endif()
endfunction()

function(get_glibc_version var)
    execute_process(COMMAND ldd --version
        OUTPUT_VARIABLE LDD_STR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REPLACE "\n" ";" LDD_LINES ${LDD_STR})
    list(GET LDD_LINES 0 LDD_FIRST_LINE)
    string(REPLACE " " ";" LDD_FIRST_LINE_LIST ${LDD_FIRST_LINE})
    list(GET LDD_FIRST_LINE_LIST -1 GLIBC_VER)
    set(${var} ${GLIBC_VER} PARENT_SCOPE)
    message(STATUS "Looking for GLIBC - Found version ${GLIBC_VER}")
endfunction()

# Base options
if(MSVC)
    # On MSVC, CMake sets /GR by default (enabling RTTI), but we set /GR-
    string(REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    list(APPEND REGOR_DEFAULT_COPTS
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/bigobj>" # C1128: number of sections exceeded object file format limit: compile with /bigobj
        "$<$<AND:$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>,$<CONFIG:Debug>>:$<IF:$<BOOL:${REGOR_ENABLE_RTTI}>,/GR,/GR->>"
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/experimental:external>"
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/external:W0>"
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/external:anglebrackets>"
        "/utf-8"
    )
else()
    list(APPEND REGOR_DEFAULT_COPTS
        "$<IF:$<BOOL:${REGOR_ENABLE_RTTI}>,-frtti,-fno-rtti>"
        "$<$<CONFIG:Debug>:-ggdb>"
        "$<$<BOOL:${REGOR_ENABLE_COVERAGE}>:-g3>"
        "$<$<BOOL:${REGOR_ENABLE_COVERAGE}>:-fprofile-arcs>"
        "$<$<BOOL:${REGOR_ENABLE_COVERAGE}>:-ftest-coverage>"
        "$<$<BOOL:${REGOR_ENABLE_PROFILING}>:-fno-omit-frame-pointer>"
        # https://gitlab.kitware.com/cmake/cmake/-/issues/23136
        "$<$<BOOL:${REGOR_ENABLE_LTO}>:$<IF:$<CXX_COMPILER_ID:GNU>,-ffat-lto-objects,-flto=full>>"
        "$<$<BOOL:${REGOR_SANITIZE}>:-fsanitize=${REGOR_SANITIZE}>"
    )
    if (REGOR_SANITIZE)
        # There's just too much code being added by the sanitizer
        checked_flag(CXX "-fno-var-tracking-assignments" REGOR_DEFAULT_COPTS)
        checked_flag(CXX "-fno-sanitize-recover=all" REGOR_DEFAULT_COPTS)
    endif()
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        # -gz via assembler
        checked_flag(ASSEMBLER "--compress-debug-sections" REGOR_DEFAULT_COPTS "-fnodebug-types-section")
        checked_flag(CXX "-gdwarf-4" REGOR_DEFAULT_COPTS)
    endif()
endif()

# Definitions
list(APPEND REGOR_DEFAULT_DOPTS
    "$<$<NOT:$<BOOL:${REGOR_ENABLE_ASSERT}>>:NDEBUG>"
    "$<$<BOOL:${REGOR_ENABLE_ASSERT}>:_DEBUG>"
    "$<$<BOOL:${REGOR_ENABLE_EXPENSIVE_CHECKS}>:_GLIBCXX_DEBUG>"
    "$<$<BOOL:${REGOR_ENABLE_EXPENSIVE_CHECKS}>:_GLIBCXX_DEBUG_PEDANTIC>"
    "$<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>"
    "$<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>"
)

# Base link flags
if(NOT MSVC)
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        checked_flag(LINKER "--compress-debug-sections=${REGOR_DEBUG_COMPRESSION}" gz_supported "-fdebug-types-section")
        list(APPEND REGOR_DEFAULT_LOPTS ${gz_supported})
        # Add gold only if compressed sections are supported
        checked_flag(LINKER "--compress-debug-sections=${REGOR_DEBUG_COMPRESSION}" gold_gz_supported "-fuse-ld=gold" "-fdebug-types-section")
        if (gz_supported AND NOT gold_gz_supported)
            set(REGOR_ENABLE_LDGOLD OFF CACHE BOOL "Enable Gold linker if available" FORCE)
        endif()
        unset(gz_supported)
        unset(gold_gz_supported)
    endif()
    list(APPEND REGOR_DEFAULT_LOPTS
        "$<$<BOOL:${REGOR_ENABLE_COVERAGE}>:-fprofile-arcs>"
        "$<$<BOOL:${REGOR_ENABLE_LDGOLD}>:-fuse-ld=gold>"
        "$<$<CONFIG:Release>:-s>"
        "$<$<BOOL:${REGOR_ENABLE_STD_STATIC}>:-static-libstdc++>"
        "$<$<BOOL:${REGOR_ENABLE_STD_STATIC}>:-static-libgcc>"
        "$<$<BOOL:${REGOR_SANITIZE}>:-fsanitize=${REGOR_SANITIZE}>"
    )
    if (REGOR_SANITIZE)
        checked_flag(LINKER "-static-libubsan" REGOR_DEFAULT_LOPTS)
    endif()
    list(APPEND REGOR_DEFAULT_LOPTS "-lm")
    if (UNIX AND NOT APPLE)
        get_glibc_version(_GLIBC_VER)
        if (_GLIBC_VER)
            if (${_GLIBC_VER} VERSION_LESS 2.17)
                list(APPEND REGOR_DEFAULT_LOPTS "-lrt")
            endif()
        endif()
    endif()
endif()

# Diagnostics
if(MSVC)
    list(APPEND REGOR_DEFAULT_COPTS
        "/W3"                                                             # Default warning level (severe + significant + production quality).
        "$<$<BOOL:${REGOR_ENABLE_WERROR}>:/WX>"
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/wd4200>"  # "nonstandard extension used : zero-sized array in struct/union"
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/wd4018>"  # "signed/unsigned mismatch in comparison"
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/wd4146>"  # operator applied to unsigned type, result still unsigned
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/wd4244>"  # possible loss of data
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/wd4267>"  # initializing: possible loss of data
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/wd4005>"  # allow: macro redefinition
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/wd4065>"  # allow: switch statement contains 'default' but no 'case' labels
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/wd4141>"  # allow: inline used more than once
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/wd4624>"  # allow: destructor was implicitly defined as deleted
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/wd4146>"  # operator applied to unsigned type, result still unsigned
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/wd4244>"  # possible loss of data
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/wd4267>"  # initializing: possible loss of data
        "$<$<OR:$<COMPILE_LANGUAGE:CXX>,$<COMPILE_LANGUAGE:C>>:/wd5105>"  # allow: macro expansion producing 'defined' has undefined behavior
    )
else()
    list(APPEND REGOR_DEFAULT_COPTS
        "-Wall"
        "-Wextra"
        "$<$<BOOL:${REGOR_ENABLE_WERROR}>:-Werror>"

        "-Wdouble-promotion"
        "-Wshadow"
        "-Wredundant-decls"
        "-Wcast-align"
        "-Wmissing-declarations"
        "-Wmissing-include-dirs"
        "-Wswitch-enum"
        "-Wswitch-default"
        "-Winvalid-pch"
        "-Wformat=2"
        "-Wmissing-format-attribute"
        "-Wformat-nonliteral"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wold-style-cast>"
        "-Wformat-security"
        "-Wimplicit-fallthrough"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>"
        "$<$<NOT:$<CXX_COMPILER_ID:GNU>>:-Woverloaded-virtual>" # Not working in GCC
        "-Wvla"
        "-Wformat-nonliteral"
        "$<$<CXX_COMPILER_ID:GNU>:-Wlogical-op>"

        # Disabled
        "-Wno-switch-enum"                            # TODO : Switch case in TFLite ops handling
        "$<$<CXX_COMPILER_ID:GNU>:-Wno-array-bounds>" # TODO : False positives on Shape operators
        "-Wno-unused-function"
        "-Wno-unused-parameter"
        "-Wno-unused"
        "-Wno-double-promotion"
    )
endif()

function(regor_add_options tgt)
    get_target_property(tp ${tgt} TYPE)
    if ("${tp}" STREQUAL "STATIC_LIBRARY")
        set(__default_scope PRIVATE)
        set(__link FALSE)
    elseif ("${tp}" STREQUAL "OBJECT_LIBRARY")
        set(__default_scope PUBLIC)
        set(__link FALSE)
    elseif ("${tp}" STREQUAL "SHARED_LIBRARY")
        set(__default_scope PRIVATE)
        set(__link TRUE)
    elseif ("${tp}" STREQUAL "MODULE_LIBRARY")
        set(__default_scope PRIVATE)
        set(__link TRUE)
    elseif ("${tp}" STREQUAL "EXECUTABLE")
        set(__default_scope PRIVATE)
        set(__link TRUE)
    else()
        return()
    endif()

    set_target_properties(${tgt} PROPERTIES
        C_STANDARD ${REGOR_C_STANDARD}
        C_STANDARD_REQUIRED ${REGOR_C_STANDARD_REQUIRED}
        C_EXTENSIONS ${REGOR_C_EXTENSIONS}
        CXX_STANDARD ${REGOR_CXX_STANDARD}
        CXX_STANDARD_REQUIRED ${REGOR_CXX_STANDARD_REQUIRED}
        CXX_EXTENSIONS ${REGOR_CXX_EXTENSIONS}
        POSITION_INDEPENDENT_CODE ${REGOR_POSITION_INDEPENDENT_CODE}
        CXX_VISIBILITY_PRESET ${REGOR_CXX_VISIBILITY_PRESET}
        VISIBILITY_INLINES_HIDDEN ${REGOR_VISIBILITY_INLINES_HIDDEN}
        INTERPROCEDURAL_OPTIMIZATION ${REGOR_ENABLE_LTO}
        INTERPROCEDURAL_OPTIMIZATION_${CMAKE_BUILD_TYPE} ${REGOR_ENABLE_LTO})

    target_compile_definitions(${tgt} ${__default_scope} ${REGOR_DEFAULT_DOPTS})
    target_compile_options(${tgt} ${__default_scope} ${REGOR_DEFAULT_COPTS})
    if (__link)
        target_link_options(${tgt} ${__default_scope} ${REGOR_DEFAULT_LOPTS})
    endif()
endfunction()
