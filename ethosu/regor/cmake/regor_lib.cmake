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

include(GNUInstallDirs)
include(utils)
include(regor_options)
include(CMakePackageConfigHelpers)


# This function installs a target in a component with a namespace and produces a CMake export
# which links both the namespace and the component.
# An optional include sub-path can be specified for public headers under the component include
# path. This can be useful for "sub-component"
function(regor_install)
    cmake_parse_arguments(_
        ""
        "COMPONENT;NAMESPACE;TARGET;INCLUDE"
        ""
        ${ARGN})

    if (NOT __COMPONENT)
        return()
    endif()

    install(TARGETS ${__TARGET}
        EXPORT ${__COMPONENT}-targets
        LIBRARY       DESTINATION "${CMAKE_INSTALL_LIBDIR}"  COMPONENT ${__COMPONENT}
        ARCHIVE       DESTINATION "${CMAKE_INSTALL_LIBDIR}"  COMPONENT ${__COMPONENT}
        RUNTIME       DESTINATION "${CMAKE_INSTALL_BINDIR}"  COMPONENT ${__COMPONENT}
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${__COMPONENT}/${__INCLUDE}" COMPONENT ${__COMPONENT}
    )
    get_target_property(tp ${__TARGET} TYPE)
    if (MSVC AND "${tp}" STREQUAL "STATIC_LIBRARY" AND NOT CMAKE_VERSION VERSION_LESS 3.15)
        install(FILES
            "$<TARGET_FILE_DIR:${__TARGET}>/$<TARGET_FILE_PREFIX:${__TARGET}>$<TARGET_FILE_BASE_NAME:${__TARGET}>.pdb"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT ${__COMPONENT}
            OPTIONAL)
    endif()
    # Emit export
    if (NOT TARGET install-${__COMPONENT})
        install(EXPORT ${__COMPONENT}-targets
            FILE ${__COMPONENT}-targets.cmake
            NAMESPACE ${__NAMESPACE}::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${__COMPONENT}"
            COMPONENT ${__COMPONENT})

        # Convenience target
        add_custom_target(install-${__COMPONENT}
            COMMAND
                "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=${__COMPONENT}
                    -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_install.cmake")

        # Package definition.
        # These calls are wrappers around configure_file producing standard
        # files to be employed by client code using find_package(${__COMPONENT})
        # They produce a relocatable package
        set(PACKAGE_NAME ${__COMPONENT})
        configure_package_config_file(${REGOR_SOURCE_DIR}/cmake/pkg-config.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/${__COMPONENT}-config.cmake
            INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${__COMPONENT}"
            PATH_VARS PACKAGE_NAME)
        unset(PACKAGE_NAME)
        write_basic_package_version_file(
            ${CMAKE_CURRENT_BINARY_DIR}/${__COMPONENT}-config-version.cmake
            VERSION "${${PROJECT_NAME}_VERSION}"
            COMPATIBILITY AnyNewerVersion
        )

        # Install for the produced configure_files above
        install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/${__COMPONENT}-config.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/${__COMPONENT}-config-version.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${__COMPONENT}"
            COMPONENT ${__COMPONENT})

        # This produces a non-relocatable "quick" package that can be directly
        # consumed after a build
        export(EXPORT ${__COMPONENT}-targets
            FILE ${CMAKE_CURRENT_BINARY_DIR}/${__COMPONENT}-targets.cmake)
    endif()
    add_dependencies(install-${__COMPONENT} ${__TARGET})
endfunction()

# This function implements an add_library wrapper for non-interface libraries.
# It makes sure:
# - flags are properly set
# - Install exports is done when the COMPONENT option is present
#
# Note all settings have a default scope pinned to PRIVATE except for OBJECT
# libraries and PUBLIC_HEADERS. Extra settings outside the scope of this function
# can be done to the target following the call
function(regor_lib)
    cmake_parse_arguments(_
        "EXCLUDE_FROM_ALL"
        "NAME;COMPONENT;TYPE;OUTPUT_NAME;INSTALL_LOCATION"
        "PUBLIC_HEADERS;SOURCES;COPTS;DEFINES;LOPTS;DEPS;INC_DIRS"
        ${ARGN})

    if ("${__TYPE}" STREQUAL "STATIC")
        set(__default_scope PRIVATE)
        set(__is_dll FALSE)
    elseif ("${__TYPE}" STREQUAL "SHARED")
        set(__default_scope PRIVATE)
        set(__is_dll TRUE)
    elseif ("${__TYPE}" STREQUAL "OBJECT")
        set(__default_scope PUBLIC)
        set(__is_dll FALSE)
    elseif ("${__TYPE}" STREQUAL "PY_MODULE")
        set(__default_scope PRIVATE)
        set(__is_dll TRUE)
    else()
        message(FATAL_ERROR "Unexpected lib type ${__TYPE}")
    endif()

    if ("${__TYPE}" STREQUAL "PY_MODULE")
        pybind11_add_module(${__NAME} ${__SOURCES})
    elseif (__EXCLUDE_FROM_ALL)
        add_library(${__NAME} ${__TYPE} EXCLUDE_FROM_ALL ${__SOURCES})
    else()
        add_library(${__NAME} ${__TYPE} ${__SOURCES})
    endif()
    if (__OUTPUT_NAME)
        if (MSVC AND "${__TYPE}" STREQUAL "SHARED")
            # MSVC generates static artifacts for DLLs
            set_target_properties(${__NAME} PROPERTIES RUNTIME_OUTPUT_NAME
                ${__OUTPUT_NAME})
            set_target_properties(${__NAME} PROPERTIES ARCHIVE_OUTPUT_NAME
                ${__OUTPUT_NAME}_st)
        else()
            set_target_properties(${__NAME} PROPERTIES OUTPUT_NAME
                ${__OUTPUT_NAME})
        endif()
    endif()

    foreach (dep IN LISTS __DEPS)
        # Workaround for https://gitlab.kitware.com/cmake/cmake/-/issues/15415
        # Recurse deps
        set(to_visit ${dep})
        set(total_deps ${dep})
        while (to_visit)
            set(_to_visit)
            foreach (tgt IN LISTS to_visit)
                # Imported libs and non-targets are linked as usual
                set(no_target FALSE)
                set(imported FALSE)
                if (TARGET ${tgt})
                    get_target_property(imported ${tgt} IMPORTED)
                    get_target_property(tp ${tgt} TYPE)
                else()
                    set(no_target TRUE)
                endif()
                if (no_target OR imported OR (__is_dll AND NOT "${tp}" STREQUAL "OBJECT_LIBRARY"))
                    target_link_libraries(${__NAME} ${__default_scope} ${tgt})
                    list(REMOVE_ITEM total_deps ${tgt})
                    continue()
                endif()

                get_target_property(ideps ${tgt} INTERFACE_LINK_LIBRARIES)
                if (NOT ideps)
                    continue()
                endif()

                foreach(idep IN LISTS ideps)
                    # Strip LINK_ONLY
                    string(REGEX REPLACE "^\\$<LINK_ONLY:([A-Za-z0-9_]+)>$" "\\1" ridep "${idep}")
                    # Collect this target
                    list(APPEND total_deps ${ridep})
                    list(APPEND _to_visit ${ridep})
                endforeach()
            endforeach()
            list(REMOVE_DUPLICATES _to_visit)
            set(to_visit ${_to_visit})
        endwhile()
        list(REMOVE_DUPLICATES total_deps)

        # Now "link" collected target dependencies
        foreach (tgt IN LISTS total_deps)
            get_target_property(tp ${tgt} TYPE)
            if (NOT "${tp}" STREQUAL "INTERFACE_LIBRARY")
                target_sources(${__NAME} ${__default_scope} $<TARGET_OBJECTS:${tgt}>)
            endif()
            # Hand-link interface
            target_link_options(${__NAME} ${__default_scope} $<TARGET_PROPERTY:${tgt},INTERFACE_LINK_OPTIONS>)
            target_include_directories(${__NAME} ${__default_scope} $<TARGET_PROPERTY:${tgt},INTERFACE_INCLUDE_DIRECTORIES>)
            target_include_directories(${__NAME} SYSTEM ${__default_scope} $<TARGET_PROPERTY:${tgt},INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>)
            target_compile_options(${__NAME} ${__default_scope} $<TARGET_PROPERTY:${tgt},INTERFACE_COMPILE_OPTIONS>)
            target_compile_definitions(${__NAME} ${__default_scope} $<TARGET_PROPERTY:${tgt},INTERFACE_COMPILE_DEFINITIONS>)
            target_sources(${__NAME} ${__default_scope} $<TARGET_PROPERTY:${tgt},INTERFACE_SOURCES>)
        endforeach()
    endforeach()

    regor_add_options(${__NAME})
    if (__INC_DIRS)
        target_include_directories(${__NAME} ${__default_scope} ${__INC_DIRS})
    endif()
    if (__DEFINES)
        target_compile_definitions(${__NAME} ${__default_scope} ${__DEFINES})
    endif()
    if (__COPTS)
        target_compile_options(${__NAME} ${__default_scope} ${__COPTS})
    endif()
    if (__is_dll AND __LOPTS)
        target_link_options(${__NAME} ${__default_scope} ${__LOPTS})
    endif()

    if (NOT "${__TYPE}" STREQUAL "PY_MODULE")
        set_target_properties(${__NAME} PROPERTIES
            VERSION ${${PROJECT_NAME}_VERSION})

        if (__PUBLIC_HEADERS)
            set_target_properties(${__NAME} PROPERTIES
                PUBLIC_HEADER "${__PUBLIC_HEADERS}")
        endif()
    endif()
    foreach (dir IN LISTS __PUBLIC_HEADERS)
        get_filename_component(ahdr ${dir} ABSOLUTE)
        get_filename_component(dir ${ahdr} DIRECTORY)
        target_include_directories(${__NAME}
            INTERFACE
                "$<BUILD_INTERFACE:${dir}>"
                "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>")
    endforeach()

    set(__NAMESPACE ${PROJECT_NAME})
    string(TOLOWER ${__NAMESPACE} __NAMESPACE)
    if (__INSTALL_LOCATION AND __COMPONENT)
        install(TARGETS ${__NAME}
            RUNTIME DESTINATION "${__INSTALL_LOCATION}" COMPONENT "${__COMPONENT}"
            LIBRARY DESTINATION "${__INSTALL_LOCATION}" COMPONENT "${__COMPONENT}"
            ARCHIVE DESTINATION "${__INSTALL_LOCATION}" COMPONENT "${__COMPONENT}"
        )
        if (NOT TARGET install-${__COMPONENT})
            add_custom_target(install-${__COMPONENT}
            COMMAND
                "${CMAKE_COMMAND}"
                -DCMAKE_INSTALL_COMPONENT=${__COMPONENT}
                    -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
            )
        endif()
        add_dependencies(install-${__COMPONENT} ${__NAME})
    elseif (NOT "${__TYPE}" STREQUAL "OBJECT")
        regor_install(
            NAMESPACE ${__NAMESPACE}
            COMPONENT ${__COMPONENT}
            TARGET ${__NAME})
    endif()
    add_library(${__NAMESPACE}::${__NAME} ALIAS ${__NAME})
endfunction()

# Same version of the above function for executables
function(regor_exe)
    cmake_parse_arguments(_
        "EXCLUDE_FROM_ALL"
        "NAME;COMPONENT;OUTPUT_NAME"
        "SOURCES;COPTS;DEFINES;LOPTS;DEPS;INC_DIRS"
        ${ARGN})

    set(__default_scope PRIVATE)

    if (__EXCLUDE_FROM_ALL)
        add_executable(${__NAME} EXCLUDE_FROM_ALL ${__SOURCES})
    else()
        add_executable(${__NAME} ${__SOURCES})
    endif()
    if (__OUTPUT_NAME)
        set_target_properties(${__NAME} PROPERTIES OUTPUT_NAME
            ${__OUTPUT_NAME})
    endif()
    regor_add_options(${__NAME})
    if (__INC_DIRS)
        target_include_directories(${__NAME} ${__default_scope} ${__INC_DIRS})
    endif()
    if (__DEFINES)
        target_compile_definitions(${__NAME} ${__default_scope} ${__DEFINES})
    endif()
    if (__COPTS)
        target_compile_options(${__NAME} ${__default_scope} ${__COPTS})
    endif()
    if (__LOPTS)
        target_link_options(${__NAME} ${__default_scope} ${__LOPTS})
    endif()
    if (__DEPS)
        target_link_libraries(${__NAME} ${__default_scope} ${__DEPS})
    endif()
    set_target_properties(${__NAME} PROPERTIES
        VERSION ${${PROJECT_NAME}_VERSION})

    set(__NAMESPACE ${PROJECT_NAME})
    string(TOLOWER ${__NAMESPACE} __NAMESPACE)
    regor_install(
        NAMESPACE ${__NAMESPACE}
        COMPONENT ${__COMPONENT}
        TARGET ${__NAME})
endfunction()
