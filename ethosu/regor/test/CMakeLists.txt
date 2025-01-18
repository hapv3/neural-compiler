
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

# Add unit test executables here. Having more than one helps multi-threading
# them

include(regor_test)
include(regor_dependencies)

regor_lib(
    NAME
        test_common
    EXCLUDE_FROM_ALL
    TYPE OBJECT
    SOURCES
        test_main.cpp
        util.cpp
    DEPS
        Catch2
        regor::regor-objects
        regor::fmt
        mlw_codec_st
)

add_catch_test(
    NAME
        unit_tests
    SOURCES
        test_mlw_encode.cpp
        test_data_type.cpp
        test_ini_reader.cpp
        test_shape.cpp
        test_tosa_validator.cpp
        test_ethos_u85_weight_encoder.cpp
        test_transpose_type.cpp
        test_reverse_type.cpp
        test_raw_writer.cpp
        test_graph_packing.cpp
        test_ordered_map.cpp
        test_arch_ethos_u85.cpp
        test_hash.cpp
        test_scheduler_decompose.cpp
        test_buffer_view.cpp
        test_scheduler_packing.cpp
        test_operation_utils.cpp
        test_graphir_optimiser.cpp
        test_fast_storage_allocator.cpp
        test_tflite_fb.cpp
    DEPS
        test_common
)
