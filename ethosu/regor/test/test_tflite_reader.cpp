//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the License); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "tflite/tflite_reader.hpp"
#include "tflite/tflite_schema_generated.hpp"

#include <catch_all.hpp>

using namespace regor;

TEST_CASE("TFLite reader ignore_ops parses exact builtin operator names")
{
    const auto ops = TfLiteReader::ParseIgnoreOpList("ADD,CONV_2D, ARGMAX ");

    REQUIRE(ops.count(tflite::BuiltinOperator::ADD) == 1);
    REQUIRE(ops.count(tflite::BuiltinOperator::CONV_2D) == 1);
    REQUIRE(ops.count(tflite::BuiltinOperator::ARGMAX) == 1);
    REQUIRE(ops.count(tflite::BuiltinOperator::RELU) == 0);

    const auto addN = TfLiteReader::ParseIgnoreOpList("ADD_N");
    REQUIRE(addN.count(tflite::BuiltinOperator::ADD_N) == 1);
    REQUIRE(addN.count(tflite::BuiltinOperator::ADD) == 0);
    REQUIRE(TfLiteReader::ParseIgnoreOpList("").empty());
    REQUIRE(TfLiteReader::ParseIgnoreOpList(" ").empty());
}

TEST_CASE("TFLite reader ignore_ops rejects unmatched or malformed names")
{
    REQUIRE_THROWS_WITH(TfLiteReader::ParseIgnoreOpList("ADD,NOT_A_TFLITE_OP"), "Unrecognised ignore_ops value 'NOT_A_TFLITE_OP'");
    REQUIRE_THROWS_WITH(TfLiteReader::ParseIgnoreOpList("ADD,,RELU"), "Unrecognised ignore_ops value 'ADD,,RELU'");
    REQUIRE_THROWS_WITH(TfLiteReader::ParseIgnoreOpList("ARG_MAX"), "Unrecognised ignore_ops value 'ARG_MAX'");
    REQUIRE_THROWS_WITH(TfLiteReader::ParseIgnoreOpList("add"), "Unrecognised ignore_ops value 'add'");
    REQUIRE_THROWS_WITH(TfLiteReader::ParseIgnoreOpList("ADD;RELU"), "Unrecognised ignore_ops value 'ADD;RELU'");
}
