//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/ini_reader.hpp"

#include <catch_all.hpp>

static const std::string config = R"(
[scheduler]
optimize=Performance
empty=
size1=12
size2 = 27kb
size3=  437mb
size4=1099511627776
verbose=true
silent=false
; a comment
int1=39
int2=156
int3=2199023255552
values_may_contain=square[brackets]
but_keys[must]=not
quoted_string="value"
quoted_empty_string=""
quoted_escaped_string="\"escaped\" value"
array=1 ,3, 4,5,6
half=0.5

unexpected_key=skipped_value
expected_key=expected_key_value
[unexpected_section]
skipped_key=skipped_key_value
square_brackets=not[a]new]section[
[expected_section]
key=value
)";

TEST_CASE("ini_reader")
{
    IniReader reader(config.data(), int(config.size()));
    std::string section;
    REQUIRE(reader.Begin(section));
    REQUIRE(section == "scheduler");
    std::string key;
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "optimize");
    std::string value;
    REQUIRE(reader.Read(value));
    REQUIRE(value == "Performance");
    reader.End();
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "empty");
    REQUIRE(reader.Read(value));
    REQUIRE(value == "");
    reader.End();
    int size;
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "size1");
    REQUIRE(reader.Read(size));
    REQUIRE(size == 12);
    std::string suffix;
    REQUIRE(reader.Read(suffix));
    REQUIRE(suffix == "");
    reader.End();
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "size2");
    REQUIRE(reader.Read(size));
    REQUIRE(size == 27);
    REQUIRE(reader.Read(suffix));
    REQUIRE(suffix == "kb");
    reader.End();
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "size3");
    REQUIRE(reader.Read(size));
    REQUIRE(size == 437);
    REQUIRE(reader.Read(suffix));
    REQUIRE(suffix == "mb");
    reader.End();
    int64_t size_64bit;
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "size4");
    REQUIRE(reader.Read(size_64bit));
    REQUIRE(size_64bit == 1099511627776);
    reader.End();
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "verbose");
    REQUIRE(reader.Get<bool>());
    reader.End();
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "silent");
    REQUIRE(!reader.Get<bool>());
    reader.End();
    int intVal;
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "int1");
    REQUIRE(reader.Read(intVal));
    REQUIRE(intVal == 39);
    reader.End();
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "int2");
    REQUIRE(reader.Get<int>() == 156);
    reader.End();
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "int3");
    REQUIRE(reader.Get<int64_t>() == 2199023255552);
    reader.End();
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "values_may_contain");
    REQUIRE(reader.Read(value));
    REQUIRE(value == "square[brackets]");
    reader.End();
    REQUIRE(!reader.Begin(key));  // but_keys[must]=not
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "quoted_string");
    REQUIRE(reader.Read(value));
    REQUIRE(value == "value");
    reader.End();
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "quoted_empty_string");
    REQUIRE(reader.Read(value));
    REQUIRE(value.empty());
    reader.End();
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "quoted_escaped_string");
    REQUIRE(reader.Read(value));
    REQUIRE(value == "\"escaped\" value");
    reader.End();
    REQUIRE(reader.Begin(key));
    auto arr = reader.Get<std::vector<int>>();
    REQUIRE(arr.size() == 5);
    REQUIRE(arr[2] == 4);
    reader.End();
    float floatVal;
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "half");
    REQUIRE(reader.Read(floatVal));
    REQUIRE(floatVal == 0.5f);
    reader.End();

    // Test prematurely ending a key/value pair
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "unexpected_key");
    reader.End();  // End line without reading value. Should skip over 'skipped_value'.
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "expected_key");
    REQUIRE(reader.Read(value));
    REQUIRE(value == "expected_key_value");
    reader.End();  // end key
    reader.End();  // end section

    // Test prematurely ending a section
    REQUIRE(reader.Begin(section));
    REQUIRE(section == "unexpected_section");
    reader.End();  // End section without beginning a key. Should skip over entire section.
    REQUIRE(reader.Begin(section));
    REQUIRE(section == "expected_section");
    REQUIRE(reader.Begin(key));
    REQUIRE(key == "key");
    REQUIRE(reader.Read(value));
    REQUIRE(value == "value");
    reader.End();  // end key
    reader.End();  // end section

    // Test attempting to read beyond end of file
    section = "should_remain_unchanged";
    REQUIRE(!reader.Begin(section));
    REQUIRE(section == "should_remain_unchanged");
}
