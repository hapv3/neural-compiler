#!/usr/bin/env python3
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
"""Cppcheck wrapper."""
import json
import os
import subprocess
import sys

# Load compile database
def get_db_cmd(src):
    """Load compile database, looks for src compile options and filters them."""
    fcmd = []
    compile_db = None
    if "CMAKE_BINARY_DIR" in os.environ:
        compile_db = os.path.join(os.environ["CMAKE_BINARY_DIR"], "compile_commands.json")
        if not os.path.exists(compile_db):
            compile_db = None
    if compile_db:
        with open(compile_db) as f:
            cmds = json.load(f)
        for cmd in cmds:
            if cmd["file"] == src:
                cmd = cmd["command"].split()
                i = 0
                while i < len(cmd):
                    if cmd[i].startswith("-std"):
                        fcmd.append("-" + cmd[i])
                    if cmd[i].startswith("-D"):
                        fcmd.append(cmd[i])
                    if cmd[i].startswith("-isystem"):
                        fcmd.append("-I")
                        i += 1
                        fcmd.append(cmd[i])
                        fcmd.append(f"--suppress=*:{cmd[i]}/*")
                        fcmd.append("--suppress=unmatchedSuppression:*")
                    if cmd[i].startswith("-I"):
                        fcmd.append(cmd[i])
                        i += 1
                        fcmd.append(cmd[i])
                    i += 1
                break
    return fcmd


# The source file to scan is the last argument
src = sys.argv[-1]
assert os.path.exists(src), f"File {src} not found"

cmd = []
# CMake options
cmd += sys.argv[1:-1]
# Options from compile database
cmd += get_db_cmd(src)
# The source file last
cmd.append(src)

subprocess.call(cmd)
