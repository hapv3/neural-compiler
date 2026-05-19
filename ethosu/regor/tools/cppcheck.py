#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2023, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


FILE_SUFFIXES = {".cpp", ".hpp"}

EXCLUDE_FILES = [
    "ethosu/regor/architecture/ethosu85/ethos_u85_interface.hpp",
    "ethosu/regor/architecture/ethosu65/ethos_u65_interface.hpp",
    "ethosu/regor/architecture/ethosu55/ethos_u55_interface.hpp",
    "ethosu/regor/common/ordered_map.hpp",
    "ethosu/regor/tflite/tflite_schema_generated.hpp",
    "ethosu/regor/tosa/tosa_schema_generated.hpp",
]

EXCLUDE_FOLDERS = [
    "ethosu/regor/dependencies",
]

CPPCHECK_ARGS = [
    "cppcheck",
    "-q",
    "--template={file}:{line} - {severity}({id}): {inconclusive:INCONCLUSIVE: }{message}",
    "--std=c++17",
    "--enable=all",
    "--disable=portability",
    "--check-level=exhaustive",
    "--suppress=missingInclude",
    "--suppress=missingIncludeSystem",
    "--suppress=unknownMacro",
    "--suppress=unmatchedSuppression",
    "--suppress=unusedFunction",
    "--suppress=unusedStructMember",
    "--suppress=useStlAlgorithm",
    "--suppress=assertWithSideEffect",
    "--suppress=clarifyCalculation",
    "--suppress=stlFindInsert",
    "--suppress=functionConst",
    "--suppress=syntaxError",
    "--suppress=duplicateBranch",
    "--suppress=unassignedVariable",
    "--suppress=accessMoved",
    "--suppress=useInitializationList",
    "--suppress=constParameter",
    "--suppress=constParameterPointer",
    "--suppress=constParameterReference",
    "--suppress=constVariable",
    "--suppress=constVariablePointer",
    "--suppress=constVariableReference",
    "--suppress=unreadVariable",
    "--suppress=returnTempReference",
    "--suppress=variableScope",
    "--suppress=noExplicitConstructor",
    "--inline-suppr",
    "--error-exitcode=1",
    "--force",
]


def fatal(*args):
    print("Fatal:", *args, file=sys.stderr)
    return 1


def command_output(args):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"command failed: {' '.join(args)}")
    return result.stdout.splitlines()


def is_cpp_file(filename):
    return Path(filename).suffix in FILE_SUFFIXES


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Check all C++ code in the current Git repository, or only modified "
            "and added files when --revision is supplied."
        )
    )
    parser.add_argument("repo", nargs="?", default=".", help="Git repo path")
    parser.add_argument("-r", "--revision", help="Git revision to check the diff against")
    parser.add_argument("-p", "--path", dest="repo_path", help="Git repo path")
    args = parser.parse_args(argv)

    if args.revision is not None and (not args.revision or args.revision.startswith("-")):
        parser.error("Invalid commit refspec")

    if args.repo_path and args.repo != ".":
        parser.error("repo path can be supplied either positionally or with --path, not both")

    args.repo = args.repo_path or args.repo

    result = subprocess.run(
        ["git", "-C", str(args.repo), "rev-parse", "--is-inside-work-tree"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return fatal("Invalid Git repo path")

    try:
        root = Path(command_output(["git", "-C", str(args.repo), "rev-parse", "--show-toplevel"])[0])
        if args.revision:
            files = command_output(["git", "-C", str(root), "diff", "--name-only", "--diff-filter=AMRC", args.revision])
        else:
            files = command_output(["git", "-C", str(root), "ls-files"])

        files_to_check = sorted(
            {
                filename
                for filename in files
                if is_cpp_file(filename)
                and filename not in EXCLUDE_FILES
                and not any(filename.startswith(folder + "/") for folder in EXCLUDE_FOLDERS)
            }
        )
    except RuntimeError as exc:
        return fatal(str(exc))

    if not files_to_check:
        print("No C++ files to check")
        return 0

    print("Files to check:")
    for filename in files_to_check:
        print(f"  {filename}")

    print("Running Cppcheck")

    suppressions_file = None
    cppcheck_args = [
        *CPPCHECK_ARGS,
        f"--relative-paths={root}",
        *[f"--suppress=*:{filename}" for filename in EXCLUDE_FILES],
    ]
    for folder in EXCLUDE_FOLDERS:
        cppcheck_args += [f"-i{folder}", f"--suppress=*:{folder}/*"]

    try:
        if args.revision:
            allowed = set(files_to_check)
            all_cpp_files = sorted(
                filename for filename in command_output(["git", "-C", str(root), "ls-files"]) if is_cpp_file(filename)
            )

            handle = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
            try:
                with handle:
                    for filename in all_cpp_files:
                        if filename not in allowed:
                            handle.write(f"*:{filename}\n")
                suppressions_file = handle.name
            except Exception:
                os.unlink(handle.name)
                raise

            cppcheck_args.append(f"--suppressions-list={suppressions_file}")

        try:
            result = subprocess.run(
                [*cppcheck_args, *files_to_check],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError:
            return fatal("cppcheck executable not found")
    except RuntimeError as exc:
        return fatal(str(exc))
    finally:
        if suppressions_file:
            os.unlink(suppressions_file)

    if result.returncode != 0:
        print(result.stdout, end="")
        return 1

    print("No issues found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
