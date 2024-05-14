<!--
SPDX-FileCopyrightText: Copyright 2020, 2022-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the License); you may
not use this file except in compliance with the License.
You may obtain a copy of the License at

www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an AS IS BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Vela Testing

## Tools

Vela's Python code is PEP8 compliant with the exception of a 120 character
line length.  The following code formatting and linting tools are run on all the
Python files (excluding some auto-generated code see `.pre-commit-config.yaml`
for details):

* mypy (code linter)
* reorder-python-import (code formatter)
* black (code formatter)
* flake8 (code linter)
* pylint (code linter)

Vela's C/C++ code is formatted using the following tools (excluding some
auto-generated and third-party code see `.pre-commit-config.yaml` for details):

* clang-format (code formatter)

All of the above tools can be installed and run using the
[pre-commit](https://pre-commit.com/) framework (see below).

In addition, there are both Python and C/C++ unit tests. These use the following
frameworks:

* pytest (Python)
* Catch2 (C/C++)

### Installation

To install the development dependencies, first install Vela with the development
option:

``` bash
pip install -e ".[dev]"
```

This will install the following tools:

* pytest
* pytest-cov
* pre-commit
* build
* setuptools_scm

Then, the remaining tools will be installed automatically upon the first use of
pre-commit.

### Add pre-commit hook (Automatically running the tools)

To support code development all the above tools can be configured to run
automatically upon any modified file. This happens when performing a
`git commit` command and is setup using:

```bash
$ pre-commit install
pre-commit installed at .git/hooks/pre-commit
```

When committing a patch, if any of the tools result in a failure (meaning an
issue was found) then the issue will need to be resolved before re-issuing the
commit.

### Manually running the tools

All of the tools can be run individually by invoking them using the following
pre-commit framework commands:

1) Run all of the commit stages on all files in the repository:

```bash
$ pre-commit run --all-files
```

2) Run a specific check on a specific file
```bash
$ pre-commit run pylint --files ethosu/vela/vela.py
```

### Manually run the pytest unit tests

To run all of the pytest unit tests use the following command:
```bash
$ pytest
```

2) Run a specific pytest unit test
```bash
$ pytest pytest ethosu/vela/test/test_architecture_allocator.py
```

### Manually run the Catch2 unit tests

To run all of the Catch2 unit tests use the following command:
```bash
$ cmake -S ethosu/regor -B build-unit-tests -DCMAKE_BUILD_TYPE=Debug -DREGOR_SANITIZE=address
$ cmake --build build-unit-tests -t check
```
