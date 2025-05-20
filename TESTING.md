<!--
SPDX-FileCopyrightText: Copyright 2020, 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

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

Vela is tested in-house by comparing the bit exact numerical behaviour of the
optimised network against that of the corresponding behaviour of the reference
code.  In addition, a developer [pre-commit](https://pre-commit.com/) performs
pytest unit tests, along with linting and formatting checks.  However, the
Catch2 unit tests need to be run manually.

## TOSA

TOSA networks are compared against the [TOSA Reference Model](https://review.mlplatform.org/admin/repos/tosa/reference_model,general).

## TensorFlow Lite/LiteRT

TensorFlow Lite/LiteRT networks are compared against the TensorFlow Lite/LiteRT
reference kernels (except for the UNIDIRECTIONAL_SEQUENCE_LSTM operator which
is compared against the TensorFlow Lite/LiteRT for Microcontrollers reference
kernel).  The following list indicates which TensorFlow version a particular
Vela version was tested against.  It also indicates which version of the
TensorFlow Lite/LiteRT flatbuffer schema was supported.

* Vela 4.3.0 to current supports TensorFlow 2.18.1
* Vela 4.0.0 to 4.2.0 supports TensorFlow 2.17
* Vela 3.12.0 supports TensorFlow 2.16
* Vela 3.11.0 supports TensorFlow 2.15
* Vela 3.10.0 supports TensorFlow 2.14
* Vela 3.9.0 supports TensorFlow 2.12
* Vela 3.8.0 supports TensorFlow 2.11
* Vela 3.6.0 to 3.7.0 supports TensorFlow 2.10
* Vela 3.5.0 supports TensorFlow 2.9
* Vela 3.4.0 supports TensorFlow 2.8
* Vela 3.3.0 supports TensorFlow 2.7
* Vela 3.1.0 to 3.2.0 supports TensorFlow 2.5
* Vela 2.1.0 to 3.0.0 supports TensorFlow 2.4
* Vela 2.0.0 to 2.0.1 supports TensorFlow 2.3
* Vela 0.1.0 to 1.2.0 supports TensorFlow 2.1

## Python Version Support

The majority of Vela's testing is done using a single version of Python, as
indicated by the non-bracketed version in the list below.  However, some
additional testing is also performed across a range of other versions starting
from a minimum version, indicated in the brackets, and going to the latest
released version at the time of testing.

* Vela 3.10.0 to current supports Python 3.10 (3.9)
* Vela 3.9.0 supports Python 3.10 (3.8)
* Vela 3.8.0 supports Python 3.9 (3.8)
* Vela 3.4.0 to 3.7.0 supports Python 3.7 (3.8)
* Vela 3.3.0 supports Python 3.8 (3.7)
* Vela 0.1.0 to 3.2.0 supports Python 3.6 (3.7)

## Developer Pre-Commit

### Installation

To install the developer pre-commit:
``` bash
pip install -e ".[dev]"
pre-commit install
```

### Running

After installation, all pre-commit stages described below will be automatically
run upon a `git commit` command.

However, the various stages can also be run manually.

To run all of the commit stages on all files:
```bash
$ pre-commit run --all-files
```

Example of how to run a specific check on a specific file:
```bash
$ pre-commit run pylint --files ethosu/vela/vela.py
```

### Linting/Formatting

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

### Unit Tests

There are both Python and C/C++ unit tests. These use the following frameworks:

* pytest (Python)
* Catch2 (C/C++)

To run all pytest unit tests:
```bash
$ pytest
```

Example of how to run a specific pytest unit test:
```bash
$ pytest ethosu/vela/test/test_architecture_allocator.py
```

To run all of the Catch2 unit tests:
```bash
$ cmake -S ethosu/regor -B build-unit-tests -DCMAKE_BUILD_TYPE=Debug -DREGOR_SANITIZE=address
$ cmake --build build-unit-tests -t check
```
