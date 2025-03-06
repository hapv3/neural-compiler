<!--
SPDX-FileCopyrightText: Copyright 2020-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

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
# Vela Building From Source

This document details how to build Vela manually from its source code.  It also
contains some known issues that can occur when the compiler is built.

### Arm's GitLab Instance

First obtain the source code by either downloading the desired TGZ file from:  
<https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela>

Or by cloning the git repository:

```bash
git clone https://git.gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela.git
```

Once you have the source code, Vela can be installed using the following
command from the root directory of the repository:

```bash
pip3 install .
```

#### Advanced Installation for Developers

If you plan to modify the Vela codebase then it is recommended to install Vela
as an editable package to avoid the need to re-install after every modification.
This is done by adding the `-e` option to the install command like so:

```bash
pip3 install -e .[dev]
```

If you plan to contribute to the Vela project (highly encouraged!) then it is
recommended to install Vela with the development dependencies (see
[Vela Testing](TESTING.md) for more details).

##### Build options for C++ files (ethosu/regor) #####

The C++ part of the the Vela compiler can be configured through the following
environment variables:

| Variable                   | Description                                             |
| :------------------------- | :------------------------------------------------------ |
| CMAKE_BUILD_TYPE           | Control cmake-build-type (Release or Debug)             |
| CMAKE_BUILD_PARALLEL_LEVEL | Control parallel build level                            |
| CMAKE_GENERATOR            | Override the default CMAKE generator                    |
| CMAKE_ARGS                 | Provide additional build-time options (see table below) |

The following build-time options can be provided through CMAKE_ARGS  
```bash
# Example (Linux Bash)
CMAKE_ARGS="-DREGOR_ENABLE_LTO=OFF -DREGOR_ENABLE_WERROR=ON" pip3 install -e ".[dev]"
```
| Option                             | Description                               | Arguments                           |
| :--------------------------------- | :---------------------------------------- | :---------------------------------- |
| REGOR_ENABLE_LTO                   | Enable Link Time Optimization             | ON/OFF                              |
| REGOR_ENABLE_LDGOLD                | Enable Gold linker if available           | ON/OFF                              |
| REGOR_ENABLE_CCACHE                | Enable ccache if available                | ON/OFF                              |
| REGOR_ENABLE_WERROR                | Enable warnings as errors                 | ON/OFF                              |
| REGOR_ENABLE_STD_STATIC            | Link libstdc and libgcc statically        | ON/OFF                              |
| REGOR_ENABLE_COVERAGE              | Enable Coverage build                     | ON/OFF                              |
| REGOR_ENABLE_PROFILING             | Enable timer based runtime profiling      | ON/OFF                              |
| REGOR_ENABLE_ASSERT                | Enable asserts                            | ON/OFF                              |
| REGOR_ENABLE_EXPENSIVE_CHECKS      | Enable expensive STL GLICXX asserts       | ON/OFF                              |
| REGOR_ENABLE_RTTI                  | Enable RTTI (run-time type information)   | ON/OFF                              |
| REGOR_ENABLE_VALGRIND              | Enable Valgrind during check target       | ON/OFF                              |
| REGOR_ENABLE_TESTING               | Enable unit testing                       | ON/OFF                              |
| REGOR_ENABLE_CPPCHECK              | Enable CPPCHECK                           | ON/OFF                              |
| REGOR_SANITIZE                     | Sanitizer setting (forwards to fsanitize) | String                              |
| REGOR_LOG_TRACE_MASK               | Log trace enable mask                     | int (0->7) (See common/logging.hpp) |
| REGOR_PACKAGE_NAME                 | CPack package name                        | String                              |
| REGOR_DEBUG_COMPRESSION            | Debug symbol compression                  | none, zlib, zlib-gnu                |
| REGOR_PYTHON_BINDINGS_DESTINATION  | Python bindings install destination       | String                              |
| REGOR_PYEXT_VERSION                | Python extension version                  | String                              |


## Known Issues

### 1. NumPy C API version change

Once ethos-u-vela is installed, the user might want to install a different NumPy
version that is still within the dependency constraints defined in pyproject.toml.

In some scenarios, doing so might prevent ethos-u-vela from functioning as
expected due to incompatibilities between the installed NumPy C headers used in
the mlw_codec and the current version of NumPy.

**Example scenario:**

In the ethos-u-vela source directory, run:

```bash
virtualenv -p 3.10 venv
. venv/bin/activate
pip install ethos-u-vela
```

Next, install a different NumPy version (e.g. 1.23.0)

```bash
pip install numpy==1.23.0 --force
```

Finally, run ethos-u-vela. You might get an error similar to this:

```
ImportError: NumPy C API version mismatch
(Build-time version: 0x10, Run-time version: 0xe)
This is a known issue most likely caused by a change in the API version in
NumPy after installing ethos-u-vela.
```

#### Solution

In order for ethos-u-vela to work with an older version of NumPy that uses
different C APIs, you will need to install the desired NumPy version first, and
then build ethos-u-vela with that specific NumPy version:

1) Uninstall ethos-u-vela and install the desired version of NumPy
   ```
   pip uninstall ethos-u-vela
   pip install numpy==1.23.0 --force
   ```

2) Install required build dependencies
   ```
   pip install "setuptools_scm[toml]<6" wheel
   ```

3) Install ethos-u-vela without build isolation. Not using build isolation
   ensures that the correct version of NumPy is used when copying the C headers
   in mlw_codec during the build process.
   ```
   pip install ethos-u-vela --no-build-isolation --no-cache-dir
   ```