# SPDX-FileCopyrightText: Copyright 2020-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-FileCopyrightText: Copyright (c) 2016 The Pybind Development Team, All rights reserved.
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
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Packaging for the Vela compiler."""
import os
import platform
import re
import subprocess
import sys

from setuptools import Extension
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools_scm import get_version


# MakeExtension
class MakeExtension(Extension):
    def __init__(self, name, sources, define_macros=None):
        super().__init__(name, sources, define_macros=define_macros)


mlw_module = MakeExtension(
    "ethosu.mlw_codec",
    ["ethosu/mlw_codec/mlw_encode.c", "ethosu/mlw_codec/mlw_decode.c", "ethosu/mlw_codec/mlw_codecmodule.c"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_9_API_VERSION")],
)


# CMakeExtension. The package_name needs to match the source tree location of the extension
class CMakeExtension(Extension):
    """CMake extension supporting the build process."""

    def __init__(self, package_name):
        """Cmake extension constructor."""
        # name must be the single output extension from the CMake build.
        Extension.__init__(self, package_name, sources=[])
        # Cache the source dir
        paths = package_name.split(".")
        sourcedir = os.path.join(*paths)
        self.sourcedir = os.path.abspath(sourcedir)


ethosu_regor = CMakeExtension("ethosu.regor")


# The actual build extension logic
class MakeOrCMakeBuild(build_ext):
    """CMake build extension main class extending setuptools build_ext."""

    def finalize_options(self):
        build_ext.finalize_options(self)
        import builtins

        # tell numpy it's not in setup anymore
        builtins.__NUMPY_SETUP__ = False
        import numpy as np

        # add the numpy headers to the mlw_codec extension
        self.include_dirs.append(np.get_include())

    def build_extension(self, ext):
        """Override build_extension. Implements the logic to build the module."""
        if isinstance(ext, MakeExtension):
            # make build
            super().build_extension(ext)
        elif isinstance(ext, CMakeExtension):
            import numpy as np

            self.include_dirs.remove(np.get_include())
            self.build_extension_cmake(ext)
        else:
            assert False, f"Unsupported build_extension of type {type(ext)}"

    def build_extension_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Build config
        cfg = os.environ.get("CMAKE_BUILD_TYPE", "Release")

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python3_ROOT_DIR pointing to a specific Python installation if
        # required. Alternatively simply resort to a virtual env to get the
        # Python version right. Constraining CMake's FindPython module to an
        # EXACT Python version is also an option
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DREGOR_PYTHON_BINDINGS_DESTINATION={extdir}",
        ]
        build_args = ["-t", "install-python-bindings"]
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # Pass version
        cmake_args += [f"-DREGOR_PYEXT_VERSION={get_version()}"]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                try:
                    import ninja  # noqa: F401

                    ninja_executable_path = os.path.join(ninja.BIN_DIR, "ninja")
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass
        else:

            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                # Windows platform specifiers to CMake -A arguments
                plat_to_cmake = {
                    "win32": "Win32",
                    "win-amd64": "x64",
                    "win-arm32": "ARM",
                    "win-arm64": "ARM64",
                }
                cmake_args += ["-A", plat_to_cmake[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        # CMake will use the latest python version installed on the system
        # to avoid that, the intended version is specified as an env variable
        # before building the extension
        env = os.environ
        env.update({"PYTHON_VERSION": platform.python_version()})
        cmake_args += [
            "-DPython3_FIND_STRATEGY=LOCATION",
            "-DPython3_FIND_REGISTRY=NEVER",
            "-DPython3_FIND_FRAMEWORK=NEVER",
            f"-DPython3_ROOT_DIR={os.path.dirname(sys.executable)}",
        ]
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp, env=env)


# Read the contents of README.md file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()
    tag = get_version()
    url = f"https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/{tag}/"
    # Find all markdown links that match the format:  [text](link)
    for match, link in re.findall(r"(\[.+?\]\((.+?)\))", long_description):
        # If the link is a file that exists, replace it with the web link to the file instead
        if os.path.exists(os.path.join(this_directory, link)):
            url_link = match.replace(link, url + link)
            long_description = long_description.replace(match, url_link)
    if os.getenv("ETHOSU_VELA_DEBUG"):
        # Verify the contents of the modifications made in a markdown renderer
        with open(os.path.join(this_directory, "PYPI.md"), "wt", encoding="utf-8") as fout:
            fout.write(long_description)


# Can also go in setup.cfg
setup(
    use_scm_version=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=[mlw_module, ethosu_regor],
    cmdclass={"build_ext": MakeOrCMakeBuild},  # type: ignore[dict-item]
)
