#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

# Script to download and prune the required thirdparty software.
# The script should be run from the directory in which it resides. It uses
# git clone to fetch the required software before copying just the files that
# required along with all of the associated license headers and attribution,
# which is assumed to reside in the files in the root directory.

set -u

cwd=$PWD
download_dir=$(mktemp -d)

# clean up function
cleanup() {
    echo "Cleaning up..."
    rm -rf ${download_dir}
}

# handle control-c
trap cleanup SIGINT

# Add THIRDPARTY.md update text
read -r -d '' thirdparty_md_text <<'__HERE_DOC__'
# Third Party Software

The following lists the Third Party software versions and licenses used by Vela:
__HERE_DOC__


# FlatBuffers
version=v24.12.23
clone_url=https://github.com/google/flatbuffers.git
output_dir=${cwd}/flatbuffers
thirdparty_md_text+="\n* FlatBuffers [${version}](https://github.com/google/flatbuffers/releases/tag/${version}) - \n"
thirdparty_md_text+="([Apache-2.0 License](https://github.com/google/flatbuffers/blob/${version}/LICENSE))\n"
git rm -rf ${output_dir}
rm -rf ${download_dir} ${output_dir}
mkdir ${output_dir}
git clone --depth=1 --branch ${version} ${clone_url} ${download_dir}
cp ${download_dir}/* ${output_dir}
cp -r ${download_dir}/include ${output_dir}
git add ${output_dir}

# fmt
version=11.1.1
clone_url=https://github.com/fmtlib/fmt.git
output_dir=${cwd}/fmt
thirdparty_md_text+="\n* fmt [${version}](https://github.com/fmtlib/fmt/releases/tag/${version}) - \n"
thirdparty_md_text+="([MIT License](https://github.com/fmtlib/fmt/blob/${version}/LICENSE))\n"
git rm -rf ${output_dir}
rm -rf ${download_dir} ${output_dir}
mkdir ${output_dir}
git clone --depth=1 --branch ${version} ${clone_url} ${download_dir}
cp ${download_dir}/* ${output_dir}
cp -r ${download_dir}/include ${output_dir}
cp -r ${download_dir}/src ${output_dir}
git add ${output_dir}

# Catch2
version=v3.8.0
clone_url=https://github.com/catchorg/Catch2.git
output_dir=${cwd}/Catch2
thirdparty_md_text+="\n* Catch2 [${version}](https://github.com/catchorg/Catch2/releases/tag/${version}) - \n"
thirdparty_md_text+="([BSL-1.0 License](https://github.com/catchorg/Catch2/blob/${version}/LICENSE.txt))\n"
git rm -rf ${output_dir}
rm -rf ${download_dir} ${output_dir}
mkdir ${output_dir}
git clone --depth=1 --branch ${version} ${clone_url} ${download_dir}
cp ${download_dir}/* ${output_dir}
cp -r ${download_dir}/CMake ${output_dir}
cp -r ${download_dir}/extras ${output_dir}
cp -r ${download_dir}/fuzzing ${output_dir}
cp -r ${download_dir}/src ${output_dir}
cp -r ${download_dir}/third_party ${output_dir}
cp -r ${download_dir}/tools ${output_dir}
git add ${output_dir}

# Gemmlowp
version=09d81e02ab15b41405caebeb5eb63fd12555aee3
clone_url=https://github.com/google/gemmlowp.git
output_dir=${cwd}/gemmlowp
thirdparty_md_text+="\n* Gemmlowp [${version}](https://github.com/google/gemmlowp/tree/${version}) - \n"
thirdparty_md_text+="([Apache-2.0 License](https://github.com/google/gemmlowp/blob/${version}/LICENSE))\n"
git rm -rf ${output_dir}
rm -rf ${download_dir} ${output_dir}
mkdir ${output_dir}
git clone ${clone_url} ${download_dir}
cd ${download_dir}
git checkout ${version}
cd ${cwd}
cp ${download_dir}/* ${output_dir}
cp -r ${download_dir}/fixedpoint ${output_dir}
cp -r ${download_dir}/internal ${output_dir}
git add ${output_dir}

# pybind11
version=v2.13.6
clone_url=https://github.com/pybind/pybind11.git
output_dir=${cwd}/pybind11
thirdparty_md_text+="\n* pybind11 [${version}](https://github.com/pybind/pybind11/releases/tag/${version}) - \n"
thirdparty_md_text+="([BSD-3-Clause License](https://github.com/pybind/pybind11/blob/${version}/LICENSE))\n"
git rm -rf ${output_dir}
rm -rf ${download_dir} ${output_dir}
mkdir ${output_dir}
git clone --depth=1 --branch ${version} ${clone_url} ${download_dir}
cp ${download_dir}/* ${output_dir}
cp -r ${download_dir}/include ${output_dir}
cp -r ${download_dir}/pybind11 ${output_dir}
cp -r ${download_dir}/tools ${output_dir}
git add ${output_dir}

# clean up
cleanup
trap - SIGINT

# report the update text for THIRDPARTY.md
echo "Update and check THIRDPARTY.md with:"
echo "$thirdparty_md_text"