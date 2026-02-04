# SPDX-FileCopyrightText: Copyright 2021-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
"""Ethos-U-Vela package test."""
import os
import subprocess

import numpy as np


def test_ethos_u_vela_with_vela(tmp_path):
    """Testing ethos-u-vela with vela compiler core."""
    from network import TEST_NETWORK

    output_dir = tmp_path
    output_file = tmp_path / "model_vela.tflite"
    network_file = tmp_path / "model.tflite"
    network_file.write_bytes(TEST_NETWORK)
    cli = [
        "vela",
        "--debug-force-legacy-core",
        "--accelerator-config",
        "ethos-u55-256",
        "--output-dir",
        output_dir,
        network_file,
    ]
    subprocess.check_call(cli)
    output = output_file.read_bytes()
    assert str.encode("COP1") in output
    assert str.encode("ethos-u") in output
    assert str.encode("_split_1_command_stream") in output


def test_ethos_u_vela_with_regor(tmp_path):
    """Testing ethos-u-vela with regor compiler core."""
    from network import TEST_NETWORK

    output_dir = tmp_path
    output_file = tmp_path / "model_vela.tflite"
    network_file = tmp_path / "model.tflite"
    network_file.write_bytes(TEST_NETWORK)
    cur_path = os.path.dirname(os.path.realpath(__file__))
    sc_path = os.path.join(cur_path, "..", "ethosu", "config_files", "Arm", "vela.ini")
    cli = [
        "vela",
        "--accelerator-config",
        "ethos-u85-512",
        "--config",
        sc_path,
        "--system-config",
        "Ethos_U85_SYS_DRAM_Mid",
        "--memory-mode",
        "Dedicated_Sram_384KB",
        "--output-dir",
        output_dir,
        network_file,
    ]
    subprocess.check_call(cli)
    output = output_file.read_bytes()
    assert output[4:8] == str.encode("TFL3")
    assert str.encode("COP1") in output
    assert str.encode("ethos-u") in output
    assert str.encode("ethos_u_command_stream") in output


def test_ethos_u_vela_with_regor_raw_output(tmp_path):
    """Testing ethos-u-vela with regor compiler core and raw output"""
    from network import TEST_NETWORK

    output_dir = tmp_path
    output_file = tmp_path / "model_vela.npz"
    network_file = tmp_path / "model.tflite"
    network_file.write_bytes(TEST_NETWORK)
    cur_path = os.path.dirname(os.path.realpath(__file__))
    sc_path = os.path.join(cur_path, "..", "ethosu", "config_files", "Arm", "vela.ini")
    cli = [
        "vela",
        "--accelerator-config",
        "ethos-u85-256",
        "--config",
        sc_path,
        "--system-config",
        "Ethos_U85_SYS_Flash_High",
        "--memory-mode",
        "Shared_Sram",
        "--output-dir",
        output_dir,
        "--output-format",
        "raw",
        network_file,
    ]
    subprocess.check_call(cli)
    raw = np.load(output_file, allow_pickle=True)
    assert len(raw["cmd_data"].data.tobytes()) > 0
    assert str.encode("COP1") in raw["cmd_data"].data.tobytes()
    assert len(raw["weight_data"].data.tobytes()) > 0
    assert raw["weight_region"] == 0
    assert all(v > 0 for v in raw["scratch_shape"].tolist())
    assert raw["scratch_region"] == 1
    assert raw["scratch_size"] > 0
    assert raw["scratch_fast_shape"].tolist() == [0]
    assert raw["scratch_fast_region"] == 0
    assert raw["scratch_fast_size"] == 0
    assert raw["input_shape"][0].tolist() == [1, 1, 1, 64, 64, 16]
    assert raw["input_elem_size"][0] == 1
    assert raw["input_region"][0] == 1
    assert raw["input_offset"][0] >= 0
    assert raw["output_shape"][0].tolist() == [1, 1, 1, 64, 64, 8]
    assert raw["output_elem_size"][0] == 1
    assert raw["output_region"][0] == 1
    assert raw["output_offset"][0] >= 0
    # Model contains no variables, check that the values are empty arrays.
    assert raw["variable_shape"].size == 0
    assert raw["variable_elem_size"].size == 0
    assert raw["variable_region"].size == 0
    assert raw["variable_offset"].size == 0
    assert "input_quantization" in raw.files
    assert "output_quantization" in raw.files
    assert "variable_quantization" in raw.files

    assert raw["input_quantization"].size == raw["input_shape"].shape[0]
    assert raw["output_quantization"].size == raw["output_shape"].shape[0]
    assert raw["variable_quantization"].size == 0

    input_q0 = raw["input_quantization"][0]
    assert isinstance(input_q0, dict)
    assert input_q0["quant_type"] in ("per_tensor", "per_channel")
    assert "scale" in input_q0
    assert "zero_point" in input_q0

    output_q0 = raw["output_quantization"][0]
    assert isinstance(output_q0, dict)
    assert output_q0["quant_type"] in ("per_tensor", "per_channel")
    assert "scale" in output_q0
    assert "zero_point" in output_q0


def test_ethos_u_vela_with_regor_raw_output_and_cop2(tmp_path):
    """Testing ethos-u-vela with regor compiler core and raw output"""
    from network import TEST_NETWORK

    output_dir = tmp_path
    output_file = tmp_path / "model_vela.npz"
    network_file = tmp_path / "model.tflite"
    network_file.write_bytes(TEST_NETWORK)
    cur_path = os.path.dirname(os.path.realpath(__file__))
    sc_path = os.path.join(cur_path, "..", "ethosu", "config_files", "Arm", "vela.ini")
    cli = [
        "vela",
        "--accelerator-config",
        "ethos-u85-256",
        "--config",
        sc_path,
        "--system-config",
        "Ethos_U85_SYS_Flash_High",
        "--memory-mode",
        "Shared_Sram",
        "--output-dir",
        output_dir,
        "--output-format",
        "raw",
        "--separate-io-regions",
        "--cop-format",
        "COP2",
        network_file,
    ]
    subprocess.check_call(cli)
    raw = np.load(output_file)
    assert len(raw["cmd_data"].data.tobytes()) > 0
    assert str.encode("COP2") in raw["cmd_data"].data.tobytes()
    assert len(raw["weight_data"].data.tobytes()) > 0
    assert raw["weight_region"] == 0
    assert all(v > 0 for v in raw["scratch_shape"].tolist())
    assert raw["scratch_region"] == 1
    assert raw["scratch_size"] > 0
    assert raw["scratch_fast_shape"].tolist() == [0]
    assert raw["scratch_fast_region"] == 0
    assert raw["scratch_fast_size"] == 0
    assert raw["input_shape"][0].tolist() == [1, 1, 1, 64, 64, 16]
    assert raw["input_elem_size"][0] == 1
    assert raw["input_region"][0] == 3
    assert raw["input_offset"][0] == 0
    assert raw["output_shape"][0].tolist() == [1, 1, 1, 64, 64, 8]
    assert raw["output_elem_size"][0] == 1
    assert raw["output_region"][0] == 4
    assert raw["output_offset"][0] == 0
    # Model contains no variables, check that the values are empty arrays.
    assert raw["variable_shape"].size == 0
    assert raw["variable_elem_size"].size == 0
    assert raw["variable_region"].size == 0
    assert raw["variable_offset"].size == 0


def test_regor():
    """Testing regor module."""
    from ethosu import regor

    assert "Regor" in dir(regor)
    assert "compile" in dir(regor)
