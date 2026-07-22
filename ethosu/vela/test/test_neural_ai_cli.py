# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from ethosu.vela import architecture_features
from ethosu.vela import vela
from ethosu.vela.tensor import MemArea
from ethosu.vela.tensor import TensorPurpose


def test_neural_ai_architecture_features_match_fixed_target():
    arch = architecture_features.create_default_arch(architecture_features.Accelerator.Neural_AI)

    assert arch.accelerator_config == architecture_features.Accelerator.Neural_AI
    assert arch.num_macs_per_cycle == 1024
    assert arch.ncores == 1
    assert arch.arena_cache_size == 512 * 1024 - 4 * 1024
    assert arch.tensor_storage_mem_area[TensorPurpose.Weights] == MemArea.Dram
    assert arch.tensor_storage_mem_area[TensorPurpose.FeatureMap] == MemArea.Sram
    assert os.path.basename(arch.vela_config_files[0]) == "neural-ai.ini"


def test_neural_ai_cli_requires_nai_output(capsys):
    with pytest.raises(SystemExit):
        vela.main(["--accelerator-config=neural-ai", "missing.tflite"])

    assert "Neural-AI requires --output-format=nai" in capsys.readouterr().err


def test_nai_output_requires_neural_ai_target(capsys):
    with pytest.raises(SystemExit):
        vela.main(["--output-format=nai", "missing.tflite"])

    assert "--output-format=nai requires --accelerator-config=neural-ai" in capsys.readouterr().err
