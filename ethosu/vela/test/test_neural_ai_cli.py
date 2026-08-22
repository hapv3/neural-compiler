# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest

from ethosu.vela import architecture_features
from ethosu.vela import stats_writer
from ethosu.vela import vela
from ethosu.vela.npu_performance import PassCycles
from ethosu.vela.tensor import BandwidthDirection
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


def test_neural_ai_dram_peak_bandwidth_uses_clock_scale(tmp_path):
    config = tmp_path / "neural-ai.ini"
    config.write_text("[System_Config.Custom]\nDram_clock_scale=0.25\n")

    arch = architecture_features.NeuralAIArchitectureFeatures(
        vela_config_files=[str(config)],
        system_config="Custom",
        memory_mode="Neural_AI_TCDM",
        arena_cache_size=None,
    )

    assert arch.memory_bandwidths_per_second[MemArea.Dram] == 8 * arch.core_clock


def test_neural_ai_report_aggregates_model_and_l2_as_dram(monkeypatch):
    def access(access_type, read, write):
        return SimpleNamespace(accessType=access_type, bytesRead=read, bytesWritten=write)

    report = SimpleNamespace(
        memories={
            "model": SimpleNamespace(
                totalAccessCycles=11,
                peakUsage=100,
                accesses={"weights": access("weights", 1000, 0)},
            ),
            "l2": SimpleNamespace(
                totalAccessCycles=13,
                peakUsage=200,
                accesses={"featuremap": access("featuremap", 20, 30)},
            ),
        },
        readOnlyPeakUsage=0,
        npuCycles=17,
        totalCycles=29,
        cpuOps=0,
        npuOps=1,
        cascadedOps=0,
        cascades=0,
        originalWeights=1000,
        encodedWeights=1000,
        macCount=1,
    )
    captured = {}

    def capture_metrics(
        _arch, _name, cycles, _macs, bandwidths, _batch_size, memory_used, *args, **kwargs
    ):
        captured["cycles"] = cycles
        captured["bandwidths"] = bandwidths
        captured["memory_used"] = memory_used

    monkeypatch.setattr(stats_writer, "print_performance_metrics_common", capture_metrics)
    monkeypatch.setattr(stats_writer, "write_summary_metrics_csv_common", lambda *args, **kwargs: None)
    arch = architecture_features.create_default_arch(architecture_features.Accelerator.Neural_AI)

    stats_writer.print_regor_performance_metrics(arch, report, "test", "unused.csv", None)

    dram = captured["bandwidths"][MemArea.Dram]
    assert captured["cycles"][PassCycles.DramAccess] == 24
    assert captured["memory_used"][MemArea.Dram] == 300
    assert dram[TensorPurpose.Weights][BandwidthDirection.Read] == 1000
    assert dram[TensorPurpose.FeatureMap][BandwidthDirection.Read] == 20
    assert dram[TensorPurpose.FeatureMap][BandwidthDirection.Write] == 30


def test_neural_ai_cli_requires_nai_output(capsys):
    with pytest.raises(SystemExit):
        vela.main(["--accelerator-config=neural-ai", "missing.tflite"])

    assert "Neural-AI requires --output-format=nai" in capsys.readouterr().err


def test_nai_output_requires_neural_ai_target(capsys):
    with pytest.raises(SystemExit):
        vela.main(["--output-format=nai", "missing.tflite"])

    assert "--output-format=nai requires --accelerator-config=neural-ai" in capsys.readouterr().err
