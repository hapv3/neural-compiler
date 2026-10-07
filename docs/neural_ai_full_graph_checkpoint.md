# Full YOLO320 / MobileNet224 execution checkpoint

Checkpoint date: 2026-10-07. This is a reproducible **full-command execution**
checkpoint, not a byte-exact correctness release or an all-regressions-green release.

## Repository pair

- Compiler: the commit containing this document; includes the Neural-AI reference
  tools from `43216f69` and the pending staging/scheduling fixes captured here.
- Hardware/firmware/test harness (`neural-ai`):
  `eb8ce962c78d18bb0ae5157b6d58cce1ff6497a4` (execution checkpoint
  `fc623bacb0ab0a6342b2284042526339d59d6a6a`, followed by the requested license copy).
- AXI simulation timing dependency (`neural-ai/hw/axi`):
  `3611c338f44d26def6f9921aa2775a639ae4a1f6` (already pinned by the parent repo).

The checkpoint captures compiler serialization of Spatz after outstanding systolic
jobs, exclusion of spilled/multi-group linebuffer tensors from compact cascade
stripe staging, firmware depthwise requant re-enable, blocking DMA queue readiness,
and full-model tests with incremental per-command PMU CSV output. It does not add
any new synthesizable RTL change.

## Completed simulation evidence

Existing runs completed on 2026-10-06; no new Verilator simulation was started when
creating this checkpoint.

| Model | Completed commands | PMU cycles | Final output vs TFLite BUILTIN_REF |
| --- | ---: | ---: | --- |
| YOLOv8n INT8 320×320 | 3,811 / 3,811 | 16,557,245 | FAIL: output differs |
| MobileNetV2 INT8 224×224 | 12,815 / 12,815 | 17,127,973 | FAIL: output differs |

Both runs reached firmware completion without a command failure or timeout.
Output assertions remain enabled and correctly make the overall tests fail.
MobileNet output CRC32 is `0x434492b9`, versus expected `0x26406c9c`.
Do not describe dispatch completion as an output-data PASS.

Local evidence root in `neural-ai`:
`hw/rtl/cluster/tb/sim/test_full_models_cursor/{yolo320,mobilenet224}/`.
Each directory contains `sim.log`, `results.xml`, `command_pmu.csv`, and frozen
`dma_events.{nai,elf,json,csv}`. PMU CSVs contain all logical command records.
Generated artifacts and logs are local, not committed; preserve these directories
when cleaning simulator build products.

Frozen artifact SHA256:

```text
YOLO NAI:      23d24dc617e6c9fe33de08149c6c9b4af62b3e4d94b94d4a7f7b7d38e1dd5c6a
MobileNet NAI: 5d972100d0ecb86958e29aee5615be56008b57763fc202ff4b0ce63343066744
Firmware ELF: 5003c93460c376b5b8c32c18cf2053168eaf1aa980bbbfc4f947d7f437976035
YOLO TFLite:   2330cae180c11c0d3203883efe63ef677935867754dc6a81d4c88c773436921f
MobileNet:     cefdc91b5270a9deff9139d3c60931178a960d214304f8ea180e385e4183422f
```

## Rebuild and compile

Use sibling `neural-compiler` and `neural-ai` workspaces, initialize the hardware
submodules at their pinned commits, and install the Spatz toolchains and project
Python/cocotb dependencies. TensorFlow 2.21.0 was used for reference comparisons.

From `neural-compiler`:

```bash
cmake -S ethosu/regor -B build-python -DCMAKE_BUILD_TYPE=Release \
  -DREGOR_PYTHON_BINDINGS_DESTINATION="$PWD/ethosu"
cmake --build build-python --target PyRegor -j10
cmake --install build-python --component python-bindings

python3 -c 'import sys; from ethosu.vela.vela import main; raise SystemExit(main(sys.argv[1:]))' \
  --accelerator-config=neural-ai --output-format=nai \
  --output-dir=out/checkpoint/yolo320 test/model/yolov8n_320_int8.tflite
python3 -c 'import sys; from ethosu.vela.vela import main; raise SystemExit(main(sys.argv[1:]))' \
  --accelerator-config=neural-ai --output-format=nai \
  --output-dir=out/checkpoint/mobilenet224 test/model/mobilenet_v2_224_int8.tflite
```

The full-graph tests compile these source models themselves. Do not substitute a
different NAI or reuse a snapshot across model hashes without validation.

At checkpoint creation, both models were recompiled through the rebuilt/installed
Python backend: each resulting NAI was byte-for-byte identical to its frozen
simulation artifact above (3,811 and 12,815 commands). The current firmware ELF
SHA256 also matches the frozen artifact. Compile-only logs and new packages are
under `out/full_graph_checkpoint_20261007/` (local).

From `neural-ai`:

```bash
make -j10 -C sw/test/compiler_runtime check NEURAL_COMPILER_ROOT="$(realpath ../neural-compiler)"
make -j10 -C sw/runtime/neural_ai all
```

The host-check command resolves the compiler root before `make -C` changes directory.
The verified PMU-enabled firmware has `.text` = 26,248 bytes within the 32 KiB ITCM.

## Run either full simulation with per-command PMU

The following commands **start simulation**. Use a fresh output directory; never
overwrite the completed evidence above. Example from the `neural-ai` root:

```bash
export NEURAL_COMPILER_ROOT="$(realpath ../neural-compiler)"
export PYTHONPATH="$PWD/sw/test/compiler_runtime"
export LIBPYTHON_LOC=/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0
export PYGPI_PYTHON_BIN=/usr/bin/python3
export PYTHONDONTWRITEBYTECODE=1
unset NAI_REPLAY_MANIFEST NAI_REPLAY_FIRMWARE_ELF NAI_PMU_COMMAND_FILTER
export YOLO320_TIMEOUT_CYCLES=50000000
export MOBILENET224_TIMEOUT_CYCLES=50000000

# Choose ONE model, then run the command below.
model=yolo320
# model=mobilenet224
checkpoint_run="$PWD/hw/rtl/cluster/tb/sim/checkpoint_$model"
test ! -e "$checkpoint_run" || { echo "Choose a fresh output directory"; exit 1; }
mkdir -p "$checkpoint_run"
export TMPDIR="$checkpoint_run"
export NAI_DMA_TRACE_CSV="$checkpoint_run/dma_events.csv"
export NAI_DMA_TRACE_START_CYCLE=0 NAI_DMA_TRACE_MAX_ROWS=1000
export YOLO320_COMMAND_PMU_CSV="$checkpoint_run/command_pmu.csv"
export MOBILENET224_COMMAND_PMU_CSV="$checkpoint_run/command_pmu.csv"

make -j10 -C hw/rtl/cluster sim VERILATOR_JOBS=10 \
  COCOTB_TEST_MODULES=test_compiled_model \
  COCOTB_TESTCASE="test_compiler_generated_selected_${model}_full_graph" \
  CLUSTER_SIM_NAME=compiled_model SIM_BUILD="$checkpoint_run/build" \
  SIM_LOG="$checkpoint_run/sim.log" \
  COCOTB_RESULTS_FILE="$checkpoint_run/results.xml"
```

Adjust libpython/interpreter paths together for other installations. Build firmware
before starting either run. Run models sequentially, or use independent simulator
build/output directories; do not rebuild a shared executable while another run uses it.
The full tests are expected to fail at their final TFLite output assertion at this
checkpoint. Inspect firmware completion, PMU counts and output assertion separately.

## Verification at checkpoint creation and remaining failures

- Firmware host tests and cross-repository ABI check: PASS (8 host executables).
- Firmware build: PASS; 26,248-byte `.text`.
- Offline snapshot-boundary and byte-exact helper tests: 6 PASS.
- New Spatz/async-linebuffer scheduling regression: PASS (13 assertions).
- Broad compiler filter `Neural-AI*`: **152/156 test cases pass**, 42,132/42,136
  assertions pass. ASan additionally reports 24,940 leaked bytes in 128 allocations.
- Full compiler log: `out/full_graph_checkpoint_20261007/compiler_tests.log` (local).

The four compiler failures are retained, not hidden by changing assertions:

1. `Neural-AI L2 placement excludes operators without L2 staging`: LUT unexpectedly
   qualifies as an L2 spill candidate.
2. `Neural-AI compiler streams a structural RGB stem from its binding`: padding
   seeds equal binding copies (16), violating the optimization expectation `<`.
3. `Neural-AI compiler fuses ReLU and ReLU6 into qparam clamps`: compilation rejects
   a pointwise Conv whose input/output are not C32-blocked.
4. `Neural-AI compiler copies multi-group compact NHWC into depthwise C32 storage`:
   expected two input DMA3D commands for 64 channels, observed zero.

These need separate triage; this checkpoint does not establish whether each is a
stale structural expectation or an implementation defect.

YOLO stem mismatch has been traced to requant rounding and verified against saved
RTL bytes. That diagnosis does not explain every full-graph mismatch or establish
the MobileNet failure's cause. The separate hybrid reference currently replaces
Conv/Depthwise/FC only; it is not yet a full RTL golden runtime. See
[reference runtime results](../tools/neural_ai_tflite_reference/RESULTS.md).

Local-only refactor scripts, planning documents, generated executables, datasets,
simulation outputs and unrelated ignore-file edits are intentionally excluded.
