# YOLO first-cascade byte-exact investigation — 2026-10-08

## Confirmed simulation result

Pinned model and firmware are unchanged from
`neural_ai_full_graph_checkpoint.md`. The corrected replay harness is neural-ai
commit `d4b74e8`. Evidence is local under
`neural-ai/hw/rtl/cluster/tb/sim/test_yolo_neural_reference_24_75_firmware_restore/`.

The simulator completed 53/53 commands: original command 23 as LUT restoration,
then commands 24–75. Firmware PASS is **not** numerical PASS.

| Check | Bytes compared | Differences |
| --- | ---: | ---: |
| Command 23 restoration versus saved physical destination | 102400 | 0 |
| Command 25 SiLU versus Neural-AI hybrid reference | 51200 | 0 |
| Command 70 Conv versus Neural-AI hybrid reference | 25600 | 25079 |
| Command 75 Conv versus Neural-AI hybrid reference | 25600 | 25079 |

Command 70 maximum absolute error is 159 INT8 levels, not a one-LSB rounding
disagreement. Its saved pre-compute active input differs from the corresponding
reference padded input at 41395/54096 bytes. The first Conv and SiLU correctness
established by the earlier snapshot therefore does not extend to this consumer.

Per-command PMU: 505600 cycles for commands 24–75, matching the previously
recorded sum for this range; restoration command 23 adds 32340 cycles. Whole-run
PMU is 610763 cycles and includes additional firmware/control work. This is a
correctness diagnostic, not a performance improvement claim.

## Independent reproduction without RTL simulation

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/neural_ai_tflite_reference/analyze_staging.py \
  --neural-ai-root /home/dev01/neural-ai \
  --case /home/dev01/neural-ai/hw/rtl/cluster/tb/sim/test_yolo_neural_reference_24_75_firmware_restore/case.json \
  --before-command 70 \
  --report /home/dev01/neural-ai/hw/rtl/cluster/tb/sim/test_yolo_neural_reference_24_75_firmware_restore/staging_analysis.json
```

Use a fresh report path for reruns. The helper verifies pinned artifacts using
the replay harness, executes only supported DMA/LUT commands 24–69, and excludes
the firmware command/scratch window from the comparison. It reproduces **all
520192 allocatable TCDM bytes with zero differences** against the saved Verilator
pre-command-70 dump. Async transfers are applied eagerly: this proves the
observed functional corruption can arise from the command stream, not that all
possible asynchronous schedules are safe.

Command 24 loads a padding seed at TCDM offset 0, where activation bytes are
still live. Subsequent padding writes and command 69's forward strided copy
also overlap the original activation. Command 70 then consumes this corrupted
stage. No RTL modification is justified by this evidence.

## Compiler corrections and remaining work

The command-generator prepass excluded L2-backed cascade inputs from staging
reservation, even though stripe emission still materialized them. A graph input
is L2-backed too. The old global maximum-size check missed a zero-sized
per-operation allocation and allowed offset 0 to be emitted.

Corrections in this increment:

- Reserve single-group cascade input staging regardless of source memory area.
- Reject a stripe exceeding its own reservation, including its padding seed.
- Estimate scheduled cascade stripe geometry/layout, not standalone spill tiles.
- Extend the existing structural RGB→LUT→Conv regression with staging/feature-map
  disjointness checks; retain the original DMA reuse assertions.

The structural regression passes **1292 assertions**, with ASan enabled and leak
detection disabled (known unrelated leak findings are documented in the full
checkpoint). This is not an all-tests-green statement.
The five stripe-staging planner cases also pass (90 assertions), as do four
Python unit tests for the offline ABI replay helper.

At that checkpoint, **full YOLO compilation was not yet restored**. Correct accounting exposed
a later schedule/fast-storage overflow: 552960 tensor bytes plus 4096 workspace
bytes versus 520192 available, at schedule time 2. Do not bypass this guard or
resume the old corrupted model as a correctness run. An exploratory shared
live-range change did not resolve the overflow and was removed.

## Scheduler feedback correction and fresh replay

Instrumentation localized the overflow to the workspace feedback loop, before
`ApplySchedule` or fast-storage allocation. At a reduced limit of 350400 bytes,
the proposal still used 552960 tensor bytes plus 12288 workspace bytes. The old
update took `min(current_limit, physical_limit - workspace)`, leaving the limit
unchanged and incorrectly treating this over-budget plateau as convergence.
The measured tensor peak did not grow during the subsequent allocation phases.

The update now subtracts the observed overflow from the current limit, bounded
by physical capacity minus workspace. The measured case retries at 305344 bytes.
An exhausted positive budget fails explicitly; capacity checks are not bypassed.
The new schedule also needs striped AFU LUT input from L2. It now uses ordinary
DMA/AFU commands with at most 4 KiB per staging buffer, allowing tiles to split
wide spatial rows. Simultaneous input/output staging uses disjoint buffers.
No RTL or firmware change is involved.

Verification:

- Budget-feedback regression: 22 assertions.
- Feedback, stripe-planner and structural RGB regression selection: 7 cases,
  1404 assertions. Separately, 6 LUT cases pass 1957 assertions.
- Replay/snapshot offline helpers: 15 tests.
- Full YOLO320 and MobileNet224 compile successfully with TCDM guards enabled.

New artifacts under `out/{yolo,mobilenet}_scheduler_debug_20261008/`:

| Model | Logical commands (excluding END) | NAI SHA-256 |
| --- | ---: | --- |
| YOLO320 | 4976 | `5158f0d25da621a8606e2d82f83ba0216bb0871a56830b1a46a62e32ce252b0a` |
| MobileNet224 | 12783 | `44c05e1525ce74e2cc641a5f371d9f93d66e0d9b9c7c1931a27e48cefb8f0d78` |

These are correctness-driven schedules, not a demonstrated performance gain.
The changed NAI invalidates old snapshots. The new bounded diagnostic starts
from reset using the unchanged firmware (`5003c93460c376b5b8c32c18cf2053168eaf1aa980bbbfc4f947d7f437976035`).
Prepared case: `neural-ai/hw/rtl/cluster/tb/sim/test_yolo_neural_reference_fresh_20261008/run/case.json`.
It executes commands 1–56 and compares commands 15/25 (35840 active SiLU bytes
each) and 56 (17920 active consumer Conv bytes) against Neural-AI reference.
Numerical simulation confirmation remains pending; compiling successfully or
passing helper tests is not byte-exact PASS.
