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

**Full YOLO compilation is not yet restored.** Correct accounting now exposes
a later schedule/fast-storage overflow: 552960 tensor bytes plus 4096 workspace
bytes versus 520192 available, at schedule time 2. Do not bypass this guard or
resume the old corrupted model as a correctness run. An exploratory shared
live-range change did not resolve the overflow and was removed.

Next: reconcile the selected schedule with the storage actually assigned after
`ApplySchedule`/fast-storage allocation; add a reproducer for that mismatch;
recompile full YOLO and MobileNet. Only after compile-time memory checks pass,
run a fresh initial-cascade simulation against Neural-AI reference. A changed
NAI invalidates the existing command-23/75 snapshots; retain them only as old-model
diagnostic evidence. No new simulation was started during this investigation.
