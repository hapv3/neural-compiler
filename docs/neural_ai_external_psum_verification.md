# External PSUM accumulation verification

## Failure and cause

The frozen YOLO320 replay from 2026-10-08 completed commands 1–56 without a
firmware error, but command 56 differed from the Neural-AI reference in
17,873 of 17,920 output bytes. The preceding two SiLU checkpoints each matched
35,840 bytes. Command completion alone therefore did not establish correctness.

The failing Conv uses a 15×161 C32 input with 16 active channels, a 3×3
stride-2 kernel, nine automatic K tiles, and an output tile with M=560.
The internal PSUM buffer holds only 256 rows. Larger automatic multi-K jobs
need an external INT32 PSUM workspace: M×32×4 = 71,680 bytes for this job.

The drain wrote intermediate INT32 results to the OFM base with the INT8 OFM
row stride, while subsequent tiles read the unchanged PSUM address. Replaying
those incorrect memory transactions reproduced the entire final 520,192-byte
allocatable TCDM snapshot. Computing all nine contributions from the saved
input, weights and ABI qparams instead matched every expected output byte.

The compiler also reserved too few PSUM rows. In the old artifact, the next
input staging allocation started only 30,720 bytes after PSUM, inside the
required 71,680-byte region. Correcting RTL alone would introduce a real
overlap with that input staging allocation.

Frozen NAI SHA256:
`5158f0d25da621a8606e2d82f83ba0216bb0871a56830b1a46a62e32ce252b0a`.
Its snapshots must not be reused with a newly compiled NAI.

## Implementation

- RTL selects external PSUM and its INT32 row stride for intermediate
  automatic K tiles above the internal capacity. The final tile selects OFM.
  Job start and tile advance account for the K index changing on the same edge.
- The planner computes external PSUM spans, including row-stride gaps.
  Command generation checks each job against its operation's reserved PSUM.
- Workspace reservation includes up to 1024 rows for direct single-group
  linebuffer jobs; spilled/multi-group paths retain their existing tile limits.
- Command memory accesses include the automatic PSUM read/write hazards so
  overlap scheduling cannot treat this workspace as unused.
- Standalone padded-input reservation follows the actual whole-input emitter,
  and staging precomputation uses the connection slice's channel count. Padding
  copies also include the selected C32 group's source offset.

No MAC array, memory port, or command ABI expansion is required.

## Verified unit and block gates

Hardware commit: `66aee9e` (`Fix external PSUM routing for automatic multi-K jobs`).
On 2026-10-12, all five sequential Verilator gates passed: output drain
(4 tests), external PSUM controller (1 test with five geometries), execution
modes (4 tests), controller (8 tests), and linebuffer matrix (8 tests).

The external PSUM test covers M=255/C16, M=256/C32, M=257/C16,
M=560/C16 and M=560/C32. Consecutive jobs run without reset. Each checks all
512 KiB of TCDM against an independent numerical reference, including dirty
PSUM, unequal OFM/PSUM row strides, stride gaps, inputs and weights. The drain
unit test additionally checks stalled OBI grants and a preceding job's K index.

Local gate evidence:
`neural-ai/hw/rtl/systolic/tb/sim/psum_external_20261012/`.
The gate runner is local only. Firmware host/ABI checks and the nine replay
helper tests also passed. Compiler ASan regressions passed 4,872 assertions
across 21 tests, including padded high-C32 slices, external PSUM hazards,
standalone reservation, cascades, pointwise spill and K3 spill paths.

Both full models compile with zero CPU operators. YOLO320 emits 5,008 logical
commands and requires 512,000 TCDM bytes; MobileNet224 emits 12,789 logical
commands and requires 504,736 bytes. Counts exclude END.

New YOLO NAI SHA256:
`09e7119e56d8a744a94f9a1c5af42ed4716ddbb11e4f63d974dd2b2790f88491`.
New MobileNet NAI SHA256:
`4fd5d48405f972f552fc930981e6a00dd9ff91da8b166adbd65bb60738308024`.
Compiler build, tests and compile logs are under
`out/psum_external_20261012/` (local).

## Verified bounded cluster gate

The new YOLO NAI and unchanged firmware have been frozen; the bounded
first-cascade reference case starts from reset and ends at command 50, with
checkpoints at commands 14, 23 and 50. The cluster simulator has been rebuilt.
The new consumer tile has M=480 and nine K tiles. Its 61,440-byte PSUM span
is `[267200,328640)`; input staging starts at 328,640, with no overlap.
On 2026-10-12 this replay passed all 50 commands and every byte comparison:

| Command | Reference tensor | Checked bytes | Mismatches |
| --- | --- | ---: | ---: |
| 14 | 159, stem SiLU | 30,720 | 0 |
| 23 | 159, stem SiLU | 30,720 | 0 |
| 50 | 161, second Conv | 15,360 | 0 |

The command PMU reports 341,489 cycles. Per-command CSV, actual/expected bytes,
`byte_exact.json` and the complete `command-50.snapshot` are retained in `run/`
under the evidence root below. This proves the first external-PSUM consumer
tile, not the rest of the graph.

Local cluster evidence root:
`neural-ai/hw/rtl/cluster/tb/sim/test_yolo_external_psum_20261012/`.

## Verified rolling-cascade cluster gate

The pinned reset replay completed all 107 commands and checked commands
14, 23, 50, 55, 66, 69, 75, 105 and 107. This covers consecutive
M=480/nine-K jobs, reused producer storage and a split SiLU activation around
an asynchronous producer's completion. The reference mapper tracks completed
physical C32 rows separately from pending producer output; it rejects LUT
access to pending or stale/noncontiguous rows.

All nine checkpoints passed: **184,320 / 184,320 bytes exact, zero mismatches**.
PMU reports **859,303 cycles**; the simulation took 1,427.48 wall-clock seconds.
The four consumer tiles cover 1,920 spatial rows of tensor 161 (61,440 bytes),
while producer SiLU checkpoints cover 122,880 bytes. These checks occur at
command completion, before rolling storage may overwrite the bytes.

Preparation had 14 offline helper tests passing. The case, per-command CSV,
actual/expected byte files, result JSON and `command-107.snapshot` are retained
in `run-107/` under the same evidence root. Reproduce preparation with a fresh
output directory:

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_external_psum_20261012/artifacts/dma_events.json \
  --from-reset --consumer-tiles 4 \
  --output hw/rtl/cluster/tb/sim/test_yolo_external_psum_20261012/run-107
```

Do not resume directly from snapshot 50: memory snapshots omit the resident
requant registers and AFU LUT SRAM, and command 55 reuses requant configuration.
The current restore helper supports only a narrowly verified idempotent LUT
preamble, not arbitrary register restoration. Reset replay avoids relying on
unrestored hardware state. Derive checkpoint numbers from the ABI each time.

## Next integration gate: full YOLO output

Run all 5,008 logical commands from reset using the **original pinned NAI**,
including its encoded affine loops, rather than a flattened replacement stream.
Retain the nine verified early checkpoints, require all 176,400 final output
bytes to equal the pinned Neural-AI hybrid reference, and collect PMU for every
command. Use a 50,000,000-cycle execution timeout and the existing 100,000-cycle
progress log. No RTL, firmware or compiled-model change is needed for this gate.

The reference still delegates only Conv/Depthwise/FC; other operators use
TFLite reference kernels. A full-output discrepancy is evidence to diagnose,
not permission to weaken the comparison or automatically classify AFU/DFL as
faulty. Agreement would establish the measured input's output, not dataset
accuracy or every intermediate tensor.

The helper's full-graph mode has 21 passing offline tests, including rejection
of partial execution/missing or truncated output oracles, strict one-byte
mismatch detection and preservation of encoded affine loops. Preparation:

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_external_psum_20261012/artifacts/dma_events.json \
  --from-reset --consumer-tiles 4 --full-graph --timeout-cycles 50000000 \
  --output hw/rtl/cluster/tb/sim/test_yolo_external_psum_20261012/run-full
```

Result scope and artifact hashes are explicit in `byte_exact.json`.
`output-actual.bin` and `output-expected.bin` retain the complete output;
dispatch errors/timeouts record `ERROR` and `failure-tcdm.bin`, not a data PASS.

Full-model byte-exactness remains unverified.
The old replay's 391,869 cycles are evidence of the faulty execution, not a
valid performance baseline for the repaired inference.
