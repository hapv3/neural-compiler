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

## Checkpoint-first YOLO verification

Full regression is deferred until the bounded numerical gates pass. A firmware
PASS or a completed command is not a byte-exact checkpoint. Each gate compares
its live output bytes with the pinned Neural-AI hybrid reference, before rolling
storage is reused, and stops on the first mismatch. Keep the original input,
model, firmware and reference identities across gates; any identity change
invalidates the previous coverage claim.

| Gate | Compared data | Status |
| --- | --- | --- |
| Through command 50 | Two stem SiLU stripes and first consumer Conv tile | PASS, 76,800 bytes |
| Through command 107 | Nine checks, including four consumer tiles | PASS, 184,320 bytes |
| Through command 157 | Previous nine plus commands 118, 121, 127 and 157 | PASS, 261,120 bytes |
| Through command 211 | Previous thirteen plus 159, 170, 173, 179, 209 and 211 | Prepared; simulation pending |
| Remaining cascade and later layers | Completion-aware tiles/groups and operator boundaries | Not yet verified |
| Full YOLO and MobileNet regression | Complete output bindings | Deferred |

The gate through command 157 checks another 61,440 producer SiLU bytes and 15,360
consumer Conv bytes. All 13 checks total 261,120 bytes. The replay ends at
command 157, with a 2,000,000-cycle timeout, and records command PMU as before.
It starts from reset: snapshot 107 lacks AFU LUT SRAM, and subsequent commands
reuse that LUT. Do not treat a memory-only restart as restored hardware state.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_external_psum_20261012/artifacts/dma_events.json \
  --from-reset --consumer-tiles 5 --fail-fast --timeout-cycles 2000000 \
  --output hw/rtl/cluster/tb/sim/test_yolo_external_psum_20261012/run-157
```

On 2026-10-12 gate 157 passed all 13 checkpoints: **261,120 / 261,120 bytes
exact**. Actual byte files and hashes were independently rechecked against the
pinned expected files. Cocotb reports PASS, the complete memory snapshot at 157
passes integrity/provenance checks, and CSV contains commands 1–157 with 169
columns. Command cycles sum to 1,017,614, matching command-active PMU;
whole-run PMU is 1,155,301 cycles. Simulation wall time was 2,107.99 seconds.
Evidence remains in `run-157/` under the pinned YOLO evidence root.

The next gate ends at **211**, adding six comparisons (107,520 bytes): consumer
Conv at 159/209/211 and rolling producer SiLU at 170/173/179. Its 19 checks total
368,640 bytes and cover eight consumer tiles. It still starts from reset because
the resident LUT is not part of the memory snapshot; no unverified register
restore is assumed. Prepare with the same command above, changing to
`--consumer-tiles 8 --timeout-cycles 2500000` and a fresh `run-211` output directory.
Retain `--fail-fast`; no full-model regression is scheduled by this gate.

Advance only after zero mismatches at every listed checkpoint and successful
firmware completion. For later layers, derive tensor/group/row mappings from
the current ABI and source graph; extend the mapper before testing unsupported
async consumers or multiple channel groups, rather than silently skipping them.
Cover remaining cascade tiles, later Conv/SiLU, residual and layout/pooling
paths, then detection heads/DFL and final output. A snapshot is reusable only
with matching artifacts, verified data and a supported register restore path.

The stopped full YOLO run reached roughly command 2500 but compared only the
same nine early checkpoints. Its report may retain `running` after process
termination; neither command progress nor that report certifies later bytes.
The 37 offline helper/snapshot/numerical tests pass, including fail-fast evidence
retention and rejection of malformed flags. No RTL or firmware changes are
part of this verification-policy update.

## Deferred regression: full YOLO output

After the bounded gates pass, run all 5,008 logical commands from reset using the **original pinned NAI**,
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

## MobileNet224 full-output gate

The same verified RTL binary and firmware also support a separate MobileNet224
reset run: 12,789 logical commands, a 50,000,000-cycle timeout and strict
comparison of all 1,000 final INT8 output bytes (source tensor 176). No YOLO
stem checkpoints are imposed. The original encoded NAI is preserved, with
SHA256 `4fd5d48405f972f552fc930981e6a00dd9ff91da8b166adbd65bb60738308024`.

The generic runner uses `MODEL_REFERENCE_CASE` and
`COCOTB_TEST_MODULES=test_model_reference_replay`. Its preparation command is:

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --source /home/dev01/neural-compiler/test/model/mobilenet_v2_224_int8.tflite \
  --manifest hw/rtl/cluster/tb/sim/test_mobilenet224_reference_20261012/artifacts/dma_events.json \
  --from-reset --consumer-tiles 0 --full-graph --timeout-cycles 50000000 \
  --output hw/rtl/cluster/tb/sim/test_mobilenet224_reference_20261012/run-full
```

Preparation passed 27 helper tests and verified complete hybrid-reference MAC
coverage. The 4,224,864-byte model requires control data at `0x81800000`, above
its encoded bytes; its 2,408,448-byte temporary binding fits between
`0x80200000` and the model at `0x81000000`. The tests cover both address limits
and preserve compatibility with the prepared full YOLO case.

Evidence is under the MobileNet run directory above: `sim.log`, continuously
appended `command_pmu.csv`, final `output-actual.bin`, `byte_exact.json` and
`command-12789.snapshot`. A separate copy of the verified simulator avoids
changing the binary used by the concurrent YOLO run. Simulation and full-output
correctness are pending; offline preparation alone is not an inference PASS.
