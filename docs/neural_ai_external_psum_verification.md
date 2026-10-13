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
storage is reused. Diagnostic runs can stop on the first mismatch; checkpoint
batches collect all comparisons and conclude at the end. Keep the original input,
model, firmware and reference identities across gates; any identity change
invalidates the previous coverage claim.

| Gate | Compared data | Status |
| --- | --- | --- |
| Through command 50 | Two stem SiLU stripes and first consumer Conv tile | PASS, 76,800 bytes |
| Through command 107 | Nine checks, including four consumer tiles | PASS, 184,320 bytes |
| Through command 157 | Previous nine plus commands 118, 121, 127 and 157 | PASS, 261,120 bytes |
| Through command 211 | Previous thirteen plus 159, 170, 173, 179, 209 and 211 | Prepared, superseded by the complete-cascade batch |
| Complete initial cascade, old NAI through 399 | All 33 planned checkpoints, both complete logical tensors | Stopped; command 159 byte-exact FAIL, diagnosed below |
| Complete initial cascade, corrected NAI through 525 | 38 live checks and complete retained consumer L2 output | PASS, 614,400 live bytes + 204,800 L2 bytes |
| Next SiLU / pointwise Conv / SiLU through 587 | All initial checks plus 59 completion-aware checks | Prepared; cluster verification pending |
| Remaining later layers | Completion-aware tiles/groups and operator boundaries | Not yet verified |
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

The intermediate gate at **211** adds six comparisons (107,520 bytes): consumer
Conv at 159/209/211 and rolling producer SiLU at 170/173/179. Its 19 checks total
368,640 bytes and cover eight consumer tiles. It still starts from reset because
the resident LUT is not part of the memory snapshot; no unverified register
restore is assumed. Prepare with the same command above, changing to
`--consumer-tiles 8 --timeout-cycles 2500000` and a fresh `run-211` output directory.
This gate was prepared but not launched; the complete-cascade batch below
supersedes repeated smaller prefix runs. No full-model regression is scheduled.

### Complete supported checkpoint list, one run

The requested batch derives all initial-cascade tiles from tensor extents rather
than a fixed tile count. It runs **33 checkpoints through command 399**, comparing
all 25,600 stem SiLU rows (409,600 bytes) and all 6,400 consumer Conv rows
(204,800 bytes): **614,400 bytes** total. Missing, overlapping or duplicate
logical rows are rejected during preparation and case loading.

The last consumer submits asynchronously at command 363. Its 5,120 output bytes
are captured after WAIT 399, not at submission. Intervening later-layer LUT/DMA
work must be disjoint from the pending output; that work is not counted as a
verified tensor. The pre-compute dump is taken before submit 363. This supports
the actual compiler schedule without modifying RTL, firmware or command payloads.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_external_psum_20261012/artifacts/dma_events.json \
  --from-reset --all-cascade-checkpoints --timeout-cycles 4000000 \
  --output hw/rtl/cluster/tb/sim/test_yolo_external_psum_20261012/run-all-cascade
```

`--fail-fast` is intentionally omitted: collect every listed comparison, keep
the live per-command CSV, and inspect the final report afterward. Any numerical
mismatch makes the final result FAIL; timeout/firmware failure remains ERROR.
There is one reset replay and one final memory snapshot, not a separate replay
for every prefix. Actual/expected data remains available for each checkpoint.
The 44 offline tests pass, including async completion, missing-wait rejection,
pending-output read/write hazards and complete row-coverage validation.

This is the full list of currently supported **initial-cascade** checkpoints,
not a claim that all YOLO layers have numerical mappings. Later operators and
full-output regression remain pending. Evidence is in `run-all-cascade/`.

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

## Cascade L2 output alias: diagnosis and correction, 2026-10-13

The old complete-cascade batch was stopped after 30 comparisons through command
326: 29 matched, but command 159 differed in 5,039 of 15,360 bytes (first offset
5,120; maximum absolute difference 129). This is not a complete batch PASS.
The raw interrupted report retains `running`; do not interpret it as completion.

Command 157 writes its INT8 output to TCDM `[61440,76800)`. DMA 158 subsequently
reads live stem activation from `[35840,102384)` into staging for Conv 159.
The earlier output destroys 7,680 active input bytes in three source image rows;
two output image rows then differ. Independently replaying DMA 158 after
injecting command 157's verified output reproduces all 66,976 staged input bytes.
Offline convolution from that corrupted staging matches all 15,360 simulator
output bytes; retaining the live input instead matches all reference bytes.

Recompilation with allocation logging reproduces the pinned old NAI hash.
Tensor 161 is allocated at offset 0 **in L2**, whereas the producer's rolling
activation starts at offset 0 **in TCDM**. The striped linebuffer emitter discarded
the region and used `AllocatedAddress()` directly as an RTL-local OFM address.
This is a compiler lowering bug, not a PSUM or rounding RTL failure.

The correction resolves a typed output reference, reserves a separate C32 output
stripe after aligned input staging, and emits a TCDM-to-L2 DMA store to the actual
tensor offset. Workspace reservation and HLC emission share the output-stripe
size calculation, including a separate padding seed. Local output remains direct.
Global scheduling retains the output write/store/read hazards and inserts waits
before that staging can be reused. No RTL, firmware, or ABI changes are involved.

The new ASan regression covers L2 offsets 0 and 1 MiB, rolling input protection,
input/weight/PSUM separation, and exactly-once stores of all consumer output
bytes. It fails with the baseline generator and passes with the correction.
The broader baseline has four independent failures: stale LUT placement
expectations, ReLU/pointwise layout, staging diagnostic text, and compact
depthwise input-DMA expectations. All four were reproduced with the baseline
generator; they are not silently counted as passing or changed here.
With those four explicitly excluded, **182 tests / 43,978 assertions pass** on
the final corrected sources. Firmware host/ABI checks also pass. Leak detection
is disabled because it cannot run under the sandbox's tracing environment;
AddressSanitizer remains enabled.

The corrected YOLO NAI has **5,254 commands** (previously 5,008), still with zero
CPU operators and 512,000 TCDM bytes. Workspace feedback changes stripe geometry
when output staging is budgeted, so the difference includes more than the L2
stores themselves. Performance must be measured again, not inferred from count.
SHA256: `937ac5fbf64d317e29f5eec2983c29789d10dc53e63143dd0e10ec4e31c95890`.
MobileNet recompilation is byte-identical to its earlier pinned NAI.

The updated checkpoint mapper follows each reused local output stripe to its
contiguous L2 store. Missing stores, overwrites, stale mappings, or missing async
completion waits are rejected. A bounded replay extends to the first globally
drained boundary; the complete batch additionally compares the retained L2 tensor
after all stores finish. The 49 offline reference/snapshot/numerical tests pass.

The new reset batch **passed on 2026-10-13**: all 38 live checkpoints match
614,400 bytes, followed by an exact 204,800-byte L2 output comparison.
All 16 consumer tiles complete locally by command 443, but the last queued store
is joined at 523 and the other DMA queue drains at **525**. Commands between these
boundaries include later-layer work that is not numerically covered by this gate.
The timeout is 4,000,000 cycles and per-command PMU CSV remains enabled.
Do not reuse old-model snapshots or hardcode old checkpoint IDs.

Whole-run PMU reports **2,917,285 cycles**; the 525 command rows sum to
**2,565,010 command-active cycles**. Cocotb reports PASS in 6,337.91 wall-clock
seconds. An independent artifact recheck compared all 38 saved actual files
with their pinned expected files, compared the final L2 bytes, verified the
snapshot's model/boundary/integrity, and checked unique CSV commands 1–525
with all 169 columns. Both command-begin and command-end counters equal 525.
There were zero mismatches, including the previously failing consumer stripes.
This does not establish correctness of later work interleaved before 525.

Local evidence:

- Compiler diagnostics, baseline failures, ASan results and host tests:
  `out/psum_external_20261012/alias_*.log`.
- Corrected models: `out/cascade_l2_20261013/{yolo320,mobilenet224}/`.
- Prepared cluster case and pinned artifacts:
  `neural-ai/hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/`.

### Next bounded gate: complete SiLU / pointwise Conv / SiLU

The mapper now follows each completed consumer L2 store through DMA1D loads
and their explicit waits, the second Conv's SiLU, the next single-C32-group
pointwise Conv, and its SiLU. Logical row identities follow the input data,
not local output addresses that the schedule reuses. Missing/incomplete loads
or stores, overwritten/stale sources, duplicate rows, partial/multi-group
pointwise jobs and unsupported layouts are rejected. Source tensor IDs are
resolved through actual Conv → Logistic/Mul edges in the TFLite graph.

The prepared reset replay retains all 38 initial checks and adds **59 checks**:
51 for tensor 163 (second SiLU), four for tensor 164 (pointwise Conv), and four
for tensor 166 (its SiLU). Each new tensor is checked completely: 6,400 rows ×
32 channels = 204,800 bytes. The **97 live comparisons total 1,228,800 bytes**;
the original retained L2 consumer is also compared again at completion.
Pointwise checkpoints are 524, 529, 560 and 579; their activation checkpoints
are 528, 530, 581 and 582. DMA loads pending at 582 drain at **587**, which is
the first safe final boundary. The timeout is 8,000,000 cycles; this is a bound,
not a performance estimate. Per-command PMU remains enabled.

Before launching, additional snapshot-525 comparisons matched the 109,568
retained bytes of tensor 163 and the first 56,320 bytes of tensor 164. These
checks do not cover their remaining tiles or tensor 166. Memory-only snapshot
525 still lacks LUT/requant state; this gate deliberately starts from reset.
The expanded offline helper/snapshot/numerical suite passes **55 tests**.
No RTL, firmware, or compiled NAI changes are involved.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/artifacts/dma_events.json \
  --from-reset --post-cascade-checkpoints --timeout-cycles 8000000 \
  --output hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-post-cascade
```

Use a fresh output directory when repeating preparation. Current case, oracles,
subsequent actual bytes, PMU and results are kept in `run-post-cascade/` under
that evidence root. **Preparation is not a cluster/data PASS**; full-model
regression remains deferred until the required bounded checkpoints pass.

## Deferred regression: full YOLO output

After the bounded gates pass, run every logical command from reset using the **newly verified pinned NAI**,
including its encoded affine loops, rather than a flattened replacement stream.
Retain the verified early checkpoints, require all 176,400 final output
bytes to equal the pinned Neural-AI hybrid reference, and collect PMU for every
command. Use a 50,000,000-cycle execution timeout and the existing 100,000-cycle
progress log. No further RTL, firmware or compiled-model change is needed for this gate.

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
