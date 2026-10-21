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
| Next SiLU / pointwise Conv / SiLU through 587 | All initial checks plus 59 completion-aware checks | PASS, 1,228,800 live bytes + 204,800 L2 bytes |
| First C2f residual/concat through 1817 | Previous checks plus complete branch, two Conv/SiLU stages, residual and concat | PASS, 2,150,400 live bytes + 204,800 early L2 bytes |
| Complete first C2f output through 2020 | Previous checks plus closing two-input-group pointwise Conv and SiLU | PASS, 2,560,000 live bytes + 204,800 early L2 bytes |
| Next K3/S2 Conv and SiLU through 2057 | Previous checks plus both C32 output groups of C64 tensors | PASS, 2,764,800 live bytes + 204,800 early L2 bytes |
| Snapshot-2057 continuation through 2067 | Retained parent + next two-group pointwise Conv / SiLU | PASS, 204,800 new live bytes + 102,400 retained parent bytes |
| Snapshot-2057 continuation through 2266 | Next complete two-bottleneck C2f, both AFU Add paths and C128 concat | PASS, 921,600 live bytes |
| Snapshot-2266 continuation through 2346 | Closing C128-to-C64 pointwise Conv, streamed SiLU and retained L2 | PASS, 204,800 live bytes + 102,400 L2 bytes |
| Snapshot-2266 continuation through 2405 | Previous output checks plus L2-fed K3/S2 Conv and C128 SiLU | PASS, 307,200 live bytes + 102,400 L2 bytes |
| Snapshot-2405 continuation through 2418 | Four-group C128 pointwise Conv and whole-tensor SiLU | PASS, 102,400 live bytes |
| Snapshot-2405 continuation through 2604 | Previous five checks plus complete C64-branch C2f, both residual Adds and C256 concat | PASS, 460,800 live bytes |
| Snapshot-2604 continuation through 2640 | Closing C256-to-C128 pointwise Conv, streamed SiLU and retained L2 | PASS, 102,400 live bytes + 51,200 L2 bytes |
| Snapshot-2604 continuation through 2731 | Previous output checks plus L2-fed K3/S2 Conv and C256 SiLU | PASS, 153,600 live bytes + 51,200 L2 bytes |
| Snapshot-2731 continuation through 5254 | All 2,523 remaining commands, nine prefix checks and final output | Commands complete; final output FAIL, 7,929 / 176,400 bytes differ |
| Snapshot-2731 continuation with detection-head observations | Same complete suffix plus 115 diagnostic checks covering all 18 head Convs | FAIL at command 4314: 2,909 / 3,200 bytes differ; Conv is exact on its incorrect input |
| Repaired pointwise cascade, snapshot-2731 continuation through 5513 | Nine prefix checks, 313 extra Conv observations and final output | All 322 live checks PASS; final FAIL in 7,047 box bytes, class bytes all exact |
| Compact plane alignment regression | Compiler-generated Add / slice / general Add, 4,200 output bytes | PASS: 14/14 commands, all 4,200 bytes exact |
| Alignment-corrected snapshot-2731 continuation through 5520 | Same 322 live checks, strict final output and command PMU | All 322 checks PASS; 56 intermediate snapshots verified; final FAIL in 3,465 box bytes, maximum 7 |
| Fused DFL numerical equivalence | Three DFL heads and all downstream binary steps from retained snapshots | Deployed arithmetic matches RTL; differs from the unchanged hybrid reference |
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

The reset replay retains all 38 initial checks and adds **59 checks**:
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

On 2026-10-13 this gate **passed all 97 checks**, with **zero mismatches** in
1,228,800 live bytes and all 204,800 retained L2 bytes. PMU reports **3,312,759
whole-run cycles** and **2,925,624 command-active cycles**, with 587 begin/end
events, 587 unique CSV commands, 169 columns and no counter overflow. Cocotb
reports PASS in **6,247.63 wall-clock seconds**. An independent recheck compared
every actual/expected file and the retained L2 bytes, verified snapshot 587,
and reconciled the CSV cycle sum with command-active PMU.

Use a fresh output directory when repeating preparation. Case, actual/expected
bytes, PMU and results are kept in `run-post-cascade/` under that evidence root.
This proves complete tensors 163, 164 and 166 in addition to the initial
cascade, not the remaining graph. Full-model regression remains deferred.

### Next bounded gate: first C2f residual/concat block

The next mapper resolves the source graph's channel slices and concat order,
then propagates byte identities through the actual DMA1D/2D/3D strides. It
supports compact C16 and C32 storage without treating padding as active data.
Async Conv outputs become readable only at SYSTOLIC_WAIT; queued DMA data
becomes readable only at the matching queue wait. Stale/misaligned residual
inputs, missing/duplicate rows and incomplete concat materialization are errors.
The mapper assigns logical Conv rows in verified raster tile order and checks
every output byte against the independently computed hybrid reference; it is
not a general numerical execution model for arbitrary graphs.

| New tensor/check | Source tensor | Live checks | Active bytes |
| --- | ---: | ---: | ---: |
| Compact branch slice | 167 | 1 | 102,400 |
| First block Conv / SiLU | 168 / 170 | 20 / 60 | 204,800 |
| Second block Conv / SiLU | 171 / 173 | 80 / 80 | 204,800 |
| AFU biased residual Add | 174 | 25 | 102,400 |
| Two-group C32 concat, 48 channels | 176 | 1 | 307,200 |

All previous 97 checks are retained: **364 live checks / 2,150,400 bytes** total.
The initial consumer L2 output is compared at **587**, before block activation
stores reuse its allocation; it is not incorrectly compared at the final
snapshot. The Python PMU monitor supports short async callbacks for this
existing backdoor read, while retaining strict context-sequence checks. Only
the 204,800-byte oracle range is read (about 6.4 ns of simulation time), not
the entire L2 binding. RTL, firmware and the pinned compiled NAI are unchanged.

The final checkpoint is concat materialization at **1817**, a drained boundary
before the next multi-input-group pointwise Conv. The timeout is 12,000,000
cycles, a conservative bound rather than a performance estimate. Before launch,
snapshot 587 also matched all 102,400 bytes of branch tensor 167. The expanded
offline helper/snapshot/numerical suite passes **64 tests**; the prepared real
case additionally rejects missing checks, stale compact layout, a wrong live-L2
boundary, wrong concat channel order and stale coverage, and still loads both
earlier gate cases. These preparation checks did not establish a C2f cluster
PASS on their own.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/artifacts/dma_events.json \
  --from-reset --c2f-checkpoints --timeout-cycles 12000000 \
  --output hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-c2f
```

Use a fresh output directory for a rerun. Evidence stays in `run-c2f/`, including
actual/expected bytes, per-command PMU and the final memory snapshot.

Verified on **2026-10-14**: gate 1817 **passed all 364 checkpoints**, with zero
mismatches in **2,150,400 bytes**. The live L2 comparison at command 587 also
matched all 204,800 bytes. PMU reports **6,904,735 whole-run cycles** and
**5,794,818 command-active cycles**, with 1,817 command begin/end events, no
counter overflow, and all 1,817 CSV rows/169 columns. Cocotb reports PASS in
**11,923.34 wall-clock seconds**. Independent saved-file comparisons, snapshot
integrity/provenance, XML status and CSV cycle reconciliation also passed.
This covers the first C2f through concat, not its following pointwise Conv or
the rest of YOLO. Full-model numerical correctness remains pending.

### Next bounded gate: closing C2f pointwise Conv and SiLU

The next gate follows tensor 176's 48 active channels across two resident C32
input groups into the closing 1×1 Conv (tensor 177), its reused local output
staging and L2 stores, then DMA loads and SiLU (tensor 179). Physical group
strides, output rows and completion points come from ABI commands; source
tensor identities come from the actual concat → Conv → Logistic/Mul edges.
The mapper rejects stale/missing input-group data, mismatched group strides,
duplicate or missing rows, missing DMA waits, and partial/OFM workspace
overlap with each other or with the resident concat. It does not perform the
numerical Conv itself: output oracles remain the independent hybrid reference.

The prepared reset gate retains all 364 checks and adds **75**: 25 Conv tiles
and 50 SiLU tiles, each new tensor covering 6,400 × 32 = 204,800 active bytes.
Total coverage is **439 checkpoints / 2,560,000 live bytes**, plus the retained
204,800-byte early L2 comparison at 587. It ends at the first drained boundary,
**2020**, before the next downsampling Conv; timeout remains 12,000,000 cycles.
Per-command PMU is enabled. The new offline suite passes **68 tests**. No RTL,
firmware or compiled-model changes are involved, and memory-only snapshot
1817 is not assumed to restore the requant registers loaded earlier.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/artifacts/dma_events.json \
  --from-reset --c2f-output-checkpoints --timeout-cycles 12000000 \
  --output hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-c2f-output
```

Choose a fresh output directory for a rerun. Artifacts, expected/actual bytes,
PMU and the final snapshot stay in `run-c2f-output/`.

Verified on **2026-10-15**: gate 2020 **passed all 439 checkpoints**, with zero
mismatches in **2,560,000 live bytes** and the 204,800-byte L2 comparison at
587. PMU reports **7,205,685 whole-run cycles** and **5,979,252 command-active
cycles**; all 2,020 command begin/end events and CSV rows are present, with
169 columns and no overflow. Cocotb reports PASS in **16,022.83 wall-clock
seconds**. Independent actual/expected file comparison, snapshot provenance,
XML status and CSV-cycle reconciliation also passed. This establishes complete
first-C2f output tensors 177/179 for the pinned input, not full-model correctness.

### Next bounded gate: grouped K3/S2 downsample Conv and SiLU

The mapper follows the next source Conv through its optional PAD edge from
tensor 179. Tensor 181 is a 40×40×64 Conv output, stored as two C32 groups;
tensor 183 is its SiLU output. Each group has independent raster rows selected
by the explicit LOAD_QPARAMS group selector. Stale selectors, noncontiguous
destinations and reads of pending async results are rejected. The whole-tensor
LUT requires every output group to have completed.

The oracle now slices both spatial rows and channels from the original NHWC
tensor. Flattening a C64 tensor into C32 rows would interleave channel groups
incorrectly; a dedicated regression distinguishes these two interpretations.
Coverage proves every `(row, channel)` cell is checked exactly once, rather
than counting both groups as duplicate spatial rows. All **439 preceding
expected byte files remain unchanged**.

The prepared reset gate adds **15 checks** (14 Conv tiles + one C64 SiLU),
covering both complete tensors: **454 checkpoints / 2,764,800 live bytes** total,
plus the early 204,800-byte L2 comparison at 587. It ends at the drained boundary
**2057**, before the next pointwise Conv. Timeout remains 12,000,000 cycles and
per-command PMU is enabled. The offline helper/snapshot/numerical suite passes
**73 tests**. RTL, firmware and the compiled NAI are unchanged; snapshot 2020
is not assumed to restore the requant configuration loaded at 1869.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/artifacts/dma_events.json \
  --from-reset --downsample-checkpoints --timeout-cycles 12000000 \
  --output hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-downsample
```

Use a fresh output directory for reruns. Byte oracles, actual outputs, per-command
PMU and the final snapshot stay in `run-downsample/`.

Verified on **2026-10-15**: all **454 checkpoints / 2,764,800 live bytes passed**,
with zero mismatches; early L2 comparison also matched 204,800 bytes. PMU
reports **7,754,503 whole-run cycles**, **6,501,144 command-active cycles**,
2,057 command begin/end events, 2,057 CSV rows/169 columns and no overflow.
Cocotb reports PASS in **18,708.93 wall-clock seconds**. Independent saved-byte,
snapshot integrity/provenance, XML and CSV-cycle reconciliation checks passed.
This establishes tensors 181/183 in addition to preceding gates, not full YOLO.

### Verified-parent continuation: grouped pointwise from snapshot 2057

This suffix explicitly configures RQ at 2058/2060 and loads a fresh LUT at
2063; no old RQ or LUT state is required. The replay resets the hardware,
restores the complete memory snapshot, and executes **2058–2067 only**, not
the already verified prefix. Its three new checks are Conv tensor 184's two
output groups at 2059/2061 and full C64 SiLU tensor 186 at 2063:
**204,800 new live bytes**. Pending constant/weight DMA loads require the final
join at **2067**, the first globally drained boundary; that later-layer setup
is not counted as numerically verified work. Timeout is 1,000,000 cycles.

Continuation is accepted only when all of the following hold:

- The preceding case/report is PASS at the exact snapshot boundary, with
  matching model, source, firmware, input and reference-build identities.
- Snapshot model/boundary/integrity and binding sizes match. The retained
  parent tensor 183 is re-compared in full: **102,400 bytes** against both the
  pinned oracle and the current independent reference before launching.
- The prepared case pins the snapshot, previous case and PASS report hashes.
  Runtime case loading repeats the parent comparison before memory restore.
- Every pointwise output group has an explicit matching RQ load; its source
  groups and partial/OFM workspaces pass provenance and overlap checks. The
  first LUT must load its configuration itself, rather than reuse missing SRAM.

The **81 offline tests** include rejected running/failed evidence, wrong
identity/boundary/reference build, corrupt or partial parent data, missing RQ
loads, missing output groups, stale input and LUT-reuse rejection. Historical
non-LUT AFU work no longer prevents a suffix whose first LUT explicitly loads
new constants; suffixes that need historical LUT residency retain the existing
strict restore checks. Earlier reset gate cases remain supported.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/artifacts/dma_events.json \
  --snapshot hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-downsample/command-2057.snapshot \
  --resume-grouped-pointwise-checkpoints --timeout-cycles 1000000 \
  --output hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-resume-pointwise
```

Evidence stays in `run-resume-pointwise/`, with original command numbers
2058–2067 in the ten PMU rows, initial-parent statistics, new actual/expected
bytes and snapshot 2067. This is a narrowly verified self-configuring restart,
not arbitrary register restoration. Other boundaries still require an explicit
safe restore plan or replay from a supported start.

Verified on **2026-10-15**: all **3 checkpoints / 204,800 new live bytes passed**,
with zero mismatches. The retained parent independently matched all 102,400
bytes. PMU reports **113,061 whole-run cycles**, **65,068 command-active cycles**,
10 command begin/end events, 10 CSV rows/169 columns and no overflow. Cocotb
reports PASS in **546.23 wall-clock seconds**. Independent saved-byte, snapshot
integrity/provenance, XML and CSV-cycle reconciliation checks passed. This
establishes tensors 184/186, not the following C2f or full-model output. No RTL,
firmware or NAI changes are involved.

### Complete resident C2f chain from the verified snapshot

`--resume-c2f-chain-checkpoints` includes the self-configuring pointwise gate
above, then resolves all residual bottlenecks and the concat through source
graph edges. For this model it executes **2058–2266**, 209 original commands,
with **18 checkpoints / 921,600 live bytes**. The next closing pointwise Conv
is excluded. Timeout is 2,000,000 cycles and `--fail-fast` preserves the first
mismatch and diagnostic TCDM instead of continuing through downstream errors.

| Commands | Reference tensors | Compared bytes |
| --- | --- | ---: |
| 2059 / 2061 / 2063 | Pointwise 184 and SiLU 186 | 204,800 |
| 2110 / 2111 / 2113 | First bottleneck Conv 189 and SiLU 191 | 102,400 |
| 2159 / 2160 / 2161 | First bottleneck Conv 192 and SiLU 194 | 102,400 |
| 2163 | Residual 195, `AFU_BINARY_QUANT` general Add | 51,200 |
| 2209 / 2210 / 2212 | Second bottleneck Conv 196 and SiLU 198 | 102,400 |
| 2257 / 2258 / 2259 | Second bottleneck Conv 199 and SiLU 201 | 102,400 |
| 2260 | Residual 202, `AFU_BINARY` biased Add | 51,200 |
| 2266 | Complete four-group C128 concat 203 | 204,800 |

These command numbers are derived results, not mapper constraints. The helper
supports resident C64 parents split into C32 branches and resolves the chain
length, source slices, concat order, ABI layers, raster tiles, addresses and
downstream input-group stride. It tracks byte provenance across local/L2 DMA,
requires current explicit RQ loads, captures async Conv at its wait, rejects
stale LUT state after non-LUT AFU work, and checks residual operands and every
concat channel group. Numerical oracles remain independent reference tensors,
not data generated by the provenance tracker. Every role must cover every
spatial/channel cell exactly once.

The **92 offline tests** pass, including one/two-bottleneck mapping, both Add
paths, source-edge/layout errors, stale data/RQ/LUT state, duplicate rows and
missing DMA/compute waits. All six earlier reset/snapshot cases still load;
the new gate's pointwise oracles match the already verified case. Seven
real-case metadata mutations were independently rejected. Offline evidence is
`resume-c2f-helper-tests.log` and preparation evidence is `prepare-resume-c2f.log`.

Use snapshot **2057**, not 2067: the memory-only snapshot at 2067 follows RQ
load 2065, so restarting after it would omit that live hardware configuration.
Replaying ten cheap verified commands avoids rerunning the 7.75M-cycle prefix
without inventing arbitrary register restoration.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/artifacts/dma_events.json \
  --snapshot hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-downsample/command-2057.snapshot \
  --resume-c2f-chain-checkpoints --timeout-cycles 2000000 --fail-fast \
  --output hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-resume-c2f
```

Simulation evidence stays in `run-resume-c2f/`: original-numbered
`command_pmu.csv` (209 rows expected), live actual/expected bytes,
`byte_exact.json`, `sim.log`, `results.xml` and, on completed execution,
`command-2266.snapshot`.

Verified on **2026-10-16**: all **18 checkpoints / 921,600 live bytes passed**,
with zero mismatches. PMU reports **871,693 whole-run cycles**,
**707,944 command-active cycles**, 209 command begin/end events, 209 CSV
rows/169 columns and no overflow. Cocotb reports PASS in **1,491.79 wall-clock
seconds**. Independent saved-byte, snapshot integrity/provenance, retained
concat, XML and CSV-cycle reconciliation checks passed. This establishes the
complete second C2f interior through tensor 203, not its closing Conv/SiLU or
full-model output.

Per-command measurements: general AFU Add 2163 took **6,788 cycles**
(6,411 AFU-active); biased AFU Add 2260 took **6,704 cycles** (6,403 AFU-active).
Each compared 51,200 bytes with zero mismatches and no Spatz-active cycles.
The six local concat-copy commands 2261–2266 each took **33,022 cycles**:
**198,132 cycles total**, approximately 28.0% of command-active time. Keep
these as measured tuning evidence, not a full-inference estimate or permission
to change the implementation. The auxiliary DMA event trace reached its
1,000-row cap; the 209-row per-command PMU CSV and all numerical checks are
complete and unaffected. No RTL, firmware or NAI changes were involved.

### Closing pointwise, streamed SiLU and retained L2 from snapshot 2266

`--resume-c2f-output-checkpoints` restores snapshot **2266** only after its
PASS case/report identities and all **204,800 retained concat bytes** have
been independently rechecked. It executes **2267–2346**, 80 original commands,
without replaying earlier work. Explicit RQ loads 2267/2269 configure the two
pointwise output groups; LUT 2271 loads fresh constants. No old hardware
register state is assumed. Timeout is 2,000,000 cycles, with fail-fast enabled.

The **27 checkpoints / 204,800 live bytes** cover:

- Tensor 204, closing Conv: four C32 input groups, two C32 output groups at
  2268/2270, all 102,400 output bytes.
- Tensor 206, SiLU: 25 physical 4 KiB LUT tiles, all 102,400 bytes. At 2307,
  one tile contains rows 1536–1599 of channels 0–31 followed by rows 0–63 of
  channels 32–63. `reference_spans` describes that physical order explicitly;
  neither NHWC flattening nor a single rectangular slice is correct there.
- After store/load queues join at 2346, an additional full **102,400-byte L2
  comparison** verifies the assembled tensor 206, including every stored
  channel group, against its independently pinned reference.

The mapper tracks source/output byte identities, current RQ selectors and LUT
residency across DMA and staging reuse. It requires complete, non-overlapping
cell coverage and locates the retained L2 tensor by provenance rather than a
hardcoded address. Partial workspace is **min(M, 256) × 128 bytes**: runtime
`runtime_pointwise_c32` reuses it for each row stripe. Treating it as M × 128
would falsely report overlap for this command. No firmware change is needed.

The **100 offline tests** pass, including a 257-row workspace-boundary case,
cross-group oracle ordering, corrupt fourth-group snapshot data, missing RQ,
stale LUT/input data and missing store completion. All seven earlier cases
still load. Seven real-case mutations (missing tile, wrong L2 location,
coverage, parent check, incomplete DMA join and wrong/unexpected spans) were
rejected. Logs: `resume-output-helper-tests.log` and `prepare-resume-output.log`.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/artifacts/dma_events.json \
  --snapshot hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-resume-c2f/command-2266.snapshot \
  --resume-c2f-output-checkpoints --timeout-cycles 2000000 --fail-fast \
  --output hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-resume-output
```

Evidence stays in `run-resume-output/`: `command_pmu.csv` (80 rows expected),
live actual/expected tiles, `resident-l2-actual.bin`, `resident-l2-expected.bin`,
`byte_exact.json`, `results.xml`, `sim.log` and `command-2346.snapshot` on
completed execution. The local launcher raises the auxiliary DMA trace cap to
50,000 rows; per-command PMU remains unfiltered. A later restart after 2346 still
needs RQ restoration or replay from a self-configuring boundary: its next Conv
would otherwise reuse the omitted hardware state loaded at 2273.

Verified on **2026-10-16**: all **27 checkpoints / 204,800 live bytes passed**,
including the cross-group tile at 2307. The complete retained tensor 206 also
matched all **102,400 L2 bytes**. PMU reports **189,129 whole-run cycles**,
**102,858 command-active cycles**, 80 command begin/end events, 80 CSV
rows/169 columns and no overflow. Cocotb reports PASS in **631.47 wall-clock
seconds**. Independent saved-byte, XML, snapshot integrity/provenance,
retained L2 and CSV-cycle reconciliation checks passed. This establishes
tensors 204/206, not the following downsample or full-model output.

### L2-fed grouped K3/S2 accumulation and C128 SiLU

`--resume-l2-downsample-checkpoints` extends the preceding gate from snapshot
**2266** through **2405**, 139 original commands. Replaying 2267–2346 keeps the
required RQ load 2273 inside the executed stream; it does not restore arbitrary
register state or rerun the large reset prefix. The new gate retains all 27
output checks and adds nine checks: **36 checkpoints / 307,200 live bytes**,
plus the complete **102,400-byte retained L2 comparison**. Timeout remains
2,000,000 cycles with fail-fast enabled.

| Finalized Conv commands | Tensor 208 channels | Compared bytes |
| --- | --- | ---: |
| 2352 / 2359 | 0–31 | 12,800 |
| 2367 / 2374 | 32–63 | 12,800 |
| 2382 / 2389 | 64–95 | 12,800 |
| 2397 / 2404 | 96–127 | 12,800 |
| 2405, SiLU tensor 210 | All 128 channels | 51,200 |

Command numbers, stripe rows and addresses are derived from source graph and
ABI geometry. Source PAD constants determine top/bottom/left/right padding;
the mapper does not infer it from a generic SAME rule. Each output stripe
requires both C32 input groups in order: initialize external pSUM, then
accumulate and requantize. Middle accumulation is also modeled and tested for
three input groups. Only finalized INT8 output becomes a numerical checkpoint;
intermediate pSUM arithmetic is not claimed byte-exact independently.

The mapper checks every staged non-padding pixel/channel against the retained
L2 parent's byte provenance, requires current replayed RQ setup, checks pSUM
continuity between input groups, excludes live workspace overlap, and requires
all four completed output groups before a freshly configured LUT can read
them. It also proves that the retained L2 input has not been overwritten.
Numerical Conv/SiLU oracles are independently generated reference tensors.
The current gate supports synchronous linebuffer jobs; async/unknown layouts
are rejected rather than silently misinterpreted.

The **110 offline tests** pass, including two/three-input-group chains,
missing/reordered partials, corrupt staged channel/padding origin, absent DMA
waits, overwritten pSUM, workspace alias, stale RQ/LUT state and clobbered L2.
All eight earlier cases still load; prior output/L2 oracles are unchanged.
Eight real-case mutations were rejected. Evidence logs are
`l2-downsample-helper-tests.log` and `prepare-l2-downsample.log`.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/artifacts/dma_events.json \
  --snapshot hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-resume-c2f/command-2266.snapshot \
  --resume-l2-downsample-checkpoints --timeout-cycles 2000000 --fail-fast \
  --output hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-l2-downsample
```

Evidence stays in `run-l2-downsample/`: original-numbered `command_pmu.csv`
(139 rows expected), live actual/expected bytes, retained L2 actual/expected,
`byte_exact.json`, `sim.log`, `results.xml` and, on completed execution,
`command-2405.snapshot`. RTL, firmware and the compiled NAI are unchanged.

Verified on **2026-10-16**: all **36 checkpoints / 307,200 live bytes passed**,
with zero mismatches. The retained L2 tensor matched all **102,400 bytes**.
PMU reports **513,419 whole-run cycles**, **387,480 command-active cycles**,
139 command begin/end events, 139 CSV rows/169 columns and no overflow.
Cocotb reports PASS in **1,119.44 wall-clock seconds**. Independent saved-byte,
XML, snapshot integrity/provenance, retained activation/L2 and CSV-cycle
reconciliation checks passed. This establishes tensors 208/210, including all
four output groups after two-input-group accumulation, not later C2f blocks
or full-model output.

### Four-group resident pointwise and SiLU from snapshot 2405

The existing `--resume-grouped-pointwise-checkpoints` mode now accepts the
verified C128 downsample activation. It selects exactly one supported parent
role at the saved boundary, requires matching PASS evidence and artifact/
reference-build identities, and rechecks all **51,200 retained parent bytes**
against both the pinned oracle and the current independent reference. It does
not permit an arbitrary tensor or memory snapshot to serve as numerical proof.

This gate executes **2406–2418 only**, 13 original commands. Its **five
checkpoints / 102,400 live bytes** are:

| Commands | Compared tensor | Bytes |
| --- | --- | ---: |
| 2407 / 2409 / 2411 / 2413 | Conv 211, channels 0–31 / 32–63 / 64–95 / 96–127 | 51,200 |
| 2415 | Complete C128 SiLU 213 | 51,200 |

Every pointwise job has an explicit matching RQ load at 2406/2408/2410/2412;
the first LUT at 2415 loads its constants itself. The helper derives C32 group
counts, selectors, raster rows and storage strides from shapes/ABI, rather
than constraining the gate to `[1,20,20,128]` or specific command numbers.
Supported resident layouts extend to eight C32 groups. Parent layout, live
input/OFM/pSUM overlap, stale input and complete per-cell coverage remain
strict. pSUM extent uses **min(M,256) × 128 bytes**, matching firmware stripe
reuse rather than conservatively treating every row as simultaneously live.

Pending next-layer constant DMA loads require the join at **2418**, the first
globally drained boundary. That setup is not numerically verified as next-layer
compute. Timeout is 1,000,000 cycles with fail-fast enabled. Snapshot 2418 does
not preserve the RQ state loaded at 2417; a later Conv restart must replay from
a supported self-configuring start or explicitly restore that configuration.

The **116 offline tests** pass. New tests cover four-group input/output and
full C128 LUT, missing/corrupted fourth-group data, overwritten RQ setup,
257-row pSUM bounds and unique supported parent selection. All nine earlier
cases still load; six real-case metadata mutations were rejected. Offline and
preparation logs are `resume-c128-pointwise-helper-tests.log` and
`prepare-resume-c128-pointwise.log`.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/artifacts/dma_events.json \
  --snapshot hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-l2-downsample/command-2405.snapshot \
  --resume-grouped-pointwise-checkpoints --timeout-cycles 1000000 --fail-fast \
  --output hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-resume-c128-pointwise
```

Evidence stays in `run-resume-c128-pointwise/`: actual/expected checkpoint
bytes, original-numbered `command_pmu.csv`, `byte_exact.json`, `sim.log`,
`results.xml` and `command-2418.snapshot`.

Verified on **2026-10-16**: all **five checkpoints / 102,400 live bytes passed**,
with zero mismatches. The retained parent also matched all **51,200 bytes**.
PMU reports **105,217 whole-run cycles**, **55,562 command-active cycles**,
13 command begin/end events, 13 CSV rows/169 columns and no overflow.
Cocotb reports PASS in **293.13 wall-clock seconds**. Independent saved-byte,
XML, snapshot integrity/provenance and CSV-cycle reconciliation checks passed.
This establishes Conv 211 and SiLU 213, not subsequent C2f or full-model output.
No RTL, firmware, compiled NAI or numerical-reference changes are involved.

### Complete multi-input-group C2f from snapshot 2405

The existing `--resume-c2f-chain-checkpoints` mode now also maps branches with
two or four C32 input/output groups. Source edges determine the two slices,
bottleneck count, residual operands and concat order; shapes and ABI determine
the spatial stripes and group strides. The already verified single-input-group
mapper, including its asynchronous compute handling, is unchanged. The new
multi-input-group mapper requires synchronous Conv commands and rejects
unsupported asynchronous compute rather than assuming completion.

For the third C2f, use **snapshot 2405**, not 2418: the latter omits RQ register
state from command 2417. Replay **2406–2604**, 199 original commands, including
the five verified head checks above. The complete gate contains **28
checkpoints / 460,800 live bytes**:

| Commands | Compared tensor | Bytes |
| --- | --- | ---: |
| 2407 / 2409 / 2411 / 2413 / 2415 | C128 pointwise Conv 211 and SiLU 213 | 102,400 |
| 2436 / 2442 / 2451 / 2457 | Conv 216, both C32 output groups and both spatial stripes | 25,600 |
| 2459 | Complete C64 SiLU 218 | 25,600 |
| 2480 / 2486 / 2495 / 2501 | Conv 219, both groups and stripes | 25,600 |
| 2502 / 2504 | Complete C64 SiLU 221 / general residual Add 222 | 51,200 |
| 2525 / 2531 / 2540 / 2546 | Conv 223, both groups and stripes | 25,600 |
| 2548 | Complete C64 SiLU 225 | 25,600 |
| 2569 / 2575 / 2584 / 2590 | Conv 226, both groups and stripes | 25,600 |
| 2591 / 2592 | Complete C64 SiLU 228 / biased residual Add 229 | 51,200 |
| 2604 | Complete C256 concat 230 after all copies finish | 102,400 |

For every K3/S1 stripe, staged pixels/channels must have provenance from the
correct branch or preceding activation/residual. External pSUM must follow
init/middle/final input-group order, retain the same stripe/output workspace
and advance weight addresses consistently. Only the final input-group job
creates an output checkpoint; partial sums are not presented as numerical
output proof. Each output cell is covered exactly once, with current RQ setup
and live workspace overlap checks. Whole-tensor LUT/Add and concat checks
preserve native group-major C32 order, not flattened NHWC order. Queued DMA
requires its matching completion before a consumer or final concat capture.

The **126 offline tests** pass, including two/four-input-group accumulation,
middle-partial corruption, staged channel corruption, missing final output,
native residual/concat order, both DMA wait directions and unsupported async
compute. All ten earlier cases still load. Six real-case mutations (missing
stripe, wrong output group, swapped concat sources, stale coverage, missing
parent verification and nonminimal boundary) are rejected. Preparation pins
the same NAI, firmware, input and independent numerical reference; the five
head oracles are unchanged. Logs are `resume-grouped-c2f-helper-tests.log`
and `prepare-resume-grouped-c2f.log`.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/artifacts/dma_events.json \
  --snapshot hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-l2-downsample/command-2405.snapshot \
  --resume-c2f-chain-checkpoints --timeout-cycles 2000000 --fail-fast \
  --output hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-resume-grouped-c2f
```

Evidence stays in `run-resume-grouped-c2f/`: expected/actual checkpoint bytes,
original-numbered `command_pmu.csv`, `byte_exact.json`, `sim.log`, `results.xml`
and `command-2604.snapshot`.

Verified on **2026-10-16**: all **28 checkpoints / 460,800 live bytes passed**,
with zero mismatches. The final snapshot retains the complete **102,400-byte
C256 concat**, independently rechecked against the pinned oracle. PMU reports
**830,279 whole-run cycles**, **661,072 command-active cycles**, 199 command
begin/end events, 199 CSV rows/169 columns and no overflow. Cocotb reports PASS
in **1,667.46 wall-clock seconds**. Independent actual/expected files, XML,
snapshot integrity/model identity and CSV-cycle reconciliation passed. This
establishes both C64 bottlenecks, both residual Add paths and concat 230;
it does not establish later layers or full-model output. RTL, firmware,
compiled NAI and numerical reference are unchanged.

### C256 concat output and streamed C128 SiLU from snapshot 2604

The unchanged `--resume-c2f-output-checkpoints` mapper accepts this verified
complete C256 parent. Preparation requires the prior PASS report and pinned
artifact/reference identities, rechecks all **102,400 retained parent bytes**
and compares them with the current independent reference. Replay begins with
fresh RQ setup at 2605, so no unsaved configuration register is assumed.

Execute **2605–2640 only**, 36 original commands, with a 2,000,000-cycle timeout
and fail-fast. There are **17 checkpoints / 102,400 live bytes**:

| Commands | Compared tensor | Bytes |
| --- | --- | ---: |
| 2606 / 2608 / 2610 / 2612 | Conv 231, all four C32 output groups after eight-input-group pointwise accumulation | 51,200 |
| 2613 / 2617 / 2619 / 2621 / 2623 / 2625 / 2627 / 2629 / 2631 / 2633 / 2635 / 2637 / 2639 | Streamed SiLU 233, every output byte once | 51,200 |

Also compare all **51,200 retained L2 bytes** at offset 102,400 after store
completion. SiLU tiles 2621/2627/2633 cross C32-group boundaries; explicit
reference spans preserve physical group-major order rather than NHWC flattening.
The last tile contains 2,048 bytes, the preceding twelve contain 4,096 each.
Boundary 2640 joins the final store before saving the next memory snapshot.

All **129 offline tests** pass. Added tests cover eight input groups/four output
groups, complete native L2 output, eighth-group parent corruption, missing
fourth output and complete C256 snapshot-parent verification. All eleven
earlier cases still load; five real-case mutations (missing output group,
wrong cross-group span, missing parent check, wrong L2 group stride and
nonminimal boundary) are rejected. Logs are `resume-c256-output-helper-tests.log`
and `prepare-resume-c256-output.log`.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/artifacts/dma_events.json \
  --snapshot hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-resume-grouped-c2f/command-2604.snapshot \
  --resume-c2f-output-checkpoints --timeout-cycles 2000000 --fail-fast \
  --output hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-resume-c256-output
```

Evidence stays in `run-resume-c256-output/`: expected/actual checkpoint and L2
bytes, `command_pmu.csv`, `byte_exact.json`, `sim.log`, `results.xml` and
`command-2640.snapshot`.

Verified on **2026-10-16**: all **17 checkpoints / 102,400 live bytes passed**,
with zero mismatches; the complete retained L2 output matched all **51,200
bytes**. PMU reports **158,229 whole-run cycles**, **95,400 command-active
cycles**, 36 command begin/end events, 36 CSV rows/169 columns and no overflow.
Cocotb reports PASS in **419.87 wall-clock seconds**. Independent saved-byte,
retained-L2, XML, snapshot integrity/model identity and CSV-cycle reconciliation
checks passed. This establishes Conv 231 and SiLU 233, not later layers or
full-model output. RTL, firmware, compiled NAI and numerical reference are unchanged.

### Four-input-group L2 downsample to C256 from snapshot 2604

The unchanged `--resume-l2-downsample-checkpoints` mapper also supports the
next **C128-to-C256 K3/S2 Conv**. It resolves explicit source PAD values
`[top,bottom,left,right] = [1,1,1,1]`; input/output geometry is derived from the
source graph and ABI, not fixed model dimensions. The complete native L2
parent is SiLU 233 (`20×20×128`), and outputs are Conv 235 / SiLU 237
(`10×10×256`).

Replay **2605–2731**, 127 original commands, from **snapshot 2604** rather than
2640: the downsample's first RQ setup at 2615 is not stored in a memory-only
snapshot. Repeating the short, self-configuring output prefix supplies that
setup and preserves its 17 verified checkpoints/oracles. Timeout is 2,000,000
cycles with fail-fast. The whole gate checks **26 checkpoints / 153,600 live
bytes**, plus all **51,200 retained L2 bytes**:

| Commands | Compared tensor | Bytes |
| --- | --- | ---: |
| Previous 17 checks through 2639 | Complete pointwise Conv 231 and streamed SiLU 233 | 102,400 |
| 2653 / 2664 / 2675 / 2686 / 2697 / 2708 / 2719 / 2730 | Conv 235, all eight C32 output groups | 25,600 |
| 2731 | Complete native C256 SiLU 237 | 25,600 |

Each finalized Conv group covers all 100 output rows only after four input
groups follow init/middle/middle/final pSUM accumulation. The mapper checks
staged source pixels/channels, weight increments, pSUM continuity, current RQ,
live workspace overlap and exactly-once output coverage. LUT capture requires
every output group and fresh configuration; the L2 parent must remain intact.

All **132 offline tests** pass. Added coverage exercises four-input/eight-output
groups, full C256 LUT, premature finalization, missing last group, fourth-input
staging corruption and stale eighth-group RQ. All twelve earlier cases still
load; six real-case metadata mutations (missing last group, wrong channel,
wrong padding origin, stale coverage, missing parent check and nonminimal
boundary) are rejected. Prior prefix and L2 oracles are unchanged. Logs are
`l2-c256-downsample-helper-tests.log` and `prepare-l2-c256-downsample.log`.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/artifacts/dma_events.json \
  --snapshot hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-resume-grouped-c2f/command-2604.snapshot \
  --resume-l2-downsample-checkpoints --timeout-cycles 2000000 --fail-fast \
  --output hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-l2-c256-downsample
```

Evidence stays in `run-l2-c256-downsample/`: expected/actual checkpoint and L2
bytes, `command_pmu.csv`, `byte_exact.json`, `sim.log`, `results.xml` and
`command-2731.snapshot`.

Verified on **2026-10-16**: all **26 checkpoints / 153,600 live bytes passed**,
with zero mismatches; retained L2 matched all **51,200 bytes**. PMU reports
**569,501 whole-run cycles**, **445,834 command-active cycles**, 127 command
begin/end events, 127 CSV rows/169 columns and no overflow. Cocotb reports PASS
in **1,242.25 wall-clock seconds**. Independent actual/oracle files, XML,
snapshot integrity/model identity, retained L2 and CSV-cycle reconciliation
passed. All **25,600 bytes of the retained C256 SiLU 237** in snapshot 2731
were also rechecked against the pinned oracle. RTL, firmware, compiled NAI
and numerical reference are unchanged.

### Single continuation for all remaining commands

At the user's request, run **2732–5254 in one invocation**, not further small
gates. `--remaining-graph` starts from the verified snapshot 2731, executes all
**2,523 remaining logical commands**, and compares all **176,400 final INT8
output bytes** (`[1,84,2100]`) with the pinned Neural-AI hybrid reference.
Timeout is **50,000,000 cycles**; per-command PMU covers the entire suffix.

The mode retains strict snapshot/PASS/artifact/reference-build provenance and
rechecks the complete retained parent against both its saved oracle and the
current reference. It also requires a self-configuring pointwise/SiLU prefix:
fresh RQ begins at 2732 and the first LUT at 2748 loads its constants itself.
No unsaved hardware register or LUT contents are assumed. Eight Conv output
groups at 2733/2735/2737/2739/2741/2743/2745/2747 and whole C256 SiLU at 2748
remain live byte-exact checkpoints (**51,200 bytes**), with fail-fast enabled.
The remaining later intermediates are not individually compared by this run.

`remaining_graph=true` and `full_graph=false` distinguish this continuation
from the reset regression below. The suffix uses the existing expanded
logical-command range extractor; all original commands/operands and their
order are checked unchanged, but affine-loop encoding is not retained.
Therefore its PMU is suffix-replay evidence, not a measurement of the original
compacted full stream from reset. Final output PASS requires both successful
command execution and zero byte mismatches, never command completion alone.

All **136 offline tests** pass. Scope tests reject partial execution, missing
snapshot/prefix metadata, ambiguous reset/continuation claims, incomplete
output oracles and a single-byte output error. All thirteen earlier cases
still load; eight real-case mutations (missing final command, stale prefix
boundary, bad evidence hash, missing parent check/oracle/output group, partial
oracle and reset claim) are rejected. Preparation independently checks the
entire 2,523-command suffix against the pinned logical stream. Logs are
`remaining-2732-5254-helper-tests.log` and `prepare-remaining-2732-5254.log`.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/artifacts/dma_events.json \
  --snapshot hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-l2-c256-downsample/command-2731.snapshot \
  --remaining-graph --timeout-cycles 50000000 --fail-fast \
  --output hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-remaining-2732-5254
```

Evidence stays in `run-remaining-2732-5254/`: prefix actual/expected files,
`output-actual.bin`, `output-expected.bin`, `command_pmu.csv` (2,523 rows
expected), `byte_exact.json`, `sim.log`, `results.xml` and
`command-5254.snapshot`. No RTL, firmware, compiled NAI or numerical-reference
changes are involved.

### Complete suffix result: command completion is not numerical PASS

Reviewed on **2026-10-19**: firmware completed **2,523 / 2,523 commands**, and all
nine prefix checks (**51,200 bytes**) passed. Nevertheless, the final comparison
is **FAIL: 7,929 / 176,400 bytes differ**, 95.5051% exact, maximum absolute
error 222. Preserve this failure; do not relabel it as inference PASS.

| Output channels | Mismatches | Maximum absolute difference |
| --- | ---: | ---: |
| 0 / 1 / 2 / 3 (boxes) | 1,781 / 2,013 / 2,016 / 2,093 | 67 / 77 / 200 / 222 |
| 4 / 10 / 11 / 63 (four class channels) | 10 / 7 / 1 / 8 | At most 2 |
| Remaining 76 class channels | 0 | 0 |

PMU reports **10,087,829 whole-run cycles**, **8,441,896 command-active cycles**,
2,523 command begin/end events, 2,523 CSV rows/169 columns and no overflow.
CSV cycle sums reconcile with command-active PMU. Cocotb correctly reports
FAIL in **15,694.45 wall-clock seconds**. Final actual bytes agree with the
saved snapshot output; the snapshot is intact, but numerical failure means it
is not a verified restart parent.

Offline analysis of the surviving TCDM bytes narrows the problem upstream:

- For the 1,600-location box head, 48,674 / 51,200 surviving C32-group-1 logits
  differ from source Conv tensor 378, maximum difference 47. The saved group-1
  input of that pointwise Conv also differs from activation 377 in 47,568 /
  51,200 bytes, maximum difference 237.
- Applying the deployed exp/reciprocal LUT and requant arithmetic to those
  actual saved logits reproduces **3,200 / 3,200** saved DFL output bytes for
  sides 2/3 exactly. This is evidence against blaming the observed large error
  on AFU DFL arithmetic; it does not certify overwritten inputs or other jobs.
- Applying the compiled preceding SiLU LUT to reference Conv 375 matches all
  **102,400** reference activation-377 bytes. The compiled three class LUTs
  also reproduce all **168,000** reference class bytes. Thus the 26 class-output
  differences are not explained by LUT-table generation alone.
- Even with correct reference logits, fused AFU DFL arithmetic differs from
  the hybrid reference's quantized Softmax/Conv/Mul sequence by up to 4 INT8
  units in this input. This smaller independent numerical-contract issue must
  not be hidden by relaxing byte-exact checks or replacing the pinned oracle.

Evidence is `offline-dfl-lut-analysis.json` in the failed run and the local
`offline-dfl-tail.log`. Analysis also found that exp-LUT entries 1–127 are
generated as positive signed-byte deltas rather than wrapped deltas -255…-129;
the reference head distributions in this input do not exercise those entries,
so that latent compiler issue does not explain this failure. No compiler
datapath, firmware or RTL fix has been applied on the basis of these findings.

### Completion-aware detection-head diagnostic continuation

Replay the same **2732–5254** suffix once with the existing nine validated
prefix checks, final-output oracle and PMU, adding **115 diagnostic checkpoints
/ 907,200 bytes** covering every output cell of all **18 detection-head Convs**.
Their source mapping uses unique complete first-C32 adjusted-bias signatures
from the independent reference and emitted qparams, not layer-number guesses.
Output groups/stripes follow ABI geometry and exactly-once source coverage.

The opt-in `--diagnostic-checkpoints` manifest pins model/source/input/reference
identities, each command body, each oracle and its geometry. These observations
are recorded separately in `diagnostic_checkpoints`; they do not replace the
validated prefix or the final-output comparison and do not prove every other
intermediate tensor. Partial sums are never treated as requantized output.
Of three async jobs in the first head, two are partial accumulations; only
finalized submit **4356** is observed, at matching wait **4357**. Early submit
capture, ambiguous waits and changed command bodies are rejected.

Run with fail-fast: the first bad observation saves its actual bytes,
`failure-tcdm.bin`, command/layer/tensor identity and PMU through that command,
then reports FAIL. This permits staged-input/Conv/RQ diagnosis at the first
observed divergence. A passing head observation does not authorize an RTL edit.

All **143 offline tests** pass. All fourteen earlier cases still load. Five
real-case mutations (manifest hash, oracle hash, geometry, premature async
capture and command hash) are rejected. Preparation uses the same pinned NAI,
firmware, input, reference build and snapshot 2731; no numerical oracle is
weakened. Logs are `remaining-head-debug-helper-tests.log` and
`prepare-remaining-head-debug.log`.

```sh
cd /home/dev01/neural-ai
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --compiler-root /home/dev01/neural-compiler \
  --manifest hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/artifacts/dma_events.json \
  --snapshot hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-l2-c256-downsample/command-2731.snapshot \
  --remaining-graph \
  --diagnostic-checkpoints hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/head-conv-diagnostic-oracles/observations.json \
  --timeout-cycles 50000000 --fail-fast \
  --output hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-remaining-head-debug
```

Evidence stays in `run-remaining-head-debug/`. The same all-command invocation
may stop early on a numerical failure; completion/output correctness is pending.

### Pointwise cascade L2 staging correction (2026-10-20)

The detection-head replay stopped at **command 4314, tensor 345, source layer
130**, with **2,909 / 3,200 mismatches**, maximum absolute difference 32.
All nine prefix checks passed. Its PMU CSV contains **1,583 rows / 169 columns**
for commands 2732–4314, summing to **4,594,768 command cycles**. Fail-fast
preserved `failure-tcdm.bin`, actual output bytes and `byte_exact.json`; it did
not reach a final aggregate PMU or create a passing end snapshot.

Offline recomputation in `offline-first-head-analysis.json` establishes:

- The complete input tensor 337 differs in **18,618 / 25,600 bytes**.
- All **73,728 packed weight bytes** and adjusted biases match the source.
- All **3,200 INT32 partial sums** match raw MACs over the actual input's first
  seven C32 groups.
- Full host Conv/requant on that actual input reproduces **all 3,200 RTL
  output bytes**; the same calculation on reference input reproduces the oracle.

Thus this particular Conv is not evidence for an RTL fix. Auditing its upstream
commands found **108 PointwiseC32 jobs with direct L2 output references**,
starting at command **2972**. Firmware programs local systolic ports; its
trusted path does not repeat the host's region checks. The standalone compiler
path staged L2 endpoints correctly, but `AppendC32MatrixStripe` did not.

The correction stages each spilled full-width Y stripe in separate TCDM input
and output ranges, uses typed DMA loads/stores and preserves rolling Y offsets,
original C32 group strides and output groups. Both workspace sizing passes now
reserve the same staging, including at least one full spatial row. Semantic IR
rejects nonlocal IFM, OFM or used partial-sum references (including nonzero local
binding indices), before trusted firmware can execute them. No RTL, firmware,
ABI or reference-arithmetic change is made.

Verification:

- Focused ASan tests: **3 cases / 2,135 assertions PASS**, including output-only
  and two-ended L2 cascade spills, non-aliasing local buffers and complete DMA
  byte coverage. The broader `Neural-AI*` run reports **158 passing / 4 known
  baseline failing cases**, 45,314 assertions; the baseline exclusions described
  above are unchanged, not counted as new passes.
- Offline replay/oracle tests: **156 PASS**, including changed command prefixes,
  constants/bindings/qparams/layout, stale compatibility proofs, nonlocal compute
  references and outstanding-DMA snapshot rejection.
- New NAI SHA256:
  `45b2c890185e06d414fe06c56392309e6f7292a6c5b972f09d50e5d3436fa972`;
  **5,513 commands**, no CPU operators, **512,000 TCDM bytes**, no remaining
  nonlocal pointwise jobs. The +259 commands are required DMA/wait work, not a
  performance optimization. The old faulty suffix is not a valid timing baseline.

The current compiler estimate is **9,125,722 cycles / 18.251444 ms at 500 MHz**,
54.79 inferences/s; these are estimates, not verified RTL performance. The
Micro-MobileNet plan's **347,992-cycle** figure is for its different 96×96 graph
and remains reference-only; do not treat it as a 1:1 YOLO target.

The first changed logical command is **2972**. Prefix **1–2731**, constants,
qparams, binding descriptors, all other section payloads and memory requirements
are byte-identical. `--allow-compatible-snapshot` proves these conditions on
preparation and every case load, including both drained restart boundaries.
The original snapshot CRC, parent case and PASS report remain untouched and are
still verified against the **original** model; firmware/source/input/reference
identities and the complete retained parent are checked normally. The new case
and report explicitly carry `snapshot_prefix_compatibility`; this is not a new
reset-regression claim or permission to relabel the failed final snapshot.

`prepare_conv_diagnostics.py` derives unique adjusted-bias source mappings,
finalized ABI output geometry and independent TensorFlow-derived oracles for all
**42 remaining source Convs**. It validates exactly-once complete group coverage,
omits eight already-proven prefix Conv checks only after comparing their oracle
bytes, and prepares **313 extra observations / 1,726,400 bytes**. Pointwise
outputs are observed locally before DMA reuse; asynchronous Conv outputs are
observed only at their matching completion wait.

The prepared all-command replay is **2732–5513**, timeout **50,000,000 cycles**,
with nine existing prefix checks, all 313 extra observations, unchanged final
176,400-byte oracle, fail-fast and per-command PMU. Artifacts are under
`artifacts-pointwise-staging-20261020/`, oracles under
`pointwise-staging-conv-oracles/`, and the prepared case/logs under
`run-remaining-pointwise-staging/`, all below the existing evidence root.
Preparation logs are `prepare-pointwise-staging-base.log`,
`prepare-pointwise-staging-oracles.log` and `prepare-pointwise-staging-debug.log`.
**No numerical PASS is claimed for the repaired suffix yet.** The independent
DFL numerical-contract discrepancy described above is still unresolved.

Reproduction after pinning the repaired NAI and unchanged ELF (choose fresh
output directories for another run):

```sh
cd /home/dev01/neural-ai
evidence_root=hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --manifest "$evidence_root/artifacts-pointwise-staging-20261020/dma_events.json" \
  --snapshot "$evidence_root/run-l2-c256-downsample/command-2731.snapshot" \
  --remaining-graph --allow-compatible-snapshot --fail-fast \
  --timeout-cycles 50000000 --output "$evidence_root/run-remaining-pointwise-staging-base"
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/prepare_conv_diagnostics.py \
  --compiler-root /home/dev01/neural-compiler \
  --case "$evidence_root/run-remaining-pointwise-staging-base/case.json" \
  --output "$evidence_root/pointwise-staging-conv-oracles"
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --manifest "$evidence_root/artifacts-pointwise-staging-20261020/dma_events.json" \
  --snapshot "$evidence_root/run-l2-c256-downsample/command-2731.snapshot" \
  --remaining-graph --allow-compatible-snapshot --fail-fast --timeout-cycles 50000000 \
  --diagnostic-checkpoints "$evidence_root/pointwise-staging-conv-oracles/observations.json" \
  --output "$evidence_root/run-remaining-pointwise-staging"
```

### AFU binary compact-slice alignment correction (2026-10-21)

The pointwise-staging replay **completed commands 2732–5513**. All **322 live
checks / 1,777,600 compared bytes PASS**, including all 42 observed remaining
source Convs. The final **168,000 class bytes are exact**, but **7,047 / 8,400
box bytes differ** (maximum absolute difference 137). Do not treat command
completion or passing Convs as full inference PASS.

Evidence: `run-remaining-pointwise-staging/byte_exact.json`, `results.xml`,
`output-actual.bin`, `output-expected.bin` and `command-5513.snapshot` under the
existing evidence root. The snapshot integrity is valid, but the final numerical
FAIL means it is **not** an accepted restart parent. Simulation time was
10,279,178 ns, wall time 16,911.29 s. PMU reports **10,243,691 whole-run cycles**,
**8,481,630 command-active cycles**, no overflow and 2,782 begin/end events.
`command_pmu.csv` contains **2,782 rows / 169 columns**, commands 2732–5513;
both its `cycle` and `command_active_cycles` sums equal the active PMU total.
These are suffix measurements on the still-faulty tail, not corrected full-model
performance or a direct Micro-MobileNet comparison.

Offline analysis (`offline-box-tail-analysis.json`, produced by the local
`analyze-pointwise-staging-tail.py`, with log `offline-pointwise-staging-tail.log`)
separates the remaining problems:

- The last AFU DFL command **5493** matches all **6,400 surviving RTL bytes**
  when evaluated on the verified logits using its emitted exp/reciprocal tables
  and requant. Differences from the unchanged hybrid reference are 333/400,
  1,042/1,600 and 2,489/6,400 bytes across the three heads, maximum 4/2/2.
- The emitted coordinate constants match all **4,200 TFLite bytes**, and the
  four emitted binary quantizations plus final LUT, when fed reference DFL
  distances, reproduce **all 8,400 reference box bytes** on the host.
- Command **5506**, layer 168, reads the right/bottom DFL slice at TCDM offset
  **14,216**, whose low five bits equal **8**. The standalone AFU binary backend
  reads aligned 32-byte beats: it starts at **14,208**, not at that slice.
  This is a missing compiler alignment/staging contract, not authorization to
  change the RTL interface.
- Replaying the emitted arithmetic with the backend's aligned addresses and
  fused AFU DFL distances reproduces **all 8,400 actual RTL box bytes**, and
  every surviving right/bottom, width/height, center and final-LUT buffer.
  With the correct byte address, the predicted residual is only **3,465 box
  mismatches, maximum 7**, attributable to DFL's separate quantization contract.
  This last figure is a host prediction, not a repaired RTL simulation result.

The compiler now stages unaligned compact-plane Add/Sub views through aligned
TCDM buffers, using the existing tiled DMA path. Each LHS/RHS/output slot has a
separately **32-byte-rounded stride**, including short tiles; aligned local
views remain direct. A shared sizing helper is used in scheduler reservation and
emission, and generation checks the reserved span. Semantic IR rejects nonlocal,
nonzero-index or unaligned native `AFUBinaryQuant` references. No RTL, firmware,
ABI or numerical oracle is changed.

Verification so far:

- Focused ASan: **4 cases / 500 assertions PASS**. The new compiler test covers
  widths **3, 16, 17, 32, 2100**, plane starts **0 and 2**, aligned controls,
  complete staged byte coverage, short-slot padding and the 4,096+104 split.
  Semantic tests reject invalid LHS, RHS and output ports for lengths 6/34/4200.
- Broad `Neural-AI*`: **160 cases PASS / 4 unchanged baseline failures**,
  **45,708 assertions**. The known four failures remain recorded above.
- Offline replay suite: **157 PASS**. The generic reset runner now permits an
  absent, unused L2 temporary binding and constructs exactly the declared binding
  table, without adding a fake binding to the compiled model. Missing input or
  output, duplicate descriptors and nonzero indices remain rejected.
- Recompiled YOLO: **5,520 commands**, zero CPU ops, **512,000 TCDM bytes**;
  SHA256 `ac9a410bd25221e1d8362211e1f30c8baae7a7b783756d9cc17bb87eeeeb464d`.
  The first changed command is 5504; all emitted native binary addresses are
  aligned. This artifact has not yet passed full numerical simulation.

First run the focused cluster regression, then return to the full YOLO suffix.
Its local directory is
`hw/rtl/cluster/tb/sim/test_afu_compact_slice_20261021/`: `build_fixture.py` creates
a single-input INT8 `Add -> plane slice -> general Add` TFLite graph. The slice
starts at byte 4,200 within its compact parent; output is 4,200 bytes. Different
consumer output scale forces general AFU quantization, not equal-scale fast Add.
The compiler emits **14 commands**, including two general jobs of **4,096 and
104 bytes**, with **29,472 TCDM bytes**, zero CPU ops, NAI SHA256
`ae5fffccd9c19ebf115dd2febfc386ac92d61fb15f429a90741fdcb92d9105d7`.
The unchanged firmware and Vtop are pinned. `run-general/case.json` starts from
reset, requires all 4,200 final bytes to match the independent reference, records
per-command PMU and has a **1,000,000-cycle timeout**. Preparation:

```sh
cd /home/dev01/neural-ai
test_root=hw/rtl/cluster/tb/sim/test_afu_compact_slice_20261021
PYTHONDONTWRITEBYTECODE=1 python3 sw/test/compiler_runtime/yolo_reference_replay.py \
  --source "$test_root/compact_slice_general_2100.tflite" \
  --manifest "$test_root/artifacts/dma_events.json" \
  --from-reset --full-graph --consumer-tiles 0 --timeout-cycles 1000000 \
  --output "$test_root/run-general"
```

The local launcher is `launch-general.sh`; it refuses an overlapping Vtop or
overwriting existing evidence.

Focused cluster result, reviewed **2026-10-21**: **PASS, 14/14 commands,
0/4,200 output mismatches**. `results.xml` also reports PASS, and the saved
`command-14.snapshot` output matches both the actual file and pinned reference
with SHA256 `dec4707eb626fc403746f79d66ffb036715b141cb2c16e2a3d8e39c46b252986`.
Simulation time is **98,661 ns**, wall time **151.89 s**. PMU records
**63,335 whole-run cycles**, **13,468 command-active cycles**, 14 begin/end
events and no overflow. The 14-row/169-column `command_pmu.csv` reconciles with
the active PMU total. General AFU commands 8 and 12 take **902 and 416 command
cycles**, respectively, including firmware configuration; they are not pure
arithmetic latencies. The other six local staging commands take **9,414 command
cycles** combined. Do not compare this small reset test directly to full YOLO or
Micro-MobileNet inference performance.

Next gate: use `artifacts-binary-alignment-20261021/` and continue original
snapshot 2731 through corrected command **5520**. The unchanged-prefix proof,
all noncommand section payloads, memory requirements, firmware and original
parent-byte evidence remain mandatory; do not use failed snapshot 5513. Fresh
oracles are stored in `binary-alignment-conv-oracles/`, with the prepared case
and all runtime evidence in `run-remaining-binary-alignment/` below the existing
YOLO evidence root. This gate retains the nine prefix checks, all 313 additional
Conv observations, the original final-output oracle, per-command PMU and a
50,000,000-cycle timeout. A final DFL discrepancy must still report FAIL rather
than accepting a tolerance or replacing that oracle. This remains a suffix
continuation, not a full reset regression.

### Intermediate full-memory snapshots (2026-10-21)

Previously, the 322 live observations saved only each tensor's output bytes;
the runner saved a complete memory snapshot only at the run's end. That caused
the avoidable return to snapshot 2731 after a later failure. The corrected replay
now requests **56 intermediate full-memory snapshots**, beginning at **2733**.
Other selected boundaries follow completed Convs/queue drains and surround
DFL at **4743/4744, 5035/5036, 5492/5493**, then the box arithmetic at
**5505, 5506, 5507, 5508, 5511, 5512, 5513, 5514, 5515**. No snapshot is taken
while an ABI DMA/systolic submission is outstanding. The end boundary 5520 is
still saved automatically by the existing final-output path.

The opt-in CLI is `--snapshot-boundaries 2733,2748,...,5515`. The complete list
is pinned in `run-remaining-binary-alignment/case.json`; omission preserves the
previous runner behavior. Boundaries must be unique, ascending and strictly
inside the requested range. Both preparation and case loading validate them.

For each requested boundary, the exact command-end PMU callback pauses the
testbench clock, reads **524,288 TCDM bytes + 204,800 L2 temporary bytes + 176,400
output bytes**, writes the standard compressed `command-N.snapshot`, verifies
its CRC/model identity and all three memory images, then resumes at the held
clock level. Capture asserts that neither the PMU cycle counter nor TCDM changed
during the L2 reads. This changes no RTL/firmware and adds no clock cycles to
the measured commands; simulation time and wall time can include capture cost.

Each `command-N.snapshot.json` sidecar pins the snapshot and originating case
hashes, model/source/ELF/input/reference identities, memory sizes and the
comparison descriptors/hashes available at that boundary. The main report's
`intermediate_snapshots` list references each sidecar. Logs contain
`NEURAL-AI SNAPSHOT command N: integrity=... comparisons=... path=...`.
Files are not overwritten. An error reports ERROR rather than publishing a
passing comparison verdict, and a completed run requires every requested file.

Memory integrity and numerical scope are deliberately separate:

- `integrity=VERIFIED` means model/CRC/full-memory readback checks passed.
- `PASS_LISTED_CHECKS` means only the listed reference observations passed;
  `FAIL_LISTED_CHECKS` and `UNVERIFIED` remain explicit alternatives. It does
  **not** establish every memory byte or final inference output.
- Earlier immutable evidence survives a later final-output FAIL. Hardware
  registers are **not** saved: reuse still needs a self-configuring suffix or
  validated configuration restoration, not just a drained async queue. Do not
  forge a parent PASS report or edit its command boundary to bypass that check.

Offline verification: **167 tests PASS**, including ten new tests for invalid
lists/ranges, outstanding DMA/systolic jobs, missing/duplicate byte checks,
memory sizes, model/case identity, no-overwrite policy, snapshot encode/readback,
preserved prefix evidence after final failure, and clock pause/resume/error paths.
Live integration subsequently captured all 56 requested boundaries, with no
clock-advance/TCDM-coherence assertion failures. Independent post-run verification
checked snapshot CRC/model identity, safe boundaries, all memory sizes, sidecar
hashes and observation scope. Total compressed snapshot size is **18,443,037
bytes**, excluding sidecars and the final command-5520 snapshot. These files
permit offline comparison at the saved boundaries even though final inference
fails; they do not imply arbitrary register-state restoration is supported.

### Alignment-corrected suffix result and DFL isolation (2026-10-21)

The corrected run in `run-remaining-binary-alignment/` completed original
commands **2732–5520**, **2,789 / 2,789**, from the verified 2731 parent. All nine
prefix checks / 51,200 bytes and all 313 additional Conv observations /
1,726,400 bytes pass. Independent comparison of the saved actual/oracle files
confirms all **1,777,600 observed bytes**. The original final oracle remains
unchanged (SHA256 `351f47bbfeb19ff821535f6895106f94870ffbdd5bcfd5f4da32fda362e28fe7`).

Final output correctly reports **FAIL: 3,465 / 176,400 mismatches**, 98.0357%
exact, maximum absolute error 7. Every mismatch is in the 8,400 box bytes;
all **168,000 class bytes** are exact. Box channel mismatch counts are
545 / 593 / 1,291 / 1,036, with maximum differences 3 / 2 / 7 / 6. This agrees
with the prediction made before the alignment correction was simulated; the
previous additional address-alignment error no longer explains any output byte.

PMU reports **10,263,621 whole-run cycles**, **8,497,586 command-active cycles**,
2,789 begin/end events and no overflow. `command_pmu.csv` has exactly 2,789
consecutive command rows and 169 columns; cycle sums equal command-active PMU.
Cocotb reports FAIL in **17,884.48 wall-clock seconds** (10,299,775.18 ns simulated).
The final snapshot output equals `output-actual.bin`, SHA256
`0d3063a0062f245cd95e2b172c5c166b8112f0a912d3e67949ae1818996366b3`.

Offline analysis uses snapshots immediately before and after each DFL command,
not logits inferred from overwritten final memory:

| DFL command | Locations | Input vs reference | Fused arithmetic vs RTL output | RTL output vs unchanged hybrid reference |
| --- | ---: | --- | --- | --- |
| 4744 | 100 | 6,400 / 6,400 exact | 400 / 400 exact | 333 mismatches, maximum 4 |
| 5036 | 400 | 25,600 / 25,600 exact | 1,600 / 1,600 exact | 1,042 mismatches, maximum 2 |
| 5493 | 1,600 | 102,400 / 102,400 exact | 6,400 / 6,400 exact | 2,489 mismatches, maximum 2 |

The independent host calculation reads the deployed exp/reciprocal tables and
requant parameters. All **8,400 DFL output bytes** agree, and their concatenation
matches the tensor saved at command 5505. The five binary jobs **5506, 5508,
5512, 5514, 5515** then match their respective snapshots exactly, including the
4,096-element tile and 104-element tail of the staged Add. Their command costs
are 794 / 902 / 416 / 794 / 794 cycles. Replaying the full box tail from these
actual DFL distances reproduces every final RTL box byte. In the counterfactual
with reference DFL distances, reconstructing the already-copied staging view as
well as later copies reproduces all **8,400 original oracle box bytes**. This
isolates the observed discrepancy upstream of the final Add/Sub and LUT chain.

The arithmetic contracts are not identical:

- The source graph quantizes Softmax probabilities to INT8, projects them with
  quantized Conv weights, requantizes that Conv output, then performs a separate
  quantized per-head Mul. The projection's raw weights are
  `[0,8,17,25,34,42,51,59,68,76,85,93,102,110,119,127]`, scale approximately
  0.11811024, zero point 0: they approximate bins 0–15 rather than represent all
  bin indices exactly. For example, bin 1 dequantizes to approximately 0.94488.
- `RewriteYoloDfl16` accepts weights within half a weight-quantization step of
  the ideal bins and replaces the projection with ideal integer-bin weights.
  `FuseDistributedDflBoxScale` folds the later Mul into DFL output quantization.
  The AFU computes a Q8 expectation using exp/reciprocal LUTs, followed by one
  output requantization; it does not materialize the original Softmax and Conv
  INT8 rounding boundaries. Better reciprocal precision alone would not restore
  those omitted boundaries or the original projection weights.
- The latent wrapped-delta LUT issue remains separate: minimum logits-minus-max
  are -44 / -43 / -48 here; no delta below -128 is exercised by these three heads.

Evidence is `offline-dfl-snapshot-analysis.json` in the run directory, with the
offline invocation log at `offline-binary-alignment-analysis.log` in its parent.
The local, uncommitted analysis script is
`neural-compiler/output/yolo320-binary-alignment-20261021/analyze_completed_replay.py`.
It validates snapshot/sidecar identities, all saved comparisons, PMU reconciliation,
DFL input placement, every DFL output and the downstream binary steps; it neither
starts a simulator nor modifies the pinned reference runtime or output oracle.

Next requires an explicit numerical-contract decision: retain fused DFL and add
an independently tested, separately identified AFU DFL reference while preserving
the TFLite/hybrid baseline, or preserve the original quantized operator sequence
and investigate an exact lowering. Neither oracle replacement nor a new RTL
change is authorized by the observed arithmetic agreement. No further simulation
is needed just to repeat this unchanged failure.

## Deferred regression: full YOLO output

After the bounded gates pass, run every logical command from reset using the **newly verified pinned NAI**,
including its encoded affine loops, rather than a flattened replacement stream.
Retain the verified early checkpoints, require all 176,400 final output
bytes to equal the pinned Neural-AI hybrid reference, and collect PMU for every
command. Use a 50,000,000-cycle execution timeout and the existing 100,000-cycle
progress log. First resolve the diagnosed compiler/DFL discrepancies and verify
the repaired suffix; do not reuse a faulty model just to complete this gate.

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
  --manifest hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/artifacts/dma_events.json \
  --from-reset --consumer-tiles 4 --full-graph --timeout-cycles 50000000 \
  --output hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-full-byte-exact
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
