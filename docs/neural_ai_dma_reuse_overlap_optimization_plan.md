# Neural-AI DMA Reuse and Overlap Optimization Plan

## 1. Purpose

Reduce real memory traffic and engine idle time in the Neural-AI schedule before
reducing the number of ABI commands. Command compaction is deliberately the
last stage: it must describe the final memory schedule rather than freeze an
unoptimized one.

The plan is compiler/firmware-first. Neural-AI RTL is frozen unless the user
explicitly approves an RTL change. Testbench, cocotb, estimator, and compiler
changes are allowed within the existing workflow.

## 2. Current baseline

The selected YOLO320 package currently contains 3,910 ABI commands. The largest
groups are:

| Command type | Count | Estimated cycles |
| --- | ---: | ---: |
| DMA2D | 1,660 | 10.26M |
| LINE_BUFFER_JOB | 592 | 1.38M |
| DMA3D | 461 | 0.54M |
| RQ_LOAD | 413 | 0.53M |
| DMA1D | 313 | 0.34M |
| POINTWISE_C32 | 239 | 1.50M |

The command-level PMU tracer records command header metadata and deltas for
cycle, Snitch, systolic, Spatz, iDMA, AFU, and TCDM counters. The baseline must
be preserved before each optimization increment.

## 3. Optimization order

```text
PMU baseline
  -> residency and DMA reuse
  -> rolling/zero-copy and ping-pong buffers
  -> asynchronous DMA/compute overlap
  -> retile and reallocate TCDM
  -> ABI command fusion and tile-loop compaction
```

DMA reuse reduces both traffic and often command count. DMA overlap primarily
reduces idle cycles; it may temporarily leave command count unchanged. These
effects must be measured separately.

## 4. Phase A: measure before changing the schedule

For every command and every selected layer, collect:

- PMU cycles and dispatch-to-command transition cost;
- DMA read/write bytes, bursts, busy cycles, and TCDM stalls;
- systolic compute, weight-load, IFM/OFM request and stall counters;
- AFU/Spatz issue, response, and TCDM activity;
- command type, layer, tile, tensor references, and residency decisions;
- peak TCDM usage and live-range lifetime at each boundary.

The estimator and PMU report must remain separate. The estimator predicts the
schedule; PMU measurements validate the implementation. A change is accepted
only when output remains byte-exact and the measured traffic/cycle delta is
reported.

## 5. Phase B: residency and DMA reuse

### 5.1 Qparams and LUTs

Regor should identify identical qparam/LUT blocks and assign a lifetime and
residency interval. Load once when the interval permits and reuse across tiles
or adjacent operators. A qparam update is required only when the next command
changes the active configuration.

### 5.2 Weights

Keep packed weights resident across output stripes when TCDM capacity and
dependency analysis permit. Prefer a larger output tile when it removes weight
reloads; otherwise spill explicitly to L2 and account for the reload.

### 5.3 Rolling input and halo reuse

For K3/K5 rolling convolutions, retain overlapping rows and DMA only newly
needed rows. The compiler must prove source strides, padding, row ownership,
and partial-tile behavior. Asymmetric padding must not cause an implicit scalar
fallback.

### 5.4 Producer/consumer reuse

Keep an intermediate tensor in native C32/ROW32 form when its consumer accepts
that layout. Use a storage-preserving alias for producer-to-consumer edges and
emit one materialization only at a real layout boundary. Avoid store-to-L2 and
reload-to-TCDM pairs when the live range fits.

### 5.5 DMA coalescing

Coalesce adjacent transfers only when address, stride, direction, alignment,
and dependency proofs hold. Prefer existing DMA1D/2D/3D encodings first; do not
introduce a generic byte loop or a broad fallback path.

## 6. Phase C: command-wide dependency scheduling and asynchronous overlap

Overlap applies to the complete ABI command stream, not only DMA commands.
Regor first lowers every command to a resource/access record:

- execution resource: DMA read, DMA write, systolic, AFU, Spatz, or Snitch;
- exact or conservative input, output, weight, qparam, LUT, and partial-sum spans;
- configuration state read or modified by the command;
- command-local staging windows and persistent tensor live ranges;
- whether the runtime can submit the operation and observe completion later.

Build a dependency DAG from RAW, WAR, WAW, engine-serialization, configuration,
and explicit graph-order edges. Schedule any ready command when its engine and
TCDM footprint are available. Insert a wait only at the first dependent command,
engine reuse, staging-window reuse, barrier, snapshot boundary, or model end.

The target pipeline includes all legal cross-engine combinations, for example:

```text
DMA-read tile N+1  ||  systolic tile N  ||  DMA-write tile N-1
AFU independent output   ||  DMA staging for the next systolic command
systolic independent tile || Spatz/AFU work on a disjoint TCDM range
```

Commands using the same engine remain serialized unless that engine exposes a
real queue. Spatz commands remain firmware-blocking initially, but can execute
while a previously submitted DMA, AFU, or systolic operation is outstanding.
RQ/LUT/config commands are state writes and cannot cross a compute command that
reads the old state.

Implement in descending breadth:

1. Replace the current DMA-only scan with the common access/dependency model.
2. Reorder independent existing commands to fill already-supported DMA overlap.
3. Add AFU submit/wait ABI operations using the existing AFU busy/done status.
4. Add systolic submit/wait for a single preloaded hardware job; split compound
   firmware loops only when Regor can represent every intermediate dependency.
5. Run list scheduling across the whole stream and retile/ping-pong where the
   original buffer allocation prevents useful overlap.

AFU/systolic submit support is an ABI and firmware change first. Existing RTL
start/done interfaces appear sufficient, but this must be proven by unit/block
tests. If concurrent engines expose a missing event, unsafe shared state, or an
RTL arbitration limitation, stop and request permission before modifying RTL.

After reuse is stable, schedule independent transfers around compute:

Use two buffers per resource only when the TCDM lifetime analysis proves the
peak fits. The compiler emits dependency/event metadata; firmware performs the
minimal submit, wait, and buffer rotation operations.

Required checks:

- no DMA writes into a buffer still read by systolic/AFU/Spatz;
- no qparam or shadow-register update before the previous owner completes;
- no output store before the final consumer response;
- event IDs and wait ordering are deterministic across reset/snapshot runs.

Use the existing intra-command shadow-register pipeline as the baseline. Extend
it across command boundaries only after dependency analysis proves that no DMA,
qparam, buffer, or configuration hazard intervenes.

If the current RTL cannot expose the required asynchronous event or queue,
stop and request permission before changing RTL. Do not silently add a new
hardware path.

## 7. Phase D: retile and reallocate

Reuse and ping-pong change TCDM pressure, so tile selection must run again after
each accepted schedule change. Evaluate:

- larger tiles that amortize weight/qparam loads;
- smaller tiles required by three-way ping-pong;
- partial tiles and row tails;
- command staging reservation (final 4 KiB);
- peak TCDM and L2 arena usage;
- DRAM latency, burst length, and outstanding read/write limits.

Do not constrain the solution to one fixed model shape. A tile is accepted only
when the same proof works for all supported shape variants in the release
corpus.

## 8. Phase E: command compaction after traffic optimization

Once the memory schedule is stable, reduce ABI overhead:

1. Fuse `RQ_LOAD` with a compute descriptor when the qparam lifetime is proven.
2. Encode a staged DMA plus line-buffer tile as one trusted tile descriptor.
3. Encode affine tile loops for repeated DMA/compute patterns.
4. Preserve logical child-command mapping for PMU, snapshots, and debugging.
5. Keep command boundaries where a dependency, snapshot, or failure diagnostic
   requires one.

The target is a measured reduction from 3,910 commands toward roughly
500--800 for YOLO320, but command count is not the primary success metric. A
smaller stream that increases L2 traffic or TCDM stalls is rejected.

## 9. Verification gates

Every increment follows this order:

1. Regor/C++ unit tests for residency, affine addresses, dependency proofs, and
   command emission.
2. ABI and trusted-firmware tests for valid and malformed descriptors.
3. Small block test with byte-exact output and TCDM/L2 checks.
4. Cluster prefix test from a clean snapshot boundary.
5. PMU comparison against the saved baseline.
6. Full selected-model output comparison and memory-budget check.

Record at least:

```text
command_count
command_bytes
L2_read_bytes / L2_write_bytes
TCDM_peak_bytes
PMU_total_cycles
DMA_busy / DMA_stall
systolic_compute / systolic_stall
AFU/Spatz activity
```

An optimization is not accepted if it changes output bytes, violates TCDM,
breaks snapshot continuation, increases the trusted firmware footprint beyond
ITCM, or falls into a scalar/one-byte slow path.

## 10. Explicit non-goals

- Do not add a generic hardware scheduler before compiler/firmware reuse and
  overlap are measured.
- Do not merge commands solely because their types match; references and
  dependencies must be affine and proven.
- Do not move validation back into firmware when Regor can prove it statically.
- Do not modify RTL without explicit approval, followed by RTL lint and a
  synthesis-oriented check.
