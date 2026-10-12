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

## Remaining integration gate

The new YOLO NAI and unchanged firmware have been frozen; the bounded
first-cascade reference case starts from reset and ends at command 50, with
checkpoints at commands 14, 23 and 50. The cluster simulator has been rebuilt.
The new consumer tile has M=480 and nine K tiles. Its 61,440-byte PSUM span
is `[267200,328640)`; input staging starts at 328,640, with no overlap.
Run this case and require zero
byte differences at every checkpoint; retain per-command PMU and complete
memory snapshots. Derive checkpoint numbers from the new ABI command list.

Local cluster evidence root:
`neural-ai/hw/rtl/cluster/tb/sim/test_yolo_external_psum_20261012/`.

Unit/block results do not yet establish cluster or full-model byte-exactness.
The old replay's 391,869 cycles are evidence of the faulty execution, not a
valid performance baseline for the repaired inference.
