# Neural-AI Phase 3 Conv Implementation Guide

## 1. Purpose

This document turns Phase 3 of
`neural_ai_full_support_implementation_plan.md` into an implementation and
verification specification. It covers:

- Conv classification and physical-layout selection.
- The C++ linebuffer tile planner.
- RGB, pointwise, generic C32 Conv, and depthwise weight encoding.
- Per-channel quantization matching the RTL rounding model.
- Runtime v2 command serialization, validation, and dispatch.
- C32 layout propagation, TCDM allocation, IC/OC group scheduling, channel
  tails, and fused activation clamps.
- Unit tests in `neural-compiler`.
- Host-runtime tests and Verilator simulations in `neural-ai`.

This is an acceptance document, not only a design description. Every task has a
fixed minimum scenario count, expected result, corner-case list, and exit gate.
An existing test may satisfy a scenario when it checks the same contract and
has the same expected result. It must not be duplicated only to satisfy a count.

## 2. Scope and Non-Goals

The stable Phase 3 graph contracts are:

| Operation | Native contract |
|---|---|
| RGB stem | INT8 NHWC/HWC input, C=3, K3, S2, P1, OC=32 |
| Pointwise Conv | INT8, K1, S1, P0, C32-blocked internal tensors |
| Generic C32 Conv | INT8, K3, S1 or S2, P1, full-group fast path and generic tail correctness path |
| Depthwise Conv | INT8, K3, S1 or S2, P1, depth multiplier 1, C32 tails masked |
| Requantization | INT32 accumulator to INT8, per-channel bias/multiplier/shift/zero-point/clamp |

The following are outside the Phase 3 stable contract:

- Batch other than 1.
- Dynamic shapes.
- Kernels other than those listed above, even when a lower-level RTL counter
  can represent them.
- Dilation other than 1.
- Depth multiplier other than 1.
- Non-symmetric IFM or weight quantization on native Conv paths.
- Implicit native-layout public model bindings.
- Runtime tile planning.
- Changes to any `neural-ai/hw/**/*.sv` file.

RTL is a read-only implementation contract for this phase. Tests, Python
oracles, firmware, HAL wrappers, command parsers, and build tooling under
`neural-ai` may be added or changed. If a required behavior cannot be achieved
without changing an `.sv` file, stop that task and record the RTL gap instead of
silently changing the graph contract.

## 3. Sources of Truth

Use the following sources in this order:

1. Current RTL behavior and limits described by:
   - `neural-ai/docs/systolic_array_spec.md`
   - `neural-ai/docs/linebuffer_architecture.md`
   - `neural-ai/docs/operator_support_matrix.md`
2. Runtime structures and HAL behavior:
   - `neural-ai/sw/lib/hal_systolic.h`
   - `neural-ai/sw/lib/hal_systolic.c`
   - `neural-ai/sw/lib/conv2d_packed.h`
   - `neural-ai/sw/lib/conv2d_packed.c`
   - `neural-ai/sw/lib/npu_cmd_desc_v2.h`
   - `neural-ai/sw/lib/npu_cmd_desc_v2.c`
   - `neural-ai/sw/lib/npu_runtime_ops.c`
3. The existing host-planned golden implementation:
   - `neural-ai/hw/rtl/cluster/tb/npu_linebuf_precompute.py`
4. The compiler ABI and generated `.nai` package:
   - `ethosu/regor/architecture/neuralai/neural_ai_abi.hpp`
   - `ethosu/regor/compiler/neural_ai_command_generator.cpp`

The Python planner is a read-only algorithm oracle while the C++ planner is
being ported. Golden data may be exported from it, but its scheduling algorithm
must not be changed in the same increment as the C++ port.

## 4. Current Baseline

The implementation is partial:

| Phase 3 task | Current state |
|---|---|
| T3.1 Classifier and constraints | Phase 3 mode classifier, shape/stride/padding/quantization gates, and scheduler lowering for the supported linebuffer modes are implemented; unsupported-shape diagnostics remain part of the negative-test matrix |
| T3.2 C++ linebuffer planner | C++ planner, M/width tiling, border geometry, checked ABI arithmetic, immutable golden vectors, TCDM checks, and grouped job integration are implemented; Python-golden parity remains open |
| T3.3 Weight encoders | GEMM, depthwise C32, and generic K3 grouped linebuffer packing are integrated with deterministic marker/golden tests; the complete byte-golden matrix remains open |
| T3.4 Per-channel quantization | Multiplier search, INT64 reference rounding, emitted per-channel qparams, activation clamp propagation, a deterministic 10,000-seed C++ reference check, and a 32-lane Verilator qparam check are implemented; broader randomized RTL parity remains open |
| T3.5 Commands | `RQ_LOAD`, `POINTWISE_C32`, group-scoped depthwise, and grouped RGB/generic K3 `LINEBUF_JOB` records are emitted; host and focused Verilator malformed-record validation are covered |
| T3.6 Runtime handlers | Pointwise/depthwise handlers and linebuffer HAL dispatch with TCDM offset relocation and accumulation modes are implemented; compiler-generated pointwise, RGB, generic-K3, depthwise, oversized-M packages, C96 grouped runtime package, and a pointwise-to-depthwise chain execute in Verilator; full E2E coverage remains open |
| T3.7 Layout and TCDM | Pointwise C32 layout, public-NHWC byte alignment, and partial scratch allocation exist; general Conv liveness is incomplete |
| T3.8 Boundary materialization | Pointwise NHWC-to/from-C32, direct-RGB input staging, and RTL unaligned compact-row iDMA execution are implemented; compiler-generated RGB boundary execution and the C3/C31 round-trip pass, while multi-group boundary cases and complete E2E proof remain open |
| T3.9 IC/OC loops and tails | Pointwise, generic K3 Conv, and depthwise group/tail command loops are implemented; compiler-generated generic-K3/depthwise and M257/M511 stripe packages plus independent C65/S2 and C96 grouped runtime tests execute in Verilator; full functional/performance E2E coverage remains open |
| T3.10 Activation fusion | Neural-AI fused ReLU/ReLU0To1/ReLU6/ReLU-1..1 activations are bypassed into producer qparam clamps; focused ReLU/ReLU6 compiler and pointwise ReLU6 Verilator coverage pass, while the complete clamp matrix and E2E coverage remain open |

No task is complete merely because its enum value, structure, or HAL function
exists. The compiler emission, runtime validation, hardware execution, and
expected-output comparison must all be present where required by the task.

## 5. Common Prerequisite Gate

All seven gates below must be recorded as passing before the final Phase 3 exit.
A task may be developed earlier, but it cannot be marked complete when a gate it
depends on is unresolved.

### G3.0.1 Physical memory contract

- One physical 512 KiB TCDM.
- The 4 KiB command staging window is excluded from compiler allocation.
- Runtime-only DTCM storage is not exposed as general model scratch.
- No compiler logic models a physical 12 KiB/4 KiB TCDM partition.

Evidence:

- Compiler architecture configuration test.
- Linker-map assertion for runtime-owned storage.
- Allocation failure test at one byte over the usable TCDM limit.

### G3.0.2 L2-aware `COPY_LAYOUT`

- External-to-local and local-to-external copies use iDMA where its tested
  contract permits.
- Local-to-local repacks use a Spatz or Snitch correctness path.
- L2-to-L2 is rejected.
- Compact NHWC rows are not required to have a 32-byte length or stride. Public
  NHWC bindings use byte alignment; ROW32/C32 and encoded-weight allocations
  retain 32-byte alignment.

Evidence:

- Host C=3/C=31 unaligned layout tests in both directions and a row crossing a
  32-byte AXI beat pass.
- The corresponding compiler-runtime Verilator test passes with unaligned
  public bases and compact rows whose later rows cross 32-byte beats.
- A local-to-local layout test proving no invalid iDMA event is created.

The host layout oracle covers C=3/C=31 compact rows, C32 grouping, unaligned
source/destination offsets, and INT32 elements. The Verilator compiled-package
matrix still needs the corresponding all-offset L2 tests.

### G3.0.3 Symmetric native Conv quantization

- IFM zero point is 0.
- Weight zero point is 0 for every channel.
- The compiler rejects a native Conv when either invariant is false.
- Output zero point may be nonzero only through the tested requant path.

Evidence:

- Positive symmetric constraint tests.
- Negative IFM and weight zero-point tests.

### G3.0.4 Mode-specific M limits

- A single runtime v2 command rejects `M > 256` where the command ABI limit is
  256.
- The compiler divides M=257 and M=511 into commands with `M <= 256`.
- On-chip group-stationary PSum is used only for `M <= 256`.
- Larger legal linebuffer work is spatially tiled or uses the documented
  external-PSum path.

Evidence:

- Runtime negative tests for single-command M=257 and 511.
- Compiler and Verilator positive tests for compiled M=257 and 511.

### G3.0.5 Frozen linebuffer wire record

- `LINEBUF_JOB` is exactly 160 bytes.
- Its payload contains the 80-byte `systolic_linebuf_cfg_t`, 36-byte
  `systolic_gemm32_req_t`, 4-byte rows, and 4-byte K-tile count.
- The final 20 bytes are zero.
- Address fields are compiler-owned offsets within the shared TCDM window; the
  firmware handler resolves them against `NPU_TCDM_BASE` immediately before
  programming the HAL. The handler applies the HAL's virtual top-padding-row
  subtraction to `input_base` after relocation, so the wire ABI never carries
  a negative offset.
- For linebuffer GEMM requests, `accum_en=0` means direct final requantization,
  `accum_en=1` means an intermediate PSum accumulation without requantization,
  and `accum_en=2` means the final PSum accumulation with requantization.
- The runtime command staging buffer accepts a complete 160-byte record.

Evidence:

- C++ `static_assert` checks.
- C host `sizeof` and `offsetof` checks.
- One complete byte-golden serialization test.

### G3.0.6 Separated C32 modes

The classifier must not represent all C32 Conv work as one mode. At minimum it
distinguishes:

- Generic merge path.
- C32 fast path.
- C32 multi-K group-stationary path.
- C32 channel-tail path.

The internal RTL group-stationary state is enabled only when all of these are
true:

```text
linebuffer enabled
coalesce enabled
KGEN enabled
C32_FAST enabled
lane_base == 0
block_valid_bytes == 32
input_c >= 32
input_c % 32 == 0
k_tiles > 1
```

The descriptor's `c32_group_stationary` bit is treated as a derived assertion,
not as an independent enable.

### G3.0.7 Clean baseline

Before each implementation increment:

- `neural-compiler` unit tests pass.
- `make -C neural-ai/sw/test/compiler_runtime check` passes.
- Relevant unchanged `neural-ai` Verilator smoke tests pass.
- `git -C ../neural-ai diff -- ':(glob)hw/**/*.sv'` is empty.

## 6. Shared Implementation Objects

The following object boundaries are used throughout the task descriptions.
Names marked "new" are proposed stable names; equivalent names are acceptable
when responsibilities remain separated.

| Object | Repository/file | Responsibility |
|---|---|---|
| `NeuralAIOpMode` | `neural_ai_op_config.hpp` | Carries the validated native datapath |
| `NeuralAIOpConfig` | `neural_ai_op_config.*` | Mode, granules, limits, layouts, scratch properties |
| `NeuralAIConstraints` | `neural_ai_constraints.cpp` | Rejects graph shapes and quantization outside the stable contract |
| `NeuralAILinebufferPlanner` (new) | `neural_ai_linebuffer_planner.*` | Produces complete linebuffer jobs; no runtime planning |
| `NeuralAIWeightEncoder` | `neural_ai_weight_encoder.*` | Packs RGB, pointwise, generic K3, and depthwise target constants |
| `NeuralAIQuantization` (new) | `neural_ai_quantization.*` | Generates and validates `QParamV1` using RTL-compatible rounding |
| `Command*V2` | `neural_ai_abi.hpp` | Fixed-width little-endian compiler wire ABI |
| `NeuralAICommandGenerator` | `neural_ai_command_generator.cpp` | Serializes already planned operations; does not reclassify |
| `NeuralAIGraphOptimiser` | `neural_ai_graph_optimiser.*` | Inserts explicit boundary conversions and propagates native layouts |
| Scheduler/TCDM allocator | `scheduler*`, `tensor_allocator*`, `neural_ai.cpp` | Layout-aware storage, liveness, staging, and partial-sum scratch |
| `nai_cmd_*_v2_t` | `neural-ai/sw/lib/npu_cmd_desc_v2.h` | Runtime mirror of fixed compiler records |
| Runtime dispatcher | `npu_cmd_desc_v2.c` | Bounds, region, size, capability, and semantic validation |
| Runtime operation handlers | `npu_runtime_ops.c` | Staging and HAL dispatch only |
| Systolic HAL | `hal_systolic.*`, `conv2d_packed.*` | Programs complete descriptors and starts tested modes |
| Build registration | `ethosu/regor/CMakeLists.txt`, `ethosu/regor/test/CMakeLists.txt` | Registers every new production source and unit-test source |

Mode classification occurs once. The selected `NeuralAIOpMode` is carried into
scheduling and command generation. The command generator must not infer a mode
again from a tensor format or kernel shape.

### 6.1 Recommended implementation order

The task numbers match the Phase 3 work-item numbers. Their implementation
order is:

1. Pass the common gates that affect the next increment.
2. T3.1 classifier and constraints.
3. T3.3 weight encoders and T3.4 quantization.
4. T3.2 planner unit tests and byte-golden output.
5. T3.5 command ABI and serialization.
6. T3.6 runtime handlers.
7. Finish the T3.2 Verilator exit scenarios through T3.5/T3.6.
8. T3.7 internal layout and allocation.
9. T3.8 public boundary materialization.
10. T3.9 complete group/tail scheduling.
11. T3.10 activation fusion.
12. Run the ordered Phase 3 E2E suite.

This order avoids making the runtime infer information that was not yet
represented by compiler objects. T3.2 may implement and validate its pure C++
planner before the linebuffer command exists, but its simulation exit remains
blocked until T3.5 and T3.6 are complete.

## 7. Test Conventions

### 7.1 Scenario counting

A "scenario" is one independently reported parameter set with one expected
result. A parameterized test function may implement several scenarios, but CI
output must identify the scenario ID on failure.

The minimum Phase 3 task suite contains:

| Test level | Minimum scenarios |
|---|---:|
| `neural-compiler` unit scenarios | 147 |
| `neural-ai` host C runtime scenarios | 52 |
| `neural-ai` Verilator scenarios | 73 |
| Total | 272 |

The final eight-model E2E suite in Section 18 is selected from these scenarios
and is not counted a second time.

### 7.2 Common positive simulation expectations

Unless a scenario specifies a rejection:

- `NPU_CMD_STATUS == NPU_CMD_STATUS_PASS`.
- `NPU_CMD_FAIL_CODE == 0`.
- `NPU_CMD_DONE_COUNT == model.command_count`.
- Every logical output byte equals the independent reference.
- C32 padding lanes are zero or ignored as specified and never appear in the
  public NHWC output.
- Guard regions before and after all bindings remain unchanged.
- The test times out if firmware or an RTL ready/valid path deadlocks.

### 7.3 Common negative simulation expectations

- `NPU_CMD_STATUS == NPU_CMD_STATUS_FAIL`.
- `NPU_CMD_FAIL_CODE` equals the documented validation class.
- `NPU_CMD_DONE_COUNT` stops before the failing command.
- The failure pointer identifies the failing record.
- Output and guard regions are unchanged after the rejection.

### 7.4 Independent references

- Conv and depthwise outputs use an INT64 accumulator in the reference.
- Quantization uses the explicit ties-away-from-zero model in T3.4.
- Weight byte goldens use direct OHWI indexing, not the compiler encoder.
- Linebuffer descriptor goldens come from
  `npu_linebuf_precompute.py` or checked-in immutable binary vectors.
- Tests must use signed extremes and nonuniform data. All-one weights alone are
  insufficient for byte-order verification.

## 8. T3.1 - Neural-AI Conv Classifier and Constraints

### Objective

Classify every accepted Conv into one explicit `NeuralAIOpMode` and reject every
shape that has no stable Phase 3 datapath.

### Prerequisites

- G3.0.3 symmetric quantization.
- G3.0.4 mode-specific M limits.
- G3.0.6 separated C32 modes.
- Logical tensors remain NHWC; physical format selection is separate.

### Objects and implementation

Modify:

- `neural_ai_op_config.hpp/.cpp`
- `neural_ai_constraints.cpp`
- `neural_ai.cpp`
- `test_neural_ai.cpp`

`NeuralAIOpConfig` must contain:

- `OpType` and `NeuralAIOpMode`.
- Required IFM, OFM, and weight formats.
- Kernel, stride, padding, and dilation limits.
- M granule and maximum command M.
- IC/OC group granules.
- Whether partial sums and requantization are required.
- Whether direct public NHWC input is allowed.

Classifier order must be deterministic:

1. Validate data type, batch, static shape, constant weights, and symmetric
   quantization.
2. Recognize the exact RGB stem contract.
3. Recognize pointwise K1/S1/P0.
4. Recognize generic K3/P1/S1 or S2.
5. Recognize depthwise K3/P1/S1 or S2 with multiplier 1.
6. Select full-group, tail, and group-stationary eligibility.
7. Otherwise return `Unsupported` with a diagnostic naming the first failed
   contract.

### Compiler unit scenarios: 18

| ID | Input | Expected |
|---|---|---|
| CL-01 | Pointwise K1/S1/P0, IC32/OC32 | `Conv2DPointwiseC32Requant` |
| CL-02 | RGB C3, K3/S2/P1, OC32 | `Conv2DRgbLinebufRequant`, direct NHWC input allowed |
| CL-03 | C32 K3/S1/P1, IC32/OC32 | C32 linebuffer requant mode |
| CL-04 | C32 K3/S2/P1, IC64/OC64 | C32 downsample linebuffer mode |
| CL-05 | Depthwise K3/S1/P1, C32, multiplier 1 | Depthwise S1 mode |
| CL-06 | Depthwise K3/S2/P1, C33, multiplier 1 | Depthwise downsample tail mode |
| CL-07 | Conv K2 | `Unsupported`, kernel diagnostic |
| CL-08 | Conv K5 | `Unsupported` for the Phase 3 graph contract |
| CL-09 | Conv stride 3 | `Unsupported`, stride diagnostic |
| CL-10 | Conv dilation 2 | `Unsupported`, dilation diagnostic |
| CL-11 | K3 with asymmetric padding | `Unsupported`, padding diagnostic |
| CL-12 | RGB-like input C4 | Must not select RGB mode |
| CL-13 | RGB C3 with OC64 | Must not select RGB mode |
| CL-14 | Generic C32 Conv with IC33 | Generic tail mode; group-stationary false |
| CL-15 | Generic C32 Conv with OC33 | Generic output-tail mode; group-stationary false |
| CL-16 | Depthwise multiplier 2 | `Unsupported`, multiplier diagnostic |
| CL-17 | Native Conv with IFM zero point 1 | `Unsupported`, quantization diagnostic |
| CL-18 | Native Conv with any weight zero point nonzero | `Unsupported`, channel diagnostic |

No Verilator scenario is required for pure classification.
Host C runtime scenarios: 0. Classification is a compiler-only responsibility.

Verilator scenarios: 0. Hardware execution is covered after the selected mode
has a command path.

### Corner cases

- H or W equal to 1.
- Output spatial size equal to zero after invalid padding.
- Weight tensor shape inconsistent with IFM/OFM channels.
- Missing constant weight or bias tensor.
- Per-tensor versus per-channel weight scale arrays.
- A depthwise operator presented as ordinary Conv2D.

### Exit criteria

- All 18 scenarios pass.
- Every supported mode has one positive scenario.
- Every rejected dimension reports a stable diagnostic.
- Scheduler and command generation consume the stored mode without
  reclassification.

## 9. T3.2 - C++ Linebuffer Tile Planner

### Objective

Port the host/Python schedule to C++ and produce complete, byte-stable
linebuffer job descriptors before firmware execution.

### Prerequisites

- T3.1 mode is available.
- G3.0.1 memory contract is frozen.
- G3.0.4 M limits are frozen.
- G3.0.5 linebuffer wire layout is frozen.
- T3.5 and T3.6 are required only for this task's Verilator exit.

### Objects and implementation

Add:

- `neural_ai_linebuffer_planner.hpp`
- `neural_ai_linebuffer_planner.cpp`
- `test_neural_ai_linebuffer_planner.cpp`
- Immutable descriptor golden vectors under `ethosu/regor/test/test_data/`.

The planner input includes:

- Full logical IFM/OFM shapes.
- Physical formats and base references.
- Kernel, stride, padding, IC, and OC.
- Input/output group index and valid lane count.
- TCDM budget and already-live allocation.
- Quantization block and partial-sum reference.

The planner output is a list of jobs containing:

- Output origin and extent.
- Clamped input extent and shifted padding.
- Input, output, weight, and partial-sum references.
- Pixel, row, C32 group, and output strides.
- `lane_base`, `block_valid_bytes`, and `channel_addr_offset`.
- `coalesce`, `coalesce_k_bytes`, K seed, and K-tile count.
- `c32_fast` and derived group-stationary assertion.
- Rows/M and first/accumulate/final-requant state.

Candidate search:

1. Enumerate legal output H/W candidates from large to small.
2. Reject kernel above 5, stride above 2, input tile width above 640, taps above
   25, and command M above the mode limit.
3. Calculate all live IFM, weight, OFM, PSum, qparam, layout, and staging bytes.
4. Reject candidates exceeding usable TCDM.
5. Estimate DMA, compute, and command cost.
6. Select lowest cost; break ties by larger M and then fewer jobs.

Use checked 64-bit arithmetic for every address and size calculation before
converting to the 32-bit ABI.

### Compiler unit scenarios: 20

| ID | Input | Expected |
|---|---|---|
| LP-01 | K3/S1 interior tile | Exact 124-byte Python-golden payload |
| LP-02 | K3/S1 top-left tile | Top/left zero injection and exact golden |
| LP-03 | K3/S1 bottom-right tile | Bottom/right clipping and exact golden |
| LP-04 | K3/S2 interior tile | Exact origin, strides, and golden |
| LP-05 | K3/S2 border tile | Shifted padding and exact golden |
| LP-06 | RGB C3 stem | Generic merge mode, `block_valid_bytes=3` |
| LP-07 | One full C32 group | `pixel_stride=32`, `lane_base=0`, C32 fast |
| LP-08 | First IC group of IC64 | First-write state and correct group base |
| LP-09 | Final IC group of IC64 | Accumulate/final-requant state |
| LP-10 | C31 tail | Valid bytes 31, fast/group-stationary false |
| LP-11 | C33 final group | Group base advanced, valid bytes 1 |
| LP-12 | Input width 639 | Legal without width split |
| LP-13 | Input width 640 | Legal boundary |
| LP-14 | Input width 641 | Split into legal stripes; no job exceeds 640 |
| LP-15 | M=256 | One legal preferred spatial job |
| LP-16 | M=257 | At least two jobs; every job M is at most 256 |
| LP-17 | Candidate exactly fills TCDM | Candidate accepted |
| LP-18 | Candidate exceeds TCDM by one byte | Smaller candidate selected or explicit failure |
| LP-19 | All group-stationary predicates true | Derived assertion true |
| LP-20 | One predicate false, including tail | Derived assertion false |

### Verilator scenarios: 6

| ID | Package | Expected |
|---|---|---|
| LP-S01 | RGB border descriptor | PASS and exact RGB reference |
| LP-S02 | C32 K3/S1 full group | PASS and exact Conv reference |
| LP-S03 | C32 K3/S2 full group | PASS and exact downsample reference |
| LP-S04 | IC64, M256 group-stationary | PASS; group-stationary predicate observed |
| LP-S05 | C33 tail generic path | PASS; tail bytes correct and guards unchanged |
| LP-S06 | Width 641 split schedule | PASS; multiple jobs reconstruct exact output |

Host C runtime scenarios: 0. The planner does not execute in firmware.

### Corner cases

- H/W smaller than the kernel.
- A tile containing only padding on one side.
- Address addition overflow.
- Zero-sized candidates.
- M=1 and M=256.
- TCDM fragmentation even when total free bytes appear sufficient.
- Tail group with a 32-byte-aligned base but fewer than 32 valid bytes.

### Exit criteria

- All 20 unit and 6 simulation scenarios pass.
- Every emitted job respects hard RTL limits.
- C++ and Python payloads are byte-identical for the golden matrix.
- Firmware performs no model-level tile calculation.

## 10. T3.3 - Conv Weight Encoders

### Objective

Encode constants from logical OHWI into the exact byte order consumed by each
native datapath.

### Prerequisites

- T3.1 mode is stored in `NeuralAIOpConfig`.
- Weight cache keys include the encoder/mode and cannot alias different packing
  formats.
- Tensor shapes and quantization axis are validated before encoding.

### Objects and implementation

Modify:

- `neural_ai_weight_encoder.hpp/.cpp`
- Scheduler weight-source selection.
- `test_neural_ai.cpp` or a new `test_neural_ai_weight_encoder.cpp`.

Add distinct encoding kinds:

- RGB: flattened `(kh, kw, ic)` K lanes padded to 32, then N lane.
- Pointwise: output group, input group, K lane, N lane.
- Generic K3: output group, input group, kh, kw, K lane, N lane.
- Depthwise: channel group, kh, kw, lane.

Each out-of-range channel lane is written as zero. Encoding must use the
original tensor strides and must not assume the FlatBuffer is contiguous in an
already-target-specific order.

### Compiler unit scenarios: 14

| ID | Input | Expected |
|---|---|---|
| WE-01 | Pointwise IC1/OC1 signed marker | Marker at K0/N0; remaining tile zero |
| WE-02 | Pointwise IC33/OC34 | Exact two-by-two group ordering |
| WE-03 | RGB C3/OC32 K3 | Exact `(kh,kw,ic,n)` order, K lanes 27..31 zero |
| WE-04 | Generic K3 IC32/OC32 | Exact `kh,kw,k,n` ordering |
| WE-05 | Generic K3 IC64/OC64 | Exact OC-group then IC-group ordering |
| WE-06 | Generic K3 IC64/OC32 | No phantom second OC group |
| WE-07 | Depthwise C1 | Nine marker bytes at lane 0; other lanes zero |
| WE-08 | Depthwise C31 | Lane 31 padding zero for every tap |
| WE-09 | Depthwise C32 | No tail padding |
| WE-10 | Depthwise C33 | Second group lane 0 valid, lanes 1..31 zero |
| WE-11 | Depthwise C65 | Three groups; final group has one valid lane |
| WE-12 | INT8 values -128 and 127 | Bytes preserved exactly |
| WE-13 | Nontrivial source strides | Same target bytes as contiguous logical OHWI |
| WE-14 | Same tensor requested in two encoding modes | Separate cache entries and different expected blobs |

### Verilator scenarios: 5

| ID | Package | Expected |
|---|---|---|
| WE-S01 | RGB marker weights | PASS and spatial/channel marker output |
| WE-S02 | Pointwise IC33/OC34 marker weights | PASS and exact output; catches group transpose |
| WE-S03 | Generic K3 IC32/OC32 marker weights | PASS and exact output |
| WE-S04 | Generic K3 IC64/OC64 marker weights | PASS and exact accumulated output |
| WE-S05 | Depthwise C33 marker weights | PASS and independent lane outputs |

### Corner cases

- Constant buffer shorter than shape storage.
- Quantization channel count different from logical output channels.
- Negative depth offset for a sliced weight source.
- A channel slice not aligned to the dimension used by the encoding.
- Integer overflow in encoded byte count.

### Exit criteria

- All 14 unit and 5 simulation scenarios pass.
- Randomized reference comparisons use at least 100 deterministic seeds across
  the 14 unit scenarios.
- Every padding byte is deterministic zero.
- Cache identity includes the encoding kind.

## 11. T3.4 - Per-Channel Quantization and RTL Rounding

### Objective

Generate per-channel qparams that reproduce RTL output bit-for-bit, including
bias, output zero point, clamp, negative rounding, and padded lanes.

### Prerequisites

- G3.0.3 symmetric native quantization.
- The `QParamV1` field widths and `RQ_LOAD` block size are frozen.
- RTL shift range is 0 through 31.

### Objects and implementation

Add or modify:

- `neural_ai_quantization.hpp/.cpp`
- `neural_ai_weight_encoder.cpp`
- `neural_ai_abi.hpp`
- `neural-ai/sw/lib/npu_quant_buffer.*`
- `neural-ai/sw/test/compiler_runtime/test_quant_buffer.c`
- `neural-ai/hw/rtl/cluster/tb/tests/test_systolic_requant.py`

For output channel `oc`:

```text
real_scale = ifm_scale * weight_scale[oc] / ofm_scale
```

Search shifts 0 through 31. For each shift, round
`real_scale * 2^shift` to the nearest legal positive INT32 multiplier. Select
the candidate with minimum absolute scale error and prefer the larger shift on
a tie.

The reference rounding operation adds half a unit to the magnitude and rounds
ties away from zero. It must not depend on the C/C++ result of right-shifting a
negative signed value.

### Compiler unit scenarios: 18

| ID | Input | Expected |
|---|---|---|
| Q-01 | Scale exactly 1 | Exact legal multiplier/shift pair |
| Q-02 | Scale exactly 0.5 | Exact result |
| Q-03 | Scale exactly representable at several shifts | Largest-shift tie winner |
| Q-04 | Very small positive representable scale | Legal nonzero multiplier |
| Q-05 | Scale below representable range | Compilation rejected |
| Q-06 | Scale causing multiplier overflow | Compilation rejected |
| Q-07 | Shift candidate 31 | Accepted |
| Q-08 | Required shift 32 | Rejected |
| Q-09 | Positive half tie | Rounded away from zero |
| Q-10 | Negative half tie | Rounded away from zero |
| Q-11 | INT32 positive accumulator edge | INT64 reference, no host overflow |
| Q-12 | INT32 negative accumulator edge | INT64 reference, no host overflow |
| Q-13 | Valid per-channel bias scales | Bias copied exactly |
| Q-14 | Invalid bias scale on one channel | Rejected with channel index |
| Q-15 | Per-tensor weight scale | Replicated across logical channels |
| Q-16 | Nonuniform per-channel scales | Distinct lane qparams |
| Q-17 | Output zero point nonzero | Added after scaling, then clamped |
| Q-18 | Padded C tail lanes | Safe neutral qparams and no logical output |

### Host C runtime scenarios: 6

| ID | Input | Expected |
|---|---|---|
| Q-H01 | Load complete 32-lane block | All runtime lanes equal source |
| Q-H02 | Qparam index at final valid block | Accepted |
| Q-H03 | Qparam range one byte short | `BAD_REFERENCE` |
| Q-H04 | Shift 32 | `BAD_COMMAND` |
| Q-H05 | Clamp min greater than max | `BAD_COMMAND` |
| Q-H06 | Command uses block not loaded | `BAD_COMMAND` |

### Verilator scenarios: 8

| ID | Data | Expected |
|---|---|---|
| Q-S01 | Positive accumulators, uniform qparams | Exact reference bytes |
| Q-S02 | Negative accumulators, uniform qparams | Exact reference bytes |
| Q-S03 | Positive and negative half ties | Ties away from zero |
| Q-S04 | 32 different lane multipliers | Exact per-lane output |
| Q-S05 | 32 different biases | Exact per-lane output |
| Q-S06 | Nonzero output zero point | Exact offset and saturation |
| Q-S07 | ReLU/ReLU6 clamp bounds | Exact quantized clamp |
| Q-S08 | Tail group with invalid lanes | Valid lanes correct; guards unchanged |

### Corner cases

- NaN, infinity, zero, or negative scales.
- Bias missing versus explicitly zero.
- INT64 intermediate overflow during worst-case proof.
- Clamp bounds outside INT8 before saturation.
- Qparam block reuse across M stripes.

### Exit criteria

- All 18 compiler, 6 host, and 8 simulation scenarios pass.
- Compiler and RTL reference are bit-exact over at least 10,000 deterministic
  randomized accumulator/qparam combinations.
- No accepted command contains a shift above 31.

## 12. T3.5 - Runtime v2 Conv Commands

### Objective

Freeze, serialize, and validate `RQ_LOAD`, `POINTWISE_C32`,
`DEPTHWISE_C32`, and `LINEBUF_JOB`.

### Prerequisites

- G3.0.5 linebuffer record.
- T3.1 mode and T3.2 jobs.
- T3.3 constant offsets.
- T3.4 qparam blocks.

### Objects and implementation

Mirror each record in:

- `ethosu/regor/architecture/neuralai/neural_ai_abi.hpp`
- `ethosu/regor/compiler/neural_ai_command_generator.cpp`
- `neural-ai/sw/lib/npu_cmd_desc_v2.h`
- `neural-ai/sw/lib/npu_cmd_desc_v2.c`

Required sizes:

| Command | Bytes |
|---|---:|
| `RQ_LOAD` | 32 |
| `POINTWISE_C32` | 96 |
| `DEPTHWISE_C32` | 96 |
| `LINEBUF_JOB` | 160 |

Every record:

- Is little-endian and a multiple of 32 bytes.
- Has zero reserved fields.
- Carries `layer_id` and `tile_id`.
- Uses relocatable `RefV1` references only.
- Is fully planned by the compiler.

`DEPTHWISE_C32` must be group-scoped before general compiler emission is
enabled:

- One command processes exactly one C32 channel group.
- Its IFM, weight, and OFM references point at that group's base.
- Its valid-channel field is in the range 1 through 32.
- The command references the qparam block for the same group.
- The compiler emits `ceil(C / 32)` depthwise commands and one `RQ_LOAD` per
  distinct group block, with legal state reuse when two blocks are identical.

This requires an ABI-minor semantic change or explicit group fields if the
existing `channels` field was documented as the full tensor channel count. The
runtime must not run C>32 as one command using one qparam block unless the
compiler proves that all group qparam blocks are byte-identical. That uniform
special case is an optimization, not the general per-channel contract.

### Compiler unit scenarios: 15

| ID | Record | Expected |
|---|---|---|
| CS-01 | `RQ_LOAD` | Exact 32-byte golden |
| CS-02 | Pointwise one group | Exact 96-byte golden |
| CS-03 | Pointwise multi-group strides | Exact references and strides |
| CS-04 | Pointwise tail | Logical counts and padded storage separated |
| CS-05 | Depthwise S1 | Exact 96-byte golden |
| CS-06 | Depthwise S2 | Exact stride fields |
| CS-07 | Depthwise C33 | Two group-scoped records with valid lanes 32 and 1 and qparam blocks 0 and 1 |
| CS-08 | RGB `LINEBUF_JOB` | Exact 160-byte golden |
| CS-09 | Generic S1 `LINEBUF_JOB` | Exact 160-byte golden |
| CS-10 | Generic S2 `LINEBUF_JOB` | Exact 160-byte golden |
| CS-11 | First IC group | First-write flags |
| CS-12 | Middle IC group | Accumulate flags |
| CS-13 | Final IC group | Final requant flags and qparam block |
| CS-14 | M257 compile | Multiple legal records, each M at most 256 |
| CS-15 | All records | Reserved bytes zero and size divisible by 32 |

### Host C runtime scenarios: 16

| ID | Input | Expected |
|---|---|---|
| CS-H01..H04 | One valid record of each command type | Parser accepts and calls matching mock once |
| CS-H05 | Unknown required type | `UNSUPPORTED` |
| CS-H06 | Record smaller than header | `BAD_COMMAND` |
| CS-H07 | Wrong known-command size | `BAD_COMMAND` |
| CS-H08 | Record crosses command-section end | `BAD_REFERENCE` |
| CS-H09 | Nonzero reserved field | `BAD_COMMAND` |
| CS-H10 | Depthwise valid lanes 0 | `BAD_COMMAND` |
| CS-H11 | Depthwise valid lanes 33 | `BAD_COMMAND` |
| CS-H12 | Weight range too short | `BAD_REFERENCE` |
| CS-H13 | Scratch/output range too short | `BAD_REFERENCE` |
| CS-H14 | Single-command M257 | `BAD_COMMAND` |
| CS-H15 | Single-command M511 | `BAD_COMMAND` |
| CS-H16 | Qparam block mismatch | `BAD_COMMAND` |

### Verilator scenarios: 5

| ID | Package | Expected |
|---|---|---|
| CS-S01 | Valid RQ plus pointwise | PASS and exact output |
| CS-S02 | Valid RQ plus depthwise | PASS and exact output |
| CS-S03 | Valid RQ plus linebuffer | PASS and exact output |
| CS-S04 | Malformed 160-byte linebuffer record | FAIL before hardware start |
| CS-S05 | Valid command after a malformed record | Not executed; done count stops |

### Corner cases

- Maximum 4 KiB stream refill boundary.
- A 160-byte command split across two refills.
- Reference offset addition overflow.
- Command count inconsistent with the section contents.
- Optional/skippable flag on a required Conv command.

### Exit criteria

- All 15 compiler, 16 host, and 5 simulation scenarios pass.
- C and C++ size/offset assertions agree.
- The compiler never emits a reserved-only command.
- Runtime rejects invalid records before programming the HAL.

## 13. T3.6 - Runtime Operation Handlers

### Objective

Keep firmware handlers small: resolve/stage references, validate state, program
an existing HAL path, wait, and report status. No handler selects tiles or
layouts.

### Prerequisites

- T3.5 records are frozen.
- T3.4 qparam state is validated.
- HAL paths are already verified against unchanged RTL.

### Objects and implementation

Modify:

- `npu_runtime_ops.h/.c`
- `npu_cmd_desc_v2.c`
- `hal_systolic.h/.c` only when a tested wrapper is missing
- `conv2d_packed.h/.c` only for shared descriptor dispatch
- `sw/test/compiler_runtime/test_cmd_v2.c`

Handlers:

- Stage weights from model constants to TCDM when the HAL cannot read L2.
- Validate required 32-byte C32/GEMM alignment.
- Preserve byte-addressed generic/RGB linebuffer fields.
- Require the referenced qparam block to be loaded.
- Disable requant state after the operation or explicitly track its ownership.
- Convert HAL failure/timeout into a stable runtime failure code.

### Host C runtime scenarios: 18

| ID | Handler input | Expected |
|---|---|---|
| RH-01 | Valid pointwise one group | One stage and one HAL call |
| RH-02 | Valid pointwise multi-IC | Correct staged byte count and PSum ref |
| RH-03 | Valid depthwise S1 group | Exactly 9*32 weight bytes and correct HAL arguments |
| RH-04 | Valid depthwise S2 | Correct stride argument |
| RH-05 | Valid linebuffer job | Descriptor copied unchanged to HAL |
| RH-06 | Reused qparam block | No redundant state error |
| RH-07 | Qparam not loaded | `BAD_COMMAND`, no HAL call |
| RH-08 | Misaligned C32 IFM | `BAD_REFERENCE` |
| RH-09 | Misaligned C32 OFM | `BAD_REFERENCE` |
| RH-10 | Misaligned weight staging destination | `BAD_REFERENCE` |
| RH-11 | IFM range one byte short | `BAD_REFERENCE` |
| RH-12 | OFM range one byte short | `BAD_REFERENCE` |
| RH-13 | Weight range one byte short | `BAD_REFERENCE` |
| RH-14 | Invalid depthwise output dimensions | `BAD_COMMAND` |
| RH-15 | Invalid stride 0 | `BAD_COMMAND` |
| RH-16 | Invalid stride 3 | `BAD_COMMAND` |
| RH-17 | HAL timeout | Stable timeout/failure code |
| RH-18 | Successful completion | Requant state disabled or ownership updated |

### Verilator scenarios: 10

The compact-row/non-32-byte endpoint cases are part of the ABI evidence.  The
host-side oracle and the compiler-runtime cocotb case cover their byte mapping
and execution with unaligned public bases.

| ID | Operation | Expected |
|---|---|---|
| RH-S01 | Pointwise C32 one group | PASS, exact output |
| RH-S02 | Pointwise C64 to C64 | PASS, exact output |
| RH-S03 | RGB stem | PASS, exact output |
| RH-S04 | Generic C32 S1 | PASS, exact output |
| RH-S05 | Generic C32 S2 | PASS, exact output |
| RH-S06 | Depthwise C31 S1 | PASS, exact tail output |
| RH-S07 | Depthwise C33 S2 | PASS, exact tail output |
| RH-S08 | Back-to-back different qparam blocks | Both outputs exact |
| RH-S09 | Invalid command followed by valid command | FAIL; second command not run |
| RH-S10 | Legacy v1 smoke after v2 changes | Existing output unchanged |

### Corner cases

- Weight blob larger than one staging copy.
- Back-to-back operations using the same scratch address after completion.
- Qparam state left enabled by a failed HAL start.
- Timeout while output backpressure is active.
- Source and destination references alias unexpectedly.

### Exit criteria

- All 18 host and 10 simulation scenarios pass.
- No handler contains a tile-search loop.
- V1 behavior remains unchanged.
- Firmware text plus read-only data remains within its enforced limit.

## 14. T3.7 - C32 Layout Propagation and TCDM Partial Sums

### Objective

Keep logical Graph IR shapes in NHWC while using C32-blocked internal storage
and correctly sized INT32 partial sums.

### Prerequisites

- G3.0.1 physical memory contract.
- T3.1 required formats.
- T3.2 tile lifetimes.
- G3.0.2 boundary conversion semantics.

### Objects and implementation

Modify:

- `neural_ai_graph_optimiser.*`
- `neural_ai.cpp`
- scheduler connection/slice handling where physical format is required
- allocator alignment and storage-size hooks
- `neural_ai_command_generator.cpp`

Rules:

- Public rank-4 input/output remains compact NHWC.
- Compatible Conv, pointwise, depthwise, and later native consumers keep
  C32-blocked internal edges.
- C32 offset is group-major:

```text
offset = (((c / 32) * (H * W) + pixel) * 32 + (c % 32)) * element_bytes
```

- Multi-group C32 is never modeled as one affine NHWC stride.
- INT8 OFM storage and INT32 PSum storage are separate live ranges.
- PSum bytes are `tile_M * 32 * 4` per active output group.
- PSum may be reused after the final requant command for that output group.
- C32 fast consumers retain 32-byte alignment. No scalar temporary alignment
  exception is frozen in the ABI until the RTL endpoint test passes.

### Compiler unit scenarios: 18

| ID | Graph/storage | Expected |
|---|---|---|
| LA-01 | Public NHWC input | Format pinned NHWC |
| LA-02 | Public NHWC output | Format pinned NHWC |
| LA-03 | Pointwise-to-pointwise | One internal C32 edge, no middle copy |
| LA-04 | Pointwise-to-depthwise | One internal C32 edge |
| LA-05 | Depthwise-to-pointwise | One internal C32 edge |
| LA-06 | Generic Conv-to-depthwise | One internal C32 edge |
| LA-07 | C31 storage | H*W*32 bytes |
| LA-08 | C32 storage | H*W*32 bytes |
| LA-09 | C33 storage | H*W*64 bytes |
| LA-10 | C64 storage | H*W*64 bytes |
| LA-11 | C65 storage | H*W*96 bytes |
| LA-12 | C32 group-aligned slice | Alias allowed with group-major offset |
| LA-13 | C33 slice starting at channel 1 | Alias rejected/materialized |
| LA-14 | M256 PSum | Exactly 256*32*4 bytes |
| LA-15 | M257 schedule | PSum sized per split tile, not 257 rows in one command |
| LA-16 | Two sequential output groups | PSum live range reused safely |
| LA-17 | Peak allocation exactly at limit | Compile succeeds |
| LA-18 | Peak allocation one byte over limit | Deterministic compile failure or smaller tile |

### Verilator scenarios: 6

| ID | Chain | Expected |
|---|---|---|
| LA-S01 | Pointwise -> pointwise C32 | PASS; no intermediate public-layout copy |
| LA-S02 | Pointwise -> depthwise C32 | PASS; exact output |
| LA-S03 | Depthwise -> pointwise C33 | PASS; tail remains private |
| LA-S04 | Generic Conv -> depthwise C64 | PASS; exact output |
| LA-S05 | Two OC groups sharing PSum scratch sequentially | PASS; no cross-group contamination |
| LA-S06 | Deliberate under-sized scratch package | FAIL before compute |

### Corner cases

- Fan-out to two consumers with different native formats.
- View/reshape over multi-group C32.
- Concat or split at a non-32 channel boundary.
- In-place alias requested while PSum is live.
- 32-bit reference overflow despite legal logical tensor size.

### Exit criteria

- All 18 compiler and 6 simulation scenarios pass.
- Logical shapes are unchanged by physical layout selection.
- Distinct depthwise qparam blocks for adjacent C32 groups.
- No compatible internal Conv chain materializes NHWC between operations.
- Peak TCDM is reported and enforced.

## 15. T3.8 - Boundary `COPY_LAYOUT` and Direct RGB Input

### Objective

Materialize compact public NHWC only at unsupported boundaries while allowing
the verified RGB stem to consume NHWC/HWC C3 directly.

### Prerequisites

- G3.0.2 L2-aware layout behavior.
- T3.1 marks direct RGB eligibility.
- T3.7 records physical formats independently from logical shapes.

### Objects and implementation

Modify:

- `neural_ai_graph_optimiser.*`
- `neural_ai_command_generator.cpp`
- `neural-ai/sw/lib/npu_layout_ops.*`
- `neural-ai/sw/lib/npu_cmd_desc_v2.c`
- `neural-ai/sw/test/compiler_runtime/test_copy_layout.c`
- `test_compiled_model.py`

Required modes:

- NHWC to ROW32.
- ROW32 to NHWC.
- NHWC to C32.
- C32 to NHWC.

For multi-group C32, issue one channel-group transfer sequence or a verified
Spatz gather/scatter kernel. A single affine stride cannot represent the full
layout.

### Compiler unit scenarios: 12

| ID | Graph boundary | Expected |
|---|---|---|
| BC-01 | NHWC C3 -> RGB stem | No input `COPY_LAYOUT` |
| BC-02 | NHWC C3 -> pointwise | NHWC-to-C32 inserted |
| BC-03 | NHWC C31 -> pointwise | Pack inserted; storage C32 |
| BC-04 | NHWC C32 -> C32 native | Copy may be eliminated only after byte-equivalence proof |
| BC-05 | NHWC C33 -> native | Multi-group pack inserted |
| BC-06 | NHWC C65 -> native | Three-group pack inserted |
| BC-07 | Native C31 -> NHWC output | Unpack exactly 31 bytes per pixel |
| BC-08 | Native C32 -> NHWC output | Legal byte-equivalent elimination or exact copy |
| BC-09 | Native C33 -> NHWC output | Multi-group gather/unpack |
| BC-10 | Pointwise -> depthwise internal | No boundary copy |
| BC-11 | Local C32 -> local ROW32 | CPU/Spatz path, not iDMA |
| BC-12 | Unsupported layout pair | Compile fails with explicit diagnostic |

### Host C runtime scenarios: 12

| ID | Transfer | Expected |
|---|---|---|
| BC-H01 | L2 NHWC C3 -> TCDM | Exact bytes, zero padded lanes |
| BC-H02 | L2 NHWC C31 -> TCDM | Exact bytes, zero lane 31 |
| BC-H03 | TCDM C31 -> L2 NHWC | Exact 31-byte compact rows |
| BC-H04 | C3 rows crossing a 32-byte beat | Exact copy |
| BC-H05 | C31 rows crossing a 32-byte beat | Exact copy |
| BC-H06 | NHWC C33 -> two C32 groups | Exact group-major bytes |
| BC-H07 | Two C32 groups -> NHWC C33 | Exact compact bytes |
| BC-H08 | NHWC C65 round-trip | Exact original 65 logical bytes per pixel |
| BC-H09 | TCDM -> TCDM layout | CPU/Spatz path selected |
| BC-H10 | L2 -> L2 | Direction/region rejection |
| BC-H11 | Direction field contradicts regions | Rejected |
| BC-H12 | Destination range one byte short | Rejected; guard unchanged |

### Verilator scenarios: 10

| ID | Boundary case | Expected |
|---|---|---|
| BC-S01 | C3 NHWC-to-C32 round-trip | Exact original bytes |
| BC-S02 | C31 NHWC-to-C32 round-trip | Exact original bytes |
| BC-S03 | C32 round-trip | Exact original bytes |
| BC-S04 | C33 round-trip | Exact original bytes |
| BC-S05 | C65 round-trip | Exact original bytes |
| BC-S06 | Aligned public base with later compact-row addresses unaligned | PASS and exact bytes |
| BC-S07 | Non-32 row length/stride | PASS and exact rows |
| BC-S08 | Direct RGB stem input | PASS without a pre-pack command |
| BC-S09 | C32 graph output | Exact compact NHWC output |
| BC-S10 | Illegal L2-to-L2 command | FAIL with direction/region code |

### Corner cases

- N=1, H=1, W=1.
- C=1, 3, 31, 32, 33, 63, 64, and 65.
- Engine-level test bases at every byte offset 0 through 31. These test iDMA
  behavior and do not by themselves freeze an ABI alignment exception.
- A final short row at the end of the binding.
- Overlapping local source and destination.

### Exit criteria

- All 12 compiler, 12 host, and 10 simulation scenarios pass.
- Public output exposes no padded lane.
- Direct RGB is selected only for the exact verified contract.
- No L2-to-L2 iDMA is issued.

## 16. T3.9 - IC/OC Group Loops and Channel Tails

### Objective

Emit complete compiler-owned loops for spatial tiles, output groups, input
groups, M stripes, accumulation, final requantization, and tail masking.

### Prerequisites

- T3.2 planner.
- T3.3 encoders.
- T3.4 qparams.
- T3.5/T3.6 command execution.
- T3.7 storage and PSum liveness.

### Objects and implementation

Modify:

- `neural_ai_command_generator.cpp`
- scheduler cost/operation metadata
- `neural_ai_performance.*`
- compiler package tests and `test_compiled_model.py`

Loop order for generic Conv:

```text
for spatial_tile
  for output_group
    for input_group
      emit first-write or accumulate linebuffer job
    load/reuse output-group qparams
    emit final-requant job
```

Rules:

- Every command M is legal for its mode.
- Weight offset changes with both IC and OC group.
- IFM C32 group base changes with IC group.
- OFM and qparam block change with OC group.
- PSum is reset before the first IC group and read only by later groups.
- Requantization occurs only after the final IC group.
- Tail lanes are zero in constants, masked in descriptors, and omitted from
  public NHWC output.
- A tail disables the full-group group-stationary fast mode.
- Depthwise has a channel-group loop rather than a dense IC/OC accumulation
  loop. Each group receives its own qparam load and group-scoped command.

### Compiler unit scenarios: 20

| ID | Shape/mode | Expected |
|---|---|---|
| GL-01 | Pointwise IC32/OC32 | One IC and one OC group |
| GL-02 | Pointwise IC33/OC34 | Two IC and two OC groups with tails |
| GL-03 | Pointwise M256 | One M stripe |
| GL-04 | Pointwise M257 | M256 plus M1 |
| GL-05 | Pointwise M511 | M256 plus M255 |
| GL-06 | Generic K3 IC32/OC32 S1 | One group loop |
| GL-07 | Generic K3 IC33/OC32 S1 | Full plus one-lane IC tail; final only requantizes |
| GL-08 | Generic K3 IC32/OC33 S1 | Full plus one-lane OC tail and two qparam blocks |
| GL-09 | Generic K3 IC64/OC64 S1 | Four IC/OC combinations |
| GL-10 | Generic K3 IC64/OC64 S2 | Correct downsample dimensions |
| GL-11 | Depthwise C31 S1 | One tail group |
| GL-12 | Depthwise C32 S1 | One full group |
| GL-13 | Depthwise C33 S1 | Two group commands; final valid lane count 1; distinct qparam blocks |
| GL-14 | Depthwise C48 S2 | Two group commands; final valid lane count 16 |
| GL-15 | Depthwise C64 S2 | Two full-group commands |
| GL-16 | Depthwise C65 S1 | Three group commands; final valid lane count 1 |
| GL-17 | Depthwise C96 S2 | Three full-group commands |
| GL-18 | Full C32 multi-K, M256 | Group-stationary predicate true |
| GL-19 | Same layer with a channel tail | Group-stationary predicate false |
| GL-20 | Command/reference offset near UINT32_MAX | Checked failure, no wrap |

### Verilator scenarios: 16

| ID | Compiled model | Expected |
|---|---|---|
| GL-S01 | Pointwise IC33/OC34 | Exact output |
| GL-S02 | Pointwise M257 | Exact output; all commands M at most 256 |
| GL-S03 | Pointwise M511 | Exact output; two M stripes |
| GL-S04 | Generic K3/S1 IC32/OC32 | Exact output |
| GL-S05 | Generic K3/S1 IC33/OC32 | Exact accumulated tail output |
| GL-S06 | Generic K3/S1 IC32/OC33 | Exact output-group tail |
| GL-S07 | Generic K3/S1 IC64/OC64 | Exact output |
| GL-S08 | Generic K3/S2 IC64/OC64 | Exact downsample output |
| GL-S09 | Depthwise S1 C31 | Exact tail output |
| GL-S10 | Depthwise S1 C33 | Exact two-group output |
| GL-S11 | Depthwise S2 C48 | Exact downsample tail output |
| GL-S12 | Depthwise S2 C65 | Exact three-group output |
| GL-S13 | Full C32 multi-K M256 | Exact output and group-stationary active |
| GL-S14 | Tail variant | Exact output and group-stationary inactive |
| GL-S15 | Depthwise S2 C96 | Exact three-group output |
| GL-S16 | Pointwise -> depthwise chain | Exact final NHWC output |

Host C runtime scenarios: 0. Runtime command validation is covered by T3.5 and
T3.6; loop ownership is verified in compiler command streams and E2E execution.

### Corner cases

- IC/OC group count multiplication overflow.
- M stripe ending at one row.
- Output H/W odd under stride 2.
- Bias/qparam arrays with logical OC rather than padded OC entries.
- PSum reuse when an output group has only one IC group.
- Depthwise groups do not use dense IC accumulation or a separate OC group
  loop; they do require one group-scoped command and qparam block per C32
  channel group.

### Exit criteria

- All 20 compiler and 16 simulation scenarios pass.
- M257 and M511 compile and execute successfully as multiple legal commands.
- No final requant occurs before the final IC group.
- C33, C48, C64, C65, and C96 are covered.
- Functional and performance results distinguish full group from tail mode.

## 17. T3.10 - Fuse ReLU, ReLU6, and Clamp

### Objective

Convert supported fused activations into per-channel output clamp bounds in the
final requantization without emitting a separate tensor operation.

### Prerequisites

- T3.4 quantization and rounding.
- T3.9 final-requant command placement.
- The frontend activation range is available in the scheduled operation.

### Objects and implementation

Modify:

- `neural_ai_quantization.*`
- `neural_ai_op_config.*`
- `neural_ai_command_generator.cpp`
- `tflite_graph_optimiser.cpp` and `architecture_constraints.hpp` for target-gated
  activation bypass and clamp preservation
- compiler tests and `test_compiled_model.py`

Clamp conversion:

1. Convert real activation bounds to output quantized units using the OFM scale
   and zero point.
2. Apply ties-away-from-zero rounding.
3. Saturate to INT8.
4. Store bounds in every logical channel's qparam.
5. Give padded lanes safe bounds.
6. Emit no standalone ReLU/ReLU6/Clamp command when fused.

The TFLite reader materializes a fused activation as a producer plus an
activation operation. The Neural-AI constraints advertise clamp-capable
requantization, so the supported-operator pass preserves the producer as an
NPU operation, copies the activation output quantization, and records the
activation bounds before disconnecting the standalone activation. Other
architectures retain their existing CPU-passthrough behavior.

### Compiler unit scenarios: 12

| ID | Activation | Expected |
|---|---|---|
| AF-01 | NONE | Clamp -128 to 127 |
| AF-02 | ReLU, zp=0 | Clamp 0 to 127 |
| AF-03 | ReLU, nonzero output zp | Quantized zero maps to output zp |
| AF-04 | ReLU6 exact scale | Exact quantized six bound |
| AF-05 | ReLU6 non-exact scale | RTL-compatible rounded bound |
| AF-06 | Clamp entirely inside INT8 | Exact quantized bounds |
| AF-07 | Clamp below INT8 minimum | Lower bound saturated |
| AF-08 | Clamp above INT8 maximum | Upper bound saturated |
| AF-09 | Clamp min greater than max | Compilation rejected |
| AF-10 | Per-channel weight scales | Same output-domain clamp, distinct multiplier |
| AF-11 | Tail lanes | Safe padded-lane clamps |
| AF-12 | Fused activation graph | No extra activation command/tensor |

### Verilator scenarios: 9

| ID | Compiled operation | Expected |
|---|---|---|
| AF-S01 | Pointwise + ReLU | Exact clamped output |
| AF-S02 | Pointwise + ReLU6 | Exact clamped output |
| AF-S03 | RGB stem + ReLU | Exact clamped output |
| AF-S04 | Generic S1 + ReLU6 | Exact clamped output |
| AF-S05 | Generic S2 + Clamp | Exact clamped output |
| AF-S06 | Depthwise S1 + ReLU | Exact clamped output |
| AF-S07 | Depthwise S2 + ReLU6 | Exact clamped output |
| AF-S08 | Negative and positive values at both boundaries | Inclusive exact bounds |
| AF-S09 | C33 tail plus activation | Valid channels exact; padding hidden |

### Corner cases

- Output zero point near -128 or 127.
- ReLU6 upper bound saturating to 127.
- A bound exactly halfway between quantized values.
- NaN or infinity in an imported activation bound.

### Exit criteria

- All 12 compiler and 9 simulation scenarios pass.
- No redundant activation command is emitted.
- Clamp values match the independent reference for every output channel.

### Compiler-generated package evidence (current increment)

The following tests compile TFLite fixtures with the Regor Neural-AI backend,
load the generated package and weights, and compare the Verilator result with
an independent integer reference:

| Test | Coverage | Result |
|---|---|---|
| `test_compiler_generated_rgb_k3_conv_package` | NHWC C3 -> C32, K3/S2/P1 direct-RGB boundary | Pass |
| `test_compiler_generated_generic_k3_conv_package` | K3/S1, IC64 -> OC64 grouped linebuffer path | Pass |
| `test_compiler_generated_depthwise_k3_conv_package` | Depthwise K3/S2, C33 group/tail path | Pass |
| `test_compiler_runtime_unaligned_row32_c3_c31` | RTL iDMA round-trip with C3/C31 compact rows, unaligned public bases, and 32-byte beat crossings | Pass |
| `test_compiler_runtime_depthwise_c65_stride2_tail` | Independent runtime package, C65, three C32 groups, final one-lane tail, S2 | Pass |
| `test_compiler_runtime_pointwise_per_channel_requant` | 32 distinct bias/multiplier/zero-point lanes with signed values | Pass |
| `test_compiler_generated_pointwise_m257_stripes` | Compiler package, M=257, non-overlapping public bindings, M256+M1 exact output | Pass |
| `test_compiler_generated_pointwise_m511_stripes` | Compiler package, M=511, non-overlapping public bindings, M256+M255 exact output | Pass |
| `test_compiler_runtime_depthwise_c96_stride2_groups` | Independent runtime package, C96, three full C32 groups, S2 | Pass |
| `test_compiler_runtime_pointwise_depthwise_chain` | Independent runtime package, internal C32 pointwise -> depthwise chain | Pass |
| `test_compiler_generated_pointwise_depthwise_chain` | Regor-generated TFLite pointwise -> depthwise package, exact public NHWC output | Pass |

These focused results do not close the complete matrix below: the full
Micro-MobileNet model and the remaining broad randomized/layout matrix remain
required for the Phase 3 exit. The oversized-M tests also use deliberately
non-overlapping public L2 bindings; this is required when a single input or
output exceeds the legacy 4 KiB fixture spacing.

## 18. Phase 3 End-to-End Acceptance Order

Run the following in order. A later model does not replace an earlier focused
test.

| ID | Model | Required properties | Expected |
|---|---|---|---|
| E2E-01 | RGB stem C3 -> C32 | Direct NHWC input, K3/S2/P1 | Exact NHWC reference after output conversion |
| E2E-02 | Pointwise C32 -> C32 | IC33/OC34, group/tail path | Exact output and no intermediate NHWC |
| E2E-03 | Conv3x3 S1 | IC64/OC64, M at most 256 per job | Exact output |
| E2E-04 | Conv3x3 S2 | Odd H/W and border padding | Exact output shape and bytes |
| E2E-05 | Depthwise S1 and S2 | C31, C33, and C65 variants | Exact lane-wise output |
| E2E-06 | Pointwise -> depthwise chain | Internal C32 retained | Exact final output |
| E2E-07 | C32 -> NHWC graph output | C33 or C65 | Exact compact output, no padding lanes |
| E2E-08 | Full Micro-MobileNet | NHWC public buffers, compiler-generated package | No hand-written graph/weights/descriptors; exact model tolerance |

For E2E-08, integer operators covered by this phase must be bit-exact. If the
model contains an operation assigned to a later phase, use a deliberately
declared fallback or keep the model outside the Phase 3 exit; do not silently
claim full-native execution.

## 19. Test Commands

### 19.1 `neural-compiler`

Configure and run the full Regor unit suite:

```bash
cmake -S ethosu/regor -B /tmp/regor-phase3-tests \
  -DCMAKE_BUILD_TYPE=Debug \
  -DREGOR_ENABLE_TESTING=ON
cmake --build /tmp/regor-phase3-tests -j
cmake --build /tmp/regor-phase3-tests -t check
```

Focused Catch2 filters may be used during development, but the task exit
requires the complete `check` target.

### 19.2 `neural-ai` host-runtime tests

```bash
make -C sw/test/compiler_runtime check
make -C sw/test/compiler_runtime firmware
```

Firmware uses the repository's Spatz LLVM/GCC toolchain. Do not replace it with
the host compiler.

### 19.3 `neural-ai` Verilator

Run all compiled-package scenarios:

```bash
make -C sw/test/compiler_runtime sim
```

Run one focused scenario:

```bash
COCOTB_TEST_FILTER=<scenario_name> \
make -C sw/test/compiler_runtime sim
```

Direct cluster tests use:

```bash
make -C hw/rtl/cluster sim \
  COCOTB_TEST_MODULES=<test_module> \
  CLUSTER_SIM_NAME=<unique_build_name>
```

Use a unique `CLUSTER_SIM_NAME` or `SIM_BUILD` for parallel or differently
parameterized runs. The simulator is Verilator by default and the toplevel is
`tb_npu_cluster`.

## 20. Task Completion Record

Each task completion must record:

```text
Task:
Compiler commit:
neural-ai commit:
ABI major/minor:
Compiler unit scenarios passed / required:
Host C scenarios passed / required:
Verilator scenarios passed / required:
Functional seed:
Peak TCDM:
Firmware text + rodata bytes:
Performance counters/baseline:
RTL .sv diff: empty
Known limitations:
```

Latest implementation increment:

```text
Task: Oversized pointwise M stripes, C96 depthwise groups, Conv chain runtime proof, unaligned layout contract, and C++ planner golden coverage
Compiler commit: `9b706dff` plus planner verification commit `f3f61488`
neural-ai commit: `a1a6c254`
ABI major/minor: 1/0
Compiler unit scenarios passed / required: 149 / 149 CTest cases (Phase 3 minimum: 147; 8 planner cases, 104 planner assertions)
Host C scenarios passed / required: 4 / 4 compiler-runtime test binaries; detailed 52-scenario accounting remains open
Verilator scenarios passed / required: M257, M511, C96/S2, pointwise->depthwise chain, unaligned C3/C31 ROW32 round-trip (5 / 5 focused)
Functional seed: deterministic byte patterns and signed per-pixel references
Peak TCDM: 32704 bytes (M511 compiler package)
Firmware text + rodata bytes: 22628 text bytes (30156 total image)
Performance counters/baseline: M511 302448 cycles; C96/S2 245500 cycles; independent chain 206210 cycles; compiler-generated chain 217516 cycles
RTL .sv diff: empty
Known limitations: full E2E-01..08 order, Micro-MobileNet compiler-generated
package, multi-group C65 compiler-runtime boundary simulation, Python-golden
parity, complete byte-golden matrix, detailed host/Verilator scenario
accounting, and broad randomized/layout coverage remain open.
```

A task is not complete when:

- A required test is skipped.
- Only command serialization is tested without runtime execution.
- Only all-one weights are used.
- A tail is padded but not verified at the public output.
- A simulation passes without checking status, done count, output, and guards.
- An `.sv` file was modified.

## 21. Phase 3 Exit Criteria

Phase 3 is complete only when:

- T3.1 through T3.10 meet their individual exit criteria.
- All 170 current compiler, 52 host C, and 73 Verilator scenarios pass.
- E2E-01 through E2E-08 pass in order.
- Micro-MobileNet needs no hand-written graph, weight, qparam, or linebuffer
  descriptor.
- Per-channel bias, scale, rounding, output zero point, and fused clamp match
  the independent reference.
- M257 and M511 are split by the compiler into commands with M at most 256 and
  execute successfully.
- C32 full-group and tail paths are functionally correct and have separate
  performance reporting.
- Peak TCDM and firmware size are reported and enforced.
- Existing legacy graph and v1 runtime tests still pass.
- No `neural-ai/hw/**/*.sv` file has changed.
