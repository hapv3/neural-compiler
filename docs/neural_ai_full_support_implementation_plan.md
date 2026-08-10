# Neural-AI Backend for Vela/Regor: Architecture Analysis and Implementation Plan

## 1. Purpose

This document describes the current `neural-compiler` architecture, the existing
Neural-AI hardware and software contract, the gaps between them, and a concrete
implementation plan for producing executables that run directly on the Neural-AI
NPU.

Allowed scope:

- `neural-compiler` may be changed.
- `neural-ai/sw` may be changed to provide the ABI, runtime, kernel wrappers, and
  tests required by the compiler.
- RTL `.sv` files under `neural-ai/hw` must not be changed. Cocotb tests,
  Makefiles, and simulation tooling may be extended for verification.
- Existing simulation flows must continue to work.

This is a design and implementation guide for running selected full-size YOLO
and MobileNet graphs. It is deliberately model-driven: it does not target full
TFLite or TOSA operator coverage, and an operator is not in scope merely because
it exists in either frontend format.

## 2. Architectural Conclusion

`neural-compiler` is still Vela/Regor with Ethos-U backends. Its README describes
the intended Neural-AI direction, but the source does not yet contain a real
Neural-AI target:

- `ethosu/regor/regor.cpp::regor_create()` only constructs `ArchEthosU55`,
  `ArchEthosU65`, and `ArchEthosU85`.
- `ethosu/regor/include/regor.h` only publishes the three Ethos-U targets.
- The Python CLI still creates `ArchitectureFeatures` from Ethos-U variants.
- The shared scheduler still hard-codes 16-byte alignment and `NHCWB16` layout.
- The linker and writers still package output as Ethos COP1/COP2 data and the
  `ethos-u` custom operator.

The reusable parts are the frontend, Graph IR, optimizer framework, scheduler,
live-range allocator, HLC representation, and architecture interfaces. Neural-AI
needs its own operator constraints, layouts, weight packing, tiling, quantization,
performance model, command generation, executable package, and runtime ABI.

The recommended implementation direction is therefore:

1. Add Neural-AI as a first-class architecture target in Regor.
2. Replace remaining shared Ethos assumptions with architecture-owned APIs.
3. Preserve NHWC logical shapes in the frontend and expose NHWC model bindings.
4. Lower internal activation storage directly to Neural-AI native `ROW32` or
   `C32_BLOCKED`; never route Neural-AI tensors through Ethos `NHCWB16`.
5. Add a versioned `.nai` model package that contains no native pointers.
6. Add a v2 runtime and command ABI in `neural-ai/sw`, while preserving v1.
7. Generate all weights, quantization parameters, tile descriptors, and commands
   in the compiler.
8. Keep model-level planning out of firmware.

The resulting activation-layout flow is:

```text
TFLite/TOSA frontend and Graph IR (logical NHWC)
    -> public input binding (physical NHWC)
    -> explicit boundary conversion or direct native first consumer
    -> internal ROW32/C32_BLOCKED tensors
    -> explicit boundary conversion or direct NHWC final producer
    -> public output binding (physical NHWC)
```

`NHCWB16` remains an Ethos-only physical format. It is neither a TFLite format
nor a Neural-AI intermediate format.

### 2.1. Current Implementation Status

At the current commit, `neural-compiler` already contains a partial Neural-AI
target. The plan must not describe the codebase as if no Neural-AI support
exists. The following status matrix reflects what has been audited:

| Feature | RTL primitive | HAL / runtime | Compiler lowering | RTL E2E |
|---|---|---|---|---|
| DMA package | Yes | Yes | Yes | Yes |
| GEMM32 | Yes | Yes | Yes | Yes |
| RQ load | Yes | Yes | Yes for implemented Conv modes | Focused compiler-generated packages |
| NHWC↔ROW32 | Not a primitive | iDMA for external↔local, scalar/Spatz for local↔local | Yes | Focused boundary regression |
| NHWC↔C32 | Not a primitive | iDMA for external↔local, scalar/Spatz for local↔local | Emitted for Conv, depthwise, and constrained Add boundaries | Focused boundary regression |
| Pointwise Conv1x1 | GEMM32 primitive | v2 `POINTWISE_C32` path | Constrained 1x1/S1/P0 lowering with C32 group/tail padding | Compiler-generated Conv package on Verilator |
| Linebuffer Conv | Yes | v2 linebuffer and depthwise dispatch | RGB K3 S2, generic full-group C32 K3, and depthwise K3 S1/S2 | Compiler-generated `.nai` packages |
| AFU commands | Hardware modes exist | v2 constrained `ADD_I8` and LUT dispatch | Equal-shape raw-safe INT8 Add, byte-exact Conv-producer canonicalization, standalone activations, and quantized YOLO SiLU LUT fusion | Compiler-generated Add, activation, and SiLU packages on Verilator |
| Spatz commands | Engine exists | v2 quantized-Add and nearest-upsample dispatch | Quantization-correct Add fallback and nearest-neighbor 2x INT8 C32 | Compiler-generated packages on Verilator |
| MaxPool | Systolic linebuffer pool mode | v2 constrained MaxPool dispatch | K5/S1/P2, batch 1, INT8 C32, out-of-place | Compiler-generated H24/W24/C32 package on Verilator |

`ArchNeuralAI` exposes `FullyConnected`, `MatMul`, `MemoryCopy`, and a
constrained CNN path containing pointwise Conv, RGB K3 S2, generic full-group
C32 K3, and depthwise K3 S1/S2. Equal-shape INT8 Add uses `ADD_I8` when Regor's
normalized scales are the raw identity, the two input zero points sum to the
output zero point, the clamp spans the full INT8 range, and the AFU output does
not overlap either input. A private Conv producer may shift its output zero
point and clamp only when that representation change is provably byte-exact;
all other scalar-quantized equal-shape Add cases use `SPATZ_ADD`.

The plan should record the hardware commit and compiler commit at which the
contract was audited.

## 3. Definition of Target Support

In this document, target support for the Neural-AI NPU means:

- The input is a selected, static-shape, batch-1 full-size YOLO or MobileNet
  model represented as TFLite, or as TOSA when the same model pattern is already
  covered by the target contract.
- Main activations and weights are signed INT8; bias and partial sums are INT32.
- Native GEMM, FullyConnected, Conv, and pointwise paths require symmetric
  quantization: IFM zero point == 0 and weight zero point == 0. OFM zero point
  is arbitrary INT8, applied by requantization. A corpus-backed Conv with a
  scalar nonzero IFM zero point may be canonicalized exactly by subtracting
  `ifm_zero_point * sum(weights)` from each constant INT32 bias and
  materializing implicit padding with the raw IFM zero-point byte before the
  connection is relabelled as symmetric. The compiler must reject nonconstant
  weights/bias, correction overflow, and every case outside that tested
  transform before scheduling.
- Every operator instance in the selected model corpus uses a documented and
  tested Neural-AI lowering. Operators and parameter combinations outside that
  corpus fail at compile time with an actionable diagnostic unless separately
  promoted into the corpus.
- The compiler produces one `.nai` file without requiring a model-specific
  `main.c`, C graph, hand-packed weights, or hand-written descriptors.
- One generic firmware image can load and execute that file.
- Rank-4 model inputs and outputs use contiguous NHWC storage by default; native
  padded layouts are private implementation details of the compiled graph.
- Output is bit-exact, or within an explicitly defined tolerance, relative to a
  reference implementation.
- Unsupported operators, shapes, layouts, or quantization parameters fail at
  compile time with actionable diagnostics.

Target support does not mean:

- Every TFLite or TOSA operator.
- Every YOLO or MobileNet version, input resolution, detection head, or optional
  block. Support is claimed for named model artifacts in the release corpus and
  for additional artifacts proven to use the same validated contracts.
- Floating-point inference.
- Dynamic shapes or dynamic allocation in firmware.
- Arbitrary kernel, stride, dilation, or padding values beyond RTL limits.
- Generic Softmax, GELU, NMS, or control flow without a verified software kernel.
- Five-cluster scheduling, because the five-cluster top level is still planned.
- Modifying RTL to make a model compilable.

### 3.1. Target Model Corpus and Expansion Policy

The release corpus, rather than the frontend operator catalog, defines scope.
It must contain at least one selected full-size YOLO graph and one selected
full-size MobileNet graph, with model hashes, input shapes, quantization
metadata, and an operator-instance inventory checked into test data or generated
reproducibly. Micro-MobileNet and Micro-YOLO remain fast regression and PMU
baselines; they do not define feature completeness.

Generate each TFLite source inventory with the Regor tool built as
`neural-ai-model-inventory`:

```sh
neural-ai-model-inventory model.tflite > model.inventory.json
```

Generate a mapping micro-graph from reviewed source operator indices with the
same tool. The command writes the TFLite artifact and prints its deterministic
provenance JSON:

```sh
neural-ai-model-inventory model.tflite \
  --micrograph 0:OP0,OP1,OP2 --input-hw 16x16 \
  --output mapping_case.tflite \
  > mapping_case.provenance.json
```

The selector is `SUBGRAPH:OP[,OP...]`. The extractor preserves the selected
operators, their constant tensors and exact producer-consumer edges, promotes
non-constant edges entering or leaving the selection to graph boundaries, and
removes unrelated model buffers. `--input-hw` is optional and changes only the
mapping fixture's spatial extent while propagating the resulting H/W through
the selected Conv, Depthwise, Pad, activation, Add, Mul, Quantize, and other
explicitly admitted topology edges. It preserves channels, weights,
quantization, stride, padding, builtin options, and operator connectivity from
the full graph. This keeps mapping independent of the later SRAM-feasibility
gate. Use the normal inventory command on the generated artifact when a review
needs the full tensor-contract table.

The deterministic JSON records the artifact basename, byte size and explicitly
labelled MD5 identity hash, TFLite schema version, sorted operator counts,
subgraph bindings, every tensor shape/type/buffer/quantization tuple, every
operator edge, and the YOLO/MobileNet-relevant builtin options such as Conv and
pool kernel/stride/padding, fused activation, concat axis, and nearest-resize
flags. It intentionally contains no target-support classification: inventory
describes what the model requests, while the separately reviewed constrained
contracts decide what Neural-AI accepts. Release metadata may additionally
record a SHA-256 digest without changing this source inventory format.

Expected recurring patterns include:

- YOLO: Conv plus fused activation, C2f/bottleneck residual Add, SPPF-style
  MaxPool, nearest-neighbor upsample, channel Concat, detection heads, and only
  the post-processing patterns included in the selected deployable artifact.
- MobileNet: pointwise and depthwise Conv, fused ReLU/ReLU6, residual Add, and
  global average pool. Hard-Swish and squeeze-excitation are required only when
  the selected MobileNet artifact contains them.

For every unsupported node discovered in that corpus:

1. Prefer an existing RTL primitive, compiler fusion, layout view, or
   quantization canonicalization.
2. Add a constrained runtime or Spatz kernel only for a recurring model pattern
   with byte-exact tests and an acceptable measured cost.
3. Extend the command ABI only when existing commands cannot describe that
   model-required lowering.
4. Do not extend RTL for generic TFLite semantics. Any future RTL proposal must
   be justified separately by a selected-model correctness or performance gap.
5. Reject all unvalidated shapes, broadcasts, quantization forms, and operator
   variants explicitly; do not silently route them through a generic firmware
   loop.

The generic firmware image is model-independent packaging and dispatch. It does
not imply generic implementations of every frontend operator.

### 3.2. Current Milestone Scope Freeze

For the current milestone, the implementation target is the smallest validated
set of contracts needed to compile and execute the named full-size YOLO and
MobileNet release artifacts. This scope freeze applies equally to RTL, command
ABI, firmware, compiler legalization, Graph IR lowering, layout conversion, and
software kernels.

An operator or parameter variant enters the implementation backlog only when
all of the following are true:

1. At least one concrete instance exists in a named release artifact and is
   recorded in its reproducible operator and quantization inventory.
2. No already-validated fusion, producer canonicalization, metadata view, or
   existing Neural-AI primitive can implement the instance byte-exactly.
3. The proposed constrained contract states the exact shapes, axes, broadcast
   form, data types, quantization form, and layout used by those model
   instances.
4. Focused negative tests reject adjacent TFLite variants that the contract
   does not cover.
5. The implementation has a model-path performance justification, not merely
   an operator-support-matrix justification.

Therefore, during this milestone:

- Do not add or widen an RTL datapath or mode to implement full TFLite operator
  semantics. RTL `.sv` remains frozen.
- Do not add a generic firmware tensor loop, command opcode, Spatz kernel,
  frontend legalization, or compiler fallback merely to accept more TFLite
  models.
- Do not broaden a supported model pattern into arbitrary broadcasting,
  per-axis quantization, dynamic shapes, ranks, axes, kernels, strides, or
  padding combinations unless another selected artifact requires the exact
  extension.
- Prefer Conv requantization fusion for ReLU/ReLU6 and byte-exact producer
  canonicalization for residual Add; prefer a generated unary LUT for the
  selected SiLU or Hard-Swish pattern. Primitive availability alone does not
  justify exposing generic Add or Mul.
- Treat NMS, decode, dynamic post-processing, generic Softmax, and other
  optional detection-tail operations as host-side graph boundaries unless the
  named deployable YOLO artifact includes them and a separate constrained
  lowering is approved.

The scope may be expanded later by adding another named full-size artifact and
repeating the inventory, contract, correctness, and performance gates. It is
not expanded by implementing the remaining TFLite operator catalog.

## 4. Current `neural-compiler` Architecture

### 4.1. Frontend and Entry Point

The Python CLI in `ethosu/vela/vela.py` currently:

1. Memory-maps a TFLite or TOSA input.
2. Converts accelerator and system configuration into C++ binding arguments.
3. Calls `regor.compile()`.
4. Writes TFLite or raw output.

The compilation core is already in C++ Regor. A separate Python compiler should
not be created for Neural-AI. Python should remain an orchestration layer that
adds a target, configuration, and output format.

Current frontend issues:

- `ArchitectureFeatures` accepts only Ethos-U enums.
- Accelerator-name conversion relies on Ethos naming conventions.
- Output format is limited to `tflite` and `raw`.
- Reporting fields such as MACs per cycle, core count, and AXI width are inferred
  using Ethos-U rules.

Recommended changes:

- Parse the target before creating target-specific architecture features.
- Preserve the existing `ArchitectureFeatures` path for Ethos-U.
- Add a small `NeuralAIArchitectureFeatures` implementation for CLI and reports.
- Add an explicit `neural-ai -> NeuralAI` mapping instead of string heuristics.
- Add a `neural-ai` or `nai` output format with the `.nai` extension.

### 4.2. C++ Compilation Pipeline

The main pipeline in `ethosu/regor/compiler/compiler.cpp` is:

1. Validate architecture configuration.
2. Build or read Graph IR.
3. Run notation-specific and generic Graph IR optimizers.
4. Use `SchedulerPacking` to linearize the graph into scheduler operations.
5. Select stripes and cascades, encode constants, and allocate live ranges.
6. Use `GraphPacking` to group consecutive NPU operations into `CustomNpuOp`.
7. Generate `HLCStripe`, `HLCDMA`, and branch commands.
8. Use the architecture command generator to emit a low-level word stream.
9. Package commands, read-only tensors, scratch, and bindings in a target writer.

Reusable classes and frameworks:

- TFLite and TOSA readers and schema mappings.
- `Graph`, `Operation`, `Tensor`, `Quantization`, and `OpType`.
- Optimizer traversal and rewrite infrastructure.
- Scheduler operation and lifetime infrastructure.
- Linear and hill-climb allocators.
- High-level command representation.
- Optimizer database and performance reporting.

Required Neural-AI implementations:

- `IArchitectureConstraints`.
- `ArchitectureOpConfig`.
- `ArchitectureOpGroup`.
- `WeightEncoder`.
- `ArchitecturePerformance`.
- `IRegisterCommandStreamGenerator`.
- A target graph legalization and layout pass.
- A target executable writer.

### 4.3. Existing Architecture Abstraction

`ethosu/regor/architecture/architecture.hpp` already exposes important hooks:

- Memory areas and their bandwidth and latency.
- Operator configurations and stripe granules.
- Operator grouping and fusion.
- Weight encoding.
- Performance estimation.
- Command stream generation.
- Constraint queries.
- Preferred buffering format.
- Maximum address, scalar support, and supported weight formats.

This abstraction is sufficient to introduce Neural-AI, but it does not fully own
layout and alignment behavior. Add architecture-owned APIs equivalent to:

```cpp
virtual int AllocationQuantum() const;
virtual int TensorAlignment(TensorUsage usage, TensorFormat format) const;
virtual TensorFormat ModelBindingFormat(TensorUsage usage) const;
virtual TensorFormat DefaultInternalTensorFormat(TensorUsage usage,
                                                 bool linearRequired) const;
virtual Shape StorageShape(const Shape &logical, TensorFormat format) const;
virtual int64_t StorageBytes(const Shape &logical,
                             TensorFormat format,
                             DataType dtype) const;
virtual bool CanAliasSlice(TensorFormat format,
                           const TensorSlice &slice) const;
virtual Address SliceOffset(const Shape &logical,
                            TensorFormat format,
                            const TensorSlice &slice,
                            DataType dtype) const;
virtual Shape RollingBufferShape(const Shape &producer,
                                 const Shape &consumer,
                                 TensorFormat format) const;
```

Exact names and signatures may follow local style, but ownership must remain in
`Architecture`, not `scheduler.cpp`.

Default implementations must preserve current Ethos behavior. Neural-AI should
override the per-engine alignment, storage, and layout rules defined in Section
6.2, including 32-byte alignment for GEMM/C32-fast buffers and byte alignment
for generic layout/scalar-only buffers. Binding format selection and internal
format selection must remain separate calls so a scheduler default cannot
accidentally expose a native padded format at the model boundary.

### 4.4. Ethos Assumptions in Shared Code

The following locations must be generalized before the Neural-AI scheduler can
be trusted:

- `compiler/scheduler.cpp`
  - `AllocationQuantum = 16`.
  - `NPUTensorAlignment = 16`.
  - `GetShapeForFormat()` only handles `NHCWB16`.
  - Feature maps default to `NHCWB16`.
  - Concat and split aliasing checks use a depth multiple of 16.
  - Read-only, staging, and I/O allocation use 16-byte alignment.
- `compiler/cascade_builder.cpp`
  - Rolling-buffer depth is always rounded to 16 for `NHCWB16`.
- `compiler/high_level_command_stream_generator.cpp`
  - Strides are only generated for NHWC and NHCWB16.
- `architecture/architecture_constraints.hpp`
  - `TransposeSupport` only describes NHWC and NHCWB16.
- `compiler/scheduler_operation.hpp`
  - Allocation size calls a global helper without architecture context.

Refactoring requirements:

- Do not change Ethos-U output.
- Add allocation and command-stream regression coverage before changing behavior.
- Obtain every target-dependent value through the architecture API.
- Do not scatter `if (target == NeuralAI)` checks across the scheduler.

### 4.5. Graph Optimizer and Supported-Operator Checker

`TFLiteGraphOptimiser` combines target-independent canonicalization with
target-dependent legalization. `MakeSupportedOpsChecker()` currently returns only
an U55/U65 or U85 checker.

Add:

- `TfLiteSupportedOperatorsNeuralAI`.
- `NeuralAIGraphOptimiser`, run after common canonicalization.
- A target hook or registry that selects target passes without asserting Ethos.

The Neural-AI legalization pass should:

- Classify Conv as RGB stem, pointwise, C32 3x3, or unsupported.
- Convert FullyConnected into a suitable MatMul representation.
- Fuse activation clamps into requantization when possible.
- Convert Sigmoid into a 256-entry LUT.
- Preserve frontend NHWC logical shapes and pin public model bindings to NHWC.
- Assign `ROW32` or `C32_BLOCKED` physical storage to internal NPU edges.
- Insert explicit NHWC-to-native and native-to-NHWC boundary conversions unless
  the first or last target operation has a verified direct boundary mode.
- Create views for reshape, slice, and split when byte order is unchanged.
- Insert explicit layout conversion or requantization where required.
- Recognize concat-consumer fusion and model-specific YOLO patterns.

### 4.6. Current Output Path

The existing output path is not suitable for Neural-AI:

- `custom_operator_ethosu.hpp` creates COP1/COP2 headers and Ethos config words.
- `tflite_writer.cpp` hard-codes the `ethos-u` custom code.
- `raw_writer.cpp` assumes `ethos_u_command_stream`, read-only, scratch, and
  scratch-fast tensor wiring.
- The Python binding classifies one blob as TFLite and multiple blobs as Ethos
  raw output.

Neural-AI descriptors must not be embedded in an Ethos COP payload. Add a separate
artifact and writer:

```text
Scheduled graph
    -> HLC
    -> target command generator
    -> neutral CompiledNpuArtifact
         commands
         constants
         tensor bindings
         memory requirements
         debug and performance map
    -> Ethos adapter, preserving current output
    -> NeuralAIWriter, producing .nai
```

To reduce blast radius, the first vertical slice may use a Neural-AI-specific
branch in the link stage. Once GEMM works end to end, extract the neutral artifact
as a shared abstraction. Do not combine a large linker refactor and full Conv
support in one change.

## 5. Current Neural-AI Target

### 5.1. Compute Architecture

The existing contract includes:

- A Snitch RV32IMAC control core.
- A 32x32 INT8 systolic array for GEMM and Conv.
- Spatz for vector, memory, and elementwise kernels.
- An AFU for LUT operations and selected fused post-processing modes.
- iDMA with AXI↔OBI datapaths for transfers between L2 (external) and local
  memory (TCDM). iDMA does not support local-to-local (OBI→OBI) transfers;
  `idma_L1ToL1()` is implemented as a CPU copy. 1D, 2D, and 3D transfer modes
  are available for external↔local only.
- A 256-bit-wide shared TCDM path.
- Command-control MMIO and interrupt completion.

The five-cluster top level and manager core remain planned. Backend ABI v1 must
target one cluster. Do not add `cluster_id`, sharding, or cross-cluster
synchronization to the first command ABI without a hardware contract.

### 5.2. Memory Contract

RTL defines:

- 32 KB ITCM.
- 32 KB DTCM.
- 16 TCDM banks x 32 KB, for 512 KB of physical local memory.
- A 256-bit AXI data path, equivalent to 32 bytes per beat.

`npu_cluster_pkg.sv` declares `I_TCDM_NUM_BANKS = 12` and
`O_TCDM_NUM_BANKS = 4`, but these constants are not used to partition SRAM.
`tcdm_interconnect` selects the bank from `address[8:5]` and all 13 masters
connect to a single 16-bank interconnect with identical SRAM banks. There is
no range check or master-to-bank restriction between "input" and "output".
Each SRAM bank uses 10 bits of bank-local word address, which produces the
following alias model:

```text
bank          = (address >> 5) & 0xF
bank_row      = (address >> 9) & 0x3FF
alias_period  = 32 bytes × 16 banks × 1024 rows
              = 0x80000 bytes
```

Addresses `0x1010_0000`, `0x1018_0000`, and `0x1020_0000` therefore map to the
same physical SRAM location. Snitch D-bus only decodes `0x1010_0000` with mask
`0xFFF0_0000` (window `0x1010_0000`–`0x101F_FFFF`); it does not decode
`0x1020_0000`. Engine ports access TCDM directly and can issue `0x102...`
addresses, but these are aliases, not a separate physical output TCDM.

The canonical TCDM model for the compiler and runtime is:

```text
Canonical TCDM base     0x1010_0000
Physical TCDM bytes     0x0008_0000
Allocatable bytes       0x0007_F000
Command staging         [0x0007_F000, 0x0008_0000)
Physical alias period   0x0008_0000
Bank selection          offset[8:5]
```

Software defines command staging as:

- Base address `0x1017F000`.
- Size 4 KB.
- The command table resides in L2 and is refilled into staging through iDMA.

The compiler allocator must:

- Emit addresses only as `0x1010_0000 + physical_offset`.
- Reject any TCDM offset >= `0x80000`.
- Reserve staging by physical offset, not by an alias address.
- Use 32-byte alignment for ABI sections, command records, GEMM tiles, and
  constant-section bases. Tensor bases, strides, and transfer lengths follow
  the per-engine alignment contract in Section 6.2; do not impose a 32-byte
  requirement on every iDMA row or generic layout buffer.
- Never allocate over command staging.
- Model one physical TCDM arena; there are no separate physical "input banks"
  and "output banks". IFM, weight, OFM, DMA, AFU, and Spatz all contend on
  the same 16 banks.
- If bank-conflict optimization is needed, use bank coloring based on
  `base_offset[8:5]`, not separate address windows.
- Reserve runtime scratch explicitly or make all scratch allocator-owned.
- Fail compilation when peak local memory exceeds the target contract.

Two inconsistencies must be resolved in Phase 0:

- `sw/lib/npu_memory_map.h` declares 8 KB DTCM while RTL and link scripts use
  32 KB.
- Older architecture documentation describes five clusters and a Spatz status
  that is not fully synchronized with current source and regression tests.

RTL source and verified tests are the source of truth. Software headers and
documentation may be corrected; RTL must not be changed.

### 5.3. Firmware Size

ITCM is limited to 32 KB. At the time of this audit:

- `sw/test/micro_yolo/micro_yolo.bin` is 30,712 bytes.
- `sw/test/conv_perf/conv_perf.bin` is 30,900 bytes.

The compiler-driven runtime must remain small:

- Planning, tiling, and layout decisions belong in the compiler.
- Firmware only parses, validates, resolves references, and dispatches HAL calls.
- Build with `-Os`, `-ffunction-sections`, `-fdata-sections`, and
  `--gc-sections`.
- CI must fail when text plus read-only data exceeds 32 KB.
- Use feature profiles only if a universal image cannot fit after dead-code
  elimination.

### 5.4. Current Tensor and Graph ABI

`sw/lib/npu_tensor.h` provides:

- I8 and I32 data types.
- HWC, ROW32, and C32_BLOCKED layouts.
- A 32-bit address, H/W/C, byte size, and quantization metadata.

The software `HWC` name is the batch-1 runtime representation of a contiguous
compiler `NHWC` tensor. Keep `NPU_LAYOUT_HWC` for software ABI compatibility, but
do not add a separate `TensorFormat::HWC` to Regor. For the initial static batch-1
target, map `TensorFormat::NHWC` to `NPU_LAYOUT_HWC` only when constructing the
runtime descriptor.

`sw/lib/npu_graph.h` provides:

- `npu_layer_t`, including tensor indices and native pointers to linebuffer jobs.
- `npu_graph_t`, including native pointers to tensor and layer arrays.
- A large `npu_graph_run()` switch that executes graph operations.

This is a firmware-static C API, not a portable file format. Do not serialize
`npu_graph_t` into `.nai`; pointer size, alignment, relocation, and lifetime are
not defined as a wire contract.

Keep `npu_graph_run()` as a compatibility and reference path. Extract shared
operation handlers so both the graph path and command v2 path call the same
implementation.

### 5.5. Command ABI v1

`sw/lib/npu_cmd_desc.h` defines:

- `NPUC` magic, version 1, and 32-byte alignment.
- `END`.
- `IDMA_1D`, `IDMA_2D`, and `IDMA_3D`.
- `SYSTOLIC_GEMM32`.
- `BARRIER`.
- `ROLLING_BUFFER`.

Limitations:

- Descriptors contain absolute 32-bit addresses.
- There is no model, binding, or constant region abstraction.
- There is no per-channel quantization command.
- There are no complete linebuffer, depthwise, AFU, or Spatz commands; the
  current pointwise path reuses GEMM32 rather than introducing a separate
  pointwise command.
- DMA handlers wait immediately after submission, so DMA and compute do not
  overlap.
- Rolling-buffer commands primarily validate bookkeeping and do not create
  execution overlap by themselves.

ABI v1 already has regression coverage and must continue to work. ABI v2 should
live in new files and be selected by magic and version.

### 5.6. Effective Operator Support

The current hardware and software primitives are broader than the compiler's
release claim. The table below is a candidate-path inventory; a row is supported
only for the constrained patterns exercised and tested by the selected YOLO and
MobileNet corpus.

| Family | Native or available path | Main limits |
|---|---|---|
| DMA | iDMA 1D/2D/3D | Existing graph path is primarily blocking |
| MatMul/GEMM | Systolic GEMM32 | K and N in groups of 32; M is tiled |
| RGB Conv | Linebuffer plus systolic | C3, OC32, K3, S2, P1 |
| Pointwise Conv | Direct GEMM32 | RTL supports grouped C32 operation; compiler currently lowers only 1x1/S1/P0 and pads IC/OC tails |
| C32 Conv | Multi-C32 linebuffer | K3, S1/S2; full C32 groups, plus a compiler-decomposed 16-lane IC tail for explicit-padded/VALID Conv; OC tails are masked at the C32 boundary |
| Depthwise Conv | Depthwise linebuffer | K3, S1/S2, P1; tail lanes masked |
| Requant | Per-channel systolic or Spatz | Shift range 0..31 |
| Logistic/Clamp/activation LUT | AFU LUT | 256-entry LUT, out-of-place; use for selected-model activation fusion or standalone activation only |
| Add | AFU fast mode or Spatz | Equal-shape residual Add patterns selected by model quantization; no generic broadcast contract |
| Mul | AFU `MUL_Q7` or fused LUT | `MUL_Q7` only for a proven Q7 stored-value contract; SiLU/Hard-Swish should be fused to a 256-entry LUT; no generic TFLite Mul contract |
| MaxPool | Systolic or Spatz | Fast path is C32 K5 S1 P2 |
| Upsample | Spatz | Graph contract is nearest-neighbor 2x |
| Global AvgPool | AFU | C32 input, 1x1 output |
| Views | Compiler metadata | Zero-copy only when storage order is preserved |
| Concat | View, fused consumer, or constrained materialization | Only selected YOLO channel-concat patterns; generic N-way concat is out of scope |
| DFL/class sigmoid | Model-specific AFU modes | Not equivalent to generic Softmax |

The backend must support only the subset of these parameter combinations needed
by the target corpus. A primitive's existence does not make the corresponding
generic TFLite operator supported. Snitch or Spatz fallback is acceptable only
for a demonstrated model requirement after byte-exact and performance tests;
otherwise compilation must reject the node.

### 5.7. Requantization Contract

RTL supports per-lane parameters:

- `bias[32]`: signed INT32.
- `multiplier[32]`: signed INT32.
- `shift[32]`: 0 through 31.
- `zero_point[32]`: signed INT32.
- Shared clamp minimum and maximum for the 32-lane group.

Pipeline behavior:

```text
scaled = (accumulator + bias[channel]) * multiplier[channel]
rounded = round_away_from_zero(scaled / 2^shift[channel])
output = clamp(rounded + zero_point[channel], clamp_min, clamp_max)
```

The existing graph convenience path repeats one multiplier and shift across all
32 lanes and sets bias and output zero point to zero. The compiler and runtime v2
must use the per-channel HAL directly.

### 5.8. Weight Layout Contract

Current host tests are the golden oracle for weight ordering:

```text
RGB stem:
  [k_lane=32][oc_lane=32]
  27 valid taps followed by 5 zero-padded taps

Depthwise 3x3:
  [channel_group][kh][kw][lane]

Pointwise:
  [output_group][input_group][k_lane][n_lane]

Conv 3x3:
  [output_group][input_group][kh][kw][k_lane][n_lane]
```

All group and channel tails must be zero padded. Every constant blob starts at a
32-byte-aligned offset.

## 6. Proposed Neural-AI Backend Architecture

### 6.1. Files in `neural-compiler`

Add:

```text
ethosu/regor/architecture/neuralai/
  neural_ai.hpp
  neural_ai.cpp
  neural_ai_constraints.hpp
  neural_ai_constraints.cpp
  neural_ai_op_config.hpp
  neural_ai_op_config.cpp
  neural_ai_weight_encoder.hpp
  neural_ai_weight_encoder.cpp
  neural_ai_performance.hpp
  neural_ai_performance.cpp
  neural_ai_command_stream_generator.hpp
  neural_ai_command_stream_generator.cpp
  neural_ai_linebuffer_planner.hpp
  neural_ai_linebuffer_planner.cpp
  neural_ai_quantization.hpp
  neural_ai_quantization.cpp
  neural_ai_abi.hpp

ethosu/regor/compiler/
  neural_ai_graph_optimiser.hpp
  neural_ai_graph_optimiser.cpp
  neural_ai_writer.hpp
  neural_ai_writer.cpp

ethosu/regor/tflite/
  tflite_supported_operators_neural_ai.hpp
  tflite_supported_operators_neural_ai.cpp

ethosu/config_files/NeuralAI/
  neural-ai.ini
```

Modify:

- `ethosu/regor/include/regor.h`: add `REGOR_ARCH_NEURALAI`.
- `ethosu/regor/regor.cpp`: add the target factory and configuration path.
- `ethosu/regor/architecture/architecture.hpp/.cpp`: add layout, storage, and
  alignment hooks.
- `ethosu/regor/compiler/scheduler.cpp`: remove fixed 16/NHCWB16 assumptions.
- `ethosu/regor/compiler/cascade_builder.cpp`: use target-owned rolling shapes.
- `ethosu/regor/compiler/high_level_command_stream_generator.cpp`: generate
  layout-aware addresses and strides.
- `ethosu/regor/compiler/compiler.hpp/.cpp`: add output format and artifact writer.
- `ethosu/regor/compiler/graph_optimiser.cpp`: add target pass and checker factory.
- `ethosu/regor/CMakeLists.txt`: add sources.
- `ethosu/regor/test/CMakeLists.txt`: add tests.
- `ethosu/vela/architecture_features.py`: add a Neural-AI CLI feature class or
  target registry.
- `ethosu/vela/vela.py`: add explicit target mapping and `.nai` output.
- Python bindings: expose `CompiledNeuralAIModel` instead of inferring output type
  from blob count.

### 6.2. `NeuralAIArchitecture`

The target object owns immutable limits matching current RTL:

```text
target name              NeuralAI
architecture ABI         1.x
clusters                 1
array dimension          32
DMA data beat            32 bytes
alignment                per-engine contract below
physical TCDM            512 KB
command staging          4 KB in the reserved top window
linebuffer max kernel    5
linebuffer max stride    2
linebuffer max input W   640
requant shift            0..31
runtime address width    32 bits after relocation
```

M dimension limits are not a single constant. The compiler must track separate
limits per mode:

```text
MaxGemmCommandMCurrentV2       = 256   (runtime v2 hard-rejects M > 256)
MaxOnChipPartialSumM           = 256   (RTL PSUM_BUF_M for group-stationary
                                        multi-K; larger M falls back to
                                        external TCDM partial-sum path)
MaxHalPlainGemmSoftwareTileM   = 1024  (HAL software tiling helper, not a
                                        command ABI limit)
MaxExternalPsumM               = mode- and TCDM-dependent
```

Because the plan requires compiler-owned tile planning, the runtime must not
tile M internally. The compiler must emit multiple commands with M values valid
for each specific mode.

System configuration may change clocks, bandwidth, and latency for performance
estimation. It must not override hard RTL limits. `CheckConfiguration()` must
reject conflicting values.

Alignment is a property of an address or operation, not of the whole target.
ABI section alignment and engine-specific tensor alignment must remain separate.
The initial contract is:

| Path | Base address | Row/plane stride | Transfer length | Contract |
|---|---:|---:|---:|---|
| `.nai` section and command record | 32 bytes | N/A | command/section size is a multiple of 32 | Software wire ABI |
| Public input/output binding | 32 bytes | Compact logical byte stride; may be non-32 | Exact logical byte size; may be non-32 | Software ABI simplification |
| iDMA 1D/2D/3D external↔local | Compact L2 endpoint is byte-addressed and may be unaligned; compiler-generated native TCDM endpoint remains 32-byte aligned | Compact L2 stride may be non-32; native ROW32/C32 TCDM stride remains 32-byte aligned | Byte units; may be non-32 | Required by NHWC boundary copies; endpoint requirements are asymmetric |
| Systolic GEMM32/psum/requant | 32 bytes | OFM/psum strides obey the command validator and are multiples of 32 | K/N tiles are 32 lanes | Hardware and command ABI |
| Linebuffer RGB/generic | Byte-addressed descriptor fields | Pixel and row strides are byte counts and may be non-32 | Internally reads 32-byte beats and merges crossing beats | Hardware descriptor contract |
| Linebuffer C32 fast/group-stationary | 32-byte C32 group base and group span | C32 pixel/group strides | Full 32-byte channel groups | Hardware fast-path contract |
| AFU | 32-byte base policy for compiler-generated tensors | No tensor row stride in the current job ABI | Length is a byte count and tails use byte enables | Compiler/runtime policy |
| Spatz or Snitch scalar layout kernel | Element aligned | Layout-dependent byte stride | Byte count | Software-kernel contract |

Before the ABI is frozen, RTL tests must verify iDMA 1D, 2D, and 3D transfers in
both directions with unaligned compact L2 endpoints, non-32-byte lengths, and
non-32-byte compact strides, including rows that cross a 32-byte beat boundary.
The native TCDM endpoint used by compiler-generated ROW32/C32 boundary
transfers remains base- and stride-aligned. In particular, test the C=3 and
C=31 NHWC patterns required by `COPY_LAYOUT`. Arbitrary local-unaligned ND
transfers are not part of the frozen compiler contract; if a future lowering
requires one, the runtime must first add a verified aligned bounce-buffer,
Spatz, or Snitch correctness path.

The compiler may retain 32-byte public base alignment as an ABI simplification.
Internal buffers used only by generic layout or scalar kernels may use byte
alignment. Buffers consumed by GEMM, C32-fast linebuffer, or another
32-byte-aligned engine must retain that engine's alignment.

### 6.3. Logical Shapes, Boundary Layout, and Native Tensor Formats

Layout is split into two independent concepts:

- **Logical axis order:** frontend and Graph IR meaning. Rank-4 feature maps stay
  `[N, H, W, C]` throughout import, validation, optimization, and diagnostics.
- **Physical tensor format:** byte ordering chosen during target lowering and
  scheduling. It changes storage and addressing, but never changes logical axis
  meaning.

The initial backend contract is:

1. Preserve TFLite/TOSA rank-4 activations as logical NHWC in the frontend.
2. Pin every public rank-4 input and output binding to contiguous physical NHWC.
   Lower-rank bindings preserve their frontend element order and use contiguous
   storage.
3. Keep `TensorFormat::NHWC` for the public format. Do not add a duplicate HWC
   compiler format; batch 1 lets the runtime map NHWC to `NPU_LAYOUT_HWC`.
4. Add `TensorFormat::ROW32` for MatMul and FullyConnected internal tensors.
5. Add `TensorFormat::C32Blocked` for Conv, pointwise, depthwise, pooling, and
   other compatible internal feature maps.
6. Add distinct target-encoded weight formats for RGB stem, pointwise, Conv, and
   depthwise constants. Keep frontend weight semantics such as OHWI unchanged
   until the weight encoder runs.
7. Never select `TensorFormat::NHCWB16` in the Neural-AI target path. Preserve it
   unchanged for Ethos-U.

The first NPU operation may read an NHWC binding directly only when it has a
verified native boundary mode, initially the batch-1 RGB C3 stem. Otherwise the
schedule contains an explicit `NHWC_TO_ROW32` or `NHWC_TO_C32` conversion. The
last native operation similarly writes NHWC directly only when that mode is
verified; otherwise the schedule contains `ROW32_TO_NHWC` or `C32_TO_NHWC`.
Consequently, padding lanes and blocked channel order are never visible through
the default model API.

Storage sizes:

```text
NHWC bytes      = N * H * W * C * element_bytes
ROW32 bytes     = rows * round_up(C, 32) * element_bytes
C32 bytes       = N * H * W * ceil(C / 32) * 32 * element_bytes
I32 ROW32 bytes = rows * round_up(C, 32) * 4
```

For a MatMul tensor, `rows` is the product of logical dimensions before the last
dimension and `C` is the logical last dimension. Padding bytes are zero-filled
on pack and discarded on unpack.

C32 does not have one affine `strideC` across the full channel dimension. The
target generator must calculate offsets as:

```text
pixel = y * W + x
offset = (((c / 32) * (H * W) + pixel) * 32 + (c % 32)) * element_bytes
```

Do not emulate C32 with an NHWC `Shape strides` value. Slice and address APIs must
be layout-aware and permit zero-copy channel slices only where the boundary is
representable. Every scheduled tensor records both its unchanged logical shape
and its selected physical format.

Native-layout model I/O may be considered later as an explicit opt-in ABI
extension. It is not part of `.nai` ABI v1 and must never be selected implicitly.

This policy keeps the model API compatible with TFLite tooling while avoiding
repeated layout conversion inside Conv or MatMul chains. Its expected cost is at
most one conversion after each unsupported input boundary and one before each
unsupported output boundary. It also prevents the Neural-AI backend from
inheriting the Ethos-specific `NHCWB16` storage contract.

The scheduler may remove a boundary conversion when byte equivalence is proven:

- Contiguous NHWC and ROW32 are identical when the logical last dimension is
  exactly 32 and all preceding dimensions are flattened in the same order.
- NHWC, ROW32, and single-group C32 are identical for a batch-1 feature map with
  exactly 32 channels.
- ROW32 and single-group C32 have the same padded bytes for at most 32 channels,
  but they are not byte-equivalent to compact NHWC when the channel count is less
  than 32.
- Multi-group C32 is group-major and is never treated as NHWC or ROW32 by a
  stride-only relabel.

### 6.4. `.nai` Model Package

Do not add a FlatBuffer parser to firmware constrained to 32 KB. Use fixed-width,
little-endian wire structures and manual serialization.

Proposed header:

```c
typedef struct {
    uint32_t magic;              // "NAIM"
    uint16_t abi_major;
    uint16_t abi_minor;
    uint32_t target_id;
    uint32_t flags;
    uint32_t total_bytes;
    uint32_t section_count;
    uint32_t section_table_off;
    uint32_t entry_command_off;
    uint32_t command_count;
    uint32_t required_tcdm_bytes;
    uint32_t required_tcdm_align;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t reserved[3];
} nai_model_header_v1_t;
```

Section entry:

```c
typedef struct {
    uint32_t type;
    uint32_t flags;
    uint32_t offset;
    uint32_t size;
    uint32_t alignment;
    uint32_t element_count;
    uint32_t reserved[2];
} nai_section_v1_t;
```

ABI 1.1 assigns both trailing section words as reserved zero. The package does
not carry or validate a checksum; firmware validates version, target, reserved
fields, alignment, overflow-safe ranges, required sections, bindings, and
commands without scanning every section payload before dispatch.

Minimum section types:

- `COMMANDS`.
- `CONSTANTS`.
- `TENSORS`.
- `BINDINGS`.
- `QPARAMS`.
- Optional `DEBUG_MAP`.

Each public binding descriptor records direction, logical rank and dimensions,
data type, quantization, byte requirement, and physical format. In ABI v1, a
rank-4 binding format must be NHWC and its byte requirement excludes all native
channel padding. `TENSORS` may describe private `ROW32` and `C32_BLOCKED`
allocations, but `BINDINGS` must not expose those formats. The loader validates
the provided binding size before command execution.

All section offsets and command sizes must be divisible by 32. The parser must
validate ranges as `offset <= total && size <= total - offset`; do not rely on
`offset + size <= total`, which can overflow.

### 6.5. Invocation and Relocatable References

The model package must not know absolute L2 addresses for input and output. The
host creates an invocation record in L2 and writes its address to
`NPU_CMD_L2_BASE`.

```c
typedef struct {
    uint32_t magic;              // "NAIV"
    uint16_t abi_major;
    uint16_t abi_minor;
    uint32_t total_bytes;
    uint32_t model_base;
    uint32_t model_bytes;
    uint32_t binding_table_base;
    uint32_t binding_count;
    uint32_t flags;
    uint32_t reserved[8];
} nai_invocation_v1_t;
```

Note: the implementation uses 8 reserved words (total struct size 64 bytes).
The runtime validates the full 64-byte header. Any new serializer must match
this size.

Every command reference uses:

```c
typedef struct {
    uint16_t region;
    uint16_t index;
    uint32_t offset;
} nai_ref_v1_t;
```

Proposed regions:

- `MODEL_CONSTANTS`.
- `MODEL_COMMANDS` when a command references additional command payload.
- `INPUT_BINDING[index]`.
- `OUTPUT_BINDING[index]`.
- `L2_TEMP_BINDING[index]`.
- `TCDM_SCRATCH`.
- `DTCM_RUNTIME`, available only to runtime-owned data, not arbitrary model refs.

The runtime resolves each reference into an absolute address after checking the
region, index, offset, size, and alignment. The compiler must not embed `0x800...`
addresses in the file.

### 6.6. Command ABI v2

Proposed command types:

```text
END
BARRIER
DMA_1D
DMA_2D
DMA_3D
DMA_SUBMIT_1D        performance phase
DMA_SUBMIT_2D        performance phase
DMA_SUBMIT_3D        performance phase
DMA_WAIT             performance phase
RQ_LOAD
GEMM32
GEMM32_ACCUM
GEMM32_REQUANT
LINEBUF_JOB
POINTWISE_C32
DEPTHWISE_C32
AFU_LUT
AFU_BINARY
AFU_GLOBAL_AVGPOOL
SPATZ_REQUANT
SPATZ_ADD
SPATZ_MUL
COPY_LAYOUT
MAXPOOL
UPSAMPLE_NEAREST
ROLLING_RESET
ROLLING_PRODUCE
ROLLING_CONSUME_RELEASE
```

Each command type must have a clear readiness status:

- Reserved enum ID.
- Wire ABI frozen.
- Runtime parser implemented.
- Hardware path implemented.
- Compiler emission implemented.
- RTL end-to-end verified.

The compiler must not emit a command solely because the enum exists. As of the
current audit, `POINTWISE_C32` also has runtime dispatch and a verified package
path. Remaining composite types are reserved with `UNSUPPORTED` unless marked
optional and skippable.

Composite commands (e.g. `POINTWISE_C32`, `DEPTHWISE_C32`, `LINEBUF_JOB`) may
only exist in the ABI when:

- Semantics are fully fixed.
- The command does not select tile, layout, or mode at runtime.
- All addresses, strides, counts, and modes are compiler-provided.
- A corresponding ABI minor/capability bit exists.

Rules:

- Preserve `type`, `size`, `flags`, `layer_id`, and `tile_id` in each command
  header for diagnostics.
- Every command size is divisible by 32 and every reserved field is zero.
- The runtime rejects unknown required command types.
- A minor-version command may be skipped only when marked optional and skippable.
- `RQ_LOAD` loads one quantization block (32 × 32 = 1024 bytes) into DTCM or
  runtime arrays, allowing several compute commands to reuse it. The load
  involves 1 KB DMA plus 132 MMIO writes (1 disable + 2 clamp + 128
  per-channel register writes + 1 enable). The scheduler should reuse qparam
  blocks across M stripes and treat the RQ register state as a machine resource.
- `LINEBUF_JOB` embeds one job descriptor. The wire format is 160 bytes:
  16-byte command header + 124-byte payload (`systolic_linebuf_cfg_t` 80 bytes +
  `systolic_gemm32_req_t` 36 bytes + `rows` 4 bytes + `k_tiles` 4 bytes) +
  20 bytes zero-reserved padding to satisfy the 32-byte size divisibility rule:
  ```c
  typedef struct {
      nai_cmd_header_v2_t header;    // 16
      nai_linebuf_job_wire_v1_t job; // 124
      uint8_t reserved[20];          // 20
  } nai_cmd_linebuf_job_v2_t;        // 160
  ```
  The runtime command buffer must be large enough to hold 160 bytes.
  Constant deduplication may be added later.
- `COPY_LAYOUT` carries source and destination formats, logical dimensions,
  valid channel count, and target-computed strides. Initial modes are
  `NHWC_TO_ROW32`, `ROW32_TO_NHWC`, `NHWC_TO_C32`, and `C32_TO_NHWC`.
- `COPY_LAYOUT` must be implemented according to the memory spaces involved,
  because Snitch does not have an L2/AXI data path on D-bus. Concrete modes:
  - **NHWC→ROW32, L2→TCDM**: Zero-fill destination native buffer in TCDM, then
    transfer the logical `C * element_bytes` row span with iDMA 2D,
    `source_stride = C * element_bytes`, `dest_stride = round_up(C, 32) *
    element_bytes`, and `repetitions = N * H * W`. The runtime decomposes a
    row span wider than one 32-byte data beat into at-most-32-byte affine 2D
    segments while preserving those strides. This is required by the current
    RTL iDMA path: a wide row combined with a non-32-byte compact stride can
    otherwise leave the engine busy indefinitely.
  - **ROW32→NHWC, TCDM→L2**: iDMA 2D with reversed source and destination
    strides and the same at-most-32-byte row segmentation; padded lanes are not
    copied.
  - **NHWC↔C32 with multiple channel groups**: Requires one transfer per
    channel group, or a verified Spatz gather/scatter kernel. Multi-group C32
    cannot be expressed with a single affine row stride.
  - **TCDM→TCDM (local-to-local)**: Must use Spatz kernel or CPU copy
    correctness path; iDMA cannot perform OBI→OBI transfers. Performance cost
    must be calculated as CPU or Spatz cycles, not as `ceil(bytes/32)`.
  The command semantics do not expose which engine performs the conversion.
- AFU commands must encode the exact hardware mode (`E8`, `E16`, `E32`,
  `MUL_Q7`, `ADD_I8`, `DFL4_ROW32_Q8`, `CLASS_SIGMOID_ROW32_HIGH16`,
  `GLOBAL_AVGPOOL_C32`). `AFU_BINARY` does not imply generic TFLite Add/Mul
  with arbitrary scales and zero points; the runtime must validate the
  quantization contract of each mode.
- Spatz commands are only supported when a tested kernel exists. Spatz is an
  integer-only configuration with two VLSU TCDM ports, not a generic
  high-bandwidth fallback for arbitrary gather/scatter.
- DMA commands: the runtime must derive transfer direction from the resolved
  region of source and destination references. The runtime must reject L2→L2
  transfers and reject a direction field that contradicts the resolved regions.
  Do not create a DMA event ID for local CPU copies.
- The correctness phase uses blocking DMA. Asynchronous commands are added only
  after dependency semantics and regression coverage are stable.

### 6.7. Runtime v2 in `neural-ai/sw`

Add:

```text
sw/lib/npu_model_abi.h
sw/lib/npu_model_loader.h
sw/lib/npu_model_loader.c
sw/lib/npu_cmd_desc_v2.h
sw/lib/npu_cmd_desc_v2.c
sw/lib/npu_runtime_ops.h
sw/lib/npu_runtime_ops.c
sw/runtime/neural_ai/main.c
sw/runtime/neural_ai/start.S
sw/runtime/neural_ai/link.ld
sw/runtime/neural_ai/Makefile
sw/test/compiler_runtime/
```

Modify carefully:

- `sw/lib/npu_cmd_desc.c`: dispatch by v1/v2 magic without changing v1 behavior.
- `sw/lib/npu_memory_map.h`: synchronize DTCM size and canonical constants.
- `sw/lib/spatz_ops.h/.c`: add `_ex` APIs that accept scratch or context while
  preserving old wrappers.
- `sw/lib/npu_graph.c`: extract shared handlers while preserving `npu_graph_run()`.
- `sw/lib/hal_systolic.*`: add wrappers only when existing APIs do not expose a
  required mode.

Runtime main loop:

1. Read command-control registers.
2. Refill the first 32 bytes from L2.
3. If magic is `NPUC`, call the v1 dispatcher.
4. If magic is `NAIV`, parse the invocation and model header.
5. Validate target, ABI, sections, bindings, and scratch requirements.
6. Stream v2 commands through the existing 4 KB staging window.
7. Resolve references and dispatch HAL or shared operation handlers.
8. Update done count, failure code, failure pointer, status, and interrupt.

Do not add model-level tile planning to runtime when the compiler can emit the
required descriptors.

## 7. Detailed Compiler Algorithms

### 7.1. Operator Classification

Each operation must be classified into a concrete `NeuralAIOpMode` in
`NeuralAIArchitectureOpConfig`. A single `OpType::Conv2D` can map to several
datapaths.

Example:

```cpp
enum class NeuralAIOpMode {
    Unsupported,
    View,
    DmaCopy,
    SystolicGemm32,
    SystolicGemm32Requant,
    Conv2DPointwiseC32Requant,
    Conv2DRgbLinebufRequant,
    Conv2DC32Linebuf,
    Conv2DC32LinebufRequant,
    Conv2DC32DownsampleLinebufRequant,
    Conv2DDualSourceC32LinebufRequantL2,
    Conv2DC32MultiLinebufRequant,
    DepthwiseConv2DC32Requant,
    DepthwiseConv2DC32DownsampleRequant,
    LogisticLutI8,
    SiluLutI8,
    HardSwishLutI8,
    ClampI8,
    AddI8,
    MulQ7I8,
    DflSoftmaxI8Q8,
    ClassSigmoidRow32High16I8,
    SpatzRequant,
    MaxPool2DI8,
    UpsampleNearestI8,
    GlobalAvgPoolC32Reduce,
};
```

C32 Conv modes must be separated because RTL only enables multi-K
group-stationary fast mode when all of the following conditions are true
simultaneously:

- `linebuffer enabled`
- `coalesce enabled`
- `kgen enabled`
- `c32_fast enabled`
- `lane_base == 0`
- `block_valid_bytes == 32`
- `input_c >= 32`
- `input_c % 32 == 0`
- `k_tiles > 1`

When these hold, RTL derives `linebuf_c32_group_stationary =
linebuf_kgen_multi`. The `cfg_linebuf_c32_group_stationary_i` field exists in
the config but is not the actual mode selector. A channel tail makes
`block_valid_bytes < 32` and disables the fast multi-K path. Setting
`c32_group_stationary = 1` in the descriptor alone is insufficient.

Consequently, `c32_group_stationary` in the wire descriptor should either be
removed (derived state) or treated as an assertion that the runtime validates
and rejects if the predicate does not match. Different performance formulas
must be used for full-group and tail paths.

AFU command mode names must map directly to hardware modes (`E8`, `E16`, `E32`,
`MUL_Q7`, `ADD_I8`, `DFL4_ROW32_Q8`, `CLASS_SIGMOID_ROW32_HIGH16`,
`GLOBAL_AVGPOOL_C32`), not generic TFLite operator names. Compiler-only modes
such as `SiluLutI8` and `HardSwishLutI8` identify a proven graph fusion and then
lower to `AFU_LUT`; they do not add an RTL mode.

Classifier inputs:

- Operation type.
- Data types and quantization.
- Kernel H/W, stride, dilation, and padding.
- Input and output channels.
- Whether weights are constant.
- Candidate layouts.
- Fused activation.

Classifier outputs:

- Mode.
- Required input, output, and weight layouts.
- Tile granules and hard limits.
- Scratch and partial-sum requirements.
- Whether sub-operations can be fused.
- A diagnostic when unsupported.

The command generator must not infer the mode again. The validated mode carried
through optimization and scheduling is the source of truth.

### 7.2. Layout Propagation

Use a deterministic initial policy:

1. Import and retain logical NHWC axis order for every rank-4 activation.
2. Mark graph inputs and outputs as fixed physical NHWC boundaries before
   selecting internal layouts.
3. Select each operation's native physical input and output formats from its
   validated `NeuralAIOpMode`.
4. Let a verified RGB C3 stem consume its NHWC input binding directly. For every
   other first consumer, insert one explicit NHWC-to-native conversion.
5. Keep activations between Conv, depthwise, pointwise, and compatible pool or
   vector operations in `C32_BLOCKED`.
6. Use `ROW32` between MatMul and FullyConnected operations.
7. Before every graph output, write contiguous NHWC directly when the final
   operation supports it; otherwise insert one explicit native-to-NHWC
   conversion that removes padded lanes.
8. Preserve storage for reshape, squeeze, and flatten only when the logical view
   is compatible with the selected physical byte order.
9. Represent channel slices and splits as views when the start and size align to
   a 32-lane group, or when a consumer descriptor can represent the tail.
10. Avoid materializing concat when the consuming operation can read multiple
    native-layout sources.
11. Insert pack, unpack, or transpose at other edges only when producer and
    consumer native formats are incompatible.

The basic assignment invariant is:

```text
tensor.logical_shape = frontend_shape                  // never reordered
tensor.format = NHWC                                   // model boundary
tensor.format = native_format(producer, consumers)     // internal edge
```

An internal tensor must not fall back to NHWC merely because its logical shape
is NHWC. NHWC physical storage inside the graph is legal only when required by a
supported operation or when conversion elimination proves that a direct boundary
mode preserves the same bytes.

A later performance phase may replace deterministic choices with cost-based
layout planning:

```text
cost(layout on edge) = producer native cost
                     + materialization DMA or vector cost
                     + sum(consumer adaptation cost)
                     + extra peak-memory penalty
```

Run a backward pass to collect required native layouts, then a forward pass to
select layouts and insert conversions. Boundary NHWC constraints are hard; cost
optimization may eliminate a conversion only through a verified direct NHWC
producer or consumer mode. Do not materialize a view only to satisfy logical
NHWC metadata when the target consumer can calculate the native address directly.

### 7.3. TCDM Allocation

Reuse the Regor live-range allocator with these changes:

- Obtain allocation quantum and per-live-range alignment from the architecture.
  Neural-AI must not use one global 32-byte tensor alignment: use the smallest
  allocator quantum supported by the live-range allocator and apply 32-byte
  alignment only to ABI, GEMM, C32-fast, AFU, and other buffers that require it
  under the Section 6.2 contract. Generic layout/scalar-only buffers may be byte
  aligned.
- Obtain storage size from layout-aware APIs.
- Keep constants in `.nai` and L2; stage only the current tile in TCDM.
- Keep contiguous NHWC input and output bindings in L2. Native local tiles and
  conversion scratch have command-schedule lifetimes.
- Reserve a small quantization runtime buffer in DTCM.
- Exclude the 4 KB command staging window from the allocatable arena.
- Remove fixed Spatz scratch addresses from the general arena.

A candidate tile fits when:

```text
live IFM tiles
+ live weight tiles
+ OFM and partial-sum tiles
+ optional ping-pong copies
+ layout conversion scratch
+ runtime local scratch
<= usable physical TCDM
```

The correctness implementation allocates one copy per tile. The performance
phase adds ping-pong buffers and recomputes peak memory.

### 7.4. MatMul and FullyConnected Lowering

Logical inputs:

```text
A: [M, K]
B: [K, N]
C: [M, N]
```

Lowering:

1. Treat external A and C bindings as contiguous frontend-order tensors.
2. Pack A into `ROW32` when it is a model input or arrives in another format.
3. Round K and N up to multiples of 32 and zero-pad input and weight tails.
4. Divide N into output groups of 32 lanes.
5. Divide K into input groups of 32 lanes.
6. Divide M into stripes that fit TCDM; prefer M <= 256 when using partial-sum
   overlap.
7. Load one `[32 K][32 N]` weight tile and the input rows.
8. The first input group writes INT32 partial sums.
9. Intermediate input groups accumulate into those partial sums.
10. The final input group enables per-channel requantization and writes INT8
    `ROW32`.
11. Keep the result in `ROW32` for a compatible internal consumer. For a model
    output, unpack the logical N values per row into the contiguous output
    binding and discard padded lanes.

TFLite FullyConnected weights are commonly `[output, input]`. The encoder must
transpose each group into systolic `[k_lane][n_lane]` order. Tests must include K
and N values that are not divisible by 32.

### 7.5. RGB Stem Conv

Initial native contract:

```text
compiler input layout  NHWC, batch 1
runtime input layout   HWC
input C       3
kernel        3x3
stride        2x2
padding       1x1
output C      32
internal output layout ROW32 or one C32 group
```

The encoder flattens `(kh, kw, ic)` into 27 K lanes and pads to 32. The compiler
precomputes the linebuffer job. This is a verified direct NHWC boundary consumer,
so no input pack command is required. If the graph does not match this contract,
the classifier must not silently select RGB mode.

### 7.6. Pointwise Conv

Lower K1 Conv into GEMM:

```text
M = output_h * output_w
K = input_channels
N = output_channels
```

Recommended loop order:

```text
for output_group in OC32 groups:
    load or reset the partial-sum tile
    for input_group in IC32 groups:
        load the weight tile
        run GEMM32, accumulating after the first group
    load quantization parameters for the output group
    apply final requantization into C32 output
```

The compiler may reuse an IFM tile across output groups when peak TCDM permits.

### 7.7. C32 Conv 3x3

Native modes:

- K3 S1 P1.
- K3 S2 P1.
- Full C32 input groups use the multi-K group-stationary path. A corpus-required
  16-lane input remainder is lowered as nine K1 jobs with external psum
  accumulation and is admitted only when the Conv itself has zero padding
  (the YOLO artifacts materialize padding before the Conv). Arbitrary input
  tails and nonzero-padded tail Conv forms remain rejected. OC tails are
  zero-padded in weights and excluded by the C32-to-NHWC boundary copy.

Main loop:

```text
for spatial tile:
    for output_group:
        for input_group:
            configure a linebuffer job for this C32 group
            the first input_group writes partial sums
            intermediate input_groups accumulate partial sums
            the final input_group requantizes and writes INT8
```

Partial-sum scratch may be reused by output group. The compiler must distinguish
INT8 OFM storage from INT32 partial-sum storage.

### 7.8. Depthwise Conv

Native contract:

- K3 S1 or S2, P1.
- C32_BLOCKED or one ROW32 group.
- Channel multiplier must match the verified kernel contract; reject multiplier
  values other than 1 until tested.
- The weight encoder zero-pads channel tails and the descriptor masks them.

Do not lower depthwise into regular 32x32 dense Conv when a native depthwise mode
exists; doing so would greatly increase MAC and weight traffic.

### 7.9. Linebuffer Tile Planner

Port the logic from
`neural-ai/hw/rtl/cluster/tb/npu_linebuf_precompute.py` to C++. Treat the Python
file as a read-only golden oracle.

Hard constraints:

```text
kernel_h, kernel_w <= 5
stride_h, stride_w <= 2
input tile width <= 640
linebuffer tap count <= 25
requant shift <= 31
preferred tile M = tile_oh * tile_ow <= 256
```

For each candidate output tile `(oh0, ow0, tile_oh, tile_ow)`:

```text
first_y_unpadded = oh0 * stride_h
last_y_kernel    = (oh0 + tile_oh - 1) * stride_h + kernel_h - 1
first_x_unpadded = ow0 * stride_w
last_x_kernel    = (ow0 + tile_ow - 1) * stride_w + kernel_w - 1

first_ih = clamp(first_y_unpadded - pad_h, 0, input_h - 1)
last_ih  = clamp(last_y_kernel - pad_h,    0, input_h - 1)
first_iw = clamp(first_x_unpadded - pad_w, 0, input_w - 1)
last_iw  = clamp(last_x_kernel - pad_w,    0, input_w - 1)
```

Then calculate:

- Input tile base and row and pixel strides.
- Shifted top and left padding.
- Output tile base.
- `block_valid_bytes` for channel tails.
- `c32_fast` and `c32_group_stationary`.
- `channel_addr_offset`.
- `coalesce_k_bytes`.
- K tile count and spatial M.
- OFM and partial-sum row strides.

Candidate search:

1. Enumerate tile H/W from large to small in legal granules.
2. Reject candidates that violate hard RTL limits.
3. Reject candidates that do not fit TCDM.
4. Estimate DMA, compute, and command overhead.
5. Select the lowest cost; break ties in favor of larger tiles and fewer
   commands.

Unit tests must byte-compare the packed 124-byte descriptor with Python golden
data for:

- Border and interior tiles.
- S1 and S2.
- RGB C3.
- One C32 group.
- Multi-C32 group-stationary mode.
- Channel tails.
- Width near 640.

### 7.10. Weight Encoding

Graph IR weights are generally in OHWI order. Emit target blobs using explicit
loops instead of reinterpreting raw tensor storage.

Pointwise and MatMul:

```text
for oc_group
  for ic_group
    for k_lane in 0..31
      for n_lane in 0..31
        dst = weight[oc_group*32+n_lane][ic_group*32+k_lane]
```

Conv 3x3:

```text
for oc_group
  for ic_group
    for kh
      for kw
        for k_lane
          for n_lane
            dst = weight[oc][kh][kw][ic]
```

Depthwise:

```text
for channel_group
  for kh
    for kw
      for lane
        dst = weight[kh][kw][channel_group*32+lane]
```

Write zero for every out-of-range channel or tap. Tests should use randomized
small tensors and an independent reference encoder, not only blob-size checks.

### 7.11. Quantization Algorithm

For output channel `oc`:

```text
real_scale[oc] = ifm_scale * weight_scale[oc] / ofm_scale
```

Find multiplier and shift with:

```text
best = none
for shift in 0..31:
    multiplier = round(real_scale * 2^shift)
    if multiplier is not in the positive signed-int32 range:
        continue
    approx = multiplier / 2^shift
    error = abs(approx - real_scale)
    choose minimum error; break ties with larger shift
```

Reject the operation if no legal representation exists or if the worst-case
product exceeds the width guaranteed by RTL. Host and reference calculations
must use at least 64-bit intermediates.

Bias handling:

- A TFLite INT32 bias is valid when
  `bias_scale[oc] == ifm_scale * weight_scale[oc]`.
- Validate this equality with a strict tolerance.
- Use the integer bias directly when valid.
- Quantize real-valued frontend bias into accumulator units when required.

Output handling:

- Set `zero_point[oc]` from OFM quantization when the datapath permits it.
- Convert fused ReLU, ReLU6, or Clamp into quantized output clamp bounds.
- Give padded tail lanes safe multiplier and bias values and exclude them from
  logical output.

The golden rounding model must match RTL: add half a unit to the magnitude and
round ties away from zero. Do not depend on implementation-defined right shift of
negative C integers.

Zero-point constraints:

- Reject nonzero weight zero points on native dense and Conv paths.
- For a selected INT8 Conv/Depthwise with scalar nonzero IFM zero point,
  constant symmetric weights, and constant INT32 bias, apply the exact
  per-channel correction `bias' = bias - ifm_zp * sum(weights)`.
- Materialize every implicit padding byte as the original raw IFM zero point
  before changing the Conv connection zero point to zero. Folding bias without
  this padding rewrite is forbidden because border positions would be wrong.
- Reject the transform if any corrected bias is outside INT32, if quantization
  is not scalar where required, or if weights/bias are not eligible constants.
- Insert an explicit INT8-to-INT8 affine requant into a symmetric internal
  representation only after that SW or AFU path has verified tests.
- Fail compilation when no loss-safe and overflow-safe internal representation
  is available.

### 7.12. Elementwise Quantization

Elementwise lowering is model-pattern driven. The backend does not promise
generic TFLite Add or Mul semantics, broadcasting, or arbitrary per-axis
quantization.

Add:

```text
real = (q0 - zp0) * scale0 + (q1 - zp1) * scale1
qout = round(real / scale_out) + zp_out
```

Mul:

```text
real = (q0 - zp0) * (q1 - zp1) * scale0 * scale1
qout = round(real / scale_out) + zp_out
```

For residual Add, use `ADD_I8` only when stored representations already match or
when changing a private Conv producer's final requantization is proven
byte-exact. Use the implemented quantization-correct Spatz Add only for a
selected-model case that cannot be canonicalized and passes the model
performance gate. Reject broadcast and unvalidated quantization forms.

Do not add a generic TFLite Mul command or widen the RTL multiplier for this
plan. Lower model-required multiplication as follows:

- Fuse YOLO SiLU (`x * sigmoid(x)`) into a generated 256-entry INT8 LUT and
  execute it with `AFU_LUT`.
- Fuse MobileNet Hard-Swish into a generated 256-entry INT8 LUT when present in
  the selected artifact.
- Use existing AFU `MUL_Q7` only when the exact stored-value Q7 representation,
  truncation behavior, clamp, and non-overlap constraints are proven.
- Add a constrained channel-broadcast multiply for squeeze-excitation only if a
  selected MobileNet artifact requires it and fusion cannot eliminate it. This
  requires its own byte-exact reference and performance gate, but does not imply
  generic two-tensor Mul support.
- Reject all other Mul variants unless a named target model demonstrates the
  need and the plan is revised with evidence.

The performance report must identify the selected fusion or primitive. A scalar
generic tensor loop is not a release path.

Sigmoid and Clamp:

- Generate a 256-byte LUT from input and output quantization.
- Generate fused SiLU and Hard-Swish LUTs from the complete quantized reference
  formula, including zero points, rounding, and clamp.
- Fuse Clamp, ReLU, and ReLU6 into Conv requantization when possible.
- Account for AFU out-of-place restrictions in allocation and liveness.

### 7.13. DMA and Command Scheduling

Initial correctness schedule:

```text
DMA input or weight -> wait
load quantization   -> complete
compute             -> wait
DMA output          -> wait
```

After bit-exact correctness is stable, add an asynchronous schedule:

```text
submit DMA for tile n+1 into a ping or pong buffer
compute tile n
wait for tile n+1 before consuming its buffer
submit output DMA for tile n
barrier only at a true dependency boundary
```

The compiler tracks dependencies by buffer and event ID. It must not reuse a
ping-pong slot before producer DMA and consumer compute are complete.
`ROLLING_*` commands validate occupancy and expected slots to expose schedule
bugs.

### 7.14. Performance Model

Start with a conservative model:

```text
dma_cycles = latency + ceil(bytes / 32)
gemm_cycles = weight_load_cost
            + M pipeline cycles
            + drain and requant overhead
linebuffer_cycles = function(M, kernel taps, K groups, mode)
rq_load_cycles = qparam_dma_cycles(1024)
               + 132 * mmio_write_cost
               + decode_cost
total_blocking = sum(dma + compute + rq_load)
total_overlap = critical_path(max(dma_stage, compute_stage))
```

RTL TCDM arbitration uses strict priority: HWPE > DMA > CORE, with round-robin
within each traffic class. Systolic, Spatz, and AFU are all HWPE class. All
masters share the same 16-bank memory with bank selection from `address[8:5]`.

Consequences for the performance model:

- DMA does not achieve 32 bytes/cycle when compute accesses the same bank.
- CPU layout loops may be delayed under sustained HWPE traffic.
- Two streams with the same base bank phase can conflict continuously.
- Using `max(dma, compute)` for overlap prediction can be overly optimistic.

The performance phase must add:

```text
bank_phase(buffer) = (base >> 5) & 0xF
contention_penalty(engine_a, engine_b, access_pattern)
priority_penalty(HWPE, DMA, CORE)
measured_effective_dma_bandwidth(concurrent_mode)
```

The allocator should prefer different bank phases for IFM, weight, OFM, and
ping-pong DMA buffers. PMU calibration must occur before the cost model is used
for tile-shape decisions, not only as a final optimization step.

For local-to-local copies (Spatz or CPU), the cost model must use CPU or Spatz
cycles, not `ceil(bytes/32)` which assumes iDMA bandwidth.

RQ_LOAD cost is non-trivial for small M with many output-channel groups. The
scheduler should reuse qparam blocks as long as possible, avoid reloading the
same block between M stripes, and treat RQ register state as a machine
resource. When adding async command prefetch, avoid concurrent use of the 4 KB
staging window for command reads and qparam loads unless dependencies are
explicit.

Calibrate with PMU regression for the shapes and paths extracted from the
selected model corpus. Keep only the hardware-boundary points needed to detect
tiling and tail regressions:

- GEMM M values used by model tiles, plus the M=256 command boundary.
- Pointwise IC/OC groups and Conv H/W tiles present in the named artifacts.
- S1/S2 linebuffer and depthwise tail forms present in those artifacts.
- DMA size/stride and AFU/Spatz element-count points generated by those model
  paths, plus minimum and maximum admitted command boundaries.
- Concurrent DMA + compute bank-conflict measurements.

The performance model must not claim overlap while the command runtime remains
blocking.

## 8. Implementation Phases

### 8.1. Mapping-First Full-Graph Bring-Up

Full-size artifacts define the required operator contracts, but they are not
the mapping-phase simulation workload. Bring-up is split into three ordered
gates:

1. **Operator mapping:** use the full graph only as a topology and contract
   source. Inventory every source instance, then generate small compiler test
   micro-graphs that preserve the relevant full-graph producer-consumer
   topology, tensor shapes, quantization, padding, fusion, layout, and fan-out.
   Close semantic lowering and byte-exact correctness only on those generated
   micro-graphs; compiling or simulating the full graph is not a mapping-phase
   test.
2. **SRAM feasibility:** after every selected instance maps, compose larger
   staged prefixes and use Regor/Vela's existing scheduler decomposition,
   cascade builder, live-range analysis, tensor allocator, fast-storage
   allocator, hill-climb allocator, and architecture cost hooks to tile or spill
   the graph within Neural-AI memory limits.
3. **Performance:** only after mapping and memory feasibility close, tune
   cascades, tile choice, DMA overlap, and bank placement against attributed PMU
   measurements.

Mapping micro-graphs are generated automatically or reproducibly from the named
artifact inventory; they are not hand-designed toy operators. Each generated
case records the source artifact hash, subgraph index, source operator indices,
and tensor-contract fields used to reproduce it. It must include enough
neighboring operations to exercise the actual fusion and layout decision, for
example `Conv -> ReLU6 -> depthwise`, a residual `Conv -> Add -> consumer`,
YOLO `Conv -> Sigmoid -> Mul`, or `Concat -> head Conv`. A single isolated
operator is acceptable only when its full-graph behavior has no
producer-consumer dependency.

During the mapping gate:

- Do not require the full graph to fit TCDM and do not optimize operator
  semantics around current SRAM pressure.
- Do not use successful or failed full-graph compilation as a mapping test or
  exit criterion. The full artifact is read only to inventory contracts and to
  select/generate topology windows.
- A conservative schedule or explicit L2 spill is acceptable for a micro-graph
  correctness test.
- Do not run full-graph Verilator. Use compiler unit tests, host reference
  verification, and focused micro-graph Verilator packages instead.
- Do not claim model support from micro-graph success alone. Full-graph compile,
  SRAM feasibility, and final simulation remain later release gates.
- Keep memory optimization in shared scheduler/allocator mechanisms wherever
  possible; do not create model-specific firmware allocation logic.

## Phase 0 - Freeze Contracts and Golden Data

### Objective

Remove ambiguity between compiler, software, and RTL before backend development.

### Work Items

1. Specify ABI version 1 for `.nai`, invocation records, references, and commands.
2. Freeze the initial target as one cluster, INT8, static shape, and batch 1.
3. Freeze physical TCDM size, 4 KB staging, and the per-engine base/stride/length
   alignment contract after the unaligned iDMA RTL gate passes.
4. Correct the software DTCM constant from 8 KB to 32 KB.
5. Freeze logical NHWC semantics, contiguous NHWC model bindings, native ROW32
   and C32 formulas, boundary conversions, and aliasing rules.
6. Document the compiler NHWC to software `NPU_LAYOUT_HWC` batch-1 mapping.
7. Freeze the four weight layouts and quantization rounding behavior.
8. Capture golden command, weight, and linebuffer blobs from existing tests.
9. Add host-side ABI layout tests for size, offset, endianness, and alignment.

### Planned Files

```text
neural-compiler/docs/neural_ai_backend_abi.md
neural-compiler/ethosu/regor/architecture/neuralai/neural_ai_abi.hpp
neural-ai/sw/lib/npu_model_abi.h
neural-ai/sw/lib/npu_memory_map.h
```

### Tests

- C and C++ `static_assert` checks for every wire structure size and offset.
- Serializer byte-golden tests.
- Invalid offset, overflow, and alignment cases.
- iDMA 1D/2D/3D tests with unaligned compact L2 endpoints, non-32-byte length,
  and non-32-byte compact stride in both external↔local directions, including
  C=3 and C=31 `COPY_LAYOUT` row patterns and 32-byte beat crossings. Native
  ROW32/C32 TCDM endpoints remain aligned.
- A cross-repository ABI manifest and version comparison.
- TCDM alias verification must use the access path implemented by RTL: Snitch
  directly verifies `0x1010...`/`0x1018...`, while iDMA verifies the
  engine-only `0x1020...` alias through L2 round trips. Then ensure runtime and
  compiler emit only canonical `0x1010...` addresses.

### Exit Criteria

- ABI review is complete and the file format contains no native pointer.
- Memory constants agree with RTL and linker scripts.
- There is no RTL `.sv` diff under `neural-ai/hw`.
- Existing software and RTL tests still pass.

### Current Verification Evidence

- Raw v2 DMA endpoint resolution is byte-addressed; section and command-record
  alignment remains unchanged.
- Host dispatcher tests cover non-32-byte lengths, offsets, and compact strides
  for DMA 1D/2D/3D.
- `make check` compares compiler and runtime ABI headers through a
  cross-repository manifest covering 6 constants/version fields, 57 enum
  values, and 20 wire-structure sizes.
- A Verilator round-trip regression executes external-to-local and
  local-to-external DMA for 1D length 37, 2D C=3, and 3D C=31, then checks the
  sparse output byte-exactly. It passes at 278,888 simulated ns.
- The `COPY_LAYOUT` C=3/C=31 regression with unaligned L2 bindings passes at
  198,258 simulated ns after wide-row chunking, versus 199,932 ns before it
  (-0.84%).
- The ROW32 boundary matrix round-trips `H=2`, `W=3`, and
  `C=3,31,32,33,63,64,65` byte-exactly. C=63 exposed a pre-fix iDMA busy
  timeout; with row spans split into 32-byte segments, all seven cases pass in
  702,378 aggregate simulated ns. Individual runtime completion is
  97,692-101,628 ns and PMU cycles are 67,574-71,510 as the segment count
  increases.
- The independent-memory regression verifies the `0x1010...`/`0x1018...`
  Snitch alias and the `0x1020...` engine alias through iDMA in both
  directions. It passes at 34,247 simulated ns.
- The runtime firmware is 23,684 bytes of `.text`; no RTL source was
  changed.

## Phase 1 - Runtime ABI v2 and GEMM Execution Skeleton

### Objective

Provide one generic firmware image that reads an invocation and `.nai` model,
executes a DMA/GEMM package, and reports status through existing command-control
MMIO.

### Work Items

1. Add model and invocation parsers with bounds checking.
2. Add region resolution and binding validation.
3. Add a v2 streaming dispatcher using the existing 4 KB staging window.
4. Preserve the v1 dispatcher; `NPUC` magic must continue to use the old path.
5. Implement v2 `DMA_*`, `GEMM32`, `BARRIER`, and `END`.
6. Add a quantization-buffer API skeleton without requiring full Conv support.
7. Add the universal runtime firmware and linker size assertion.
8. Add a host test builder under `sw/test/compiler_runtime`, not under `hw`.

### Planned Files

```text
neural-ai/sw/lib/npu_model_loader.*
neural-ai/sw/lib/npu_cmd_desc_v2.*
neural-ai/sw/lib/npu_runtime_ops.*
neural-ai/sw/runtime/neural_ai/*
neural-ai/sw/test/compiler_runtime/*
```

### Tests

- The unchanged legacy v1 MatMul test still passes.
- Malformed v2 header, section, and reference tests.
- V2 DMA in -> GEMM32 -> DMA out end-to-end test.
- Positive single-command M values 1, 31, 32, 33, 255, and 256.
- Negative single-command M values 257 and 511; each must be rejected with the
  expected bad-command status because runtime v2 does not tile M.
- K values 1, 31, 32, 33, 63, 64, and 65.
- N values 1, 31, 32, 33, 63, 64, and 65.
- Misaligned or out-of-range bindings fail with the expected error code.
- DMA direction validation: reject L2->L2 and direction/region mismatch.
- Quantization rejection: IFM zp != 0, weight zp != 0, shift > 31, invalid
  clamp, mismatched qparam block.
- Firmware text plus read-only data remains at or below 32 KB.

### Exit Criteria

- A GEMM package needs no model-specific C graph.
- V1 and v2 coexist.
- Runtime performs no tile planning.
- Neural-AI RTL `.sv` sources remain unchanged.

### Current Verification Evidence

- The compiler derives `DMA_1D` and `DMA_2D` direction from source and
  destination regions. External-to-external region pairs are rejected during
  command generation.
- The runtime independently validates `DMA_1D`, `DMA_2D`, and `DMA_3D`
  direction against the referenced regions before address resolution. Host
  tests cover matching external-to-local, local-to-external, and
  local-to-local commands, plus direction mismatch and L2-to-L2 rejection
  through both direct and streaming dispatch.
- Runtime host tests reject raw single-command GEMM `M=257` and `M=511` with
  `NAI_DISPATCH_BAD_COMMAND` before invoking the GEMM operation callback.
- Runtime host tests accept raw single-command GEMM M values 1, 31, 32, 33,
  255, and 256 and pass the exact M value to the operation callback.
- Compiler constraints reject nonzero IFM and weight zero points for native
  matrix operations while accepting the full signed INT8 OFM zero-point range.
- Runtime quant-buffer tests reject shift 32, clamps outside `[-128, 127]`,
  inverted clamps, and non-uniform clamp values within one C32 block.
- A Verilator negative package loads qparam block 0 and requests block 1 from
  `POINTWISE_C32`; it fails before compute with the expected operation status
  at 176,665 simulated ns and leaves output untouched.
- Compiler C++ tests pass 183/183 and the Neural-AI compiler-runtime host checks
  pass.
- Compiler-generated Verilator regressions still pass after adding the runtime
  validation and wide-row layout chunking: public Reshape at 97,644 simulated
  ns before the chunking change, generic K3 at 494,008 simulated ns, and RGB K3
  at 195,997 simulated ns.
- Runtime firmware `.text` is 23,684 bytes, below the 32 KB limit.

## Phase 2 - Neural-AI Target Skeleton in Regor

### Objective

Compile an INT8 TFLite FullyConnected, MatMul, or constrained pointwise Conv2D
model into `.nai` and execute it using the Phase 1 runtime.

### Work Items

1. Generalize architecture alignment, layout, and storage hooks.
2. Add Ethos default implementations and regression tests before changing the
   scheduler.
3. Add `REGOR_ARCH_NEURALAI` and the architecture factory entry.
4. Add the Neural-AI memory and configuration object.
5. Add initial MatMul, FullyConnected, and constrained pointwise Conv2D
   constraints (`1x1`, stride 1, dilation 1, zero padding); reject other NPU
   operations.
6. Keep frontend shapes and public bindings in NHWC or their original lower-rank
   contiguous order.
7. Add ROW32 internal layout plus explicit boundary pack and unpack lowering.
8. Add the initial Neural-AI weight encoder.
9. Implement the minimal v2 `COPY_LAYOUT` runtime handler for contiguous-to-ROW32
   pack and ROW32-to-contiguous unpack, including zero-filled tails.
10. Add DMA, layout-conversion, and GEMM command generation. Stage each model
    constant tile in the reserved local TCDM command window before invoking the
    systolic engine; the engine does not consume L2 model-constant addresses
    directly.
11. Add the `.nai` writer, Python result type, and CLI output.
12. Add a Neural-AI config file and explicit CLI target mapping.

### Compiler Tests

- Factory and configuration parsing.
- Per-engine allocation alignment: 32-byte GEMM/C32-fast buffers and byte-aligned
  generic layout/scalar-only buffers.
- ROW32 storage size and tail padding.
- NHWC logical shape preservation through legalization and scheduling.
- Contiguous model input pack to ROW32 and ROW32 output unpack with K/N tails.
- Public binding descriptors reject native formats in ABI v1.
- MatMul weight transpose and packing.
- Command serialization golden data.
- Deterministic `.nai`: compiling twice produces identical bytes.
- Unsupported-operation diagnostics.
- All Ethos unit tests pass without changing their golden output.

### End-to-End Tests

- TFLite MatMul and FullyConnected where K and N are divisible by 32.
- K and N tails not divisible by 32.
- Contiguous frontend-order input and output buffers with no host-side packing.
- Per-channel bias and output clamp.
- Compiler output loads directly without patching absolute command addresses.
- NHWC L2 -> ROW32 TCDM -> NHWC L2 boundary layout round-trip with
  C = 3, 31, 32, 33, 63, 64, 65 and H/W > 1.
- M = 257 and M = 511 to verify that the compiler emits multiple commands, each
  with M <= 256, and that the complete tiled execution succeeds and matches the
  reference.
- Multi-K accumulation with K tail not divisible by 32.

### Exit Criteria

- The CLI successfully creates `.nai`.
- The generic runtime executes compiler-generated files.
- Output matches the reference.
- The model API exposes no ROW32 padding and requires no layout knowledge from
  the host.
- The compiler E2E test contains no manual command builder.

### Current Verification Evidence

- Compiler C++ tests compile pointwise M=257 and M=511 into one full-row
  `POINTWISE_C32` command per output group. The runtime decomposes each command
  into at most 256-row systolic stripes while caching up to three input-group
  weight tiles. This changes compiler/runtime scheduling only; it requires no
  ABI wire-layout or RTL change.
- GEMM weight-packing tests cover K and N boundaries 1, 31, 32, 33, 63, 64,
  and 65, checking every valid byte and every zero-padded lane.
- Focused compiler-generated pointwise M=257 and M=511 packages both pass
  byte-exactly on Verilator with the full-row command contract.
- Two independent compiler instances produce byte-identical `.nai` blobs for
  the same FullyConnected K33/N34 model and configuration.
- A Verilator ROW32 boundary package round-trips contiguous public data through
  native TCDM storage for `H=2`, `W=3`, and
  `C=3,31,32,33,63,64,65`, including the wide compact-stride cases that require
  multiple iDMA row segments.
- This does not relax the raw systolic GEMM M limit: only `POINTWISE_C32`
  carries the full row count, and its runtime handler performs the legal
  at-most-256-row decomposition.

## Phase 3 - Conv, Linebuffer, and Per-Channel Quantization

### Objective

Support the Conv backbones required by the selected full-size YOLO and
MobileNet artifacts: RGB stem, pointwise Conv, C32 Conv, depthwise Conv, and
their fused final-requantization activations. Channel, stride, and padding
coverage is expanded from concrete model instances, not from generic TFLite
Conv parameter space.

Current progress: constrained pointwise Conv1x1, RGB K3 S2, generic full-group
C32 K3, the corpus-required 16-lane IC-tail decomposition, and depthwise K3
S1/S2 lowering are implemented. The generated command streams include
NHWC↔C32 boundaries, multi-group accumulation, per-channel requantization, and
fused ReLU/ReLU6 clamps. Generic K3 accepts full C32 groups and a 16-lane IC
remainder only for zero-padded/VALID Conv; the tail is emitted as nine K1 jobs
because the frozen RTL multi-K fast predicate requires `block_valid_bytes=32`.
Arbitrary input tails and nonzero-padded tail Conv forms remain rejected; OC
tails are zero-padded and masked at the public boundary. Pointwise and
depthwise retain their existing C32 tail support. The
direct RGB stem accepts the corpus-required partial output group (OC 1..32),
and sliced materialized padding supports the full-depth C32 rectangles required
between MobileNet producer and depthwise consumer. Focused
compiler-generated Verilator E2E tests pass for RGB, generic C32, depthwise C33
S2, and a pointwise-plus-depthwise chain. A 13-stage, native-like
Micro-MobileNet topology now compiles through RGB K3 S2, depthwise and
pointwise Conv, two residual Adds, generic K3, GlobalAvgPool, and the
classifier. This required TFLite SAME padding with asymmetric 0/1 sides for
even-sized K3 S2 inputs. ReLU6 is fused into the producing Conv's existing
per-channel requantization, so this graph does not emit a standalone AFU LUT
command. The full compiler-generated graph now passes byte-exactly on the
current Verilator model. Pointwise command coalescing reduces its PMU result to
393,750 cycles, so correctness remains closed while the performance gate stays
open against the documented 347,992-cycle native Micro-MobileNet record.

### Prerequisite Gate

The following must be complete before Conv compiler lowering begins:

1. Physical memory contract frozen: one canonical 512 KB TCDM, no physical
   12/4 partition.
2. `COPY_LAYOUT` L2-aware: iDMA for external↔local transfer; Spatz or CPU for
   local repack.
3. DMA validation by memory region: reject L2→L2 and direction mismatch.
4. Symmetric IFM/weight quantization elevated to target invariant.
5. Hard M limits defined per GEMM/linebuffer mode, not a single `M<=256`
   preferred constant.
6. `LINEBUF_JOB` wire size frozen at 160 bytes (or referenced descriptor
   alternative).
7. C32 generic, fast, group-stationary, and tail modes separated in the
   classifier.

### Work Items

1. Implement the Neural-AI Conv classifier and constraints.
2. Port the linebuffer tile planner from Python to C++.
3. Implement RGB, pointwise, selected C32 K3 Conv2D, and depthwise weight
   encoders required by the corpus.
4. Implement per-channel multiplier and shift search plus the RTL rounding model.
5. Add `RQ_LOAD`, linebuffer, pointwise, and depthwise commands.
6. Extract runtime operation handlers from `npu_graph_run()` where appropriate.
7. Add C32 layout propagation and TCDM partial-sum allocation while leaving
   logical Graph IR shapes in NHWC.
8. Extend the v2 `COPY_LAYOUT` handler and command generation with NHWC-to-C32
   and C32-to-NHWC boundary materialization. Stage compact public RGB C3 inputs
   in TCDM before direct-NHWC linebuffer consumption; public binding metadata
   must retain the original TFLite shape, type, scale, and zero point.
9. Implement the IC/OC group loops and channel-tail forms observed in the
   selected artifacts; reject other tail/mode combinations.
10. Fuse ReLU, ReLU6, and Clamp into final requantization.

### Compiler Tests

- A constraint matrix for every corpus-backed Conv contract, with nearby
  kernel/stride/padding/quantization variants proving fail-closed behavior.
- Weight byte-golden tests for all four formats.
- Randomized quantization tests against an RTL-compatible reference.
- Border and interior linebuffer descriptor byte-golden tests.
- 160-byte `LINEBUF_JOB` command serialization golden tests.
- Tile-fit and overflow tests at the TCDM boundary.
- Channel/group cases extracted from the selected artifacts. Retain C32
  boundary and one tail case per admitted lowering as safety tests; do not add
  arbitrary channel sweeps as a feature target.
- Full C32 group and corpus-required tail cases with separate performance
  validation.
- M = 256 and M > 256 external-psum path tests.
- NHWC L2 → C32 TCDM → NHWC L2 boundary layout round-trip.
- NHWC input and output byte-order tests across H/W/C and channel-tail cases.

### Mapping Micro-Graph Order

1. RGB stem C3 -> one C32 storage group, including the corpus-required OC16 and
   OC32 logical tails.
2. Pointwise C32 -> C32.
3. Conv3x3 S1.
4. Conv3x3 S2.
5. Depthwise S1/S2 with channel tails.
6. Pointwise plus depthwise chain.
7. C32 -> NHWC graph output.
8. Full Micro-MobileNet graph with NHWC host buffers as a fast regression.
9. Corpus-derived full-size MobileNet topology micro-graphs, including adjacent
   producer/activation/consumer and residual contexts. Full-size graph compile,
   TCDM-fit work, and end-to-end simulation are deferred to the SRAM-feasibility
   and release gates.

### Current Verification Evidence

- Compiler C++ tests: 221/221 pass after the current asymmetric-Conv,
  constant-padding serialization, sliced C32 boundary, wide compact-NHWC
  boundary, and mapping-fixture changes. The command generator now places
  nonzero Pad fill tensors in `ModelConstants`; padding DMAs no longer read an
  implicitly zeroed TCDM address.
- A topology-derived MobileNet fixture selected from full-graph operators
  `Conv -> Depthwise -> Pointwise`, spatially cropped to 16x16, invokes in the
  TFLite reference interpreter and compiles to a 7,360-byte `.nai` package with
  0 CPU operators, 12 NPU operators, 5.12 KiB peak SRAM, and 104,448 MACs.
- A topology-derived YOLOv8 fixture selected from full-graph operators
  `Pad -> Conv(OC16) -> Logistic -> Mul`, spatially cropped to 16x16, invokes in
  the TFLite reference interpreter and compiles to a 3,808-byte `.nai` package
  with 0 CPU operators, 8 NPU operators, 4.00 KiB peak SRAM, and 19,456 MACs.
  The compiler fuses `Logistic -> Mul` into one `AFU_LUT`; Verilator completes
  all 10 runtime commands and matches all 1,024 output bytes against TFLite
  `BUILTIN_REF` at 67,650 PMU cycles. The same TensorFlow interpreter with its
  default XNNPACK delegate differs at 7 of 1,024 outputs by one LSB, so mapping
  goldens must explicitly select `BUILTIN_REF` rather than silently changing
  the compiler's documented integer-reference contract. The focused run is
  19.4% of the native Micro-MobileNet record and 17.4% of the native
  Micro-YOLO record. These are diagnostic mapping results, not full-graph SRAM
  or equivalent performance claims.
- The isolated corpus-derived YOLOv8 stem Conv
  (`18x18x3 -> 8x8x16`, K3/S2/VALID) stages its 972-byte compact public input
  into TCDM and passes all 1,024 output bytes against TFLite on Verilator. The
  public `.nai` binding retains NHWC `1x18x18x3`, scale `1/255`, and zero point
  `-128`; the staged DMA destination is the same address consumed by the
  linebuffer. The run completes at 55,592 PMU cycles, 16.0% of the native
  Micro-MobileNet record and 14.3% of the native Micro-YOLO record. These are
  diagnostic ratios for a cropped single operator, not equivalent full-graph
  performance comparisons.
- The next isolated YOLOv8 Conv (`18x18x16 -> 8x8x32`, K3/S2/VALID) now maps
  the corpus-required C16 input tail without widening RTL. Compact C16 boundary
  and Pad slices use 16-byte-pixel DMA3D copies into C32 storage; weights are
  encoded as nine tap-major 32x32 tiles with lanes 16..31 zero. Because the
  frozen RTL enables automatic group-stationary multi-K only for a full
  32-byte block, the compiler emits nine K1 linebuffer jobs and accumulates
  them as `initialize, intermediate..., final-requant` rather than incorrectly
  advertising one nine-tile job. The 12,608-byte package completes all 13
  runtime commands at 89,110 PMU cycles, 25.6% of the 347,992-cycle native
  Micro-MobileNet record and 23.0% of the 388,146-cycle native Micro-YOLO
  record. It matches the default TFLite XNNPACK execution in all 2,048 bytes;
  TFLite `BUILTIN_REF` differs at eight occurrences in one channel by exactly
  one LSB (`-26` versus `-27`). This is recorded as the same frozen
  single-round semantic-fidelity issue already accepted for the MobileNet RGB
  stem, not hidden as a byte-exact `BUILTIN_REF` result and not used to justify
  a generic TFLite or RTL extension. These ratios are diagnostic for a cropped
  operator, not equivalent full-graph performance claims.
- The isolated YOLO topology tail Conv (`8x8x48 -> 6x6x32`, K3/S1/VALID)
  validates the generic input-channel remainder path. The compiler emits one
  full C32 linebuffer job followed by nine K1 jobs for the 16-lane tail,
  preserving the external partial-sum sequence and the C32 plane addresses;
  no RTL source is changed. Vela reports 0 CPU operators and 3 NPU operators,
  and the 21,984-byte package completes all 14 runtime commands byte-exactly
  on Verilator at 91,916 PMU cycles. That is 26.4% of the native
  Micro-MobileNet record and 23.7% of the native Micro-YOLO record; these are
  diagnostic single-operator ratios. A C32 IC32 control run passes at 58,476
  PMU cycles, isolating the tail decomposition cost.
- The complete 13-stage Micro-MobileNet compiler test passes every graph prefix
  and the full graph byte-exactly. The full package contains two `AFU_BINARY`
  residual Adds, one `AFU_GLOBAL_AVGPOOL`, and the expected Conv/linebuffer
  commands, with no `AFU_LUT` activation command.
- The compiler-generated Micro-MobileNet stage-1 stem Conv
  (`96x96x3 -> 48x48x32`) is a byte-exact PASS on Verilator at 83,786 PMU
  cycles. Its PMU attribution is 41,801 retired Snitch instructions, 2,304
  systolic compute cycles, 3,426 iDMA busy cycles, and 248 TCDM stall cycles.
  This is a single stem-Conv attribution run from
  `test_compiler_generated_micro_mobilenet_stage1`, not an equivalent
  full-graph comparison.
- The compiler-generated Micro-MobileNet stage-10 prefix remains a byte-exact
  PASS on Verilator after pointwise coalescing. Its command count falls from 63
  to 39, PMU cycles from 338,498 to 287,654 (-15.0%), and completed iDMA
  operations from 227 to 139. This remains attribution only: the prefix has a
  36,864-byte public output boundary and is not equivalent to either full-graph
  workload. The source test is
  `test_compiler_generated_micro_mobilenet_stage10`.
- Neural-AI compiler-runtime host ABI/layout/quantization checks pass.
- Compiler-generated Verilator packages pass byte-exactly:
  - RGB K3 S2 C3 -> C32: 195,997 simulated ns after wide-row chunking, versus
    195,851 ns immediately before it (+0.07%) and 194,651 ns before DMA
    direction validation (+0.69% cumulative).
  - Generic K3 S1 C32 -> C32: 494,008 simulated ns after wide-row chunking,
    versus 493,826 ns immediately before it (+0.04%) and 492,662 ns before DMA
    direction validation (+0.27% cumulative).
  - Depthwise K3 S2 C33 tail: 228,121 simulated ns.
- The corpus-derived MobileNet RGB stem (`16x16x3 -> 8x8x32`) passes
  byte-exactly against the frozen RTL single-round golden at 58,874 PMU cycles.
  Its raw `ifm_zp=-1` right/bottom padding is copied from serialized
  `ModelConstants`, closing a pre-fix 38-byte border mismatch. Against the
  TFLite interpreter, 64 of 2,048 outputs in one extremely small-scale channel
  differ by exactly one LSB because TFLite double-rounding is not the frozen
  systolic single-round contract. This is recorded as an open semantic-fidelity
  decision for the later release gate, not hidden by the mapping gate and not
  used to justify an RTL extension. The focused operator consumes 16.9% of the
  347,992-cycle Micro-MobileNet record and 15.2% of the 388,146-cycle
  Micro-YOLO record; those ratios are diagnostic only because the workloads are
  not equivalent full graphs.
- The isolated corpus-derived MobileNet Depthwise stage (`8x8x32 -> 8x8x32`)
  compiles with 0 CPU operators and passes byte-exactly against TFLite on
  Verilator at 59,840 PMU cycles. Its asymmetric public input uses one
  constrained iDMA 2D rectangle from compact NHWC `C=32` into padded C32
  storage (256-byte source rows and 320-byte destination rows). This does not
  admit generic NHWC channel tails. The focused run is 17.2% of the
  Micro-MobileNet cycle record and 15.4% of the Micro-YOLO record; these are
  diagnostic ratios rather than equivalent full-graph comparisons.
- The selected full-size MobileNet final depthwise topology (`7x7x960`,
  `DepthwiseConv2D`) now compiles with 0 CPU operators and 7 NPU operators.
  The current compiler emits one compact-NHWC-to-C32 DMA3D transfer per C32
  group (30 input transfers for C=960), and the fresh package SHA-256
  (`7309ca6d6059c0068448441e527771fb6777b94a805f83e9f18a0d8d60eeb0ad`)
  matches the package consumed by simulation. The corrected runtime harness
  keeps disjoint ranges: input `0x80000000`, output `0x80020000`, invocation
  `0x80040000`, binding table `0x80048000`, and model `0x80100000`. The full
  Verilator runtime completes all commands at 521,518 PMU cycles. Its output
  differs from the TFLite `BUILTIN_REF` golden at 13 of 47,040 bytes, every
  delta `-1`; a compute-prefix run stopping before `CopyLayout` reproduces the
  same 13 bytes, proving the layout and DMA are exact. TensorFlow's default
  XNNPACK output matches RTL in all 47,040 bytes, while `BUILTIN_REF` differs
  at those 13 bytes. This is the already documented frozen single-round versus
  double-round semantic-fidelity issue, not an RTL defect and not an accepted
  byte-exact `BUILTIN_REF` result. The earlier 1,566-byte/210-DMA2D report is
  invalid because its ad-hoc harness aliased input with output and model with
  the binding table; no DMA2D mitigation is part of the current mapping.
- The following corpus-derived MobileNet Pointwise stage
  (`8x8x32 -> 8x8x16`) compiles with 0 CPU operators and passes byte-exactly
  against TFLite on Verilator at 53,174 PMU cycles. The focused run is 15.3%
  of the Micro-MobileNet cycle record and 13.7% of the Micro-YOLO record; these
  remain diagnostic, non-equivalent workload ratios.
- The complete corpus-derived MobileNet mapping chain
  `RGB Conv -> Depthwise -> Pointwise` (`16x16` crop) passes byte-exactly
  against TFLite with all 16 runtime commands completed at 105,334 PMU cycles.
  Serializing the asymmetric padding constants reduced this from the pre-fix
  114,096-cycle run by 8,762 cycles (7.68%). The chain is 30.3% of the
  Micro-MobileNet record and 27.1% of the Micro-YOLO record; these remain
  diagnostic ratios for a spatially cropped mapping graph, not equivalent
  full-graph performance claims. The RGB stem's isolated one-LSB rounding
  difference is absorbed by downstream requantization and does not appear at
  the chain output.
- The optimized full package is 87,008 bytes, contains 57 runtime commands
  (down from 81), and retains a 290,816-byte peak TCDM allocation. Direct
  external-to-local compact tensor transfers avoid CPU-backed local-to-local
  bounce copies, and one pointwise `RQ_LOAD` is reused by the full-row
  `POINTWISE_C32` command for each output group. The model reader copies aligned
  ABI records as 32-bit words while retaining a byte-copy fallback for unaligned
  reads.
- The equivalent full-graph run remains a byte-exact PASS at 393,750 PMU cycles,
  down from 443,524 (-11.2%). Pointwise tile reuse reduces iDMA busy cycles from
  6,959 to 5,231.
- The current native Micro-MobileNet rerun is a PASS at 344,454 total PMU
  cycles, a -3,538-cycle difference from the documented 347,992-cycle
  regression reference (-1.02%); the documented value remains the regression
  baseline. The optimized compiler full graph is 1.143x this current native
  run and 1.131x the documented baseline. Its 1.014x ratio to the 388,146-cycle
  Micro-YOLO record is diagnostic only because the graphs differ. Native layers
  0-21 sum to 212,026 cycles, and all 28 traced layers sum to 324,860 cycles,
  leaving 19,594 cycles of total-run overhead outside the traced layers. The
  native rerun artifacts are
  `/tmp/neural-ai-native-mobilenet-pmu.log` and
  `/tmp/neural-ai-native-mobilenet-results.xml`.
- Optimization evidence logs are `/tmp/neural-ai-perf-opt1-iter2-m257.log`,
  `/tmp/neural-ai-perf-opt1-iter2-m511.log`,
  `/tmp/neural-ai-perf-opt1-stage10-iter1.log`, and
  `/tmp/neural-ai-perf-opt1-full-iter1.log`; each has a matching
  `-results.xml` artifact.
- The rebuilt runtime firmware reports `.text=29,936`, `.data=68`, and
  `.bss=6,081` bytes via `llvm-size -A`. Text remains below the 32 KiB ITCM
  limit, and data plus BSS is 6,149 bytes, below the 32 KiB DTCM limit.
- The focused package times include boot, command loading,
  boundary layout DMA, and output checking. They must not be compared as
  operator latency against the PMU-only Micro-MobileNet and Micro-YOLO records.
  The documented regression baselines remain 347,992 total cycles for
  Micro-MobileNet and 388,146 total cycles for the Micro-YOLO raw-head graph.
  The optimized compiler graph is 1.131x the matching documented
  Micro-MobileNet record; its 1.014x Micro-YOLO ratio is diagnostic only because
  the graph differs. The matching MobileNet gap does not close the performance
  gate. Further work must attribute the remaining command/runtime overhead
  without weakening byte-exact correctness or changing frozen RTL.

### Exit Criteria

- The selected full-size MobileNet artifact no longer requires a hand-written
  graph, weights, or descriptor, and every Conv/depthwise instance matches a
  validated constrained mode.
- Micro-MobileNet remains byte-exact and its 347,992-cycle native record remains
  the regression/performance attribution baseline.
- Per-channel bias and scale match the reference.
- Peak TCDM and firmware size are reported and enforced.
- Existing hand-written model tests still pass.

## Phase 4 - Vector, AFU, View, and Layout Operations

### Objective

Support only the non-Conv patterns exercised by the selected full-size YOLO and
MobileNet graphs. The operator matrix is a primitive inventory, not a mandate
to implement generic frontend semantics.

Current progress: Add uses a three-way lowering policy. The 64-byte v2
`AFU_BINARY` `ADD_I8` command is selected when the normalized Add is a raw
stored-value sum: both input scales are the input identity, the output scale is
the inverse identity, the input zero points sum to the output zero point, and
the output clamp is full INT8. If exactly one input is the private output of a
Conv, the graph optimiser may shift that Conv's requantization zero point and
clamp together with the Add input representation. It does so only when both
shifted clamps and the shifted zero point stay in INT8, which proves the stored
bytes are exactly `q' = q + delta`; the resulting Add then uses AFU. Other
equal-shape, batch-1, scalar-quantized INT8 Add cases use the 96-byte v2
`SPATZ_ADD` command and an integer-only quantization-correct kernel. All three
paths keep tensors in C32 blocked storage and require out-of-place output.
Broadcasting and non-scalar quantization remain unsupported and are rejected.

Internal `Reshape`, `Squeeze`, and `ExpandDims` are admitted to Regor's existing
reshape-removal pass; a view-to-Add regression verifies that none of the three
adds a runtime command.
The admitted zero-copy subset must preserve the innermost channel depth, which
keeps the existing ROW32/C32 physical interpretation stable; depth-changing
views are rejected by both the TFLite checker and the Graph IR alias hook until
an explicit materialization path is implemented.
When one of these views is a public graph output, the compiler materializes it
through two `DMA1D` commands and a TCDM bounce buffer, because the current
runtime cannot issue a direct L2-to-L2 copy. C++ tests cover public `Reshape`,
`Squeeze`, and `ExpandDims`; a compiler-generated public `Reshape` package also
passes byte-exactly on Verilator at 97,644 simulated ns after DMA direction
validation, versus the previous 97,150 ns (+0.51%). The existing two-DMA
runtime fixture previously passed on the same simulator build at 91,973
simulated ns. That fixture moves 32 bytes while the public view moves 128 bytes,
so their completion times are only focused sanity checks and not an
operator-latency comparison. View offset/slice cases remain open; command
generation now rejects sliced `MemoryCopy` connections instead of incorrectly
treating them as full-volume copies.

The raw-AFU compiler-generated Add package passes byte-exactly on Verilator at
127,826 simulated ns. The equivalent hand-written package passes at 113,958
simulated ns; the 13,868 ns (12.17%) difference is consistent with additional
parsing/validation overhead for a 1,184-byte compiler artifact versus an
800-byte fixture, while both execute four commands over the same 128-byte
tensors. These focused completion times include boot and runtime overhead and
are not substitutes for the 347,992-cycle Micro-MobileNet or 388,146-cycle
Micro-YOLO full-graph PMU gates.

The compiler-generated requantized H2/W2/C32 Add package emits
`COPY_LAYOUT`, `COPY_LAYOUT`, `SPATZ_ADD`, `COPY_LAYOUT`, and `END`, and passes
an independent byte-exact reference on Verilator at 96,178 simulated ns. Its
57,386 PMU cycles include firmware boot, command parsing, public L2 transfers,
and three boundary conversions, so this focused package is a correctness and
ABI check rather than an Add datapath comparison. The standalone 32-element
kernel also passes byte-exactly at 13,017 simulated ns.

A diagnostic standalone run over 73,728 elements, matching the tensor volume
of the first Micro-MobileNet residual, did not finish within 2,000,000 cluster
cycles. That incomplete lower bound is already 5.75x the 347,992-cycle native
Micro-MobileNet record and 5.15x the 388,146-cycle Micro-YOLO raw-head record.
It therefore rejects generic `SPATZ_ADD` as a model hot path: model performance
depends on selecting raw AFU Add directly or proving the Conv-producer
canonicalization byte-exact. `SPATZ_ADD` remains the quantization-correct
fallback for uncommon, non-canonicalizable cases.

Full-spatial INT8 AvgPool is now lowered through a separate 64-byte
`AFU_GLOBAL_AVGPOOL` command. The supported subset is batch 1, output 1x1,
stride and dilation 1, zero padding, equal scalar IFM/OFM quantization, no
fused clamp, and C32-blocked internal input/output. Asymmetric INT8 zero points
are valid when they are equal because averaging preserves the stored-value zero
point. Other AvgPool shapes and any required requantization are rejected.
Compiler tests cover positive C33 and MobileNet-scale H24/W24/C64 cases plus
non-global, requantized, and fused-ReLU rejection. Host runtime tests cover
dispatch, zero dimensions, overflow-safe ranges, and input/output overlap.

The compiler-generated H2/W3/C33 package passes byte-exactly on Verilator at
114,570 simulated ns and 83,712 PMU cycles. It observes 14 AFU TCDM requests,
which matches 6 pixels times 2 C32 groups plus 2 output groups. The existing
native H7/W5/C65 AFU regression remains byte-exact at 514 PMU cycles and 108
AFU TCDM requests, exactly 35 pixels times 3 groups plus 3 output groups. The
native regression starts with tensors already in TCDM, while the compiler
package includes boot, package validation, 21 DMA operations, two boundary
layout conversions, and public L2 I/O. Their cycle counts are therefore
datapath and integration checks, not an operator-speed ratio. The complete
compiler-generated graph now supplies the missing full-graph comparison:
393,750 cycles, or 1.131x the 347,992-cycle Micro-MobileNet record. Its 1.014x
ratio to the 388,146-cycle Micro-YOLO record is diagnostic because that graph
differs. Focused AFU results remain useful for attribution but do not replace
the matching end-to-end gate.

The Neural-AI AFU is covered by standalone `afu_ops` tests and by the native
Micro-MobileNet/Micro-YOLO graphs. The generic stream FSM prioritizes final
tensor completion over a coincident 32-byte destination boundary, so an aligned
final E8, E16, or E32 beat performs the final flush instead of requesting a
nonexistent next input beat. Block-level final-boundary cases pass in all three
element widths. Focused full-length Clamp/ReLU6 and Logistic tests pass, and the
native Micro-MobileNet graph passes byte-exactly with its three standalone
Clamp layers. Conv clamps remain fused into requantization because that is the
more efficient graph lowering.

The 64-byte v2 `AFU_LUT` command path is implemented for standalone INT8
Sigmoid and clipping. The compiler substitutes Sigmoid, ReLU, ReLU0To1, ReLU6,
ReLUN1To1, and explicitly bounded Clamp with a 256-entry LUT when the clipping
operation is standalone. A source-fused activation remains attached to its
producer and uses the existing final-requantization clamp. LUT lowering requires
equal static batch-1 IFM/OFM shapes, inserts C32 boundary conversions, allocates
the output out-of-place, and rotates Regor's signed `-128..127` LUT order into
the AFU's raw-byte `0..255` index order. Runtime dispatch accepts LUT data in
model constants or TCDM. Model-constant LUTs are copied from L2 into the
reserved TCDM command-staging window with iDMA before firmware programs the AFU
MMIO entries; the control core does not directly load L2 bytes.

Compiler tests validate all 256 Sigmoid and ReLU6 LUT entries and reject invalid
shape, batch, and transform cases. They also verify that source-fused Conv ReLU
and ReLU6 remain in qparams and emit no `AFU_LUT`. Command generation and
op-group rules enforce out-of-place storage and prevent unsupported activation
fusion. The compiler-generated H2/W3/C33 Sigmoid and ReLU6 packages both pass
byte-exactly on Verilator at 88,766 simulated ns. This focused time includes
boot, package parsing, boundary copies, LUT DMA staging, 256 MMIO writes, and
output transfer. It is not comparable to the full-graph PMU records. The
current compiler Micro-MobileNet graph fuses Conv clamps and emits no LUT, while
the native Micro-YOLO record uses its specialized raw-head class-sigmoid path.
The relevant matching full-graph performance comparison is therefore 393,750
compiler Micro-MobileNet cycles versus the documented 347,992-cycle record and
the current 344,454-cycle native rerun. The 388,146-cycle Micro-YOLO comparison
remains diagnostic because it is a different graph.

Quantized YOLO SiLU is implemented as a graph-pattern fusion rather than as
generic TFLite Mul. Before supported-operator checks, the TFLite optimiser
recognizes `x * Sigmoid(x)` in either Mul input order and replaces both
operators with one generated `AFU_LUT`. The admitted contract is static batch-1
INT8 with equal shapes, scalar TFLite quantization, the canonical INT8 Sigmoid
representation (`scale=1/256`, `zero_point=-128`), a Mul rescale no greater than
one, a single-use Sigmoid result, and no public Sigmoid intermediate. LUT
generation emulates the quantized Sigmoid result followed by TFLite integer Mul
rounding, rather than evaluating an unconstrained floating-point SiLU formula.
All other Mul forms remain outside the Neural-AI supported-operator set.

Compiler tests verify both Mul input orders and every one of the 256 LUT bytes;
the complete Regor suite passes 640,735 assertions in 205 cases in the recorded
run. The compiler-generated H2/W3/C33 SiLU package is 1,152 bytes, requires 768
bytes of TCDM, and emits three executable commands: input `COPY_LAYOUT`, one
`AFU_LUT`, and output `COPY_LAYOUT`. It passes byte-exactly on Verilator at
91,394.006 total simulated ns, with 89,682 ns measured invocation time and
52,674 PMU cycles. A same-build, same-shape Sigmoid control also completes at
91,394.006 total simulated ns, demonstrating that fused SiLU has the cost of one
unary LUT and does not execute a second generic Mul path. Both are 2.96% above
the older 88,766 ns activation record because the current control moved by the
same amount. The focused SiLU cycles are 0.151x the 347,992-cycle
Micro-MobileNet record and 0.136x the 388,146-cycle Micro-YOLO record, but those
ratios compare one operator package with full graphs and are diagnostic only.

Nearest-neighbor 2x upsample is implemented through the 64-byte v2
`UPSAMPLE_NEAREST` command. The supported subset is static batch-1 INT8 with
exactly 32 channels, equal input/output quantization, and an exact 2x increase
in both spatial dimensions. The compiler legalizes TFLite
`ResizeNearestNeighbor` into the existing nearest-resampling AvgPool form,
keeps both internal tensors C32 blocked, and requires an out-of-place output.
The runtime validates the C32/2x contract, overflow-safe input/output ranges,
TCDM references, and non-overlap before calling the existing
`spatz_upsample_nearest2x_c32_i8` assembly kernel used by native Micro-YOLO.
C33 and non-2x shapes are intentionally rejected rather than routed through an
untested generic fallback.

The compiler-generated H2/W3/C32 package passes byte-exactly on Verilator at
86,036 simulated ns, with 83,998 ns measured invocation time and 50,260 PMU
cycles. The native Micro-YOLO upsample layer record is 35,462 PMU cycles. The
compiler package is 14,798 cycles higher (+41.7%), but includes package
parsing, public L2 I/O, and two boundary layout conversions, whereas the native
layer record measures the already-internal C32 path. Because both paths call
the same assembly kernel, this focused result confirms datapath reuse and
integration rather than establishing an operator-speed regression. A
compiler-generated full YOLO graph is still required for an equivalent
comparison with the 388,146-cycle native full-graph record.

YOLO-style MaxPool is implemented through the 96-byte v2 `MAXPOOL` command.
The supported subset is static batch-1 INT8, exactly 32 channels, K5/S1/P2,
same input/output shape and quantization, no fused activation, C32-blocked
internal tensors, and out-of-place output. The runtime validates all fixed
kernel fields, overflow-safe tensor ranges, TCDM references, reserved fields,
and non-overlap before calling
`systolic_maxpool5x5s1p2_c32_linebuf()`, the same linebuffer pool fast path used
by native Micro-YOLO. C33, K3, requantized, and fused-ReLU forms are rejected.

The compiler-generated H24/W24/C32 operator package passes byte-exactly on
Verilator at 289,729 simulated ns, with 251,203 ns measured invocation time and
216,420 PMU cycles. The equivalent native Micro-YOLO MaxPool layer record is
18,474 PMU cycles. The isolated compiler package contains two full 18,432-byte
public boundary conversions plus package parsing and L2 I/O; the native layer
starts and ends in internal C32 storage. In a YOLO graph, adjacent native C32
operators remove those public boundary conversions. Because both paths invoke
the same systolic helper, this result verifies compiler/runtime integration but
must not be interpreted as a 11.7x kernel slowdown. An equivalent
compiler-generated full YOLO graph remains the performance gate against the
388,146-cycle native record.

The model-required materialized Concat fallback is implemented for exactly two
static batch-1 INT8 inputs on the channel axis. Both input depths must be C32
aligned, spatial shapes and quantization must match exactly, and the output
depth must be their sum. The compiler preserves this subset as a native
operation, assigns C32-blocked storage, requires an out-of-place output, and
emits two existing local-to-local `DMA1D` commands. N-way, non-channel,
requantized, and channel-tail forms remain rejected. No Concat command or ABI
extension is added.

The compiler-generated H24/W24/C32+C32 package contains three public boundary
layout conversions and two 18,432-byte local copies. It passes byte-exactly on
Verilator at 192,093 simulated ns, with 115,789 ns measured invocation time and
81,006 PMU cycles. Firmware uses the existing Spatz vector copy and zero
primitives for local materialization and C32 boundary initialization; this
removes the scalar byte loops without increasing the 27,416-byte firmware text
baseline. The focused package moves 110,592 tensor bytes across its three
boundary conversions and two local copies. Native Micro-YOLO instead fuses the
logical H48/W48/C64 Concat into its dual-source head Conv, avoiding a 147,456-byte
materialized tensor and contributing to the 388,146-cycle full-graph record.
Therefore the measured materialized path is an appropriate correctness fallback
for standalone or non-fusible Concat, while concat-consumer Conv fusion remains
the required Phase 5 performance path for YOLO heads.

### Work Items

1. Implement zero-copy reshape, flatten, squeeze, and unsqueeze views only for
   storage-preserving instances present in the corpus.
2. Implement slice and split views only when required by a selected graph, with
   exact alignment and consumer contracts.
3. Implement only the DMA 2D/3D tensor-copy and regular layout movements emitted
   by those admitted model paths.
4. Optimize the four boundary `COPY_LAYOUT` modes and add native-to-native
   materialization required by view, concat, and vector operations.
5. Maintain the implemented two-input C32 materialized Concat fallback and add
   concat-consumer Conv fusion for YOLO head performance in Phase 5.
6. Maintain residual Add canonicalization, the implemented
   quantization-correct Add fallback, and the implemented YOLO SiLU AFU LUT
   fusion. When present in the selected MobileNet artifact, implement
   Hard-Swish as a generated AFU LUT fusion.
   Retain `MUL_Q7` only for proven Q7 patterns. Do not implement generic TFLite
   Mul or standalone Requant solely for operator coverage; add a constrained
   squeeze-excitation channel multiply only if the target corpus requires it.
7. Maintain AFU LUT lowering for standalone Sigmoid and clipping while keeping
   source-fused activation clamps in final requantization.
8. Maintain the implemented MaxPool K5 S1 P2, nearest-neighbor 2x C32, and
   global average pool paths while expanding only model-required shapes.
9. Add out-of-place and liveness constraints for AFU operations.

### Tests

- Every admitted corpus view validates metadata, offset, and no-command
  behavior; nearby non-storage-preserving forms are rejected.
- Every required materialization validates byte order across its model H/W/C
  shapes and tails.
- Every public graph output is contiguous frontend order, including outputs from
  native C32 and ROW32 producers.
- Add scale, zero-point, clamp, raw-AFU, byte-exact Conv canonicalization, and
  quantization-correct fallback tests; add randomized coverage for each residual
  pattern found in the target corpus.
- Exhaustive 256-input LUT tests for Sigmoid, clipping, SiLU, and Hard-Swish
  modes that are required by selected models, using their admitted
  quantization contracts and rounding boundaries. Unrelated scale/zero-point
  combinations are negative tests, not new supported modes.
- If `MUL_Q7` or squeeze-excitation channel multiply is selected, compare every
  supported quantization/shape case with an independent integer reference and
  reject nearby unsupported broadcasts and representations.
- Two-input aligned materialized Concat plus rejection tests for channel tails,
  non-channel axes, and mismatched quantization; fused-consumer coverage belongs
  to Phase 5.
- Multi-operation chains that expose lifetime and aliasing bugs.

### Exit Criteria

- Every Phase 4 operator instance in the selected full-size YOLO and MobileNet
  artifacts has compiler lowering plus a corpus-derived topology micro-graph
  test. Nearby unvalidated variants stop with an actionable diagnostic.
- No operator is advertised from primitive availability alone, and there is no
  silent fallback for unvalidated TFLite variants.
- No new generic tensor loop is added to firmware when a Spatz or iDMA path
  already exists.
- Full-graph operator inventories contain no unexpected host/CPU fallback,
  every inventory instance maps to a verified micro-graph contract, and
  Micro-MobileNet/Micro-YOLO regressions remain byte-exact. Full-size graph SRAM
  fit and simulation are not Phase 4 exit criteria.

## Phase 5 - DMA Overlap, Performance Model, and YOLO Patterns

### Objective

After operator mapping closes, move from micro-graphs to an SRAM-feasible and
then optimized schedule for the selected full-size YOLO and MobileNet flows.
Reuse Regor/Vela scheduling, cascading, tiling, live-range, and allocation
infrastructure before adding Neural-AI-specific policy. Micro-YOLO remains the
fast native performance baseline, not the feature target.

### Work Items

1. Add asynchronous DMA submit and wait commands with event IDs.
2. Add ping-pong IFM, weight, and OFM allocation.
3. Add dependency-aware rolling-buffer scheduling.
4. Calibrate the architecture performance model using PMU data.
5. Enumerate tile candidates in the scheduler and select by measured cost.
6. Fuse logical concat into dual-source Conv when the contract matches.
7. Add DFL softmax4 and class-sigmoid pattern matching only when those exact
   head patterns are present in the selected deployable YOLO artifact.
8. Add a debug map from layer and tile IDs to command byte ranges.

### Tests

- Asynchronous DMA dependency, stall, and error tests.
- Ping-pong buffer reuse under artificial TCDM stalls.
- Blocking and overlapping schedules produce identical output.
- PMU estimates remain within a defined error threshold.
- Selected full-size YOLO compiler graph, with staged or subgraph simulations
  during development and a final full-graph run under the agreed long-simulation
  workflow.
- Micro-YOLO end-to-end regression and PMU baseline comparison.
- Command-count and L2-traffic regression thresholds.

### Exit Criteria

- DMA and compute actually overlap; rolling metadata alone is not sufficient.
- The selected full-size YOLO artifact compiles without hand-written descriptors
  and every operator instance is accounted for by a validated lowering.
- The generated graph is compared with the 388,146-cycle Micro-YOLO record using
  normalized layer/path measurements where shapes differ; no unsupported claim
  is inferred from raw total-cycle ratios between different model sizes.
- Performance reports separate compute, DMA, AFU or Spatz, and stall time.

## Phase 6 - Hardening and Release Gates

### Objective

Turn the model-driven YOLO/MobileNet backend into a maintainable target without
expanding it into a generic TFLite or TOSA backend.

### Work Items

1. Validate TOSA or GraphAPI input only for patterns already required and proven
   by the selected model corpus. Frontend parity is not a release blocker when
   no selected artifact uses that frontend.
2. Audit all remaining `NHCWB16`, 16-byte alignment, and Ethos-target assertions;
   prove that `NHCWB16` is unreachable from the Neural-AI target.
3. Fuzz and negative-test the package parser.
4. Add ABI forward- and backward-compatibility tests.
5. Add reproducible-build and deterministic-artifact checks.
6. Add compiler diagnostics with operation name, source ID, shape, and violated
   constraint.
7. Add version negotiation and target-mismatch errors.
8. Document CLI usage, package inspection, and runtime integration.
9. Run all neural-compiler unit tests and Neural-AI RTL regressions.

### Exit Criteria

- Any supported TOSA/GraphAPI pattern uses the same target contracts as its
  already-supported TFLite counterpart; no independent generic operator subset
  is created.
- There is no known silent fallback.
- Runtime safely rejects malformed packages.
- Ethos output and regression results remain unchanged.
- `git diff -- neural-ai/hw` is empty.

## 9. Test Strategy Without Modifying `neural-ai/hw`

### 9.1. Compiler Unit Tests

Place tests in `neural-compiler/ethosu/regor/test/`:

```text
test_arch_neural_ai.cpp
test_neural_ai_constraints.cpp
test_neural_ai_layout.cpp
test_neural_ai_boundary_layout.cpp
test_neural_ai_weight_encoder.cpp
test_neural_ai_quantization.cpp
test_neural_ai_linebuffer_planner.cpp
test_neural_ai_command_stream.cpp
test_neural_ai_writer.cpp
```

Build and run:

```bash
cd /home/dev01/neural-compiler
cmake -S ethosu/regor -B build-unit-tests -DCMAKE_BUILD_TYPE=Debug
cmake --build build-unit-tests -t check
```

### 9.2. Software and Runtime Tests

Place all new test source under:

```text
/home/dev01/neural-ai/sw/test/compiler_runtime/
```

Compiler-generated model tests belong under `sw/test/compiler_runtime`.
Existing Cocotb tests under `hw/rtl/cluster/tb/tests` may be extended when the
test must directly verify an RTL access path or hardware-only contract. The
software test directory should contain:

- Runtime firmware build files.
- An invocation or package builder for negative tests where needed.
- Compiler invocation helpers.
- A Cocotb module named `test_compiled_model.py`.
- Golden and reference model utilities.

### 9.3. Use the Existing Simulator

The cluster Makefile prepends its own test directories while preserving incoming
`PYTHONPATH`. An external test module can therefore be run with:

```bash
cd /home/dev01/neural-ai
PYTHONPATH="$PWD/sw/test/compiler_runtime:$PWD/hw/rtl/cluster/tb/tests" \
make -C hw/rtl/cluster sim \
  COCOTB_TEST_MODULES=test_compiled_model \
  CLUSTER_SIM_NAME=compiled_model \
  SIM_BUILD=/tmp/neural-ai-compiled-model
```

Using `SIM_BUILD=/tmp/...` avoids adding build output to the source tree. RTL
`.sv` sources remain unchanged.

### 9.4. Regression Order

1. Neural-compiler unit tests.
2. Software ABI and parser host tests.
3. Runtime firmware build and size gate.
4. Existing v1 MatMul simulation.
5. Compiler-generated GEMM end-to-end test.
6. Per-operation Conv and vector end-to-end tests.
7. Micro-MobileNet.
8. Micro-YOLO.
9. Compiler and operator-inventory checks for selected full-size MobileNet and
   YOLO artifacts.
10. Staged full-size model subgraphs, then explicit full-graph simulation only
    at the release gate and under the agreed long-simulation workflow.
11. Full existing Neural-AI cluster regression.
12. Git check confirming no RTL `.sv` diff under `hw`.

## 10. Diagnostics and Unsupported Behavior

The compiler must fail early. It must not silently send an operation to a CPU in
a raw `.nai` execution flow.

Proposed diagnostic format:

```text
NeuralAI: unsupported Conv2D at op "block3/conv" (source id 42):
kernel=7x7 exceeds linebuffer limit 5x5; shape=[1,224,224,32],
stride=2x2, dilation=1x1, padding=SAME.
```

Each error should include:

- Target name and version.
- Operation name and source ID.
- Input and output shapes and data types.
- Quantization details when relevant.
- The violated parameter or constraint.
- A supported alternative when it is unambiguous, such as K3/S1/P1.

Do not use `assert` for an invalid user model. Reserve assertions for compiler
internal invariants.

## 11. Pull Request and Commit Strategy

Recommended independent changes:

1. ABI contract, memory constant correction, and golden fixtures.
2. Software runtime v2 DMA/GEMM with unchanged legacy v1 behavior.
3. Architecture-owned alignment and layout refactor with unchanged Ethos output.
4. Neural-AI factory, config, CLI, `.nai` writer, NHWC boundary conversion, and
   MatMul.
5. Quantization and weight encoders.
6. Linebuffer planner and RGB/C32 Conv.
7. Pointwise and depthwise plus Micro-MobileNet.
8. Views, layouts, vector, and AFU operations.
9. Asynchronous DMA and performance model.
10. Selected full-size YOLO/MobileNet patterns and micro-model regressions.
11. Hardening, optional frontend parity for already-supported patterns, and
    documentation.

Each change must:

- Include focused tests.
- Avoid unrelated refactoring.
- Leave Neural-AI RTL `.sv` sources unchanged.
- State the ABI version when changing a wire format.
- Run Ethos regression tests when touching shared code.

## 12. Risks and Mitigations

### 12.1. Shared Scheduler Is Too Ethos-Specific

Risk: Neural-AI compilation succeeds but storage offsets are wrong because of a
hidden NHCWB16 assumption.

Mitigation: move alignment and layout behavior behind architecture APIs in an
independent change, with Ethos golden and regression tests before and after. Add
an invariant that Neural-AI public bindings are NHWC and its internal tensors are
never NHCWB16.

### 12.2. ABI Drift Between Repositories

Risk: the C++ serializer and C runtime disagree on structure size or field order.

Mitigation: use fixed-width manual serialization, static assertions, an ABI
manifest, and cross-repository golden byte tests. Increment the major version for
every incompatible change.

### 12.3. Quantization Produces Plausible but Incorrect Output

Risk: negative rounding, bias scale, activation clamp, or zero-point handling is
wrong while output still appears reasonable.

Mitigation: add exhaustive boundary vectors, randomized accumulator tests, an
RTL-compatible reference, and per-channel end-to-end tests. Do not reuse Ethos
scaling encoding.

### 12.4. Asymmetric Input Zero Point with Padded Conv

Risk: folding the zero point into bias produces position-dependent border errors.

Mitigation: require symmetric internal Conv input, insert an explicit verified
requantization where possible, and otherwise fail compilation.

### 12.5. Firmware Exceeds ITCM

Risk: a universal runtime plus all handlers exceeds 32 KB.

Mitigation: keep planning in the compiler, enable section garbage collection,
enforce a size gate, share handlers, and introduce feature profiles only if
necessary. Do not duplicate the large graph switch in a new dispatcher.

### 12.6. Incorrect TCDM Alias or Port Modeling

Risk: the allocator treats logical address windows as independent physical
memories and over-allocates TCDM.

Mitigation: model one physical arena with target access constraints and add peak
memory and aliasing regression tests.

### 12.7. Claiming Double Buffering While Runtime Is Blocking

Risk: performance estimates and documentation overstate actual overlap.

Mitigation: calculate summed blocking cost in the correctness phase. Enable the
overlap model only after asynchronous submit and wait commands have PMU coverage.

### 12.8. Generic Fallback Increases Latency or Firmware Size

Risk: every unsupported operation becomes a scalar loop on Snitch.

Mitigation: derive the supported subset from the selected full-size model
inventory, prefer fusion and existing AFU, Spatz, and iDMA primitives, and add a
fallback only for a demonstrated recurring model requirement with byte-exact
and performance evidence. Generic TFLite coverage must not drive RTL, ABI, or
firmware growth.

## 13. Definition of Done

The Neural-AI backend is complete for the current hardware contract when:

- `vela --accelerator-config neural-ai --output-format nai model.tflite`
  produces a `.nai` file for each named full-size YOLO and MobileNet artifact in
  the release corpus.
- Builds are reproducible and the file contains no absolute runtime address.
- Generic firmware loads the invocation and model without model-specific C code.
- Every operator instance in those artifacts maps to a constrained, documented,
  tested lowering or fusion. Primitive availability and generic TFLite operator
  names are not completion criteria.
- Required Conv, depthwise, residual Add, activation, pooling, upsample, concat,
  and selected detection-head patterns have focused byte-exact tests and
  model-level coverage.
- SiLU and Hard-Swish use LUT fusion when required; generic TFLite Mul and full
  generic TFLite operator coverage remain explicitly out of scope.
- Per-channel bias, multiplier, shift, and zero point follow RTL semantics.
- Frontend and Graph IR keep logical NHWC shapes, and all rank-4 public bindings
  use contiguous NHWC storage.
- Internal NPU activations use Neural-AI native ROW32 or C32_BLOCKED unless a
  supported operation explicitly requires another physical format.
- No default host API exposes native padding or blocked channel order.
- Views avoid unnecessary materialization; every layout conversion is explicit
  and tested.
- Unsupported graphs fail compilation with actionable diagnostics.
- The runtime bounds-checks every section, reference, and command.
- Firmware fits 32 KB ITCM and runtime data fits 32 KB DTCM.
- Peak TCDM, command bytes, and constant bytes are reported.
- Full-size model performance is reported by graph and attributed by operator
  path. Micro-MobileNet (347,992 cycles) and Micro-YOLO raw-head (388,146
  cycles) remain regression references; comparisons across different model
  sizes use normalized layer/path data rather than raw total-cycle equivalence.
- Existing ABI v1 and hand-written model tests still pass.
- Existing Ethos-U compiler tests and output still pass.
- There is no RTL `.sv` source change under `/home/dev01/neural-ai/hw`.

## 14. Recommended Starting Point

The foundational MatMul/Conv/runtime slices already exist. The next vertical
slice should start from the selected full-size model artifacts rather than from
another generic operator:

```text
selected full-size YOLO and MobileNet artifacts
  -> reproducible operator-instance and quantization inventory
  -> classify each instance against existing constrained lowerings
  -> prioritize the first unsupported recurring hot-path pattern
  -> prefer fusion or an existing RTL primitive
  -> add focused compiler, host-runtime, and single-operator RTL tests
  -> run staged model subgraphs
  -> compare attributed PMU data with Micro-YOLO/Micro-MobileNet records
  -> run the final full graph only at the release gate
```

Byte-exact SiLU LUT fusion is now implemented without a Mul datapath extension.
The next gap must come from the reproducible operator inventory of the selected
full-size artifacts. Revise that inventory and its constrained contract before
implementation; do not add generic TFLite semantics or RTL merely to broaden an
operator-support table.

Estimated effort for one engineer, assuming the hardware contract remains fixed:

| Phase | Estimate |
|---|---:|
| Phase 0 | 3-5 days |
| Phase 1 | 1-2 weeks |
| Phase 2 | 1-2 weeks |
| Phase 3 | 2-3 weeks |
| Phase 4 | 1-2 weeks |
| Phase 5 | 1-2 weeks |
| Phase 6 | About 1 week |
| Total | About 8-12 weeks |

## 15. Source References

Compiler:

- `../README.md`
- `../ethosu/vela/vela.py`
- `../ethosu/vela/architecture_features.py`
- `../ethosu/vela/shape4d.py`
- `../ethosu/vela/api.py`
- `../ethosu/regor/regor.cpp`
- `../ethosu/regor/include/regor.h`
- `../ethosu/regor/architecture/architecture.hpp`
- `../ethosu/regor/architecture/architecture_constraints.hpp`
- `../ethosu/regor/architecture/register_command_stream_generator.hpp`
- `../ethosu/regor/architecture/weight_encoder.hpp`
- `../ethosu/regor/compiler/compiler.cpp`
- `../ethosu/regor/compiler/scheduler.cpp`
- `../ethosu/regor/compiler/cascade_builder.cpp`
- `../ethosu/regor/compiler/high_level_command_stream.hpp`
- `../ethosu/regor/compiler/high_level_command_stream_generator.cpp`
- `../ethosu/regor/compiler/graph_packing.cpp`
- `../ethosu/regor/compiler/raw_writer.cpp`
- `../ethosu/regor/tflite/custom_operator_ethosu.hpp`
- `../ethosu/regor/tflite/tflite_writer.cpp`
- `../ethosu/regor/tflite/tflite_supported_operators.cpp`

Neural-AI:

- `../../neural-ai/README.md`
- `../../neural-ai/docs/architecture.md`
- `../../neural-ai/docs/operator_support_matrix.md`
- `../../neural-ai/docs/systolic_array_spec.md`
- `../../neural-ai/hw/rtl/cluster/npu_cluster_pkg.sv`
- `../../neural-ai/hw/rtl/cluster/tb/npu_linebuf_precompute.py`
- `../../neural-ai/hw/rtl/cluster/tb/tests/npu_test_utils.py`
- `../../neural-ai/hw/rtl/cluster/tb/tests/test_matmul.py`
- `../../neural-ai/hw/rtl/cluster/tb/tests/test_micro_mobilenet_e2e.py`
- `../../neural-ai/sw/lib/npu_memory_map.h`
- `../../neural-ai/sw/lib/npu_tensor.h`
- `../../neural-ai/sw/lib/npu_graph.h`
- `../../neural-ai/sw/lib/npu_graph.c`
- `../../neural-ai/sw/lib/npu_cmd_desc.h`
- `../../neural-ai/sw/lib/npu_cmd_desc.c`
- `../../neural-ai/sw/lib/hal_systolic.h`
- `../../neural-ai/sw/lib/hal_systolic.c`
- `../../neural-ai/sw/lib/conv2d_packed.h`
- `../../neural-ai/sw/lib/spatz_ops.h`
- `../../neural-ai/sw/lib/spatz_ops.c`
