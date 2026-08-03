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

This is a design and implementation guide. It does not claim that the current
hardware supports every TFLite or TOSA operator.

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
| AFU commands | Hardware modes exist | v2 constrained `ADD_I8` dispatch | Equal-shape, symmetric, raw-safe INT8 Add | Compiler-generated Add package on Verilator |
| Spatz commands | Engine exists | Incomplete kernels | No | No |

`ArchNeuralAI` exposes `FullyConnected`, `MatMul`, `MemoryCopy`, and a
constrained CNN path containing pointwise Conv, RGB K3 S2, generic full-group
C32 K3, and depthwise K3 S1/S2. It also exposes raw-safe `AddI8` only when
Regor's normalized quantization proves equal source/output scales, all zero
points are zero, the clamp spans the full INT8 range, shapes are equal, and the
AFU output allocation does not overlap either input.

The plan should record the hardware commit and compiler commit at which the
contract was audited.

## 3. Definition of Full Support

In this document, full support for the Neural-AI NPU means:

- The input is a static-shape, batch-1 TFLite or TOSA model.
- Main activations and weights are signed INT8; bias and partial sums are INT32.
- Native GEMM, FullyConnected, Conv, and pointwise paths require symmetric
  quantization: IFM zero point == 0 and weight zero point == 0. OFM zero point
  is arbitrary INT8, applied by requantization. Nonzero IFM zero point is
  unsupported unless an explicit, tested affine-to-symmetric conversion is
  inserted before the native operation. The operator checker must reject
  violations before scheduling.
- The graph only uses operators and parameter combinations covered by the
  Neural-AI contract.
- The compiler produces one `.nai` file without requiring a model-specific
  `main.c`, C graph, hand-packed weights, or hand-written descriptors.
- One generic firmware image can load and execute that file.
- Rank-4 model inputs and outputs use contiguous NHWC storage by default; native
  padded layouts are private implementation details of the compiled graph.
- Output is bit-exact, or within an explicitly defined tolerance, relative to a
  reference implementation.
- Unsupported operators, shapes, layouts, or quantization parameters fail at
  compile time with actionable diagnostics.

Full support does not mean:

- Every TFLite or TOSA operator.
- Floating-point inference.
- Dynamic shapes or dynamic allocation in firmware.
- Arbitrary kernel, stride, dilation, or padding values beyond RTL limits.
- Generic Softmax, GELU, NMS, or control flow without a verified software kernel.
- Five-cluster scheduling, because the five-cluster top level is still planned.
- Modifying RTL to make a model compilable.

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

The current contract in `neural-ai/docs/operator_support_matrix.md` is:

| Family | Native or available path | Main limits |
|---|---|---|
| DMA | iDMA 1D/2D/3D | Existing graph path is primarily blocking |
| MatMul/GEMM | Systolic GEMM32 | K and N in groups of 32; M is tiled |
| RGB Conv | Linebuffer plus systolic | C3, OC32, K3, S2, P1 |
| Pointwise Conv | Direct GEMM32 | RTL supports grouped C32 operation; compiler currently lowers only 1x1/S1/P0 and pads IC/OC tails |
| C32 Conv | Multi-C32 linebuffer | K3, S1/S2, P1; IC/OC multiples of 32 |
| Depthwise Conv | Depthwise linebuffer | K3, S1/S2, P1; tail lanes masked |
| Requant | Per-channel systolic or Spatz | Shift range 0..31 |
| Logistic/Clamp | AFU LUT | 256-entry LUT, out-of-place |
| Add/Mul | AFU fast mode or Spatz | Fast mode covers selected quantization |
| MaxPool | Systolic or Spatz | Fast path is C32 K5 S1 P2 |
| Upsample | Spatz | Graph contract is nearest-neighbor 2x |
| Global AvgPool | AFU | C32 input, 1x1 output |
| Views | Compiler metadata | Zero-copy only when storage order is preserved |
| Concat | View, fused consumer, or Spatz | Generic N-way concat is not stable yet |
| DFL/class sigmoid | Model-specific AFU modes | Not equivalent to generic Softmax |

The backend must first support exactly these parameter combinations. A generic
fallback on Snitch or Spatz may only be advertised after the kernel and tests
exist.

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
    ClampI8,
    AddI8,
    MulI8,
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

AFU mode names must map directly to hardware modes (`E8`, `E16`, `E32`,
`MUL_Q7`, `ADD_I8`, `DFL4_ROW32_Q8`, `CLASS_SIGMOID_ROW32_HIGH16`,
`GLOBAL_AVGPOOL_C32`), not generic TFLite operator names.

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
- IC and OC multiples of 32 for the multi-C32 path.

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
- Do not fold a nonzero input zero point into bias for padded linebuffer Conv;
  border padding would become position-dependent and incorrect.
- Insert an explicit INT8-to-INT8 affine requant into a symmetric internal
  representation only after that SW or AFU path has verified tests.
- Fail compilation when no loss-safe and overflow-safe internal representation
  is available.

### 7.12. Elementwise Quantization

The current raw INT8 Add and Mul wrappers are not sufficient for generic TFLite
quantization. Runtime v2 commands must carry complete quantization parameters.

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

Use the AFU fast mode only when the configuration matches exactly. Use a Spatz
vector kernel or a scalar correctness fallback on Snitch for remaining cases,
then optimize. The performance report must identify the selected path.

Sigmoid and Clamp:

- Generate a 256-byte LUT from input and output quantization.
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

Calibrate with PMU regression for:

- GEMM M sweeps.
- Pointwise IC and OC group sweeps.
- Conv tile H and W sweeps.
- S1 and S2 linebuffer modes.
- Depthwise channel and tail sweeps.
- DMA 1D, 2D, and 3D size and stride sweeps.
- AFU and Spatz element-count sweeps.
- Concurrent DMA + compute bank-conflict measurements.

The performance model must not claim overlap while the command runtime remains
blocking.

## 8. Implementation Phases

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

- Compiler C++ tests compile pointwise M=257 and M=511 into multiple
  `POINTWISE_C32` commands and assert every command has `0 < M <= 256`.
- GEMM weight-packing tests cover K and N boundaries 1, 31, 32, 33, 63, 64,
  and 65, checking every valid byte and every zero-padded lane.
- Compiler-generated pointwise packages execute byte-exactly on Verilator:
  M=257 passes at 286,687 simulated ns and M=511 passes at 369,129 simulated
  ns.
- Two independent compiler instances produce byte-identical `.nai` blobs for
  the same FullyConnected K33/N34 model and configuration.
- A Verilator ROW32 boundary package round-trips contiguous public data through
  native TCDM storage for `H=2`, `W=3`, and
  `C=3,31,32,33,63,64,65`, including the wide compact-stride cases that require
  multiple iDMA row segments.
- This evidence does not relax the Phase 1 command limit: a raw command with
  either oversized M value remains invalid.

## Phase 3 - Conv, Linebuffer, and Per-Channel Quantization

### Objective

Support CNN backbones containing an RGB stem, pointwise Conv, C32 Conv, and
depthwise Conv.

Current progress: constrained pointwise Conv1x1, RGB K3 S2, generic full-group
C32 K3, and depthwise K3 S1/S2 lowering are implemented. The generated command
streams include NHWC↔C32 boundaries, multi-group accumulation, per-channel
requantization, and fused ReLU/ReLU6 clamps. Generic K3 requires both IC and OC
to be divisible by 32; pointwise and depthwise retain C32 tail support. Focused
compiler-generated Verilator E2E tests pass for RGB, generic C32, depthwise C33
S2, and a pointwise-plus-depthwise chain. A 13-stage, native-like
Micro-MobileNet topology now compiles through RGB K3 S2, depthwise and
pointwise Conv, two residual Adds, generic K3, GlobalAvgPool, and the
classifier. This required TFLite SAME padding with asymmetric 0/1 sides for
even-sized K3 S2 inputs. ReLU6 is fused into the producing Conv's existing
per-channel requantization, so this graph does not emit a standalone AFU LUT
command. The full compiler-generated graph now passes byte-exactly on the
current Verilator model. Its PMU result is 522,900 cycles, so correctness is
closed for this topology while the performance gate remains open against the
347,992-cycle native Micro-MobileNet record.

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
3. Implement RGB, pointwise, generic Conv2D, and depthwise weight encoders.
4. Implement per-channel multiplier and shift search plus the RTL rounding model.
5. Add `RQ_LOAD`, linebuffer, pointwise, and depthwise commands.
6. Extract runtime operation handlers from `npu_graph_run()` where appropriate.
7. Add C32 layout propagation and TCDM partial-sum allocation while leaving
   logical Graph IR shapes in NHWC.
8. Extend the v2 `COPY_LAYOUT` handler and command generation with NHWC-to-C32
   and C32-to-NHWC boundary materialization, with direct RGB C3 input
   consumption where supported.
9. Implement IC/OC group loops and channel tails.
10. Fuse ReLU, ReLU6, and Clamp into final requantization.

### Compiler Tests

- A constraint matrix for every supported and unsupported kernel combination.
- Weight byte-golden tests for all four formats.
- Randomized quantization tests against an RTL-compatible reference.
- Border and interior linebuffer descriptor byte-golden tests.
- 160-byte `LINEBUF_JOB` command serialization golden tests.
- Tile-fit and overflow tests at the TCDM boundary.
- C33, C48, C64, C65, and C96 channel cases.
- Full C32 group and tail C32 cases with separate performance validation.
- M = 256 and M > 256 external-psum path tests.
- NHWC L2 → C32 TCDM → NHWC L2 boundary layout round-trip.
- NHWC input and output byte-order tests across H/W/C and channel-tail cases.

### End-to-End Order

1. RGB stem C3 -> C32.
2. Pointwise C32 -> C32.
3. Conv3x3 S1.
4. Conv3x3 S2.
5. Depthwise S1/S2 with channel tails.
6. Pointwise plus depthwise chain.
7. C32 -> NHWC graph output.
8. Full Micro-MobileNet graph with NHWC host buffers.

### Current Verification Evidence

- Compiler C++ tests: 188/188 pass.
- The complete 13-stage Micro-MobileNet compiler test passes every graph prefix
  and the full graph byte-exactly. The full package contains two `AFU_BINARY`
  residual Adds, one `AFU_GLOBAL_AVGPOOL`, and the expected Conv/linebuffer
  commands, with no `AFU_LUT` activation command.
- Neural-AI compiler-runtime host ABI/layout/quantization checks pass.
- Compiler-generated Verilator packages pass byte-exactly:
  - RGB K3 S2 C3 -> C32: 195,997 simulated ns after wide-row chunking, versus
    195,851 ns immediately before it (+0.07%) and 194,651 ns before DMA
    direction validation (+0.69% cumulative).
  - Generic K3 S1 C32 -> C32: 494,008 simulated ns after wide-row chunking,
    versus 493,826 ns immediately before it (+0.04%) and 492,662 ns before DMA
    direction validation (+0.27% cumulative).
  - Depthwise K3 S2 C33 tail: 228,121 simulated ns.
- The current full package is 89,312 bytes, contains 81 runtime commands, and
  has a 290,816-byte peak TCDM allocation. Direct external-to-local compact
  tensor transfers avoid CPU-backed local-to-local bounce copies, and one
  pointwise `RQ_LOAD` is reused by every M stripe in the same output group.
- The equivalent full-graph PMU run completes byte-exactly at 522,900 cycles.
  The principal counters are 261,093 Snitch instructions, 35,714 systolic
  compute cycles, 6,959 iDMA busy cycles, 350,281 AFU done events, and 5,482
  TCDM stall cycles.
- Runtime firmware `.text` is 23,988 bytes with the ABI 1.1 section layout,
  below the 32 KB limit.
- The focused package times include boot, command loading,
  boundary layout DMA, and output checking. They must not be compared as
  operator latency against the PMU-only Micro-MobileNet and Micro-YOLO records.
  The current native baselines remain 347,992 total cycles for Micro-MobileNet
  and 388,146 total cycles for the Micro-YOLO raw-head graph. The measured
  compiler graph is therefore 1.503x the Micro-MobileNet record and 1.347x the
  Micro-YOLO record. These are equivalent full-graph PMU comparisons, but the
  50.3% gap to the matching MobileNet record does not close the performance
  gate. Further work must attribute the remaining command/runtime overhead
  without weakening byte-exact correctness or changing frozen RTL.

### Exit Criteria

- Micro-MobileNet no longer requires a hand-written graph, weights, or
  descriptors.
- Per-channel bias and scale match the reference.
- Peak TCDM and firmware size are reported and enforced.
- Existing hand-written model tests still pass.

## Phase 4 - Vector, AFU, View, and Layout Operations

### Objective

Support the non-Conv operations in the current Neural-AI operator matrix.

Current progress: the 64-byte v2 `AFU_BINARY` ABI and runtime `ADD_I8`
dispatcher are implemented. The compiler lowers only equal-shape, batch-1,
raw-safe symmetric INT8 Add to this mode, keeps tensors in C32 blocked storage,
and disables allocator IFM reuse so the AFU output remains out-of-place.
Requantized Add, nonzero zero points, broadcasting, and fused activation clamps
remain unsupported and are rejected. Internal `Reshape`, `Squeeze`, and
`ExpandDims` are admitted to Regor's existing reshape-removal pass; a
view-to-Add regression verifies that none of the three adds a runtime command.
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

The compiler-generated Add package passes byte-exactly on Verilator at 127,826
simulated ns. The equivalent hand-written package passes at 113,958 simulated
ns; the 13,868 ns (12.17%) difference is consistent with additional
parsing/validation overhead for a 1,184-byte compiler artifact versus an
800-byte fixture, while both execute four commands over the same 128-byte
tensors. These focused completion times include boot and runtime overhead and
are not substitutes for the 347,992-cycle Micro-MobileNet or 388,146-cycle
Micro-YOLO full-graph PMU gates.

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
522,900 cycles, or 1.503x the 347,992-cycle Micro-MobileNet record and 1.347x
the 388,146-cycle Micro-YOLO record. Focused AFU results remain useful for
attribution but do not replace that end-to-end gate.

The Neural-AI AFU is covered by standalone `afu_ops` tests and by the native
Micro-MobileNet/Micro-YOLO graphs; lack of tests is not the current limitation.
On the current RTL, `test_afu_op_clamp_relu6` times out after consuming all
21,216 input bytes and remains in `ST_READ_IN`. The generic E8 stream path tests
the 32-byte destination boundary before the final-element condition, selects a
mid-stream flush on an aligned final byte, and then waits for a nonexistent
next RFIFO beat. This ordering entered with AFU refactor `41cce28`; the recorded
347,992-cycle Micro-MobileNet result predates that refactor and therefore does
not establish that the current RTL still passes its three standalone clamp
layers. `ADD_I8` and `GLOBAL_AVGPOOL_C32` pass their current focused tests, so
this finding is specific to the generic E8 clamp/LUT stream rather than the
whole AFU. RTL is frozen for this compiler plan; Conv clamps must remain fused
into requantization, and standalone AFU LUT lowering cannot close until the
native clamp regression passes again.

### Work Items

1. Implement zero-copy reshape, flatten, squeeze, and unsqueeze views.
2. Implement slice and split views with alignment and consumer checks.
3. Implement DMA 2D/3D tensor copies and regular layout movement.
4. Optimize the four boundary `COPY_LAYOUT` modes and add native-to-native
   materialization required by view, concat, and vector operations.
5. Implement concat fusion and a Spatz concat fallback where required.
6. Implement quantization-correct Add, Mul, and Requant software kernels.
7. Generate AFU LUTs for Sigmoid and Clamp.
8. Implement MaxPool K5 S1 P2, nearest-neighbor 2x, and global average pool.
9. Add out-of-place and liveness constraints for AFU operations.

### Tests

- Every view validates metadata, offset, and no-command behavior.
- Every materialization validates byte order across H/W/C tails.
- Every public graph output is contiguous frontend order, including outputs from
  native C32 and ROW32 producers.
- Add and Mul randomized scale, zero-point, and reference tests.
- Exhaustive 256-value LUT tests.
- Aligned, unaligned, fused, and materialized concat cases.
- Multi-operation chains that expose lifetime and aliasing bugs.

### Exit Criteria

- Every row marked Supported in the operator matrix has compiler lowering and an
  end-to-end test.
- Every Partial row has a documented path or diagnostic; there is no silent
  incorrect result.
- No new generic tensor loop is added to firmware when a Spatz or iDMA path
  already exists.

## Phase 5 - DMA Overlap, Performance Model, and YOLO Patterns

### Objective

Move from a correctness schedule to an optimized schedule and cover the existing
YOLO model flow.

### Work Items

1. Add asynchronous DMA submit and wait commands with event IDs.
2. Add ping-pong IFM, weight, and OFM allocation.
3. Add dependency-aware rolling-buffer scheduling.
4. Calibrate the architecture performance model using PMU data.
5. Enumerate tile candidates in the scheduler and select by measured cost.
6. Fuse logical concat into dual-source Conv when the contract matches.
7. Add DFL softmax4 and class-sigmoid pattern matching.
8. Add a debug map from layer and tile IDs to command byte ranges.

### Tests

- Asynchronous DMA dependency, stall, and error tests.
- Ping-pong buffer reuse under artificial TCDM stalls.
- Blocking and overlapping schedules produce identical output.
- PMU estimates remain within a defined error threshold.
- Full Micro-YOLO end-to-end graph.
- Command-count and L2-traffic regression thresholds.

### Exit Criteria

- DMA and compute actually overlap; rolling metadata alone is not sufficient.
- Micro-YOLO compiles and runs without hand-written descriptors.
- Performance reports separate compute, DMA, AFU or Spatz, and stall time.

## Phase 6 - Hardening, TOSA, and Release Gates

### Objective

Turn the backend into a maintainable target rather than a model-specific
prototype.

### Work Items

1. Run the same backend from TOSA and GraphAPI canonical IR.
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

- The supported TFLite and TOSA subsets use the same target backend.
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
9. Full existing Neural-AI cluster regression.
10. Git check confirming no RTL `.sv` diff under `hw`.

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
10. YOLO patterns and Micro-YOLO.
11. TOSA, hardening, and documentation.

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

Mitigation: keep the supported subset explicit, prefer AFU, Spatz, and iDMA, and
add fallbacks only for demonstrated model requirements with performance labels.

## 13. Definition of Done

The Neural-AI backend is complete for the current hardware contract when:

- `vela --accelerator-config neural-ai --output-format nai model.tflite` produces
  a `.nai` file.
- Builds are reproducible and the file contains no absolute runtime address.
- Generic firmware loads the invocation and model without model-specific C code.
- MatMul, RGB Conv, pointwise, C32 Conv, depthwise, and the current vector and AFU
  operator matrix have compiler lowering and end-to-end tests.
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
- Existing ABI v1 and hand-written model tests still pass.
- Existing Ethos-U compiler tests and output still pass.
- There is no RTL `.sv` source change under `/home/dev01/neural-ai/hw`.

## 14. Recommended Starting Point

Do not start with full Conv implementation. The first vertical slice should be:

```text
TFLite MatMul or FullyConnected
  -> Regor Graph IR
  -> preserved frontend-order model input
  -> NHWC/contiguous-to-ROW32 boundary pack
  -> Neural-AI scheduler and alignment
  -> packed 32x32 weights
  -> relocatable v2 commands
  -> ROW32-to-frontend-order boundary unpack
  -> .nai writer
  -> generic software runtime
  -> existing cluster simulator
  -> reference output comparison
```

This slice exercises every important boundary: target factory, CLI, allocator,
encoder, command ABI, writer, loader, runtime, and simulator. Port linebuffer and
Conv only after the slice is stable. This avoids debugging the compiler,
quantization, descriptors, and runtime simultaneously in one large graph.

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
