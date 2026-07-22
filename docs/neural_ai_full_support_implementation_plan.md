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
- Source files under `neural-ai/hw` must not be changed.
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

## 3. Definition of Full Support

In this document, full support for the Neural-AI NPU means:

- The input is a static-shape, batch-1 TFLite or TOSA model.
- Main activations and weights are signed INT8; bias and partial sums are INT32.
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
override 32-byte alignment and its own storage and layout rules. Binding format
selection and internal format selection must remain separate calls so a scheduler
default cannot accidentally expose a native padded format at the model boundary.

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
- iDMA 1D, 2D, and 3D transfers between L2 and local memory.
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
- 12 logical input-side banks and 4 logical output-side banks.
- A 256-bit AXI data path, equivalent to 32 bytes per beat.

Software defines command staging as:

- Base address `0x1017F000`.
- Size 4 KB.
- The command table resides in L2 and is refilled into staging through iDMA.

The compiler allocator must:

- Use 32-byte alignment for tensors, commands, and constant transfers.
- Never allocate over command staging.
- Model one physical TCDM arena rather than adding logical aliases as separate
  physical memories.
- Respect access restrictions for IFM, weights, partial sums, and OFM.
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
- There are no complete linebuffer, pointwise, depthwise, AFU, or Spatz commands.
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
| Pointwise Conv | Direct GEMM32 | IC and OC are multiples of 32 |
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
DMA beat and alignment   32 bytes
physical TCDM            512 KB
command staging          4 KB in the reserved top window
linebuffer max kernel    5
linebuffer max stride    2
linebuffer max input W   640
preferred partial-sum M  <= 256
requant shift            0..31
runtime address width    32 bits after relocation
```

System configuration may change clocks, bandwidth, and latency for performance
estimation. It must not override hard RTL limits. `CheckConfiguration()` must
reject conflicting values.

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
    uint32_t crc32;
    uint32_t reserved;
} nai_section_v1_t;
```

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
    uint32_t reserved[7];
} nai_invocation_v1_t;
```

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

Rules:

- Preserve `type`, `size`, `flags`, `layer_id`, and `tile_id` in each command
  header for diagnostics.
- Every command size is divisible by 32 and every reserved field is zero.
- The runtime rejects unknown required command types.
- A minor-version command may be skipped only when marked optional and skippable.
- `RQ_LOAD` loads one quantization block into DTCM or runtime arrays, allowing
  several compute commands to reuse it.
- `LINEBUF_JOB` should initially embed one job descriptor for parser simplicity.
  Constant deduplication may be added later.
- `COPY_LAYOUT` carries source and destination formats, logical dimensions,
  valid channel count, and target-computed strides. Initial modes are
  `NHWC_TO_ROW32`, `ROW32_TO_NHWC`, `NHWC_TO_C32`, and `C32_TO_NHWC`.
- The runtime implements `COPY_LAYOUT` through a zero-fill plus iDMA 2D/3D plan
  when regular, with a Spatz fallback for gather/scatter cases. The command
  semantics do not expose which engine performs the conversion.
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
    Gemm32,
    FullyConnected,
    RgbStem3x3S2P1,
    PointwiseC32,
    Conv3x3C32S1P1,
    Conv3x3C32S2P1,
    Depthwise3x3C32S1P1,
    Depthwise3x3C32S2P1,
    AfuLut,
    AfuBinary,
    SpatzElementwise,
    MaxPool5x5S1P2,
    UpsampleNearest2x,
    GlobalAvgPoolC32,
    DflSoftmax4,
    ClassSigmoidHigh16,
};
```

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

- Obtain allocation quantum and alignment from the architecture; Neural-AI uses
  32 bytes.
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
total_blocking = sum(dma + compute)
total_overlap = critical_path(max(dma_stage, compute_stage))
```

Calibrate with PMU regression for:

- GEMM M sweeps.
- Pointwise IC and OC group sweeps.
- Conv tile H and W sweeps.
- S1 and S2 linebuffer modes.
- Depthwise channel and tail sweeps.
- DMA 1D, 2D, and 3D size and stride sweeps.
- AFU and Spatz element-count sweeps.

The performance model must not claim overlap while the command runtime remains
blocking.

## 8. Implementation Phases

## Phase 0 - Freeze Contracts and Golden Data

### Objective

Remove ambiguity between compiler, software, and RTL before backend development.

### Work Items

1. Specify ABI version 1 for `.nai`, invocation records, references, and commands.
2. Freeze the initial target as one cluster, INT8, static shape, and batch 1.
3. Freeze physical TCDM size, 4 KB staging, and 32-byte alignment.
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
- A cross-repository ABI manifest and version comparison.

### Exit Criteria

- ABI review is complete and the file format contains no native pointer.
- Memory constants agree with RTL and linker scripts.
- There is no source diff under `neural-ai/hw`.
- Existing software and RTL tests still pass.

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
- M values 1, 31, 32, 33, and 256.
- Misaligned or out-of-range bindings fail with the expected error code.
- Firmware text plus read-only data remains at or below 32 KB.

### Exit Criteria

- A GEMM package needs no model-specific C graph.
- V1 and v2 coexist.
- Runtime performs no tile planning.
- `neural-ai/hw` remains unchanged.

## Phase 2 - Neural-AI Target Skeleton in Regor

### Objective

Compile an INT8 TFLite FullyConnected or MatMul model into `.nai` and execute it
using the Phase 1 runtime.

### Work Items

1. Generalize architecture alignment, layout, and storage hooks.
2. Add Ethos default implementations and regression tests before changing the
   scheduler.
3. Add `REGOR_ARCH_NEURALAI` and the architecture factory entry.
4. Add the Neural-AI memory and configuration object.
5. Add initial MatMul and FullyConnected constraints and reject all other NPU
   operations.
6. Keep frontend shapes and public bindings in NHWC or their original lower-rank
   contiguous order.
7. Add ROW32 internal layout plus explicit boundary pack and unpack lowering.
8. Add the initial Neural-AI weight encoder.
9. Implement the minimal v2 `COPY_LAYOUT` runtime handler for contiguous-to-ROW32
   pack and ROW32-to-contiguous unpack, including zero-filled tails.
10. Add DMA, layout-conversion, and GEMM command generation.
11. Add the `.nai` writer, Python result type, and CLI output.
12. Add a Neural-AI config file and explicit CLI target mapping.

### Compiler Tests

- Factory and configuration parsing.
- 32-byte allocation and alignment.
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

### Exit Criteria

- The CLI successfully creates `.nai`.
- The generic runtime executes compiler-generated files.
- Output matches the reference.
- The model API exposes no ROW32 padding and requires no layout knowledge from
  the host.
- The compiler E2E test contains no manual command builder.

## Phase 3 - Conv, Linebuffer, and Per-Channel Quantization

### Objective

Support CNN backbones containing an RGB stem, pointwise Conv, C32 Conv, and
depthwise Conv.

### Work Items

1. Implement the Neural-AI Conv classifier and constraints.
2. Port the linebuffer tile planner from Python to C++.
3. Implement RGB, pointwise, Conv3x3, and depthwise weight encoders.
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
- Tile-fit and overflow tests at the TCDM boundary.
- C33, C48, C64, C65, and C96 channel cases.
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

### Exit Criteria

- Micro-MobileNet no longer requires a hand-written graph, weights, or
  descriptors.
- Per-channel bias and scale match the reference.
- Peak TCDM and firmware size are reported and enforced.
- Existing hand-written model tests still pass.

## Phase 4 - Vector, AFU, View, and Layout Operations

### Objective

Support the non-Conv operations in the current Neural-AI operator matrix.

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

Do not copy or modify test source under `hw`. The software test directory should
contain:

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

Using `SIM_BUILD=/tmp/...` avoids adding build output to the source tree. Existing
Makefiles and RTL source remain unchanged.

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
10. Git check confirming no source diff under `hw`.

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
- Leave `neural-ai/hw` unchanged.
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
- There is no source change under `/home/dev01/neural-ai/hw`.

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
