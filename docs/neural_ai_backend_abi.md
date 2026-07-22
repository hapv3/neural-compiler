# Neural-AI Model and Runtime ABI v1.0

## Scope

This document freezes the portable contract between Regor-generated `.nai`
packages and the single-cluster Neural-AI runtime. The initial target supports
static-shape, batch-1 INT8 graphs, INT32 bias and partial sums, 32-byte transfer
alignment, one 512 KiB TCDM arena, 32 KiB ITCM, and 32 KiB DTCM. The final 4 KiB
of TCDM at `0x1017f000` is reserved for command staging and is not allocatable
model scratch.

RTL and verified hardware regressions are authoritative for physical limits.
The wire contract is represented in C++ by
`architecture/neuralai/neural_ai_abi.hpp` and in firmware by
`sw/lib/npu_model_abi.h`. Both sides have size and offset assertions.

## Encoding rules

- Every integer is fixed-width and little-endian.
- A `.nai` file contains no native pointer and no absolute runtime address.
- The model header, section table, every section start, every section size, and
  every command are aligned to 32 bytes.
- All reserved fields are zero and must be rejected when non-zero unless a later
  compatible minor version explicitly assigns them.
- Ranges are checked as `offset <= total && size <= total - offset` to prevent
  overflow.
- ABI major 1 readers reject unknown required commands. A command may be skipped
  only when both OPTIONAL and SKIPPABLE flags are set.

## Package structure

`nai_model_header_v1_t` is 64 bytes and is followed by a table of 32-byte
`nai_section_v1_t` entries. Required section types are COMMANDS, CONSTANTS,
TENSORS, BINDINGS, and QPARAMS; DEBUG_MAP is optional. CRC32 covers section
payload bytes exactly, including deterministic zero padding.

The header identifies target 1 (single-cluster Neural-AI), ABI 1.0, the first
command, command count, model TCDM requirement, and public input/output counts.
TCDM requirements exclude the fixed command staging window.

The invocation record is 64 bytes. The earlier design sketch used seven reserved
words, which produced a 60-byte structure; ABI 1.0 uses eight reserved words so
the record is exactly two 32-byte beats.

## References and relocation

Each `nai_ref_v1_t` is an 8-byte tuple `(region, index, offset)`. Defined regions
are model constants, model commands, input binding, output binding, L2 temporary
binding, TCDM scratch, and runtime-owned DTCM. The loader validates region,
index, byte range, and command-specific alignment before producing a 32-bit
physical address. Model commands cannot make arbitrary references into DTCM.

## Tensors and public bindings

Frontend rank-4 logical shapes remain `[N,H,W,C]`. Public rank-4 bindings use
compact NHWC bytes; firmware maps this to the existing batch-1 `NPU_LAYOUT_HWC`
name. Public bindings never expose ROW32 or C32 padding.

Private tensors may use:

- `ROW32`: `rows * round_up(C, 32) * element_bytes`.
- `C32_BLOCKED`: `N * H * W * ceil(C / 32) * 32 * element_bytes`.

For C32, byte offset is
`(((c / 32) * (H * W) + y * W + x) * 32 + c % 32) * element_bytes`.
Padding lanes are zero on pack and discarded on unpack. Multi-group C32 is never
treated as affine NHWC by changing strides.

## Weight ordering

- RGB stem: `[k_lane=32][oc_lane=32]`; 27 valid C3/K3 taps and five zero taps.
- Pointwise and MatMul: `[output_group][input_group][k_lane][n_lane]`.
- Conv 3x3: `[output_group][input_group][kh][kw][k_lane][n_lane]`.
- Depthwise 3x3: `[channel_group][kh][kw][lane]`.

Out-of-range channels and taps are zero. Each constant blob begins at a 32-byte
aligned section-relative offset.

## Quantization

For output channel `oc`, `real_scale = ifm_scale * weight_scale / ofm_scale`.
The compiler searches shifts 0 through 31, selects a positive signed INT32
multiplier minimizing absolute scale error, and breaks ties toward the larger
shift. Bias uses accumulator units. Native dense and Conv modes require zero
weight zero-point.

The runtime computes with at least signed 64-bit intermediates:

`scaled = (accumulator + bias) * multiplier`

It divides by `2^shift`, rounds halfway cases away from zero, adds output zero
point, and applies the shared clamp bounds. No implementation-defined signed
right shift is part of the reference semantics.

## Compatibility

The legacy `NPUC` command table remains ABI v1 and is dispatched unchanged.
`NAIV` selects the invocation/model loader and command ABI v2. An incompatible
wire change increments the model ABI major. Additive fields or optional,
skippable commands increment the minor version.
