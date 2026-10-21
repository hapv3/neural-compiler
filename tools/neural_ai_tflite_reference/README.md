# Neural-AI TFLite reference runtime

An independent **host-side, hybrid numerical reference**. MAC kernels use a TFLite external delegate;
an optional graph adapter models fused AFU DFL16 without changing the default runtime.
It does not modify the model, installed TensorFlow, compiler, firmware, or RTL.

## Numerical scope

| Operation | Runtime implementation |
| --- | --- |
| INT8 Conv2D, DepthwiseConv2D, FullyConnected | Independent scalar C++ accumulation + Neural-AI multiplier/shift + single rounding, ties away from zero |
| Other operators | Unmodified TFLite `BUILTIN_REF` kernels |

The default `neural_ai_reference` remains the original MAC-only hybrid baseline.
Opt-in `neural_ai_afu_dfl_reference` additionally replaces one recognized private
`Softmax -> 16-bin projection Conv -> views -> constant box-scale Mul -> view`
chain with independent deployed AFU arithmetic. `tflite_afu_dfl_reference` applies
the same DFL arithmetic with original TFLite MACs. Unsupported DFL graphs raise an
error; they do not silently fall back. No fixed tensor IDs or 2,100-location constraint
is used. Current recognition requires static INT8 `[1,4,L,16]` logits, beta 1,
projection weights approximating bins 0–15, zero projection bias, private
quantization-preserving views, and positive box scales equal across the four sides.

MAC kernels support per-tensor/per-channel symmetric INT8 weights, INT32 bias, input/output zero points,
SAME/VALID padding, stride/dilation, depth multiplier, and None/Relu/Relu6/ReluN1To1 activation clamps.
FullyConnected accepts optional bias and flattens the input as specified by its static output shape.
Each MAC operator is its own delegate partition, retaining ordinary CPU intermediate tensors.
Only resolved, positive shapes and constant, non-sparse weights are accepted; unsupported MAC nodes
remain on TFLite and are listed in `mac_fallback_nodes`. The benchmark refuses these fallbacks.
Recreate the interpreter for a different input shape; this runtime is not a dynamic-shape backend.

Multiplier selection searches shifts 0..31 with positive signed-32-bit multipliers, minimizes the real-scale
error, and breaks ties in favor of the larger shift. Requantization uses a wide integer product and one
rounding operation, adds output zero point, then clamps. This mirrors the compiler quantization contract
without calling compiler code. Accumulation uses int64 and rejects a biased result outside signed 34-bit
range. **Intermediate systolic overflow/wrap behavior is not modeled**; this reference is for non-overflowing
MACs. It does not model TCDM layouts, DMA, tiling, scheduling, or cycles.

AFU LUT fusion, hardware Mean and general binary requant are not independently modeled here.
DFL is hardware-modeled only in the explicitly selected AFU mode. Therefore **this
is not an unrestricted, full-graph RTL golden runtime**. Agreement on a retained
input does not establish dataset-wide RTL byte-exactness or hardware timing.

## Build and tests

From the compiler repository root (Python with TensorFlow, NumPy, FlatBuffers, Pillow; g++ required):

```bash
python3 tools/neural_ai_tflite_reference/build.py --sanitize
python3 tools/neural_ai_tflite_reference/test_runtime.py -v
PYTHONPATH=tools/neural_ai_tflite_reference python3 -m unittest test_afu_dfl test_runtime test_benchmark -v

# COCO evaluator is isolated; do not replace system packages.
python3 -m pip install --no-deps \
  --target tools/neural_ai_tflite_reference/build/python-deps pycocotools==2.0.11
python3 tools/neural_ai_tflite_reference/test_benchmark.py -v

# Use an unsanitized build for timing. Do not rebuild during a benchmark.
python3 tools/neural_ai_tflite_reference/build.py
```

The build fetches only official headers from the tag matching installed TensorFlow (validated with 2.21.0),
not TensorFlow sources/libraries. Its manifest records header, source and shared-library SHA256 values.
The wrapper rejects a TensorFlow version mismatch or a stale source/library build.
`--sanitize` enables UBSan and aborts on undefined behavior. No synthesis or Verilator is involved.

## Use as an interpreter

```python
# Add tools/neural_ai_tflite_reference to PYTHONPATH.
from runtime import make_interpreter, coverage

runtime = make_interpreter("model.tflite", preserve=True)
print(coverage(runtime))
runtime.set_tensor(runtime.get_input_details()[0]["index"], input_int8)
runtime.invoke()
output = runtime.get_tensor(runtime.get_output_details()[0]["index"])
```

Use `kind="tflite_reference"` for the original reference interpreter, or `kind="tflite_optimized"`
for original default CPU kernels/delegates (XNNPACK in this environment). Keep a runtime alive only while
needed; explicitly delete interpreter references before Python shutdown when embedding it in scripts.

## Deployed AFU DFL contract and accuracy isolation

```python
runtime = make_interpreter("model.tflite", kind="neural_ai_afu_dfl_reference", preserve=True)
runtime.set_tensor(runtime.get_input_details()[0]["index"], input_int8)
runtime.invoke()
output = runtime.get_tensor(runtime.get_output_details()[0]["index"])

# Accuracy control: execute one shared prefix, source DFL and AFU DFL, then the
# same tail twice. The returned baseline must match an unchanged hybrid run.
baseline, afu_output, source_distances = runtime.invoke_comparison()
print(runtime.last_dfl_stats)  # Includes use of deltas below -128.
```

The wrapper constructs prefix/middle/tail graphs **in memory** from the original
TFLite edges; it never edits the source file. Prefix MACs and class scores stay
identical between the paired paths. The AFU reference independently generates
exp/reciprocal tables and 16-bit multiplier/right-shift parameters from source
quantization, computes the unsigned Q8 expectation with integer rounding, and
applies the folded box-scale requantization. It reads no compiler-generated LUT
or RTL output to produce a prediction. Removed Softmax/projection intermediates
are explicitly unavailable through `get_tensor()` rather than fabricated.

Contract ID **`afu_dfl16_deployed_signed_delta_v1`** deliberately includes the
current compiler's signed-byte exp-index behavior: wrapped differences -255…-129
alias indices 1…127 whose entries are 32768. This is a known latent LUT-generation
issue, not an ideal Softmax definition. It must not be silently corrected inside
the reference. Counters expose whether an input exercises it; a future compiler
fix requires a separately identified contract and regenerated verification cases.

Check the new reference against the already-retained three DFL snapshots and full
output without starting a simulator:

```bash
python3 tools/neural_ai_tflite_reference/verify_afu_dfl.py \
  --case /home/dev01/neural-ai/hw/rtl/cluster/tb/sim/test_yolo_cascade_l2_20261013/run-remaining-binary-alignment/case.json \
  --neural-ai-root /home/dev01/neural-ai \
  --output tools/neural_ai_tflite_reference/results/afu_dfl_check.json
```

Evaluate the isolated DFL effect on the pinned 100-image set:

```bash
python3 tools/neural_ai_tflite_reference/benchmark_dfl.py \
  --model test/model/yolov8n_320_int8.tflite \
  --baseline tools/neural_ai_tflite_reference/results/yolo320_val100 \
  --output tools/neural_ai_tflite_reference/results/yolo320_afu_dfl_val100
```

This accuracy-only benchmark validates model/dataset/MAC-build identities, input
hashes and cached output hashes. Every freshly recomputed split control must be
byte-exact with the original hybrid output. It reports original TFLite, unchanged
hybrid and hybrid-plus-AFU-DFL AP/AR, per-class AP, small/medium/large AP, raw score
invariance, NMS anchor changes, high-confidence aligned box IoU, and wrapped-delta
counts. Outputs, per-image CSV, detections, provenance and full evaluator logs are
saved in a new directory; previous evidence is never overwritten. This is not a
benchmark of 100 RTL inferences or NPU performance.

## Download the requested 100-image set

```bash
python3 tools/neural_ai_tflite_reference/download_demo.py
```

Source: [detection-datasets/coco](https://huggingface.co/datasets/detection-datasets/coco).
The downloader saves the first 100 `val` viewer rows, original-resolution images and all supplied
annotations to `data/coco_val100/`. It does not download the full dataset. The current selection contains
714 boxes. `manifest.json` records image IDs, hashes, dimensions, class names, xyxy boxes, areas and
the dataset revision observed at download. Saved image hashes and annotations identify the exact local
subset; the viewer cache is not guaranteed to track the observed repository revision instantaneously.
Re-running verifies the pinned local images instead of silently replacing the sample set.
Observe the source dataset and original image licensing when redistributing; images are not committed.

The supplied annotations omit `iscrowd`; the evaluator assumes zero. Results are **subset detection
metrics**, not an official full COCO benchmark. The first 100 rows are not a random representative sample.

## Benchmark both models

```bash
python3 tools/neural_ai_tflite_reference/benchmark.py \
  --model test/model/yolov8n_320_int8.tflite --task yolo \
  --output tools/neural_ai_tflite_reference/results/yolo320_val100

python3 tools/neural_ai_tflite_reference/benchmark.py \
  --model test/model/mobilenet_v2_224_int8.tflite --task mobilenet \
  --output tools/neural_ai_tflite_reference/results/mobilenet224_val100
```

Choose a new output directory for reruns. Run the two benchmarks sequentially to avoid competing with
one another. By default, each runtime warms up twice and invokes once per image (100 timed invocations),
one configured interpreter thread. Execution order rotates across images. `--limit` supports smoke tests.

Three runtimes receive identical input bytes. Timing covers `invoke()` only: no model loading, image
preprocessing, tensor copies, NMS, evaluation or output compression. CPU model/load, model and input
hashes, build identity and coverage are recorded. **These CPU times cannot predict Neural-AI NPU speed.**
The C++ reference favors clarity/wide arithmetic, not optimized CPU throughput; compare numerical
quality separately from its speed relative to TFLite reference and XNNPACK.

YOLO adapter: RGB, PIL bilinear resize, centered 114 letterbox, /255 then input quantization. The provided
export has `[1,84,2100]` output: normalized xywh and 80 class scores. Detection uses multilabel confidence
>=0.001, class-aware NMS IoU=0.7 and at most 300 predictions/image; COCO evaluation uses maxDets=100
and IoU thresholds 0.50:0.05:0.95. All runtime variants share exactly the same adapter.

MobileNet adapter: RGB bilinear short-side 256, center crop 224, /127.5-1 then quantization. COCO labels
are not ImageNet labels: report top-1 **agreement** versus TFLite, not classification accuracy.

Outputs in each run directory:

- `report.json`: aggregate timing, inference/s, numerical errors, coverage and YOLO AP/AR.
- `per_image.csv`: invocation time, mismatches, max/mean/RMS LSB error, input/output hashes; flushed per image.
- `outputs/<image_id>.npz`: all three raw output tensors, for independent re-analysis.
- `*_detections.json`, `ground_truth.json`, `coco_eval.log`: YOLO predictions, subset GT and evaluator details.

Re-analyze saved outputs (no inference) to separate coordinates from scores, or list changed top-1 images:

```bash
python3 tools/neural_ai_tflite_reference/analyze_results.py \
  tools/neural_ai_tflite_reference/results/yolo320_val100
python3 tools/neural_ai_tflite_reference/analyze_results.py \
  tools/neural_ai_tflite_reference/results/mobilenet224_val100
```

This writes `output_analysis.json`. See [RESULTS.md](RESULTS.md) for the measured 100-image comparison.

## Layer-level and saved RTL verification

```bash
python3 tools/neural_ai_tflite_reference/compare_tensors.py \
  --model test/model/yolov8n_320_int8.tflite --task yolo \
  --image tools/neural_ai_tflite_reference/data/coco_val100/000000000139.jpg \
  --output tools/neural_ai_tflite_reference/results/yolo_first_image_tensors.csv

python3 tools/neural_ai_tflite_reference/verify_checkpoints.py \
  --model test/model/yolov8n_320_int8.tflite \
  --checkpoints /home/dev01/neural-ai/hw/rtl/cluster/tb/sim/test_yolo_stem_byte_exact \
  --output tools/neural_ai_tflite_reference/results/stem_checkpoints.json
```

Layer CSV differences include propagation from earlier operators, not just local operator error.
The checkpoint verifier checks source-model and saved-byte hashes, feeds the saved input into both
interpreters and compares logical INT8 bytes. It fails if any checked Neural-AI reference tensor differs
from the saved RTL. It never launches or changes simulation and never edits the original TFLite model.

Current saved-stem check: 102,400 Conv bytes plus 51,200 SiLU bytes match the hybrid runtime exactly;
the original TFLite reference differs in 100 Conv bytes and 31 SiLU bytes respectively.

Generated headers, binaries, datasets and benchmark outputs remain local/ignored. Source and instructions
are intended to be versioned; no changes to RTL or compiler lowering are required for this runtime.
