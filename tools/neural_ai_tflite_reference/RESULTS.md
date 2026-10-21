# Neural-AI reference versus original TFLite — 100-image demo

Measured 2026-10-07, TensorFlow 2.21.0, Intel Core i5-1145G7, one interpreter thread.
Dataset: first 100 `val` rows from [detection-datasets/coco](https://huggingface.co/datasets/detection-datasets/coco),
714 supplied object annotations. Identical preprocessed input bytes are used by all runtimes.
No Verilator run, RTL change, model edit, or compiler-lowering change was needed.

## Detection quality: YOLOv8n INT8, 320×320

| Runtime | AP50–95 | AP50 | AP75 |
| --- | ---: | ---: | ---: |
| Original TFLite reference (`BUILTIN_REF`) | 36.195% | 50.092% | 37.549% |
| Neural-AI hybrid reference | 35.184% | 48.832% | 37.921% |
| Original TFLite optimized (default/XNNPACK) | 34.742% | 48.677% | 37.005% |

Neural-AI reference is **−1.011 percentage points AP50–95** and −1.261 pp AP50 versus TFLite reference
on this sample. It is +0.441 pp AP50–95 versus default optimized TFLite. The latter also differs numerically
from TFLite reference; these results do not establish that any implementation is generally more accurate.
No statistical significance or full-dataset accuracy claim is made.

Rounding differences can therefore change final detections; they are not merely harmless byte-level noise.
Whether this quality delta is acceptable remains a product decision requiring representative validation.
The 100 ordered images are a small demo, not a random sample. HF annotations omit `iscrowd`; all supplied
boxes are treated as non-crowd. AP is computed on this subset using pycocotools 2.0.11, not official full COCO.

Numerical differences versus `BUILTIN_REF`:

| Output group | Neural-AI exact bytes | Mean absolute error (LSB) | Max absolute error (LSB) |
| --- | ---: | ---: | ---: |
| All 17,640,000 output elements | 97.177% | 0.06473 | 197 |
| 840,000 coordinate elements | 42.787% | 1.29361 | 127 |
| 16,800,000 class-score elements | 99.897% | 0.00328 | 197 |
| Coordinates at anchors with score ≥0.25 in either runtime | 63.047% | 0.55823 | 70 |

One coordinate LSB is approximately 1.342 pixels in the 320×320 letterboxed image. These are aligned
pre-NMS anchor comparisons, not matched final detections. Scores dominate the tensor and many are zero;
**97.177% byte agreement alone is not an accuracy measure**. Large outliers exist despite low average error.

## MobileNetV2 INT8, 224×224

| Runtime | Top-1 agreement with TFLite reference | Exact output bytes | Mean absolute error (LSB) | Max error |
| --- | ---: | ---: | ---: | ---: |
| Neural-AI hybrid reference | 96/100 | 25.706% | 1.30008 | 11 |
| Original TFLite optimized | 95/100 | 25.785% | 1.30552 | 12 |

The Neural-AI top-1 changes occur at image IDs 5600, 6763, 7108, 7511. Default optimized TFLite changes
these same four and image 7574. These are disagreements, **not measured errors**: COCO annotations do
not provide ImageNet ground-truth classes, so classification accuracy was not evaluated.

## Host CPU timing — not NPU performance

Two warmups/runtime; one invocation for each of 100 images, rotated execution order. Only `invoke()`
is timed; preprocessing, copies, NMS, evaluation and file output are excluded. Models run sequentially.

| Model/runtime | Mean ms | Median ms | P95 ms | Inferences/s |
| --- | ---: | ---: | ---: | ---: |
| YOLO / TFLite reference | 267.30 | 259.32 | 306.87 | 3.74 |
| YOLO / Neural-AI reference | 585.57 | 570.18 | 693.95 | 1.71 |
| YOLO / TFLite optimized | 7.09 | 6.97 | 7.95 | 141.10 |
| MobileNet / TFLite reference | 165.95 | 155.08 | 205.66 | 6.03 |
| MobileNet / Neural-AI reference | 304.13 | 282.78 | 395.41 | 3.29 |
| MobileNet / TFLite optimized | 3.73 | 3.59 | 4.87 | 268.17 |

The Neural-AI host reference is about 2.19×/1.83× slower than TFLite reference on YOLO/MobileNet.
It uses straightforward wide-integer loops, no tuned CPU convolution implementation. This overhead
does **not** measure the cost of single rounding in RTL and says nothing about hardware inference/s.
Host scheduling/frequency variation is visible in the latency tail; treat these as demo timings.
No Vtop process was active when timing started.

## Coverage and validation

- YOLO: 64/64 Conv operators use Neural-AI kernels.
- MobileNet: 35 Conv + 17 Depthwise + 1 FullyConnected use Neural-AI kernels.
- All other operators retain original TFLite reference semantics. No unsupported MAC fallback occurred.
- Six numerical unit-test groups pass in UBSan and release builds, including eight Conv parameter/activation
  subcases, signed ties, known stem discrepancies, per-channel scales, depth multiplier, optional FC bias,
  adjacent delegated nodes and explicit unsupported-scale fallback.
- Three evaluation tests pass: signed-byte errors, normalized box/NMS decoding, perfect/empty COCO predictions.
- Saved RTL stem: 102,400 Conv elements + 51,200 SiLU elements match Neural-AI reference exactly.
  TFLite reference has 100 Conv and 31 SiLU mismatches at these same checkpoints.
- First real image has per-tensor CSV comparisons for all 253 YOLO and 68 MobileNet operators. First Conv
  differs by at most 1 LSB (YOLO: 262/409,600 elements; MobileNet: 13,078/401,408). Subsequent differences
  include propagation through nonlinear operations and cannot be attributed solely to the local operator.

The original experiment above validates the MAC/requant reference and saved stem only.
**The default hybrid runtime is not an end-to-end RTL golden model**. The following
experiment adds an optional AFU DFL contract; AFU LUT/Mean/binary semantics still
need further independent modeling and hardware checkpoint checks.

## AFU DFL-only accuracy experiment — 2026-10-21

The new opt-in `neural_ai_afu_dfl_reference` models deployed AFU DFL16 arithmetic.
The default hybrid runtime, original TFLite file, delegate library, cached baseline
outputs and previous failing RTL comparison are unchanged. Contract ID is
`afu_dfl16_deployed_signed_delta_v1`; see [the contract and commands](README.md#deployed-afu-dfl-contract-and-accuracy-isolation).

### Hardware checkpoint validation

Using the retained corrected YOLO suffix and its intermediate snapshots, independently
generated LUTs and requant parameters match the deployed model. Inputs of all three
DFL jobs (134,400 logits bytes), all 8,400 DFL output bytes, and **all 176,400 final
RTL output bytes** match the new reference. The split source-DFL control still equals
the original pinned hybrid oracle; that oracle differs from the new reference in
3,465 box bytes, maximum 7. The old FAIL report is preserved, not relabeled.
This is an offline check of a retained input, **not a new reset simulation or proof
of RTL equality on the following 100 images**.

### Paired accuracy: same MAC prefix, only DFL changes

All 100 original images and 714 supplied boxes are reused. Each newly recomputed
split-graph source-DFL control matches its cached unchanged hybrid output byte-for-byte.
Every AFU-DFL image uses the identical Conv prefix and original TFLite tail. All
**16,800,000 raw class-score bytes remain exact** between these paired paths.
Cached TFLite/hybrid model, build, image/input/output identities are checked.

| Runtime | AP50–95 | AP50 | AP75 | AR@100 |
| --- | ---: | ---: | ---: | ---: |
| Original TFLite reference | 36.1950% | 50.0923% | 37.5486% | 51.1212% |
| Unchanged Neural-AI hybrid | 35.1838% | 48.8316% | 37.9213% | 50.8868% |
| Hybrid + deployed AFU DFL | 35.1726% | 48.6777% | 37.6612% | 50.9974% |
| **DFL-only change (percentage points)** | **−0.0111** | **−0.1538** | **−0.2601** | **+0.1106** |

The complete hybrid-plus-DFL change versus original TFLite is **−1.0224 pp
AP50–95**, of which only **−0.0111 pp** is added by DFL on this sample. It would
be incorrect to attribute the full −1.0224 pp to DFL: the earlier MAC/reference
differences already account for −1.0112 pp.

| Object area | Unchanged hybrid AP | Hybrid + AFU DFL AP | DFL-only change |
| --- | ---: | ---: | ---: |
| Small | 12.3347% | 11.6070% | −0.7277 pp |
| Medium | 31.3196% | 31.4319% | +0.1123 pp |
| Large | 64.7933% | 65.1133% | +0.3200 pp |

Overall AP is nearly unchanged on this demo, but small-object AP and AP75 show a
larger decrease. No statistical significance or general equivalence is claimed.
This is a deployment-validation question, not evidence that all quantization
differences are harmless.

### Box and detection behavior

- Of 840,000 raw box elements, 318,771 differ; mean absolute error is **0.4916
  LSB** and maximum **23 LSB**, not a global bound of 7 from the synthetic test.
  Per-channel maxima for normalized x/y/w/h are 14 / 7 / 23 / 11 LSB. At 320×320,
  one LSB is approximately 1.342 pixels; the largest width difference is about
  30.87 pixels, including low-confidence candidates.
- At the **2,718 anchors with confidence ≥0.25**, coordinate MAE is **0.5556 LSB**
  and maximum **5 LSB** (about 6.71 pixels). Aligned pre-NMS box IoU has median
  **0.9722**, fifth percentile **0.8947**, and minimum **0.5294**. These are
  comparisons at corresponding anchors, not matched ground-truth detections.
- With the unchanged evaluation postprocess (multilabel confidence ≥0.001,
  class-aware NMS IoU=0.7, max300), detection count changes from **12,250 to
  12,182**. Relative to the original hybrid anchor/class selections, **786 are
  removed and 718 added**, over 94 images. These are selection changes, not
  counts of missed or newly recognized ground-truth objects.
- Raw class scores and per-anchor class choices are unchanged. DFL does not
  directly swap cat/dog scores; changed localization can still affect NMS,
  matching against ground truth, and final detection metrics.
- None of **840,000 DFL distributions** exercises a logits-minus-max delta below
  −128 in these 100 images. The known wrapped-index LUT issue is modeled but its
  accuracy impact is **not** covered by this dataset result.

Reproduction artifacts (local, ignored):

- `results/afu_dfl_snapshot_check_provenance_20261021.json` and `.log`: retained-RTL
  checks with reference-source hashes (`afu_dfl_snapshot_check_20261021.*` is the
  initial identical numerical check).
- `results/yolo320_afu_dfl_val100_20261021/report.json`: metrics, per-class AP,
  exact contract, source/build/model/dataset hashes and limitations.
- The same directory contains `per_image.csv`, three detection JSONs, original
  and AFU outputs for every image, `ground_truth.json` and `coco_eval.log`.
- `results/yolo320_afu_dfl_val100_20261021.log`: benchmark progress/results.

The accuracy run took 66.86 seconds including paired source/AFU evaluation and
file output; this is **not** an inference-latency or NPU-performance measurement.
All 21 host regression tests pass (arithmetic, graph-control, MAC, evaluator and
staging helpers). No RTL,
firmware, compiler-lowering or installed TensorFlow changes are involved.

## Reproduction artifacts

All generated artifacts are local under `tools/neural_ai_tflite_reference/`:

- `data/coco_val100/manifest.json`: pinned images, IDs, classes, boxes, dimensions and hashes.
- `results/yolo320_val100/`, `results/mobilenet224_val100/`: report JSON, per-image CSV, raw NPZ outputs,
  output analysis, and YOLO detection/evaluation files.
- `results/{yolo,mobilenet}_first_image_tensors.csv`: layer-level comparisons.
- `results/stem_checkpoints.json`: saved-RTL comparison.
- `results/yolo320_val100.log`, `results/mobilenet224_val100.log`: full benchmark console logs.

SHA256 identities:

```text
YOLO TFLite:     2330cae180c11c0d3203883efe63ef677935867754dc6a81d4c88c773436921f
MobileNet:      cefdc91b5270a9deff9139d3c60931178a960d214304f8ea180e385e4183422f
Data manifest:  5125a16ffa9c94bc41418f2a4b178d410ba14f3a51cd9a526aa8e3808ca66d28
```

Build/header/library hashes and exact environment are embedded in each `report.json`.
See [README.md](README.md) for commands and methodology. Images and generated outputs are not committed.
