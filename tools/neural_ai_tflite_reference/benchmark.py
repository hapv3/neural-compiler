"""Compare unchanged TFLite models on a pinned local image set (host CPU only)."""

import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image
import tensorflow as tf

from runtime import ROOT, coverage, make_interpreter

KINDS = ("tflite_reference", "neural_ai_reference", "tflite_optimized")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_image(path, task, detail):
    _, height, width, channels = map(int, detail["shape"])
    if channels != 3 or detail["dtype"] != np.int8:
        raise ValueError("This benchmark adapter requires NHWC RGB INT8 models")
    with Image.open(path) as image:
        image = image.convert("RGB")
        original = image.size
        if task == "yolo":
            ratio = min(width / image.width, height / image.height)
            resized = (round(image.width * ratio), round(image.height * ratio))
            left, top = (width - resized[0]) // 2, (height - resized[1]) // 2
            canvas = Image.new("RGB", (width, height), (114, 114, 114))
            canvas.paste(image.resize(resized, Image.Resampling.BILINEAR), (left, top))
            values = np.asarray(canvas, np.float32) / 255.0
            transform = {"original_width": original[0], "original_height": original[1],
                         "scale_x": resized[0] / original[0], "scale_y": resized[1] / original[1],
                         "left": left, "top": top, "width": width, "height": height}
        else:
            # Conventional MobileNet evaluation adapter, not an ImageNet accuracy claim.
            resize_short = round(min(width, height) / 0.875)
            ratio = resize_short / min(original)
            resized = (round(original[0] * ratio), round(original[1] * ratio))
            image = image.resize(resized, Image.Resampling.BILINEAR)
            left, top = (image.width - width) // 2, (image.height - height) // 2
            values = np.asarray(image.crop((left, top, left + width, top + height)), np.float32) / 127.5 - 1.0
            transform = None
    scale, zp = detail["quantization"]
    if scale <= 0:
        raise ValueError("Missing input quantization")
    quantized = np.clip(np.rint(values / scale) + zp, -128, 127).astype(np.int8)[None]
    return quantized, transform


def difference(actual, reference):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError("Output dtype/shape mismatch")
    delta = actual.astype(np.int64) - reference.astype(np.int64)
    return {"elements": int(delta.size), "mismatches": int(np.count_nonzero(delta)),
            "max_abs_lsb": int(np.abs(delta).max(initial=0)), "mae_lsb": float(np.abs(delta).mean()),
            "rmse_lsb": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2)))}


def box_iou(box, boxes):
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-12)


def decode_yolo(output, detail, transform, image_id, confidence=0.001, nms_iou=0.7, include_anchor=False):
    """Adapter for this exported YOLOv8: [1,84,N], normalized xywh + 80 scores."""
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 84:
        raise ValueError("Expected YOLOv8 [1,84,N] output, without embedded NMS")
    scale, zp = detail["quantization"]
    prediction = (output[0].astype(np.float64).T - zp) * scale
    anchor, category = np.nonzero(prediction[:, 4:] >= confidence)
    scores = prediction[anchor, category + 4]
    xywh = prediction[anchor, :4] * [transform["width"], transform["height"], transform["width"], transform["height"]]
    boxes = np.concatenate((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), axis=1)
    order = np.argsort(-scores, kind="stable")[:30000]
    keep = []
    while len(order) and len(keep) < 300:
        index = order[0]
        keep.append(index)
        rest = order[1:]
        suppressed = (category[rest] == category[index]) & (box_iou(boxes[index], boxes[rest]) > nms_iou)
        order = rest[~suppressed]
    detections = []
    for index in keep:
        x1, y1, x2, y2 = boxes[index]
        x1, x2 = np.clip((np.array([x1, x2]) - transform["left"]) / transform["scale_x"], 0, transform["original_width"])
        y1, y2 = np.clip((np.array([y1, y2]) - transform["top"]) / transform["scale_y"], 0, transform["original_height"])
        if x2 > x1 and y2 > y1:
            detections.append({"image_id": image_id, "category_id": int(category[index]),
                               "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                               "score": float(scores[index])})
            if include_anchor:
                detections[-1]['anchor_index'] = int(anchor[index])
    return detections


def evaluate_coco(manifest, samples, predictions, output, detailed=False):
    sys.path.insert(0, str(ROOT / "build/python-deps"))
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = {"info": {"description": "HF COCO val subset; iscrowd unavailable, assumed zero"},
                    "images": [{"id": s["image_id"], "width": s["width"], "height": s["height"]} for s in samples],
                    "categories": [{"id": i, "name": name} for i, name in enumerate(manifest["classes"])],
                    "annotations": []}
    for sample in samples:
        objects = sample["objects"]
        for box, category, area in zip(objects["bbox"], objects["category"], objects["area"]):
            x1, y1, x2, y2 = box
            ground_truth["annotations"].append({"id": len(ground_truth["annotations"]) + 1,
                "image_id": sample["image_id"], "category_id": category, "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": area, "iscrowd": 0})
    (output / "ground_truth.json").write_text(json.dumps(ground_truth))
    metrics = {}
    log = io.StringIO()
    with redirect_stdout(log):
        coco = COCO(str(output / "ground_truth.json"))
        for kind, detections in predictions.items():
            (output / f"{kind}_detections.json").write_text(json.dumps(detections))
            if detections:
                result = coco.loadRes(detections)
            else:
                result = COCO()
                result.dataset = {**ground_truth, "annotations": []}
                result.createIndex()
            evaluator = COCOeval(coco, result, "bbox")
            evaluator.params.imgIds = [s["image_id"] for s in samples]
            evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
            metrics[kind] = {"AP_50_95": float(evaluator.stats[0]), "AP_50": float(evaluator.stats[1]),
                             "AP_75": float(evaluator.stats[2]), "AR_max100": float(evaluator.stats[8]),
                             "detections": len(detections)}
            if detailed:
                metrics[kind].update(AP_small=float(evaluator.stats[3]), AP_medium=float(evaluator.stats[4]),
                                     AP_large=float(evaluator.stats[5]), per_category={})
                for index, name in enumerate(manifest['classes']):
                    precision = evaluator.eval['precision'][:, :, index, 0, -1]
                    valid = precision[precision >= 0]
                    p50 = precision[0]
                    p75 = precision[5]
                    metrics[kind]['per_category'][name] = {
                        'AP_50_95': float(valid.mean()) if valid.size else None,
                        'AP_50': float(p50[p50 >= 0].mean()) if np.any(p50 >= 0) else None,
                        'AP_75': float(p75[p75 >= 0].mean()) if np.any(p75 >= 0) else None}
    (output / "coco_eval.log").write_text(log.getvalue())
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task", choices=["yolo", "mobilenet"], required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/coco_val100")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.limit <= 0 or args.warmup < 0:
        parser.error("--limit must be positive; --warmup must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "per_image.csv").exists():
        parser.error("Output already has a benchmark; choose another output directory")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    samples = manifest["samples"][:args.limit]
    if len(samples) != args.limit:
        raise ValueError("Not enough pinned samples")
    for sample in samples:
        if digest(args.dataset / sample["file"]) != sample["sha256"]:
            raise ValueError(f"Image hash changed: {sample['file']}")
    runners = {kind: make_interpreter(args.model, kind, threads=1) for kind in KINDS}
    report = {"model": str(args.model.resolve()), "model_sha256": digest(args.model),
              "dataset_manifest_sha256": digest(args.dataset / "manifest.json"),
              "images": len(samples), "boxes": sum(len(s["objects"]["bbox"]) for s in samples),
              "tensorflow": tf.__version__, "python": platform.python_version(), "host": platform.platform(),
              "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), "unknown"),
              "loadavg_start": os.getloadavg(), "threads": 1, "warmup": args.warmup,
              "timing": "invoke only, no preprocessing/copy/NMS; alternating runtime order for each image; host CPU, not NPU cycles",
              "numeric_contract": "Conv2D/Depthwise/FC use Neural-AI single-round requant. All other ops remain TFLite reference.",
              "limitations": ["Not a full RTL golden runtime; fallback LUT/Mean/binary/DFL semantics are not hardware-modeled.",
                              "100 ordered validation images, not full COCO accuracy; no statistical significance claim.",
                              manifest["limitations"], "Host timing may be affected by concurrent simulation/system load."],
              "preprocessing": "RGB PIL bilinear; YOLO centered letterbox 114, /255; MobileNet short-side 256 + center224, /127.5-1",
              "yolo_postprocess": "normalized xywh, multilabel confidence>=0.001, class-aware NMS IoU=0.7, max300",
              "build": json.loads((ROOT / "build/manifest.json").read_text()),
              "coverage": coverage(runners["neural_ai_reference"])}
    if report["coverage"]["mac_fallback_nodes"]:
        raise RuntimeError(f"Unsupported MAC nodes: {report['coverage']['mac_fallback_nodes']}")
    details = runners["tflite_reference"].get_input_details()
    if len(details) != 1 or any(len(r.get_output_details()) != 1 for r in runners.values()):
        raise ValueError("Benchmark adapters require exactly one input/output")
    first, _ = prepare_image(args.dataset / samples[0]["file"], args.task, details[0])
    for runner in runners.values():
        runner.set_tensor(runner.get_input_details()[0]["index"], first)
        for _ in range(args.warmup):
            runner.invoke()
    fields = ["image_id", "runtime", "invoke_ms", "elements", "mismatches", "max_abs_lsb", "mae_lsb", "rmse_lsb",
              "dequant_mae", "top1", "top1_reference", "output_sha256", "input_sha256"]
    records = []
    predictions = {kind: [] for kind in KINDS}
    (args.output / "outputs").mkdir(exist_ok=True)
    with (args.output / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, sample in enumerate(samples):
            values, transform = prepare_image(args.dataset / sample["file"], args.task, details[0])
            outputs, times = {}, {}
            order = KINDS[index % len(KINDS):] + KINDS[:index % len(KINDS)]
            for kind in order:
                runner = runners[kind]
                runner.set_tensor(runner.get_input_details()[0]["index"], values)
                start = time.perf_counter_ns(); runner.invoke(); end = time.perf_counter_ns()
                times[kind] = (end - start) / 1e6
                outputs[kind] = runner.get_tensor(runner.get_output_details()[0]["index"])
            np.savez_compressed(args.output / "outputs" / f"{sample['image_id']:012d}.npz", **outputs)
            for kind in KINDS:
                output_detail = runners[kind].get_output_details()[0]
                stats = difference(outputs[kind], outputs["tflite_reference"])
                record = {"image_id": sample["image_id"], "runtime": kind, "invoke_ms": times[kind], **stats,
                          "dequant_mae": stats["mae_lsb"] * output_detail["quantization"][0],
                          "top1": int(outputs[kind].argmax()) if args.task == "mobilenet" else "",
                          "top1_reference": int(outputs["tflite_reference"].argmax()) if args.task == "mobilenet" else "",
                          "output_sha256": hashlib.sha256(outputs[kind].tobytes()).hexdigest(),
                          "input_sha256": hashlib.sha256(values.tobytes()).hexdigest()}
                writer.writerow(record); records.append(record)
                if args.task == "yolo":
                    predictions[kind].extend(decode_yolo(outputs[kind], output_detail, transform, sample["image_id"]))
            handle.flush()
            if (index + 1) % 10 == 0 or index == 0:
                print(f"{args.task}: {index + 1}/{len(samples)} images; invoke ms {times}", flush=True)
    report["results"] = {}
    for kind in KINDS:
        rows = [r for r in records if r["runtime"] == kind]
        milliseconds = np.array([r["invoke_ms"] for r in rows])
        elements = sum(r["elements"] for r in rows)
        mismatches = sum(r["mismatches"] for r in rows)
        result = {"mean_ms": float(milliseconds.mean()), "median_ms": float(np.median(milliseconds)),
                  "p95_ms": float(np.percentile(milliseconds, 95)), "min_ms": float(milliseconds.min()),
                  "max_ms": float(milliseconds.max()), "inferences_per_second": 1000 / float(milliseconds.mean()),
                  "elements": elements, "mismatches": mismatches, "exact_percent": 100 * (1 - mismatches / elements),
                  "max_abs_lsb": max(r["max_abs_lsb"] for r in rows),
                  "mae_lsb": sum(r["mae_lsb"] * r["elements"] for r in rows) / elements,
                  "dequant_mae": sum(r["dequant_mae"] * r["elements"] for r in rows) / elements}
        if args.task == "mobilenet":
            result["top1_agreement_percent"] = 100 * sum(r["top1"] == r["top1_reference"] for r in rows) / len(rows)
            result["accuracy"] = "Not measured: COCO labels are not ImageNet labels"
        report["results"][kind] = result
    if args.task == "yolo":
        report["detection_metrics"] = evaluate_coco(manifest, samples, predictions, args.output)
    report["loadavg_end"] = os.getloadavg()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k in ("results", "detection_metrics")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
