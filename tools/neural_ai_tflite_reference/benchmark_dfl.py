"""Isolate deployed AFU DFL accuracy on the pinned COCO demo, host-only.

Every split-graph control must exactly match the cached unchanged hybrid
output. Cached original TFLite outputs, images and preprocessing are hash-checked.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from afu_dfl import CONTRACT
from benchmark import decode_yolo, difference, digest, evaluate_coco, prepare_image
from runtime import ROOT, coverage, make_interpreter

KINDS = ('tflite_reference', 'neural_ai_reference', 'neural_ai_afu_dfl_reference')


def aligned_iou(actual, expected, quantization):
    scale, zp = quantization
    a, b = ((v[0, :4].astype(np.float64).T - zp) * scale for v in (actual, expected))
    a = np.concatenate((a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2), axis=1)
    b = np.concatenate((b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2), axis=1)
    intersection = np.maximum(0, np.minimum(a[:, 2:], b[:, 2:]) - np.maximum(a[:, :2], b[:, :2])).prod(axis=1)
    aa = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    ab = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    return intersection / np.maximum(aa + ab - intersection, 1e-12)


def require_cached_output(folder, rows, image_id, kind, input_sha):
    record = rows.get((image_id, kind))
    if record is None or record['input_sha256'] != input_sha:
        raise ValueError('Cached baseline input identity differs')
    with np.load(folder / 'outputs' / f'{image_id:012d}.npz', allow_pickle=False) as outputs:
        result = outputs[kind]
    if hashlib.sha256(result.tobytes()).hexdigest() != record['output_sha256']:
        raise ValueError('Cached baseline output hash differs')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/coco_val100')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'results/yolo320_val100')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error('--limit must be positive')
    if args.output.exists():
        parser.error('Choose a new output directory; do not overwrite evidence')
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    samples = manifest['samples'][:args.limit]
    if len(samples) != args.limit:
        raise ValueError('Not enough pinned images')
    base_report = json.loads((args.baseline / 'report.json').read_text())
    build = json.loads((ROOT / 'build/manifest.json').read_text())
    if (base_report['model_sha256'] != digest(args.model) or
            base_report['dataset_manifest_sha256'] != digest(manifest_path) or base_report['build'] != build):
        raise ValueError('Cached model/dataset/MAC reference build identity differs')
    rows = list(csv.DictReader((args.baseline / 'per_image.csv').open()))
    cached = {(int(r['image_id']), r['runtime']): r for r in rows}
    if len(cached) != len(rows):
        raise ValueError('Duplicate cached benchmark rows')
    runner = make_interpreter(args.model, 'neural_ai_afu_dfl_reference', preserve=True)
    cov = coverage(runner)
    if cov['mac_fallback_nodes']:
        raise ValueError('Unsupported MACs in the DFL prefix/tail')
    args.output.mkdir(parents=True)
    (args.output / 'outputs').mkdir()
    predictions = {kind: [] for kind in KINDS}
    records, high_ious = [], []
    all_delta = []
    wrapped_elements = wrapped_distributions = 0
    start = time.perf_counter()
    fields = ['image_id', 'input_sha256', 'control_output_sha256', 'afu_output_sha256',
              'box_elements', 'box_mismatches', 'box_max_lsb', 'box_mae_lsb',
              'class_mismatches', 'high_score_anchors', 'high_score_box_max_lsb',
              'high_score_box_mae_lsb', 'nms_baseline', 'nms_afu', 'nms_removed', 'nms_added',
              'wrapped_delta_elements', 'wrapped_delta_distributions', 'dfl_min_delta']
    with (args.output / 'per_image.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, sample in enumerate(samples):
            path = args.dataset / sample['file']
            if digest(path) != sample['sha256']:
                raise ValueError('Pinned image changed')
            values, transform = prepare_image(path, 'yolo', runner.get_input_details()[0])
            input_sha = hashlib.sha256(values.tobytes()).hexdigest()
            original = require_cached_output(args.baseline, cached, sample['image_id'], KINDS[0], input_sha)
            baseline = require_cached_output(args.baseline, cached, sample['image_id'], KINDS[1], input_sha)
            runner.set_tensor(runner.get_input_details()[0]['index'], values)
            control, actual, _ = runner.invoke_comparison()
            if not np.array_equal(control, baseline):
                raise AssertionError('Split-graph control differs from the unchanged hybrid baseline')
            if not np.array_equal(actual[:, 4:], baseline[:, 4:]):
                raise AssertionError('DFL must not change raw class scores')
            detail = runner.get_output_details()[0]
            outputs = dict(zip(KINDS, (original, baseline, actual)))
            np.savez_compressed(args.output / 'outputs' / f"{sample['image_id']:012d}.npz", **outputs)
            detections = {}
            for kind, output in outputs.items():
                detections[kind] = decode_yolo(output, detail, transform, sample['image_id'], include_anchor=True)
                predictions[kind].extend(detections[kind])
            before = {(v['anchor_index'], v['category_id']) for v in detections[KINDS[1]]}
            after = {(v['anchor_index'], v['category_id']) for v in detections[KINDS[2]]}
            stats = difference(actual[:, :4], baseline[:, :4])
            scores = (baseline[0, 4:].astype(np.int16) - detail['quantization'][1]) * detail['quantization'][0]
            high = scores.max(axis=0) >= 0.25
            high_stats = difference(actual[:, :4, high], baseline[:, :4, high]) if high.any() else None
            high_ious.extend(aligned_iou(actual, baseline, detail['quantization'])[high].tolist())
            all_delta.append(actual[0, :4].astype(np.int16) - baseline[0, :4].astype(np.int16))
            dfl = runner.last_dfl_stats
            wrapped_elements += dfl['wrapped_delta_elements']
            wrapped_distributions += dfl['wrapped_delta_distributions']
            record = dict(image_id=sample['image_id'], input_sha256=input_sha,
                          control_output_sha256=hashlib.sha256(control.tobytes()).hexdigest(),
                          afu_output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                          box_elements=stats['elements'], box_mismatches=stats['mismatches'],
                          box_max_lsb=stats['max_abs_lsb'], box_mae_lsb=stats['mae_lsb'], class_mismatches=0,
                          high_score_anchors=int(high.sum()), high_score_box_max_lsb=high_stats['max_abs_lsb'] if high_stats else 0,
                          high_score_box_mae_lsb=high_stats['mae_lsb'] if high_stats else 0,
                          nms_baseline=len(before), nms_afu=len(after), nms_removed=len(before - after), nms_added=len(after - before),
                          wrapped_delta_elements=dfl['wrapped_delta_elements'],
                          wrapped_delta_distributions=dfl['wrapped_delta_distributions'], dfl_min_delta=dfl['min_delta'])
            writer.writerow(record)
            stream.flush()
            records.append(record)
            if (index + 1) % 10 == 0 or index == 0:
                print(f"DFL {index + 1}/{len(samples)}: control exact; box max={record['box_max_lsb']} LSB; "
                      f"wrapped distributions={wrapped_distributions}", flush=True)
    metrics = evaluate_coco(manifest, samples, predictions, args.output, detailed=True)
    delta = np.concatenate(all_delta, axis=1)
    all_high = sum(r['high_score_anchors'] for r in records)
    report = dict(contract=CONTRACT, model_sha256=digest(args.model), dataset_manifest_sha256=digest(manifest_path),
                  baseline_report_sha256=digest(args.baseline / 'report.json'), baseline_csv_sha256=digest(args.baseline / 'per_image.csv'),
                  build=build, reference_sources_sha256={f: digest(ROOT / f) for f in ('runtime.py', 'afu_dfl.py', 'benchmark.py', 'benchmark_dfl.py')},
                  images=len(samples), annotations=sum(len(s['objects']['bbox']) for s in samples),
                  baseline_control_exact_images=len(samples), class_score_mismatches=0,
                  coverage=cov, detection_metrics=metrics, wall_seconds=time.perf_counter() - start,
                  box_difference=dict(elements=int(delta.size), mismatches=int(np.count_nonzero(delta)),
                                      max_lsb=int(np.abs(delta).max()), mae_lsb=float(np.abs(delta).mean()),
                                      per_channel_max_lsb=np.abs(delta).max(axis=1).tolist(),
                                      high_score_anchors=all_high,
                                      high_score_max_lsb=max(r['high_score_box_max_lsb'] for r in records),
                                      high_score_mae_lsb=sum(r['high_score_box_mae_lsb'] * r['high_score_anchors'] for r in records) / max(all_high, 1),
                                      high_score_aligned_iou_percentiles={str(p): float(np.percentile(high_ious, p)) for p in (0, 5, 50, 95, 100)} if high_ious else {}),
                  nms_changes=dict(removed=sum(r['nms_removed'] for r in records), added=sum(r['nms_added'] for r in records)),
                  wrapped_delta=dict(elements=wrapped_elements, distributions=wrapped_distributions,
                                     total_distributions=4 * runner.plan.locations * len(samples)),
                  limitations=[manifest['limitations'], '100 ordered images, not full COCO or statistical significance.',
                               'Host reference only: not 100 RTL inferences. DFL arithmetic is checked separately against retained RTL snapshots.',
                               'Same MAC prefix and TFLite tail for the DFL-only comparison; other AFU operator semantics are not generalized here.',
                               'Cached TFLite/hybrid outputs are hash-checked; every freshly recomputed split control is byte-exact.',
                               'No NPU performance claim; elapsed time includes paired reference evaluation and file IO.'])
    keys = ('AP_50_95', 'AP_50', 'AP_75', 'AP_small', 'AP_medium', 'AP_large', 'AR_max100')
    report['dfl_only_delta_percentage_points'] = {k: 100 * (metrics[KINDS[2]][k] - metrics[KINDS[1]][k]) for k in keys}
    report['total_vs_original_tflite_delta_percentage_points'] = {k: 100 * (metrics[KINDS[2]][k] - metrics[KINDS[0]][k]) for k in keys}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'detection_metrics': {k: {f: v for f, v in m.items() if f != 'per_category'} for k, m in metrics.items()},
                      'dfl_only_delta_percentage_points': report['dfl_only_delta_percentage_points'],
                      'box_difference': report['box_difference'], 'wrapped_delta': report['wrapped_delta']}, indent=2), flush=True)
    del runner


if __name__ == '__main__':
    main()
