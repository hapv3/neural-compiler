"""Re-analyze saved outputs without invoking a runtime or simulator."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    groups = {}
    changed_top1 = []
    output_quantization = None
    for filename in sorted((args.run / "outputs").glob("*.npz")):
        with np.load(filename) as outputs:
            reference = outputs["tflite_reference"].astype(np.int64)
            for kind in ("neural_ai_reference", "tflite_optimized"):
                actual = outputs[kind].astype(np.int64)
                difference = np.abs(actual - reference)
                if reference.ndim == 3:
                    # Output scale is model-specific and recorded in the benchmark adapter.
                    # Derive score thresholds from original .tflite output quantization.
                    from runtime import make_interpreter
                    if output_quantization is None:
                        runtime = make_interpreter(report["model"], "tflite_reference")
                        output_quantization = runtime.get_output_details()[0]["quantization"]
                        del runtime
                    output_scale, zp = output_quantization
                    confident = np.maximum(actual[:, 4:].max(axis=1), reference[:, 4:].max(axis=1)) >= (0.25 / output_scale + zp)
                    selected = {"coordinates_all": difference[:, :4], "scores_all": difference[:, 4:],
                                "coordinates_score_ge_025_either_runtime": difference[:, :4].transpose(0, 2, 1)[confident]}
                else:
                    selected = {"logits": difference}
                    if actual.argmax() != reference.argmax():
                        changed_top1.append({"runtime": kind, "image_id": int(filename.stem),
                                             "reference_class": int(reference.argmax()), "actual_class": int(actual.argmax())})
                for name, values in selected.items():
                    key = f"{kind}/{name}"
                    totals = groups.setdefault(key, {"elements": 0, "mismatches": 0, "sum_abs_lsb": 0, "max_abs_lsb": 0})
                    totals["elements"] += int(values.size)
                    totals["mismatches"] += int(np.count_nonzero(values))
                    totals["sum_abs_lsb"] += int(values.sum())
                    totals["max_abs_lsb"] = max(totals["max_abs_lsb"], int(values.max(initial=0)))
    for totals in groups.values():
        count = totals["elements"]
        totals["mae_lsb"] = totals["sum_abs_lsb"] / count if count else None
        totals["exact_percent"] = 100 * (1 - totals["mismatches"] / count) if count else None
    result = {"groups": groups, "changed_top1": changed_top1}
    (args.run / "output_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
