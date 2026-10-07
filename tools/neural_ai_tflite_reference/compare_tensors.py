"""Per-operator tensor differences for one image (includes propagated differences)."""

import argparse
import csv
from pathlib import Path

from benchmark import difference, prepare_image
from runtime import coverage, make_interpreter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--task", choices=["yolo", "mobilenet"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = make_interpreter(args.model, "tflite_reference", preserve=True)
    neural = make_interpreter(args.model, preserve=True)
    values, _ = prepare_image(args.image, args.task, reference.get_input_details()[0])
    for runner in (reference, neural):
        runner.set_tensor(runner.get_input_details()[0]["index"], values)
        runner.invoke()
    delegated = {node["index"]: node["neural_ai_kernel"] for node in coverage(neural)["nodes"]}
    details = {d["index"]: d for d in reference.get_tensor_details()}
    rows = []
    for node in reference._get_ops_details():
        for tensor in node["outputs"]:
            if tensor < 0:
                continue
            rows.append({"node": int(node["index"]), "operator": node["op_name"], "tensor": int(tensor),
                         "name": details[tensor]["name"], "neural_ai_kernel": delegated[node["index"]],
                         **difference(neural.get_tensor(tensor), reference.get_tensor(tensor))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Saved {len(rows)} tensor comparisons: {args.output}")


if __name__ == "__main__":
    main()
