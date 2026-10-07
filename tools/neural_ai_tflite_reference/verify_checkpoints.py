"""Compare the hybrid reference against saved RTL stem bytes; never run simulation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark import difference
from runtime import make_interpreter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = json.loads((args.checkpoints / "byte_exact.json").read_text())
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != saved["source_sha256"]:
        raise ValueError("Snapshot belongs to another TFLite source model")
    reports = {}
    failed = False
    for kind in ("tflite_reference", "neural_ai_reference"):
        runtime = make_interpreter(args.model, kind, preserve=True)
        detail = runtime.get_input_details()[0]
        values = np.frombuffer((args.checkpoints / "input.bin").read_bytes(), detail["dtype"]).reshape(detail["shape"])
        runtime.set_tensor(detail["index"], values); runtime.invoke()
        rows = []
        for checkpoint in saved["checkpoints"]:
            command = checkpoint["command"]
            raw = (args.checkpoints / f"command-{command}-actual.bin").read_bytes()
            if hashlib.sha256(raw).hexdigest() != checkpoint["actual_sha256"]:
                raise ValueError(f"Snapshot bytes changed for command {command}")
            actual = np.frombuffer(raw, np.int8).reshape(checkpoint["rows"], checkpoint["channels"])
            tensor = runtime.get_tensor(checkpoint["tensor"]).reshape(-1, checkpoint["channels"])
            offset = checkpoint["reference_first_row"]
            expected = tensor[offset:offset + checkpoint["rows"]]
            stats = difference(actual, expected)
            rows.append({"command": command, "tensor": checkpoint["tensor"], **stats})
            if kind == "neural_ai_reference" and stats["mismatches"]:
                failed = True
        reports[kind] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))
    if failed:
        raise SystemExit("Saved RTL checkpoints differ from the hybrid reference; see report")


if __name__ == "__main__":
    main()
