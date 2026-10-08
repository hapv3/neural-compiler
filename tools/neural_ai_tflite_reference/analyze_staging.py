"""Reproduce a saved pre-Conv TCDM image from pinned DMA/LUT commands.

This is a bounded functional diagnostic, not a cycle/queue simulator. Async DMA
submissions are applied eagerly. Local copies are limited to one <=32-byte beat
per repetition, matching the saved case without guessing vector overlap rules.
"""

import argparse
import json
from pathlib import Path
import struct
import sys


def replay_staging(commands, memory, constants, first, end):
    memory = bytearray(memory)
    transfers = []
    for number in range(first + 1, end + 1):
        command = commands[number - 1]
        kind = struct.unpack_from("<H", command)[0]
        if kind in (5, 27):  # Requant register load or DMA wait; no tensor write.
            continue
        if kind not in (2, 3, 4, 12, 24, 25, 26):
            raise ValueError(f"Unsupported command {number}: {kind}")
        sr, si, src = struct.unpack_from("<HHI", command, 16)
        dr, di, dst = struct.unpack_from("<HHI", command, 24)
        if si or di or dr != 6 or sr not in (1, 6):
            raise ValueError("Only model constants/TCDM sources and TCDM destinations are supported")
        source = constants if sr == 1 else memory
        if kind == 12:
            lr, li, offset = struct.unpack_from("<HHI", command, 32)
            length = struct.unpack_from("<I", command, 40)[0]
            if (lr, li) != (1, 0) or offset + 256 > len(constants):
                raise ValueError("Invalid constant LUT reference")
            if src + length > len(source) or dst + length > len(memory):
                raise ValueError("LUT tensor exceeds memory")
            memory[dst:dst + length] = bytes(constants[offset + value]
                                            for value in bytes(source[src:src + length]))
            continue
        length = struct.unpack_from("<I", command, 32)[0]
        ss = ds = s3 = d3 = 0
        r2 = r3 = 1
        if kind in (3, 4, 25, 26):
            ss, ds, r2 = struct.unpack_from("<3I", command, 36)
        if kind in (4, 26):
            s3, d3, r3 = struct.unpack_from("<3I", command, 48)
        if length == 0 or r2 == 0 or r3 == 0 or (sr == 6 and length > 32):
            raise ValueError("Unsupported local copy length or empty transfer")
        for z in range(r3):
            for y in range(r2):
                a, d = src + z * s3 + y * ss, dst + z * d3 + y * ds
                if a + length > len(source) or d + length > len(memory):
                    raise ValueError("DMA transfer exceeds memory")
                memory[d:d + length] = source[a:a + length]
        source_end = src + (r3 - 1) * s3 + (r2 - 1) * ss + length
        destination_end = dst + (r3 - 1) * d3 + (r2 - 1) * ds + length
        transfers.append({"command": number, "source_region": sr, "source": src,
                          "source_end": source_end, "destination": dst,
                          "destination_end": destination_end,
                          "bounding_ranges_overlap": sr == 6 and
                          max(src, dst) < min(source_end, destination_end)})
    return bytes(memory), transfers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-ai-root", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--before-command", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report exists; choose a fresh path")
    sys.path.insert(0, str(args.neural_ai_root / "sw/test/compiler_runtime"))
    from yolo_reference_replay import artifact, load_case, runtime

    case, model, memory, _, _ = load_case(args.case)
    if not case["first_command"] < args.before_command <= case["end_command"]:
        parser.error("Requested pre-command image is outside the case")
    commands, _ = runtime._logical_command_records(model)
    count, table = struct.unpack_from("<2I", model, 20)
    constants = None
    for index in range(count):
        kind, _, offset, size = struct.unpack_from("<4I", model, table + index * 32)
        if kind == 2:
            constants = model[offset:offset + size]
    if constants is None:
        raise ValueError("Missing constants section")
    predicted, transfers = replay_staging(commands, memory, constants,
                                         case["first_command"], args.before_command - 1)
    observed_path = args.case.parent / f"before-command-{args.before_command}-tcdm.bin"
    observed = observed_path.read_bytes()
    if len(observed) != len(predicted):
        raise ValueError("Observed TCDM size mismatch")
    compared = 0x7F000  # Exclude the firmware command/scratch window.
    mismatches = [i for i in range(compared) if predicted[i] != observed[i]]
    report = {"model": case["model"], "snapshot": case["snapshot"],
              "observed": artifact(observed_path), "first_command": case["first_command"] + 1,
              "last_command": args.before_command - 1, "compared_bytes": compared,
              "mismatches": len(mismatches), "first_mismatch": next(iter(mismatches), None),
              "scope": "Functional memory replay only; does not validate asynchronous timing or Conv outputs",
              "transfers": transfers}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Offline ABI replay vs saved TCDM: {len(mismatches)}/{compared} byte mismatches; {args.report}")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
