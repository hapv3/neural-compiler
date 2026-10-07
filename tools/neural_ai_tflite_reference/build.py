"""Build a standalone delegate against headers matching the installed TensorFlow."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent


def main():
    import tensorflow as tf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    version = tf.__version__
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"Release TensorFlow required, found {version}")
    build = ROOT / "build"
    includes = build / f"tensorflow-{version}"
    pending = ["tensorflow/lite/c/common.h", "tensorflow/lite/c/builtin_op_data.h",
               "tensorflow/lite/builtin_ops.h",
               "tensorflow/lite/delegates/external/external_delegate_interface.h"]
    hashes = {}
    while pending:
        name = pending.pop()
        if name in hashes:
            continue
        path = includes / name
        if not path.exists():
            url = f"https://raw.githubusercontent.com/tensorflow/tensorflow/v{version}/{name}"
            print(f"Fetch {url}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        data = path.read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        pending.extend(re.findall(r'^#include "(tensorflow/[^\"]+)"', data.decode(), re.MULTILINE))
    output = build / "libneural_ai_reference.so"
    command = [os.environ.get("CXX", "g++"), "-std=c++17", "-O3", "-Wall", "-Wextra", "-Werror",
               "-fPIC", "-shared", "-fvisibility=hidden", "-Wl,--no-undefined",
               "-I", str(includes), str(ROOT / "delegate.cpp"), "-o", str(output)]
    if args.sanitize:
        command[2:2] = ["-fsanitize=undefined", "-fno-sanitize-recover=all"]
    subprocess.run(command, check=True)
    manifest = {"tensorflow_version": version, "headers_sha256": hashes, "command": command,
                "source_sha256": hashlib.sha256((ROOT / "delegate.cpp").read_bytes()).hexdigest(),
                "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
