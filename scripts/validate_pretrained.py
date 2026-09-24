"""Verify a pinned model hash, then run cached pretrained acceptance."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "MODEL_ACCEPTANCE_MANIFEST.json")
    parser.add_argument("--id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", help="GPU expert device, e.g. CUDA0; omitted selects CPU cache")
    parser.add_argument("--cpu-miss-percent", type=int, default=0, help="-1 adaptive or 1..100 fixed CPU share of GPU cache misses")
    args = parser.parse_args()
    if not -1 <= args.cpu_miss_percent <= 100 or (args.cpu_miss_percent and not args.device):
        parser.error("CPU miss split requires --device and -1..100")
    models = json.loads(args.manifest.read_text(encoding="utf-8"))["models"]
    model = next((m for m in models if m["id"] == args.id), None)
    if model is None:
        parser.error("unknown model id")
    if args.model.stat().st_size != model["size_bytes"]:
        parser.error("model size differs from the pinned artifact")
    model_hash = sha256(args.model)
    if model_hash != model["publisher_sha256"]:
        parser.error("model SHA-256 differs from the pinned artifact")
    executable = args.executable.resolve(strict=True)
    command = [str(executable)] + (["--device", args.device] if args.device else []) + (["--cpu-miss-percent", str(args.cpu_miss_percent)] if args.cpu_miss_percent else []) + ["--model", str(args.model.resolve())]
    log_path = args.output.with_suffix(".log")
    if args.output.exists() or log_path.exists():
        parser.error("choose a new output path to preserve prior acceptance evidence")
    start = time.time()
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    report = {"status": "passed" if result.returncode == 0 else "failed", "scope": f"{args.device or 'CPU'} cache pretrained parity; not P100 acceptance",
              "model": model, "local_sha256": model_hash, "executable_sha256": sha256(executable),
              "platform": platform.platform(), "command": command, "returncode": result.returncode,
              "elapsed_seconds": time.time() - start, "log": str(log_path.resolve())}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
