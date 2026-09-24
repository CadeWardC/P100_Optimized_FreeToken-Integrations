"""M0 inventory and local, single-file GGUF benchmark recording (standard library only)."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(command, cwd=ROOT):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=60)
        return {"command": list(map(str, command)), "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": list(map(str, command)), "returncode": None, "error": str(error)}


def inventory():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    checks = []
    for entry in [manifest["combined_patch"], *manifest["patch_source"]["patches"]]:
        path = ROOT / entry["file"]
        actual = sha256(path)
        normalized = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        checks.append({"file": entry["file"], "sha256": actual,
                       "matches_manifest": actual == entry["sha256"] or
                       normalized == entry.get("sha256_lf")})
    repositories = {}
    for key in ("baseline", "patch_source", "strategy_reference"):
        entry = manifest[key]
        repo = ROOT / entry["directory"]
        git = ["git", "-c", "safe.directory=" + repo.as_posix(), "-C", str(repo)]
        head = capture(git + ["rev-parse", "HEAD"])
        repositories[key] = {
            "head": head, "matches_pin": head.get("stdout", "").strip() == entry["commit"],
            "status": capture(git + ["status", "--short"]),
            "diff": capture(git + ["diff", "--binary", "HEAD"]),
        }
        untracked = capture(git + ["ls-files", "--others", "--exclude-standard", "-z"])
        repositories[key]["untracked_sha256"] = {
            name: sha256(repo / name) for name in untracked.get("stdout", "").split("\0") if name
        }
    host = {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count()}
    if sys.platform == "win32":
        host["hardware"] = capture(["powershell", "-NoProfile", "-Command",
            "@{computer=(Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory); "
            "cpu=(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors)} "
            "| ConvertTo-Json -Depth 4"])
    elif sys.platform.startswith("linux"):
        host["hardware"] = capture(["lscpu"])
        host["memory"] = Path("/proc/meminfo").read_text()
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "validation_scope": "development host; does not certify P100 support",
            "host": host, "repositories": repositories, "artifact_checks": checks,
            "cmake": capture(["cmake", "--version"]),
            "gpu": capture(["nvidia-smi", "-q"]),
            "default_nvcc": capture(["nvcc", "--version"])}


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def bench_command(args):
    return [str(args.binary.resolve()), "-m", str(args.model.resolve()),
            "-p", str(args.prompt), "-n", str(args.generate), "-t", str(args.threads),
            "-b", "512", "-ub", "128", "-ngl", str(args.gpu_layers),
            "-fa", "off", "-r", str(args.repetitions), "-o", "json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    inv = commands.add_parser("inventory")
    inv.add_argument("--output", type=Path, required=True)
    bench = commands.add_parser("bench")
    bench.add_argument("--output", type=Path, required=True, help="new directory; never overwritten")
    bench.add_argument("--binary", type=Path, required=True)
    bench.add_argument("--model", type=Path, required=True)
    bench.add_argument("--arm", choices=["unpatched", "p100-baseline", "freetoken"], required=True)
    bench.add_argument("--threads", type=positive, required=True)
    bench.add_argument("--gpu-layers", type=int, default=0)
    bench.add_argument("--prompt", type=positive, default=512)
    bench.add_argument("--generate", type=positive, default=256)
    bench.add_argument("--repetitions", type=positive, default=5)
    bench.add_argument("--cmake-cache", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "inventory":
        report = inventory()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        return 0 if (all(c["matches_manifest"] for c in report["artifact_checks"]) and
                     all(r["matches_pin"] for r in report["repositories"].values())) else 1
    for path in (args.binary, args.model, args.cmake_cache):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    if re.search(r"-\d{5}-of-\d{5}\.gguf$", args.model.name, re.IGNORECASE):
        parser.error("split GGUF is not supported by this initial recorder; use a single-file model")
    with args.model.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            parser.error("model is not a GGUF file")
    if args.repetitions < 5 or args.generate < 256 or args.gpu_layers < 0:
        parser.error("require >=5 repetitions, >=256 generated tokens, and >=0 GPU layers")
    args.output.mkdir(parents=True, exist_ok=False)
    command = bench_command(args)
    report = inventory()
    report.update({"arm_label": args.arm, "arm_label_is_user_supplied": True,
                   "command": command, "cwd": str(ROOT),
                   "model": {"path": str(args.model.resolve()), "bytes": args.model.stat().st_size,
                             "sha256": sha256(args.model)},
                   "binary_sha256": sha256(args.binary),
                   "cmake_cache": args.cmake_cache.read_text(encoding="utf-8"),
                   "status": "started"})
    metadata = args.output / "metadata.json"
    metadata.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output / "stdout.json").open("wb") as stdout, (args.output / "stderr.log").open("wb") as stderr:
        try:
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
            report["returncode"] = result.returncode
            report["status"] = "completed" if result.returncode == 0 else "failed"
        except OSError as error:
            report.update(status="failed", error=str(error), returncode=1)
    if report["returncode"] == 0:
        try:
            rows = json.loads((args.output / "stdout.json").read_text(encoding="utf-8"))
            if not isinstance(rows, list) or not rows:
                raise ValueError("expected nonempty benchmark JSON array")
            for row in rows:
                for field in ("samples_ns", "samples_ts"):
                    samples = row.get(field, []) if isinstance(row, dict) else []
                    if (not isinstance(samples, list) or len(samples) != args.repetitions or
                            any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in samples)):
                        raise ValueError(f"invalid or missing {field} measurements")
        except (ValueError, OSError) as error:
            report.update(status="failed", error=str(error), returncode=1)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    metadata.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report["returncode"]


if __name__ == "__main__":
    sys.exit(main())
