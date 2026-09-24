"""Build and smoke-test a Windows CUDA development archive, never a certified P100 release."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import zipfile


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--cuda-root", required=True, type=Path)
    parser.add_argument("--vs-redist", required=True, type=Path, help="MSVC redist version directory containing x64")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("this packager is for native Windows")
    root = Path(__file__).resolve().parents[1]
    for name in ("cuda-memcheck.log", "cuda-initcheck.log"):
        content = (root / "reports" / name).read_text(encoding="utf-8", errors="replace")
        summaries = re.findall(r"ERROR SUMMARY: (\d+) errors", content)
        if not summaries or any(int(n) for n in summaries) or "Target application returned an error" in content:
            raise ValueError(f"{name} does not record a clean sanitizer run")
    repo = root / "llama.cpp"
    archive = args.output.resolve()
    stage = archive.with_suffix("")
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    if archive.suffix != ".zip" or any(p.exists() for p in (archive, stage, checksum)):
        parser.error("choose an unused .zip output path")
    stage.mkdir(parents=True)
    for name in ("test-moe-offload", "test-moe-integration", "llama-cli", "llama-server"):
        shutil.copy2(args.bin_dir / (name + ".exe"), stage)
    for name in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"):
        shutil.copy2(args.cuda_root / "bin" / name, stage)
    for directory in ("Microsoft.VC143.CRT", "Microsoft.VC143.OpenMP"):
        candidates = list((args.vs_redist / "x64" / directory).glob("*.dll"))
        if not candidates:
            raise ValueError(f"missing runtime directory {directory}")
        for path in candidates:
            shutil.copy2(path, stage)
    licenses = stage / "licenses"
    licenses.mkdir()
    shutil.copy2(args.cuda_root / "EULA.txt", licenses / "NVIDIA-CUDA-EULA.txt")
    for pattern in ("*LICENSE*", "*COPYING*", "*NOTICE*"):
        for path in repo.rglob(pattern):
            if path.is_file() and ".git" not in path.relative_to(repo).parts:
                target = licenses / path.relative_to(repo)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    for name in ("DEVELOPMENT_PACKAGE.md", "MODEL_ACCEPTANCE_MANIFEST.json", "M2_CUDA_EXECUTOR.md"):
        shutil.copy2(root / name, stage)
    sources = stage / "source-changes"
    sources.mkdir()
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    (sources / "base-revision.txt").write_text(revision + "\n", encoding="utf-8")
    (sources / "working-tree.patch").write_bytes(subprocess.check_output(["git", "-C", str(repo), "diff", "--binary", "HEAD"]))
    untracked = subprocess.check_output(["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "-z"]).decode().split("\0")
    for relative in filter(None, untracked):
        path = repo / relative
        if path.is_file():
            target = sources / "additions" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    shutil.copytree(root / "scripts", sources / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(root / "cmake", sources / "cmake")
    reports = stage / "verification"
    reports.mkdir()
    for name in ("gpu-development-inventory.json", "qwen-pretrained-header.json", "gemma-pretrained-header.json",
                 "cuda-memcheck.log", "cuda-initcheck.log", "cuda-matmul-regressions.log",
                 "gpu-executor-cpu-tests.log", "cuda-build-cpu-regressions.log"):
        shutil.copy2(root / "reports" / name, reports)
    env = dict(os.environ)
    system_root = os.environ["SystemRoot"]
    env["PATH"] = os.pathsep.join((str(stage), str(Path(system_root) / "System32"), system_root))
    env.pop("GGML_CUDA_DISABLE_FUSION", None)
    env.pop("GGML_CUDA_NO_PINNED", None)
    checks = []
    for name, arguments, log in (("test-moe-offload", ["--device", "CUDA0"], "gpu"),
                                  ("test-moe-offload", [], "cpu-executor"),
                                  ("test-moe-integration", [], "model-integration"),
                                  ("llama-cli", ["--help"], "cli-help"),
                                  ("llama-server", ["--help"], "server-help")):
        with (reports / (log + ".log")).open("w", encoding="utf-8") as output:
            command = [str(stage / (name + ".exe")), *arguments]
            result = subprocess.run(command, cwd=stage, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=180)
        checks.append({"executable": name, "arguments": arguments, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"staged {name} failed; evidence retained in {reports}; archive not created")
    (stage / "package-status.json").write_text(json.dumps({"status": "development", "gpu_tested": "RTX 4060 Laptop",
            "p100_tested": False, "linux_tested": False, "pretrained_outputs_verified": False,
            "gpu_model_integration": False, "source_base": revision, "checks": checks}, indent=2) + "\n", encoding="utf-8")
    members = sorted(p for p in stage.rglob("*") if p.is_file())
    (stage / "SHA256SUMS.txt").write_text("".join(f"{digest(p)}  {p.relative_to(stage).as_posix()}\n" for p in members), encoding="utf-8")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(stage).as_posix())
    checksum.write_text(f"{digest(archive)}  {archive.name}\n", encoding="utf-8")
    print(archive)


if __name__ == "__main__":
    main()
