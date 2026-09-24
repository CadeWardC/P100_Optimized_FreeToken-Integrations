"""Run matched local CLI/server acceptance and record RAM, VRAM and token output.

Requires psutil. Uses a pinned, locally SHA-256 verified GGUF. Results describe
this host only; small-cache pressure is not an operating-system OOM test.
"""
import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

import psutil
from validate_pretrained import sha256


class GPUAllocation:
    """Own an unrelated allocation to exercise coexistence with other GPU users."""
    def __init__(self, mib):
        self.pointer = ctypes.c_void_p()
        if os.name == "nt":
            root = Path(os.environ.get("FT_CUDA_ROOT", os.environ.get("CUDA_PATH",
                        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")))
            library = next((root / "bin").glob("cudart64_*.dll"))
            self.dll_dir = os.add_dll_directory(str(root / "bin"))
        else:
            library = ctypes.util.find_library("cudart")
        self.runtime = ctypes.CDLL(str(library))
        self.runtime.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.runtime.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.runtime.cudaFree.argtypes = [ctypes.c_void_p]
        self.runtime.cudaSetDevice.argtypes = [ctypes.c_int]
        self.check(self.runtime.cudaSetDevice(0))
        self.check(self.runtime.cudaMalloc(ctypes.byref(self.pointer), mib * 1024 * 1024))
        try:
            self.check(self.runtime.cudaMemset(self.pointer, 0, mib * 1024 * 1024))
            self.check(self.runtime.cudaDeviceSynchronize())
        except Exception:
            self.close()
            raise

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"CUDA pressure allocation failed: {code}")

    def close(self):
        if self.pointer.value:
            self.check(self.runtime.cudaFree(self.pointer))
            self.pointer.value = None


def request(port, route, body=None, timeout=1800):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


class Monitor:
    def __init__(self, pid):
        self.process = psutil.Process(pid)
        self.done = threading.Event()
        self.samples = []
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def run(self):
        while not self.done.is_set():
            try:
                mem = self.process.memory_info()
                sample = {"time": time.time(), "rss": mem.rss, "vms": mem.vms,
                          "system_available": psutil.virtual_memory().available}
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                     timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if gpu.returncode == 0:
                    sample["gpu"] = gpu.stdout.strip()
                self.samples.append(sample)
            except (psutil.Error, OSError, subprocess.TimeoutExpired):
                pass
            self.done.wait(1)

    def stop(self):
        self.done.set()
        self.worker.join(timeout=6)
        return {"peak_rss": max((s["rss"] for s in self.samples), default=0),
                "minimum_system_available": min((s["system_available"] for s in self.samples), default=0),
                "samples": self.samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id", default="gemma26b-qat-q4")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cache-mib", type=int, default=1024)
    parser.add_argument("--pressure-cache-mib", type=int, default=16)
    parser.add_argument("--pressure-vram-mib", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-cli", action="store_true")
    args = parser.parse_args()
    if min(args.tokens, args.repeats, args.cache_mib, args.pressure_cache_mib, args.threads) <= 0 or args.pressure_vram_mib < 0:
        parser.error("counts and budgets must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((Path(__file__).resolve().parents[1] / "MODEL_ACCEPTANCE_MANIFEST.json").read_text())
    model = next(m for m in manifest["models"] if m["id"] == args.id)
    if args.model.stat().st_size != model["size_bytes"] or sha256(args.model) != model["publisher_sha256"]:
        parser.error("model does not match pinned size/SHA-256")
    suffix = ".exe" if os.name == "nt" else ""
    common = ["-m", str(args.model.resolve()), "-ngl", "0", "--no-op-offload", "--no-kv-offload",
              "-np", "1", "-c", str(max(256, args.tokens + 128)), "-b", "128", "-ub", "8",
              "-t", str(args.threads), "-tb", str(args.threads), "-fa", "off", "--load-mode", "mmap",
              "--no-warmup", "--no-repack", "--fit", "off"]
    prompt = "The capital of France is"
    modes = [("off", []), ("gpu", ["--moe-gpu-cache-mib", str(args.cache_mib)]),
             ("pressure", ["--moe-gpu-cache-mib", str(args.pressure_cache_mib)])]
    report = {"model": model, "settings": vars(args).copy(), "runs": [],
              "scope": "CPU attention/shared layers; GPU experts; global VRAM includes other processes"}
    report["settings"] = {k: str(v) if isinstance(v, Path) else v for k, v in report["settings"].items()}
    report["binaries"] = {name: sha256(args.bin / (name + suffix)) for name in ("llama-cli", "llama-server")}
    report["system_ram"] = psutil.virtual_memory().total
    def save():
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    pressure = None
    try:
        for name, flags in modes:
            if name == "pressure" and args.pressure_vram_mib:
                pressure = GPUAllocation(args.pressure_vram_mib)
            if flags:
                flags += ["--moe-device", "CUDA0", "--moe-staging-mib", "1", "--moe-compute-mib", "64",
                          "--moe-gpu-reserve-mib", "512"]
            if not args.skip_cli:
                command = [str((args.bin / ("llama-cli" + suffix)).resolve())] + common + flags + [
                    "--single-turn", "--no-conversation", "-p", prompt, "-n", str(args.tokens),
                    "--seed", "1234", "--temp", "0", "--ignore-eos"]
                with (args.output / f"{name}-cli.log").open("w", encoding="utf-8") as log:
                    start = time.monotonic()
                    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    monitor = Monitor(proc.pid)
                    try:
                        code = proc.wait(timeout=3600)
                    finally:
                        if proc.poll() is None:
                            proc.terminate()
                            proc.wait(timeout=30)
                        memory = monitor.stop()
                report["runs"].append({"mode": name, "surface": "cli", "command": command,
                                       "returncode": code, "seconds": time.monotonic() - start, "memory": memory})
                save()
                if code:
                    raise RuntimeError(f"{name} CLI failed: {code}")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            command = [str((args.bin / ("llama-server" + suffix)).resolve())] + common + flags + [
                "--host", "127.0.0.1", "--port", str(port), "--metrics"]
            run = {"mode": name, "surface": "server", "command": command, "requests": []}
            report["runs"].append(run)
            with (args.output / f"{name}-server.log").open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                monitor = Monitor(proc.pid)
                try:
                    deadline = time.monotonic() + 900
                    while True:
                        if proc.poll() is not None:
                            raise RuntimeError(f"{name} server exited during load: {proc.returncode}")
                        try:
                            health = request(port, "/health", timeout=2)
                            if health.get("status") == "ok":
                                break
                        except (urllib.error.URLError, TimeoutError):
                            pass
                        if time.monotonic() > deadline:
                            raise TimeoutError("server readiness")
                        time.sleep(1)
                    for index in range(args.repeats):
                        response = request(port, "/completion", {"prompt": prompt, "n_predict": args.tokens,
                            "seed": 1234, "temperature": 0, "cache_prompt": False, "return_tokens": True,
                            "ignore_eos": True, "stream": False})
                        if len(response.get("tokens", [])) != args.tokens:
                            raise RuntimeError("server did not produce requested token count")
                        run["requests"].append(response)
                        save()
                        print(f"{name} request {index + 1}: {response.get('timings')}", flush=True)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                    proc.wait(timeout=30)
                    run["memory"] = monitor.stop()
                    save()
        servers = [r for r in report["runs"] if r["surface"] == "server"]
        baseline = servers[0]["requests"][0]["tokens"]
        report["identical_tokens"] = all(r["tokens"] == baseline for s in servers for r in s["requests"])
        report["status"] = "passed" if report["identical_tokens"] else "token-divergence"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        if pressure:
            pressure.close()
        save()
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
