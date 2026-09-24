"""Bounded, sequential desktop-engine tuning; preserves saved conversation/preferences."""
import argparse
import json
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import psutil

BASE = "http://127.0.0.1:18432"
OUT = Path("reports/system-tuning-20260913")


def request(path, body=None, base=BASE, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def ready():
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        state = request("/api/state")["engine"]
        if state["state"] == "ready":
            return state
        if state["state"] == "error":
            raise RuntimeError(state["detail"])
        time.sleep(0.5)
    raise TimeoutError("Model load exceeded 120 seconds")


def load(patch):
    if request("/api/state")["engine"]["generating"]:
        raise RuntimeError("User generation is active; tuning stopped")
    request("/api/engine/unload", {})
    request("/api/settings", patch)
    request("/api/engine/load", {})
    return ready()


def monitor(stop, rows):
    while not stop.is_set():
        try:
            raw = subprocess.check_output([
                "nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu,power.draw,clocks.sm",
                "--format=csv,noheader,nounits"], creationflags=subprocess.CREATE_NO_WINDOW, timeout=3).decode()
            gpu = [float(x.strip()) for x in raw.splitlines()[0].split(",")]
            ram = psutil.virtual_memory()
            rows.append({"time": time.time(), "gpu_used_mib": gpu[0], "gpu_free_mib": gpu[1],
                         "gpu_util": gpu[2], "watts": gpu[3], "clock_mhz": gpu[4],
                         "ram_available_mib": ram.available / 2**20, "cpu_percent": psutil.cpu_percent()})
        except Exception:
            pass
        stop.wait(1)


def bench(engine, label):
    base = f"http://127.0.0.1:{engine['port']}"
    prompts = ["Explain how a GPU works.",
               "Summarize the following passage in plain language, then identify the main bottleneck.\n" +
               ("A GPU executes many operations at once. Its processor needs a steady supply of data. "
                "Weights that do not fit in video memory must be transferred or processed elsewhere. "
                "A larger cache can avoid repeated transfers, but it must leave enough space for attention "
                "and temporary calculations. CPU work is useful only when it overlaps effectively. " * 4)]
    samples, telemetry = [], []
    stop = threading.Event()
    worker = threading.Thread(target=monitor, args=(stop, telemetry), daemon=True)
    worker.start()
    try:
        for index, prompt in enumerate([prompts[0], prompts[0], prompts[1]]):
            body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
                    "max_tokens": 64, "stream": False, "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False}, "reasoning_budget": 0}
            start = time.monotonic()
            result = request("/v1/chat/completions", body, base, timeout=150)
            sample = {"kind": "warmup" if index == 0 else "short" if index == 1 else "long",
                      "wall_seconds": time.monotonic() - start, "response": result}
            samples.append(sample)
            print(label, sample["kind"], result.get("timings", {}), flush=True)
    finally:
        stop.set()
        worker.join(5)
    return {"samples": samples, "telemetry": telemetry}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="baseline,cache3712,batch256,flashq8,hybrid25")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    original = request("/api/state")["settings"]
    (OUT / "settings-before.json").write_text(json.dumps(original, indent=2))
    common = {k: original[k] for k in ["gpu_cache", "ubatch", "batch", "flash", "cache_k", "cache_v", "miss", "auto_settings"]}
    candidates = {"baseline": {}, "cache3712": {"gpu_cache": "3712"},
                  "batch256": {"gpu_cache": "3712", "ubatch": "256"},
                  "flashq8": {"gpu_cache": "3712", "flash": "on", "cache_k": "q8_0", "cache_v": "q8_0"},
                  "hybrid25": {"gpu_cache": "3712", "miss": "25"}}
    summary = []
    try:
        for label in args.candidates.split(","):
            patch = common | candidates[label] | {"auto_settings": "0"}
            print("Loading", label, patch, flush=True)
            row = {"label": label, "settings": patch}
            try:
                engine = load(patch)
                row.update(bench(engine, label))
            except Exception as error:
                row["error"] = str(error)
                print(label, "FAILED", str(error), flush=True)
            (OUT / f"{label}.json").write_text(json.dumps(row, indent=2))
            summary.append(row)
            (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        if not request("/api/state")["engine"]["generating"]:
            load(common)
            print("Original settings restored", flush=True)


if __name__ == "__main__":
    main()
