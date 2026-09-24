"""Smoke-test the local mixed-quant GGUF through the rebuilt GPU server."""
import json
import subprocess
import time
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
model = root.parent / "models/Qwen3.6-35B-A3B-I-COMPACT/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"
command = [str(root / "build/ft-windows-cuda-dev/bin/llama-server.exe"),
           "-m", str(model), "--host", "127.0.0.1", "--port", "18973",
           "-ngl", "99", "-c", "1024", "-b", "64", "-ub", "32", "-np", "1",
           "--moe-gpu-cache-mib", "512", "--moe-compute-mib", "128",
           "--no-context-shift"]
with (root / "reports/mixed-quant-server.log").open("w") as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        deadline = time.monotonic() + 240
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited with {process.returncode}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:18973/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Server did not become ready")
            time.sleep(1)
        payload = {"prompt": "The capital of France is", "n_predict": 16,
                   "temperature": 0, "seed": 123, "cache_prompt": False}
        request = urllib.request.Request("http://127.0.0.1:18973/completion",
                    json.dumps(payload).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        (root / "reports/mixed-quant-server-result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"content": result.get("content"), "timings": result.get("timings")}))
        if "Paris" not in result.get("content", ""):
            raise RuntimeError("Unexpected smoke-test completion")
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
