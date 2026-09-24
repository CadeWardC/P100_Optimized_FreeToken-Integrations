# Hybrid scheduler benchmark

Measured on 2026-09-09. GPU-only expert execution was fastest for this short, repeated Gemma workload. None of the tested CPU miss splits improved throughput. Keep `--moe-cpu-miss-percent 0` for this configuration; the adaptive policy needs further calibration before it can be recommended for performance.

| CPU share of cache misses | Median decode tokens/s | Measured range | Change vs GPU-only | Median prompt tokens/s | Median request seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0% (GPU-only) | 2.908 | 2.565-2.953 | baseline | 3.675 | 12.578 |
| 50% | 2.165 | 2.155-2.223 | -25.6% | 2.564 | 17.079 |
| 25% | 1.931 | 1.881-2.191 | -33.6% | 2.478 | 18.890 |
| Adaptive (-1) | 1.747 | 1.651-1.868 | -39.9% | 1.378 | 22.828 |
| 100% | 1.471 | 1.402-1.516 | -49.4% | 1.735 | 25.578 |

All 20 requests (five warm-ups and 15 measured requests) produced identical 32-token sequences. Every request processed all seven prompt tokens. This checks output consistency for this prompt, not full-vocabulary numerical parity. Existing numerical acceptance remains documented in [GPU_HYBRID_SCHEDULER.md](GPU_HYBRID_SCHEDULER.md).

## Configuration and method

- Windows 11 host (Python reports Windows build 26200), AMD Ryzen 7 7840HS, eight physical cores and 16 logical processors.
- NVIDIA RTX 4060 Laptop GPU, 8188 MiB VRAM, driver 610.62. This is not a P100 benchmark.
- Pinned Gemma 4 26B-A4B QAT Q4_0 model, SHA-256 checked before execution against `MODEL_ACCEPTANCE_MANIFEST.json`.
- Existing CUDA development server, SHA-256 `6995391ff2236e9b0a5ab2d049ef47c3d2c725a53ba9a5601a91d53763d36891`.
- CPU attention/shared layers, eight CPU threads, one sequence, context 256, batch 128, microbatch 8, flash attention disabled, mmap enabled, repacking disabled.
- GPU expert cache 1024 MiB, staging 1 MiB, compute budget 64 MiB, reserve 512 MiB; device CUDA0.
- Prompt: `Explain how a GPU works.`; 32 generated tokens, greedy sampling, seed 1234, EOS ignored.
- Each policy used a fresh server process. One full request warmed the model and expert cache, followed by three measured requests. Prompt KV reuse was disabled; expert cache and adaptive state persisted within each process.
- Policy order was shuffled with seed 1234: 25%, 50%, adaptive, GPU-only, 100%. Modes ran sequentially.
- Decode and prompt rates are the server's reported rates. Request wall time includes HTTP overhead but excludes model startup. Warm-up is excluded from the table.

Sampled peak process RSS ranged from 7.40 to 7.68 GiB. Sampled global GPU memory peaked at 1107-1111 MiB across policies; that includes other GPU users and is not process-specific allocation accounting. Monitoring ran once per second with `psutil` and `nvidia-smi`, including startup and warm-up.

## Limits and interpretation

This is a small development benchmark with one seven-token prompt and short generation. It does not characterize long prefill, varied prompts, cache-budget or thread-count sweeps, concurrent workloads, Linux, Qwen, or P100 hardware. Some modes improved across repetitions, so one warm-up did not establish a fully stable steady state. Sequential policy blocks and ordinary laptop clock/power variation can influence results; the ranges are observations, not confidence intervals.

The results show a throughput regression for each tested hybrid policy on this workload. They do not isolate its cause. The server response exposes end-to-end inference timings, but this benchmark does not capture scheduler branch counters, realized CPU assignment shares, or overlap. CPU contention, transfer savings and adaptive policy choices require separate profiling before changing the scheduler.

## Reproduce

The harness requires Python and `psutil`. Use a new output directory on every invocation to preserve evidence:

```powershell
python scripts/benchmark_hybrid.py `
  --model models/gemma-4-26B_q4_0-it.gguf `
  --server build/ft-windows-cuda-dev/bin/llama-server.exe `
  --output reports/hybrid-benchmark-repeat `
  --tokens 32 --repeats 3 --cache-mib 1024 --threads 8
```

[Raw results](reports/hybrid-benchmark-20260909/results.json) contain all commands, responses, token IDs, measured timings, memory samples, hashes and the sorted summary. The same directory contains one server log per mode. The runner is [scripts/benchmark_hybrid.py](scripts/benchmark_hybrid.py).
