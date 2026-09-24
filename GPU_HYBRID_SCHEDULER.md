# GPU-resident activation and hybrid expert scheduling

Scheduling and overlap below describe the earlier implementation. See [FreeToken runtime gap work](FREETOKEN_GAP_CLOSURE.md) for the bandwidth-calibrated policy, GPU transfer overlap and new controls.

Implemented on 2026-09-09 in the Windows CUDA development build. This supersedes the CPU GeGLU and serialized CPU/GPU status in GPU_CORRECTNESS_FIX.md. GPU cache size is unchanged.

## Execution

Gemma GeGLU now stays on CUDA between gate/up and down projections. A persistent 256 KiB F32 table contains the CPU implementation's half-indexed GELU values. The kernel preserves the original-input cutoffs, half rounding, signed zero and separate F32 multiplication. The table is uploaded once per executor and charged against the existing GPU compute budget together with the graph arena. Ordinary CUDA GeGLU remains unchanged; the eager graph supplies the compatibility table explicitly.

The optional hybrid scheduler partitions unique cache misses between a separate CPU worker and the existing GPU executor. Resident experts stay on GPU. Every assignment for a given expert follows the same branch, including repeated routes across tokens. The CPU worker starts before GPU weight loading, so CPU work overlaps GPU transfers and computation. Each branch owns its backend; only the scheduler touches the GPU cache and application abort callback. Failure or cancellation joins the worker before returning. Final contributions are merged in original routing order, with original-ID output scales.

CPU misses use the existing reference executor with a one-expert weight tile. They are not inserted into the GPU cache. CPU graph allocation and host copying still have costs. GPU weight transfer and GPU compute remain sequential within that branch: this is CPU/GPU concurrency, not double-buffered DMA/compute overlap.

## Controls

Add one of these flags to the existing GPU-cache invocation:

- `--moe-cpu-miss-percent 0`: default; GPU-only expert execution.
- `--moe-cpu-miss-percent 50`: send approximately half of unique cache misses to CPU, rounded to a whole expert.
- `--moe-cpu-miss-percent 100`: send all misses to CPU; resident experts still use GPU.
- `--moe-cpu-miss-percent -1`: initial adaptive policy using moving-average elapsed branch cost per expert. It starts at an even split and samples both branches when there are multiple misses.

The adaptive policy is experimental. It does not yet distinguish cache-hit cost from miss cost, assignment frequency, decode from prefill, or changing thread counts. Fixed splits remain available for comparison. No performance-optimality claim is made.

Example:

```powershell
./build/ft-windows-cuda-dev/bin/llama-cli.exe -m models/gemma-4-26B_q4_0-it.gguf `
  -ngl 0 --no-op-offload --no-kv-offload -np 1 `
  --moe-gpu-cache-mib 1024 --moe-device CUDA0 --moe-cpu-miss-percent -1 `
  -c 256 -b 8 -ub 8 -t 8 -n 64 -p "Explain how a GPU works."
```

The same flags are available to the server. Existing single-sequence and CPU attention/shared-layer restrictions remain.

## Instrumentation

`llama_moe_memory()` and `llama_perf_context_print()` expose cumulative successful-executor timings in host microseconds: weight load, input upload, complete graph compute, output readback, merge and total. In hybrid mode, the stage counters describe the GPU branch; separate CPU/GPU branch totals and assignment counts describe the split. Total includes dispatch, waiting and the final merge. The current CLI prints its own short throughput summary; library consumers can read these detailed counters or call the performance-print API. These are host wall times including completion waits, not CUDA-event kernel timings or total decode latency.

`overlap_us` is a conservative branch-overlap estimate: `max(0, CPU elapsed + GPU elapsed - elapsed through both completions)`. Setup and joining overhead reduce this estimate. It proves concurrent branch lifetimes, not continuous simultaneous GPU kernel and CPU arithmetic activity. Input/output byte counters cover the executor boundary; the activation table has its own retained-allocation counter. Hybrid copied-weight bytes include CPU tile copies, while cache-copied bytes count only GPU cache transfers. Summed branch memory peaks are conservative; hash-container bookkeeping and driver allocations remain outside accounting. The public C structs changed; rebuild callers with the new headers.

## Verification

- `reports/hybrid-build-final.log`: Windows CUDA 12.6 build targeting SM60 and SM89.
- `reports/hybrid-executor.log`: exhaustive GeGLU half inputs, half-rounding midpoints and neighbors, cutoff neighbors, signed zero, infinities and NaNs; existing F32/F16/Q4_0/Q8_0 executor matrix; fixed/adaptive splits, repeated routes, warm reuse, eviction, cancellation and retry.
- `reports/resident-integration.log`, `reports/hybrid-integration.log`, `reports/hybrid-adaptive-integration.log`: 135 synthetic GPU model comparisons and pretrained-harness smoke test per mode, covering GPU-only, fixed 50% and adaptive execution.
- `reports/hybrid-cpu-tests.log`: all three focused CPU CTests pass.
- `reports/hybrid-python-tests.log`: ten Python tests pass.
- `reports/hybrid-memcheck.log`: Compute Sanitizer memcheck reports zero errors.
- `reports/hybrid-gemma-cli.log`: adaptive CLI smoke completes 16 generated tokens with the new option.
- `reports/hybrid-gemma-pretrained.json` and its log: SHA-256 verified Gemma QAT artifact, adaptive GPU/CPU execution, full-vocabulary logits, 16 greedy tokens and state replay pass. Every reported comparison has zero logit error.

The 32-token synthetic hybrid fixture measured CPU 3796 us, GPU 8538 us, total 8677 us and overlap lower bound 3781 us on an RTX 4060 Laptop GPU. This is one diagnostic observation, not a throughput benchmark or a speedup claim. Native P100 and Linux execution, concurrent-workload calibration, double-buffered GPU transfers and long-duration pressure acceptance remain open.

## Model throughput benchmark

[The Gemma benchmark](HYBRID_BENCHMARK.md) compares five policies with one warm-up and three measured 32-token requests each. GPU-only reached 2.908 median decode tokens/s; fixed 50% reached 2.165 (-25.6%) and adaptive reached 1.747 (-39.9%). All 20 requests produced identical tokens. The short repeated workload showed no hybrid speedup on this RTX 4060 host; cache warm-up and laptop variability limit generalization. Raw commands, responses and logs are retained in `reports/hybrid-benchmark-20260909/`.
