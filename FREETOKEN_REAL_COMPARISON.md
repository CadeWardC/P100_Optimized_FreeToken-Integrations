# Installed FreeToken comparison — September 9, 2026

Historical comparison of the earlier build. See [the fresh graph-enabled comparison](FREETOKEN_CUDA_GRAPH_COMPARISON.md) for current measurements; the implementation discussion and results below describe the earlier state.

The installed FreeToken really does exceed 30 tokens/second on this laptop. The current project is not performance-equivalent to it. Adding scheduling and cache features did not reproduce its execution engine.

## Conversational comparison

| Engine | Median decode tokens/s | Measured range | Median first text | Median response time |
|---|---:|---:|---:|---:|
| Installed FreeToken | **24.42** | 23.81–25.83 | 1.81 s | 7.10 s |
| Project, native GPU + pipeline, 3,060 MiB expert cache | **10.86** | 10.73–11.22 | 3.37 s | 15.07 s |

FreeToken is approximately **2.25× faster** on this short conversational workload. Its previously logged 32.24 tokens/second and live raw-completion rates above 30 are real, but throughput varies with the generated text and measurement interval. The 21 tokens/second project result below comes from repetitive raw output and should not replace the conversational result.

The primary comparison uses `/v1/chat/completions`, the same user message, explicit `enable_thinking: false`, temperature zero, 128 requested output tokens, and the same streaming measurement script (`scripts/benchmark_openai_stream.py`). Both engines report **19 prompt tokens** and produce coherent explanations of GPUs. Their answers are not token-identical; this is not a numerical-equivalence or comprehensive quality test.

Final results are in `reports/freetoken-installed-comparison/chat-summary.json`. Each engine gets one excluded warmup and three measured responses, running alone. The project's expert budget is 3,060 MiB, native GPU kernels and pipeline are enabled, and its context is 512 tokens. FreeToken uses automatic cache sizing. Both reuse prompt prefixes, but FreeToken reuses 18 prompt tokens and the project reuses 14; startup and prefill latency are reported separately from the approximate streaming decode rate.

The streaming rate counts engine-reported output tokens over the interval between the first and last nonempty content/reasoning event, subtracting one token. Both engines stream approximately one token per event here. FreeToken reports 127 generated tokens versus the project's 128 for the same 128-token request. Timing and reporting differences of this size do not establish identical execution. Full events and usage are retained for auditing.

An initial chat diagnostic left template defaults unchanged: the project emitted a reasoning phase while FreeToken answered directly. Those results are retained as `project-chat-default-thinking.json` and `freetoken-chat-default.json`, and are excluded from the primary comparison.

## Raw completion diagnostic

**The raw-prompt runs below produced repetitive text in both engines.** They reveal cache sensitivity, but are not representative conversational answers or a quality validation. Properly formatted chat measurements are recorded separately in `freetoken-chat.json` and `project-chat.json`.

| Configuration | Median decode tokens/s | Measured range | Median response seconds |
|---|---:|---:|---:|
| Installed FreeToken, automatic ~3 GiB expert cache | 36.25 | 36.24–37.32 | 4.61 |
| Project native GPU + pipeline, 1 GiB expert cache | 7.80 | 7.71–7.81 | 17.92 |
| Project native GPU + pipeline, 3,060 MiB expert cache | 21.05 | 20.67–22.01 | 7.30 |

Increasing the project's expert cache improves measured decode speed by **2.70×**. FreeToken remains approximately **1.72×** faster than that larger-cache configuration. The previous comparison understated the project's speed by using a much smaller expert cache. End-to-end medians are 27.53, 7.14 and 17.54 output tokens/second respectively. These are short, warm-cache measurements, not universal performance guarantees; the larger project cache's speed was still rising across the three measured repetitions.

Live measurements and final settings are in `reports/freetoken-installed-comparison/summary.json`. Raw streaming events are in `freetoken-results.json`; project responses and exact launch arguments are in each `project-native*/results.json` below that directory.

- Hardware: RTX 4060 Laptop GPU, 8 GiB; Ryzen 7 7840HS; approximately 16 GB system RAM.
- Installed FreeToken: `0.1.2+g141c31a8d`, native Windows installation, CUDA 13 runtime. The reference checkout is a different revision (`af71ba43206e124f5ff6419b47ee36c6e9981078`); it is not treated as the exact installed binary's source.
- Both Gemma files are 14,439,363,584 bytes with SHA-256 `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`. Full paths and hashes are in `models.json`.
- Same raw prompt: `Explain how a GPU works.` Temperature zero, ignore EOS, requested 128 output tokens, one warmup and three measured responses. Engines run separately to avoid competing for VRAM.
- FreeToken streaming usage reports 127 output tokens and 127 text chunks for this request. Its approximate decode rate uses `(reported tokens - 1)/(last text arrival - first text arrival)`. End-to-end rate includes the initial wait. Raw events are retained rather than assuming each chunk is always one token.
- Project results use the server's decode timer and return 128 token IDs. The tokenizer reports seven prompt tokens versus FreeToken's six. FreeToken reuses five prompt tokens on repeat requests; the project's prompt cache is disabled. Thus this is a practical same-model/workload comparison, not an identical-token, identical-cache-policy microbenchmark. The tiny prompt's cache difference is excluded from the reported decode intervals but affects total latency.
- Project configuration: native kernels, GPU attention/shared layers (`-ngl 99`), pipeline enabled, GPU-only misses, eight CPU threads, flash attention disabled, 256-token context. Both 1,024 MiB and 3,060 MiB expert cache budgets are measured. Other settings remain unchanged.

## What the installed engine actually selected

The live startup log selects `offload` MoE, Triton attention, BF16 activations, and batch-size-one CUDA graph capture. It automatically locks layers 0–7 and 23–29 to CPU expert execution because all expert banks exceed the pinned-memory budget. Its persistent CPU pool uses six physical cores and reports `avx512bf16+q4_0-w4a8`.

It automatically allocates 959 expert slots (approximately 3,059.62 MiB of packed expert weights) and 8,232 KV pages. Prefill overlap is explicitly **disabled** by split residency. Semantic checkpoints are also disabled. This is not the adaptive hybrid miss-splitting backend.

The earlier desktop log independently records 32.24 tokens/second. That session initially used 1,004 expert slots and later rebuilt its caches to 554 expert slots with larger KV/SWA capacity. The live comparison uses a separate server on port 1920 with automatic defaults, not an exact restoration of that earlier cache slider state.

## Implementation differences

| Area | Installed FreeToken evidence | Current project evidence |
|---|---|---|
| Repeated decode work | Captures and replays CUDA graphs; CPU expert wrapper supports graph-capturable host callbacks | CUDA graphs disabled in build; expert executor constructs fresh contexts, graphs and allocators inside each tile batch |
| CPU experts | Persistent pinned pool, specialized Q4_0 W4A8 CPU path, 15 entire MoE layers assigned to CPU | Conservative development build disables AVX/AVX2/FMA/F16C; CPU miss path uses temporary ggml graphs and compact weight copies |
| GPU expert projections | Grouped Q4_0 kernel, combined gate/up projection, then activation and down projection | Separate gate and up operations, then down; GPU tile batches limited to eight assignments |
| Layer boundaries | Graph-oriented execution and device tensor operations in the grouped GPU expert path | Reads activations, expert IDs and weights to host at each expert boundary; synchronizes and reads expert outputs back for host merging |
| Memory allocation | Automatically balances expert slots and KV resources | Explicit budgets; earlier speed tests reserved only 1 GiB for expert weights |
| GPU attention | Triton attention selected | Native GPU placement available, but these benchmark settings disable flash attention |

Evidence locations:

- Installed package: `C:/Users/cadel/AppData/Local/FreeToken/venv/Lib/site-packages/freetoken/engine/graph.py`, `moe/fused_q4_0.py`, `moe/cpu_offload.py`, and captured startup logs. Several CPU/cache modules are compiled `.pyd` files; no claim is made to have audited their exact machine-code implementation.
- Project: `scripts/build-cuda-dev.cmd:8`; `llama.cpp/src/llama-moe-offload.cpp:129` (host boundary), `:463` (batch size and graph construction), `:507` (allocator), `:540` (synchronization), `:548` (readback).
- The project log's “graphs reused” counter refers to outer ggml graph reuse. It does not mean CUDA graph replay is enabled or that the expert subgraphs above are persistent.

These are concrete architectural differences and plausible contributors, not an attribution of a measured number of milliseconds to each feature. No isolated ablations or full-engine profiler trace were collected in this comparison.

## Engineering implications

The next performance work should first make expert buffers, graphs and allocations persistent, reduce host/device round trips, and establish a graph-capturable decode path. Enabling the CUDA graph build flag alone cannot remove the current eager expert boundary.

The CPU path needs an optimized persistent executor and measured layer placement on this processor. The GPU path should use grouped combined gate/up execution with device-side weighting and reduction. Cache budgets and attention settings should then be tuned under matched workloads. Each step needs output validation and a real-model benchmark; no specific speedup is guaranteed.

Bandwidth policy, double-buffered prefill and semantic checkpoints are not sufficient to explain or reproduce this observed decode speed. In particular, the measured FreeToken run has prefill overlap disabled and does not select adaptive hybrid MoE. The earlier emphasis on adaptive scheduling overstated its relevance to this user's configuration.

No execution code was changed for this comparison. These results concern this RTX 4060/AMD laptop, not P100 acceptance or numerical equivalence between engines.

The installed FreeToken engine was left running on `http://127.0.0.1:1920` with health status `ok`, confirmed after the final chat benchmark. Its launch PID is recorded in `reports/freetoken-installed-comparison/server.pid`; final startup and request logs are `final-server.out.log` and `final-server.err.log`. No desktop settings were changed.
