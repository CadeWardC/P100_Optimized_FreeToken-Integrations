# Graph-enabled project versus installed FreeToken

Fresh matched comparison, September 9, 2026, RTX 4060 Laptop GPU.

The project is competitive with installed FreeToken on this short chat workload. Its median decode rate was 9.6% higher, while FreeToken delivered the first text sooner and completed responses slightly sooner. The decode ranges overlap; this does not establish a general engine-wide performance lead.

| Engine | Median decode tokens/s | Decode range | Median first text | Median response |
| --- | ---: | ---: | ---: | ---: |
| Project, GPU-resident decode + CUDA replay | **20.95** | 18.79-21.02 | 2.37 s | 8.53 s |
| Installed FreeToken 0.1.2+g141c31a8d | **19.13** | 18.28-22.52 | 1.28 s | 7.99 s |

Each engine has six measured requests across two independent server sessions. Order was project, FreeToken, FreeToken, project. Each session excluded one warmup and measured three requests. Servers ran separately, with no concurrent build or GPU test. All benchmark servers were stopped afterward.

Both engines loaded the same physical GGUF file, `models/gemma-4-26B_q4_0-it.gguf`, with SHA-256 `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`. Both used the same OpenAI chat request: `Explain how a GPU works.`, thinking disabled, temperature zero, ignore EOS and 128 requested output tokens. Both reported 19 prompt tokens. The existing streaming benchmark client computes decode rate from first to last nonempty text event using engine-reported completion tokens minus one.

FreeToken was fixed to its previously selected 959 expert slots, about 3059.62 MiB, and 8232 KV tokens. This prevents automatic cache-size changes between sessions. Its startup selected 15 CPU expert layers (0-7 and 23-29), six CPU cores, `avx512bf16+q4_0-w4a8`, Triton attention, BF16 activations and batch-size-one CUDA graphs. Prefill overlap was disabled by split residency. Concurrency was explicitly one. An initial startup with default concurrency four was stopped before measurement and excluded.

The project used the existing matched command: 3060 MiB expert cache, 512-token context, eight CPU threads, native GPU kernels, pipeline enabled, GPU-only misses and flash attention disabled. CUDA graphs were enabled. Its binary SHA-256 is `034f6c2e3886fcf91d4327fb537b4ba04c9141668835978f914d612f6a3caf8c`.

This is a practical same-model chat comparison, not an identical-token or numerical-equivalence test. FreeToken reported 127 completion tokens per request; the project reported 128. Generated texts differ across engines, and FreeToken's responses also varied across its repeated requests. The project's repeated texts matched. Both produced explanatory prose. Prefix-cache policies and KV capacity differ, which affects latency; no inference is made that first-text differences are exclusively compute performance.

The earlier FreeToken median of 24.42 tokens/s and project median of 10.86 tokens/s describe the historical comparison in `FREETOKEN_REAL_COMPARISON.md`. FreeToken measured lower in this fresh run, so those historical numbers should not be mixed with the updated project result or used to attribute the entire changed gap to the new code. The separately controlled CUDA replay ablation remains +15.6%, as recorded in `GPU_RESIDENT_EXPERTS.md`.

Evidence is in `reports/freetoken-graphs-comparison/summary.json`, with per-session rates, exact commands, health/version responses, startup logs and complete streaming events. `project-responses.json` and `freetoken-responses.json` retain generated text. `run_comparison.py` and `summarize.py` reproduce the procedure and summaries. Use a new output directory for an independent repeat.
