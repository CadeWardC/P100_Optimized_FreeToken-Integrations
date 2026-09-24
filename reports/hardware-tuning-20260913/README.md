# Hardware-aware desktop tuning — September 13, 2026

Updated and rebuilt the local desktop host. Both model-switch recommendations and the FreeToken automatic profile use the same policy.

Changes:

- Read GGUF tensor offsets to separate expert weights from shared weights without loading the model. Split GGUFs with incomplete tensor accounting retain conservative placement.
- Read per-layer KV head arrays and head dimensions. Previously Gemma's KV-head array was skipped and replaced with the query-head count, substantially overestimating cache memory.
- Fully fitting models use ordinary GPU placement. Oversized supported MoE models use GPU attention/shared weights and a budgeted native expert cache, GPU-only misses, transfer pipelining and backend-controlled CUDA graphs. P100 replay remains disabled.
- Reserve memory for KV, scratch, display headroom and the selected vision projector. Context can decrease from 4096 to 2048 to retain a useful expert cache. Smaller GPUs use smaller physical prompt batches; CPU fallbacks do not create extra caches or semantic checkpoints.
- Prefer detected physical CPU cores on Windows. Sampling, thinking and output-length preferences are preserved.
- Recalculate untouched automatic profiles after unloading the old engine, before launch. This prevents the engine's own VRAM usage from shrinking its replacement profile. Manual overrides remain manual. Restore returns to the saved profile.

Applied RTX 4060 Laptop / 16 GB RAM profile with Gemma 4 26B Q4_0 and its vision projector: 2048 context, 2688 MiB GPU expert cache, all shared/attention layers offloaded, 8 CPU workers, 512 logical batch / 128 physical batch, f16 KV, native kernels on, semantic checkpoints off, CUDA graphs auto. Exact saved settings are in `applied-settings.json`; cache size may be recalculated when available memory changes. Previous settings were backed up in the application data folder.

Measurements from the actual app-owned engine:

| Workload | Prompt tokens | Output tokens | Prefill tokens/s | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Initial revised profile, before KV metadata/batch correction | 548 | 48 | 7.28 | 3.64 |
| Final profile, same synthetic raw prompt | 548 | 48 | 18.45 | 5.44 |
| Final profile, short formatted chat | 19 | 64 | 4.72 | 9.65 |

The raw prompt repeats a short GPU explanation thirty times and asks for a summary. Requests used temperature zero and disabled prompt reuse. The short chat asks “Explain how a GPU works.” with thinking disabled for that request; it returned coherent text and stopped at the requested token limit. Requests went directly to the local engine without altering the user's saved conversation or inference preferences. The two raw runs occurred in separate loads; OS file cache and expert-cache state were not controlled. These are diagnostic observations, not a controlled speedup claim or a FreeToken comparison. Full responses and engine timings are retained in the adjacent JSON files.

Validation: Release host/core build; core regression tests including tensor accounting, per-layer KV arrays, CPU-only and busy-GPU fallbacks, a 28-case VRAM matrix, P100 guards, projector budgeting and preference preservation; host integration covering save/restore, reload, streaming chat, unload and model switching; real Gemma loading and inference with the vision projector enabled.

Limits: this is a memory-aware heuristic, not an autotuned benchmark for every machine. Only this RTX 4060 laptop was physically tested. Native numerical mode can change generated text. Low RAM can still cause paging, and long prompts remain slower than desired. CUDA replay availability follows backend eligibility; the UI's “graphs reused” counter alone does not prove replay.
