# P100 FreeToken

**Ubuntu:** see [Ubuntu build and run instructions](UBUNTU_BUILD.md). Run `sh scripts/build-ubuntu.sh cpu`, or use `p100` with a CUDA 12.x toolkit. The browser interface builds without the optional native UI toolkits.

Generated Windows builds, old distribution packages and temporary audit files have been cleaned out. Paths to these outputs in the historical reports describe earlier runs; rebuild current binaries from source.

Reusable integration: [modular FreeToken runtime](MODULAR_FREETOKEN.md). The `llama.cpp/freetoken` directory can be copied into another GGML-based engine and linked as `freetoken::runtime`; see its [integration guide](llama.cpp/freetoken/README.md).

Native GUI preview: [build, verification and current limitations](NATIVE_GUI_BUILD.md). The C++/FLTK Windows executable is `build/desktop/FreeP100.exe` (formerly `freetoken-native.exe`; now with live tok/s, loading stages, VRAM/RAM gauges and a streaming server log); [usage and Linux build instructions](desktop/README.md).

Latest comparison: [graph-enabled project versus installed FreeToken](FREETOKEN_CUDA_GRAPH_COMPARISON.md). A fresh matched Gemma chat run measured 20.95 versus 19.13 median decode tokens/s; FreeToken delivered first text sooner and had slightly lower total response time. The ranges overlap, so this is workload-specific evidence rather than a general performance lead.

Latest execution work: [GPU-resident expert decode and CUDA graph replay](GPU_RESIDENT_EXPERTS.md) keeps single-token GPU-only expert activations, routing weights and merging on device. Correctness, cancellation, cache lifecycle and CUDA replay checks pass. Replay measured 20.824 versus 18.020 tokens/s (+15.6%) with the same binary; the separate boundary-only comparison measured -4.4%. See the report for methodology and scope.

This folder contains the downloaded P100-patched llama.cpp source and a detailed plan for implementing FreeToken-inspired strategies on native Linux and Windows.

Start with [FREETOKEN_IMPLEMENTATION_PLAN.md](FREETOKEN_IMPLEMENTATION_PLAN.md).

Latest work: [FreeToken runtime gap work](FREETOKEN_GAP_CLOSURE.md) adds concurrent bandwidth calibration, pipelined expert tiles, bounded full semantic checkpoints, coordinated context rebuilding and opt-in native GPU placement/kernels. Compatibility and native numerical modes have separate acceptance boundaries; see the report for tested combinations and remaining limitations.

Previous work: [GPU-resident activation and hybrid scheduler](GPU_HYBRID_SCHEDULER.md). Gemma GeGLU now stays on GPU with CPU-compatible lookup arithmetic. Timing counters and an opt-in fixed/adaptive CPU miss split expose and enable CPU/GPU overlap. Synthetic GPU-only, fixed and adaptive integration checks pass. [Gemma throughput benchmarking](HYBRID_BENCHMARK.md) found GPU-only fastest at 2.908 median decode tokens/s; fixed 50% was 25.6% slower and adaptive was 39.9% slower, with identical generated tokens. This remains a short RTX 4060 development benchmark; P100 acceptance and broader model-level performance gains are separate gates.

The primary targets are **Qwen3.6-35B-A3B and Gemma 4 26B-A4B**, with equal priority for implementation and acceptance. Both use the experimental CPU offloading path and persistent expert cache through `--moe-cpu-cache-mib`. Synthetic comparisons pass; exact pretrained GGUF and CUDA/P100 acceptance remain open. See [target models and usage](TARGET_MODELS.md).

Qwen preserves its gated shared expert and hybrid recurrent state. Gemma preserves GeGLU experts, original-ID output scales, shared dense FFN and sliding/full attention. The integration suite covers 107 synthetic configurations, including 17 cached Qwen and 28 cached Gemma cases. See [Qwen verification](QWEN_CPU_OFFLOAD.md) and [Gemma verification](GEMMA_CPU_OFFLOAD.md).

M0 implementation has started: [M0_BASELINE.md](M0_BASELINE.md) documents the new CPU/P100 build presets, inventory/benchmark recorder, passing Windows CPU checks, and work that remains pending until P100 hardware is available.

Baseline hardware measurements are now deferred. [M1_EXPERT_BOUNDARY.md](M1_EXPERT_BOUNDARY.md) documents the CPU expert-bank, route-planning and bounded FFN executor, now integrated with the Llama MoE GGUF loader and `llama_decode` behind `--moe-cpu-tile-mib`.

The plan has been source-audited; see [INTEGRATION_SANITY_CHECK.md](INTEGRATION_SANITY_CHECK.md) for corrections, verification results, and the implementation gates that remain. The combined baseline patch was corrected to include both newly added P100 kernel files.

CPU MoE also exposes `llama_moe_memory(ctx)` and performance-log memory summaries: retained backend allocations, tracked temporary peaks, and lifetime FFN/tile/copy counters. Sustained generated-model checks pass. These counters do not measure complete process RAM or OS residency; see the M1 notes for coverage and exclusions.

M2 now includes uniform cache state, compact backend storage and a reusable cached CPU executor connected to the inference boundary. One bounded cache is shared across layers within each context. Tests verify cached FFN parity, resident-hit protection, eviction, warm reuse and context telemetry. A delayed test backend checks event ordering and allocation-failure cleanup. The CUDA execution boundary and P100 validation remain open. See [M2_CACHED_EXECUTOR.md](M2_CACHED_EXECUTOR.md).

| Item | Contents |
| --- | --- |
| `llama.cpp/` | Full llama.cpp v0.2.0 source with all 31 P100 patches applied on local branch `freetoken-p100` |
| `p100-patches/` | Original ordered patch collection and documentation |
| `FreeToken-reference/` (archived) | Research-only checkout moved to the sibling cleanup recovery folder; revision retained in `SOURCE_MANIFEST.json` |
| `SOURCE_MANIFEST.json` | Exact repository revisions and individual patch SHA-256 hashes |
| `P100_BASELINE.patch` | Combined diff of the applied P100 changes against the pinned llama.cpp base |
| `FREETOKEN_IMPLEMENTATION_PLAN.md` | Architecture, implementation milestones, tests, and Linux/Windows packaging plan |

The source changes remain uncommitted. Keep the manifests, patches, and all modified and newly added source files when backing up this workspace. A Git archive of HEAD alone would omit the implementation. CUDA development builds run on the RTX 4060; P100 and production numerical acceptance remain separate gates.

To reproduce the current source, clone llama.cpp at the commit in the manifest and apply either the individual patches in manifest order **or** the combined baseline patch. Do not apply both. CUDA builds for P100 must use a compatible CUDA 12.x toolchain and explicitly target architecture 60; see the plan for candidate Linux and Windows commands and required hardware validation.

The combined patch uses canonical LF line endings and was verified against an LF-normalized Git index. Individual patch hashes include the original local checkout hash and an LF-normalized hash, since Git on Windows can convert patch files to CRLF. Keep patch and target line endings consistent when applying to working files, or apply to an index/checkout configured for LF. Do not overwrite a modified working tree to resolve a line-ending mismatch.

