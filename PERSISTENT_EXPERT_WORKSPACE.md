# Persistent GPU expert workspace

Implemented September 9, 2026 as the first slice of `FREETOKEN_EXECUTION_GAP_PLAN.md`.

Follow-up: [GPU-resident decode and CUDA graph replay](GPU_RESIDENT_EXPERTS.md) implements and validates the next slice. The report below describes the earlier workspace-only build.

The cached GPU expert executor now retains its ggml context, FFN graph, compute allocator, input/route tensors and host packing vector between calls. Fixed-shape decode updates tensor contents and reuses the allocation. One replaceable workspace bounds retained arena memory independently of layer count. CPU reference execution retains its previous temporary ownership behavior.

The workspace is shared across uniformly shaped layer banks. Physical slot IDs remain dynamic; cache eviction does not change the backing tensor pointers. A batch-size or activation change rebuilds the workspace. Successful cache resize invalidates it before old storage is released. Failed cache allocation preserves the existing workspace and cache. Existing synchronous drain guards continue to protect cancellation and resource lifetimes.

The compute budget still includes the retained GPU arena and compatibility activation table. `llama_moe_memory()` now exposes `retained_expert_compute_bytes`, `expert_workspace_builds` and `expert_workspace_reuses`; performance logging includes them. Existing peak accounting still describes the working set during execution, so do not add the retained arena to an overlapping peak to estimate total process memory. These counters do not account for all backend or process overhead.

The public memory-report struct has new fields. Library consumers must rebuild. CUDA graph replay remains disabled, so no captured executables retain references to resized storage in this build.

Changed runtime files:

- `llama.cpp/src/llama-moe-offload.cpp` and `.h`: persistent ownership, reuse and invalidation, internal counters.
- `llama.cpp/src/llama-context.cpp` and `llama.cpp/include/llama.h`: memory-report and logging fields.
- `llama.cpp/tests/test-moe-offload.cpp`: changing-input/route/layer regression, eviction, shape/activation changes, failed and successful resize.

Verification artifacts are in `reports/persistent-workspace/`. The focused persistence test makes twelve calls with changed inputs/routes/layers and observes one workspace build and eleven reuses, while comparing outputs with the ordinary executor. Existing GPU tests also exercise native mode, fused banks, Q4_0/F32/F16/Q8_0, cancellation, budget rejection, pipelining and hybrid splits.

This slice does not remove the host activation boundary, introduce persistent CPU workers, change numerical mode, enable CUDA graphs or tune kernel dispatch. Variable batch shapes can still rebuild the single workspace. The later execution-gap steps remain separate work.

## Measured result

The new executable does not demonstrate a chat throughput improvement on this laptop. AB and BA sessions used the saved Gemma chat launch settings, one excluded warmup per session and six measured requests per executable in total. All six paired response texts matched exactly.

| Measurement | Original executable | Persistent workspace |
| --- | ---: | ---: |
| Median decode tokens/s | 10.952 | 10.610 |
| Decode range | 9.745-11.375 | 10.488-11.368 |
| Median first text, seconds | 3.487 | 3.432 |
| Median response, seconds | 14.919 | 15.054 |

The updated pooled median decode is 3.1% lower; the observed ranges overlap. These short measurements do not establish a speedup or rule out a small regression. The initial exploratory run overlapped compilation and is excluded. See `reports/persistent-workspace/summary.json`, exact launch commands, full streaming events and saved executable hashes.

The implemented benefit is removal of repeated graph/arena construction for stable shapes and explicit bounded ownership. It is groundwork for device-resident execution and graph replay, not evidence that the FreeToken performance gap is closed.

## Validation

- CUDA build: `build-final.log`.
- CPU executor and cache lifecycle tests: `cpu.log`, `cache.log`.
- GPU executor and focused persistence tests: `device.log`.
- Compute Sanitizer: `memcheck.log`, zero reported errors.
- Compatibility and native GPU integration: `integration.log`, `integration-native.log`; 135 comparisons each, including Qwen/Gemma, state continuation and recovery.
- Real Gemma chat regression: all six paired response texts matched across the two benchmark orders.
- Pretrained Gemma acceptance: `pretrained.log`; full-vocabulary logits, 16 greedy tokens and state replay passed. The logged per-position GPU logit differences were zero in this compatibility-mode check. Its raw diagnostic text is not the conversational quality benchmark.
