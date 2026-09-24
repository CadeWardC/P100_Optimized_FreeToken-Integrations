# M2 uniform expert-cache state

Status: the hardware-independent cache state module is implemented and tested on Windows CPU. This starts M2; it does not complete the GPU cache milestone or close the remaining M1 acceptance gates.

Update (2026-09-09): a backend storage wrapper owns compact projection buffers and connects tickets to eager transfer/execution completion. A reusable CPU executor now uses it during inference behind `--moe-cpu-cache-mib`, with resident-hit protection and context-lifetime telemetry. See [M2_CACHE_STORAGE.md](M2_CACHE_STORAGE.md) and [M2_CACHED_EXECUTOR.md](M2_CACHED_EXECUTOR.md). CUDA/P100 validation remains open.

`llama.cpp/ggml/src/ggml-moe-cache.{h,cpp}` provides the fixed uniform slot pool and backend storage used by the optional context-owned CPU cached executor. The state class remains hardware-independent; the wrapper owns completion and storage. Both are compiled into `ggml-base`.

## Contract

- Capacity is `floor(budget / padded_slot_bytes)`; zero capacity and slot counts beyond positive I32 indexing are rejected. The reported arena size stays within the byte budget. The caller must compute the complete padded gate/up/down slot size from the backend. The module allocates metadata only, not that arena.
- Keys contain model identity, layout identity, adapter generation, layer and original expert ID. Model and layout identities must be nonzero. The caller assigns identities without collisions and retains authoritative host storage. A pool accepts exactly one layout identity; mixed size classes require separate pools and are not implemented here.
- A miss reserves an empty or least-recently-used ready slot. Loading and active slots cannot be evicted. Repeated requests for those keys return `pending` or `busy`; exhausted capacity returns `full` without mutation. A caller must tile work or wait for completion, and must deduplicate keys within each tile.
- A successful load completion publishes the whole bundle and pins it for execution. Failed loads empty their slot only after the caller has drained submitted work. Release makes a completed execution eligible for eviction and updates recency. The API does not poll backend events: calling completion or release certifies that the corresponding backend work has finished.
- Tickets contain cache owner, slot index, content generation and lease serial. Content generation changes on replacement, while lease serial changes on every acquisition. This rejects stale completions both after eviction and after reacquiring unchanged contents. A ticket is usable only while its matching bundle is pinned. The pool is fixed; no arena relocation, resize or graph invalidation is implemented.
- All calls are serialized by the owner. Tickets are non-owning and cannot outlive the cache. The backend owner must drain transfers and kernels before destroying either metadata or storage.
- Counters track hits, load reservations, evictions, failed loads and padded slot bytes for successful loads. Counters saturate rather than wrap. Padded slot bytes are not measured transfer traffic, allocation telemetry or timings. A failed attempt may have transferred bytes which this module cannot observe.

## Verification

The `test-moe-cache` CTest checks invalid geometry and identities, pending and active slot protection, out-of-order load completion, failed-load retry, LRU selection, cross-model/layer/adapter isolation, foreign and stale tickets, and unchanged content generations on hits. An independent map-based LRU oracle checks 30,000 acquisitions at capacities 1, 2 and 7. Simulated compact slot contents verify duplicate route assignments and capacity-one execution; warm routes fitting in the pool require no new loads.

The simulation supplies completion notifications directly. It does not execute CUDA events, transfer real tensors or establish numerical kernel correctness. The M1 executor's separate numerical tests remain in the regression suite.

Windows CPU builds of the cache test, existing MoE tests, CLI and server pass. Six focused CTests pass: cache state, MoE integration, expert offload, sampling, grammar parser and quantization. See [build log](reports/m2-cache-build.log) and [test log](reports/m2-cache-tests.log).

Reproduce from the workspace:

```powershell
cmake --build build/ft-windows-cpu --config Release --target test-moe-cache test-moe-offload test-moe-integration llama-cli llama-server
ctest --test-dir build/ft-windows-cpu -C Release --output-on-failure -R '^(test-moe-cache|test-moe-offload|test-moe-integration|test-sampling|test-grammar-parser|test-quantize-fns)$'
```

## Next integration

The user selected Qwen3.6-35B-A3B and Gemma 4 26B-A4B as equal primary targets on 2026-09-09. The CPU boundary and generated-model cache comparisons cover both architectures, preserving Qwen's gated shared expert/hybrid state and Gemma's GeGLU/scales/shared FFN/sliding attention. Use both for backend integration and acceptance; keep Llama MoE tests as regression controls. See [target models](TARGET_MODELS.md).

Backend-owned compact storage and the reusable context-owned CPU cached executor are now implemented; see the notes above. Next, add a backend-neutral post-routing boundary for eager CUDA and bounded staging. CUDA/P100 allocation, events, cancellation and numerical validation, native Linux validation, pretrained-model acceptance and remaining host-memory accounting are still open.

The historical baseline patch and manifest do not contain this module. Preserve these new files and the changed CMake files with the other uncommitted implementation work.
