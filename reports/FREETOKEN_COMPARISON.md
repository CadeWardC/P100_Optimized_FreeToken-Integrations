# This project compared with FreeToken

## Overall finding

This project implements a **subset of FreeToken-inspired inference strategies inside a P100-patched llama.cpp runtime**. It is not a FreeToken fork, a complete translation of FreeToken into C++, or a demonstrated performance-equivalent implementation. The shared ideas are host-resident expert weights, a bounded persistent expert cache, correct remapping of expert IDs, and CPU/GPU execution of different expert contributions. The surrounding runtime, scheduling policy, numerical kernels, loading system, and serving restrictions differ substantially.[^1][^2]

The most important distinction is between **intentional adaptation** and **unfinished implementation**. Native C++/GGUF integration and a conservative Pascal execution path are deliberate architectural choices. Missing double-buffered prefill, semantic checkpoints, and runtime memory redistribution are unfinished features. The current adaptive split and numerical compatibility kernels are implemented intermediate designs, with limitations that materially affect speed.[^2][^3]

There is meaningful correctness evidence: saved results show synthetic GPU integration and one exact Gemma QAT model passing against the ordinary CPU inference path on an RTX 4060 Laptop GPU. There is no evidence here of a completed FreeToken-versus-project benchmark, actual P100 acceptance, or matching pretrained Qwen acceptance. The latest hybrid benchmark shows slower throughput for every tested CPU miss split than for GPU-only expert execution.[^12][^13]

## Comparison baseline and evidence

The inspected project is the working tree in `llama.cpp`, including modified tracked files and new untracked source. Its base is llama.cpp `bb4caa7540188872173c44d161602d9271386413`, identified locally as v0.2.0. The 31 initial P100 patches come from `shinbunbun/llama-cpp-p100-patches` at `7c4cb31ca7bc47b25f3cb5b35d5b436d56bedcc8`. Subsequent expert-offload work remains outside that original baseline patch.[^1]

The included, clean FreeToken reference is `af71ba43206e124f5ff6419b47ee36c6e9981078`. Upstream `main`, checked on September 9, 2026, was `3d919e9bd94fc5454bdb50e09659648443e30f5e`, three commits ahead. The comparison below uses the included reference for detailed code tracing and separately accounts for the upstream delta. This prevents a renamed upstream component from being incorrectly described as a missing feature.[^15]

Implementation claims are based on source inspection. Test and performance claims refer to existing saved reports, not newly executed inference tests. Documented reasons are distinguished from engineering inferences; shallow history and uncommitted changes do not establish the original motivation for every line. Source presence alone does not establish production reliability or performance.

## Main differences

| Area | FreeToken reference | This project's current implementation | Why it differs |
|---|---|---|---|
| Runtime | Python/PyTorch orchestration with compiled CUDA, Triton and C++ components | Native llama.cpp/GGML C++ runtime and CUDA backend | Deliberate reuse of llama.cpp, GGUF and native deployment infrastructure |
| Hardware/toolchain | Source installation documents Linux x86-64 and CUDA 13; README targets modern RTX families | P100/SM60 target with CUDA 12.x; development runs on RTX 4060 | Pascal compatibility and native Windows/Linux goals |
| Expert cache | Shared slot pool, LRU policy, format-specific bank layouts and device-side movement/planning | Per-context shared-across-layers uniform slot pool, host-side leases and protected hits | Smaller, explicit ownership model suited to GGML integration |
| Cache minimum | Reference cache validation requires at least a full layer's expert count | Can execute selected experts through smaller tiles and cache capacities | Bounded-memory operation on constrained hardware |
| CPU/GPU split | Decode policy informed by CPU/PCIe bandwidth profiles, including concurrent measurements | Fixed CPU miss percentage or elapsed-time-per-expert adaptive heuristic | Partial scheduling implementation; calibration remains incomplete |
| CPU execution | Persistent worker infrastructure and pinned I/O; reads registered expert banks | A separately launched CPU task uses the reference FFN executor and a one-expert tile | Reuse of the existing correctness-tested executor |
| GPU transfer overlap | Dedicated prefill copy stream and two full-layer buffers, where configuration permits | GPU cache transfers complete before dependent GPU compute; CPU branch can overlap | DMA/compute pipeline remains unfinished |
| Prefill/decode | Distinct strategies; full-layer prefill streaming available | Same general routed tiled executor; GPU groups capped at eight assignments | Numerical consistency and bounded execution prioritized |
| Model graph placement | GPU-oriented model execution with expert-offload backends | Experimental host-bank path requires CPU attention/shared layers and one sequence | Conservative post-routing integration boundary |
| Arithmetic | Multiple format-specific fused kernels, including packed Q4_0 | Explicit F32 compatibility path and ordered Q4/Q8 reductions; table-compatible GPU GeGLU | Fix observed divergence from the portable CPU baseline |
| Semantic caching | Tool-call anchors tied to recurrent state and retained KV prefixes | Existing llama.cpp state/cache facilities; no new semantic-anchor policy | Separate planned milestone |
| Elastic memory | Idle-only cache rebuild and graph recapture without reloading weights | Fixed cache budget and startup admission check | Resize coordination and recovery not implemented |
| Loading/formats | HF/FTW plus supported GGUF Q4_0 machinery and modern quantized formats | GGUF host-bank path accepts F32, F16, Q4_0, Q8_0 experts only | Bounded validated scope; not all inherited GGUF formats are supported here |
| Validation | Separate engine, tests and hardware assumptions | Synthetic matrix and short Gemma acceptance on RTX 4060 | Available development hardware and staged implementation |

Sources: runtime and scope [^1][^2]; cache and execution [^3][^4][^5][^6]; semantic/elastic facilities [^9][^10]; validation [^12][^13].

## 1. The runtime was intentionally rebuilt around llama.cpp

FreeToken contains a complete serving engine: request scheduling, model loaders, attention and KV subsystems, expert offloading, checkpoint conversion, and HTTP APIs. Its Python layer delegates substantial work to compiled components; describing it simply as “Python versus C++” would obscure where the computation actually runs.[^5][^16]

The project plan explicitly chooses llama.cpp plus the P100 patches to preserve the existing CLI, server and GGUF ecosystem. FreeToken is a strategy reference. Consequently, copying its Python classes would not integrate them into GGML's graph scheduler, tensor ownership, model loading, or backend synchronization.[^2]

The local implementation adds `llama-moe-offload.cpp` for expert-bank handling and execution, `ggml-moe-cache.cpp` for cache ownership/storage, and hooks in the model, graph and context code. It intercepts the complete routed expert FFN after routing, computes contributions, then resumes the ordinary graph. This is a defensible way to preserve model semantics while changing expert execution, but it brings a distinct synchronization and allocation cost profile.[^3][^4][^7]

**Practical consequence:** this project does not inherit FreeToken improvements automatically. An upstream scheduling or kernel change must be interpreted and reimplemented against GGML interfaces. Conversely, not every inherited llama.cpp feature is compatible with the experimental host-bank path.

## 2. P100 support changes the execution assumptions

The reference installation instructions specify Linux x86-64 and CUDA 13, while the project targets CUDA 12.x and SM60. More decisively, the inspected llama.cpp CUDA backend explicitly disables CUDA graphs for devices below Volta. The local plan preserves that guard for P100 and treats graph capture as separate research or a newer-GPU feature.[^2][^11][^16]

FreeToken's CPU executor integrates submit/sync work with CUDA execution and owns persistent task descriptors and buffers. The project instead uses an eager scheduler boundary and explicit backend completion. Adopting graph-dependent machinery unchanged would conflict with its supported P100 execution path.[^5][^7]

The distinction is narrower than “P100 cannot overlap work.” The current graph guard does not prevent a future explicit transfer/compute pipeline, bandwidth profiling, semantic checkpoints, or cache resizing. Those omissions cannot all be attributed to Pascal hardware. They are additional engineering work.

The 31 original P100 patches also include kernel tuning, operator fusion, routing and sampling changes. Those are a separate source lineage from the new FreeToken-inspired cache/executor. A future performance study must distinguish ordinary llama.cpp, the P100-patched baseline, and the added offload implementation.[^1][^2]

## 3. The cache implements the same broad idea with different mechanics

The local cache stores compact gate/up/down expert bundles in fixed uniform slots. Keys contain model, layout, adapter, layer and expert identity, while the active executor supplies fixed model/layout/adapter values within its own context. Tickets track ownership, slot generations and leases. A slot being loaded or used cannot simply be evicted; resident routed experts are acquired before misses are reserved.[^3][^4]

This is real caching: weights survive across calls, hits avoid reloading, and capacity pressure leads to eviction. “Shared across layers” means within one context, not a process-wide multi-model cache. Uniform projection layouts are checked when the executor is constructed.[^3]

FreeToken also uses a shared LRU slot cache, but integrates it with device-side planning, quantized bank schemas, registered host bank addresses and prefill buffers. In the inspected reference, cache-size validation requires at least `num_experts` slots; overlap requires at least twice that many. The local tiling design can work below a full layer's size, provided an expert bundle and required overhead fit.[^4][^6]

**Why:** the local design makes bounded storage and lifetime correctness explicit before adding more concurrency. **Tradeoff:** host-side planning and serial cache operations are simpler to inspect, but may add latency and constrain overlap. This is an implementation tradeoff, not proof of a measured cache-management bottleneck.

## 4. The adaptive policy is not FreeToken's bandwidth policy

FreeToken's `bench_profile.py::load_hybrid_fetch_fraction` prefers measurements of CPU expert execution and PCIe gather obtained while both run concurrently. Its GPU fetch fraction is:

`GPU fetch fraction = PCIe bandwidth under overlap / (PCIe bandwidth under overlap + CPU expert bandwidth under overlap)`

Older profiles fall back to a standalone-bandwidth model. Profile matching takes hardware and expert format into account. FreeToken also supports a fixed fetch cap. This is a calibrated policy with configuration-dependent behavior, rather than an assurance that every request automatically receives an optimal split.[^5]

The local adaptive implementation instead estimates:

`CPU miss fraction = measured GPU cost per expert / (measured CPU cost per expert + measured GPU cost per expert)`

It starts at 50%, rounds to a whole number of unique missing experts, and generally updates costs using 75% previous estimate plus 25% latest sample. With multiple misses it forces at least one expert onto each branch. Resident hits stay on GPU. CPU-selected misses are not admitted to the GPU cache.[^3]

Several differences matter. GPU cost includes work on resident hits as well as fetched experts. Unique-expert count does not capture how many token assignments each expert receives. Prefill and decode share the policy, and there is no hardware/format calibration profile comparable to FreeToken's. Forcing both branches to run also prevents the adaptive mode from choosing an all-GPU split when there are multiple misses.[^3]

**Why:** this is an initial feedback heuristic built on available timing counters. **Inference:** its averages can conflate transfer, computation, cache residency and request size; that makes poor assignments plausible. The benchmark does not isolate which of those factors caused its regression.

## 5. CPU/GPU concurrency exists; GPU DMA/compute overlap does not

The local scheduler launches a CPU task with `std::async`, then runs the GPU branch, waits for completion, and merges contributions in original routing order. It joins the CPU worker on failures or cancellation. This is genuine concurrent branch execution.[^3]

Inside the GPU cache, however, missing weights are copied in staging chunks. The code drains pending work before reusing a staging buffer and completes loads before use. The graph then computes with those weights. A call named `tensor_set_async` does not by itself make the overall path a transfer/compute pipeline.[^4]

FreeToken's prefill path maintains two full-layer buffers, a copy stream, readiness events and release events. This allows transfer of another layer while the current layer computes, subject to capacity and host-residency requirements. Its source explicitly disables incompatible combinations rather than guaranteeing overlap for every configuration.[^6]

The local CPU worker also copies selected weights into its one-expert tile and constructs ordinary GGML execution work. FreeToken owns persistent CPU workers and I/O/task structures that read its bank layouts. Eliminating repeated setup and tile-copy costs is a separate opportunity from changing the split percentage.[^3][^5]

**Why:** the project first established correct bounded execution, cancellation and numerical parity. **Impact:** potentially useful PCIe and launch amortization remains unrealized, especially for long prefill. It should be measured before estimating its benefit.

## 6. The model graph remains substantially on CPU

The host-bank loader restricts the path to Llama MoE, Qwen35MoE and Gemma4 MoE, and requires zero GPU-offloaded model layers. Context construction requires one sequence, operation offloading disabled and KV offloading disabled. Other incompatible model options are rejected. The documented invocation is `-ngl 0 --no-op-offload --no-kv-offload -np 1`.[^7]

Thus the GPU expert cache does not turn the entire model into a GPU-resident inference graph. Attention, shared FFNs, normalization, routing and related state remain in the existing CPU-side execution configuration. The GPU receives routed expert work through the explicit boundary.[^7]

**Documented reason:** preserve a complete expert-FFN boundary and existing model semantics while validating the new subsystem. **Inference:** CPU work outside that boundary can limit end-to-end speedups even if expert execution improves. Layer-level profiling is needed to quantify that limit.

The Gemma adapter preserves GeGLU, fused gate/up handling, original expert-ID output scales, shared dense FFN and attention state. The Qwen adapter preserves its gated shared expert and recurrent state. Supporting those semantics in synthetic tests does not establish numerical or performance acceptance for every checkpoint bearing the same model family name.[^7][^12]

## 7. Numerical compatibility is a substantial implementation choice

The local compatibility CUDA kernel quantizes activations to Q8_0, performs integer block dot products and accumulates results using explicit rounded F32 operations. Q4 zero-point handling subtracts eight from packed values before the integer dot product. This is more specific than simply requesting “FP32 precision.”[^8]

The GPU executor also caps work groups at eight routed assignments. For Gemma it uploads a 65,536-entry F32 GELU lookup table, occupying 256 KiB, to reproduce the portable CPU's half-indexed activation behavior. The current activation stays on GPU; older documents describing intermediate GPU-to-CPU activation transfers are historical.[^3][^8]

FreeToken's Q4_0 implementation likewise keeps weights packed and uses a vector-kernel family for both prefill and decode. However, it calls its own fused activation and GGUF kernel machinery. Similar consistency goals do not make the two arithmetic implementations identical.[^8]

**Why:** earlier local GPU acceptance produced divergent logits/tokens and cache-size-sensitive behavior. The compatibility work was introduced to match the chosen ordinary CPU baseline. Saved later results report zero error at checked Gemma positions.[^12]

**Tradeoff:** ordered scalar-style reduction work and small execution groups prioritize reproducibility over peak throughput. They are candidates for optimization, but removing them without numerical testing could reintroduce the original failures. The demonstrated equality is against this build's CPU behavior on tested inputs, not against FreeToken or every CPU instruction-set implementation.

## 8. Semantic checkpoints and elastic memory remain missing

FreeToken's semantic caching code records tool-call anchors and coordinates saved recurrent state with retained attention context. These anchors identify useful exact-prefix restoration positions; they do not authorize reuse across merely similar meanings.[^9]

The local project has not added the corresponding anchor policy. Existing llama.cpp state saving, prompt caching and recurrent-state replay tests are useful foundations, but are not equivalent to semantic-boundary selection. The plan specifically notes that partial checkpoints may omit attention KV and that a correct restore contract must validate retained attention coverage and exact token position.[^2]

FreeToken also implements `rebuild_runtime_cache`, which resizes selected expert/KV/state pools without reloading weights. Its contract is **idle-only**, requiring no in-flight prefill or decode; it includes validation, teardown and graph recapture. It should not be described as arbitrary transparent resizing during active generation or universal preservation of every cached prefix.[^10]

The local cache has a fixed constructor budget. Its GPU reserve is a startup free-memory admission check, not an allocation held against future users of VRAM. There is no equivalent coordinated runtime resizing interface in the added cache/executor.[^4][^7]

**Why:** both features are separate planned milestones involving state correctness, scheduler coordination and failure recovery. Neither is inherently ruled out by a P100 target. They remain real feature gaps if “complete FreeToken strategies” is the intended destination.

## 9. Format support and deployment are narrower than the project name implies

The new expert path explicitly accepts F32, F16, Q4_0 and Q8_0 tensor types. Ordinary llama.cpp's broader GGUF support does not extend that allowlist automatically. A model filename or nominal quantization label is insufficient: actual routed expert tensor types, shapes and scales must satisfy the loader and uniform cache layout checks.[^3][^7]

FreeToken supports several other bank/kernel formats and a fast checkpoint format, FTW. It also has a GGUF Q4_0 path, so “FreeToken uses FTW, this project uses GGUF” would be an incomplete comparison. The difference is broader loading and kernel infrastructure, with format-dependent support.[^6][^8][^16]

The local plan explicitly defers full FTW compatibility, new FP4/FP8 kernels, multi-GPU placement, a desktop GUI and SSD expert paging. The engine still needs a usable host expert pool; a small GPU cache is not a total model memory limit. Reported process RSS for memory-mapped weights should not be confused with proof that the complete workload fits comfortably in available RAM.[^2][^13]

The project reuses llama.cpp's CLI/server rather than FreeToken's serving stack. Existing development archives are older snapshots, and the packaging notes describe earlier CPU-only integration. They cannot be assumed to contain the latest hybrid or correctness changes. A reproducible deliverable must include untracked additions as well as the tracked diff; archiving Git HEAD alone would omit the implementation.[^1][^14]

## 10. What the saved measurements establish

The latest policy comparison uses one Gemma QAT Q4_0 model, Windows, an RTX 4060 Laptop GPU, eight CPU threads, CPU attention/shared layers, a 1 GiB GPU expert cache, one seven-token prompt and 32 generated tokens per request. Each policy had one warm-up followed by three measured requests.[^13]

| CPU share of cache misses | Median decode tokens/s | Change versus GPU-only experts |
|---|---:|---:|
| 0% | 2.908 | Baseline |
| 50% | 2.165 | -25.6% |
| 25% | 1.931 | -33.6% |
| Adaptive | 1.747 | -39.9% |
| 100% | 1.471 | -49.4% |

All 20 requests, including warm-ups, produced identical token sequences. That establishes consistency for this workload, not a comprehensive model-quality evaluation. The earlier full-vocabulary Gemma acceptance provides a different and stronger numerical check for its tested positions.[^12][^13]

The latest benchmark contains **no feature-disabled CPU baseline and no FreeToken engine baseline**. It establishes that the tested hybrid policies were slower than the project's GPU-only expert mode. It does not establish that the project is a particular percentage slower than FreeToken. Older feature-disabled comparisons used different execution revisions/settings and should not be pooled into this table.[^13]

The saved integration logs report 135 synthetic GPU model comparisons per tested policy mode, plus an acceptance-harness smoke check. Focused CPU test logs report three passing tests. The pretrained hybrid report records a passing hash-verified Gemma artifact and RTX 4060 scope.[^12]

Open acceptance gaps include actual P100 execution, native Linux execution, exact pretrained Qwen validation, larger prompt/context sweeps, and long-duration pressure tests. Laptop clock variation, sequential policy blocks and limited warm-up constrain the performance evidence. CPU contention, staging overhead and adaptive miscalibration are plausible mechanisms, not isolated experimental conclusions.[^12][^13]

## 11. New upstream changes since the included reference

Upstream was three commits ahead at inspection. The main change, `477c8601`, introduces quantization configuration, scheme and method layers. It also replaces portions of the older backend organization, generalizes expert bank roles/layouts and changes relevant CLI terminology from `--moe-backend` to `--moe-strategy`. Some older marker modules, including `moe/cpu_offload.py`, are removed as part of that restructuring.[^15]

The other two commits change nightly wheel publication and correct placement of the Qwen3.8-Flash-Next PLE table alongside FTW checkpoints. These are additional differences from the included snapshot, not evidence that its scheduling strategies disappeared.[^15]

The retrieved cache patch preserves the essential overlap/rebuild machinery while adding layout specifications and slot limits supplied by expert kernels. The detailed bandwidth and prefill conclusions above therefore remain relevant, while any future port should use the newer organization as its integration reference. This report does not claim a complete independent execution audit of every refactored quantization backend.

## 12. Why it is different, and what should happen next

There are four distinct explanations:

1. **Deliberate platform adaptation.** Native llama.cpp/GGML, GGUF ownership, P100 patches and eager execution fit the stated project goals. These are not defects simply because FreeToken chose different infrastructure.
2. **Conservative correctness choices.** CPU graph placement, explicit expert leases, uniform layouts, bounded tiles, ordered arithmetic and a single sequence reduce the number of interacting behaviors under initial validation.
3. **Incomplete strategy implementation.** Bandwidth calibration, persistent optimized CPU execution, double-buffered prefill, semantic anchors and elastic cache coordination are not all present yet. P100 hardware does not explain away these gaps.
4. **Evidence and documentation lag.** Newer source and saved tests supersede statements in the initial manifest, implementation plan, README and old package notes. In particular, it is wrong to say GPU integration and all pretrained acceptance are still entirely absent; it is equally wrong to claim P100 or full FreeToken parity.[^1][^2][^3][^12][^14]

For performance work, first preserve a versioned source/binary/model baseline and capture branch-level timing alongside end-to-end latency. Profile CPU attention/shared work, CPU worker setup and tile copies, cache transfers, GPU kernels and synchronization separately. This resolves which changes can materially improve the target workload.

Next, evaluate a bandwidth-calibrated split against fixed policies and GPU-only execution, including the option to select zero CPU misses. Reuse CPU execution resources where measurements justify it. Introduce transfer/compute overlap as a separately measured change; start only where buffer capacity and host residency make it feasible.

P100 and exact Qwen acceptance should precede claims that this is a usable P100 implementation for both priority models. Semantic checkpoints and elastic memory should follow their own state-correctness tests. The appropriate current description is **an experimental FreeToken-inspired llama.cpp expert-offload implementation with a validated Gemma development path and substantial scheduling/serving work remaining**.

## Sources

Local files below are working-tree sources or saved evidence inspected September 9, 2026. Code locations may move as the uncommitted implementation changes. Upstream links are pinned where applicable.

[^1]: Project provenance: [SOURCE_MANIFEST.json](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/SOURCE_MANIFEST.json>), [README.md](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/README.md>). The manifest's verification flags describe its initial preparation, not all later implementation work.
[^2]: Project architecture and documented sequencing: [FREETOKEN_IMPLEMENTATION_PLAN.md](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/FREETOKEN_IMPLEMENTATION_PLAN.md>), especially sections on goals, scope, platform policy, expert boundary, prefill, checkpoints and milestones.
[^3]: Current local executor: [llama-moe-offload.cpp](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/llama.cpp/src/llama-moe-offload.cpp:645>), including `execute_ffn`, constructor checks and `llama_moe_cached_executor::execute`; [GPU_HYBRID_SCHEDULER.md](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/GPU_HYBRID_SCHEDULER.md>).
[^4]: Local cache implementation: [ggml-moe-cache.h](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/llama.cpp/ggml/src/ggml-moe-cache.h>), [ggml-moe-cache.cpp](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/llama.cpp/ggml/src/ggml-moe-cache.cpp:225>), especially acquire, staging/drain, release and fixed storage construction.
[^5]: FlashML-org, FreeToken pinned CPU/hybrid execution: [bench_profile.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/moe/bench_profile.py#L156), [benchbw.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/moe/benchbw.py#L540), [cpu_executor.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/moe/cpu_executor.py), [cpu_offload.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/moe/cpu_offload.py). Inspected through the clean local reference.
[^6]: FlashML-org, FreeToken pinned [offload_cache.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/moe/offload_cache.py), particularly `validate_rebuild`, `_init_prefill_overlap_buffers`, `prefetch_prefill_layer`, `wait_prefill_layer` and `release_prefill_layer`. Inspected through the local reference.
[^7]: Local integration restrictions and model preservation: [llama-context.cpp](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/llama.cpp/src/llama-context.cpp:97>), [llama-model.cpp](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/llama.cpp/src/llama-model.cpp:1291>), [GPU_GENERATION.md](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/GPU_GENERATION.md>). Historical status paragraphs in the latter are superseded by source 3.
[^8]: Local [moe-precise.cuh](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/llama.cpp/ggml/src/ggml-cuda/moe-precise.cuh>), [GPU_CORRECTNESS_FIX.md](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/GPU_CORRECTNESS_FIX.md>); FlashML-org, pinned [fused_q4_0.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/moe/fused_q4_0.py). Current GPU GeGLU behavior is described in source 3.
[^9]: FlashML-org, pinned [scheduler/cache.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/scheduler/cache.py#L151), `snapshot_toolcall_anchor` and associated retention logic. Local reference source inspected.
[^10]: FlashML-org, pinned [engine/engine.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/engine/engine.py#L779), `rebuild_runtime_cache`; [cache_budget.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/engine/cache_budget.py). Local reference source inspected.
[^11]: Local CUDA graph architecture guard: [ggml-cuda.cu](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4746>).
[^12]: Saved local validation: [hybrid-gemma-pretrained.json](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/reports/hybrid-gemma-pretrained.json>), [hybrid-adaptive-integration.log](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/reports/hybrid-adaptive-integration.log>), [hybrid-cpu-tests.log](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/reports/hybrid-cpu-tests.log>), [MODEL_ACCEPTANCE_MANIFEST.json](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/MODEL_ACCEPTANCE_MANIFEST.json>). The manifest contains stale status/report pointers; the specific latest reports take precedence.
[^13]: Saved local benchmark: [HYBRID_BENCHMARK.md](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/HYBRID_BENCHMARK.md>), [raw results.json](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/reports/hybrid-benchmark-20260909/results.json>), September 9, 2026.
[^14]: Historical packaging scope: [DEVELOPMENT_PACKAGE.md](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/DEVELOPMENT_PACKAGE.md>). Compare current source and source 3 before using an archive as evidence of current behavior.
[^15]: FlashML-org, [upstream comparison](https://github.com/FlashML-org/FreeToken/compare/af71ba43206e124f5ff6419b47ee36c6e9981078...3d919e9bd94fc5454bdb50e09659648443e30f5e), retrieved September 9, 2026. [Saved GitHub API comparison](<C:/Users/cadel/Documents/Coding_Projects/p100 FreeToken/reports/freetoken-upstream-comparison-20260909.json>) contains commit metadata, file list and patches.
[^16]: FlashML-org, pinned [installation requirements](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/docs/install.md), and [FreeToken repository overview](https://github.com/FlashML-org/FreeToken). Engine source requirements are distinct from the advertised Windows/Linux desktop distribution.
