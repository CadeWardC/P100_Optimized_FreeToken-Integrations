# FreeToken strategies for a P100 llama.cpp fork

Latest implementation: [GPU-resident activation and hybrid scheduling](GPU_HYBRID_SCHEDULER.md) supersedes the historical CPU activation and missing CPU/GPU split statements below.

Latest update (2026-09-09): standalone eager CUDA expert execution, bounded staging/cancellation, fused Gemma host-bank loading and exact pretrained artifact selection are implemented. The generated-model suite now has 135 comparisons plus an acceptance-harness smoke test. GPU model integration, adaptive splitting/overlap, actual pretrained outputs and P100/native Linux acceptance remain open. See [M2_CUDA_EXECUTOR.md](M2_CUDA_EXECUTOR.md) for current evidence and limitations; it supersedes older status statements below.

Prepared: 2026-09-08. Status: P100 patches applied; experimental CPU expert tiling is integrated for Qwen35MoE, Gemma4 MoE and Llama controls. GPU caching and the remaining FreeToken-inspired policies are planned, not implemented or benchmarked.

M2 update (2026-09-09): uniform cache state, compact backend storage and a reusable cached CPU executor are implemented. `--moe-cpu-cache-mib` connects a context-owned cache to the existing inference boundary, protecting all routed resident hits before reserving misses. The integration suite has 107 generated-model cases, including 17 cached Qwen and 28 cached Gemma comparisons covering shared experts, recurrent state and failure recovery. Cache allocation/copy/hit/eviction telemetry is exposed through the context API and logs. The CUDA execution boundary and P100 validation remain open. See [M2_CACHED_EXECUTOR.md](M2_CACHED_EXECUTOR.md). Remaining M1 acceptance gates are unchanged.

Gemma update (2026-09-09): Gemma 4 26B-A4B has a CPU architecture adapter and cached parity coverage alongside Qwen. The executor supports its GeGLU activation and per-expert output scales while preserving the shared dense FFN. Small-cache and split-file eight-expert routing tests cover all four allowlisted types. See [GEMMA_CPU_OFFLOAD.md](GEMMA_CPU_OFFLOAD.md). Exact pretrained GGUF and hardware acceptance remain open for both models.

Target selection (2026-09-09): the user selected **Qwen3.6-35B-A3B and Gemma 4 26B-A4B** as equal primary models for this implementation. Their text inference CPU architecture and cache integration are implemented and synthetically tested. Subsequent GPU cache and release acceptance must cover both. Existing Llama/Mixtral-style fixtures remain regression controls; Flash-Next remains out of scope. Pretrained-model and GPU acceptance remain open. See [TARGET_MODELS.md](TARGET_MODELS.md).

The official [model configuration](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json), checked on 2026-09-09, declares `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe`. The pinned converter recognizes that architecture and the runtime has a `qwen35moe` implementation. The CPU offload allowlist now accepts it for text generation, one sequence and MTP disabled. Only routed expert execution is replaced; the gated shared expert and hybrid attention remain in the ordinary graph. Synthetic fixtures verify routed outputs, shared-expert merge, logits and recurrent-state continuation with feature-off execution across all four accepted expert types. See [Qwen CPU verification](QWEN_CPU_OFFLOAD.md).

Select and hash an exact GGUF for each of Qwen3.6-35B-A3B and Gemma 4 26B-A4B before pretrained acceptance. No quantization, download or target host memory configuration has been selected yet. Common K/I quantizations are not covered by the current F32/F16/Q4_0/Q8_0 expert allowlist; inspect each chosen GGUF's per-tensor types and extend tests as needed. The complete host expert pool plus runtime overhead must fit RAM under the current design; active parameter counts do not represent total weight storage.

Implementation update: M0 preparation has started without P100 hardware. Windows CPU builds and focused tests pass; build presets and an initial inventory/benchmark recorder are available. See [M0_BASELINE.md](M0_BASELINE.md) for exact evidence and remaining gates. The original planning-time hardware/build observations below are historical.

Sequencing update (2026-09-08): the user has deferred baseline measurements to continue implementation without a P100. The CPU expert-bank and post-routing FFN executor is integrated with the Llama MoE loader/context and tested on generated GGUF models; see [M1_EXPERT_BOUNDARY.md](M1_EXPERT_BOUNDARY.md). Pretrained-model acceptance, whole-context accounting and P100 allocation/execution validation remain open.

Memory telemetry update (2026-09-08): CPU MoE now exposes context-lifetime execution/copy counters, retained backend allocation categories, and tracked temporary peaks including vector capacity, GGML metadata arenas and kernel workspace. Sustained generated-model tests pass. Host bookkeeping, OS residency and failed-FFN partial peaks remain outside these counters; see [M1_EXPERT_BOUNDARY.md](M1_EXPERT_BOUNDARY.md).

Sanity-audited against the pinned source on 2026-09-08. See [INTEGRATION_SANITY_CHECK.md](INTEGRATION_SANITY_CHECK.md) for evidence and unresolved proof obligations. The design is suitable for staged prototyping; it is not a certification of an unimplemented integration.

## 1. Outcome and baseline

Build a native C++/CUDA inference runtime for Linux x86-64 and Windows x64 that runs supported GGUF MoE models with an expert pool larger than P100 VRAM. Preserve llama.cpp's CLI, server, quantization support, and ordinary execution path. Add FreeToken-inspired policies behind explicit options and validate them on actual P100 hardware on both operating systems.

The selected foundation is **ggml-org/llama.cpp v0.2.0 plus shinbunbun/llama-cpp-p100-patches**. This keeps the fork close to upstream llama.cpp. The P100 project is an ordered patch collection, not a complete source fork; the full source has therefore been downloaded separately and patched locally. Other projects, including poisonxa16/PXA_llama, are alternatives but are not mixed into this baseline.

| Local directory | Origin | Pinned revision |
| --- | --- | --- |
| `llama.cpp/` | https://github.com/ggml-org/llama.cpp | `bb4caa7540188872173c44d161602d9271386413` (`v0.2.0`) |
| `p100-patches/` | https://github.com/shinbunbun/llama-cpp-p100-patches | `7c4cb31ca7bc47b25f3cb5b35d5b436d56bedcc8` |
| `FreeToken-reference/` | https://github.com/FlashML-org/FreeToken | `af71ba43206e124f5ff6419b47ee36c6e9981078` |

`llama.cpp` is on local branch `freetoken-p100`. All 31 patches from `p100-patches/nix/patches.nix` were applied in their declared order, without reported offsets or fuzzy matching. They remain uncommitted working-tree changes. These repositories are shallow clones. No remote fork was created, and nothing was published. `SOURCE_MANIFEST.json` records revisions and patch hashes for reproduction.

The inspected development computer has an RTX 4060 Laptop GPU (8 GB), driver 610.62, and CUDA toolkit directories 12.6 and 13.0; the default `nvcc` points at 13.0. This is a development host, not evidence of P100 compatibility. No compilation or inference benchmarks were run for this planning task.

## 2. Scope and priorities

### First usable release

1. A verified P100 build of the patched baseline for both target operating systems.
2. Host-resident GGUF expert banks with bounded pinned staging memory.
3. Persistent GPU expert caching and correct expert-ID remapping.
4. Adaptive placement of decode misses between GPU execution and CPU execution.
5. Prefill streaming with transfer/compute overlap where memory permits it.
6. Flags, metrics, correctness tests, reproducible benchmark reports, and downloadable release archives.

### Complete strategy release

Add semantic-boundary checkpoints and live expert-cache resizing. The pinned CUDA backend explicitly disables CUDA graphs below Volta, including P100. Keep this guard intact for supported P100 releases; graph capture is optional research or a newer-GPU feature, not a P100 release requirement. Live redistribution of KV memory follows only after state-preserving resizing is proven. These are separately gated milestones rather than requirements for the first experimental binary.

Start with one GPU and one active sequence; validate an explicitly allowlisted GGUF/quantization combination for each primary model. Conventional-attention Llama MoE synthetic fixtures already provide initial executor validation. The `qwen35moe` CPU integration for the selected Qwen3.6-35B-A3B target now has synthetic coverage for hybrid recurrent-state behavior and the gated shared expert. Exact model files and hashes must be selected based on available host RAM. A dense model provides a regression control but cannot validate expert offloading.

Defer multi-GPU placement, training, new model architectures, native FP4/FP8 kernels, a desktop GUI, SSD-based per-token expert paging, and full FTW checkpoint compatibility. The initial runtime must fit the complete expert pool in host RAM. Publish an explicit model/quantization allowlist; unsupported configurations use the ordinary llama.cpp path or fail validation before allocation.

## 3. What to transfer from FreeToken

FreeToken's published design combines pipelined prefill, expert reuse, bandwidth-aware CPU/GPU execution, prompt-state checkpoints, and memory-budget changes. Its released runtime targets newer NVIDIA cards and depends on a different Python/kernel stack. Transfer the scheduling concepts into GGML rather than importing that runtime into llama.cpp. The paper is a design reference, not a P100 performance guarantee. [FreeToken paper](https://arxiv.org/abs/2608.16157), [reference repository](https://github.com/FlashML-org/FreeToken).

| Strategy | Inspected FreeToken reference | Proposed adaptation |
| --- | --- | --- |
| Global expert LRU | `python/freetoken/moe/offload_cache.py`, `offload_kernels.py` | C++ byte-budgeted cache with GPU slot maps and explicit event lifetimes |
| Bandwidth-based miss split | `moe/benchbw.py`, `moe/bench_profile.py`, `moe/cpu_offload.py` | Calibrate actual GGUF CPU kernels and P100 transfers under concurrent load |
| Double-buffered prefill | `OffloadMoeCache.prefetch_prefill_layer`, `wait_prefill_layer`, `release_prefill_layer` | Two budgeted layer buffers, or bounded expert tiles when a layer is too large |
| Semantic anchors | `scheduler/cache.py::snapshot_toolcall_anchor`, `kvcache/hybrid_radix_cache.py` | Extend llama.cpp's existing prompt checkpoints with exact token-prefix validation |
| Elastic cache budget | `engine/cache_budget.py`, `OffloadMoeCache.rebuild` | Safe-point resize with explicit failure recovery and graph invalidation |
| Fast weight loading | `checkpoint/ftw.py`, `checkpoint/convert.py` | Retain GGUF and first remove unnecessary host copies; consider a derived cache only after measurement |

All reference paths above are relative to `FreeToken-reference/python/freetoken/` where shortened. The implementation sections below are proposed engineering work, not claims that the downloaded fork already contains these features.

## 4. Hardware and portability constraints

P100 uses compute capability 6.0. It has useful FP16 arithmetic but lacks tensor cores and DP4A. Keep the supplied sm_60 kernel gates, and validate accumulation precision rather than assuming faster FP16 is always accurate enough.

**Pin CUDA 12.9.x as the candidate release toolchain and explicitly compile architecture 60.** CUDA 13 removed offline compilation for Pascal. Locally installed CUDA 12.6 is a possible development fallback, subject to compilation and hardware tests. Do not allow CMake to silently select the default CUDA 13 installation. NVIDIA identifies R580 as the last driver branch for these older architectures; the precise P100 driver package and OS combination must be verified and recorded on each test machine. [CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html), [NVIDIA architecture-support announcement](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/).

Linux target: x86-64, initially Ubuntu 22.04 or 24.04 with a CUDA-supported host compiler and a P100-capable proprietary driver. Windows target: native x64 build using Visual Studio 2022 toolchain supported by the pinned CUDA version. Confirm the target Windows edition and P100 driver support before labeling that combination supported. WSL is not a substitute for native Windows validation.

Use portable C++ synchronization and GGML backend events. Put host allocation, pinning, memory-pressure probes, and optional affinity behind small platform adapters. Avoid mandatory Linux-only APIs. Windows processor groups, NUMA, locked-memory behavior, DLL discovery, and cancellation all require native tests. Treat CUDA host registration failure as a normal fallback condition, not a reason to crash.

## 5. Architecture and source integration

### Existing hooks to extend

| Existing llama.cpp location | Role and change |
| --- | --- |
| `src/llama-model-loader.cpp`, `src/llama-model.cpp` | Discover expert tensor layouts; preserve host backing; exclude expert banks from ordinary full GPU placement when the feature is enabled |
| `src/llama-graph.cpp::llm_graph_context::build_moe_ffn` | Identify a complete routed FFN region after routing; preserve router weights, activations, biases, scales, and aggregation semantics |
| `ggml/src/ggml-backend.cpp` | Existing selected-expert transfer path for `GGML_OP_MUL_MAT_ID`; add an explicit execution boundary and avoid duplicate normal scheduler copies |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Stream/event lifecycle, cached tensor dispatch, and later graph capture compatibility |
| `ggml/src/ggml-cuda/mmvq.cu`, `mmq.cu` | Reuse patched kernels with packed expert tensors and remapped IDs; extend only when measurements justify it |
| `ggml/src/ggml-cpu/` | Reuse quantized CPU execution with a dedicated bounded executor |
| `src/llama-context.cpp`, `src/llama-context.h` | Own per-context execution state and invalidate scheduler/graph reuse when storage generations change |
| `tools/server/server-context.cpp`, `include/llama.h` | Reuse prompt checkpoint creation and sequence state APIs |
| `src/llama-memory-hybrid.cpp`, `src/llama-memory-recurrent.cpp` | Validate complete attention and recurrent-state restore |
| `common/arg.cpp`, `common/common.h` | Feature options, validation, and startup report |

The current GGML scheduler already copies only selected experts in one host-weight `MUL_MAT_ID` path. It reads routing IDs back and copies selected ranges into the normal destination layout. That is useful infrastructure, but it is not a compact persistent LRU cache or a whole-FFN CPU/GPU co-executor. Simply skipping copies on a supposed hit would not establish correct residency or reduce the allocated destination size.

Proposed new modules, names subject to review:

- `src/llama-moe-offload.{h,cpp}`: model-level expert descriptors, configuration, compatibility checks, and context ownership.
- `ggml/src/ggml-moe-cache.{h,cpp}`: residency, slot allocation, replacement, generation counters, and pure planning logic.
- `ggml/src/ggml-moe-executor.{h,cpp}`: routed FFN plan and CPU/GPU task coordination through explicit backend interfaces.
- `ggml/src/ggml-cuda/moe-cache.{cu,cuh}`: compact ID maps and bounded gather/scatter support if existing tensor operations are insufficient.

Keep model-specific semantics in the llama layer and memory/execution primitives in GGML. First prototype an explicit graph partition after routing: route -> plan -> CPU/GPU FFN execution -> merge -> downstream graph. Compare that boundary with a dedicated GGML operation before expanding support. Do not call CUDA directly from generic graph construction or split arbitrary matrix nodes without a complete FFN ownership plan.

**M1 must prove allocation and scheduling, not only the FFN arithmetic.** Partition before scheduler reservation/allocation. The original full expert tensors must not remain inputs that cause the scheduler to allocate full-size GPU copies. Backend placement alone does not establish this. Use explicit subgraphs or a reviewed backend operation contract with owned outputs. The existing evaluation callback is an observation/synchronization hook; it does not provide a supported replacement-node executor. Do not use it to overwrite outputs while the original FFN still executes. Preserve output lifetimes across every subgraph and prevent fusion across the offload boundary until tested.

### Expert identity and storage

An expert key includes model identity, layer, expert ID, quantization/layout, and adapter generation. Its descriptor covers gate/up/down tensors, fused gate-up when present, scales, biases, strides, alignment, and padding. All constituent tensors become visible atomically as one ready expert bundle.

Start with uniform expert layouts for one architecture. Later use pools grouped by layout/size class under a global byte budget; a single uniform slot shape cannot represent arbitrary mixed GGUF quantizations efficiently. Keep host GGUF bytes authoritative. Never reinterpret quantized blocks as half-precision weights or silently requantize them.

Do not assume every CPU weight buffer still contains canonical GGUF bytes: the existing CPU repack backend transforms weights on upload and attaches layout-specific tensor traits. M1 uses ordinary CPU buffers for authoritative expert banks and standard CPU kernels. Any later repacked CPU representation is separately identified and budgeted; it must never be copied into the GPU cache as if it were the GGUF layout. Calibration must use the actual selected representation.

Each slot has identity, generation, state (`empty`, `loading`, `ready`, `in_use`), last-use counter, and completion event. A hit requires matching identity and generation plus completed transfer. Never evict an active expert. Keep original expert IDs separate from slot indices, including IDs used for scale and bias lookup. Respect kernel padding requirements observed in the existing scheduler's expert-copy code.

GPU kernels require concrete tensor strides, dimensions, and valid IDs, not just an array of expert pointers. Create compact tensor descriptors for each compatible bank. Either pack bias/scale banks into the same slot order or gather them using original IDs outside the fused kernel; a fused kernel cannot index weights by slot and biases by original ID using the same ID tensor. Reject incompatible fusion shapes. Do not place `-1` in `MUL_MAT_ID` inputs as a CPU-work sentinel: use compact valid route tensors and an explicit scatter map. Disable affected fusions until these layouts pass backend-operation tests.

Separate **slot content generations** from **arena/layout generations**. Ordinary eviction changes slot identity and content; it does not automatically require rebuilding every scheduler graph. Address, stride, topology, operation-parameter, or allocation changes do require invalidation. Every plan records and verifies the generations of slots it uses. This prevents stale reads without making each cache miss trigger a full graph rebuild.

For the first release, own caches per context and share them across that context's layers. Avoid cross-context sharing until synchronization is proven. Prefill must not destroy useful decode residency accidentally; use separate staging buffers or an explicit partition of the same arena.

## 6. Detailed strategy implementation

### A. Host expert banks and safe GPU caching

1. Enumerate GGUF expert spans at load time and calculate exact byte costs.
2. Reuse mapped host weights where possible. Allocate bounded pinned staging buffers first; make registration of populated host banks an optional optimization after platform tests. Avoid a second full-model RAM copy.
3. Allocate a compact GPU cache and metadata before serving; subtract it from the normal model/KV allocation budget.
4. After routing, deduplicate expert IDs while retaining every token-to-expert assignment. Protect hits, reserve miss slots, copy complete bundles, and publish readiness only after events complete.
5. Remap selected IDs to the compact cache layout. Execute all required projections and restore the original token/expert ordering before aggregation.
6. Begin with serialized GPU miss handling. If the active set exceeds capacity, process bounded tiles and accumulate all contributions; never drop experts or alter top-k.

Keep every expert's mapped span alive through loader cleanup: `llama-model-loader.cpp` unmaps unused offloaded ranges after loading. Update retained-range bookkeeping or use an explicit owned host bank, including split GGUF files. Model ownership must outlive contexts, outstanding CPU reads, and DMA; unregister host ranges only after all users drain. A mapped file fitting in virtual address space does not mean its pages are resident in RAM. Budget and monitor the host working set and page faults; report when OS paging invalidates performance assumptions.

The initial staging path is `pageable canonical bank -> pinned staging -> GPU`. Include the first CPU copy, its extra DRAM traffic, and its lifetime in the profiler and policy. A staging buffer is reusable only after its H2D completion event. Do not assume an API with an `async` name provides overlap from pageable memory. Optional direct host registration changes the transfer path and requires a separately keyed calibration.

Acceptance: all-hit, all-miss, repeated IDs, eviction, minimal capacity, fused/separate gate-up, padding, and unsupported-layout cases match the uncached reference within established numerical tolerances. Transferred bytes fall on repeated routes, and allocated device weight memory reflects the configured cache capacity.

### B. Adaptive CPU/GPU decode execution

Profile CPU expert execution, pinned H2D transfers, resident GPU FFNs, router-ID readback, and output transfers. Measure CPU and DMA **simultaneously**, because they compete for host DRAM. Key calibration by GPU identity and PCIe topology, CPU/NUMA configuration, thread count, quantization, expert dimensions, toolkit/build version, and batch regime.

For single-token decode with `M` equal-sized unique misses and a calibrated direct-transfer path, use the following initial estimate:

```text
f_gpu = B_pcie_overlap / (B_pcie_overlap + B_cpu_overlap)
q0 = clamp(round(M * f_gpu), 0, M)
```

`B_cpu_overlap` is effective expert-weight bytes processed per second, not a synthetic memory-copy rate. `q0` counts misses to fetch and compute on the GPU; remaining misses run on CPU. This follows the inspected reference's bandwidth split. It is a starting point, not a complete latency model.

Evaluate nearby integer splits using:

```text
T(q) ~= max(T_gpu_hits + T_h2d(q) + T_gpu_misses(q),
            T_input_d2h + T_cpu(M-q) + T_result_h2d)
        + T_route_sync + T_merge
```

Measure whether transfers and GPU work actually overlap, then refine this conservative model. Include launch overhead and minimum work granularity. For mixed expert sizes, partition by predicted service cost/bytes instead of a raw count. Use stable fallback when calibration is missing and rate-limit adaptations to prevent oscillation.

For bounded staging, replace `T_h2d` with measured end-to-end host-stage/transfer service time and account for contention with CPU FFNs. Validate finite nonnegative timings and bandwidths; avoid division by zero and treat `M=0` as a separate fast path. Constrain candidate splits by free cache slots, layout compatibility, scratch memory, and executor availability before selecting one. Test both endpoints even if they are not near `q0`.

For multi-token batches, unique expert misses measure transfer demand but not computation: one expert can serve many token assignments. Group work by expert, count each transfer once, and model CPU/GPU work by the number and shape of assignments. Keep the simple bandwidth fraction restricted to single-token decode until this extension is validated. Prefer a deterministic tie-break for equal predicted costs.

CPU work must evaluate a **complete expert FFN** using the same quantized weights and model semantics. Assign each token/expert contribution to exactly one execution path. Preserve whether router weighting occurs before or after the FFN, activation/clamping, normalization, and shared-expert behavior. Copy only needed activations to CPU and merge partial outputs in a defined order, preferably with FP32 accumulation. Do not send every intermediate projection across PCIe.

The first hybrid implementation synchronizes routing and runs without CUDA graph capture on P100. Correctness comes first; instrument that synchronization cost. Add a bounded CPU pool that does not oversubscribe the existing GGML worker pool. Request cancellation must drain every task/transfer that references request storage before releasing slots or outputs. A detached task is allowed only if it retains ownership of every referenced resource until completion; cancellation alone never permits immediate reuse.

Scatter each contribution into its original `(token, top-k position)` slot, then reduce in the same expert order as the reference FFN for the initial correctness path. Combining an already-summed CPU result with an already-summed GPU result changes reduction order as the partition changes; allow that optimization only after numerical tests. Carry original token and sequence associations through packing, including mixed server slots. Keep router and shared-expert branches outside the routed-expert cache unless explicitly supported.

Acceptance: test `q=0`, `q=M`, mixed splits, identical routes under different partitions, invalid calibration, CPU contention, and repeated cache churn. Automatic mode must be compared against tuned static CPU offload and GPU streaming, not just an untuned baseline.

### C. Prefill overlap

Long prefill batches may use most experts. Begin with full-layer double buffering: while layer L computes, transfer L+1's expert weights into the alternate buffer. Full-layer transfer can begin without knowing L+1 routes; selective next-layer loading cannot, unless a separately validated prediction mechanism exists.

Reserve space for two maximum-size layer bundles plus workspace before enabling this mode. If it does not fit, use expert tiles after current-layer routing, or fall back to static/serialized execution. Do not force two full layers into a 12/16 GB card. Keep ready/release events per buffer, and prevent overwrite until the previous compute has completed. Handle model layer order, chunked prefill, cancellation, and transitions back to decode explicitly.

Acceptance: logits within the frozen numerical tolerances of serialized prefill, bounded peak VRAM, and timeline evidence that H2D and compute overlap. Bitwise equality is not generally guaranteed when tiling changes kernel selection. Benchmark both cold prompts and warm multi-turn workloads; small prompts may be slower with extra staging. For fallback tiling, preserve original route weights and normalize only once across the original top-k set, never independently per tile.

### D. Semantic-boundary state checkpoints

Extend the existing server checkpoint system rather than adding an independent session store. Candidate boundaries include completed messages, tool-call boundaries, and model-specific thinking delimiters from the actual chat template/parser.

Reuse is valid only for an **exact common token prefix** with compatible model, adapter, template, positional configuration, and complete saved state. Semantic boundaries choose where to save state; they do not make semantically similar prompts interchangeable. On edits, restore the latest valid checkpoint before the changed token, then recompute the suffix.

For hybrid models, capture attention KV, recurrent/convolution state, sequence positions, and any speculative draft state coherently after the boundary token has been consumed. Audit the existing `create_checkpoint` comment about incorrectly inferred state coverage before depending on its position range. Verify the saved data really covers the positions claimed by the metadata. Limit checkpoint bytes as well as count, and isolate request/session identities.

The current server calls `create_checkpoint` **before** `llama_decode`; the queued batch is not in the saved state. It uses `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`, and `llama_memory_hybrid::state_write` then omits attention KV. Therefore, choose one explicit restore contract: (a) partial recurrent/SWA snapshot plus independently retained and validated attention-prefix KV, or (b) full sequence snapshot using `LLAMA_STATE_SEQ_FLAGS_NONE`, with its larger memory cost. Do not treat existing checkpoints as self-contained full-state snapshots. If an anchor falls inside a batch, split the batch so state is captured at the exact boundary, or defer to a valid earlier boundary; never label the end-of-batch recurrent state as an earlier anchor.

Define a boundary as an exclusive token count: a checkpoint at `p` represents exactly tokens `[0,p)`, with token `p-1` consumed. Token sampling/emission is not equivalent to consumption by the model. Validate checkpoint restore return sizes, attention coverage, sequence positions, and draft/target alignment. `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` invalidates earlier on-device snapshots for the same sequence; it cannot back a multi-anchor history without additional independent storage. Failed validation triggers recomputation, not best-effort reuse.

Acceptance: append-only conversations, removed thinking spans, rewritten tool calls, edits before every checkpoint, sliding-window attention, slot reuse, and MTP rollback all reproduce a fresh evaluation within the agreed tolerance. If a state form cannot be restored safely, invalidate and recompute.

### E. CUDA graphs and elastic memory

**P100 production path: CUDA graphs remain disabled.** `ggml_cuda_graph_set_enabled` disables them for devices below `GGML_CUDA_CC_VOLTA`. This is the pinned backend's policy, not a claim that the CUDA API never supported graphs on Pascal. Re-enabling them needs its own evidence, implementation review, and P100 numerical/performance tests; do not remove the guard as part of cache integration.

For optional graph research or newer-GPU support, stable device arenas and device-resident slot maps can change contents between replays; changing allocation addresses or geometry requires graph invalidation. Keep variable CPU work outside captured regions, join with events, and capture only stable GPU segments. Never assume a whole hybrid CPU/GPU step can be replayed as one CUDA graph.

An arena/layout generation change must invalidate llama.cpp's scheduler/graph reuse and any enabled CUDA captures. Slot-content changes must update and validate the maps rather than indiscriminately rebuilding the scheduler. Re-review supplied patches 21 and 22, which optimize scheduler reset and decode slot reuse, for stale address assumptions. Also test MTP, which changes decode batch shapes and rollback requirements.

Budget device memory using disjoint categories:

```text
expert_budget = usable_device_budget
              - resident_nonexpert_weights
              - attention_and_recurrent_state
              - activations_and_graph_workspace
              - prefill_staging
              - explicit_headroom
```

Use measured peak allocations and avoid counting an aliased staging arena twice. Separately budget host expert pages, pinned staging, and checkpoints.

Resize at scheduler safe points: stop admissions for the affected context, drain work/events, validate the target geometry, invalidate graphs/maps, adjust the arena, rebuild views, cold-fill as needed, then resume. Prevalidate minimum working-set capacity. For growth, keep the old arena if a new allocation fails. Under pressure where old and new cannot coexist, drain and release only cache storage, then fall back to a smaller cache or CPU execution if reallocation fails; retain host weights and sequence state. Do not promise an atomic rollback when the old arena has already been freed.

Implement recovery through an allocation path that returns an error. The CUDA backend's ordinary buffer allocator can return null, but its legacy scratch pool retries allocation and then calls `CUDA_CHECK`, which can terminate execution. Catching a C++ exception around existing allocation calls is not sufficient. Reserve execution scratch ahead of admission and test the exact failure paths used by cache resizing. Pure CPU fallback also needs a valid prebuilt executor and scratch budget. If those are unavailable, reject the request or resize explicitly without corrupting state; do not promise successful inference under arbitrary memory exhaustion.

First resize only expert storage while preserving a fixed KV reservation. True live KV redistribution is a later sub-milestone requiring validated state migration or compatible in-place capacity changes. Never discard active KV/recurrent state to satisfy a cache target. Use hysteresis and a cooldown; respond to allocation failures as well as advisory free-memory readings.

Acceptance: grow/shrink cycles, allocator failure injection, pressure during concurrent requests, capture invalidation, minimum budget, and long-running leak tests on both operating systems.

### F. Loading efficiency

Measure cold and warm startup separately. First improve GGUF reads, host placement, and staging without introducing another required weight format. If repeated conversion/layout work is a proven bottleneck, add an optional derived expert-bank cache keyed by source GGUF digest, layout version, quantization, endianness, and architecture. Write atomically, validate bounds/checksums, and fall back to GGUF on corruption. FTW support is not required for the FreeToken scheduling strategies.

## 7. Proposed user interface and metrics

These options do not exist yet; implement and document them in the milestones below:

| Proposed option | Meaning |
| --- | --- |
| `--moe-offload off\|cache\|hybrid\|auto` | Baseline, cached GPU execution, CPU/GPU split, or calibrated selection |
| `--moe-cache-mib N` | Expert cache byte cap; auto sizing must log the resolved value |
| `--moe-host-staging-mib N` | Maximum pinned staging allocation |
| `--moe-prefill serial\|overlap\|auto` | Prefill transfer policy |
| `--moe-policy-profile PATH` | Load/save versioned machine calibration |
| `--moe-fetch-fraction F` | Manual diagnostic override for hybrid miss partition |
| `--moe-elastic off\|on` | Safe-point cache resizing |
| `--semantic-checkpoints off\|on` | Add semantic anchors to existing prompt checkpoint policy |

Reuse existing checkpoint count/spacing controls and server metric conventions where practical. Validate conflicts with existing `--cpu-moe`, `--n-cpu-moe`, tensor placement overrides, adapters, split modes, and speculative decoding. Do not silently ignore an incompatible option.

Expose cache capacity/hit rate, unique misses, evictions, fetched bytes, CPU/GPU expert counts, policy split, copy/compute/merge timing, prefill overlap, checkpoint reused/recomputed tokens, resize counts/failures, host pinned bytes, peak VRAM, TTFT, inter-token latency, and throughput. Metrics should be aggregated without logging prompt content by default.

## 8. Linux and Windows build/release plan

Use one CMake codebase with shared feature flags. Add `CMakePresets.json` entries for Linux CPU, Linux CUDA sm_60, Windows CPU, and Windows CUDA sm_60. Packaging scripts may differ by OS; runtime algorithms must not fork by OS.

Candidate build commands from this project folder, to validate in milestone 0:

```bash
# Linux: install a CUDA 12.9-supported host compiler and select it explicitly if needed.
cmake -S llama.cpp -B build/linux-p100 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 \
  -DCMAKE_CUDA_ARCHITECTURES=60 -DGGML_CUDA_GRAPHS=OFF -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_OPENSSL=OFF
cmake --build build/linux-p100 --parallel 4
ctest --test-dir build/linux-p100 --output-on-failure
```

```powershell
# Windows: x64 developer PowerShell for a CUDA-supported VS 2022 toolset, with Ninja.
cmake -S llama.cpp -B build/windows-p100 -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON `
  '-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe' `
  '-DCUDAToolkit_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9' `
  -DCMAKE_CUDA_ARCHITECTURES=60 -DGGML_CUDA_GRAPHS=OFF -DGGML_NATIVE=OFF `
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_OPENSSL=OFF
cmake --build build/windows-p100 --parallel 4
ctest --test-dir build/windows-p100 --output-on-failure
```

The examples disable OpenSSL to minimize the baseline build dependency; they assume local model files and do not provide HTTPS support. The release decision should either include tested OpenSSL dependencies or document local HTTP/reverse-proxy operation. `GGML_NATIVE=OFF` is only the start of a portable CPU build: choose and document an explicit supported CPU ISA floor and verify dispatch on older P100 host systems. Do not solve host-compiler incompatibility with an unchecked unsupported-compiler override.

In this source, disabling `GGML_NATIVE` can leave AVX, AVX2, BMI2, and (outside MSVC) FMA/F16C enabled by default. The sample commands therefore do not promise compatibility with every x86-64 CPU. Before packaging, either declare and test that ISA floor or produce tested CPU-dispatch/fallback variants with explicit flags. Run `ctest -N` to inspect registered tests and execute the affected CUDA backend tests explicitly on P100; `ctest` passing without a CUDA device is not a GPU test. Compile checks must verify the selected compiler/toolkit and actual sm_60 cubin, not just a cache variable or PTX compatibility.

CI plan:

- Every change: Windows/Linux CPU builds, pure planner tests, checkpoint logic tests, and formatting/build checks.
- CUDA compile jobs: pinned CUDA 12.x, sm_60 embedded-code inspection, CPU feature portability, and dependency packaging checks.
- Self-hosted P100 jobs on **both operating systems**: CUDA backend correctness, representative GGUF inference, concurrent request/cancellation tests, and benchmarks. Ordinary hosted build success is not GPU runtime validation.
- Release jobs: clean-machine archive extraction, DLL/shared-library resolution, CLI/server launch, local model inference, checksums, source provenance, and license inventory.

Deliver `llama-freetoken-p100-linux-x86_64.tar.gz` and `llama-freetoken-p100-windows-x64.zip`, plus a source archive containing the actual patched source, build instructions, benchmark report, and SHA-256 checksums. Include server, CLI, benchmark tools, required redistributable libraries, and notices; exclude model weights and NVIDIA drivers. Confirm redistribution terms for bundled third-party runtimes during packaging. Keep the original license files: llama.cpp/P100 patches use MIT and the inspected FreeToken repository uses Apache-2.0. Track any copied material and preserve its notices rather than assuming an MIT-only distribution.

## 9. Ordered milestones and effort

Estimates are engineering work ranges for an experienced C++/CUDA developer with access to both test platforms, not delivery commitments. Hardware access and unforeseen GGML integration work can extend them.

| Milestone | Work | Dependency | Rough effort | Exit gate |
| --- | --- | --- | --- | --- |
| M0 | Pinned baseline, builds, P100 precision audit, models/hardware inventory, benchmark harness | None | 3-5 days | Both native OS baselines run with recorded toolchains and results |
| M1 | Expert descriptors, host bank ownership, memory accounting, explicit FFN execution boundary | M0 | 1-2 weeks | Reference executor agrees with ordinary path on first model |
| M2 | Compact GPU cache, remapping, eager execution, transfer metrics | M1 | 1-2 weeks | Eviction/capacity tests pass and repeated routes reduce transfer |
| M3 | CPU full-expert executor, parallel merge, calibration, adaptive miss split | M2 | 2-3 weeks | All split cases correct; automatic policy evaluated against tuned baselines |
| M4 | Prefill double buffering with tile/fallback path | M2, integrated with M3 | 1-2 weeks | Measured overlap and bounded memory on both OSes |
| M5 | Multi-slot safety, allowlist expansion, CLI, initial experimental archives | M3-M4 | 1-2 weeks | First usable release passes native Linux/Windows smoke and stress tests |
| M6 | Semantic checkpoints including hybrid state and edits | M0, final validation after M5 | 1-2 weeks | Fresh-vs-resumed evaluation tests pass |
| M7 | Expert-cache elasticity and scheduler invalidation; graph research optional | M5 | 2-3 weeks | P100 eager resize/cancellation/OOM tests pass; any optional graph feature has separate gates |
| M8 | State-preserving KV redistribution, loading refinements if justified, release hardening | M6-M7 | 1-3 weeks | Complete strategy release and reproducible performance report |

Expected order of magnitude: roughly 2-3 months for a useful experimental release and 3-5 months for the broader scope, assuming no major redesign. Re-estimate after M1 establishes whether the explicit FFN execution boundary fits the scheduler cleanly. Keep each milestone reviewable and preserve a feature-off baseline throughout.

## 10. Validation and performance acceptance

### Correctness

Use existing `tests/test-backend-ops.cpp` for affected operations and add focused tests for the planner/cache/executor. Compare routing IDs, intermediate expert outputs, full logits, and teacher-forced perplexity against the same patched baseline and GGUF bytes. Calibrate dtype-specific absolute/relative tolerances from normal CPU/CUDA differences in M0; freeze thresholds before evaluating the new policy. Require exact discrete IDs/counts and no missing/duplicated contributions. Greedy text agreement is useful but insufficient because small valid numerical changes can alter later generated tokens.

For exact routing assertions, supply identical router inputs or replay recorded routing decisions into both executors. End-to-end floating-point differences can change a near-tied routing decision at a later layer; distinguish that numerical sensitivity from incorrect slot-ID remapping. Add adversarial near-tie cases and report them. Compare the P100-patched baseline with unpatched and higher-precision reference evaluation before accepting it as the integration oracle; an error shared by both optimized arms must not pass unnoticed.

Use synthetic expert banks with unique, easily distinguished outputs before loading large models: two layers sharing the same expert ID must never alias; two tokens using the same expert must retain both outputs; CPU/GPU partitions must cover every original assignment exactly once. Test packed ID/bias alignment, backend padding, and mapping retention after loader cleanup. Capacity smaller than top-k is supported only after the tiled executor proves its valid rectangular ID/activation layouts; until then, reject it during configuration instead of improvising sentinel IDs or dropping routes.

Cover zero/all/mixed misses, duplicate routes, quantization layouts, minimum cache, odd dimensions/padding, failed host registration, failed allocations, stale generations, concurrent slots, cancellation, long sessions, model reload, unsupported adapters, and dense-model fallback. Add hybrid sliding-window/recurrent checkpoint tests and speculative decoding only when those combinations are allowlisted. Require no NaNs, out-of-bounds access, stale cache reads, or silent state loss.

### Benchmark matrix

Compare three primary arms: unpatched pinned llama.cpp, the 31-patch P100 baseline, and each FreeToken milestone. For the new runtime include ablations: cache only, fixed hybrid split, adaptive split, prefill overlap, semantic checkpoints, and elasticity. Graph ablations apply only to the separately validated optional graph branch or a supported newer GPU; P100 production comparisons keep graphs disabled. Hold GGUF digest, prompt tokens, context, batch, thread count, sampler, GPU layers, and speculative settings constant. Disable MTP initially; test it separately so it cannot hide scheduling regressions.

Test a dense control, conventional MoE regression controls, Qwen3.6-35B-A3B and Gemma 4 26B-A4B. Include a model that fits VRAM and an MoE that exceeds VRAM but fits host RAM. Cover short/long cold prompts, warm repeated prompts, tool-edit conversations, decode lengths of at least 256 tokens, multiple cache budgets, and 1/2/4 request slots once supported. Include P100 12 GB and 16 GB variants if claiming both; otherwise label the tested SKU precisely.

Record CPU/RAM/NUMA, PCIe link width/speed, GPU clocks/temperature/power/ECC, OS, driver/toolkit/compiler, model and input hashes, source revisions, and every command. Run at least five measured repetitions after warmup, interleave A/B arms, and report distributions rather than a single best run. Separate cold startup, prefill, and steady decode. Use CUDA events and a profiler confirmed to support P100; do not assume current profiling tools support Pascal.

Proposed performance gates, to ratify after M0 measurements:

- Feature-off regression no worse than 3% beyond the measured noise band on the agreed suite.
- Adaptive mode's median decode latency within 5% of the best tested static policy per representative regime; otherwise fix selection or narrow the auto-mode allowlist.
- Aim for at least 15% better steady decode throughput on a representative memory-constrained MoE workload versus tuned P100-patched static offload. This is a target, not a promised outcome.
- Checkpoint reuse must reduce recomputed tokens and improve TTFT on the chosen edited-conversation suite without invalid state reuse.
- No unbounded pinned-RAM/VRAM growth; complete multi-hour pressure and request-cancellation runs on native Linux and Windows.

If measured gains are absent, retain diagnostics, keep the feature experimental/off by default, and revisit the cost model instead of advertising speedups.

## 11. Main risks and decisions still needed

| Risk or unknown | Resolution |
| --- | --- |
| Which P100 variant and target hosts? | Record card count, VRAM, CPU, RAM, PCIe topology, and Linux/Windows editions in M0 |
| Which model/quantization matters most? | Pick one concrete GGUF and retain hashes; do not promise all FreeToken-supported models |
| GGML's static graph conflicts with dynamic expert placement | Prove a complete-FFN boundary in M1 before implementing advanced policies |
| CPU+DMA contention wipes out hybrid gains | Calibrate concurrent workloads and retain cache-only/CPU fallback |
| Host readback costs dominate short decode steps | Measure first; move cache planning/ID handling onto device later only where beneficial |
| P100 patch precision or unsupported Windows behavior | Establish the baseline with numerical checks on both OSes before stacking features |
| Large MoE exceeds host RAM | Reject with a sizing explanation; SSD decode paging is outside initial scope |
| Expert caching competes with KV and prefill storage | Use exact byte accounting and conservative minimum budgets |
| Upstream changes invalidate the patch stack | Keep immutable pins, test patch replay, and rebase only as a separate measured change |
| Modern FreeToken kernels cannot target Pascal | Reuse GGML quantized kernels; port concepts, not hardware assumptions |

## 12. Immediate next work and completion checklist

The M1 CPU executor and M2 reusable cached CPU executor are connected to the model/context graph boundary, with generated GGUF comparisons, graph reuse, split-file ownership and failure tests. See M1_EXPERT_BOUNDARY.md and M2_CACHED_EXECUTOR.md. Qwen cache integration preserves its shared expert and hybrid state; Gemma preserves GeGLU, output scales, shared FFN and sliding/full attention. Retained cache telemetry is implemented for both. Next: implement a backend-neutral post-routing boundary for eager CUDA, explicit backend selection, bounded pinned staging/fallback and failure/cancellation cleanup. The existing CPU custom op is not a GPU execution boundary. Select and hash compatible pretrained Qwen and Gemma GGUFs that fit host RAM, run real-workload/API and CLI/server comparisons, and extend whole-context memory accounting. Baseline hardware measurements remain deferred. When the P100 is available, record baseline correctness/performance and validate CUDA allocation and execution before claiming support or speedups. Do not start with the full elastic-memory or graph-capture feature set.

Current planning task:

- [x] Download full llama.cpp source at the patch-compatible revision.
- [x] Download and apply the ordered P100 patches.
- [x] Download FreeToken for source-level reference.
- [x] Verify patch application and whitespace consistency.
- [x] Write a source-grounded implementation plan in this folder.
- [ ] Build and test on P100 Linux and native Windows (future implementation work).
- [ ] Implement and benchmark the planned strategies (future implementation work).
- [ ] Publish tested binaries for both platforms (future release work).

Additional primary references: [P100 patch source](https://github.com/shinbunbun/llama-cpp-p100-patches/tree/7c4cb31ca7bc47b25f3cb5b35d5b436d56bedcc8), [pinned llama.cpp](https://github.com/ggml-org/llama.cpp/tree/bb4caa7540188872173c44d161602d9271386413), [pinned FreeToken](https://github.com/FlashML-org/FreeToken/tree/af71ba43206e124f5ff6419b47ee36c6e9981078), [CUDA 12.9 Windows installation guide](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-installation-guide-microsoft-windows/index.html).
