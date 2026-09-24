# Implementing the installed FreeToken execution advantages

Source audit: September 9, 2026. Scope: the local modified llama.cpp and installed Windows FreeToken, Gemma 4 26B Q4_0 on the RTX 4060 laptop. This is an implementation design, not completed runtime work or a predicted speedup.

Implementation update: persistent cached-GPU workspaces, the single-token GPU-resident boundary and verified CUDA replay are now implemented. See [GPU-resident execution and measured results](GPU_RESIDENT_EXPERTS.md) and the earlier [persistent-workspace results](PERSISTENT_EXPERT_WORKSPACE.md). Multi-token device execution and the later CPU/transfer/kernel stages remain pending. The original design below is retained.

## Target and baseline

The saved matched chat benchmark reports 24.42 tokens/s for installed FreeToken and 10.86 for this project, with approximately 3 GiB of expert cache. Use `reports/freetoken-installed-comparison/chat-summary.json` and `scripts/benchmark_openai_stream.py` as the starting evidence. Repeat the baseline before changing execution; do not mix raw repetitive completions with chat results.

Keep model hash, prompt/template, thinking setting, requested output length, context, cache budget, thread count and measurement method fixed for each A/B pair. Run engines separately. Separate startup, cold prompt, warm prefix, first-text latency and decode. Add several coherent prompts and longer generations to the short existing workload. Retain generated text, tokens, usage and launch settings.

## What the code establishes

- `src/llama-moe-offload.cpp::moe_compute` reads inputs, route IDs and routing weights to host vectors, invokes a vector-based executor, then writes a host result to the destination.
- `execute_ffn` creates graph contexts and allocators inside the tile/batch loop. The uncached CPU branch also creates a backend and compact weight buffers. GPU batches are capped at eight assignments.
- The cached executor owns stable weight storage already. `ggml_moe_cache_storage::tensor` returns the full projection tensor; route values choose physical cache slots. This is a useful foundation for persistent graphs: expert selection can change as tensor contents without changing graph structure.
- `resize` replaces cache storage. Persistent graphs must never retain pointers into the replaced cache.
- The scheduler's external node callback in `src/llama-context.cpp` is the integration point for expert execution. Removing readbacks only inside a CUDA kernel will not remove this boundary.
- `ggml-moe-cache.cpp::acquire` drains after each staging chunk and after the expert load. Existing transfer/compute overlap therefore still involves host waits.
- The actual CUDA build cache has CPU AVX/AVX2 and CUDA graphs disabled.
- CUDA graph compatibility rejects `MUL_MAT_ID` paths that need synchronization. The local backend disables graph replay below Volta, so its existing graph path is unavailable on P100.
- Correction to the earlier report: the CUDA optimizer already recognizes `MUL_MAT_ID, MUL_MAT_ID, GLU` and can pass both projections to a fused MMVQ kernel. Separate graph operations do not prove separate kernel launches in native mode. Instrument actual dispatch before proposing a new fusion kernel.

## 1. Establish useful profiling and an optimized laptop build

Files: `scripts/build-cuda-dev.cmd`, benchmark scripts, executor metrics, CUDA dispatch logging.

Create a separate laptop build directory/preset with supported CPU vector instructions enabled. Keep the portable/P100 development build intact. First change CPU flags alone; separately test CUDA graphs and flash attention. Confirm actual CMake cache values and runtime CPU feature selection. Do not assume the existing ggml CPU path becomes FreeToken's W4A8 implementation merely by enabling instructions.

Record graph construction/allocation counts, retained compute bytes, host boundary copy bytes, CPU worker launches, cache misses, CUDA capture/replay counts and real fused kernel dispatches. Use a short full-engine Nsight trace to attribute GPU idle gaps, transfers and kernel launches. Existing host timings overlap; do not sum them as exclusive stage costs.

Acceptance: reproducible baseline and a trace identifying the actual critical path. Keep only build-setting changes that pass model checks and improve measured workloads.

## 2. Persist expert compute state

Files: `src/llama-moe-offload.cpp` and `.h`; `tests/test-moe-offload.cpp` and integration tests.

Add an internal workspace owned by `llama_moe_cached_executor::impl`: context, input/route tensors, FFN graph, allocator, staging vectors and allocated-byte accounting. Reuse the existing persistent backend and cache tensors. `execute_ffn` should obtain a workspace instead of creating it inside every batch.

Start with cached GPU execution and exact existing batch shapes. Bound the workspace inventory, for example one replaceable workspace initially; do not retain one arena per layer. Subsequent shape caching must fit one aggregate compute budget, including the compatibility GELU table. The uniform bank-layout invariant permits reuse across layers when only route slots and input data change.

Workspace identity must cover backend/cache generation, projection types and geometry, assignment shape, activation and numerical mode. Keep original expert scales in dynamic data. Update input and route tensor contents on each execution; never freeze the routing decision into the graph. Changing cache contents does not invalidate pointers, but replacing the cache does.

On resize, construct the replacement first, then drain outstanding compute, invalidate old workspaces/captures and swap storage. Allocation failure preserves the old usable executor. Destruction and cancellation must drain before releasing referenced memory or cache leases. Count retained arenas as retained memory, not just temporary peaks.

Acceptance: identical same-mode outputs with changing routes, cache eviction, repeated layers, varying batch shapes, cancellation and resize; no new graph/arena allocations in warmed fixed-shape decode. Existing compute-budget failure tests must still pass.

## 3. Keep activations and reduction on the device

Files: `src/llama-moe-offload.cpp`, `.h`, `src/llama-context.cpp`, scheduler integration in `ggml/src/ggml-backend.cpp`; CUDA helpers only if existing ggml operations are insufficient.

Add a tensor-based device execution entry point, conceptually `execute_device(layer, input, ids, weights, dst, ...)`. Retain the vector entry point as a reference/CPU fallback. Start with one-token, GPU-only expert execution.

Initially copy only the small route-ID array to host for cache lookup and miss loading. Keep input activations and routing weights on GPU. Upload physical slot IDs and original-expert scales into persistent tensors. Gather/repeat activations on device, execute experts, then apply scales and routing weights and reduce in original assignment order on device. Write directly to the outer graph destination, or use a device-to-device copy where stable workspace pointers are needed.

Do not use floating-point atomics for the first reduction implementation. Store contributions by original assignment index and reduce deterministically across top-k; tile order must not change arithmetic order. For multiple tiles, keep a bounded contribution workspace and enforce token chunk limits from the compute budget. Avoid full expert-output readbacks except in explicit diagnostic mode.

Inspect scheduler placement of the custom node: removing host vectors is insufficient if the scheduler still allocates the destination on CPU. Establish a GPU-owned node destination and explicit dependencies between outer-graph compute, the expert backend and the resumed outer graph. Initially preserve a safe boundary wait; replace it with events only after ordering is tested.

Acceptance: warmed GPU-only expert execution has zero activation/output D2H transfers, apart from explicit diagnostics; route control traffic remains allowed. Verify scales use original expert IDs, including when physical slots are reused. Preserve Gemma shared FFN and Qwen shared-expert behavior outside the routed boundary.

## 4. Capture stable compute segments

Files: laptop build preset, executor workspace lifecycle, `ggml/src/ggml-cuda/ggml-cuda.cu` only where required.

Enable the existing CUDA graph implementation for the RTX 4060 build after step 2. Start by capturing persistent expert compute segments, with dynamic input and slot tensors updated outside capture. Keep cache lookup, allocation, transfers and CPU control outside captured segments. Audit the selected `MUL_MAT_ID` kernel through `ggml_cuda_mul_mat_id_needs_sync`; an unsupported fallback must remain eager.

Measure actual captures/replays and warmup resets, not the outer ggml graph-reuse counter. Stable graph keys, tensor addresses and graph shapes must survive across tokens. Cache resize must also retire captured executables that reference old storage; audit backend graph ownership before implementing invalidation.

Full-token capture is a later project: the current host-controlled cache misses and custom scheduler boundary prevent treating it as a build-flag change. Do not launch CUDA API work from a captured host callback as a shortcut. A future design needs explicit host work completion and preallocated transfer/compute dependencies.

Acceptance: trace confirms replay with changing routes and correct cache misses. Test eager versus replay under the same numerical mode. P100 retains the eager optimized path; do not remove the architecture guard without separate hardware validation.

## 5. Replace the temporary CPU expert path

Files: executor implementation and existing ggml CPU backend/threadpool interfaces.

Own a persistent CPU backend, worker/threadpool and reusable scratch storage. Replace per-call `std::async` scheduling with a bounded persistent worker. First reuse compact buffers; then investigate canonical-bank views or direct selected-expert dot products to avoid copying every selected expert into temporary storage. Any direct-bank graph must remain explicitly on CPU so the scheduler cannot migrate a full expert bank to GPU.

Use the optimized ggml Q4_0 CPU path first. A specialized W4A8 kernel is a separate numerical/performance change: validate activation quantization, scales, accumulation and GeGLU before enabling it. Installed FreeToken logs describe a specialized CPU path, but some implementation is compiled; exact replication is not established by this audit.

Add an explicit CPU-layer placement policy and benchmark per-layer execution/transfer costs. FreeToken's 15 CPU layers demonstrate a viable arrangement, not an optimum to copy blindly. Choose placement only after the CPU executor is optimized, using real contention and pinned-memory limits. Recalibrate hybrid policy against the new executor; preserve GPU-only as a measured control.

Acceptance: no repeated worker/backend creation, bounded scratch memory, cancellation and shutdown join correctly, same-mode numerical checks pass, and actual mixed placement improves end-to-end chat before becoming a default.

## 6. Tune kernels, transfers and memory after the boundary is fixed

Kernel work: first verify whether current native Q4_0 + GeGLU dispatch fuses. If it does, benchmark that against packed gate/up before changing cache layout. If it does not, identify the precise shape/eligibility constraint and extend existing MMVQ machinery where possible. Remove the fixed eight-assignment limit only with a budgeted shape policy and measured kernel support. Compatibility mode must retain its validated arithmetic.

Transfer work: use a bounded ring of pinned staging buffers and completion events instead of synchronizing each chunk. Cache entries need a loading state and a completion dependency; consumers wait on that event before compute, and eviction cannot reuse a leased or in-flight slot. The staging slot cannot be overwritten before its DMA completes. Test failures and cancellation with the delayed backend and Compute Sanitizer.

Memory work: allocate outer-model/KV resources and account for persistent arenas, staging, captures and safety headroom before assigning the remaining expert budget. CUDA graph warmup may allocate additional storage; verify headroom after warmup. Explicit budgets remain available. Live resizing remains an empty-context transaction until separate state-migration support exists.

Attention work: benchmark the existing flash-attention implementation with Gemma sliding/full attention and the intended KV types. Do not infer that matching FreeToken's Triton backend is necessary. Keep unsupported checkpoint/attention combinations rejected until replay tests pass.

Acceptance: separate A/B results for each change, long enough to expose cache churn and memory pressure. No single feature gets credit for the full observed 2.25x gap without an ablation.

## Delivery sequence and correctness gates

First implementation slice: profiling counters plus persistent cached-GPU workspaces, with unchanged arithmetic and routing. This removes a concrete source of repeated work without requiring a new kernel or scheduler redesign.

Then: device-resident boundary; verified graph replay; persistent CPU executor and measured placement; remaining kernel/transfer/memory tuning. CPU build and flash-attention experiments can be measured independently of the structural work.

For each slice run the relevant existing executor and synthetic integration tests, a pretrained Gemma same-mode regression, and the matched chat benchmark. Structural changes should preserve the current mode's behavior. Changes to quantization, fusion or reduction need fixed-prefix logit checks as well as coherent generated-output checks; token identity alone is neither a full quality test nor an appropriate cross-engine requirement. Retain compatibility mode and test synthetic Qwen regressions because the executor is shared.

The performance target is to close the measured 10.86 versus 24.42 tokens/s gap. This audit identifies implementable mechanisms and dependencies; it does not establish that any proposed slice will reach a particular throughput. No runtime code was changed and no new benchmark was run for this design.
