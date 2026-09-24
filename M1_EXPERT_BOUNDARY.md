# M1 expert-bank and FFN boundary prototype

Status: CPU prototype, experimental Llama MoE loader/`llama_decode` integration, and context memory telemetry implemented on 2026-09-08. Qwen35MoE support and synthetic hybrid/shared-expert parity tests were added on 2026-09-09; see [Qwen verification](QWEN_CPU_OFFLOAD.md). P100 baseline measurements are deferred at the user's request. M1 remains open for a pretrained model acceptance run, complete host-overhead/working-set accounting, and device-allocation proof.

## Implemented

- `llama.cpp/src/llama-moe-offload.{h,cpp}` retains an explicit shared lifetime owner for gate/up/down tensors and their backing storage. It does not copy the full bank. The caller must keep that storage immutable and include both GGML contexts and buffers in the owner.
- Validation accepts separate, contiguous, canonical ordinary or mapped CPU banks with matching SwiGLU shapes. Supported projection types are F32, F16, Q4_0, and Q8_0, including different types across projections. Repacked buffers, tensor views, extra tensor traits, unsupported types, invalid shapes, and spans outside their backing buffer are rejected. Mapped buffers are identified by their buffer type, not their display name.
- Exact expert and host tensor-byte accounting uses GGML strides. Tile capacity uses the ordinary CPU backend's padded allocation size, not a guessed bytes-per-slot estimate. A budget below one complete expert is rejected.
- A deterministic route planner deduplicates expert copies, assigns valid compact IDs, and preserves every original token/top-k assignment. Active sets exceeding capacity use multiple tiles. No sentinel IDs or dropped contributions are used.
- A synchronous reference executor builds independent post-routing GGML graphs. Each graph receives only compact gate/up/down tensors, gathered activations, and compact routing IDs. The authoritative full banks are not sources of these graphs. Graph construction precedes scheduler allocation; the evaluation callback is not used.
- The executor runs all three projections with ordinary CPU kernels, scatters unweighted expert outputs back to original route positions, and aggregates in top-k order. It returns owned outputs plus host-bank bytes, peak tile allocation, peak scheduler arena, copied weight bytes, and tile count.

The executor accepts bias-free, separate-projection SwiGLU or GeGLU with routing weights applied after the FFN. Optional per-expert F32 output scales are applied before routing weights. Gemma4 uses GeGLU and these scales; see [Gemma CPU verification](GEMMA_CPU_OFFLOAD.md). Fused gate/up, biases, input/gate/up scales, adapters, other activations, clamping, and pre-FFN routing weights remain outside this API. Model adapters must reject unsupported configurations before calling it.

The module now compiles into the llama library when `GGML_CPU` is enabled. The ordinary path remains the default. CLI/server expose `--moe-cpu-tile-mib N`; this enables the CPU reference integration and does not enable GPU caching.

## Model integration

- Architecture contracts are `LLM_ARCH_LLAMA` with routed, separate SwiGLU experts (including Mixtral), and `LLM_ARCH_QWEN35MOE` with its gated shared expert preserved outside the executor. Dense Llama, Llama shared experts, unsupported expert types, scales, clamping, GPU layer placement, tensor overrides, enabled MTP and allocation simulation are rejected. Budgets smaller than one complete expert are rejected before loading tensor data.
- `LLM_ARCH_GEMMA4` MoE is also accepted with separate GeGLU experts and optional per-expert output scales. Its shared dense FFN and normalization remain outside the executor. The general scale rejection above has this explicit Gemma exception. The current integration suite has 51 configurations; the older counts below record earlier verification runs.
- The loader selects canonical CPU buffers. Existing mmap bookkeeping retains every loaded expert span, including spans in separate GGUF shards. Expert descriptors share ownership of the model's backing contexts, buffers and mappings; no second full-bank copy is created for descriptors.
- The graph uses an existing `MAP_CUSTOM3` operation with activation, expert ID and normalized routing-weight inputs. The ordinary router remains in the graph. The full expert banks have no tensor edges into this operation, and the original FFN is omitted before reservation. GGML owns the output tensor and downstream dependencies.
- Each graph result owns its callback state and shared bank references across reuse. The custom operation runs one outer task and invokes a separate CPU backend for the bounded tiles. It catches worker exceptions; the context synchronizes, checks errors, and returns a failed decode. The evaluation callback is used only by tests to observe outputs or inject failures.
- The initial context contract requires one sequence with operation and KV offload disabled. LoRA is rejected. GPU memory auto-fitting is skipped for this CPU-only mode.

Example, after selecting an allowlisted GGUF that fits host RAM:

```powershell
./build/ft-windows-cpu/bin/Release/llama-cli.exe -m 'D:/models/model.gguf' --moe-cpu-tile-mib 128 -ngl 0 --no-op-offload --no-kv-offload -np 1 -c 2048 -p 'Hello'
```

The model path is a placeholder. The budget must hold one complete gate/up/down expert bundle; 128 MiB is an example, not a size recommendation. Use the same options with `llama-server.exe`. The public C API adds `llama_model_params::moe_cpu_tile_bytes`, defaulting to zero; consumers must rebuild against this fork's header and library.

This executor allocates a backend and compact graphs during each FFN call. It is a correctness implementation, with no speedup claim. Cancellation is observed by the outer CPU backend between operations; a running expert operation completes before cancellation is observed. The budget limits tile weights, not total RAM. Host banks, activations, gathered route vectors, per-assignment expert outputs, kernel workspace, KV memory and the outer scheduler are additional allocations.

## Integration verification

### Context memory telemetry

`llama_moe_memory(ctx)` returns retained model, KV, outer scheduler, output-buffer and CPU kernel-workspace bytes, plus the host expert tensor bytes (already included in model bytes). These are allocation sizes, not physical residency or process RSS. Query only while the context is idle, without concurrent decode. A null context returns zeros.

Executor counters persist for the context lifetime across graph replacement, reuse and microbatch splitting. They report complete FFN calls, tiles, copied weight bytes, and observed peaks for tile allocations, inner scheduler arenas, vector capacity, GGML tensor/graph metadata arenas and CPU kernel workspace. `peak_temporary_bytes` measures the sum of these live allocations for a tile, including boundary input/route/weight vectors; it is not a sum of unrelated component peaks. Layers execute serially, so their temporary peaks are maximized, not added. Temporary storage is released after each FFN and is not included in retained backend bytes. Performance-counter reset does not reset these lifetime memory counters.

The usual performance report prints retained allocation categories, tracked temporary peak, tile peak/budget, completed FFNs/tiles and copied bytes when CPU tiling is enabled. The API exposes the detailed categories. The tile budget still bounds only compact expert weights.

Accounting covers completed FFNs, including completed work before a later operation fails. A failed FFN does not publish its partial allocation/copy counters. Allocator overhead, the route planner's temporary hash table, scheduler/backend bookkeeping, thread stacks, other model/context C++ metadata, and OS working-set/page-fault measurements remain untracked. Thus this is expanded model/context telemetry, not a complete process memory bound.

Memory regression checks cover all 17 generated model configurations and add 40 decode steps per configuration. They verify exact completed-FFN counts across microbatches, preserved counters across graph changes, bounded tile allocations, stable decode temporary peaks, retained model/KV sizes, fresh-context isolation, and no stale-work counting after cancellation or injected worker failure. Five focused CTests pass; see [memory tests](reports/m1-memory-tests.log).

`test-moe-integration` generates complete two-layer Llama MoE GGUF fixtures and loads them through the normal file loader. These contain deterministic synthetic weights and a no-vocabulary tokenizer; they are not pretrained model quality tests.

- 17 baseline/tiled model comparisons: F32/F16/Q4_0/Q8_0, copied/mmap loading, capacity-one and full-bank budgets, plus a seven-expert Q8_0 model split across two shards.
- Intermediate FFN outputs and logits agree within `2e-5 + 2e-5 * abs(reference)`. Tests cover prefill, repeated single-token decode, changed batch shapes and microbatch splitting. Additional runs omit the evaluation callback entirely.
- Reserved tiled graphs contain exactly one custom FFN per layer, no `MUL_MAT_ID`, and no full expert-bank source edges.
- Cancellation and injected non-finite routing weights return failure; clean retries agree with the baseline. Invalid budgets, GPU placement, allocation simulation, unsupported types, multiple sequences and context offload settings are rejected.
- Retaining a bank preserves its tensor data after the context and model are destroyed, including mapped split-file storage.
- Windows CPU CLI/server builds and five focused CTests pass: integration, synthetic expert offload, sampling, grammar parsing and quantization. See [build](reports/m1-integration-build-final.log) and [tests](reports/m1-integration-tests.log). CLI help confirms the new flag. CLI/server inference with a pretrained model has not been tested.

## Verification

Windows CPU build: MSVC using the existing `ft-windows-cpu` preset. The test target explicitly enables MSVC exception unwinding so failure paths release owned resources.

- 156 tiled executions compare individual expert outputs and weighted FFN results with an ordinary full-bank GGML `MUL_MAT_ID` graph.
- Cases cover F32/F16/Q4_0/Q8_0, mixed Q4_0/Q8_0, decode and multiple tokens, top-1/top-3, capacities 1/2/7, more active experts than slots, repeated IDs, zero route weights, non-square matrices, and allocation padding.
- F32 square cases additionally compare against an independent scalar SwiGLU calculation.
- Tests verify exact route preservation, unique expert-copy byte counts, bounded tile allocation, ownership survival/release, invalid inputs, and integer overflow rejection.
- Increasing the full host bank from 7 to 19 experts leaves the capacity-one weight allocation unchanged for every tested uniform type.
- The synthetic CPU comparison threshold is `2e-5 + 2e-5 * abs(reference)`. This is a unit-test threshold, not a calibrated P100 or full-model tolerance.

Build and run from the workspace:

```powershell
cd llama.cpp
cmake --preset ft-windows-cpu
cmake --build --preset ft-windows-cpu --target test-moe-offload
cd ..
ctest --test-dir build/ft-windows-cpu -C Release --output-on-failure -R '^test-moe-offload$'
```

Logs: [build](reports/m1-cpu-build.log), [focused CTests](reports/m1-cpu-tests.log). The focused regression selection also runs sampling, grammar parsing, and quantization tests. Linux and CUDA execution have not been tested.

## Remaining M1 integration

1. Select and hash a pretrained GGUF within the Llama MoE/type allowlist and available host RAM. Validate real-workload intermediate outputs and logits against the feature-off patched baseline. The generated full-model tests do not close this acceptance gate.
2. Extend lifecycle coverage to sustained workloads and native Linux. CPU graph structure does not prove bounded CUDA allocations or P100 execution.
3. Validate actual CLI/server inference with that pretrained model, including tokenization and request handling.
4. Finish untracked model/context host overhead and OS working-set/page-fault accounting. Tensor/graph metadata arenas, host route/input/output vectors, backend kernel workspace and retained backend allocations now have telemetry as described above. Future pinned staging will need separate accounting. The configured budget bounds tile weights only.

There is no persistent inference cache, CUDA executor, pinned staging, CPU/GPU overlap, or measured speedup in this slice. In particular, CPU allocation tests do not prove device allocation or P100 kernel compatibility. M2 now has a separate tested cache state module for identity, slot generations and leases; backend residency and transfer events remain future work. See [M2_CACHE_STATE.md](M2_CACHE_STATE.md).

The historical `P100_BASELINE.patch` and `SOURCE_MANIFEST.json` remain unchanged. Back up the new source/test files and the changed test CMake file separately; neither a Git archive of HEAD nor the historical baseline patch includes this work.
