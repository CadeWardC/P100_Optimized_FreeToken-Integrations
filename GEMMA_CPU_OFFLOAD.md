# Gemma 4 26B-A4B CPU offloading

Implemented on 2026-09-09 for the `gemma4` MoE text graph. The [official target configuration](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json) uses GELU, eight selected experts, sliding/full attention, and no recurrent layers or shared KV layers. Gemma 4 26B-A4B and Qwen3.6-35B-A3B are equal primary P100 targets; see [TARGET_MODELS.md](TARGET_MODELS.md).

Gemma uses `--moe-cpu-cache-mib` for persistent context-owned CPU caching, including GeGLU and original-expert-ID output scales. Its expanded cache checks cover one-slot caches and split-file top-8 routing across every allowlisted weight type. See [cached executor](M2_CACHED_EXECUTOR.md).

## Behavior

The bounded executor now selects SwiGLU or GeGLU explicitly. Gemma selects GeGLU; existing Llama and Qwen paths retain SwiGLU. Gemma's per-expert F32 down-projection scales are retained with the host bank and applied after each expert's down projection, before routing weights and top-k summation. The existing router normalization/scaling, shared dense FFN, branch normalization, combined FFN normalization, attention and logit softcap remain ordinary llama.cpp graph operations.

Only routed gate/up/down projection weights are copied into bounded tiles. Per-expert output scalars are read directly from the retained CPU bank without creating a full-bank graph input or extra per-call allocation. Host-bank byte accounting includes those scalars. Expert-copy counters and the tile weight budget cover the three copied projection matrices. Shared weights, complete host banks, KV storage and other execution overhead remain outside the tile budget.

Supported routed projection formats are F32, F16, Q4_0 and Q8_0, with separate or fused gate/up tensors. Fused support was added after inspecting the selected pretrained Gemma GGUF and adds 28 generated-model comparisons (135 total). See [M2_CUDA_EXECUTOR.md](M2_CUDA_EXECUTOR.md). K/I quantizations, projection input/gate/up scales, adapters, enabled MTP, GPU model placement and multiple sequences remain unsupported. Output scales must be finite F32 scalars, one per expert; malformed scales fail model loading. The ordinary path remains the default.

## Verification

`test-moe-integration` now runs 107 synthetic model configurations: 25 Llama controls, 34 Qwen and 48 Gemma. Gemma includes 20 uncached/tiled comparisons and 28 persistent-cache comparisons. It covers all four expert formats, copied/mapped loading, one-slot/fitting caches, plus split nine-expert fixtures selecting eight experts per token with one-slot, two-slot and fitting caches across all four formats.

The two-layer Gemma fixtures exercise sliding-window and full attention with different head dimensions, omitted V projection (K=V), router scales, nonuniform output scales, nonzero shared FFNs, layer output scales and final logit softcapping.

- Prefill, repeated decode, microbatch splitting, shape changes and sustained decoding beyond the eight-token synthetic sliding window compare against ordinary execution with tiling disabled in this same fork.
- Routed outputs before normalization, shared FFN outputs, normalized routed outputs, merged FFNs and logits agree within `2e-5 + 2e-5 * abs(reference)`.
- State save/restore reproduces continuation, including restoration of an ordinary-path state into a tiled context. Fresh contexts also compare without observation callbacks. Gemma has KV-cache state rather than Qwen-style recurrent state.
- Graph checks require one custom routed operation per layer, no full-bank `MUL_MAT_ID` and no routed weight-bank graph inputs. Existing bounded allocation, memory accounting, cancellation and failure-recovery tests remain active.
- Lower-level executor tests cover GELU with positive, negative and zero expert scales, repeated routes, capacities 1/2/7, one/five tokens, top-1/top-3, all four projection types and 7/19 host experts. They compare individual expert outputs and aggregate outputs with an ordinary GGML graph. F32 tests also use an independent scalar GELU formula with the CPU kernel's F16 table rounding.
- Invalid activation values, scale shape, non-finite scales and out-of-range scale indices are rejected.

Current dual-target evidence: [Windows CPU test build](reports/dual-target-build.log) and [focused CTests](reports/dual-target-tests.log). Original adapter evidence: [CPU build](reports/gemma-cpu-build.log), [tests](reports/gemma-cpu-tests.log), and [executor tests](reports/gemma-executor-tests.log). CLI/server already expose the shared cache option.

## Remaining acceptance

These are deterministic synthetic weights, not a pretrained 26B GGUF. No pretrained CLI/server request, multimodal input, Linux build, CUDA execution or P100 run was verified. An exact GGUF and its per-tensor types must be selected before pretrained acceptance. No performance improvement or GPU cache support is claimed.
