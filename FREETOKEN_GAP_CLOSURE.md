# FreeToken runtime gap work

Follow-up: [the installed FreeToken comparison](FREETOKEN_REAL_COMPARISON.md) measures the actual Windows engine on the same Gemma file. It identifies larger execution differences and cache-budget sensitivity; the short raw-prompt benchmarks here should not be treated as conversational throughput or FreeToken parity evidence.

Implemented in the local Windows CUDA development tree on 2026-09-09. These changes narrow the differences from the checked-out FreeToken reference; they do not establish full FreeToken parity or P100 acceptance. Existing uncommitted changes were retained.

## Scheduling

`--moe-cpu-miss-percent -1` now calibrates achieved CPU expert and host-to-device transfer bandwidth concurrently. The CPU probe runs the actual one-expert executor while the transfer probe cycles canonical expert slices across banks. Both include their real executor/staging overhead. Calibration uses the loaded layout, selected GPU, staging configuration and current CPU thread count. The CPU miss fraction is `CPU_BW / (CPU_BW + H2D_BW)`. Hits stay on GPU. The policy can choose zero or all CPU misses; it no longer forces both branches to run. Prefill uses GPU experts rather than applying a decode bandwidth model to repeated token assignments.

The first execution calibrates, and a thread-count change recalibrates. Calibration clears cache residency and has startup cost. Probe transfers are excluded from inference cache counters; achieved bandwidth is exposed through `llama_moe_memory()` and performance logging. Fixed percentages remain available. This is an in-process effective-bandwidth calibration, not FreeToken's persisted hardware profile or a claim of optimal scheduling.

## Transfer/compute overlap and prefill

`--moe-pipeline` gives weight storage a separate backend/transfer stream. The cache budget holds two expert tiles; each tile is at most half the slot capacity. After submitting a compute graph asynchronously, the scheduler loads the next routed tile into other leased slots. Transfers complete before those slots are consumed. Graphs drain before current leases or graph allocations are released, including cancellation and exceptions.

This implements the bounded tile fallback for double-buffered prefill described in the implementation plan. It also works during decode. It does not prefetch an entire next layer. A one-slot cache or a fully pinned cache safely falls back to serialized loading. Original expert IDs, assignment order, route weights and scales are preserved. Weight capacity and the pinned staging budget do not grow with the number of tiles.

`prefetched_experts` counts successful lookahead loads. Host compute time includes waiting while prefetch runs, so stage timings overlap and must not be summed as exclusive durations.

Nsight Systems captured **77 full weight H2D chunks overlapping GPU kernels on different streams**, with **117.54 microseconds of union overlap** in the synthetic executor test. This proves device-side overlap for that workload, not a throughput gain. The tiny 4093-byte test chunks are deliberately a staging stress case. See `reports/gap-pipeline-overlap.json`, the `.nsys-rep`/SQLite trace, and `scripts/analyze_moe_overlap.py`.

## Semantic checkpoints

The server's `--semantic-checkpoints` option extends its existing checkpoint system to chat-template message starts, including assistant and tool messages. An anchor at token count `p` contains exactly the consumed prefix `[0,p)`; queued batch tokens are excluded. The batch is split at message boundaries.

Snapshots use `LLAMA_STATE_SEQ_FLAGS_NONE`, retaining full sequence KV and recurrent/SWA state rather than depending on independently retained attention KV. Reuse uses the server's exact common token prefix. Deserialization size and the restored final position are checked; failure falls back to prompt recomputation. Both the existing checkpoint count and `--semantic-checkpoint-mib` (default 512 MiB per slot) limit storage. Oversized snapshots are skipped before allocating them.

Supported configuration: host-bank MoE, `-ngl 0 --no-op-offload --no-kv-offload`, compatibility kernels, one sequence, positive checkpoint count, no draft model. GPU experts and pipelining are supported. GPU-resident attention is explicitly rejected in semantic mode because testing found a full-sequence replay discrepancy. Thinking-delimiter anchors, draft/MTP state and multimodal position layouts are not added here. This is exact-prefix caching, not approximate semantic similarity caching.

The real Gemma server test saved 15 snapshots, restored an edited chat at token 32, and matched a fresh slot's response. See `reports/gap-semantic-server-v2/results.json` and `scripts/validate_semantic_checkpoints.py`.

## Coordinated runtime resizing

`llama_moe_reconfigure(&ctx, params)` replaces an **empty**, single-sequence host-bank context. It rebuilds expert storage, KV/SWA/recurrent memory, graph arenas and the scheduler together. It rejects populated sequence state. Allocation failure leaves the old context intact; success replaces the pointer and destroys the old context. The model remains caller-owned.

The caller must have exclusive access, invalidate external prefix/semantic checkpoints after success, and reattach application samplers, adapters or threadpools as needed. The old and new allocations coexist during construction, so this transaction needs temporary memory headroom. It is a library API, not an HTTP endpoint, automatic memory-pressure policy, or in-place live-session resize.

The internal executor also supports transactional expert-only `resize(budget)`. Tests cover failure preservation, growth, return to the original configuration, rejected populated contexts, new KV capacity and new expert capacity.

## GPU placement and native kernels

`-ngl` can now place attention and shared layers on GPU while the loader keeps routed expert weights and their scales in canonical CPU buffers. GPU outer layers require a GPU expert cache. The one-sequence restriction remains.

`--moe-fast` selects native GPU projection/activation behavior, allowing backend fusion and omitting the compatibility GELU table and forced projection precision. This is an opt-in numerical mode. It can change intermediate values, routing and generated tokens; the old compatibility mode remains the default.

The native-placement integration matrix compares against an ordinary model with the same GPU placement. Decoded logits retain the existing `0.005 + 0.05 * abs(reference)` tolerance. Intermediate activations are recorded as diagnostics in this mode (maximum observed absolute difference 0.0127391815), not claimed to match CPU arithmetic. Same-context state replay is tested. Cross-mode checkpoint interchange and GPU full-sequence semantic snapshots are excluded. CPU attention plus compatibility kernels retains its original stricter intermediate checks.

## Commands

```powershell
# Calibrated decode, bounded prefill overlap, full semantic snapshots
./build/ft-windows-cuda-dev/bin/llama-server.exe -m models/gemma-4-26B_q4_0-it.gguf `
  -ngl 0 --no-op-offload --no-kv-offload -np 1 `
  --moe-gpu-cache-mib 1024 --moe-device CUDA0 `
  --moe-cpu-miss-percent -1 --moe-pipeline --semantic-checkpoints

# Experimental native GPU attention/shared layers and experts
./build/ft-windows-cuda-dev/bin/llama-server.exe -m models/gemma-4-26B_q4_0-it.gguf `
  -ngl 99 -np 1 --moe-gpu-cache-mib 1024 --moe-device CUDA0 --moe-pipeline --moe-fast
```

The public C structs changed; rebuild library consumers. The CUDA development build targets SM60 and SM89 with CUDA 12.6. Hardware execution here is on an RTX 4060 Laptop GPU. Native Linux/P100, pretrained Qwen, sustained memory pressure and broader workload acceptance remain open.

## Verification artifacts

- `reports/gap-build-final-verified.log`: CUDA build, CLI/server and test targets.
- `reports/gap-cpu.log`: focused CPU CTests.
- `reports/gap-device.log`: executor parity, pipelined tiles, calibration, cancellation, warm reuse and resize.
- `reports/gap-integration.log`, `reports/gap-calibrated-integration.log`: synthetic compatibility model runs.
- `reports/gap-gpu-native-integration.log`: native GPU placement/model-output checks.
- `reports/gap-memcheck.log`: Compute Sanitizer, zero reported errors.
- `reports/gap-semantic-server-v2/`: pretrained Gemma edited/fresh chat comparison.
- `reports/gap-bench-*/`: short matched performance runs; see each report's exact commands and scope.

## Short Gemma throughput comparison

One warm-up and two measured 16-token requests per configuration, with a 1024 MiB expert cache, eight CPU threads, the pinned local Gemma GGUF, no prompt KV reuse, and the same prompt/sampler. These are development measurements on this laptop, not a quality or production benchmark.

| Configuration | Median decode tokens/s | Change from serialized | Same tokens as serialized |
| --- | ---: | ---: | --- |
| CPU attention, compatible GPU experts, serialized | 2.904 | baseline | yes |
| CPU attention, compatible GPU experts, pipeline | 2.898 | -0.2% | yes |
| CPU attention, calibrated hybrid, pipeline | 2.118 | -27.1% | yes |
| Native GPU attention/shared layers and experts, pipeline | 8.474 | +191.9% | **no** |

Native placement improved throughput substantially but changed generated tokens. It is not an equivalent-output speedup claim, and pretrained native-mode quality/logit acceptance remains open. Hybrid calibration still loses to GPU-only experts on this workload. Keep GPU-only compatible execution as the default; choose native mode explicitly when accepting its numerical tradeoff. The benchmark helper now accepts `--pipeline`, `--fast`, `--gpu-layers` and `--policies` for reproducible ablations. See `reports/gap-benchmark-summary.json` and the full per-run reports.
