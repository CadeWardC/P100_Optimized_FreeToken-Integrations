# M2 persistent CPU cached executor

Implemented 2026-09-09. The reusable cached FFN executor is connected to the existing inference boundary through a context-owned cache. Qwen3.6-35B-A3B and Gemma 4 26B-A4B are equal primary integration targets; see [target models](TARGET_MODELS.md). This closes the CPU integration step after [backend storage](M2_CACHE_STORAGE.md), not the CUDA/P100 milestone.

## Usage

CLI and server accept `--moe-cpu-cache-mib N`. For a compatible local model:

```powershell
./build/ft-windows-cpu/bin/Release/llama-cli.exe -m 'D:/models/compatible.gguf' -ngl 0 --no-op-offload --no-kv-offload -np 1 --moe-cpu-cache-mib 64
```

The model path is a placeholder. The option automatically enables the existing canonical host-bank loader. It does not require `--moe-cpu-tile-mib`; if both are supplied, the tile budget remains the loader/reference-path budget, while cached contexts use the cache budget. Automatic model fitting is bypassed as it is for the existing CPU tile mode. Existing architecture/type, one-sequence, CPU-placement, no-LoRA and no-MTP restrictions apply.

C API callers set `llama_model_params.moe_cpu_tile_bytes` to enable host banks and set `llama_context_params.moe_cpu_cache_bytes` to a nonzero cache budget. A cached context requires those banks and rejects a budget smaller than one complete padded expert. Zero cache bytes preserves the previous inference path. Rebuild consumers against the changed public parameter/telemetry structs.

The byte budget applies to one cache shared across all layers in the context, not to each layer separately. All layers must have identical gate/up/down shapes and types; each projection can have a different supported type. The complete authoritative expert pool remains in host RAM, and the CPU cache adds another bounded allocation. This is a correctness/integration prototype, not a claim of lower host RAM use or faster CPU inference.

## Execution and ownership

`llama_moe_cached_executor` retains the layer banks, a dedicated CPU backend and the compact storage wrapper. Both reference and cached execution use the same FFN graph/merge implementation. No full expert banks become inputs to the outer inference graph or to cached compute graphs. The Qwen shared expert, hybrid attention and recurrent state stay in the ordinary model graph.

Routing is fully validated before acquiring cache tickets. The executor deduplicates experts and probes all routed keys, pinning resident hits without reserving misses. It processes those hits first, so a miss early in the router's order cannot evict a hit needed by a later tile. Active sets larger than capacity are tiled. Each assignment gets a valid cache slot ID, then its output is scattered back and merged in the original routing order. Duplicate assignments, original-ID scale lookup, SwiGLU and GeGLU are preserved.

Keys distinguish layers and original experts. Model/adapter identity is fixed within each private context cache; banks are immutable and adapters are rejected. Ticket guards release pinned slots on exception. All execution is serialized and synchronous at the existing CPU custom-op boundary. Cancellation is observed by the outer graph; this does not add interruption inside a running FFN tile. Cache contents survive graph reuse and attention/recurrent state restore because they contain only immutable weights. Fresh contexts start with empty caches and counters.

## Accounting

`llama_moe_memory(ctx)` and performance logs now expose:

- Retained cache weight bytes versus configured budget, slot/tensor metadata bytes, and the dedicated CPU backend's retained workspace.
- Lifetime cache hits, loads, evictions, submitted canonical copy bytes and host load submission/wait microseconds.
- Existing completed-FFN/tile/copy counts and temporary peaks for routing vectors, packed activations, graph metadata and compute buffers.

Retained cache allocations are separate from temporary peaks and the existing model/KV/outer-compute categories. A cached call has no temporary expert weight tile. Cache copy counters include transfers from calls that later fail; the older completed-FFN counters retain their previous meaning. General allocator overhead, route hash maps, backend bookkeeping and process RSS remain outside these measurements. No pinned staging allocation or PCIe traffic measurement is introduced by this CPU path.

## Verification

Windows CPU builds of CLI, server and the six focused test targets pass. All six CTests pass. [Build log](reports/m2-executor-build.log), [test log](reports/m2-executor-tests.log).

The integration suite now performs 107 generated-model comparisons, including 17 cached Qwen and 28 cached Gemma cases: F32/F16/Q4_0/Q8_0, ordinary/mapped loading, one-slot/fitting caches and split-file ownership. Gemma split models select eight experts and run with one-slot, two-slot and fitting caches across all four types. Eight cached Llama cases remain regression controls. Checks cover logits, routed/shared/combined outputs, recurrent or KV state restore, graph reuse, sustained bounded allocations, outer-graph cancellation, invalid-route recovery and fresh-context isolation. CLI/server argument conversion enables both model host banks and context caching. Expanded coverage: [build](reports/dual-target-build.log), [tests](reports/dual-target-tests.log).

The executor tests compare cached and uncached FFNs across the existing type/layout matrix, including mixed quantization, odd dimensions, duplicate routes, GELU/output scales and capacities below top-k. An adversarial two-slot case warms experts 1 and 2 before requesting `[0, 1, 2]`: only expert 0 loads, despite appearing first and requiring another tile. Separate checks cover cross-layer isolation, invalid-route recovery, incompatible layouts and zero additional copies on warm execution. Existing storage/event/failure and 30,000-step cache-state tests remain in the suite.

CLI/server help smoke checks expose the new option, and a zero cache budget is rejected by the CLI. These are not pretrained CLI/server inference tests.

```powershell
cmake --build build/ft-windows-cpu --config Release --target test-moe-cache test-moe-offload test-moe-integration test-sampling test-grammar-parser test-quantize-fns llama-cli llama-server --parallel 4
ctest --test-dir build/ft-windows-cpu -C Release --output-on-failure -R '^(test-moe-cache|test-moe-offload|test-moe-integration|test-sampling|test-grammar-parser|test-quantize-fns)$'
```

## Next work

Implement and validate a backend-neutral post-routing execution boundary for eager CUDA execution, with explicit backend selection, bounded pinned staging/fallback, compute-buffer accounting and failure/cancellation cleanup. The current CPU custom-op implementation is not a GPU execution boundary. Preserve the existing CPU controls while adding that path; do not enable CUDA merely by substituting a backend in the current callback.

Then validate actual CUDA transfers, padding, events, kernel outputs and allocation limits for both Qwen 35B and Gemma 4 26B before enabling them on P100. Exact pretrained GGUF selection/hashing, native Linux checks, P100 Windows/Linux validation and performance measurements remain open. No hardware support, overlap or speedup is claimed. Preserve all uncommitted implementation files with the existing baseline artifacts.
