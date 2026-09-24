# M2 backend expert-cache storage

Implemented 2026-09-09 as the next step after the uniform cache state module. The new `ggml_moe_cache_storage` in `llama.cpp/ggml/src/ggml-moe-cache.{h,cpp}` owns compact backend buffers and uses the existing cache tickets. It is compiled into `ggml-base`. A subsequent CPU cached executor now connects it to inference; see [M2_CACHED_EXECUTOR.md](M2_CACHED_EXECUTOR.md). CUDA/P100 integration remains open.

## Storage and completion contract

- The caller supplies one backend, a byte budget, a layout identity, embedding/FFN dimensions and gate/up/down types. F32, F16, Q4_0 and Q8_0 are accepted, including different types across projections. Shape and quantization-block checks precede tensor creation.
- The pool holds three contiguous `[input, output, capacity]` projection tensors. Capacity uses the backend's padded allocation estimate for one complete expert. The actual packed allocation is checked against the budget before allocation and checked again afterward. This conservative geometry can leave budget unused. Backend tail padding is zeroed before use. Metadata, backend internal allocations and compute workspace are outside the weight-buffer budget.
- A source bundle provides three exact-sized canonical expert slices and a shared lifetime owner. The caller is responsible for matching those bytes to the model/layer/expert/layout/adapter key. Repacked weights must not be supplied. No conversion or requantization occurs, and no full host bank is copied into the arena.
- A miss submits all three slices through GGML backend transfer calls. The owner remains retained until completion. When available, a backend event is recorded and synchronized; otherwise the backend is synchronized. Only then does the cache publish the complete bundle as usable. Loading is serialized and eager, with no transfer/compute overlap or additional pinned staging allocation.
- Hits retain their contents and submit no weight transfers. A tensor lookup requires a valid live ticket and matching key. Returned tensors describe the whole pool: every slot referenced by a kernel must have a live ticket. Original expert IDs must be remapped to ticket slot indices, and optional scales must continue to use original IDs.
- Release records and waits for backend completion before making a slot evictable. All kernels referencing the cache must be submitted on the supplied backend. The caller serializes access and keeps that backend alive through cache destruction. Destruction drains outstanding work before freeing events, buffers and tensor metadata.
- Allocation failure raises `std::bad_alloc`. Invalid sources are rejected before cache mutation. The exception cleanup path drains submitted copies before marking a failed bundle empty; backend APIs that terminate the process on device errors are not made recoverable by this wrapper.
- `allocated_bytes()` reports retained backend buffer bytes. Storage counters report submitted canonical weight bytes and host-side load submission/wait microseconds, separately from the state module's padded-slot counters. These are not PCIe bus traffic measurements or device-only timings.

## Verification

The existing `test-moe-offload` now executes complete SwiGLU gate/up/down graphs using the persistent cache. It compares expert outputs and weighted merges against ordinary CPU FFN execution using the same source bytes. Tests cover all four types, mixed Q4_0/Q8_0 projections, rectangular dimensions, odd F32 dimensions, capacities 1/2/7, repeated routes, eviction, altered warm slot ordering, and warm repeated workloads requiring no additional copies when they fit. Readback checks verify exact canonical bytes, and copied-byte totals must equal successful loads times the unpadded expert bundle size.

Invalid budget, backend, layout, shape, type, source ownership and slice lengths are rejected. Active slots cannot be replaced, and released or mismatched tickets cannot access tensors. A test backend delays actual CPU-buffer copies until event synchronization to verify publication, release and destruction ordering. It also injects an allocation failure and checks cleanup. This test backend exercises the event API contract; it is not a CUDA event test.

Windows CPU builds of the three MoE tests, CLI and server pass. Six focused CTests pass: MoE cache state, MoE integration, MoE offload, sampling, grammar parser and quantization. Logs: [build](reports/m2-storage-build.log), [tests](reports/m2-storage-tests.log). The CLI/server build retains the existing UI asset download fallback and missing-gzip warning; those are unrelated to cache storage.

Reproduce from the workspace:

```powershell
cmake --build build/ft-windows-cpu --config Release --target test-moe-cache test-moe-offload test-moe-integration llama-cli llama-server --parallel 4
ctest --test-dir build/ft-windows-cpu -C Release --output-on-failure -R '^(test-moe-cache|test-moe-offload|test-moe-integration|test-sampling|test-grammar-parser|test-quantize-fns)$'
```

## Next work

The reusable executor, resident-hit protection, context-owned Qwen CPU integration and retained/temporary telemetry are now implemented behind `--moe-cpu-cache-mib`; see the executor notes. Next, implement a backend-neutral post-routing boundary for eager CUDA, bounded pinned staging/fallback and backend failure/cancellation cleanup.

Still open: real CUDA/P100 allocation, events, numerical execution, device OOM/cancellation, native Linux testing, pinned staging, Qwen pretrained GGUF selection and acceptance, and the remaining M0/M1 memory and hardware gates. No GPU support or speedup is claimed. With caching disabled, the previous inference paths are preserved. Preserve these uncommitted changes with the other implementation files; the historical baseline patch does not contain them.
