# FreeToken runtime

Copy this directory into another GGML/llama.cpp project to reuse the expert cache, bounded expert execution, CPU/GPU scheduling, and tensor graph boundary. It builds as `freetoken::runtime`, a C++17 static library with position-independent code. It does not link llama, its model loader, server, desktop app, or P100 kernels. GGML and its CPU backend are required; CUDA is optional.

## Add to an existing project

Add the module after your project's GGML targets, then link the target that calls FreeToken:

```cmake
# Your project already defines ggml and ggml-cpu.
add_subdirectory(vendor/freetoken)
target_link_libraries(my-inference-engine PRIVATE freetoken::runtime)
```

Use `#include "freetoken/moe.h"`. Do not compile the module sources separately or add a second GGML copy. This is a source integration, not a stable binary plugin ABI. Rebuild consumers when changing GGML or FreeToken.

Alternatively, install a matching GGML/FreeToken build with `cmake --install build-freetoken --config Release --prefix /path/to/sdk`, then use `find_package(freetoken CONFIG REQUIRED)` and the same `freetoken::runtime` target. Set `CMAKE_PREFIX_PATH` to the SDK prefix. The `examples/installed` project verifies this route. With shared GGML on Windows, put the SDK's `bin` directory on `PATH` or deploy those DLLs beside the executable. Install only compatible configurations into a given prefix; use separate prefixes for Debug and Release.

## Standalone build and example

```sh
cmake -S freetoken -B build-freetoken \
  -DFREETOKEN_GGML_SOURCE_DIR=/path/to/llama.cpp/ggml \
  -DFREETOKEN_BUILD_EXAMPLES=ON -DGGML_CUDA=OFF
cmake --build build-freetoken --config Release --parallel 4
ctest --test-dir build-freetoken -C Release --output-on-failure
```

On PowerShell, put the configure command on one line. `examples/smoke.cpp` is a complete model-free consumer using only the public API. It verifies CPU SiLU/GeGLU output parity, shared weight ownership, cache reuse, resizing and invalid route rejection.

## Connect your inference loop

1. Load the canonical expert tensors into ordinary CPU buffers. Construct one `freetoken::host_bank` per expert layer with a shared owner that retains the tensors, GGML context and buffer. Gate/up are `[embedding, hidden, experts]`; down is `[hidden, embedding, experts]`. Pass a null up tensor for a fused gate/up bank. Optional output scales are indexed by the original expert ID.
2. Construct one `freetoken::cached_executor` per inference context from those banks and a byte budget. Banks in an executor must have matching embedding/hidden dimensions; projection quantization types may differ. Use `min_tile_bytes()` to size the minimum bundle per layout; different quantization layouts each need at least one slot. Keep the executor alive across decode calls so it can reuse resident experts and workspaces.
3. After routing, call `execute(layer, input, n_tokens, top_k, ids, weights, n_threads, activation)`. Input/output are flat embedding-major arrays, `[embedding, tokens]`. IDs and weights are `[top_k, tokens]`, with the top-k entries for each token adjacent. `layer` is the index into the bank vector, not necessarily the model's layer number. Routing weights are applied once by the runtime.
4. Add any shared dense expert or residual contribution in the host engine. Routing, model loading, attention/KV state, adapters, server requests, and model-specific graph construction remain the host's responsibility.

```cpp
auto bank = std::make_shared<freetoken::host_bank>(model_owner, gate, up, down);
freetoken::executor_options options;
freetoken::cached_executor experts({bank}, cache_bytes, options);
auto result = experts.execute(0, activations, tokens, top_k,
                              expert_ids, routing_weights, threads,
                              freetoken::activation::silu);
// Feed result.output into the remainder of the model graph.
```

Banks are immutable for the executor lifetime. Recreate them and the executor after changing weights or applying an adapter. Serialize execution, resize, metrics reads, and destruction. Tensor owners must retain their buffers through all transfers. An abort callback can cancel work; failures throw C++ exceptions. Do not let exceptions cross a C ABI.

For a GGML custom graph node, `cpu_op` and `cpu_graph()` retain the same boundary used by this fork. Keep the operation state alive as long as its graph. CPU callbacks must stay CPU-only. GPU execution must run outside backend callbacks: drain producer backends, call `execute_device()` or `eager_compute()` from the inference/scheduler thread, and inspect the returned status and `cpu_op::error`. `execute_device()` currently supports single-token decode, contiguous device tensors, and the selected executor device. Use `supports_device()` to choose the vector path for batches. The host must schedule the boundary; linking the library alone does not insert model hooks.

## Backend compatibility

The default `FREETOKEN_PATCHED_GGML=OFF` uses public GGML headers only. CPU execution works without this project's custom backend symbols. CPU workspace telemetry is zero when unavailable; this is not a measurement of total process memory.

For GPU execution in another fork, select a device in `executor_options` and set `fast=true` to use that backend's native numerical behavior. Native GPU results need model-level validation on the destination backend. Compatibility GPU mode (`fast=false`) fails explicitly in portable builds rather than assuming the destination implements this fork's precise kernels.

This project's llama integration enables `FREETOKEN_PATCHED_GGML=ON`. That mode requires the CPU workspace/GELU-table functions, MUL_MAT_ID precision support, and precise CUDA kernels already present here. It preserves the existing compatibility path, pipelining and optional CUDA graph hooks. Copying this module does not copy those backend patches. Full GPU graph integration also needs the host scheduler boundary currently implemented by `ggml_backend_sched_set_node_executor` in this fork, or an equivalent host-controlled split.

GGML evolves and this is not a claim of compatibility with every historical fork. The tested baseline and verification results are recorded in [VERIFICATION.md](VERIFICATION.md). Run the smoke example and your model parity tests against the exact GGML revision you use.

The optional device-tensor shortcut additionally requires the `ggml_backend_cuda_moe_read_routes` backend hook. When it is absent, `supports_device()` returns false and the vector execution API remains available.

## Layout

- `include/freetoken/moe.h`: model-independent expert banks, execution API, options, metrics, and optional GGML graph boundary.
- `include/freetoken/cache.h`: lower-level uniform cache tickets, eviction and backend storage. Existing `ggml_moe_*` names are retained.
- `src/`: the single implementation of caching and expert execution.
- `examples/smoke.cpp`: a standalone consumer without llama headers or model files.

In the surrounding fork, `src/llama-moe-offload.h` and `ggml/src/ggml-moe-cache.h` provide source compatibility aliases/includes. Model/context/graph hooks and CLI settings continue to use those adapters. Do not copy those adapters to a new project unless porting its llama integration too.
