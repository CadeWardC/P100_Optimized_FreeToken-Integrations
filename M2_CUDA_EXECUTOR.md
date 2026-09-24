# M2 eager GPU executor and model selection

Inference integration is now described in [GPU_GENERATION.md](GPU_GENERATION.md). The remainder of this file records the earlier standalone milestone.

Development work dated 2026-09-09. This is a standalone GPU expert executor, not GPU model integration or a P100 release. The CLI/server MoE option still executes experts on CPU.

## Implemented

- Explicit device selection on the internal cached executor. A dedicated backend executes the complete compact gate/up/activation/down graph with original expert-ID scales and routing-order aggregation. It runs outside backend callbacks. Unsupported operations fail rather than falling back silently.
- Device-specific padded cache sizing, a bounded staging allocation, chunked transfers, and pinned-host allocation with pageable fallback. The source owner and staging contents remain alive through completion. Transfers are serialized; there is no overlap yet.
- A separate compute-arena budget checked before allocation. GPU cache weights, staging and compute arenas are separate budgets. CUDA driver allocations, cuBLAS workspace and the backend scratch pool are not bounded or fully reported by these counters. This is not a total VRAM limit.
- Cooperative cancellation before tiles, between transfer chunks, and around synchronous tile computation. Partial failed loads are drained and invalidated; ticket guards release acquired slots. An in-flight CUDA kernel is not interrupted. Fatal CUDA errors in the underlying backend can still abort the process; device-loss recovery is not implemented.
- Fused gate/up host banks, required by the selected Gemma GGUF. Retained tensor descriptors select contiguous per-expert halves using the original expert stride. No second full weight pool is constructed. Copied/mapped and split-file inference tests cover the layout.
- An opt-in pretrained acceptance mode in `test-moe-integration --model <file>` comparing full-vocabulary logits, 16 greedy tokens and state replay between ordinary CPU inference and cached CPU inference. A checksum-verifying wrapper records results. This does not test GPU model inference.

## CUDA issues found during validation

The eager allocator requires an actual reserve after its sizing-only call. That sequence is now explicit, and allocations are drained before destruction on failure.

The existing fused CUDA FFN path produced inconsistent results for these eager graphs. Keeping gate/up intermediate tensors materialized disables that fusion locally. The default model graph and global CUDA environment are unchanged. Fused execution needs separate diagnosis before enabling it here.

Compute Sanitizer initialization checks found partial quantized tiles reading unused routing-ID entries. Three MMQ loads now guard their column index against the current expert's active-column count. Valid output columns retain the same mapping.

## Validation and reproduction

The Windows development build uses CUDA 12.6, MSVC 14.43, architectures `60;89`, CUDA graphs disabled, and the baseline's portable CPU settings. `scripts/build-cuda-dev.cmd` builds CLI, server and both executor/integration tests. `FT_CUDA_ROOT` and `FT_VS_ROOT` may override local tool locations.

The test host is an RTX 4060 Laptop GPU with about 16 GB system RAM. Compiling architecture 60 is not P100 runtime acceptance. Hardware inventory is in `reports/gpu-development-inventory.json`.

```powershell
./scripts/build-cuda-dev.cmd
$env:PATH = "$env:FT_CUDA_ROOT/bin;$env:PATH"
./build/ft-windows-cuda-dev/bin/test-moe-offload.exe --device CUDA0
ctest --test-dir build/ft-windows-cuda-dev --output-on-failure -R '^(test-moe-offload|test-moe-integration)$'
```

The standalone GPU matrix covers F32/F16/Q4_0/Q8_0, separate/fused host layouts, cache capacities 1/2/7, one/five input tokens, both activations, expert scales, eviction, warm reuse and cancellation. It also tests cancellation after partial GPU transfer, retry, compute-budget rejection and a numerical cancellation fixture. GPU comparisons use a maximum element error of `0.003 + 0.05*abs(reference)` and RMSE of `0.0001 + 0.05*RMS(reference)` against CPU; these are development tolerances, not pretrained quality criteria.

CPU integration covers 135 generated-model comparisons (28 new fused Gemma cases) plus a synthetic smoke test of the pretrained acceptance harness. Its existing `2e-5 + 2e-5*abs(reference)` logit tolerance remains unchanged. The six focused CPU CTests pass. Three Python GGUF-header tests cover fused directory parsing, truncation and corrupt counts.

Evidence: `reports/gpu-executor-cpu-tests.log`, `reports/cuda-executor-tests.log`, `reports/cuda-build-cpu-regressions.log`, `reports/cuda-memcheck.log`, `reports/cuda-initcheck.log`, and `reports/model-header-tests.log`. Check the final summaries in sanitizer logs; a numerical pass alone does not establish clean memory checks.

Final RTX 4060 checks pass: both Compute Sanitizer memcheck and initcheck report zero errors. The standard backend suite passes 290 selected `MUL_MAT_ID` comparisons across F32/F16/Q4_0/Q8_0, including partial tiles (`reports/cuda-matmul-regressions.log`). These checks do not substitute for P100 execution or long-running memory-pressure acceptance.

## Selected pretrained artifacts

`MODEL_ACCEPTANCE_MANIFEST.json` records immutable repository revisions, sizes and publisher SHA-256 hashes. Bounded HTTP range reads inspected the actual GGUF tensor directories; full weights have not been downloaded or hashed locally.

| Target | Selected artifact | Header result | Planning RAM |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B | ggml-org Q8_0, 36.90 GB | 40 layers, separate Q8_0 experts | 64 GiB |
| Gemma 4 26B-A4B-it | ggml-org Q4_0, 14.62 GB | 30 layers, fused gate/up and Q4_0 down experts | 32 GiB |

These RAM figures are estimates, not measured acceptance. An inspected bartowski Qwen Q4_0 candidate contains Q4_1/Q4_0/Q8_0 expert tensors and does not satisfy the current uniform cache layout; its report is retained in `reports/qwen-q4-pretrained-header.json`. Filenames are not a quantization allowlist.

On a sufficiently sized host with a complete selected file:

```powershell
python scripts/validate_pretrained.py --id qwen35b-q8 --model D:/models/Qwen3.6-35B-A3B-Q8_0.gguf --executable build/ft-windows-cpu/bin/Release/test-moe-integration.exe --output reports/qwen-pretrained-local.json
```

Use `gemma26b-q4` and the Gemma filename for the second target. The wrapper rejects incorrect size/hash before launching inference and preserves a new log/report for each run. Its real-model path is prepared but has not run here.

## Remaining release gates

1. Integrate an eager backend-neutral post-routing boundary into `llama_decode`, with public device/budget options and whole-context memory accounting. Do not pass a CUDA executor into the existing CPU custom callback.
2. Run the pinned pretrained files through logits/state, CLI and server acceptance; add actual GPU model comparisons after integration.
3. Add adaptive CPU/GPU splitting and overlapping prefill transfers only after the integrated path is correct and measured. Neither feature is implemented here.
4. Native P100 Windows/Linux correctness, pressure, cancellation and performance acceptance. No P100 host is available in this session.
5. Clean-machine dependency tests and release archives for both operating systems. Any Windows archive produced by `scripts/package_development.py` is labeled development and records the missing gates explicitly.
