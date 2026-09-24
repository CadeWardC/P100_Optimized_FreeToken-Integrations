# GPU expert correctness fix

Latest implementation: [GPU-resident activation and hybrid scheduling](GPU_HYBRID_SCHEDULER.md) supersedes the historical CPU activation and missing CPU/GPU split statements below.

This supersedes the failed GPU acceptance status in the earlier development reports. The P100-patched llama.cpp development build now passes the Gemma QAT pretrained comparison on this RTX 4060 Laptop GPU. This is not P100 hardware acceptance or a complete FreeToken scheduler port.

## Changes

- Q4 vector dot products now apply the zero point to quantized input values instead of using the original floating-point input sum. A zero-weight regression exercises decode, prefill and two cache capacities.
- Cached expert projections request explicit F32 precision. The CUDA floating-point dispatch respects this request and preserves it when creating per-expert fallback views, avoiding reduced-precision matrix arithmetic.
- The cached Q4/Q8 CUDA path uses Q8_0 activation quantization and ordered F32 accumulation compatible with this build's portable CPU kernels. Weights remain packed in the bounded GPU cache. This compatibility kernel favors reproducibility over maximum throughput.
- GPU assignments are processed in batches of at most eight. This keeps the execution family consistent across prefill, decode and cache capacities.
- Gemma's small GeGLU activation runs through the existing CPU lookup-table implementation between GPU gate/up and down projections. Both intermediate transfers are synchronous and the host vectors are included in temporary-memory telemetry. Attention and shared layers also remain on CPU, as before.
- The CUDA build script now builds the cache-state test as well as the executor and integration tests.

The FreeToken reference was inspected directly, especially `python/freetoken/moe/fused_q4_0.py` and `python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp`. Its Q4 path uses a consistent vector-kernel family for prefill and decode and aligns CPU/GPU quantized execution. This port still does not implement FreeToken's adaptive CPU/GPU miss split, overlapping transfers, double-buffered prefill, or elastic cache resizing.

## Verification

- `reports/gemma-qat-fix-final.json` and its log: exact Gemma QAT artifact, full-vocabulary logits, 16 greedy tokens and state replay.
- `reports/fix-final-integration.log`: 135 synthetic GPU model comparisons plus pretrained-harness smoke test pass.
- `reports/fix-final-executor.log`: GPU executor matrix and Q4 zero-point regression pass.
- `reports/fix-final-cpu-tests-rerun.log`: all three focused CPU tests pass.
- `reports/fix-final-python-tests.log`: ten script tests pass.
- `reports/qat-operation-fixed/results.json`: fresh enabled/disabled CLI and repeated server comparison, including a 16 MiB cache with 2 GiB additional GPU allocation.
- `reports/fix-final-memcheck.log`: CUDA Compute Sanitizer reports zero errors for the GPU executor matrix and regression.
- `reports/fix-final-matmul.log`: the standard CUDA expert-matmul comparisons pass.

The final pretrained run reports zero logit error at every checked position. All six server requests produce identical 64-token sequences. All three CLI runs exit successfully. Together these operation checks generate 576 tokens. This is a short repeat/pressure check, not a long-running soak test.

## Measured performance

Same exact model, raw six-token server prompt, 64 generated tokens, eight CPU threads, context 256, microbatch 8, deterministic sampling, and CPU attention/shared layers:

| Mode | First request, tokens/s | Repeat, tokens/s | Sampled global VRAM peak, MiB |
| --- | ---: | ---: | ---: |
| Disabled | 2.49 | 2.41 | 83 |
| GPU expert cache, 1024 MiB | 1.46 | 1.33 | 1111 |
| GPU expert cache, 16 MiB, plus 2048 MiB pressure | 1.20 | 1.19 | 2236 |

The corrected implementation is slower in this run. No speedup is claimed. Available system RAM fell to roughly 42-75 MiB depending on mode, so host memory pressure also affects these numbers. VRAM samples include other applications and can miss transient peaks. The next performance work is FreeToken-style CPU/GPU miss sharing and overlapped transfers; neither is part of this correctness fix.

Historical failed attempts and their reports are retained. Do not infer correctness or speed from those earlier binaries or the previously packaged development archives. Use `build/ft-windows-cuda-dev/bin/llama-cli.exe` or `llama-server.exe` from the current build.

## Limitations

The verified host has approximately 16 GB RAM and 8 GB VRAM. Performance measurements are specific to this host and its memory pressure. Exact agreement is an observed result for the tested portable CPU build and workloads, not a guarantee across CPU instruction sets, models or arbitrary prompts. The CPU activation step and ordered GPU reductions are compatibility choices; further optimization must retain numerical and cache-size/replay checks. Native P100 and Linux execution remain untested.
