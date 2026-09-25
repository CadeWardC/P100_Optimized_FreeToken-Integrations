# Modular runtime verification

Verified on Windows with MSVC 19.43 on 2026-09-14.

| Configuration | Result |
| --- | --- |
| Existing CPU llama integration | `test-moe-cache`, `test-moe-offload`, `test-moe-integration` pass |
| Existing CUDA integration | `test-moe-offload --device CUDA0` and `test-moe-integration --device CUDA0` pass on the local RTX 4060 |
| Standalone portable module, local GGML, static libraries | `freetoken-smoke` passes |
| Standalone portable module, baseline GGML, shared GGML libraries | Build and `freetoken-smoke` pass |
| Installed package consumed by a separate project | `find_package(freetoken)` configuration, link and `freetoken-consumer` pass |

The baseline GGML was extracted from this workspace's llama.cpp HEAD, `bb4caa7540188872173c44d161602d9271386413` (GGML 0.21.0), excluding the working tree's FreeToken backend extensions. It includes the pre-existing P100 patch baseline; this is not a test against every upstream release or fork.

The integration harness reports 135 synthetic model comparisons covering Qwen/Gemma, fused experts, mixed quantization, scales, state continuation and recovery. CUDA executor checks also cover cache capacities, decode/prefill, cancellation, bandwidth calibration and fixed/adaptive CPU/GPU splits.

This refactor did not validate Linux/macOS builds, P100 hardware, native GPU mode in another fork, or pretrained model throughput. Portability beyond the tested GGML revision requires building and running the examples plus the destination engine's numerical tests.
