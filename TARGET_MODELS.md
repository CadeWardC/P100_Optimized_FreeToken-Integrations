# Primary model targets

Update (2026-09-09): exact Qwen Q8_0 and Gemma Q4_0 artifacts, revisions and publisher hashes are selected in [MODEL_ACCEPTANCE_MANIFEST.json](MODEL_ACCEPTANCE_MANIFEST.json). Their actual GGUF headers were inspected. The chosen Gemma uses fused gate/up tensors, now supported with 28 additional synthetic comparisons (135 total). Full-file hashes and pretrained outputs remain unverified. A standalone CUDA executor is implemented; model inference still uses the CPU boundary. See [M2_CUDA_EXECUTOR.md](M2_CUDA_EXECUTOR.md) for current evidence, RAM estimates and remaining gates. This update supersedes the earlier selection/verification status below.

Qwen3.6-35B-A3B and Gemma 4 26B-A4B are equal primary targets for the FreeToken/P100 implementation. Subsequent cache, CUDA and release acceptance work must validate both. Llama/Mixtral-style synthetic models remain regression controls.

| Target | GGUF architecture | Routed expert behavior | State and shared computation to preserve |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B | `qwen35moe` | SwiGLU | Gated shared expert, hybrid full/recurrent attention and recurrent state |
| Gemma 4 26B-A4B | `gemma4` with MoE tensors | GeGLU, per-expert output scales, eight selected experts | Shared dense FFN, branch normalization, sliding/full attention, KV state and logit softcap |

Both already use the same context-owned CPU cache and tiled executor. The runtime selects architecture behavior from GGUF metadata. Model filenames do not select behavior.

## Current use

Both CLI and server accept:

```text
-m <compatible-model.gguf> -ngl 0 --no-op-offload --no-kv-offload -np 1 --moe-cpu-cache-mib 64
```

Replace the model placeholder with either target's local GGUF. The 64 MiB value is an example cache budget; it must hold at least one complete padded expert. The cache budget is shared across layers in one context. The full expert pool, ordinary weights and runtime overhead must fit host RAM in addition to the cache.

Current supported routed weight types are F32, F16, Q4_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K and IQ2_S. Types may differ across projections and layers. Each distinct gate/up/down type layout has separate compressed storage within the configured total weight-cache budget; projection dimensions must still match across layers. The budget must hold at least one padded expert for each layout. Remaining capacity is divided by layer count and expert size; unused capacity in one layout is not dynamically borrowed by another. Other K/I formats remain unsupported. See [mixed-quantization verification](reports/MIXED_QUANT_SUPPORT.md). Gemma output scales remain F32 and are looked up with original expert IDs after remapping. Text inference, one sequence, CPU placement, no LoRA and MTP disabled remain the supported prototype configuration. The ordinary path remains the default.

## Verification and acceptance

The integration suite covers 107 generated-model comparisons: 34 Qwen, 48 Gemma and 25 Llama controls. This includes 17 cached Qwen cases and 28 cached Gemma cases. Gemma now has copied/mapped one-slot cache checks and split-file eight-expert routing checks across all four types with one-slot, two-slot and fitting cache budgets. The tests compare routed/shared outputs, logits, state continuation, bounded storage and failure recovery against ordinary execution.

Evidence: [build log](reports/dual-target-build.log), [focused test log](reports/dual-target-tests.log). See [cache usage and telemetry](M2_CACHED_EXECUTOR.md), [Qwen details](QWEN_CPU_OFFLOAD.md) and [Gemma details](GEMMA_CPU_OFFLOAD.md).

The results above are historical synthetic CPU tests. Subsequent Gemma QAT and RTX 4060 validation is described below. Qwen pretrained validation, native Linux verification and P100 Windows/Linux acceptance remain open.

## Gemma QAT GPU generation

The exact Google QAT artifact is now recorded as `gemma26b-qat-q4` in `MODEL_ACCEPTANCE_MANIFEST.json` and locally verified. GPU expert generation and its memory flags are documented in [GPU_GENERATION.md](GPU_GENERATION.md). Attention and shared layers remain on CPU. CPU pretrained parity passes, but GPU numerical acceptance fails. Matched CLI/server and memory-pressure runs complete with token divergence and no consistent speedup; the GPU path remains experimental.
