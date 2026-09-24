# Qwen CPU offloading verification

Implemented and tested on Windows CPU on 2026-09-09 for the `qwen35moe` architecture used by the selected Qwen3.6-35B-A3B target.

Persistent CPU cache integration is now available through `--moe-cpu-cache-mib`, with 17 additional generated Qwen comparisons covering one-slot/fitting caches and split-file loading. See [M2_CACHED_EXECUTOR.md](M2_CACHED_EXECUTOR.md) for usage, telemetry and the remaining CUDA boundary work.

## Behavior

The loader and graph allowlists now accept Qwen35MoE. Its existing router supplies selected IDs and normalized weights to the bounded executor. Only routed gate/up/down experts pass through that executor. The existing shared SwiGLU expert, its sigmoid gate, the shared/routed sum, full attention and recurrent attention remain ordinary llama.cpp graph operations.

Host banks cover main decoder layers only. GGUF files may contain unused MTP tensors with MTP loading disabled; enabling MTP remains rejected. The other existing restrictions remain: CPU placement, one sequence, no operation/KV offload, no LoRA, separate unscaled and unclamped routed projections, and routed expert types F32, F16, Q4_0 or Q8_0. The tile budget bounds compact routed weights, not shared weights, full host banks or total RAM.

## Tests

`test-moe-integration` generates deterministic synthetic GGUF models and compares the same loaded weights with CPU tiling disabled and enabled. This uses ordinary llama.cpp execution in the same fork as the reference, not a separately built upstream executable.

- 34 model configurations: 17 existing Llama controls and 17 Qwen configurations. Qwen covers F32/F16/Q4_0/Q8_0, copied/mapped loading, capacity-one/full-bank budgets, and a seven-expert split GGUF with an unused MTP block.
- Each Qwen model has one gated-delta recurrent layer, one full-attention layer, and nonzero gated shared experts. Key/value recurrent head counts differ to exercise grouped heads.
- Prefill, repeated decode, changed batch shapes, microbatch splitting and 40 sustained decode steps compare logits and routed FFN outputs within `2e-5 + 2e-5 * abs(reference)`.
- Gated shared outputs, merged FFN outputs and recurrent state inputs use the same tolerance. Assertions check nonzero shared contributions/state and the shared-plus-routed sum.
- Saved states reproduce three-token continuations in both paths. An ordinary-path state also restores into a tiled context and reproduces the continuation. Fresh contexts are compared without observation callbacks.
- Reserved graphs contain two custom routed FFNs, no `MUL_MAT_ID`, and no full expert-bank source edges. Tile and retained-memory accounting checks remain active.
- Cancellation and injected invalid routing fail as expected. Qwen failure retries restore the saved state first because recurrent operations can advance state before an FFN failure. This does not promise automatic rollback after failed decode.

Build: [Windows CPU build log](reports/qwen-cpu-build.log). Regression results: [six focused CTests](reports/qwen-cpu-tests.log). CLI and server are rebuilt against the updated interface.

## Remaining acceptance work

No pretrained 35B GGUF, CLI/server inference request, Linux build, CUDA execution or P100 hardware was tested here. Select an exact compatible GGUF/quantization and hash it before pretrained acceptance. GPU cache execution and speedup claims remain outside this change.
