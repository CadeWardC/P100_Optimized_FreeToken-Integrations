# Model settings

The web GUI has Inference, Model loading, and FreeToken tabs, search, per-field reset, and persistent settings. Loading controls come from the backend's setting registry, so their keys match the actual engine arguments.

Response length accepts a positive token count or **No output token limit** (`max_tokens=-1`). Unlimited removes the application output cap; end-of-sequence, stop strings, model context capacity and user cancellation still apply. The optional output counter appears during generation and on saved assistant messages. Changing an inference setting applies to the next request. Load settings require a reload.

Inference controls include the system prompt, temperature, seed, Top K/P, Min P, repetition/frequency/presence penalties, typical and XTC sampling, Mirostat, dynamic temperature, DRY, thinking and its budget/message, stop strings, JSON schema and GBNF grammar. A full-context repetition window is converted to the configured context size because this engine requires nonnegative window values.

Load controls include context, decode/prefill threads, batch/microbatch, GPU layers, memory fitting, multi-GPU split and ratios, CPU expert placement, KV placement and quantization, flash attention, memory mapping/locking, weight repacking, warmup, context overflow/checkpoints, unified cache, parallel engine sessions, chat templates, reasoning parsing and speculative draft model settings. The GUI itself maintains one chat at a time.

## FreeToken

**Enable & auto-configure FreeToken** detects current hardware and the selected model, applies a compatible profile, and saves that profile as a baseline for this model. It preserves sampling and output-limit preferences. Edits save independently; **Restore automatic settings** restores the saved baseline without re-detecting hardware. **Disable FreeToken** suppresses custom expert and semantic-checkpoint flags while retaining their values.

Dense models receive hardware placement tuning. Supported MoE architectures can use GPU expert caching, transfer pipelining, adaptive CPU/GPU miss scheduling and semantic checkpoints. Low-VRAM/CPU-only machines receive a CPU cache profile. Native numerical mode remains manually selectable but cannot coexist with semantic checkpoints. Custom caching uses one engine session, no draft model, and explicit budgets. Model tensor formats and backend kernels can impose further restrictions at load time. These are conservative starting profiles, not a speed benchmark or a guarantee of fit.

## LM Studio comparison

Compared on 2026-09-11 against LM Studio's [configuration inventory](https://github.com/lmstudio-ai/localization/blob/main/en/config.json), [load configuration](https://lmstudio.ai/docs/typescript/api-reference/llm-load-model-config), and [prediction parameter documentation](https://lmstudio.ai/docs/typescript/llm-prediction/parameters).

This is a set of llama.cpp model/inference equivalents, not a clone of LM Studio's application preferences. Apple MLX and ONNX controls, its GPU driver memory cap, legacy tail-free sampling and its tool-execution system are unavailable here. Context overflow uses this engine's context-shift/stop policies. The UI explicitly identifies these differences instead of offering ineffective controls.

## Verification

`desktop-tests` covers numeric/structured inference payloads, unlimited limits, compatible CPU/GPU profiles and existing core behavior. `scripts/test-host-engine.ps1` exercises load/reload, rejected starts, persisted multi-setting edits, FreeToken baseline restore, unlimited-output chat and unload. Use `-RealEngine -ModelPath <GGUF>` for a real model; default mode uses the test server. All integration settings/conversations are isolated under `reports/engine-test-*`.
