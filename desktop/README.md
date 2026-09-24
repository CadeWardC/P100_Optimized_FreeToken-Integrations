# FreeToken desktop

For Ubuntu, follow [the build guide](../UBUNTU_BUILD.md). The default Linux build produces the browser host; FLTK and Slint are optional CMake targets. Slint now requires `-DFT_BUILD_SLINT=ON` on all platforms.

Two applications share the engine core in this folder:

- **`llamacpp-p100.exe`** — the new LlamaCPP P100 GUI (recommended). Build with `scripts/build-desktop.cmd`; the executable lands in `build/desktop/`. Launch it and it opens `http://127.0.0.1:18432/` in your browser automatically (`--no-browser` to disable, `--port N` to change the UI port). It manages llama-server for you and shows what is happening: live model loading stages, per-message tokens/s and time-to-first-token, VRAM/RAM gauges, context fill, and a streaming server log. Settings and conversations migrate automatically from the old FLTK app's data folder. Flags: `--port`, `--data-dir`, `--no-browser`.

  **Memory**: built to stay out of the model's way. The GUI never holds model weights (llama-server is a separate process), the host's working set is ~4 MB (trimmed; private ~37 MB, pageable), and the page's JS heap is ~4 MB. Closing or hiding the browser tab during long runs returns the renderer's memory — the engine keeps running, and a hidden tab drops its connection until you come back.
- **`FreeP100.exe`** (formerly `freetoken-native.exe`) — the FLTK desktop app you liked, now upgraded with the same live telemetry as the other UIs:

  - **Tokens/s + TTFT** — live tok/s in the status line while generating; each assistant reply ends with a stats line (`[20.8 tok/s | TTFT 284 ms | 318 tokens | completed]`) built from the server's `timings` stream and usage chunk.
  - **Model-loading progress** — staged status with percentage (`Reading model file → Loading tensors → Placing layers on GPU → Allocating context → Starting API`), streamed from llama-server's output through a live pipe.
  - **VRAM/RAM gauges** — sidebar bars polled at 1 Hz via NVML / GlobalMemoryStatusEx.
  - **Context fill + session counters** — `ctx used/total` (from `--slots`), requests and tokens in/out, in the sidebar chip line.
  - **Live server log** — Diagnostics streams the engine output (bounded ring, auto-scroll) instead of tailing the file on refresh.
  - **Startup review memory plan** — an estimated VRAM/RAM budget (weights, KV, expert caches, reserve) computed from the selected model's GGUF dimensions.

  The engine is launched with `--metrics --slots` to feed the context gauge and counters. Shares `settings.json` / `conversation.json` with the other UIs — run one UI at a time (all three manage llama-server on the same port).

## Model library

In the web UI, selecting a model starts loading it automatically. Reload stops the managed engine before checking its port. Failed starts display the error and keep a Retry Load action available.

In **Settings**, click **Browse** beside **Model folder** to choose a folder with the Windows folder picker, or paste a path. Click **Save Settings** to remember the folder and scan it. Canceling the picker or Settings leaves the folder unchanged. The folder is remembered for future launches; **Rescan** in the Models tab refreshes the saved folder. Invalid or inaccessible folders show an error without replacing the saved path. Scanning includes subfolders and runs in a background thread. The header dropdown selects a discovered model from any screen. Model metadata includes architecture, total shard size, layers and maximum context. Unreadable files remain listed with an error. Projectors and subsequent GGUF shards are not offered as standalone language models.

The eye icon means a projector was paired: a single model/projector in its own folder, or an unambiguous filename match in a shared folder. Pair ambiguous projectors manually. This indicates a local projector association, not a verified compatibility test. The selected projector is passed to the server using `--mmproj`; image attachments in the chat composer are not implemented.

## Recommendations and overrides

**FreeToken Auto** applies a recommended profile when selecting a different model. Disabling it retains manual values. **Detect hardware** recalculates and applies a new profile. **Restore recommended** applies the saved baseline without recalculating it. Re-selecting the same model retains overrides. Valid settings edits save automatically; load-related changes require **Load / reload model**. Selecting another model disables sending until that model is loaded.

Recommendations inspect GGUF dimensions and detect logical CPU threads, available RAM, and the first CUDA device's name and total memory on Windows. The GPU layer estimate reserves 30% of device memory plus estimated F16 KV storage and 512 MiB of scratch space. Unknown dimensions or unavailable CUDA detection use CPU placement. Split-file sizes are summed before estimating placement. P100 graphs default to Off.

These are conservative starting profiles, not calibrated fastest settings or a guarantee that a model will fit. On systems with a single NVML GPU, free-memory telemetry further limits the budget to 85% of free VRAM. Without that telemetry the estimate uses total capacity; the hardware summary displays total capacity. Other workloads, architecture-specific buffers, and nonuniform layer sizes affect actual usage. Auto leaves experimental expert-cache budgets empty; it does not automatically enable unverified GPU numerical modes. Manual expert budgets live in their own collapsible section. The expert-caching switch can bypass them while retaining their values.

## Settings

The web GUI now provides searchable model-loading, inference and FreeToken controls, unlimited response length, output-token counts and an editable/restorable automatic FreeToken profile. See [Model settings](SETTINGS.md) for behavior, supported LM Studio equivalents and verification.

- Context and GPU offload: context, GPU layers, CPU threads and batching.
- Attention and memory: dropdowns for flash attention and KV types, memory switches.
- Inference and sampling: system prompt, thinking, temperature, response length, top K/P, min P and repeat penalty. Sampling values are sent with chat requests.
- FreeToken expert caching and advanced execution: experimental controls are separated from routine settings.
- Diagnostics: server executable, port, logs and an explicit save action.

Click a slider's number for exact entry. Model downloading is deferred.

## Verification

The standard build runs core and streaming integration tests. Core coverage includes GGUF inspection, folder scanning, invalid files, split models, projector ambiguity, recommendation fallback, profile persistence and expert-toggle arguments. The integration test covers generation, reload and cancellation.

`freetoken-native.exe --visual-check <output-folder>` renders the application's own widgets without opening a desktop window. It uses isolated settings under that output folder, checks override/restore behavior, and writes screenshots plus `verification.json`. It does not load model weights or start an inference server.

## Chat and Q4 KV cache update

The sidebar's **New chat** button starts a separate conversation. Previous chats remain selectable and are saved in `chats.json`; existing `conversation.json` history migrates on startup. Switching is disabled while generating. New replies persist generation **t/s** and TTFT across refreshes and restarts. Old replies without saved speed measurements cannot recover them retroactively.

With Auto enabled, changing KV types recomputes the memory profile at the selected context size. The chosen types survive model reload, and quantized values enable Flash Attention. Pascal automatic profiles reject quantized values because they disable Flash Attention; Q4 keys with F16 values remain available. Mixed-type KV estimates are conservative for aggregate layer metadata. Small GPUs reserve additional runtime headroom instead of assigning all estimated spare memory to expert cache.

TTFT starts at the inference request and ends at the first nonempty content or reasoning delta. Empty role events no longer count. Thinking can delay the final answer after first reasoning text, and cold prompt processing/expert transfers still contribute to TTFT. Prompt caching is requested explicitly for follow-up turns.

