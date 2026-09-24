# Lightweight native GUI and automatic startup planner

Planning proposal, September 9, 2026. No application implementation or new benchmarks performed for this plan.

## Recommended direction

Build a C++17 desktop application using FLTK 1.4 and CMake. It controls this workspace's patched llama-server as a separate child process over loopback HTTP, with streamed chat responses. Provide model management, chat, complete supported settings, startup planning, optional calibration and reusable profiles.

Treat the patched llama.cpp engine as the primary backend. Expose its existing FreeToken-inspired features directly. Add an optional adapter for an installed FreeToken engine so FreeToken-only settings can also be used without implying that the llama.cpp fork implements them. The GUI and primary engine must run natively on Windows and Linux; support for an external FreeToken installation depends on that installation's actual capabilities.

## Language and toolkit decisions

| Choice | Decision and reason |
| --- | --- |
| C++17 | Fits the existing C++/CMake code, has no interpreter or managed runtime requirement, and allows small native OS process and hardware adapters. Use RAII, bounded queues and explicit ownership for worker tasks. |
| FLTK 1.4 | Recommended because minimum overhead is the strongest stated priority. Compiled desktop GUI with Windows and Linux Wayland/X11 support. Use its normal event loop and software drawing; no continuous rendering loop or GPU context needed for the interface. |
| Native appearance | FLTK draws its own controls. It is a native compiled application, but does not use standard OS widgets throughout. This is the main visual/accessibility tradeoff; keyboard navigation, IME, Unicode and assistive technology need an early acceptance check. |
| wxWidgets alternative | Prefer this if standard Windows/GTK controls and accessibility take priority over the smallest footprint. Decide after the initial UI spike, before building the full settings editor. |
| Qt Widgets | Viable for a richer interface, but its additional deployment components are not my first choice for this footprint target. |
| Rust | Viable, but adds another language/toolchain and GUI integration choices without a clear benefit for this existing C++ project. |
| Python, Electron, webview UI | Avoid for the primary desktop application because a runtime/browser dependency conflicts with the requested emphasis on footprint. Python can remain in existing offline development scripts. |
| HTTP client / JSON | Use libcurl for HTTP and streaming, and the JSON library already available in the engine dependency tree after checking reuse boundaries. Pin dependencies; avoid introducing a separate web framework. |
| Persistence | Versioned JSON for preferences and profiles; append-only per-conversation records with lazy loading for chat history. Atomic replacement for configuration files. No always-running database service. |

Toolkit references: [FLTK manual](https://www.fltk.org/doc-1.4/preface.html), [FLTK basics](https://www.fltk.org/doc-1.4/basics.html), [wxWidgets native controls](https://wxwidgets.org/about/), [Qt deployment](https://doc.qt.io/qt-6/deployment.html). These establish platform/toolkit characteristics; application footprint must be measured.

## Footprint goals

These are engineering targets, not measured results or guarantees:

- GUI alone: aim for <= 40 MiB idle private memory, with 60 MiB as an initial investigation threshold. Report Windows private bytes and Linux private/PSS measurements explicitly; they are not identical metrics.
- Idle CPU: under 0.5% of one logical core over a 60-second idle interval on the reference machines.
- Window visible and interactive in under one second on a warm local launch, excluding model loading.
- Aim for <= 20 MiB compressed GUI-only package, accounting for bundled dependencies. Report engine, CUDA libraries and models separately.
- No dedicated GPU allocations for GUI rendering, no background inference, and no full model scan on every launch.
- Telemetry updates at roughly 1 Hz while visible, slower or suspended while hidden. Coalesce streaming text updates and bound log/history memory.
- Measure with the engine stopped and running, with a large saved history, and during token streaming. Reject unbounded transcript growth in the display widget.

## Architecture

```text
FLTK desktop window
  |-- Chat, Models, Model Settings, Startup Planner, Diagnostics
  |-- Settings registry + validation + profiles
  |-- Pure C++ memory and candidate planner
  |-- Worker queue: metadata, HTTP/SSE, calibration, telemetry
  `-- Backend adapter and process supervisor
        |-- patched llama-server: primary, native Windows/Linux
        `-- installed FreeToken: optional, capability checked
```

Keep CUDA and model allocations out of the GUI process. A server crash should leave the window, history and diagnosis available. The process separation also lets a stopped model release its full engine allocation and avoids binding the GUI to unstable llama.cpp context structs.

Use a small engine-side probe/helper or a narrowly scoped server extension for structured information missing from current APIs. Compile metadata inspection against the pinned GGUF/ggml readers instead of implementing a second GGUF parser. Read tensor descriptors without loading weight payloads; support split GGUF files.

Use existing chat and status endpoints where available. The local server documents `/health`, `/props`, `/slots`, `/metrics`, `/v1/models` and `/v1/chat/completions`; individual optional endpoints require capability checks. Do not infer fork behavior from the current upstream README.

Supervise child processes with CreateProcessW and a Job Object on Windows, and an argument-vector spawn plus process group on Linux. Capture stdout/stderr asynchronously. Own and stop only processes launched by this app. Handle port conflicts, readiness timeouts, cancellation and crash recovery. An attach-to-server mode must not terminate an externally owned server.

Bind managed servers to loopback by default. Keep environment overrides scoped to the child. Pass arguments without a shell; provide correctly escaped command exports separately. Use the platform configuration directory, including a deliberate portable-mode option. Redact credentials from exported diagnostics.

## User interface

- **Chat:** streaming replies, stop, regenerate, edit-and-resend, system prompt, conversation history, copy/export, supported reasoning display and model selection. Add modest formatting and code blocks without a browser renderer.
- **Models:** local model folders, import existing files, metadata, quantization, estimated RAM/VRAM, load/unload, per-model defaults and current engine. Later add explicit Hugging Face search/download with progress, cancellation, resume and disk-space checks.
- **Model Settings:** searchable categories, Basic/Advanced/All views, defaults, units, help, dependent controls and reset per field. Do not allocate all setting widgets until their category is opened.
- **Startup Planner:** context and concurrency goals, Balanced/Fastest Decode/Low Memory presets, automatic recommendations, manual locks, estimated memory breakdown and an explanation for each selected or rejected setting.
- **Diagnostics:** current requested and effective settings, engine version, launch command, RAM/VRAM, prompt/decode rate, time to first token, cache hit/miss and graph capture/replay where available. Distinguish engine counters from whole-process memory.

Most users should be able to select a model and click Load using Auto. Advanced users can inspect every applicable engine setting and override individual planner choices.

## Complete settings coverage

Define 'all settings' as every public launch option and generation/request setting in the selected supported backend version, plus explicitly supported tuning environment variables. Build-time features and read-only model architecture metadata are visible with their status, not presented as working runtime toggles. Internal developer fields are classified separately; exposing an arbitrary dataclass field does not make it configurable.

Create a versioned settings registry containing ID, backend, CLI/API/environment mapping, type, range/choices, unit, default source, dependencies, applicability, persistence scope and application timing. Mark each field as next-request, server restart, supported runtime action, or build-time. Distinguish inherited/Auto from explicit zero, false and empty values.

For the fork, add structured CLI schema export at the existing argument registration layer. Supplement with the request schema and cross-field validation. For FreeToken, use a versioned adapter and, where practical, export argparse metadata from its installed environment. Do not import its Python engine into the GUI. Until structured export exists, use an audited registry for the pinned binary and compare it with `--help`; help parsing alone cannot discover safe dependencies.

Add a coverage check: every public option in the pinned parser must have a control or an explicit action/read-only/build-only classification with a reason. Unknown new options are shown in an advanced discovery list, not silently discarded or guessed. A raw argument editor is an escape hatch, not the definition of full GUI support.

### llama.cpp settings groups

| Group | Coverage |
| --- | --- |
| Model loading | Model/shards, adapter paths/scales, supported projector inputs, memory mapping/locking, tensor placement, devices, GPU layers and supported split modes. Quantization is file metadata; changing it requires selecting/converting a model file. |
| Context and memory | Context length, logical/physical batch sizes, KV types and placement, attention settings, RoPE/YARN overrides, context shifting, prefix/slot cache settings. Preserve model-specific defaults unless deliberately overridden. |
| CPU | Generation and batch threads, affinity and NUMA controls, priorities and applicable polling settings. |
| Generation | Sampler chain, temperature/dynamic temperature, top-k/top-p/min-p/typical sampling, supported additional samplers, seed, penalties, DRY, Mirostat, output limit, stops, logit bias, grammar/JSON schema. |
| Chat and API | Chat template and template arguments, reasoning options, system prompt, host/port, authentication, concurrency, timeouts, metrics and logging. |
| Additional capabilities | Speculative decoding/draft configuration, embeddings, reranking, multimodal input and tool configuration where supported. UI availability follows actual engine and model capabilities. |

### Existing FreeToken-inspired settings in this fork

| GUI control | Current mapping / constraints |
| --- | --- |
| GPU expert cache | `--moe-gpu-cache-mib`; single parallel sequence (`-np 1`) required. |
| CPU expert cache / tile budget | `--moe-cpu-cache-mib`, `--moe-cpu-tile-mib`; current CPU path requires `-ngl 0 --no-op-offload --no-kv-offload -np 1`. |
| Expert GPU | `--moe-device`. |
| Memory headroom | `--moe-gpu-reserve-mib`; GUI also estimates untracked backend scratch and graph overhead. |
| Transfer and compute budget | `--moe-staging-mib`, `--moe-compute-mib`; compute arena does not include all backend scratch. |
| CPU/GPU miss scheduling | `--moe-cpu-miss-percent`: 0 GPU misses, -1 calibrated split, 1..100 fixed CPU share. |
| Transfer overlap | `--moe-pipeline`; separate from FreeToken's prefill overlap implementation. |
| Native numerical mode | `--moe-fast`; explicit opt-in, since it may change numerical results and generated tokens. |
| Semantic checkpoints | `--semantic-checkpoints`, `--semantic-checkpoint-mib`; enforce CPU attention, compatible GPU experts and no draft restrictions from the implementation. |
| CUDA graphs | Built with `GGML_CUDA_GRAPHS`; runtime disable through presence of `GGML_CUDA_DISABLE_GRAPHS`. Unset the variable to permit graphs: setting it to `0` still disables them in this fork. |

Source: [fork argument definitions](llama.cpp/common/arg.cpp), [runtime gap report](FREETOKEN_GAP_CLOSURE.md), [GPU graph report](GPU_RESIDENT_EXPERTS.md).

### Additional FreeToken settings

Expose these through the optional FreeToken adapter when the installed version supports them. For the primary fork, show the actual equivalent or an explanatory unavailable state; never send FreeToken flags to llama-server.

| Group | FreeToken controls and mapping decisions |
| --- | --- |
| Cache planning | `--memory-ratio`, mutually exclusive expert slots/rate/auto, `--kv-reserve-tokens`, KV pages/tokens, page size. Implement the planning concept for the fork using its byte budgets, not a direct slots-to-MiB identity. |
| Execution | Attention backend, MoE backend, NVFP4 backend, dtype, PLE backend and expert loading strategy. Triton, FlashInfer and Marlin choices are not automatically available in llama.cpp. |
| Hybrid scheduling | CPU worker threads, CPU layer selection and hybrid maximum fetch. CPU layer selection is not equivalent to the fork's per-step miss percentage. |
| Prefix and state caches | Radix/naive, applicable recurrent-state and SWA controls, special-token checkpoints and cache reporting. FreeToken radix caching and the fork's bounded full semantic snapshots have different semantics. |
| Scheduling and graphs | Concurrent requests, maximum sequence length, prefill chunk length, supported graph batch controls, prefill overlap and cache-hit device copies. |
| Operations | Server defaults, reasoning/tool parsers, sampling defaults, logging, metrics and supported cache management actions. |

Some state-cache ratios and graph batch lists exist in internal configuration without a public CLI flag in the inspected version. Mark them read-only/internal unless an explicit supported engine interface is added. Current cache eviction choices only include LRU; do not invent alternatives.

The reference has a CUDA minimum-version inconsistency for prefill device-side hit copies: an EngineConfig comment says 12.8 while CLI help says 13.0. Resolve against the installed implementation/build capability before enabling it. This illustrates why the registry needs a backend version and actual capability probe.

Sources: [FreeToken CLI](FreeToken-reference/docs/cli.md), [server argument definitions](FreeToken-reference/python/freetoken/server/args.py), [engine configuration](FreeToken-reference/python/freetoken/engine/config.py), [cache budget policy](FreeToken-reference/python/freetoken/engine/cache_budget.py).

## Automatic startup planner

### 1. Detect hardware and model

- CPU model, physical/logical topology, hybrid core layout where available, usable instruction sets and NUMA nodes.
- Available and total RAM, OS/container memory limits, and relevant pinned/locked memory limits. Do not budget against total RAM alone.
- GPUs by stable UUID, architecture/compute capability, total/free VRAM, driver and runtime; discover compiled backend/kernel capabilities through the selected engine.
- GGUF architecture, layer count, dimensions, attention/KV heads, expert count/top-k, tensor quantization and actual tensor storage sizes, context limit, sliding-window/recurrent state and split files. Use an engine-specific metadata probe for other checkpoint formats.
- Missing probes produce an 'unknown' state and conservative defaults. Optional NVML or `nvidia-smi` can provide telemetry; starting the GUI must not require CUDA.

### 2. Establish requirements and memory budget

Respect user-locked context, concurrency, numerical mode and device choices. Default to one active chat/model for the custom offload path. Offer Balanced, Fastest Decode and Low Memory objectives without assuming a larger expert cache always wins.

Estimate resident non-expert weights, expert cache, KV cache, SWA/recurrent state, activations, staging, workspace, graph overhead and a reserve. Account for both RAM and VRAM, loading peaks, per-device placement, alignment and actual quantized tensor sizes.

```text
usable VRAM = measured free VRAM - safety reserve
available for caches = usable VRAM - resident weights - workspace/activations
                       - graph allowance - other fixed device state
```

For ordinary attention, per-layer K/V dimensions and cache element/block sizes yield an initial KV estimate. SWA and recurrent architectures require their own formulas and allocation rules. Validate estimates against the engine's allocated sizes after startup. Track pre-load versus post-load measurements explicitly so weights or reserve are not subtracted twice.

Reserve the requested context/concurrency capacity first, then allocate an expert cache within the remaining budget. Consider smaller batches, expert cache and GPU placement if necessary. Never silently lower a locked context target or enable a different numerical mode; report infeasibility with concrete alternatives. Recheck free memory immediately before allocation and allow a bounded conservative retry for unlocked settings.

### 3. Select threads and supported execution paths

Start from usable physical cores while leaving the desktop responsive. Tune prefill and decode threads separately; account for CPU expert work competing with attention and transfer threads. NUMA/affinity are explicit advanced choices until measured.

Kernel eligibility depends on hardware ISA, engine build, driver/runtime, model architecture, quantization, tensor shape and execution mode. Prefer engine Auto dispatch with legal settings; expose a manual kernel override only when the engine implements it. Do not claim NVFP4, BF16 or modern attention kernels on hardware merely because a menu lists them.

Keep compatibility numerical mode as the automatic default. Benchmark GPU-only, fixed and adaptive misses where legal: existing project measurements already show hybrid execution can be slower.

### 4. Enable compatible CUDA graphs

Use Auto / Off / On-if-supported, with separate requested, eligible, captured and replaying states. Build support and a CUDA driver alone are insufficient; check engine architecture guards, capture-safe operations, stable buffers/shapes and available graph headroom.

Critical current constraint: `ggml-cuda.cu` disables graphs below `GGML_CUDA_CC_VOLTA`. P100 is SM60 and therefore uses eager execution under the current implementation. This is an engine policy, not proof that P100 lacks basic CUDA Graph support. The GUI must not silently bypass the guard. Add a separate experimental P100 graph investigation, described below, before promoting any SM60 graph path into Auto.

On supported hardware, run warmup and confirm nonzero replay counters before claiming graph acceleration. Add structured telemetry if counters are only available in library reports/logs. On capture failure, disable graphs for that profile and retry safely. Shape/cache changes invalidate captures and may require a server restart. Distinguish expert-subgraph replay from full-token graph capture; the project currently implements the former.

### 5. Optional bounded calibration

Provide a cancellable Quick Calibration action. Aim for 30-90 seconds of measurement after model loading/warmup; loading, compilation and graph capture can make total elapsed time longer. A strict overall deadline stops the candidate search and keeps the best validated completed candidate.

First measure CPU expert work and host-to-device transfer, including concurrent behavior where relevant. Then evaluate a small candidate set for threads, batch sizes, expert/KV split, miss scheduling, pipeline and graphs. Keep prompt, output length, context and numerical mode fixed when comparing candidates. Avoid an exhaustive search or switching several variables without a baseline.

Record prompt tokens/s, decode tokens/s, first-token latency, peak memory, cache hits/misses, replay counts and errors. Exclude warmup and repeat finalists if the budget allows. Optimize the chosen objective, retain the conservative candidate when differences are within measurement noise, and label partial calibration honestly.

Run a finite-output/smoke check and same-mode consistency checks for candidates; calibration is not a substitute for pretrained numerical acceptance. Never change quantization or enable `--moe-fast` just to win a speed measurement.

### 6. Save and reuse machine/model profiles

Persist hardware fingerprint, GPU UUID/architecture, model identity/quantization, engine binary/build hash, driver/runtime, OS, planner version, context/concurrency, numerical mode, effective settings, measurements and reasons. Use a fast file identity for startup; compute/cache a strong model fingerprint asynchronously and invalidate on changes, including split shards.

Revalidate current free memory on every load even if the profile matches. Invalidate or mark calibration stale after a relevant model, engine, driver, hardware or workload change. Store user overrides separately and preserve them during replanning. Keep a last-known-good profile and a visible reset/recalibrate action.

Precedence: explicit session overrides, model-specific user overrides, valid machine/model recommendations, application defaults, backend defaults. Saved measured recommendations never override a user lock.

## Applying settings correctly

Generation settings apply to the next request. Load-time settings trigger a controlled stop/restart with the transcript preserved and reprocessed as necessary; a transcript is not a portable KV checkpoint. Show the pending change summary and restart requirement in the UI.

The fork's `llama_moe_reconfigure` currently requires an empty single-sequence context, exclusive access and temporary memory for overlapping old/new allocations. It is a library API, not an HTTP cache rebuild endpoint. Initial GUI delivery uses restart. A later engine endpoint would need draining, checkpoint invalidation, failure recovery and explicit active-chat behavior.

FreeToken has a separate `/v1/cache/rebuild` control interface. Use it only through its adapter after verifying version, capacity units and supported semantics. Do not reuse it for the fork.

## Delivery order and acceptance gates

1. **Footprint and usability spike:** CMake GUI target, model picker, transcript widget, worker events and minimal server connection on Windows and Linux. Measure footprint, Unicode/IME, keyboard access, scaling and X11/Wayland. Decide FLTK versus wxWidgets here.
2. **Useful chat application:** process supervision, local model library, load/unload, streaming/cancel, history and core sampling. Verify crash recovery, port conflicts, long replies, paths with spaces/non-ASCII characters, and responsive cancellation.
3. **Full settings registry:** structured schema support, complete pinned-backend coverage, per-model persistence, dependency rules and restart flow. Every public setting must round-trip without unit loss and reach the intended CLI/API/environment field.
4. **Deterministic startup planner:** metadata/hardware probes, architecture-aware budgeting, eligibility explanations and manual locks. Unit tests cover dense/MoE, SWA/recurrent layouts, quantized tensors, missing probes and insufficient memory. Validate estimates against actual allocations.
5. **Calibration and graph verification:** bounded search, warmup/replay telemetry, cancellation, profile invalidation and last-good recovery. Test changing memory availability and graph failures without losing history or leaving orphaned servers.
6. **FreeToken adapter and extended workflows:** installed-version capability checks, its full public settings, cache controls, supported model downloads and relevant multimodal/advanced UI. Backend-specific limitations stay visible.
7. **Distribution and hardware acceptance:** Windows portable ZIP and Linux package/portable bundle after dependency validation. Ship the GUI separately from optional engine/CUDA packages. Maintain a CUDA 12.x SM60 engine build for P100 and suitable builds for newer GPUs; never assume one binary works everywhere.

Release matrix: native Windows and Linux CPU; available RTX hardware; actual P100 on the promised platforms; both target pretrained Qwen and Gemma models; relevant context lengths; cancellation/reload and sustained memory pressure. Unsupported combinations must be clearly identified. Existing RTX results do not establish P100 acceptance.

Proposed source layout: `desktop/` for application, `desktop/core/` for planner/settings/profiles, `desktop/platform/` for OS adapters, `desktop/backends/` for engine adapters and `desktop/tests/` for behavior tests. Make only narrow changes inside llama.cpp for schema, metadata and telemetry exports. Keep inference optimization work independent of UI changes.

## What success looks like

Select a model, choose a context goal, and click Load. The app explains its memory/thread/kernel choices, starts a compatible engine, streams chat, exposes all supported settings, and optionally calibrates once for that machine/model. On later launches it reuses validated choices while checking available memory. On a P100 it clearly reports that the current backend uses eager execution because CUDA graphs are architecture-disabled.

## Follow-up research: P100 graph enablement

Online research on September 9 found a concrete starting point, but no verified drop-in fix for this fork on P100:

- An [NVIDIA support example](https://forums.developer.nvidia.com/t/cuda-graph-in-cuda-fortran/136290/1) confirms basic CUDA graph code running on P100 under Linux. Basic API support does not establish capture compatibility or performance for this engine.
- In [upstream PR 25749](https://github.com/ggml-org/llama.cpp/pull/25749), a maintainer explains that the original older-GPU restriction reflected performance regressions with earlier graph-management logic. Volta/Turing were subsequently enabled.
- [Open PR 27721](https://github.com/ggml-org/llama.cpp/pull/27721) lowers the graph gate to SM61 and reports about 40% higher decode throughput for one MoE workload and 6-8% for a dense workload on GTX 1080/Windows. Its author explicitly leaves SM60/P100 excluded because they lack that hardware. Those results cannot be assigned to P100 or this fork.
- [Merged PR 26574](https://github.com/ggml-org/llama.cpp/pull/26574) addresses cuBLAS workspace handling associated with older-GPU graph issues. The local fork already contains explicit per-stream cuBLAS workspaces; audit equivalence and lifetime behavior instead of blindly applying the patch again.

Proposed experiment: build with CUDA 12.x targeting SM60 and graphs enabled, add a narrowly scoped experimental SM60 opt-in to the architecture policy, and retain operation-level capture compatibility checks. The local `ggml_cuda_mul_mat_id_needs_sync` path still blocks capture where CPU routing/stream synchronization is needed. Start with stable single-token expert subgraphs and keep unsupported sections eager. Audit capture failures, workspace lifetime and cache invalidation before expecting automatic recovery.

On real P100 hardware, first run a basic capture/replay probe; then numerical parity, actual replay-counter checks, long generation, cancellation, eviction/resize and memory-soak checks. Benchmark the same binary with graphs permitted versus disabled. Enable a saved experimental profile only when that machine/model passes correctness and shows a repeatable gain. Windows and Linux need separate validation.

The cache HTTP limitation is independently addressable: add an engine-owned control endpoint that drains requests, clears the old sequence state, rebuilds the empty context while retaining the loaded model, reattaches runtime state and invalidates incompatible checkpoints. The GUI can replay the transcript afterward. This can avoid reloading model weights, but is not seamless resizing of an active KV cache. The current transactional API also needs temporary allocation headroom; retain full restart as a fallback.
