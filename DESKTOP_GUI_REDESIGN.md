# LlamaCPP P100 desktop GUI redesign — plan

Product name for the GUI: **LlamaCPP P100** (renamed from "FreeToken" on September 10, 2026 — see Naming and brand in §5). References to "FreeToken" elsewhere in this document refer to the upstream FreeToken reference project, not this GUI.

September 10, 2026. Originally research and plan only.

> **UPDATE — implemented the same day.** The first working version ships in this repo: [desktop/control.cpp](desktop/control.cpp) + [desktop/host.cpp](desktop/host.cpp) (C++ control plane) and [desktop/ui/](desktop/ui/) (the web frontend), built as **`llamacpp-p100.exe`**. See [Implementation status](#11-implementation-status--what-shipped-on-day-one) and [desktop/README.md](desktop/README.md). Run `build\desktop\llamacpp-p100.exe`; it opens `http://127.0.0.1:18432/` automatically. End-to-end verified on the RTX 4060 machine: model load, streaming chat at 17.2 tok/s, TTFT 293 ms, live VRAM/RAM telemetry, and automatic settings migration from the old FLTK app. The IA shipped with the §4 reorganization: four pages (Chat / Models / Tune / Monitor) instead of the five mockup pages.

## Problem statement

The current FLTK app ([desktop/main.cpp](desktop/main.cpp)) has a solid engine core but a presentation layer that cannot reach the polish of LM Studio or the FreeToken Desktop console. Concretely:

| User-visible gap | Today |
| --- | --- |
| Tokens/s (t/s) | Never shown; timings stream past invisibly |
| Model loading | A status label "Loading model - Stop cancels startup"; no progress %, stages, VRAM view, or log while loading |
| Live information | One muted status line at the bottom; no gauges, counters, context fill, or cache telemetry |
| Organization | 5 pages with monolithic panels; chat is a plain-text transcript; startup review is a monospace text dump |
| Slickness | Hand-drawn FLTK widgets ([Switch/Slider/Disclosure in main.cpp:80-133](desktop/main.cpp#L80)) cap out before markdown, code blocks, animation, and responsive layout |

The two reference UIs the user admires are both web-tech UIs: LM Studio is Electron + React; the FreeToken Desktop console (see [FreeToken-reference/assets/desktop-console.png](FreeToken-reference/assets/desktop-console.png)) is a web console backed by an HTTP control plane ([FreeToken-reference/python/freetoken/daemon/app.py](FreeToken-reference/python/freetoken/daemon/app.py)). Even the bundled llama-server now embeds its own web UI (`llama_ui_get_assets()` in [llama.cpp/tools/server/server-http.cpp](llama.cpp/tools/server/server-http.cpp)) — opening `http://127.0.0.1:18080` today already shows a modern chat UI. The plan below keeps the C++ engine core and replaces the presentation layer with the same architecture: **C++ control plane + web frontend + native window shell**.

This deliberately revisits one decision in [NATIVE_GUI_PLAN.md](NATIVE_GUI_PLAN.md) ("avoid webview UI"). That decision optimized for footprint; this redesign optimizes for the user's stated priority — LM Studio-class presentation and information density. Footprint impact is quantified under Risks.

---

## 1. Architecture

```text
+--------------------------------------------------------------+
|  WebView2 shell window (Win32, thin)   or  default browser   |
|  loads http://127.0.0.1:<app-port>/  (SPA served by app)     |
+--------------------------------------------------------------+
|  Frontend: Vite + TypeScript + React + Tailwind (embedded)   |
|    pages: Chat / Models / Engine / Logs / Settings           |
|    REST for actions + one SSE stream for live events         |
+--------------------------------------------------------------+
|  App control plane (existing C++ desktop core, embedded      |
|  cpp-httplib already linked):                                |
|    /app/api/state  /app/api/events (SSE)  /app/api/...       |
|    settings registry · GGUF reader · recommender             |
|    process supervisor · telemetry · log ring                 |
+--------------------------+-----------------------------------+
                           | loopback HTTP (existing flow)
                  +--------v--------+
                  | patched         |
                  | llama-server    |
                  +-----------------+
```

Why this and not the alternatives:

| Option | Verdict |
| --- | --- |
| **Keep C++ core; web UI served by the app; WebView2 shell (recommended)** | Reuses 100% of [desktop/core.cpp](desktop/core.cpp) (settings registry, GGUF inspector, recommender), [desktop/library.cpp](desktop/library.cpp) (scanner, NVML), [desktop/platform.cpp](desktop/platform.cpp) (Job-Object supervisor). The UI layer gets the full web design surface: markdown, code highlighting, animated gauges, sliders with live chips, progress bars, sparklines. Same architecture as LM Studio and FreeToken Desktop. |
| Qt / QML rewrite | Viable slickness, but huge dependency, C++ UI code remains slow to iterate, and design ceiling is still below web for markdown/chat rendering. |
| Electron/Tauri + Rust/Go rewrite | Discards the working C++ core or needs an FFI layer; new toolchain for no user-visible gain over WebView2 hosting. |
| Polish FLTK further | Every target feature (markdown chat, progress staging, live gauges, sparklines) is custom drawing; [main.cpp](desktop/main.cpp) already spends ~130 lines faking switches and sliders. This is the dead end. |

The frontend is developed with Vite (`desktop/ui/`), built to static assets, and embedded into the executable the same way llama-server embeds its UI assets. During development the SPA runs on the Vite dev server with a proxy to the app's control plane. The WebView2 shell is a thin Win32 window hosting the OS-shared WebView2 runtime (preinstalled on Windows 10/11); **the browser fallback is always available** — if the shell is absent, the app opens the default browser. That keeps the app usable even with a broken runtime and makes the UI testable headlessly.

---

## 2. Control plane (C++ side)

The GUI process gains an embedded HTTP server on a fixed loopback port (default `18432`, configurable, probed for availability). The frontend talks **only** to this control plane; the app proxies inference to llama-server so the server can restart mid-conversation without breaking the chat page. JSON is camelCase, matching FreeToken's daemon convention.

```text
GET  /app/api/state          full snapshot: version, settings, library, hardware, engine status, active model
GET  /app/api/events         SSE: every live event (see schema below)
POST /app/api/engine/load    {modelId} – start/restart server with current profile
POST /app/api/engine/unload
GET  /app/api/models         scanned library incl. per-model fit estimate
POST /app/api/models/scan    {folder}
POST /app/api/models/select  {modelId, projectorPath?}
POST /app/api/models/import  {path}
GET/PUT /app/api/settings    validated registry round-trip (existing ft::validate)
GET  /app/api/startup-review structured plan (replaces describe_plan text dump)
POST /app/api/chat           {messages, sampling…} → SSE proxy of the completion stream
POST /app/api/chat/cancel
GET  /app/api/logs           SSE log tail from an in-process ring buffer (last 5k lines)
GET  /app/api/telemetry      1 Hz tick: RAM/VRAM, engine tok/s, context fill (polled by state, not SSE, if simpler)
```

Two structural changes to existing code:

1. **Live stdout pipe.** [platform.cpp](desktop/platform.cpp) currently redirects llama-server stdout/stderr to `server.log` and the GUI only tails the file on demand. The supervisor instead creates a pipe, feeds an in-process ring buffer (line count + seq numbers, like FreeToken's log ring), and flushes to `server.log` in the background. This one change unlocks live load progress, the Logs page, and crash forensics.
2. **Event bus.** The `Update` queue in [main.cpp:280-284](desktop/main.cpp#L280) already merges token deltas and lifecycle events — formalize it: a single thread-safe queue drained by the control-plane thread, published to SSE subscribers as typed JSON events.

### SSE event schema (first pass)

```jsonc
{"type":"engine.state","state":"loading|ready|stopped|error","detail":"…"}
{"type":"engine.progress","stage":"text_model","value":0.42}          // see §4
{"type":"chat.token","convId":"…","delta":"…"}
{"type":"chat.reasoning","convId":"…","delta":"…"}
{"type":"chat.usage","convId":"…","ttftMs":284,"promptN":42,"predN":318,
 "predPerSecond":20.8,"promptPerSecond":412.1,"stopReason":"eos"}
{"type":"library.updated","models":[…]}
{"type":"log.line","seq":1041,"level":"INFO","source":"llama-server","text":"…"}
{"type":"telemetry.tick","ramUsedMib":5102,"ramTotalMib":15872,
 "vramUsedMib":6100,"vramTotalMib":8187}
```

---

## 3. Where every number comes from (verified against this repo's llama-server)

The "way more information" goal is not aspirational — this exact server build already exposes nearly all of it. Verified sources:

| Widget / stat | Source (verified in tree) |
| --- | --- |
| Per-message **tok/s**, TTFT, prompt/gen token split | Client-side chunk timing plus the `timings` object the server attaches to stream chunks and the final `usage` chunk via `stream_options.include_usage` — both present in [server-task.cpp:495-516](llama.cpp/tools/server/server-task.cpp#L495); timings JSON (`prompt_per_second`, `predicted_per_second`, …) is populated at [server-task.cpp:1050-1060](llama.cpp/tools/server/server-task.cpp#L1050) |
| **Prefill progress bar** while the prompt processes | `prompt_progress` field on stream chunks (`is_progress` / `progress.to_json()` in [server-task.cpp:1063-1067](llama.cpp/tools/server/server-task.cpp#L1063)) |
| **Live engine tok/s** on the Engine page | Computed in the proxy layer from per-second deltas; cross-checkable against `llama_prediction_per_second` in `/metrics` (enable `--metrics`, [arg.cpp:3641](llama.cpp/common/arg.cpp#L3641)) |
| **Context fill + KV cache hits** gauge | `GET /slots` (enable `--slots`): slot JSON carries `n_ctx`, `n_prompt_tokens_processed`, `n_prompt_tokens_cache`, `n_decoded`, `is_processing` ([server-context.cpp:642-667](llama.cpp/tools/server/server-context.cpp#L642)) |
| Model metadata (arch, layers, max ctx, experts) | Existing GUI GGUF reader ([library.cpp:42-59](desktop/library.cpp#L42)) plus `GET /props` at runtime |
| **VRAM / RAM gauges** | NVML loaded dynamically already ([library.cpp:100-109](desktop/library.cpp#L100)) — extract into a telemetry module that refreshes at 1 Hz; RAM via `GlobalMemoryStatusEx` (already in [core.cpp:133-141](desktop/core.cpp#L133)) |
| **Load progress %** | See §4 |
| Lifetime counters (tokens processed, requests) | `/metrics` Prometheus counters, scraped once per minute into the Engine page |

Launch-flag changes only: the app adds `--metrics --slots` (both request-scoped, no behavior change to the engine) to its argument builder in [core.cpp:81-103](desktop/core.cpp#L81).

### §4 Model-load progress — three tiers

In single-model mode (`--model <path>`, what the GUI launches today) the server does **not** expose load progress over HTTP; the state callback that feeds `/models/sse` in router mode is wired only via `set_state_callback` ([server-context.cpp:4210](llama.cpp/tools/server/server-context.cpp#L4210)). The loader itself already produces throttled 0..1 progress with stages (`text_model`, `mmproj_model`, `spec_model`) in `load_progress_callback` ([server-context.cpp:924-962](llama.cpp/tools/server/server-context.cpp#L924)). Three options, in increasing strength:

1. **Tier 1 (no patch): staged log heuristic.** Parse the piped stdout ring for known stage markers (`llama_model_loader`, `load_tensors: offloaded X/Y layers`, `warmup`, `server is listening`). Show a 5-stage checklist with a spinner instead of a fake percentage. Deliverable immediately with the control plane.
2. **Tier 2 (recommended, ~10-line fork patch): emit progress lines.** In `load_progress_callback`, when no router parent is attached, also `fprintf(stderr, "FT_LOAD_PROGRESS %s %.3f\n", stage, value)`. The GUI regexes the ring and drives a real 0-100% bar with stage labels — the exact pattern FreeToken's daemon uses for bench progress (`FTBENCH done total label` in [FreeToken-reference/python/freetoken/daemon/app.py:105-113](FreeToken-reference/python/freetoken/daemon/app.py#L105)). Consistent with this workspace's existing practice of narrow llama.cpp changes; gated behind an `#ifdef`-free one-liner that upstream would likely accept.
3. **Tier 3 (alternative, no patch): router mode.** Launch llama-server with `--models-dir` and use `GET /models`, `POST /models/load`, `GET /models/sse` ([server.cpp:227-231](llama.cpp/tools/server/server.cpp#L227), [server-models.cpp:2060-2072](llama.cpp/tools/server/server-models.cpp#L2060)) for real progress events. Deferred: per-model custom args (`--moe-*`) through the router are unverified in this fork, and it changes the lifecycle model.

The loading UI (shown as a full-height card in Chat and mini-bar in the top status pill): stage checklist → percentage bar (Tier 2) → live VRAM gauge rising as tensors land → last 3 log lines → Cancel. This is the single biggest "show what's going on" win.

---

## 4. Information architecture (the reorganization)

Replace the current 5 unequal pages with 5 purposeful pages + persistent live chrome. The sidebar and top bar follow LM Studio's proven shell; the Engine page adapts the FreeToken console's signature elements.

```text
┌─┬──────────────────────────────────────────────────────────┐
││  FreeToken            [● Ready · gemma-4-26B · 20.8 t/s] │  ← status pill (click = Engine)
││                                                          │  ← model quick-picker, port chip
││  💬 Chat        ┌────────────────────────────────────┐   │
││  📚 Models      │  page content                      │   │
││  ⚙ Engine       │                                    │   │
││  📜 Logs        │                                    │   │
││  🔧 Settings    │                                    │   │
││                └────────────────────────────────────┘   │
│  ── gauges ──                                             │
│  VRAM ▓▓▓▓▓▓▓░░ 6.1/8.0 GB   (sidebar bottom, 1 Hz)      │
│  RAM  ▓▓▓░░░░░░ 5.1/15.5 GB  GPU: RTX 4060  v0.3.0       │
└─┴──────────────────────────────────────────────────────────┘
```

Sidebar is collapsible to icons (FreeToken console does this), shows a live dot + badge (model count), and hosts the two memory gauges permanently — the user should never wonder "is the model on the GPU right now?"

### Chat (default page)
- Real message bubbles: user right-ish cards vs assistant plain-width rows, avatar chips, per-message action row (copy, regenerate, edit-and-resend).
- **Markdown + syntax-highlighted code blocks with copy buttons** (DOMPurify-sanitized rendering).
- Reasoning stream renders in a collapsible "Thought for 2.3 s" disclosure above the answer (the SSE plumbing for `reasoning_content` already exists at [main.cpp:386-391](desktop/main.cpp#L386)).
- **Per-message footer stats**: `21 tok/s · TTFT 284 ms · 318 tokens · eos` — sourced per §3.
- Composer: auto-growing textarea, Enter to send / Shift+Enter newline, Stop button morph, model chip showing loaded model + context fill `1.2k / 4k`, **image attach button enabled when a vision projector is paired** (mmproj plumbing already passes `--mmproj`; [desktop/README.md](desktop/README.md) notes attachments were "not implemented" — this closes that loop).
- Prefill progress bar appears between send and first token (`prompt_progress`).
- Conversation management: list of saved conversations (the JSON-per-conversation file convention from NATIVE_GUI_PLAN), rename/delete/export, "New chat". Empty state keeps the existing "A little space for big ideas." artwork.

### Models
- Card grid, one card per model: name, arch badge (gemma4/qwen), size, layers, max context, expert count, vision eye when a projector is paired, **fit estimate bar** (green/amber/red) computed from the existing `ft::recommend()` budget math.
- Card actions: Load (primary), Unload (when live), details drawer with the full GGUF metadata + recommended profile diff.
- Library tools: folder picker + rescan (existing), drag-and-drop GGUF import, manual projector pairing UI (upgrades today's raw path inputs), split-shard awareness (already handled by the scanner).
- While a load runs, the card shows the same progress bar as the Chat overlay.

### Engine (FreeToken console, adapted)
- Model status card: logo/initial, quant · params · ctx line, `● Running · 1h 12m` pill, live **TOKENS/S**, **REQUESTS**, **TTFT**, Chat and Stop buttons — directly modeled on the FreeToken console screenshot.
- Stat cards row: tokens processed this launch (in/out split), uptime, model file size, backend/build info (from `/props`).
- **Cache & memory config**: presets **Lean / Chat / Long-context** that map to the MoE knob bundles (`--moe-gpu-cache-mib` / `--moe-cpu-cache-mib` / reserve / pipeline); each slider shows a live derived-value chip (GiB, or tokens of expert coverage computed from GGUF dims) — FreeToken's KV/MoE sliders with `12.8 GiB` chips, applied to this fork's real flags. Dirty state shows "Apply & reload" (server restart flow already exists).
- **Startup review, visualized**: instead of the monospace dump in `describe_plan` ([core.cpp:142-152](desktop/core.cpp#L142)), a stacked horizontal memory budget bar (weights / KV / expert cache / graphs+scratch / reserve) for VRAM and RAM, each segment labeled with MiB and its driver, plus warnings (e.g. "P100: CUDA graphs disabled by architecture guard") as callout cards. Same honesty, readable presentation. The API returns this structured (`/app/api/startup-review`); the old text becomes the accessible fallback.

### Logs
- Live streaming view (SSE from the ring buffer): level filter, text search, autoscroll toggle, pause, copy/save, per-line severity color. Shows llama-server lines and app-side events in one timeline with source chips. The launch command (redacted) is displayed above — Diagnostics merges into this page, so the old Diagnostics page disappears as a separate destination.

### Settings
- Same registry and validation (unchanged C++), re-grouped and **searchable**: Basics (sampling, system prompt, thinking), Context & GPU, Attention & memory, FreeToken expert caching (kept visually separated as experimental), Advanced execution, App preferences (theme, port, server executable).
- Every control keeps its help text; gains: units in the value, "restart required" pill vs "next request" pill (the registry already knows this distinction), FreeToken Auto toggle with recommended-profile restore (existing), and a sticky "unsaved changes" bar.

---

## 5. Design system

- **Theme**: keep the current identity so it still feels like FreeToken — near-black surfaces `#111418`/`#16191d`, panel `#1e2227`, border `#31383f`, text `#e7ebef`, muted `#919ca7`, accent mint `#99E2C3` (these are the existing theme constants, [main.cpp:38-42](desktop/main.cpp#L38)). Add a proper light theme.
- **Type**: Segoe UI Variable (fallback Inter); tabular numerals for all live numbers so digits don't jitter at 1 Hz.
- **Space & shape**: 4/8 px spacing grid, 10-12 px card radius, 1 px borders, soft shadows only on overlays.
- **Motion**: 150-200 ms ease-out on hover/expand; animated progress bars and gauge transitions; streaming caret. Nothing bounces.
- **States**: every async control has idle/busy/success/error treatments (the current app communicates state only through one muted label).
- Dark/light via CSS custom properties; honor `prefers-color-scheme` on first run.

### Naming and brand (renamed September 10, 2026)

The GUI is named **LlamaCPP P100**. Applied at implementation time:

- Window title, sidebar brand block, taskbar/shortcut names, About text: **LlamaCPP P100**.
- Executable: `llamacpp-p100.exe` (Phase 6 shell; the FLTK `freetoken-native.exe` is deleted at parity, so no dual naming).
- Data directory: move from `%LOCALAPPDATA%\FreeTokenDesktop` to `%LOCALAPPDATA%\LlamaCppP100` ([core.cpp:109-118](desktop/core.cpp#L109)); on first run, copy `settings.json` and `conversation.json` from the old directory if the new one does not exist, so users keep their configuration.
- Keep the mint accent, palette, tagline ("Intelligence, on your terms.") and empty-state copy — the identity stays; only the name changes.
- Engine-level names (`--moe-*` flags, FreeToken Auto) keep their names: "FreeToken Auto" is a feature label carried over from the strategy's origin, not the product name. Renaming it is a user-facing decision, not part of this rename.

---

## 6. Frontend stack

| Choice | Pick | Why |
| --- | --- | --- |
| Build | Vite + TypeScript | Fast dev loop with HMR; static output embeds cleanly |
| UI | React 18 + Tailwind CSS | Same family as LM Studio; largest component ecosystem; Tailwind keeps styling consistent with the tokens above |
| State | Zustand (or nanostores) | Tiny, no provider boilerplate |
| Live data | native `EventSource` for `/app/api/events` + fetch-streaming for chat | No socket.io-style dependency |
| Markdown | marked + highlight.js + DOMPurify | Standard trio; sanitize everything |
| Charts | hand-rolled SVG (gauge, sparkline, stacked bar) | Three small components; avoids a chart library |
| Embedding | CMake: gzip assets → generated header, served with `Content-Encoding` | Same approach llama-server uses for its UI assets |

Accessibility baseline the FLTK version lacked: full keyboard nav, focus rings, ARIA roles on gauges/progress, HiDPI-correct rendering for free (browser renderer), IME support for free.

---

## 7. Delivery phases

Each phase ends usable; the FLTK binary stays shippable until the parity gate at Phase 5.

| Phase | Contents | Exit check |
| --- | --- | --- |
| **0. Control plane + browser mode** | httplib server in the app process, state/ events SSE, log pipe + ring, serve built SPA from `dist/`, open in browser. No shell yet. | `curl` shows state; browser UI renders shell with live engine status |
| **1. Shell + design system** | Sidebar/top bar, routing, theme tokens, **bottom status bar with live telemetry** (per §10), **LlamaCPP P100 branding + settings migration** from `FreeTokenDesktop` | Navigation + live RAM/VRAM and ready-state chips working off a running server |
| **2. Chat MVP** | Streaming bubbles, markdown/code, reasoning disclosure, stop, per-message stats, prefill bar, composer, conversation save (ports existing flows) | Integration scenario from [desktop/tests.cpp](desktop/tests.cpp) passes driven through the API; visual check via Playwright screenshots |
| **3. Models page** | Scanner UI, cards with fit bars, projector pairing, load/unload + Tier-1 staged loading UX | Load gemma + MiniCPM end-to-end from the page |
| **4. Load progress Tier 2 + Engine page** | `FT_LOAD_PROGRESS` fork patch (opt-in flag `--emit-progress` not needed; app-only), Engine console cards, startup-review visual, cache presets | Real 0-100% load bar; engine stats match `/metrics` within noise |
| **5. Settings + Logs polish** | Searchable settings, restart-required pills, logs page with filters, diagnostics merge | Registry round-trip test; log stream survives 10k lines |
| **6. WebView2 shell + packaging + cleanup** | Thin Win32 host window as `llamacpp-p100.exe` (window icon, taskbar, single-instance, safe-close → existing stop_request flow), script updates, delete FLTK UI layer (keep core/library/platform/tests) | Footprint measured and documented; `build-desktop.cmd` produces one exe |

Estimate: Phases 0-3 are the bulk of the user-visible win (loading, t/s, chat, organization). Phases 4-6 are where it stops feeling like a developer tool.

---

## 8. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Footprint regression vs NATIVE_GUI_PLAN goals (≤40 MiB idle) | WebView2 shares the OS runtime; measure private bytes at Phase 1 and Phase 6 and publish in the build doc. Browser fallback mode has near-zero shell cost. Acceptance is explicitly re-baselined: presentation priority beat the old target by user decision. |
| WebView2 runtime missing (rare Win10) | Detect and fall back to default browser with a one-line explanation; never block. |
| httplib SSE robustness (long-lived connections) | Keep SSE responses small and heartbeat-ed (15 s comment pings, like FreeToken's `_log_stream`); reconnect with `Last-Event-ID`; cap subscribers. |
| Frontend complexity creep | No router beyond 5 pages, no CSS framework beyond Tailwind, no chart lib; component count budget per page noted in Phase tasks. |
| `--metrics`/`--slots` overhead | Negligible for a single-slot local server; both are opt-in upstream flags. |
| Fork patch (Tier 2) divergence | One fprintf in an existing callback; wrapped so router mode is untouched; documented in the next `SOURCE_MANIFEST.json` revision alongside existing patches. |

## 9. Explicitly kept from the current implementation

- The entire C++ core: settings registry/validation, GGUF inspector, scanner, recommender, process supervisor, tests philosophy (headless integration scenario + visual-check becomes Playwright + API tests).
- The brand palette, tagline, and empty-state copy, under the new product name **LlamaCPP P100** (see Naming and brand in §5).

---

## 10. Mockup review — September 10, 2026 (adopted visual baseline)

Four full-page mockups (Model library, Model settings, Startup review, Diagnostics) were reviewed against this plan and are **adopted as the visual target**: same sidebar navigation, 12-column content cards, the hardware chips row, steppers-with-sliders settings pattern, and the new bottom status bar (`● Local inference ready · Model: … · CPU · RAM 6.7/6.9 GiB · N models`). Everything below is either confirmation or a gap to close during implementation.

### What the mockups get right (adopted as-is)

| Mockup element | Verdict |
| --- | --- |
| Bottom status bar with ready dot, active model, RAM/CPU chips, model count | Best addition. Realizes §4's persistent live chrome. **Requirement: the RAM readout and ready dot must be live (1 Hz telemetry), not static text; clicking the model chip opens Engine state.** Placement (bottom bar instead of sidebar gauges) supersedes §4's sidebar-gauge sketch. |
| Model library table + search + capabilities column + Vision badge, details rail with Family/Parameters/Context/Capabilities and per-field copy buttons | Exactly the §4 Models page; the details rail is a cleaner solution than the proposed card grid. Projector pairing ("Files & projectors") is properly surfaced. |
| Model settings: FreeToken Auto card with Recommended badge, hardware chips row, number-stepper + slider pairs, collapsible advanced sections | Matches the §4 Settings direction and LM Studio's pattern; range hints (`512 – 32768`) are a good addition worth keeping for every numeric control. |
| Diagnostics: server endpoint card with copyable URL, log streaming pill with Pause/Clear | Matches the §4 Logs direction; keep filters (level/text) from the plan on the roadmap. |
| Startup review: four stat cards (RAM/threads/file size/context) on top | Right instinct; keep these cards. |

### Gaps against the original request (fold into the phases)

1. **No t/s anywhere.** The four pages shown are all management pages. Per-message `tok/s · TTFT` (Chat) and a live tok/s readout are absent — add live t/s to the status bar while generating and keep the per-message stats in the Chat page per §3/§4.
2. **No loading experience.** Every mockup shows the ready state. The staged load card (checklist → % → VRAM rising → log tail → Cancel, §4 Tier 2) and a `Loading 42%` status-bar state are required — this was one of the original asks ("showing the loading").
3. **Startup review is still the text dump in a nicer frame.** The monospace output should be a secondary "Output" tab; the hero must be the visual memory budget bar (weights / KV / expert cache / reserve) with warning callouts, per §4. Keep the Copy output button.
4. **No Engine/console destination.** FreeToken's strongest idea (live model card, tokens/s, requests, TTFT, cache presets with live GiB chips) has no page. Either add an Engine page or fold the live model card into Startup review; decide at Phase 4 start.
5. **Chat page not shown.** It is the default destination and the hardest page; treat §4 Chat as the spec and mock it before implementation.
6. Mockup data inconsistencies to not carry into the spec: Diagnostics log shows an RTX 4090 24 GB while Model settings shows RTX 4060 8 GB; the library "Modified" column is all "—" (wire real mtimes); status-bar RAM should match the hardware chips row.

Phase mapping unchanged, with these additions: Phase 1 ships the status bar (live) and the rename; Phase 2 (Chat) closes gap 1; Phase 3 (Models) closes gap 6; Phase 4 (load progress + Engine) closes gaps 2–4.
- Honest telemetry labeling: every estimated number keeps a "not calibrated" tooltip; graphs/P100 caveats move from the monospace dump into warning callouts rather than disappearing.

---

## 11. Implementation status — what shipped on day one

The reorganized IA was implemented as described below. Committed as working code in this workspace (not yet a git commit — source changes remain uncommitted per workspace convention).

### IA reorganization (v2 — replaces the five-page mockup layout)

| Was (mockups / FLTK) | Now | Why |
| --- | --- | --- |
| Chat | **Chat** (default) | Unchanged mission; gains per-message tok/s/TTFT stats, reasoning disclosure, prefill state, suggestion chips, live loading card. |
| Model library | **Models** | Same table + details rail from the approved mockup, with per-model loading progress and one-click Load/Unload. |
| Model settings + Startup review | **Tune** | Merged. The launch summary (memory budget bars, warnings, Load button) sits beside the settings that change it and updates live — the review is no longer a separate page with a text dump. |
| Diagnostics (+ FreeToken console ideas) | **Monitor** | Merged. Live model card (tok/s, TTFT, requests, context fill, uptime), VRAM/RAM gauges, endpoint/launch info, and the streaming server log with filter/pause/save in one place. |
| (global) | **Bottom status bar** | Always-visible state dot, load stage/percent, live tok/s while generating, model chip, VRAM/RAM, model count. |
| Settings app-level | **Settings modal** | Server executable, port, theme, data folder — off the main nav. |

### Shipped components

- `desktop/control.h|cpp` — loopback control plane (`/api/state`, `/api/events` SSE, `/api/chat`(+`/cancel`), `/api/engine/load|unload`, `/api/models/*`, `/api/settings`, `/api/profile/restore`, `/api/logs`, `/api/review`, static UI serving), event bus, 1 Hz telemetry thread (NVML + `GlobalMemoryStatusEx` + `/slots` context fill), log ring fed by a new live stdout pipe, structured startup review with device-attributed memory budget, engine lifecycle with load-stage detection, settings registry with reload-vs-next-request tracking, and one-time migration of `settings.json`/`conversation.json` from the legacy `FreeTokenDesktop` folder.
- `desktop/platform.cpp` — `Process::start` now pipes child output live (tee to `server.log` + per-line callback) while keeping the Job Object supervision.
- `desktop/ui/` — dependency-free frontend (no npm/build step): design system matching the approved mockups, markdown renderer (escape-first, safe), SSE-driven live updates, chat with streaming + stats, models table + details rail, Tune page with sliders/steppers/switches and launch summary, Monitor with live log, status bar, settings modal, dark/light themes, toasts and confirm dialogs.
- `desktop/CMakeLists.txt` — new FLTK-free target `freetoken-host` → **`llamacpp-p100.exe`**; copies `ui/` next to the exe at build.

### Verified on this machine (RTX 4060 Laptop, Windows)

- Library scan finds both models (MiniCPM5-2B-F16, gemma-4-26B q4) with correct arch/layers/context/expert metadata.
- Full loop with MiniCPM5-2B: select → Auto profile (37 GPU layers) → load → chat → **17.2 tok/s, TTFT 293 ms**, context fill 38/4096 from `/slots` → unload. Old FLTK conversation and settings migrated automatically.
- Load pipeline streams staged progress events (Reading model file → Loading tensors → Placing layers → Allocating context) from the live log pipe.
- All four pages + status bar verified visually in a real browser.

### Known limitations (next up)

- Load progress is stage-based (Tier 1); exact percentage needs the planned `FT_LOAD_PROGRESS` fork patch (§4 Tier 2).
- Image attachments: projector plumbing exists end-to-end, composer button is present but inert until the multimodal request path is added.
- Regenerate/edit-and-resend and conversation list (multiple saved chats) are stubbed with friendly notices.
- Historical assistant messages show stats only for messages sent in the current session (usage is not persisted per message yet).
- `timings_per_token`/`include_usage` give exact server-side t/s; the Monitor "Requests" counter counts this launch, not all-time.

---

## 12. Memory budget (measured September 10, 2026)

Design constraint from the user: the GUI must take as little memory as possible so the model gets everything. The architecture already guarantees the important part — **the GUI process never touches model weights**; llama-server is a separate child process. All numbers below measured on the RTX 4060 reference machine, engine idle, one UI tab open.

| Component | Before optimization | After | Notes |
| --- | --- | --- | --- |
| `llamacpp-p100.exe` working set | 60.8 MB | **3.8 MB** | Trimmed after startup and after NVML's first tick; cold pages yield to the LLM under memory pressure. |
| `llamacpp-p100.exe` private (committed) | 39.9 MB | **37.3 MB** | Mostly NVML client runtime, static CRT, httplib buffers and thread stacks; pageable. |
| `llamacpp-p100.exe` threads | 19 | **12** | httplib pool cut from ~15 to 6 workers (SSE holds one worker per open tab); telemetry + main + aux. |
| UI page JS heap | — | **4.1 MB** (129 DOM nodes) | Dependency-free vanilla JS; no framework, no chart/markdown libraries. |
| Browser tab total | — | renderer-dependent | Chromium's per-tab baseline dominates and is shared across tabs. |

What was done to get there:

1. **Working-set trims** (`SetProcessWorkingSetSizeEx -1,-1`) after bootstrap, after the first telemetry tick (NVML allocates then), and after each model load completes. Working set is what competes with the LLM for resident memory — it is now ~4 MB and Windows shrinks it further under pressure.
2. **httplib thread pool 15 → 6**: fewer 1 MB thread-stack reservations. Six because each open UI tab holds one SSE worker; two tabs plus API calls still fit comfortably.
3. **NVML cached**: loaded once per process instead of load/free every second (less churn, fewer transient allocations).
4. **Adaptive telemetry cadence**: 1 Hz while a UI is connected or the engine is up; 6 s when idle and unwatched. `/slots` client is reused rather than rebuilt per tick.
5. **Hidden tabs release their SSE connection** and reconnect on return (the app is also discard-safe for browser Memory Saver: it refetches full state on any reconnect).

Guidance baked into the design:

- **Close (or just hide) the UI tab while a long generation runs.** The engine is supervised by `llamacpp-p100.exe`, not by the browser — closing the tab never stops the model, and a hidden tab drops its connection to near-zero traffic.
- Chromium's **Memory Saver** can discard the tab entirely; on return the UI reconnects and rebuilds from `/api/state` with no loss.
- The remaining footprint is dedicated to things that earn their place: live VRAM/RAM truth (NVML), the streaming log ring (bounded at ~6k lines), and the SSE bus that makes the UI real-time.

---

## 13. Native Slint UI (September 10, 2026)

The web UI was remade natively with **Slint's C++ API** (chosen over Rust to keep the existing, tested `desktop-core` in one language; see the language discussion in the conversation history — the toolkit, not the language, determines the footprint). New target `freetoken-slint` → **`llamacpp-p100-native.exe`**:

- `desktop/native/app.slint` — the four-page interface (Chat / Models / Tune / Monitor) with sidebar gauges and status bar, fluent-dark widgets over the project's color tokens.
- `desktop/native/native.cpp` — in-process controller: no HTTP layer at all. It reuses `desktop-core` directly (settings registry, GGUF inspector, recommender, `ft::Process` supervisor) and drives the UI through Slint properties/callbacks from worker threads via `slint::invoke_from_event_loop`. Shares `settings.json` / `conversation.json` with the web host (run one UI at a time — both manage llama-server on the same port).
- Renderer policy (§12 constraint): compiled with **only Slint's CPU software renderer** (`SLINT_FEATURE_RENDERER_SOFTWARE=ON`, femtovg/Skia/Qt backends OFF) — **zero VRAM**, no OpenGL in the GUI process.

Native v1 scope notes: chat renders as clean plain text (markdown is stripped; rich text is the one real cost of leaving the browser — see §6). The experimental MoE keys beyond gpu/cpu cache and reserve, projector pairing by folder scan, and image attachments remain web-UI features for now.

---

## 14. FreeP100 — the FLTK app, upgraded (September 10, 2026)

The user preferred the FLTK app's look, so the new telemetry was ported **into it** instead of replacing it. The FLTK target (`freetoken-desktop`) now outputs **`FreeP100.exe`** (1.8 MB, statically linked with the core — no runtime DLLs). Additions, all reusing the desktop-core plumbing:

- Live **tok/s** in the status line while generating; per-reply stats line (`[20.8 tok/s | TTFT 284 ms | 318 tokens | completed]`) in the transcript, from the server's `timings` stream + `stream_options.include_usage` (§3).
- **TTFT** measured at first streamed token.
- **Staged load progress** with percentage in the status line, via the new `Process` stdout pipe and the same stage detection as the control plane (§4 Tier 1).
- **VRAM/RAM sidebar gauges** (new `ft::gpu_memory`/`ft::ram_memory` cached-NVML helpers in [desktop/library.cpp](desktop/library.cpp)), 1 Hz telemetry thread.
- **Context fill + session counters** (requests, tokens in/out) via `--slots`, shown in a sidebar chip line.
- **Live server log** in Diagnostics — bounded ring fed by the pipe, auto-scrolled, replacing refresh-to-tail.
- **Startup review** now appends the estimated VRAM/RAM memory plan (device-attributed, §10) to the planner text.

Also: window/brand renamed to FreeP100; llama-server launches with `--metrics --slots`; all three UIs (FreeP100, llamacpp-p100-native, web host) share the same settings and conversation files and the same llama-server port — run one at a time. All existing tests (core + streaming integration) pass unchanged.
