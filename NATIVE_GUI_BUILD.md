# Native desktop build 0.1

Built September 9, 2026. This is the first runnable preview of the
[native GUI plan](NATIVE_GUI_PLAN.md), using C++17, FLTK 1.4.5 and the existing
vendored cpp-httplib/JSON libraries. No inference-engine source was changed.

## Deliverables

- `build/desktop/freetoken-native.exe`: Windows native GUI, static runtime.
- `dist/freetoken-native-windows-preview-0.1/`: portable package with buildable
  application source and third-party license notices.
- `desktop/`: application, process adapters and tests.
- `scripts/build-desktop.cmd` and `scripts/build-desktop.sh`: build entrypoints.
- [Usage and limitations](desktop/README.md).

The GUI manages a selected patched llama-server and GGUF, streams chat, allows
cancellation, retains the current conversation, and saves engine settings.
The settings screen exposes core loading/generation controls and existing
FreeToken-inspired cache, scheduling, pipeline and numerical-mode options.
Invalid CPU/GPU cache combinations and incompatible semantic checkpoint modes
are rejected before launch.

## Verification performed

- Windows Release build with MSVC 19.43 and CMake 4.1.2: passed.
- Core test: fragmented SSE, invalid settings, nonfinite temperature rejection,
  Windows path quoting, JSON persistence/replacement, process cleanup and HTTP.
- Native UI integration: starts a mock server, sends correctly shaped chat
  messages, renders streamed content, reloads the server without losing chat
  history, and cancels a second streaming response. Passed.
- Real model smoke: GUI started the local patched CUDA server with the saved
  Gemma GGUF on RTX 4060, streamed 32 output tokens, saved the response, and
  closed the window/server. Passed. This is functional validation, not a
  performance comparison or numerical-equivalence test.
- Native window inspection: chat, settings and model-selection tabs were
  visually inspected through Computer Use.
- Dependency inspection: recorded in `reports/desktop-dependencies.txt`.

The real-server test initially found incorrectly shaped system-message JSON;
this was corrected and the mock now rejects that shape. The passing run is
`reports/desktop-real-smoke-v2/`. Earlier diagnostic evidence is retained.

## Measured footprint

On this Windows machine with the GUI open and no engine loaded:

| Metric | Measurement |
| --- | ---: |
| Executable | 1,511,424 bytes (1.44 MiB) |
| Private bytes | 3,354,624 bytes (3.20 MiB) |
| Working set | 21,049,344 bytes (20.07 MiB) |
| Idle CPU time | 0.015625 seconds over 30.02 seconds |
| Idle CPU as fraction of one core | Approximately 0.052% |

This is a short Windows sample, not a Linux measurement or the complete
60-second acceptance benchmark. It excludes the inference server and models.
Evidence: `reports/desktop-footprint.json` and `reports/desktop-build.log`.

## Graph and cache work retained in the design

- Auto/Off graphs are scoped to the child environment. Auto respects the
  backend's architecture and operation eligibility; it does not claim replay.
- No SM60 guard bypass is shipped. Experimental P100 capture and real-hardware
  parity/replay/memory testing remain required; the public SM61 patch is not
  sufficient validation for P100.
- Cache changes currently use Apply & reload with conversation text retained.
  The proposed engine-owned HTTP operation will drain work, rebuild an empty
  context and reattach state while retaining model weights where possible.
  Active KV state is not promised to survive that operation.

## Remaining plan milestones

The full settings schema/coverage system, GGUF and GPU capability probes,
automatic cache budgeting, calibration/profiles, installed FreeToken adapter,
model library/downloads and HTTP cache resizing are not implemented yet.
The startup tab is a review, not an automatic hardware optimizer.

Linux has a POSIX process implementation and CMake/shell build entrypoint,
but no native Linux build/runtime validation was possible here: WSL has no
installed distribution. P100 was not tested. Existing FreeToken installations
and inference-engine source were not modified.
