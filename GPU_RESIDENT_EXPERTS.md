# GPU-resident expert decode and CUDA graph replay

Implementation and validation: September 9, 2026, RTX 4060 Laptop GPU.

The GPU-only one-token expert boundary now keeps activations, routing weights and expert outputs on the device. It reads back one small expert-ID list after a CUDA kernel marks nonfinite routing weights invalid. The cache still uses those original expert IDs for loading, eviction and output scales. Physical cache slots are uploaded separately.

The existing persistent expert workspace receives activations through asynchronous device copies. A second persistent, budgeted workspace holds expert contributions in original assignment order, multiplies by expert scales and routing weights, then sums them in that order. The zero tensor used to start the sum is protected from allocator reuse. No floating-point atomics are used. Both arenas and the compatibility activation table share the existing compute budget.

Automatic placement applies to layers already on the selected GPU. CPU-placed layers, multi-token prefill and hybrid CPU/GPU execution keep the vector path. `llama_context_params::moe_device_tensors` defaults to true and can be disabled for numerical ablations. Producer and completion waits remain at the scheduler boundary; this does not implement an asynchronous cache transfer ring or full-token graph capture.

CUDA captures are owned by the expert backend and keyed by the persistent graphs. Destruction, shape changes and successful cache resize explicitly retire captures before releasing referenced storage. Failed cache resize preserves the old workspaces. CUDA capture/replay counters measure capture completion and replay launches, excluding the initial launch after capture. The architecture guard and synchronization-dependent matmul fallback remain intact.

Use `scripts/build-cuda-graphs.cmd` to build the existing CUDA development directory with graph support enabled. `scripts/build-cuda-dev.cmd` keeps its default of graphs disabled, with `FT_CUDA_GRAPHS` available as an override. `GGML_CUDA_DISABLE_GRAPHS=1` disables replay at runtime for ablations. The public context and memory-report structs changed; library consumers must rebuild.

Validation and matched benchmark results are recorded in `reports/gpu-resident/`. The benchmark uses the saved Gemma model and launch settings, the same chat template and prompt, 128 greedy output tokens, a 3060 MiB expert cache and eight CPU threads. Every session excludes one warmup and measures three requests. Executables run separately, with no concurrent builds or GPU tests.

The boundary-only matched comparison measured 18.727 median decode tokens/s for the prior persistent-workspace executable and 17.906 for the GPU-resident eager executable (4.4% lower). Their ranges were 16.913-19.620 and 15.824-18.055. All six paired response texts matched. This short run does not demonstrate a transfer-boundary speedup. The historical 10.6 tokens/s result was recorded in a different session and is not the comparator for this change.

Before enabling graphs, the focused device tests passed changing inputs/routes/layers, duplicate assignments, original-expert scales, ordered reduction, eviction, failed/successful resize, cancellation/retry and compute-budget rejection. Same-mode output comparisons used a 1e-6 absolute bound. Both 135-case synthetic integration suites passed (compatibility and GPU-layer/native mode). A pretrained Gemma comparison of host versus device boundary, with GPU placement and native mode, reported zero full-vocabulary logit differences across the tested prefix, 16 greedy tokens and state replay. The raw pretrained diagnostic continuation is not a chat-quality benchmark.

With graphs enabled, all focused tests and both 135-case integration suites passed again. Compute Sanitizer reported zero errors. The persistent expert FFN test recorded one capture and ten replays; the boundary tests recorded two captures and twelve replays in each numerical mode while changing routes and exercising eviction/resize.

Pretrained host-versus-device boundary checks also passed in both compatibility and native mode with zero full-vocabulary logit differences, matching greedy tokens and successful state replay. The native run recorded 46 captures and 1,933 replays. Across that mixed prefill/decode test, expert-output readback fell from 102,727,680 to 16,220,160 bytes; multi-token prefill still uses the host boundary. Focused one-token device calls reported zero expert-output readback and uploaded only control data/scales.

Capture objects and backend kernel scratch are outside the arena telemetry; the existing GPU reserve remains relevant. No P100 hardware validation was performed.

The separate replay ablation used the **same graph-enabled binary** with `GGML_CUDA_DISABLE_GRAPHS=1` versus the variable unset, in AB/BA order. Each side had six measured requests. Replay improved the median decode rate from **18.020 to 20.824 tokens/s (15.6%)**. All six paired response texts matched. Median response time decreased from 9.452 to 8.563 seconds; median first-text time was 2.354 versus 2.389 seconds. The observed decode ranges were 15.969-18.171 and 18.107-20.916 tokens/s. This is a short laptop result, not a general performance guarantee.

| Separate comparison | Control median tokens/s | Updated median tokens/s | Measured change |
| --- | ---: | ---: | ---: |
| Prior workspace build vs GPU-resident eager boundary | 18.727 | 17.906 | -4.4% |
| Same binary, replay disabled vs enabled | 18.020 | 20.824 | +15.6% |

These are two separate ablations. Do not attribute the replay gain to removal of transfers alone, or compare these rates directly with historical sessions using different machine conditions. CUDA graphs are enabled in the current `build/ft-windows-cuda-dev` build.

Reproduction and evidence:

- `reports/gpu-resident/validate_graphs.py`: CPU/device tests, sanitizer, both integration suites and both pretrained numerical modes. Set `FT_TEST_REQUIRE_GRAPHS=1` for tests that must demonstrate actual replay.
- `reports/gpu-resident/run_chat.py`: exact saved chat launch settings and per-run environment, with streaming events retained.
- `reports/gpu-resident/boundary-summary.json` and `replay-summary.json`: per-comparison medians, ranges, text matches and executable hashes.
- `reports/gpu-resident/model-verification.json`: verified QAT GGUF hash and pinned manifest entry.
- `reports/gpu-resident/graphs-pretrained-native.log` and `graphs-pretrained-compatibility.log`: full-vocabulary logit diagnostics, readback bytes and capture/replay counts.
- `reports/gpu-resident/graphs-memcheck.log`: zero Compute Sanitizer errors.
