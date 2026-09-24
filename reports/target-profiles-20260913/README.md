# Automatic profiles for P100 + N150 and the RTX 4060 laptop

The rebuilt desktop host uses one hardware-aware policy with capability-specific safeguards. This supersedes the earlier fixed-percentage memory reserve policy.

Changes:

- Query CUDA compute capability. SM60/P100 disables CUDA replay; Pascal profiles disable flash attention. The RTX 4060 retains backend-controlled CUDA graphs.
- Replace the blanket 30% VRAM reduction and 15% full-model surcharge with explicit allowances for model weights, KV, projector, working buffers and spare memory. P100 headroom is 512 MiB; the laptop profile reserves 640 MiB in addition to its working buffers.
- Prefer full GPU residency when the model and working memory fit. Otherwise budget the available VRAM for expert weights while retaining GPU attention and GPU-only cache misses. A four-core CPU is not assigned additional expert work simply to increase CPU utilization.
- Detect physical cores, and do not halve four logical workers if topology information is unavailable. P100 + N150 policy tests select four decode and four prefill workers.
- Explicit auto-configure preserves the selected context length, subject to the model's supported limit, and re-enables automatic settings. Default model-switch recommendations may select a shorter context to improve placement. Sampling, thinking and output limits are unchanged.
- Untouched automatic profiles are recalculated after unloading the old engine, so the engine's own allocations are not mistaken for another workload. Manual overrides remain manual.

## Validated profiles

The real Gemma file is 14,439,363,584 bytes; its expert tensors occupy 12,846,382,080 bytes. The vision projector is 1,194,828,160 bytes.

| Machine | Context | Placement | Expert cache |
| --- | ---: | --- | ---: |
| This RTX 4060 laptop, with vision | 2048, preserved | GPU attention/shared weights; native GPU experts | 3648 MiB |
| Simulated 16 GiB P100 + four-core CPU, text only | 4096 | Full normal GPU model loading | None needed |
| Simulated 16 GiB P100 + four-core CPU, with vision | 4096 | GPU attention/shared weights; native GPU experts | 11520 MiB |
| Simulated 12 GiB P100 + four-core CPU, with vision | 4096 | GPU attention/shared weights; native GPU experts | 7424 MiB |

P100 scenarios assume the listed VRAM is available and 12 GiB of free system RAM. Other workloads and the selected context change the resulting budgets. The P100 JSON files explicitly mark their hardware as simulated; no physical P100 or N150 performance claim is made.

The actual laptop loaded successfully at 7147 MiB GPU usage, leaving 810 MiB free according to NVML. Its previous 2688 MiB expert profile used 6187 MiB. The current application settings and a real chat smoke-test response are recorded beside this report. These are observed memory use and functional checks, not proof of optimal throughput.

Validation: Release desktop build; core tests including tensor accounting, per-layer KV, 28 VRAM/free-memory combinations, P100 12/16 GiB, four-core fallback, SM60/name safeguards, explicit context preservation, projector budgets and RTX 4060 differentiation; host integration for save/restore, load/reload, streaming, unload and model switching; real laptop model load and chat inference.

Remaining limits: automatic settings are an allocation heuristic, not per-computer throughput calibration. The inference engine's CPU instruction-set build was not changed in this update. Native numerical mode can change generated text. Headroom remains necessary for transient allocations; filling VRAM to 100% is not the objective.
