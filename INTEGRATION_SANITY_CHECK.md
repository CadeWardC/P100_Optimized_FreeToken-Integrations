# Integration sanity check

Audit date: 2026-09-08. Scope: pinned llama.cpp with P100 patches, pinned FreeToken reference, implementation plan, and source-reproduction artifacts.

**Verdict: proceed with a gated prototype, not a claim of proven integration.** The overall direction is feasible enough to investigate, but the first plan contained unsupported assumptions and one incomplete reproduction artifact. Those are corrected. No FreeToken integration exists yet, and no P100 build or runtime tests were performed. A source audit cannot guarantee a future implementation has no mistakes.

## Findings and corrections

| Priority | Finding | Evidence in the pinned source | Correction |
| --- | --- | --- | --- |
| High | The combined baseline patch omitted two new kernel files, although both files were present in the downloaded working source. A plain `git diff` excludes untracked additions. | Patch 12 adds `ggml/src/ggml-cuda/mmvq-f16-sm60.cu` and `.cuh`. The initial combined patch contained only 31 modified files. | Regenerated `P100_BASELINE.patch` from an isolated index containing the complete ordered patch set: 33 changed/added files. Independently replayed it and refreshed the manifest hash. |
| High | CUDA graphs were treated as a normal future P100 feature without acknowledging the backend guard. | `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_graph_set_enabled` disables them below Volta. | Keep graphs disabled in P100 build examples and release requirements. Re-enabling is separate optional research; eager execution is the supported plan. |
| High | Existing checkpoints cannot be assumed to contain complete hybrid-model state. | `tools/server/server-context.cpp::create_checkpoint` uses `PARTIAL_ONLY`; `src/llama-memory-hybrid.cpp::state_write/state_read` skip attention KV with that flag. | Require either partial snapshot plus independently retained/validated attention KV, or a full sequence snapshot with explicit memory accounting. |
| High | Checkpoint labels could be off by a batch or a token. | Server checkpoints are made before `llama_decode`; the current queued batch is excluded. `include/llama.h` also states that a new `ON_DEVICE` snapshot invalidates earlier snapshots for the same sequence. | Specify exclusive token boundary `[0,p)`, split batches when needed, distinguish token emission from consumption, and forbid on-device snapshots as an independent multi-anchor history. |
| High | CPU backing buffers are not necessarily in GPU-compatible GGUF layout. | `ggml/src/ggml-cpu/repack.cpp::ggml_backend_cpu_repack_buffer_set_tensor` transforms weight data; repack supports `MUL_MAT_ID`. | Use canonical ordinary host buffers initially; identify and budget any separate CPU-repacked copy. Never DMA repacked bytes as canonical GPU weights. |
| High | Remapping expert IDs alone can misindex bias/scale tensors or violate fused-kernel layouts. | `ggml/src/ggml-cuda/mmvq.cu::ggml_cuda_mul_mat_vec_q` checks fusion tensor dimensions/strides and scale counts against expert dimensions. `ggml/src/ggml.c::ggml_mul_mat_id` requires valid typed ID/tensor shapes. | Require aligned slot order across fused banks or separate original-ID gathers; use valid compact IDs and scatter maps, no negative sentinel IDs. Gate fusions and sub-top-k capacity until tested. |
| High | A graph boundary does not automatically prevent full-bank GPU allocation. The evaluation callback is not a replacement executor. | `ggml/src/ggml-backend.cpp` schedules copies and executes graph views around callback observation points. | Prove partitioning before allocation; ensure original full expert banks no longer induce device copies. Explicitly own subgraph outputs and lifetimes. This is M1's architectural gate. |
| High | Promised resize recovery may fail if it uses an allocator that terminates on OOM. | The CUDA ordinary buffer allocator returns null; the legacy scratch pool retries then calls `CUDA_CHECK(err)`. | Use recoverable allocation interfaces, reserve scratch ahead of execution, and inject failures in the exact used paths. Reject safely when fallback resources do not exist. |
| Medium | Authoritative mapped expert bytes can be unmapped after loading unless marked retained. | `src/llama-model-loader.cpp` calls `unmap_fragment` during final mapping cleanup. | Retain every expert span and split-file mapping through all CPU/DMA users; test loader cleanup and destruction. |
| Medium | The initial bandwidth fraction does not include the bounded-staging path's host copy, nor multi-token expert fanout. | FreeToken's `moe/bench_profile.py::load_hybrid_fetch_fraction` uses overlapped measured bandwidths. Our proposed bounded staging adds a distinct host-copy stage. | Calibrate the actual staging pipeline; use the simple formula only in its single-token regime. Model assignments separately from unique transfers for batches. |
| Medium | A generic generation invalidation rule would rebuild graphs on every cache miss. | Slot contents change frequently; scheduler/capture lifetime is tied to storage and graph properties. | Separate slot-content generations from arena/layout generations, with independent validity rules. |
| Medium | Prefill demanded identical logits despite allowing kernel-changing tiling. | MMVQ/MMQ dispatch and reductions can change with batch shape. | Require frozen numerical tolerances; preserve routing weights and original aggregation order in the correctness path. |
| Medium | CPU compatibility remained ambiguous in candidate build commands. | `ggml/CMakeLists.txt` can enable AVX2/BMI2/FMA/F16C with `GGML_NATIVE=OFF`. | Explicitly label the commands as non-universal x86 builds; require a tested ISA floor or dispatch variants before release. |

The implementation plan now contains these constraints in the relevant architecture, strategy, build, and test sections. The local source itself was not changed by this audit.

## Verification actually performed

1. Checked all 31 individual patch SHA-256 hashes against the existing manifest.
2. Replayed all patches in declared order into an isolated Git index based on the pinned upstream commit. The Windows patch checkout has CRLF line endings, so replay into Git's LF-normalized index required LF-normalized temporary copies. The initial direct index replay failed on that line-ending mismatch; the normalized replay succeeded. Original patch files were preserved.
3. Compared the resulting index against the downloaded working source, including both patch-added kernel files: no differences.
4. Exported the complete combined patch from that index, then replayed it into a second clean isolated index. Confirmed it reproduces the source and passes forward/reverse application checks.
5. Checked patched-index whitespace consistency. Recorded the resulting complete Git tree ID, corrected combined-patch hash, and LF-normalized per-patch hashes in `SOURCE_MANIFEST.json`.
6. Queried the local CUDA 12.6 compiler's advertised code targets: it lists `sm_60`. This is a capability listing, not a compile test.
7. Rechecked NVIDIA's CUDA 13 release notes and CUDA 12.9 Windows installation documentation. The CUDA 12.x target remains appropriate; an actual supported P100 driver/Windows edition/compiler combination still requires validation. [CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html), [CUDA 12.9 Windows guide](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-installation-guide-microsoft-windows/index.html).

Isolated audit indexes and LF-normalized patch copies are under `audit-artifacts/`. They are verification artifacts, not additional source changes. The normal source index was not staged or committed.

## Required proof before expanding implementation

| Gate | Required evidence | Failure action |
| --- | --- | --- |
| Baseline | Actual native Linux and Windows P100 compilation, inference, numerical comparison, and recorded driver/toolchain | Fix the baseline before adding offload changes |
| Graph ownership | M1 synthetic FFN matches ordinary execution and device allocations scale with cache capacity rather than total experts | Redesign the execution boundary before building caching policies |
| Expert identity | Same numeric expert ID in two layers remains distinct; packed weights/biases/scales agree; every route contributes once | Keep cache mode disabled |
| Weight lifetime | Canonical source survives loader cleanup, split GGUF handling, asynchronous copies, cancellation, and context destruction | Fix ownership before enabling overlap |
| Hybrid math | Fixed-input routing replay across CPU-only/GPU-only/mixed splits meets frozen tolerances and original reduction order | Fix executor semantics before tuning policy |
| State restore | Exact-boundary full or partial-plus-retained-KV restore agrees with fresh evaluation, including batch boundaries and edits | Fall back to recomputation |
| Memory pressure | Every resize/fallback allocation path fails without stale pointers, state loss, or an unexpected fatal allocator | Disable elasticity until recovery works |
| Platform parity | Real P100 tests pass on both native operating systems with packaged dependencies | Label only tested platforms supported |

No graph capture, memory recovery, checkpoint correctness, throughput improvement, or Windows P100 compatibility is marked as proven solely from this review.
