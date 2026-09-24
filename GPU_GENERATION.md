# GPU expert generation

Latest implementation: [GPU-resident activation and hybrid scheduling](GPU_HYBRID_SCHEDULER.md) supersedes the historical CPU activation and missing CPU/GPU split statements below.

**Current status:** [GPU correctness fix](GPU_CORRECTNESS_FIX.md) supersedes the failed numerical acceptance results below. The current RTX 4060 build passes the exact Gemma QAT pretrained comparison. Gate/up/down projections use GPU execution; Gemma GeGLU uses the original CPU table between those projections. The validation section below preserves the historical pre-fix measurements.

The cached executor is connected to token generation through a synchronous scheduler node boundary. Selected expert gate/up and down projections execute on the selected GPU. SiLU runs on GPU; Gemma GeGLU uses the CPU implementation for compatibility. The scheduler waits for the routed inputs, invokes the dedicated executor outside backend computation, then resumes the model graph with the expert output. Evaluation callbacks remain available. A missed GPU boundary fails instead of launching CUDA inside the CPU worker callback.

Gemma attention, sliding/full KV state, shared dense FFN, residuals, normalization, routing and original-ID expert output scales retain their existing implementation. This first integration keeps attention and shared layers on CPU. It requires `-ngl 0 --no-op-offload --no-kv-offload -np 1`; layer offloading, multiple sequences and LoRA remain unsupported with host banks. Transfers and expert computation are serialized.

## Settings

Both CLI and server accept these options:

| Option | Purpose |
| --- | --- |
| `--moe-gpu-cache-mib 1024` | Enable GPU experts with a persistent 1 GiB weight cache shared across layers |
| `--moe-device CUDA0` | Select the expert execution device |
| `--moe-staging-mib 1` | Limit the pinned host transfer buffer, with pageable fallback |
| `--moe-compute-mib 64` | Limit the expert graph compute arena |
| `--moe-gpu-reserve-mib 512` | Require this much additional free VRAM at context construction |

Cache, compute arena and reserve must fit in currently available device memory. The check occurs before cache allocation. It is a startup admission check, not a reservation against future allocations by other applications. CUDA context, cuBLAS and backend scratch allocations are outside the explicit arena limits. The staging buffer uses host memory. `llama_moe_memory()` reports staging, configured compute/reserve limits, peak compute allocation, retained weights, transfers and cache counters; it does not report total process or driver memory.

Do not combine CPU and GPU cache options. Omit `--moe-gpu-cache-mib` for ordinary inference. Use the same model, context, batches, threads, sampling and offload settings when comparing.

```powershell
$env:PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin;$env:PATH"
./build/ft-windows-cuda-dev/bin/llama-cli.exe `
  -m models/gemma-4-26B_q4_0-it.gguf `
  -ngl 0 --no-op-offload --no-kv-offload -np 1 `
  -c 256 -b 128 -ub 8 -t 8 -tb 8 -fa off --load-mode mmap --no-repack --fit off `
  --moe-gpu-cache-mib 1024 --moe-device CUDA0 `
  --moe-staging-mib 1 --moe-compute-mib 64 --moe-gpu-reserve-mib 512 `
  --single-turn --no-conversation --temp 0 --seed 1234 -n 64 -p "The capital of France is"
```

For the HTTP server, replace the executable with `llama-server.exe`, omit the last line and add `--host 127.0.0.1 --port 8080`. Send requests to `/completion`. The application remains single-sequence for this experimental path.

## Exact QAT artifact

`MODEL_ACCEPTANCE_MANIFEST.json` contains `gemma26b-qat-q4`, Google's `gemma-4-26B_q4_0-it.gguf` at revision `d1c082be9cf3c8a514acf63b8761f4b41935842e`. Its size is 14,439,363,584 bytes and SHA-256 is `3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d`. The local download is `models/gemma-4-26B_q4_0-it.gguf`.

```powershell
python scripts/validate_pretrained.py --id gemma26b-qat-q4 `
  --model models/gemma-4-26B_q4_0-it.gguf `
  --executable build/ft-windows-cuda-dev/bin/test-moe-integration.exe `
  --device CUDA0 --output reports/qat-gpu-new.json

python scripts/validate_operation.py --model models/gemma-4-26B_q4_0-it.gguf `
  --bin build/ft-windows-cuda-dev/bin --output reports/qat-operation-new
```

The pretrained harness compares full-vocabulary logits, 16 greedy tokens and state replay against ordinary CPU inference. The operation harness checks the hash again, runs CLI and repeated HTTP generation with matching settings, and records token IDs, server timings, RSS, available system RAM and observed global VRAM. Its pressure mode combines a 16 MiB cache with a separate 2 GiB CUDA allocation. It requires Python `psutil`; pressure allocation also requires the CUDA runtime. Results are saved even on failure.

## Historical validation before the correctness fix

The integration is implemented, but **GPU numerical acceptance fails**. It is experimental and is not a validated replacement for ordinary inference. This host is an RTX 4060 Laptop GPU with 8 GiB VRAM and approximately 16 GB system RAM; these results do not establish P100 acceptance.

On 2026-09-09:

- The exact QAT download passed size and SHA-256 verification. CPU cached inference passed full-logit comparison, 16 greedy tokens and state replay: [CPU acceptance](reports/gemma-qat-cpu-pretrained.json).
- The final native GPU executor failed pretrained numerical acceptance: [GPU acceptance](reports/gemma-qat-gpu-native-final.json) and [full log](reports/gemma-qat-gpu-native-final.log). Greedy tokens differ from ordinary inference. Thresholds are recorded by the harness; GPU distribution checks require KL <= 0.01 and total variation <= 0.05, in addition to greedy agreement. They are acceptance gates, not a claim of equivalence.
- All three focused CPU CTests and ten Python tests pass. The scheduler test covers dependency order, observers, repeated execution, failure propagation and removing the hook. The standalone native GPU executor passes under Compute Sanitizer memcheck with **zero reported errors**. Full GPU synthetic model integration still fails numerical comparison. See `reports/gpu-native-final-{cpu-tests,python-tests,memcheck,integration}.log`.
- CLI and HTTP server complete generation with FreeToken off, with a 1 GiB GPU cache, and with a 16 MiB cache plus a separate touched 2 GiB CUDA allocation. Each mode runs one 64-token CLI generation and two 64-token HTTP requests, for 576 generated tokens total. This is a short sustained-generation check, not a long-duration soak test.

The operation run uses identical model, context, batches, threads and deterministic sampling settings within each interface. Server requests use the same raw six-token prompt. CLI applies its model prompt formatting, so CLI and server are not compared against each other. All commands, token IDs, binary hashes, timings and memory samples are in [operation results](reports/qat-operation/results.json).

| Mode | First server request, tokens/s | Repeat, tokens/s | Sampled global VRAM, MiB (max) | Server RSS, GiB (max) |
| --- | ---: | ---: | ---: | ---: |
| FreeToken off | 2.75 | 3.13 | 83 | 8.21 |
| GPU cache 1024 MiB | 2.22 | 3.21 | 1119 | 8.08 |
| GPU cache 16 MiB + 2048 MiB pressure | 1.80 | 1.70 | 2244 | 8.23 |

There is no consistent speedup in this run. Off and 1 GiB-cache repeats are individually deterministic, but their token sequences differ from each other. The two pressure-mode sequences also differ from each other. Therefore the operation report correctly returns `token-divergence`, despite successful CLI/server execution. This cache-size/repeat sensitivity remains unresolved.

Memory values are sampled approximately once per second, include other GPU processes, and can miss transient peaks. Available system RAM fell as low as 29 MiB during the pressure server run; these timings include substantial host memory pressure and should not be generalized to a machine with more RAM. No allocation failure occurred in this workload. The explicit reserve check does not guarantee future allocation success.

The remaining release gate is to resolve GPU arithmetic/output divergence, rerun full synthetic and QAT acceptance, and then repeat performance measurements. The integration preserves the existing attention/shared-layer graph; a diagnostic CPU executor through the same scheduler boundary produced exact agreement in the observed prompt and initial decode steps, but that partial diagnostic is not full GPU acceptance. Historical experimental arithmetic logs are retained separately; the final implementation uses the original native CUDA expert executor.

The added context and telemetry fields change the public C struct layouts; rebuild callers against the updated headers and library together.
