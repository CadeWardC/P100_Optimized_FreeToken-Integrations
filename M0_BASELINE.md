# M0 baseline preparation

Status: first implementation slice completed on the development Windows host. M0 remains open until actual P100 and native Linux validation. No FreeToken executor or caching policy is enabled. No P100 is needed for the CPU work below.

## Implemented

- Four CMake presets: `ft-windows-cpu`, `ft-linux-cpu`, `ft-windows-p100`, and `ft-linux-p100`. Existing upstream presets remain available.
- Explicit CUDA toolkit selection with a pre-configuration CUDA 12.x guard, architecture 60, and graphs disabled. CUDA 12.9.x remains the release candidate; 12.6 is a development fallback.
- A standard-library Python inventory and benchmark recorder, with source pins, patch verification, source diffs/untracked-file hashes, hardware queries, model/executable SHA-256, CMake cache, commands, raw samples, and failure records.
- Recorder checks for failed processes, malformed/missing measurements, accidental result overwrite, missing tools, and single-file model handling.

The presets build static libraries and use an explicit SSE4.2 CPU configuration with AVX/AVX2/BMI2/FMA/F16C disabled. This is a conservative development baseline, not a tuned performance build or a tested universal x86 binary. Server UI and OpenSSL are disabled to avoid download/build dependencies; CLI and HTTP server are built. Use local models. The upstream benchmark reports asserts enabled; retain that condition across comparison arms.

## Run now without a P100

Requires CMake 3.24 or newer (preset includes and top-level guard). Windows CPU builds use Visual Studio 2022 with the x64 C++ toolchain; Linux uses Ninja and an installed C/C++ compiler.

From this folder:

```powershell
python scripts/baseline.py inventory --output reports/my-host.json
python -m unittest discover -s scripts -p test_baseline.py -v
cd llama.cpp
cmake --preset ft-windows-cpu
cmake --build --preset ft-windows-cpu --target llama-cli llama-server llama-bench test-backend-ops test-quantize-fns test-sampling test-grammar-parser
cd ..
ctest --test-dir build/ft-windows-cpu -C Release --output-on-failure -R '^(test-quantize-fns|test-sampling|test-grammar-parser)$'
./build/ft-windows-cpu/bin/Release/test-backend-ops.exe test -b CPU -o MUL_MAT_ID
```

For Linux substitute `ft-linux-cpu`, omit `-C Release`, and use executables in `build/ft-linux-cpu/bin/` without `.exe`. Linux commands are prepared but not executed here. The focused CTest selection requires no downloaded model. A full CTest run includes other targets, model fixtures, Python dependencies, and potentially network-dependent tests; do not interpret the focused selection as the full suite.

Inventory files and benchmark directories must have new names. The inventory returns nonzero for mismatched pins or patch hashes and records unavailable hardware tools explicitly. It does not prove the working tree equals the original patch set: source differences are retained for review. No global Git configuration is changed.

## CUDA build later

Use an x64 VS developer shell with Ninja on Windows, and a CUDA-supported compiler. Explicitly choose the toolkit before configuring:

```powershell
$env:FT_CUDA_ROOT = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9'
cd llama.cpp
cmake --preset ft-windows-p100
cmake --build --preset ft-windows-p100
```

On Linux:

```bash
export FT_CUDA_ROOT=/usr/local/cuda-12.9
cd llama.cpp
cmake --preset ft-linux-p100
cmake --build --preset ft-linux-p100
```

A P100 is not required to compile, but compilation still requires a compatible installed CUDA/host-compiler pair. These full CUDA builds have not been run. Do not bypass unsupported-compiler checks. After compilation, inspect the CUDA output with the selected toolkit's `cuobjdump` for embedded sm_60 code. On the P100, run backend tests with `-b CUDA0` and record actual device selection; CPU tests cannot validate CUDA kernels.

## Record a model benchmark

After selecting a local, single-file GGUF that fits available RAM:

```powershell
python scripts/baseline.py bench --binary build/ft-windows-cpu/bin/Release/llama-bench.exe --model 'D:/models/control.gguf' --arm p100-baseline --threads 8 --gpu-layers 0 --cmake-cache build/ft-windows-cpu/CMakeCache.txt --output reports/control-cpu-01
```

The example path is a placeholder. The recorder uses llama-bench's built-in warmup, at least five repetitions, 512 prompt tokens, 256 generated tokens, batch 512, microbatch 128, and flash attention off. It preserves per-repetition timings; it does not measure tokenization, sampling, TTFT, logits, or perplexity. The generated prompt is llama-bench's synthetic workload, not a user prompt. No model downloads occur. Split GGUF is rejected until all-shard provenance is implemented.

The arm label is supplied by the caller, not inferred from the binary. The caller must pair the executable with its actual CMake cache and source checkout; the current recorder inventories this workspace. Cross-check source provenance before comparing external/unpatched binaries. This is the initial recorder, not the full interleaved A/B harness or precision auditor.

## Verification on this host

- Windows CPU configure and all requested build targets passed with MSVC 19.43.34808.0; [build log](reports/m0-cpu-build.log).
- CLI/server version and benchmark help commands launched successfully.
- Sampling, grammar parser, and quantization CTests: 3/3 passed; [test log](reports/m0-cpu-tests.log).
- CPU `MUL_MAT_ID`: 872/872 passed; [operation log](reports/m0-cpu-mul-mat-id.log). This compares CPU execution with the test's reference and is not a CUDA/P100 comparison.
- Recorder unit tests: 7/7 passed using synthetic process results, not model performance measurements.
- CUDA guard: installed 13.0 rejected, installed 12.6 accepted. This tested the guard only, not CUDA compilation.
- All 31 individual patch artifacts, the combined patch hash, and all three pinned repository revisions matched the manifest. `git diff --check` passed.
- [Host inventory](reports/m0-host-inventory.json): RTX 4060 Laptop GPU, 16,309,932,032 bytes RAM (about 15.2 GiB). The initial [sandbox inventory](reports/m0-development-inventory.json) records denied Windows hardware queries; the host inventory was collected with access to those queries.

Sandbox configure/build attempts failed from compiler access and duplicate Path/PATH environment entries; running the same commands outside the sandbox succeeded. Existing compiler warnings are retained in the build log. No inference-source changes were needed.

## Remaining M0 gates

- Select and hash the dense control and first conventional-attention MoE after confirming target host RAM, desired model, and P100 SKU. No model has been downloaded or approved as an integration oracle.
- Compile CUDA sm_60 with a supported toolchain and inspect embedded code.
- Run native Linux builds and tests.
- Run actual P100 Linux/Windows inference, precision comparisons against unpatched/higher-precision references, and baseline performance measurements; freeze numerical tolerances from that evidence.
- Expand the recorder into interleaved A/B comparisons and add the numerical comparison harness.

The user has deferred M0 hardware measurements to allow M1 development. A CPU host-bank and explicit FFN boundary prototype is now implemented; see [M1_EXPERT_BOUNDARY.md](M1_EXPERT_BOUNDARY.md) for tests and remaining integration work. The original `SOURCE_MANIFEST.json` and `P100_BASELINE.patch` describe the historical pinned baseline and are unchanged. Back up the new presets, `cmake/`, `scripts/`, and implementation files alongside those artifacts; the historical patch alone does not include them.
