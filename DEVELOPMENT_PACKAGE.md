# Windows CUDA development package

This package is for testing the experimental executor. It is not a supported P100 release. The standalone GPU executor is tested on an RTX 4060 Laptop; the CLI/server expert-cache option still uses CPU execution. Pretrained output validation, P100 runtime tests and native Linux tests are pending. Adaptive splitting and transfer/compute overlap are not implemented.

## Run

Extract the complete archive to one folder. Keep the bundled DLLs beside the executables. A compatible NVIDIA driver must be installed separately; the package does not include a driver or model weights. Inspect `package-status.json` and `verification/` for the exact completed checks.

```powershell
./test-moe-offload.exe --device CUDA0
./test-moe-offload.exe
./test-moe-integration.exe
./llama-cli.exe --help
./llama-server.exe --help
```

The GPU command checks compact expert execution, transfer staging and cancellation. The model integration command creates tiny temporary GGUFs and checks the CPU inference path. Both return a nonzero exit status on failure.

For CPU-cache text inference with a compatible GGUF that fits host RAM:

```powershell
./llama-cli.exe -m D:/models/compatible.gguf -ngl 0 --no-op-offload --no-kv-offload -np 1 --moe-cpu-cache-mib 64
```

The model path is a placeholder. Selected model revisions and publisher hashes are in `MODEL_ACCEPTANCE_MANIFEST.json`. The complete weights and runtime overhead must fit RAM. A 64 MiB cache is not a total memory limit.

For pretrained CPU parity, use `test-moe-integration.exe --model <complete.gguf>` or the hash-verifying `source-changes/scripts/validate_pretrained.py` wrapper. When using the wrapper from this archive, explicitly pass `--manifest MODEL_ACCEPTANCE_MANIFEST.json`.

## Contents and integrity

CUDA 12 runtime/cuBLAS and MSVC C++/OpenMP runtime DLLs are included. The packager tests these staged binaries with the CUDA toolkit removed from PATH, but this is not a clean-machine installation test. License files are in `licenses/`.

`SHA256SUMS.txt` hashes every other packaged file. The adjacent `.zip.sha256` file hashes the archive. Use PowerShell `Get-FileHash -Algorithm SHA256 <file>` to check a file.

`source-changes/` records the pinned base revision, the complete tracked working-tree patch, untracked source additions and helper scripts. These include earlier P100/CPU work as well as this executor work. Apply the patch only to a fresh checkout of that revision, then copy additions to their recorded relative paths. Do not apply it on top of the separate P100 baseline patch.
