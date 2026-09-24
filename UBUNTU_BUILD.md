# Ubuntu build

Build from the complete workspace, including `llama.cpp`, `desktop`, `cmake`, and `scripts`. The engine contains local changes that are not in upstream Git. Do not replace it with a fresh clone or export only its committed HEAD.

## Dependencies

Ubuntu 24.04 LTS, x86-64:

```sh
sudo apt update
sudo apt install build-essential cmake ninja-build git pkg-config
```

The default browser interface uses the included `desktop/ui` assets. Building it does not require Node, Rust, FLTK, Slint, or an active desktop session. To regenerate the frontend after editing `desktop/frontend/src`, use a supported Node.js installation and run `npm ci && npm run build` from `desktop/frontend`.

## CPU build

From the workspace root:

```sh
sh scripts/build-ubuntu.sh cpu
./build/ubuntu-cpu/llamacpp-p100
```

Open `http://127.0.0.1:18432/` if the browser does not open automatically. On a headless machine, pass `--no-browser` and use an SSH tunnel to port 18432. Paste your model folder path into Settings; the native folder picker is Windows-only.

## Tesla P100 build

Install a CUDA 12.x toolkit and a compatible NVIDIA driver separately. The P100 preset requires compute capability 6.0 and disables CUDA graphs. CUDA 13 is not supported by this preset.

```sh
export FT_CUDA_ROOT=/usr/local/cuda-12.9
sh scripts/build-ubuntu.sh p100
./build/ubuntu-p100/llamacpp-p100
```

Set `FT_CUDA_ROOT` to the actual toolkit location. Set `FT_BUILD_JOBS=2` before building on machines with limited RAM. The script builds the server, CLI and browser host, runs the three MoE tests and desktop core tests, and stages the executables beside the UI assets. Keep that output folder together. CUDA runtime libraries and the driver must remain installed on the target machine.

Linux GPU auto-detection and GPU telemetry are not implemented in this UI. Set GPU offload and cache budgets manually; automatic profiles can fall back to CPU. No model weights are included. Build/test success does not establish pretrained model correctness or P100 performance.

## Optional native interfaces

The browser host is the default Linux interface. For the FLTK interface, install the X11 development dependencies and configure with `-DFT_BUILD_FLTK=ON`. Slint is explicitly opt-in with `-DFT_BUILD_SLINT=ON` and additionally requires its Rust/platform dependencies. These optional builds fetch their own upstream dependencies.

## Verification limits

This workspace was prepared on Windows without an installed WSL distribution or Docker. Native Ubuntu execution and CUDA/P100 validation still need to be run on the target machine. See `reports/ubuntu-cleanup.md` for the checks performed here.
