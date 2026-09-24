#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
mode=${1:-cpu}
case "$mode" in
    cpu|p100) ;;
    *) echo "Usage: sh scripts/build-ubuntu.sh [cpu|p100]" >&2; exit 2 ;;
esac
if [ "$(uname -s)" != Linux ]; then
    echo "Run this script on Ubuntu Linux." >&2
    exit 2
fi
if [ "$mode" = p100 ] && [ -z "${FT_CUDA_ROOT:-}" ]; then
    echo "Set FT_CUDA_ROOT to your CUDA 12.x toolkit directory." >&2
    exit 2
fi
jobs=${FT_BUILD_JOBS:-4}
cd "$project_dir/llama.cpp"
cmake --preset "ft-linux-$mode" -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build --preset "ft-linux-$mode" --parallel "$jobs" --target llama-server llama-cli test-moe-cache test-moe-offload test-moe-integration
ctest --test-dir "$project_dir/build/ft-linux-$mode" --output-on-failure --no-tests=error -R '^test-moe-(cache|offload|integration)$'
cmake -S "$project_dir/desktop" -B "$project_dir/build/desktop-linux" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DFT_BUILD_FLTK=OFF -DFT_BUILD_SLINT=OFF
cmake --build "$project_dir/build/desktop-linux" --parallel "$jobs"
ctest --test-dir "$project_dir/build/desktop-linux" --output-on-failure --no-tests=error
stage="$project_dir/build/ubuntu-$mode"
cmake --install "$project_dir/build/desktop-linux" --prefix "$stage"
cmake -E copy "$project_dir/build/ft-linux-$mode/bin/llama-server" "$stage/llama-server"
cmake -E copy "$project_dir/build/ft-linux-$mode/bin/llama-cli" "$stage/llama-cli"
echo "Built application: $stage/llamacpp-p100"
echo "Server: $stage/llama-server"
