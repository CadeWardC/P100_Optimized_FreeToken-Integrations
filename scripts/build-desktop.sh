#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cmake -S "$project_dir/desktop" -B "$project_dir/build/desktop-linux" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$project_dir/build/desktop-linux" --parallel "${FT_BUILD_JOBS:-4}"
ctest --test-dir "$project_dir/build/desktop-linux" --output-on-failure
