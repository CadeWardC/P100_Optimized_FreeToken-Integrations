# Reusable FreeToken runtime

The canonical expert execution and cache implementation now lives in `llama.cpp/freetoken`, independently of llama's loader, inference context, server and desktop UI. Copy that directory into another GGML project, add it with CMake and link `freetoken::runtime`. It can also be installed and consumed with `find_package(freetoken CONFIG REQUIRED)`.

See the [integration guide](llama.cpp/freetoken/README.md), [public API](llama.cpp/freetoken/include/freetoken/moe.h), [standalone example](llama.cpp/freetoken/examples/smoke.cpp), and [verification results](llama.cpp/freetoken/VERIFICATION.md).

The module owns host banks, route tiling, persistent expert caching, backend transfers, CPU/GPU execution, scheduling, cancellation and metrics. The existing llama integration uses compatibility aliases, preserving its model/context/graph call sites and public `llama.h` behavior. Cache code is no longer compiled into `ggml-base`, and tests link the same library used by inference. Old internal C++ symbol names are not binary-compatible: rebuild dependent binaries.

The destination engine must still connect its model tensors and post-routing execution boundary. Portable CPU builds avoid this fork's private extensions. Precise CUDA behavior, device-resident route handling and full scheduler integration require the corresponding backend/host hooks described in the guide; they are not automatically installed by copying the library.

CPU regression suites, CUDA executor/integration suites, standalone static/shared builds, and an installed-package consumer passed. No new P100 or pretrained throughput claim is made.

These changes are uncommitted, as were the existing implementation changes. Include the new `llama.cpp/freetoken` directory in source backups; archiving Git HEAD alone omits it.
