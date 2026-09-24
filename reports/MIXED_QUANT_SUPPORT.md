# Mixed K/I quantization support

Implemented for the local FreeToken/P100 fork on 2026-09-13.

The rebuilt engine is `build/ft-windows-cuda-dev/bin/llama-server.exe`. This is the default development engine discovered by the desktop application. Existing installations configured to use another executable must select this engine to use the change.

## Change

- Centralized the host-bank and cache-storage type check; added Q2_K, Q3_K, Q4_K, Q5_K and IQ2_S alongside F32, F16, Q4_0 and Q8_0.
- Partitioned compressed cache storage by each layer's gate/up/down type combination. Layer identity remains part of cache keys.
- Reserved one padded expert per layout, then apportioned the remaining weight budget by expert size and layer count. Transfer staging is divided across layouts. Cache allocation and telemetry cover all layouts.
- Retained one compute workspace, invalidated when storage changes. Resize builds replacement storage before swapping and preserves the old cache on failure.
- Unsupported-type loader errors now name the tensor and quantization format.

## Exact model

`Qwen3.6-35B-A3B-I-COMPACT/Qwen3.6-35B-A3B-APEX-I-Mini.gguf`, architecture `qwen35moe`:

- 20 layers: IQ2_S gate, up and down.
- 18 layers: Q3_K gate/up, Q4_K down.
- 2 layers: Q3_K gate/up, Q5_K down.

Header inspection: `qwen-apex-mini-header.json`. No model weights were converted or modified.

## Verification

- Cache lifecycle suite: 30,000 LRU steps passed (`mixed-quant-cache.log`).
- CPU executor suite passed, including mixed layouts, distinct projections, duplicate layouts, decode/prefill, cold/warm cache, eviction, failed resize and recovery (`mixed-quant-cpu.log`).
- CUDA executor suite passed on an RTX 4060 Laptop GPU, including the new formats compared against ordinary GPU execution, plus existing cancellation, hybrid scheduling and resize coverage (`mixed-quant-cuda.log`). The ordinary GPU reference explicitly places its operations on GPU to avoid an unintended CPU reference.
- All 135 existing synthetic model integration comparisons passed (`mixed-quant-integration.log`).
- Exact pretrained model: ordinary CPU versus cached CPU passed full-vocabulary logits, 16 greedy tokens and state replay (`mixed-quant-pretrained-cpu.log`).
- Exact model server smoke test passed with GPU layers 99, 512 MiB expert cache, 128 MiB expert compute budget and context 1024. Default numerical mode was used. Model loaded in approximately 13 seconds; this short run measured 1.40 decode tokens/s (`mixed-quant-server.log`, `mixed-quant-server-result.json`). This is a correctness smoke test, not a tuned performance benchmark.
- Both CPU and GPU smoke runs completed “The capital of France is” with “ Paris, a city renowned for its iconic landmarks such as the Eiffel Tower,”.

## Limits

The weight budget is statically partitioned by layout; eviction cannot borrow free slots from a different layout. Projection dimensions still must match across layers. The adaptive bandwidth calibration samples a single layout and remains an approximation for mixed formats.

K/I GPU arithmetic is not guaranteed to match CPU arithmetic exactly, including in default numerical mode. GPU tests establish agreement with ordinary GPU execution; the short real-model completion is not full-vocabulary GPU/CPU parity or a general quality assessment. P100 hardware was unavailable: the build targets CUDA architectures 60 and 89, but execution was verified only on the RTX 4060. The separate upstream FreeToken application's architecture support is unchanged.
