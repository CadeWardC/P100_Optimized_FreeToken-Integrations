# K2-Horizon model support

The FreeToken engine includes the model authors' K2-Horizon implementation from
[MBZUAI-IFM/llama.cpp, branch model/K2Horizon](https://github.com/MBZUAI-IFM/llama.cpp/tree/model/K2Horizon),
pinned at `35999d101` (the five K2 commits after `4e97ac8`).
The original announcement is [llama.cpp discussion 28308](https://github.com/ggml-org/llama.cpp/discussions/28308).

This is a selective backport, not a replacement of the FreeToken fork. It adds
the `k2-horizon` architecture, its tensor and metadata definitions, grouped RMS
normalization, attention graph, tokenizer, and the authors' conversion/template
files. Existing FreeToken CUDA and expert-cache changes are retained.

Local adaptations resolve registration differences against the older base,
identify the 36-layer/4096-hidden dense model as 7B, and use the existing Unicode
splitter for K2's combining marks and zero-width joiners. The fork's escaped
Unicode regex fails with the Windows standard regex library. A regression test
covers accents, both joiners, contractions, digit groups, and line endings.

Model re-saving is disabled because the generic saver does not preserve all K2
metadata. Tensor-parallel splitting is also disabled because its generic split
rules do not cover this graph. Normal single-GPU and layer offload remain
available.

## Local model

`K2-Horizon-7B-Q4_K_M.gguf` has 36 dense layers, hidden size 4096, 32 attention
heads, 8 KV heads, and four normalization groups. The file has 327 tensors and
5,592,217,984 bytes. It declares a 524,288-token training context; this is not
a suitable default allocation for a small GPU. Verification uses 2,048 tokens.

The fork includes MoE/MoVA paths, but validation with the local 7B file does not
establish correctness for those other models. The authors' converter retains
its original supported Hugging Face model classes; existing GGUF loading does
not require re-conversion.

## Rebuild and verify

Run `scripts/build-cuda-dev.cmd` to rebuild the CUDA engine for SM60 and SM89.
The desktop application's configured server is
`build/ft-windows-cuda-dev/bin/llama-server.exe`.

Run the local-model smoke check:

```powershell
python scripts/test_k2_horizon.py --server build/ft-windows-cuda-dev/bin/llama-server.exe --model "C:\Users\cadel\Documents\Coding_Projects\models\K2 7B\K2-Horizon-7B-Q4_K_M.gguf" --output reports/k2-horizon-smoke
```

The check starts an isolated loopback server, exercises completion and chat,
records responses and logs, then stops its own server. It does not change app
settings or model files. It verifies usable output, not benchmark accuracy or
long-context quality. Inspect the recorded answers as well as the pass result.

The original upstream patch and the local GGUF header inspection are recorded
in `reports/k2-horizon-upstream.patch` and `reports/k2-horizon-header.json`.

## Verified locally

- CUDA build succeeded for SM60 and SM89; runtime checks used an RTX 4060 Laptop GPU.
- The real Q4_K_M GGUF loaded and completed text with "Paris" for France's capital.
- The model's embedded chat template produced the final answer "4" to "2 + 2".
- The dense synthetic graph matched CPU output on CUDA with normalized mean
  squared error `8.45e-08` (seed 42). Tensor splitting and saver roundtrips are
  intentionally skipped.
- `test-unicode` passed, including the K2 tokenizer regression.
- Existing MoE cache, offload, and integration checks passed, including 135 CPU
  model comparisons. Existing Llama dense and MoE CPU/CUDA comparisons passed.

Full responses and engine timing logs are in `reports/k2-horizon-smoke/`.
These checks do not establish P100 runtime performance, MoVA correctness,
long-context accuracy, or benchmark quality.
