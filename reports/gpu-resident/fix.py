from pathlib import Path
p=Path('llama.cpp/src/llama-moe-offload.cpp');s=p.read_text().replace('        auto * scaled = ggml_mul(ctx, ws->contributions, ws->scales);','        ggml_set_output(zero); // Keep the additive identity immutable across calls.\n        auto * scaled = ggml_mul(ctx, ws->contributions, ws->scales);');p.write_text(s)
