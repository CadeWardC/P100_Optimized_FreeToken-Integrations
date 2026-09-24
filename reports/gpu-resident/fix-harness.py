from pathlib import Path
p=Path('llama.cpp/tests/test-moe-integration.cpp');s=p.read_text();a=s.index('    if (device_boundary) { p.n_gpu_layers');b=s.index('static void pretrained');assert a<b;s=s[:a]+s[a:].replace('    if (device_boundary) { p.n_gpu_layers = 99; p.moe_cpu_tile_bytes = 64 * 1024 * 1024; }\n','',1);p.write_text(s)
