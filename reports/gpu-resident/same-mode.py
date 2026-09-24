from pathlib import Path
p=Path('llama.cpp/include/llama.h');s=p.read_text().replace('        bool moe_fast;', '        bool moe_fast;\n        bool moe_device_tensors; // Keep GPU-layer decode activations and reduction on device.');p.write_text(s)
p=Path('llama.cpp/src/llama-context.cpp');s=p.read_text().replace('        options.fast = params.moe_fast;', '        options.fast = params.moe_fast;\n        options.device_tensors = params.moe_device_tensors;').replace('        /*.moe_fast                     =*/ false,','        /*.moe_fast                     =*/ false,\n        /*.moe_device_tensors           =*/ true,');p.write_text(s)
p=Path('llama.cpp/src/llama-moe-offload.h');s=p.read_text().replace('    bool fast = false;', '    bool fast = false;\n    bool device_tensors = true;');p.write_text(s)
p=Path('llama.cpp/src/llama-moe-offload.cpp');s=p.read_text().replace('return state->options.device && !state->options.cpu_miss_percent', 'return state->options.device_tensors && state->options.device && !state->options.cpu_miss_percent');p.write_text(s)
p=Path('llama.cpp/tests/test-moe-integration.cpp');s=p.read_text().replace('static void pretrained(const std::string & path, bool synthetic = false) {','''static void pretrained(const std::string & path, bool synthetic = false) {
    const bool device_boundary = test_device && std::getenv("FT_TEST_DEVICE_BOUNDARY");''');s=s.replace('    p.use_extra_bufts = false;\n    llama_model_ptr baseline', '''    p.use_extra_bufts = false;
    if (device_boundary) { p.n_gpu_layers = 99; p.moe_cpu_tile_bytes = 64 * 1024 * 1024; }
    llama_model_ptr baseline''');s=s.replace('    llama_context_ptr a(llama_init_from_model(baseline.get(), cp));\n    cache_params(cp, 64 * 1024 * 1024);','''    if (device_boundary) {
        cache_params(cp, 64 * 1024 * 1024);
        cp.moe_device_tensors = false;
        cp.op_offload = cp.offload_kqv = true;
    }
    llama_context_ptr a(llama_init_from_model(baseline.get(), cp));
    cache_params(cp, 64 * 1024 * 1024);
    cp.moe_device_tensors = true;''');s=s.replace('                gpu_quality_ok &= kl <= 0.01 && tv <= 0.05;','''                gpu_quality_ok &= kl <= 0.01 && tv <= 0.05;
                if (device_boundary) { check(maximum <= 1e-5, "device boundary changed same-mode logits"); }''');p.write_text(s)
