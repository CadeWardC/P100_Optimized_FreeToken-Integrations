from pathlib import Path
p=Path('llama.cpp/tests/test-moe-offload.cpp');s=p.read_text();pos=s.index('static void test_device_executor(');s=s[:pos]+'''static void test_device_boundary(ggml_backend_dev_t device) {
    auto owner = std::make_shared<fixture>(GGML_TYPE_Q4_0, 7, false, 256, 256, true, true, true);
    auto bank = std::make_shared<llama_moe_host_bank>(owner, owner->tensors[0], nullptr, owner->tensors[2], owner->scales);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    ggml_init_params params{32 * ggml_tensor_overhead(), nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 256, 1);
    auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 3, 1);
    auto * w = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, 3, 1);
    auto * out = ggml_dup_tensor(ctx.get(), x);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    check(bool(buffer), "device boundary buffer");
    for (bool fast : {false, true}) {
        int checks = 0, cancel_at = 0;
        llama_moe_executor_options options;
        options.device = device;
        options.fast = fast;
        options.pipeline = true;
        options.staging_bytes = 4093;
        options.abort_callback = [](void * p) {
            auto * values = static_cast<std::pair<int *, int *> *>(p);
            return ++*values->first == *values->second;
        };
        std::pair<int *, int *> cancellation{&checks, &cancel_at};
        options.abort_data = &cancellation;
        const size_t budget = llama_moe_min_tile_bytes(*bank, ggml_backend_dev_buffer_type(device));
        llama_moe_cached_executor executor({bank, bank}, budget, options);
        llama_moe_cached_executor reference({bank, bank}, budget, options);
        for (int step = 0; step < 16; ++step) {
            std::vector<float> input(256), weights{0.3f, -0.2f, 0.7f}, actual(256);
            for (size_t d = 0; d < input.size(); ++d) { input[d] = 0.1f * std::cos(float(d + step)); }
            std::vector<int32_t> routes{step % 7, (step + 3) % 7, step % 7};
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
            ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
            const auto activation = step < 12 ? llama_moe_activation::gelu : llama_moe_activation::silu;
            const auto stats = executor.execute_device(step % 2, x, ids, w, out, activation);
            ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
            const auto expected = reference.execute(step % 2, input, 1, 3, routes, weights, 2, activation);
            device_close(actual, expected.output);
            check(stats.output_readback_bytes == 0 && stats.input_upload_bytes == 3 * (sizeof(float) + sizeof(int32_t)),
                    "device boundary transferred activations or outputs");
            check(executor.metrics().retained_compute_bytes + executor.metrics().activation_table <= options.compute_bytes,
                    "device boundary compute budget");
            if (step == 5) {
                const auto retained = executor.metrics().retained_compute_bytes;
                rejects([&] { executor.resize(1); });
                check(executor.metrics().retained_compute_bytes == retained, "failed device resize lost workspaces");
            }
            if (step == 8) {
                executor.resize(2 * budget);
                check(executor.metrics().retained_compute_bytes == 0, "device resize retained workspaces");
            }
        }
        check(executor.metrics().evictions > 0, "device boundary did not evict");
        for (int point : {1, 3, 10, 30}) {
            executor.resize(budget);
            checks = 0;
            cancel_at = point;
            bool failed = false;
            try { executor.execute_device(0, x, ids, w, out, llama_moe_activation::gelu); }
            catch (const std::runtime_error &) { failed = true; }
            check(failed, "device cancellation point not reached");
            cancel_at = 0;
            executor.execute_device(0, x, ids, w, out, llama_moe_activation::gelu);
        }
        options.compute_bytes = 1;
        llama_moe_cached_executor limited({bank}, budget, options);
        rejects([&] { limited.execute_device(0, x, ids, w, out, llama_moe_activation::silu); });
    }
    std::puts("PASS GPU tensor boundary: scales, duplicate routes, ordered merge, eviction, resize, cancellation and budget");
}

''' +s[pos:];s=s.replace('    test_persistent_workspace(device);','    test_device_boundary(device);\n    test_persistent_workspace(device);');p.write_text(s)
