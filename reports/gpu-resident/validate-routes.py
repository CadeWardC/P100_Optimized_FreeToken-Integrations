from pathlib import Path
p=Path('llama.cpp/src/llama-moe-offload.cpp');s=p.read_text();s=s.replace('    ggml_tensor * contributions = nullptr, * scales = nullptr, * weights = nullptr, * out = nullptr;','    ggml_tensor * contributions = nullptr, * scales = nullptr, * weights = nullptr, * out = nullptr, * checked_ids = nullptr;')
s=s.replace('    return state->options.device && !state->options.cpu_miss_percent && n_tokens == 1;','''    return state->options.device && !state->options.cpu_miss_percent && n_tokens == 1 &&
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(state->options.device),
                "ggml_backend_cuda_moe_read_routes") != nullptr;''')
s=s.replace('    ggml_backend_tensor_get(ids, routes.data(), 0, ggml_nbytes(ids));\n    std::vector<float> scales(top_k);\n    for (size_t a = 0; a < top_k; ++a) { scales[a] = bank->output_scale(routes[a]); }','    std::vector<float> scales(top_k);')
s=s.replace('        ws->weights = ggml_dup_tensor(ctx, ws->scales);','        ws->weights = ggml_dup_tensor(ctx, ws->scales);\n        ws->checked_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, int64_t(top_k));\n        ggml_set_input(ws->checked_ids);\n        ggml_set_output(ws->checked_ids);')
s=s.replace('        ggml_build_forward_expand(ws->graph, sum);','        ggml_build_forward_expand(ws->graph, ws->checked_ids);\n        ggml_build_forward_expand(ws->graph, sum);')
s=s.replace('    auto & merge = *state->merge;\n    copy_device_rows','''    auto & merge = *state->merge;
    using read_routes_fn = void (*)(ggml_backend_t, const ggml_tensor *, const ggml_tensor *, ggml_tensor *, int32_t *);
    auto read_routes = reinterpret_cast<read_routes_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(state->options.device), "ggml_backend_cuda_moe_read_routes"));
    read_routes(backend, ids, weights, merge.checked_ids, routes.data());
    for (size_t a = 0; a < top_k; ++a) { scales[a] = bank->output_scale(routes[a]); }
    copy_device_rows''')
p.write_text(s)
p=Path('llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu');s=p.read_text();pos=s.index('static void * ggml_backend_cuda_reg_get_proc_address(');s=s[:pos]+'''static __global__ void moe_checked_routes(const int32_t * ids, const float * weights, int32_t * checked, int64_t count) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { checked[i] = isfinite(weights[i]) ? ids[i] : -1; }
}

static void ggml_backend_cuda_moe_read_routes(ggml_backend_t backend, const ggml_tensor * ids,
        const ggml_tensor * weights, ggml_tensor * checked, int32_t * host) {
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx->device);
    const int64_t count = ggml_nelements(ids);
    moe_checked_routes<<<(count + 255) / 256, 256, 0, ctx->stream()>>>(
            static_cast<const int32_t *>(ids->data), static_cast<const float *>(weights->data),
            static_cast<int32_t *>(checked->data), count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(host, checked->data, count * sizeof(int32_t), cudaMemcpyDeviceToHost, ctx->stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream()));
}

''' +s[pos:];s=s.replace('    GGML_UNUSED(reg);\n    if (strcmp(name, "ggml_backend_comm_init")','    GGML_UNUSED(reg);\n    if (strcmp(name, "ggml_backend_cuda_moe_read_routes") == 0) { return (void *)ggml_backend_cuda_moe_read_routes; }\n    if (strcmp(name, "ggml_backend_comm_init")');p.write_text(s)
