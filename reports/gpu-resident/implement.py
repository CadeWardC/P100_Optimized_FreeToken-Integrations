from pathlib import Path
p=Path('llama.cpp/src/llama-moe-offload.cpp')
s=p.read_text()
s=s.replace('context_ptr make_context() {','context_ptr make_context(size_t nodes = 64) {').replace('64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false)','(nodes + 32) * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false)')
s=s.replace('        std::vector<float> x(ggml_nelements(input));','''        if (external && op.executor && op.executor->supports_device(input->ne[1])) {
            op.metrics = op.executor->execute_device(op.layer, input, ids, weights, dst, op.activation);
            return;
        }
        std::vector<float> x(ggml_nelements(input));''')
pos=s.index('struct expert_workspace {')
s=s[:pos]+'''// Copy contiguous rows without moving tensor contents through host memory.
void copy_device_rows(const ggml_tensor * src, size_t src_row, ggml_tensor * dst, size_t dst_row, size_t width) {
    auto a = *src;
    auto b = *dst;
    for (auto * t : {&a, &b}) {
        t->ne[0] = int64_t(width);
        t->nb[0] = sizeof(float);
        for (int d = 1; d < GGML_MAX_DIMS; ++d) { t->ne[d] = 1; t->nb[d] = width * sizeof(float); }
        t->view_src = nullptr;
        t->view_offs = 0;
    }
    a.data = static_cast<char *>(src->data) + src_row * width * sizeof(float);
    b.data = static_cast<char *>(dst->data) + dst_row * width * sizeof(float);
    ggml_backend_tensor_copy(&a, &b);
}

struct device_merge_workspace {
    context_ptr ctx{nullptr, ggml_free};
    allocator_ptr alloc{nullptr, ggml_gallocr_free};
    ggml_tensor * contributions = nullptr, * scales = nullptr, * weights = nullptr, * out = nullptr;
    ggml_cgraph * graph = nullptr;
    size_t top_k = 0, compute_bytes = 0;
};

''' +s[pos:]
s=s.replace('const std::unordered_set<int32_t> * selected = nullptr, expert_workspace_cache * workspaces = nullptr) {','''const std::unordered_set<int32_t> * selected = nullptr, expert_workspace_cache * workspaces = nullptr,
        const ggml_tensor * device_input = nullptr, device_merge_workspace * merge = nullptr) {''')
s=s.replace('input.size() != checked_mul(n_tokens, embd) ||','(!device_input && input.size() != checked_mul(n_tokens, embd)) ||').replace('ids.size() != assignments || weights.size() != assignments','ids.size() != assignments || (!device_input && weights.size() != assignments)')
s=s.replace('    result.output.resize(input.size(), 0.0f);\n    result.expert_outputs.resize(checked_mul(assignments, embd));','''    if (!device_input) {
        result.output.resize(input.size(), 0.0f);
        result.expert_outputs.resize(checked_mul(assignments, embd));
    }''')
s=s.replace('            for (size_t a = 0; a < count; ++a) {\n                const size_t token = tile.assignments[begin + a] / top_k;','            for (size_t a = 0; !device_input && a < count; ++a) {\n                const size_t token = tile.assignments[begin + a] / top_k;')
s=s.replace('            ggml_backend_tensor_set(x, packed.data(), 0, ggml_nbytes(x));','''            if (device_input) {
                for (size_t a = 0; a < count; ++a) { copy_device_rows(device_input, 0, x, a, embd); }
            } else {
                ggml_backend_tensor_set(x, packed.data(), 0, ggml_nbytes(x));
            }''')
s=s.replace('result.input_upload_bytes += ggml_nbytes(x) + ggml_nbytes(route);','result.input_upload_bytes += (device_input ? 0 : ggml_nbytes(x)) + ggml_nbytes(route);')
s=s.replace('''            ggml_backend_tensor_get(out, packed.data(), 0, ggml_nbytes(out));
            result.output_readback_us += elapsed_us(readback_start);
            result.output_readback_bytes += ggml_nbytes(out);''','''            if (device_input) {
                for (size_t a = 0; a < count; ++a) {
                    copy_device_rows(out, a, merge->contributions, tile.assignments[begin + a], embd);
                }
            } else {
                ggml_backend_tensor_get(out, packed.data(), 0, ggml_nbytes(out));
                result.output_readback_us += elapsed_us(readback_start);
                result.output_readback_bytes += ggml_nbytes(out);
            }''')
s=s.replace('for (size_t a = 0; a < count; ++a) {\n                const float scale', 'for (size_t a = 0; !device_input && a < count; ++a) {\n                const float scale')
s=s.replace('for (size_t a = 0; a < assignments; ++a) {\n        for (size_t d', 'for (size_t a = 0; !device_input && a < assignments; ++a) {\n        for (size_t d')
s=s.replace('    expert_workspace_cache workspaces;','    expert_workspace_cache workspaces;\n    std::unique_ptr<device_merge_workspace> merge;')
pos=s.index('llama_moe_reference_result llama_moe_cached_executor::execute(')
s=s[:pos]+'''bool llama_moe_cached_executor::supports_device(size_t n_tokens) const {
    return state->options.device && !state->options.cpu_miss_percent && n_tokens == 1;
}

ggml_backend_dev_t llama_moe_cached_executor::device() const { return state->options.device; }

llama_moe_execution_stats llama_moe_cached_executor::execute_device(size_t layer, const ggml_tensor * input,
        const ggml_tensor * ids, const ggml_tensor * weights, ggml_tensor * dst, llama_moe_activation activation) {
    const auto start = moe_clock::now();
    const auto & bank = state->banks.at(layer);
    if (!input || !ids || !weights || !dst || !supports_device(input->ne[1]) ||
            input->ne[0] != bank->n_embd() || input->ne[2] != 1 || input->ne[3] != 1 ||
            ids->type != GGML_TYPE_I32 || ids->ne[0] <= 0 || ids->ne[0] > bank->n_expert() ||
            ggml_nelements(ids) != ids->ne[0] || weights->ne[0] != 1 || weights->ne[1] != ids->ne[0] ||
            ggml_nelements(weights) != ids->ne[0] || !ggml_are_same_shape(input, dst)) {
        throw std::invalid_argument("Invalid device MoE boundary");
    }
    auto * backend = state->backend.get();
    for (const auto * t : {input, ids, weights, static_cast<const ggml_tensor *>(dst)}) {
        if (!t->buffer || !ggml_is_contiguous(t) || (t != ids && t->type != GGML_TYPE_F32) ||
                ggml_backend_buffer_is_host(t->buffer) ||
                ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer)) != state->options.device) {
            throw std::invalid_argument("Device MoE tensors must reside on the executor GPU");
        }
    }
    check_cancel(state->options);
    backend_drain drain{backend};
    const size_t top_k = size_t(ids->ne[0]);
    std::vector<int32_t> routes(top_k);
    ggml_backend_tensor_get(ids, routes.data(), 0, ggml_nbytes(ids));
    std::vector<float> scales(top_k);
    for (size_t a = 0; a < top_k; ++a) { scales[a] = bank->output_scale(routes[a]); }
    if (!state->merge || state->merge->top_k != top_k) {
        state->merge.reset();
        auto ws = std::make_unique<device_merge_workspace>();
        const size_t nodes = checked_add(32, checked_mul(top_k, 4));
        ws->ctx = make_context(nodes);
        auto * ctx = ws->ctx.get();
        ws->top_k = top_k;
        ws->contributions = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, bank->n_embd(), int64_t(top_k));
        ws->scales = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, int64_t(top_k));
        ws->weights = ggml_dup_tensor(ctx, ws->scales);
        auto * zero = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bank->n_embd());
        for (auto * t : {ws->contributions, ws->scales, ws->weights, zero}) { ggml_set_input(t); }
        auto * scaled = ggml_mul(ctx, ws->contributions, ws->scales);
        ggml_set_output(scaled);
        auto * weighted = ggml_mul(ctx, scaled, ws->weights);
        ggml_set_output(weighted);
        auto * sum = zero;
        for (size_t a = 0; a < top_k; ++a) {
            sum = ggml_add(ctx, sum, ggml_view_1d(ctx, weighted, bank->n_embd(), a * weighted->nb[1]));
        }
        ws->out = sum;
        ggml_set_output(sum);
        ws->graph = ggml_new_graph_custom(ctx, nodes, false);
        ggml_build_forward_expand(ws->graph, sum);
        for (int i = 0; i < ggml_graph_n_nodes(ws->graph); ++i) {
            if (!ggml_backend_supports_op(backend, ggml_graph_node(ws->graph, i))) {
                throw std::invalid_argument("MoE device does not support output reduction");
            }
        }
        ws->alloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
        if (!ws->alloc) { throw std::bad_alloc(); }
        ggml_gallocr_reserve_n_size(ws->alloc.get(), ws->graph, nullptr, nullptr, &ws->compute_bytes);
        const size_t retained = state->workspaces.workspace ? state->workspaces.workspace->compute_bytes : 0;
        const size_t table = state->activation_buffer ? ggml_backend_buffer_get_size(state->activation_buffer.get()) : 0;
        if (checked_add(checked_add(ws->compute_bytes, retained), table) > state->options.compute_bytes) {
            throw std::runtime_error("MoE device merge budget exceeded");
        }
        if (!ggml_gallocr_reserve(ws->alloc.get(), ws->graph) || !ggml_gallocr_alloc_graph(ws->alloc.get(), ws->graph)) {
            throw std::bad_alloc();
        }
        ws->compute_bytes = ggml_gallocr_get_buffer_size(ws->alloc.get(), 0);
        if (checked_add(checked_add(ws->compute_bytes, retained), table) > state->options.compute_bytes) {
            throw std::runtime_error("MoE device merge allocation exceeded budget");
        }
        ggml_backend_tensor_memset(zero, 0, 0, ggml_nbytes(zero));
        state->merge = std::move(ws);
    }
    auto & merge = *state->merge;
    copy_device_rows(weights, 0, merge.weights, 0, top_k);
    ggml_backend_tensor_set(merge.scales, scales.data(), 0, ggml_nbytes(merge.scales));
    auto options = state->options;
    options.compute_bytes -= merge.compute_bytes;
    auto result = execute_ffn(*bank, {}, 1, top_k, routes, {}, 0, 1, activation,
            state->cache.get(), backend, uint32_t(layer), bank, options, state->gelu_table,
            nullptr, &state->workspaces, input, &merge);
    const auto merge_start = moe_clock::now();
    if (ggml_backend_graph_compute(backend, merge.graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("MoE device merge failed");
    }
    check_cancel(options);
    copy_device_rows(merge.out, 0, dst, 0, size_t(bank->n_embd()));
    ggml_backend_synchronize(backend);
    result.merge_us = elapsed_us(merge_start);
    result.input_upload_bytes += ggml_nbytes(merge.scales);
    result.peak_compute_bytes += merge.compute_bytes;
    result.peak_metadata_bytes += ggml_get_mem_size(merge.ctx.get());
    result.peak_host_vector_bytes += (routes.capacity() + scales.capacity()) * sizeof(float);
    result.peak_temporary_bytes += merge.compute_bytes + ggml_get_mem_size(merge.ctx.get()) +
            (routes.capacity() + scales.capacity()) * sizeof(float);
    result.total_us = result.gpu_branch_us = elapsed_us(start);
    return result;
}

''' +s[pos:]
s=s.replace('    const auto & bank = state->banks.at(layer);\n    if (!state->options.cpu_miss_percent)', '    state->merge.reset();\n    const auto & bank = state->banks.at(layer);\n    if (!state->options.cpu_miss_percent)')
s=s.replace('state->workspaces.workspace ? state->workspaces.workspace->compute_bytes : 0,','(state->workspaces.workspace ? state->workspaces.workspace->compute_bytes : 0) +\n                    (state->merge ? state->merge->compute_bytes : 0),')
s=s.replace('    state->workspaces.workspace.reset();\n    state->cache.swap', '    state->workspaces.workspace.reset();\n    state->merge.reset();\n    state->cache.swap')
p.write_text(s)
p=Path('llama.cpp/src/llama-moe-offload.h');s=p.read_text().replace('    llama_moe_cache_metrics metrics() const;','''    // The caller drains producer backends before entry. This call drains before returning.
    bool supports_device(size_t n_tokens) const;
    ggml_backend_dev_t device() const;
    llama_moe_execution_stats execute_device(size_t layer, const ggml_tensor * input, const ggml_tensor * ids,
            const ggml_tensor * weights, ggml_tensor * dst, llama_moe_activation activation);
    llama_moe_cache_metrics metrics() const;''');p.write_text(s)
p=Path('llama.cpp/src/llama-context.cpp');s=p.read_text();needle='        // - norm may be automatically assigned'
pos=s.index(needle);s=s[:pos]+'''#ifdef LLAMA_MOE_CPU
        if (moe_executor && moe_executor->supports_device(ubatch.n_tokens) && strcmp(name, "ffn_moe_cpu") == 0) {
            for (const auto & backend : backends) {
                if (ggml_backend_get_device(backend.get()) == moe_executor->device()) {
                    // The scheduler executes this custom node outside the backend graph.
                    ggml_backend_sched_set_tensor_backend(sched_active(), cur, backend.get());
                    for (int i = 0; i < 3; ++i) {
                        ggml_backend_sched_set_tensor_backend(sched_active(), cur->src[i], backend.get());
                    }
                    break;
                }
            }
        }
#endif

''' +s[pos:];p.write_text(s)
