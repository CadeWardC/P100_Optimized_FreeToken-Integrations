from pathlib import Path
p=Path('llama.cpp/ggml/src/ggml-cuda/common.cuh');s=p.read_text().replace('    // Map from first_node_ptr to cuda_graph - allows multiple graphs per context','    uint64_t graph_captures = 0, graph_replays = 0;\n    // Map from first_node_ptr to cuda_graph - allows multiple graphs per context');p.write_text(s)
p=Path('llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu');s=p.read_text();s=s.replace('            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));','            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));\n            ++cuda_ctx->graph_captures;');s=s.replace('        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));','        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));\n        if (!cuda_graph_update_required) { ++cuda_ctx->graph_replays; }')
pos=s.index('static __global__ void moe_checked_routes(')
s=s[:pos]+'''static void ggml_backend_cuda_graph_release(ggml_backend_t backend, const ggml_cgraph * graph) {
#ifdef USE_CUDA_GRAPH
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx->device);
    ggml_backend_synchronize(backend);
    if (graph && graph->n_nodes) { ctx->cuda_graphs.erase(graph->nodes[0]); }
#else
    GGML_UNUSED(backend);
    GGML_UNUSED(graph);
#endif
}

static void ggml_backend_cuda_graph_stats(ggml_backend_t backend, uint64_t * captures, uint64_t * replays) {
    *captures = *replays = 0;
#ifdef USE_CUDA_GRAPH
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    *captures = ctx->graph_captures;
    *replays = ctx->graph_replays;
#else
    GGML_UNUSED(backend);
#endif
}

''' +s[pos:]
s=s.replace('    if (strcmp(name, "ggml_backend_cuda_moe_read_routes")', '    if (strcmp(name, "ggml_backend_cuda_graph_release") == 0) { return (void *)ggml_backend_cuda_graph_release; }\n    if (strcmp(name, "ggml_backend_cuda_graph_stats") == 0) { return (void *)ggml_backend_cuda_graph_stats; }\n    if (strcmp(name, "ggml_backend_cuda_moe_read_routes")')
p.write_text(s)
p=Path('llama.cpp/src/llama-moe-offload.cpp');s=p.read_text();pos=s.index('struct device_merge_workspace {');s=s[:pos]+'''void release_device_graph(ggml_backend_t backend, ggml_cgraph * graph) {
    if (!backend || !graph) { return; }
    using release_fn = void (*)(ggml_backend_t, const ggml_cgraph *);
    auto release = reinterpret_cast<release_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "ggml_backend_cuda_graph_release"));
    if (release) { release(backend, graph); }
}

''' +s[pos:]
s=s.replace('struct device_merge_workspace {','struct device_merge_workspace {\n    ggml_backend_t backend = nullptr;\n    ~device_merge_workspace() { release_device_graph(backend, graph); }')
s=s.replace('struct expert_workspace {','struct expert_workspace {\n    ggml_backend_t backend = nullptr;\n    ~expert_workspace() { release_device_graph(backend, graph); }')
s=s.replace('                ws.ctx = make_context();','                ws.backend = cpu ? nullptr : backend;\n                ws.ctx = make_context();')
s=s.replace('        auto ws = std::make_unique<device_merge_workspace>();','        auto ws = std::make_unique<device_merge_workspace>();\n        ws->backend = backend;')
s=s.replace('    const auto & storage = state->cache->storage_stats();','''    const auto & storage = state->cache->storage_stats();
    uint64_t captures = 0, replays = 0;
    using stats_fn = void (*)(ggml_backend_t, uint64_t *, uint64_t *);
    auto graph_stats = reinterpret_cast<stats_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(state->backend.get())), "ggml_backend_cuda_graph_stats"));
    if (graph_stats) { graph_stats(state->backend.get(), &captures, &replays); }''')
s=s.replace('state->workspaces.builds, state->workspaces.reuses};','state->workspaces.builds, state->workspaces.reuses, captures, replays};')
p.write_text(s)
p=Path('llama.cpp/src/llama-moe-offload.h');s=p.read_text().replace('    uint64_t workspace_builds = 0, workspace_reuses = 0;','    uint64_t workspace_builds = 0, workspace_reuses = 0;\n    uint64_t graph_captures = 0, graph_replays = 0;');p.write_text(s)
p=Path('llama.cpp/include/llama.h');s=p.read_text().replace('        uint64_t expert_workspace_builds, expert_workspace_reuses;','        uint64_t expert_workspace_builds, expert_workspace_reuses;\n        uint64_t expert_graph_captures, expert_graph_replays;');p.write_text(s)
p=Path('llama.cpp/src/llama-context.cpp');s=p.read_text().replace('        data.expert_workspace_reuses = m.workspace_reuses;', '        data.expert_workspace_reuses = m.workspace_reuses;\n        data.expert_graph_captures = m.graph_captures;\n        data.expert_graph_replays = m.graph_replays;');needle='        LLAMA_LOG_INFO("%s: MoE persistent compute=';pos=s.index(needle);s=s[:pos]+'''        LLAMA_LOG_INFO("%s: MoE CUDA graph captures/replays=%" PRIu64 "/%" PRIu64 "\\n",
                __func__, moe.expert_graph_captures, moe.expert_graph_replays);
''' +s[pos:];p.write_text(s)
p=Path('scripts/build-cuda-dev.cmd');s=p.read_text().replace('setlocal','setlocal\nif not defined FT_CUDA_GRAPHS set "FT_CUDA_GRAPHS=OFF"').replace('-DGGML_CUDA_GRAPHS=OFF','-DGGML_CUDA_GRAPHS=%FT_CUDA_GRAPHS%');p.write_text(s)
Path('scripts/build-cuda-graphs.cmd').write_text('@echo off\nsetlocal\nset "FT_CUDA_GRAPHS=ON"\ncall "%~dp0build-cuda-dev.cmd"\nexit /b %errorlevel%\n')
