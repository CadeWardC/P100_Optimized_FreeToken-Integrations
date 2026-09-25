#pragma once

#include "../freetoken/include/freetoken/moe.h"

using llama_moe_activation = freetoken::activation;
using llama_moe_cache_metrics = freetoken::cache_metrics;
using llama_moe_cached_executor = freetoken::cached_executor;
using llama_moe_cpu_op = freetoken::cpu_op;
using llama_moe_execution_stats = freetoken::execution_stats;
using llama_moe_executor_options = freetoken::executor_options;
using llama_moe_host_bank = freetoken::host_bank;
using llama_moe_reference_result = freetoken::reference_result;
using llama_moe_route_tile = freetoken::route_tile;

inline std::vector<llama_moe_route_tile> llama_moe_plan_routes(const std::vector<int32_t> & ids, int64_t n_expert, size_t capacity) {
    return freetoken::plan_routes(ids, n_expert, capacity);
}
inline size_t llama_moe_min_tile_bytes(const llama_moe_host_bank & bank, ggml_backend_buffer_type_t buft = nullptr) {
    return freetoken::min_tile_bytes(bank, buft);
}
inline size_t llama_moe_min_tile_bytes(const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down) {
    return freetoken::min_tile_bytes(gate, up, down);
}
inline llama_moe_reference_result llama_moe_reference_execute(const llama_moe_host_bank & bank,
        const std::vector<float> & input, size_t n_tokens, size_t top_k,
        const std::vector<int32_t> & ids, const std::vector<float> & weights,
        size_t budget, int n_threads = 1, llama_moe_activation activation = llama_moe_activation::silu) {
    return freetoken::reference_execute(bank, input, n_tokens, top_k, ids, weights, budget, n_threads, activation);
}
inline ggml_status llama_moe_eager_compute(ggml_tensor * dst, llama_moe_cpu_op & op) {
    return freetoken::eager_compute(dst, op);
}
inline ggml_tensor * llama_moe_cpu_graph(ggml_context * ctx, ggml_tensor * input,
        ggml_tensor * ids, ggml_tensor * weights, llama_moe_cpu_op * op) {
    return freetoken::cpu_graph(ctx, input, ids, weights, op);
}
