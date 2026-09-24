#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <exception>
#include <vector>
#include <array>

enum class llama_moe_activation { silu, gelu };

// Experimental post-routing boundary for bias-free gated experts.
// The owner must retain immutable tensors, their contexts, and their buffers.
class llama_moe_host_bank {
public:
    // Null up selects a fused [n_embd, 2*n_ff, n_expert] gate/up tensor.
    llama_moe_host_bank(std::shared_ptr<const void> owner,
                       const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
                       const ggml_tensor * down_scale = nullptr);

    int64_t n_embd() const;
    int64_t n_ff() const;
    int64_t n_expert() const;
    size_t host_bytes() const;
    size_t expert_bytes() const;
    size_t projection_bytes(size_t projection) const;
    const ggml_tensor * tensor(size_t projection) const;
    float output_scale(int32_t expert) const;

private:
    std::shared_ptr<const void> owner;
    const ggml_tensor * tensors[3];
    const ggml_tensor * down_scale;
    std::shared_ptr<std::array<ggml_tensor, 2>> split_tensors;
    size_t bytes_per_expert = 0;
};

struct llama_moe_route_tile {
    std::vector<int32_t> experts;
    std::vector<size_t> assignments;
    std::vector<int32_t> slots;
};

// IDs are in [top_k, n_tokens] order. Every assignment survives deduplication.
std::vector<llama_moe_route_tile> llama_moe_plan_routes(
        const std::vector<int32_t> & ids, int64_t n_expert, size_t capacity);

struct llama_moe_execution_stats {
    // Host elapsed times include submission and completion waits.
    uint64_t weight_load_us = 0;
    uint64_t input_upload_us = 0;
    uint64_t compute_us = 0;
    uint64_t output_readback_us = 0;
    uint64_t merge_us = 0;
    uint64_t total_us = 0;
    uint64_t cpu_branch_us = 0, gpu_branch_us = 0, overlap_us = 0;
    uint64_t cpu_assignments = 0, gpu_assignments = 0;
    uint64_t input_upload_bytes = 0;
    uint64_t prefetched_experts = 0;
    uint64_t output_readback_bytes = 0;

    size_t host_bank_bytes = 0;
    size_t peak_tile_weight_bytes = 0; // backend allocation, including padding
    size_t peak_compute_bytes = 0;     // scheduler arena, workspace reported separately
    size_t copied_weight_bytes = 0;
    size_t tiles = 0;
    size_t peak_host_vector_bytes = 0;
    size_t peak_metadata_bytes = 0;
    size_t peak_workspace_bytes = 0;
    size_t peak_temporary_bytes = 0;
};

struct llama_moe_reference_result : llama_moe_execution_stats {
    std::vector<float> output;
    std::vector<float> expert_outputs; // [n_embd, top_k, n_tokens], before routing weights
};

struct llama_moe_cache_metrics {
    size_t budget = 0, weights = 0, metadata = 0, workspace = 0;
    uint64_t hits = 0, loads = 0, evictions = 0, copied_bytes = 0, load_wait_us = 0;
    size_t staging = 0;
    size_t activation_table = 0;
    double cpu_bytes_per_us = 0, transfer_bytes_per_us = 0;
};

struct llama_moe_executor_options {
    ggml_backend_dev_t device = nullptr; // Null keeps the CPU integration path.
    size_t staging_bytes = 1024 * 1024;
    size_t compute_bytes = 64 * 1024 * 1024;
    ggml_abort_callback abort_callback = nullptr;
    void * abort_data = nullptr;
    int cpu_miss_percent = 0; // 0 disables; -1 calibrates concurrent CPU/transfer bandwidth; 1..100 fixes the split.
    bool pipeline = false;
    bool fast = false;
};

// Serialized eager executor. Device execution must be called outside a backend callback.
class llama_moe_cached_executor {
public:
    llama_moe_cached_executor(const std::vector<std::shared_ptr<llama_moe_host_bank>> & banks, size_t budget,
            llama_moe_executor_options options = {});
    ~llama_moe_cached_executor();
    llama_moe_cached_executor(const llama_moe_cached_executor &) = delete;
    llama_moe_cached_executor & operator=(const llama_moe_cached_executor &) = delete;
    llama_moe_reference_result execute(size_t layer, const std::vector<float> & input, size_t n_tokens, size_t top_k,
            const std::vector<int32_t> & ids, const std::vector<float> & weights, int n_threads,
            llama_moe_activation activation);
    llama_moe_cache_metrics metrics() const;
    // Caller must serialize with execute. Failed allocation preserves the old cache.
    void resize(size_t budget);
private:
    struct impl;
    std::unique_ptr<impl> state;
};

// Synchronous CPU prototype. Routing is complete before any graph allocation.
// Each graph owns compact weights; the full host banks are never graph inputs.
// Supports optional per-expert output scales retained by the bank.
llama_moe_reference_result llama_moe_reference_execute(
        const llama_moe_host_bank & bank,
        const std::vector<float> & input, size_t n_tokens, size_t top_k,
        const std::vector<int32_t> & ids, const std::vector<float> & weights,
        size_t tile_weight_budget, int n_threads = 1,
        llama_moe_activation activation = llama_moe_activation::silu);

size_t llama_moe_min_tile_bytes(const llama_moe_host_bank & bank, ggml_backend_buffer_type_t buft = nullptr);
size_t llama_moe_min_tile_bytes(const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down);

// Owned by the graph result, with no full-bank tensor edges in the graph.
struct llama_moe_cpu_op {
    std::shared_ptr<llama_moe_host_bank> bank;
    std::shared_ptr<llama_moe_cached_executor> executor;
    ggml_tensor * output = nullptr;
    bool eager = false;
    size_t layer = 0;
    size_t budget = 0;
    int n_threads = 1;
    llama_moe_activation activation = llama_moe_activation::silu;
    std::exception_ptr error;
    llama_moe_execution_stats metrics;
};

ggml_status llama_moe_eager_compute(ggml_tensor * dst, llama_moe_cpu_op & op);

ggml_tensor * llama_moe_cpu_graph(ggml_context * ctx, ggml_tensor * input,
        ggml_tensor * ids, ggml_tensor * weights, llama_moe_cpu_op * op);
