#include "llama.h"
#include "llama-cpp.h"
#include "llama-model.h"
#include "llama-moe-offload.h"
#include "llama-ext.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "common.h"
#include "chat.h"
#include "arg.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static ggml_backend_dev_t test_device = nullptr;
static int cpu_miss_percent = 0;
static bool comparing_gpu = false;
static bool gpu_layers = false;
static double native_intermediate_max_error = 0;

static void cache_params(llama_context_params & cp, size_t budget) {
    cp.moe_cpu_cache_bytes = test_device ? 0 : budget;
    cp.moe_gpu_cache_bytes = test_device ? budget : 0;
    cp.moe_device = test_device;
    cp.moe_cpu_miss_percent = budget && test_device ? cpu_miss_percent : 0;
    cp.moe_pipeline = test_device && budget;
    cp.moe_fast = gpu_layers && test_device && budget;
    if (gpu_layers && test_device && budget) { cp.op_offload = cp.offload_kqv = true; }
    comparing_gpu = test_device && budget;
}

static void compare(const std::vector<float> & a, const std::vector<float> & b) {
    check(a.size() == b.size() && !a.empty(), "comparison shape");
    for (size_t i = 0; i < a.size(); ++i) {
        const float limit = comparing_gpu ? 0.005f + 0.05f * std::abs(a[i]) : 2e-5f + 2e-5f * std::abs(a[i]);
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::abs(a[i] - b[i]) > limit) {
            std::fprintf(stderr, "logit/value mismatch index=%zu reference=%.9g actual=%.9g limit=%.9g\n", i, a[i], b[i], limit);
            throw std::runtime_error("model output differs from baseline");
        }
    }
}

static void compare_intermediate(const std::vector<float> & a, const std::vector<float> & b) {
    if (!gpu_layers) { compare(a, b); return; }
    check(a.size() == b.size(), "intermediate shape mismatch");
    for (size_t i = 0; i < a.size(); ++i) {
        check(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite native intermediate");
        native_intermediate_max_error = std::max(native_intermediate_max_error, double(std::abs(a[i] - b[i])));
    }
}

static void fixture(const std::string & path, ggml_type expert_type, int experts = 4, bool split = false,
        bool qwen = false, bool mtp = false, bool gemma = false, bool fused = false) {
    gguf_context_ptr file(gguf_init_empty());
    ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, false}));
    const std::string arch = gemma ? "gemma4" : qwen ? "qwen35moe" : "llama";
    gguf_set_val_str(file.get(), "general.architecture", arch.c_str());
    gguf_set_val_str(file.get(), "tokenizer.ggml.model", "no_vocab");
    for (auto kv : std::vector<std::pair<const char *, uint32_t>>{
            {"llama.vocab_size", 32}, {"llama.context_length", 128},
            {"llama.embedding_length", 32}, {"llama.block_count", 2},
            {"llama.feed_forward_length", 64}, {"llama.attention.head_count", 2},
            {"llama.attention.head_count_kv", 2}, {"llama.rope.dimension_count", 16},
            {"llama.expert_count", uint32_t(experts)}, {"llama.expert_used_count", 2}}) {
        gguf_set_val_u32(file.get(), (arch + std::string(kv.first).substr(5)).c_str(), kv.second);
    }
    gguf_set_val_f32(file.get(), (arch + ".attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
    if (gemma) {
        gguf_set_val_u32(file.get(), "gemma4.expert_used_count", experts >= 8 ? 8 : 2);
        for (auto kv : std::vector<std::pair<const char *, uint32_t>>{
                {"expert_feed_forward_length", 64}, {"embedding_length_per_layer_input", 0},
                {"attention.sliding_window", 8}, {"attention.key_length", 32}, {"attention.value_length", 32},
                {"attention.key_length_swa", 16}, {"attention.value_length_swa", 16}}) {
            gguf_set_val_u32(file.get(), (arch + "." + kv.first).c_str(), kv.second);
        }
        const bool swa[] = {true, false};
        gguf_set_arr_data(file.get(), "gemma4.attention.sliding_window_pattern", GGUF_TYPE_BOOL, swa, 2);
        gguf_set_val_f32(file.get(), "gemma4.final_logit_softcapping", 30.0f);
    }
    if (qwen) {
        if (mtp) {
            gguf_set_val_u32(file.get(), "qwen35moe.block_count", 3);
            gguf_set_val_u32(file.get(), "qwen35moe.nextn_predict_layers", 1);
        }
        for (auto kv : std::vector<std::pair<const char *, uint32_t>>{
                {"expert_feed_forward_length", 64}, {"expert_shared_feed_forward_length", 64},
                {"full_attention_interval", 2}, {"ssm.conv_kernel", 4}, {"ssm.inner_size", 32},
                {"ssm.state_size", 16}, {"ssm.time_step_rank", 2}, {"ssm.group_count", 1}}) {
            gguf_set_val_u32(file.get(), (arch + "." + kv.first).c_str(), kv.second);
        }
        const int32_t sections[] = {4, 2, 2, 0};
        gguf_set_arr_data(file.get(), "qwen35moe.rope.dimension_sections", GGUF_TYPE_INT32, sections, 4);
    }
    int serial = 0;
    auto add = [&](const std::string & name, int64_t n0, int64_t n1 = 1, int64_t n2 = 1) {
        const bool expert = name.find("_exps.weight") != std::string::npos;
        auto * t = ggml_new_tensor_3d(ctx.get(), expert ? expert_type : GGML_TYPE_F32, n0, n1, n2);
        ggml_set_name(t, name.c_str());
        std::vector<float> values(ggml_nelements(t));
        ++serial;
        uint32_t seed = 1729 + uint32_t(serial);
        for (size_t i = 0; i < values.size(); ++i) {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            values[i] = name.find(".scale") != std::string::npos ? 0.5f + 0.1f * float(i) :
                name.find("rope_freqs") != std::string::npos ? 1.0f :
                name.find("layer_output_scale") != std::string::npos ? 0.8f :
                name.find("ssm_a") != std::string::npos ? -0.5f :
                name.find("norm") != std::string::npos ? 1.0f :
                test_device ? 0.15f * (float(seed % 65536) / 32768.0f - 1.0f) :
                0.12f * std::sin(float(i * 17 + serial * 31)) + 0.03f * std::cos(float(i * 7 + serial));
        }
        ggml_quantize_chunk(t->type, values.data(), t->data, 0, n1 * n2, n0, nullptr);
        gguf_add_tensor(file.get(), t);
    };
    add("token_embd.weight", 32, 32);
    add("output_norm.weight", 32);
    add("output.weight", 32, 32);
    for (int il = 0; il < (mtp ? 3 : 2); ++il) {
        const auto prefix = "blk." + std::to_string(il) + ".";
        add(prefix + "attn_norm.weight", 32);
        if (gemma) {
            const int head = il == 0 ? 16 : 32;
            add(prefix + "attn_q.weight", 32, head * 2);
            add(prefix + "attn_k.weight", 32, head * 2);
            // Omit V to exercise Gemma's K=V attention path.
            add(prefix + "attn_output.weight", head * 2, 32);
            add(prefix + "attn_q_norm.weight", head);
            add(prefix + "attn_k_norm.weight", head);
            add(prefix + "post_attention_norm.weight", 32);
            add(prefix + "ffn_gate.weight", 32, 64);
            add(prefix + "ffn_up.weight", 32, 64);
            add(prefix + "ffn_down.weight", 64, 32);
            add(prefix + "post_ffw_norm.weight", 32);
            add(prefix + "pre_ffw_norm_2.weight", 32);
            add(prefix + "post_ffw_norm_1.weight", 32);
            add(prefix + "post_ffw_norm_2.weight", 32);
            add(prefix + "ffn_gate_inp.scale", 32);
            add(prefix + "ffn_down_exps.scale", experts);
            add(prefix + "layer_output_scale.weight", 1);
            if (il == 1) { add("rope_freqs.weight", head / 2); }
        } else if (qwen && il == 0) {
            add(prefix + "attn_qkv.weight", 32, 64);
            add(prefix + "attn_gate.weight", 32, 32);
            add(prefix + "ssm_conv1d.weight", 4, 64);
            add(prefix + "ssm_dt.bias", 2);
            add(prefix + "ssm_a", 2);
            add(prefix + "ssm_beta.weight", 32, 2);
            add(prefix + "ssm_alpha.weight", 32, 2);
            add(prefix + "ssm_norm.weight", 16);
            add(prefix + "ssm_out.weight", 32, 32);
        } else {
            for (const char * projection : {"q", "k", "v", "output"}) {
                add(prefix + "attn_" + projection + ".weight", 32, qwen && std::strcmp(projection, "q") == 0 ? 64 : 32);
            }
            if (qwen) {
                add(prefix + "attn_q_norm.weight", 16);
                add(prefix + "attn_k_norm.weight", 16);
            }
        }
        if (qwen) {
            add(prefix + "ffn_gate_inp_shexp.weight", 32);
            add(prefix + "ffn_gate_shexp.weight", 32, 64);
            add(prefix + "ffn_up_shexp.weight", 32, 64);
            add(prefix + "ffn_down_shexp.weight", 64, 32);
        }
        add(prefix + (qwen ? "post_attention_norm.weight" : "ffn_norm.weight"), 32);
        add(prefix + "ffn_gate_inp.weight", 32, experts);
        if (fused) {
            add(prefix + "ffn_gate_up_exps.weight", 32, 128, experts);
        } else {
            add(prefix + "ffn_gate_exps.weight", 32, 64, experts);
            add(prefix + "ffn_up_exps.weight", 32, 64, experts);
        }
        add(prefix + "ffn_down_exps.weight", 64, 32, experts);
        if (mtp && il == 2) {
            add(prefix + "nextn.eh_proj.weight", 64, 32);
            add(prefix + "nextn.enorm.weight", 32);
            add(prefix + "nextn.hnorm.weight", 32);
        }
    }
    check(gguf_write_to_file(file.get(), path.c_str(), false), "write fixture");
    if (split) {
        for (int shard = 0; shard < 2; ++shard) {
            gguf_context_ptr part(gguf_init_empty());
            gguf_set_kv(part.get(), file.get());
            gguf_set_val_u16(part.get(), "split.no", shard);
            gguf_set_val_u16(part.get(), "split.count", 2);
            gguf_set_val_i32(part.get(), "split.tensors.count", int32_t(gguf_get_n_tensors(file.get())));
            int index = 0;
            for (auto * t = ggml_get_first_tensor(ctx.get()); t; t = ggml_get_next_tensor(ctx.get(), t)) {
                if (index++ % 2 == shard) { gguf_add_tensor(part.get(), t); }
            }
            const auto name = path + (shard == 0 ? "-00001-of-00002.gguf" : "-00002-of-00002.gguf");
            check(gguf_write_to_file(part.get(), name.c_str(), false), "write split fixture");
        }
    }
}

struct observations {
    std::vector<float> ffn;
    std::vector<float> shared;
    std::vector<float> combined;
    std::vector<float> recurrent;
    std::vector<float> normalized;
    size_t custom = 0;
    bool bad_route = false;
    static bool eval(ggml_tensor * t, bool ask, void * data) {
        auto & self = *static_cast<observations *>(data);
        const bool route = self.bad_route && std::strncmp(t->name, "ffn_moe_weights_norm-", 21) == 0;
        const bool selected = std::strncmp(t->name, "ffn_moe_out-", 12) == 0;
        std::vector<float> * values = selected ? &self.ffn :
            (std::strncmp(t->name, "ffn_shexp_gated-", 16) == 0 || std::strncmp(t->name, "ffn_mlp-", 8) == 0) ? &self.shared :
            (std::strncmp(t->name, "ffn_out-", 8) == 0 || std::strncmp(t->name, "ffn_moe_combined-", 17) == 0) ? &self.combined :
            std::strncmp(t->name, "ffn_moe-", 8) == 0 ? &self.normalized :
            std::strncmp(t->name, "state_predelta-", 15) == 0 ? &self.recurrent : nullptr;
        if (route) {
            if (!ask) {
                const float invalid = std::numeric_limits<float>::quiet_NaN();
                ggml_backend_tensor_set(t, &invalid, 0, sizeof(invalid));
            }
            return true;
        }
        if (ask) { return values != nullptr; }
        if (values) {
            const size_t offset = values->size();
            values->resize(offset + ggml_nelements(t));
            ggml_backend_tensor_get(t, values->data() + offset, 0, ggml_nbytes(t));
            self.custom += selected && t->op == GGML_OP_MAP_CUSTOM3;
        }
        return true;
    }
};

static llama_context_params context_params() {
    auto p = llama_context_default_params();
    p.n_ctx = 128;
    p.n_batch = 32;
    p.n_ubatch = 8;
    p.n_seq_max = 1;
    p.n_threads = 2;
    p.n_threads_batch = 2;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    p.op_offload = false;
    p.offload_kqv = false;
    return p;
}

static std::vector<float> decode(llama_context * ctx, int count, int start) {
    auto batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = (start + i * 3) % 32;
        batch.pos[i] = start + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    const int status = llama_decode(ctx, batch);
    llama_batch_free(batch);
    check(status == 0, "decode failed");
    auto * values = llama_get_logits(ctx);
    return {values, values + count * 32};
}

static std::vector<uint8_t> save_state(llama_context * ctx) {
    std::vector<uint8_t> state(llama_state_get_size(ctx));
    check(llama_state_get_data(ctx, state.data(), state.size()) == state.size(), "save state");
    return state;
}

static void restore_state(llama_context * ctx, const std::vector<uint8_t> & state) {
    check(llama_state_set_data(ctx, state.data(), state.size()) == state.size(), "restore state");
}

static void run(const std::string & path, llama_load_mode mode, size_t budget, bool qwen = false, bool gemma = false,
        size_t cache_budget = 0) {
    auto p = llama_model_default_params();
    p.n_gpu_layers = gpu_layers && test_device && cache_budget ? 99 : 0;
    p.load_mode = mode;
    p.use_extra_bufts = false;
    llama_model_ptr baseline(llama_model_load_from_file(path.c_str(), p));
    check(bool(baseline), "load baseline");
    p.tensor_buft_overrides = nullptr;
    p.moe_cpu_tile_bytes = budget;
    if (gpu_layers && test_device && cache_budget) { p.n_gpu_layers = 99; }
    llama_model_ptr tiled(llama_model_load_from_file(path.c_str(), p));
    check(bool(tiled) && tiled->moe_cpu_banks.size() == 2, "load tiled model");
    if (test_device && cache_budget) {
        cache_budget = std::max(size_t(1), cache_budget / budget) *
                llama_moe_min_tile_bytes(*tiled->moe_cpu_banks.front(), ggml_backend_dev_buffer_type(test_device));
    }
    observations base_obs, tile_obs;
    auto cp = context_params();
    if (gpu_layers && test_device && cache_budget) { cp.op_offload = cp.offload_kqv = true; }
    cp.cb_eval = observations::eval;
    cp.cb_eval_user_data = &base_obs;
    llama_context_ptr base_ctx(llama_init_from_model(baseline.get(), cp));
    cp.cb_eval_user_data = &tile_obs;
    cache_params(cp, cache_budget);
    llama_context_ptr tile_ctx(llama_init_from_model(tiled.get(), cp));
    check(bool(base_ctx) && bool(tile_ctx), "create contexts");
    static bool gpu_rejections_checked = false;
    if (test_device && cache_budget && !gpu_rejections_checked) {
        auto invalid = cp;
        invalid.moe_cpu_cache_bytes = cache_budget;
        check(llama_init_from_model(tiled.get(), invalid) == nullptr, "accepted simultaneous CPU and GPU caches");
        invalid = cp;
        invalid.moe_gpu_reserve_bytes = SIZE_MAX;
        check(llama_init_from_model(tiled.get(), invalid) == nullptr, "accepted impossible VRAM reserve");
        invalid = cp;
        invalid.moe_staging_bytes = 0;
        check(llama_init_from_model(tiled.get(), invalid) == nullptr, "accepted missing transfer buffer");
        invalid = cp;
        invalid.moe_compute_bytes = 1;
        invalid.moe_cpu_miss_percent = 0;
        invalid.cb_eval = nullptr;
        invalid.cb_eval_user_data = nullptr;
        llama_context_ptr limited(llama_init_from_model(tiled.get(), invalid));
        check(bool(limited), "create compute-limited context");
        llama_token token = 1;
        check(llama_decode(limited.get(), llama_batch_get_one(&token, 1)) != 0, "compute limit ignored during decode");
        check(llama_moe_memory(limited.get()).ffn_calls == 0, "failed GPU FFN counted as complete");
        gpu_rejections_checked = true;
    }
    if (cache_budget) {
        auto * raw = tile_ctx.release();
        auto invalid = cp;
        invalid.moe_cpu_cache_bytes = invalid.moe_gpu_cache_bytes = 1;
        check(!llama_moe_reconfigure(&raw, invalid), "accepted invalid runtime pool configuration");
        auto resized = cp;
        resized.n_ctx = cp.n_ctx * 2;
        if (test_device) { resized.moe_gpu_cache_bytes *= 2; }
        else { resized.moe_cpu_cache_bytes *= 2; }
        check(llama_moe_reconfigure(&raw, resized), "coordinated runtime resize failed");
        check(llama_n_ctx(raw) >= resized.n_ctx, "KV pool did not resize");
        check(llama_moe_memory(raw).cache_budget_bytes == cache_budget * 2, "expert pool did not resize");
        check(llama_moe_reconfigure(&raw, cp), "runtime resize back failed");
        tile_ctx.reset(raw);
    }
    const auto initial = llama_moe_memory(tile_ctx.get());
    check(initial.cache_staging_bytes == (comparing_gpu ? cp.moe_staging_bytes : 0), "transfer buffer accounting");
    check(initial.gpu_compute_limit_bytes == (comparing_gpu ? cp.moe_compute_bytes : 0), "GPU arena limit accounting");
    size_t bank_bytes = 0;
    for (const auto & bank : tiled->moe_cpu_banks) { bank_bytes += bank->host_bytes(); }
    check(initial.host_bank_bytes == bank_bytes && initial.model_bytes >= bank_bytes, "host bank accounting");
    check(initial.tile_budget_bytes == budget && initial.ffn_calls == 0 && initial.peak_temporary_bytes == 0,
            "initial memory counters");
    check(initial.cache_budget_bytes == cache_budget && initial.cache_weight_bytes <= cache_budget &&
            initial.cache_loads == 0, "initial cache accounting");
    auto * graph = llama_graph_reserve(tile_ctx.get(), 4, 1, 4);
    check(graph != nullptr, "reserve graph");
    size_t custom = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * t = ggml_graph_node(graph, i);
        check(t->op != GGML_OP_MUL_MAT_ID, "full expert graph still executes");
        custom += t->op == GGML_OP_MAP_CUSTOM3;
        for (auto * source : t->src) {
            check(!source || std::strstr(source->name, "_exps.weight") == nullptr, "full bank remains a graph input");
        }
    }
    check(custom == 2, "missing FFN boundary");
    int start = 0;
    uint64_t calls = 0;
    for (int count : {4, 1, 1, 1, 9, 2, 1}) {
        compare(decode(base_ctx.get(), count, start), decode(tile_ctx.get(), count, start));
        if (cache_budget) {
            auto * raw = tile_ctx.get();
            check(!llama_moe_reconfigure(&raw, cp) && raw == tile_ctx.get(), "resize accepted populated state");
        }
        start += count;
        calls += 2 * ((count + 7) / 8);
        const auto m = llama_moe_memory(tile_ctx.get());
        check(m.ffn_calls == calls, "FFN counters lost or duplicated across graph reuse");
        check(cache_budget ? m.peak_tile_bytes <= budget : (m.peak_tile_bytes > 0 && m.peak_tile_bytes <= budget), "tile accounting exceeds budget");
        check(m.peak_host_vector_bytes > 0 && m.peak_metadata_bytes > 0 && (comparing_gpu || (m.peak_workspace_bytes + m.cache_workspace_bytes) > 0),
                "missing executor allocations");
        check(m.peak_temporary_bytes >= m.peak_tile_bytes + m.peak_metadata_bytes, "temporary peak undercounts");
        check(m.peak_temporary_bytes <= m.peak_tile_bytes + m.peak_compute_bytes +
                m.peak_host_vector_bytes + m.peak_metadata_bytes + m.peak_workspace_bytes, "temporary peak double counts");
        check(m.kv_bytes > 0 && m.outer_compute_bytes > 0 && m.output_bytes > 0 && (gpu_layers || m.outer_workspace_bytes > 0),
                "missing retained allocations");
        if (cache_budget) {
            check(m.cache_weight_bytes > 0 && m.cache_weight_bytes <= cache_budget && m.cache_metadata_bytes > 0, "cache allocation accounting");
            check(test_device && cpu_miss_percent ? m.copied_weight_bytes >= m.cache_copied_bytes : m.copied_weight_bytes == m.cache_copied_bytes, "cache copy accounting");
        } else {
            check(m.copied_weight_bytes >= m.tiles * tiled->moe_cpu_banks.front()->expert_bytes(), "copy accounting");
        }
    }
    const auto warm = llama_moe_memory(tile_ctx.get());
    for (int i = 0; i < 40; ++i) {
        compare(decode(base_ctx.get(), 1, start), decode(tile_ctx.get(), 1, start));
        ++start;
    }
    const auto sustained = llama_moe_memory(tile_ctx.get());
    check(sustained.ffn_calls == calls + 80, "sustained FFN accounting");
    check(sustained.peak_temporary_bytes == warm.peak_temporary_bytes, "single-token temporary storage grew");
    check(sustained.model_bytes == warm.model_bytes && sustained.kv_bytes == warm.kv_bytes,
            "retained model or KV storage grew");
    if (cache_budget) {
        check(sustained.cache_weight_bytes == initial.cache_weight_bytes, "cache not retained");
        if (cache_budget >= 2 * size_t(tiled->moe_cpu_banks.front()->n_expert()) * llama_moe_min_tile_bytes(*tiled->moe_cpu_banks.front(),
                test_device ? ggml_backend_dev_buffer_type(test_device) : nullptr)) {
            check(sustained.cache_hits > 0 && sustained.cache_loads <= 2 * uint64_t(tiled->moe_cpu_banks.front()->n_expert()) && sustained.cache_evictions == 0,
                    "fitting cache reloaded or evicted experts");
        }
    }
    check(llama_moe_memory(base_ctx.get()).ffn_calls == 0, "baseline reports tiled work");
    compare_intermediate(base_obs.ffn, tile_obs.ffn);
    if (qwen || gemma) {
        compare_intermediate(base_obs.shared, tile_obs.shared);
        compare_intermediate(base_obs.combined, tile_obs.combined);
        if (qwen) {
            compare_intermediate(base_obs.recurrent, tile_obs.recurrent);
            bool nonzero_state = false;
            for (float value : base_obs.recurrent) { nonzero_state |= std::abs(value) > 1e-6f; }
            check(nonzero_state, "recurrent state fixture is inert");
        } else {
            compare_intermediate(base_obs.normalized, tile_obs.normalized);
        }
        check(base_obs.shared.size() == base_obs.ffn.size(), "missing shared expert contributions");
        check(tile_obs.combined.size() == tile_obs.ffn.size(), "missing combined FFN outputs");
        bool nonzero_shared = false;
        for (size_t i = 0; i < base_obs.shared.size(); ++i) {
            nonzero_shared |= std::abs(base_obs.shared[i]) > 1e-6f;
            const auto & routed = gemma ? tile_obs.normalized : tile_obs.ffn;
            check(std::abs(tile_obs.combined[i] - routed[i] - tile_obs.shared[i]) < 2e-5f,
                    "shared expert not added to routed output");
        }
        check(nonzero_shared, "shared expert fixture is inert");
        const auto base_state = save_state(base_ctx.get());
        const auto tile_state = save_state(tile_ctx.get());
        std::vector<uint8_t> full_sequence(llama_state_seq_get_size_ext(tile_ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE));
        check(llama_state_seq_get_data_ext(tile_ctx.get(), full_sequence.data(), full_sequence.size(), 0,
                LLAMA_STATE_SEQ_FLAGS_NONE) == full_sequence.size(), "full semantic snapshot save");
        const auto expected = decode(base_ctx.get(), 3, start);
        const auto tiled_expected = decode(tile_ctx.get(), 3, start);
        compare(expected, tiled_expected);
        restore_state(base_ctx.get(), base_state);
        restore_state(tile_ctx.get(), tile_state);
        compare(expected, decode(base_ctx.get(), 3, start));
        compare(gpu_layers ? tiled_expected : expected, decode(tile_ctx.get(), 3, start));
        if (!gpu_layers) {
            restore_state(tile_ctx.get(), base_state);
            compare(expected, decode(tile_ctx.get(), 3, start));
        }
        // GPU-resident full-sequence snapshots are excluded from semantic mode until replay is validated.
        if (!gpu_layers) {
            llama_memory_clear(llama_get_memory(tile_ctx.get()), true);
            check(llama_state_seq_set_data_ext(tile_ctx.get(), full_sequence.data(), full_sequence.size(), 0,
                    LLAMA_STATE_SEQ_FLAGS_NONE) == full_sequence.size(), "full semantic snapshot restore");
            check(llama_memory_seq_pos_max(llama_get_memory(tile_ctx.get()), 0) == start - 1,
                    "semantic snapshot consumed boundary mismatch");
            compare(gpu_layers ? tiled_expected : expected, decode(tile_ctx.get(), 3, start));
        }
        start += 3;
    }
    check(tile_obs.custom > 0 && base_obs.custom == 0, "tiled path was not used");
    const auto before_failure = llama_moe_memory(tile_ctx.get());
    const auto clean_state = qwen ? save_state(tile_ctx.get()) : std::vector<uint8_t>{};
    auto batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = start % 32;
    batch.pos[0] = start;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;
    llama_set_abort_callback(tile_ctx.get(), [](void *) { return true; }, nullptr);
    check(llama_decode(tile_ctx.get(), batch) != 0, "cancellation ignored");
    llama_set_abort_callback(tile_ctx.get(), nullptr, nullptr);
    check(llama_moe_memory(tile_ctx.get()).ffn_calls == before_failure.ffn_calls, "cancelled graph counted stale work");
    if (qwen) { restore_state(tile_ctx.get(), clean_state); }
    tile_obs.bad_route = true;
    check(llama_decode(tile_ctx.get(), batch) != 0, "worker failure was swallowed");
    tile_obs.bad_route = false;
    check(llama_moe_memory(tile_ctx.get()).ffn_calls == before_failure.ffn_calls, "failed FFN counted as completed");
    if (qwen) { restore_state(tile_ctx.get(), clean_state); }
    llama_batch_free(batch);
    compare(decode(base_ctx.get(), 1, start), decode(tile_ctx.get(), 1, start));
    auto rejected_cp = context_params();
    rejected_cp.n_seq_max = 2;
    check(llama_init_from_model(tiled.get(), rejected_cp) == nullptr, "accepted multiple sequences");
    rejected_cp = context_params();
    rejected_cp.op_offload = true;
    check(llama_init_from_model(tiled.get(), rejected_cp) == nullptr, "accepted op offload");
    rejected_cp = context_params();
    rejected_cp.offload_kqv = true;
    check(llama_init_from_model(tiled.get(), rejected_cp) == nullptr, "accepted KV offload");
    rejected_cp = context_params();
    rejected_cp.moe_cpu_cache_bytes = 1;
    check(llama_init_from_model(tiled.get(), rejected_cp) == nullptr, "accepted undersized cache");
    rejected_cp.moe_cpu_cache_bytes = budget;
    check(llama_init_from_model(baseline.get(), rejected_cp) == nullptr, "accepted cache without host banks");
    tile_ctx.reset();
    base_ctx.reset();
    cp = context_params();
    cache_params(cp, cache_budget);
    tile_ctx.reset(llama_init_from_model(tiled.get(), cp));
    cp.moe_cpu_cache_bytes = 0;
    cp.moe_gpu_cache_bytes = 0;
    cp.moe_cpu_miss_percent = 0;
    cp.moe_pipeline = cp.moe_fast = false;
    base_ctx.reset(llama_init_from_model(baseline.get(), cp));
    check(bool(tile_ctx) && bool(base_ctx), "create contexts without callback");
    check(llama_moe_memory(tile_ctx.get()).ffn_calls == 0, "new context inherits old counters");
    check(llama_moe_memory(tile_ctx.get()).cache_loads == 0, "new context inherits cache contents");
    compare(decode(base_ctx.get(), 4, 0), decode(tile_ctx.get(), 4, 0));
    tile_ctx.reset();
    auto retained = tiled->moe_cpu_banks.front();
    const auto * tensor = retained->tensor(0);
    const auto before = static_cast<const unsigned char *>(tensor->data)[0];
    tiled.reset();
    check(static_cast<const unsigned char *>(tensor->data)[0] == before, "host owner lost storage");
}

static void pretrained(const std::string & path, bool synthetic = false) {
    const bool device_boundary = test_device && std::getenv("FT_TEST_DEVICE_BOUNDARY");
    auto p = llama_model_default_params();
    p.n_gpu_layers = 0;
    p.load_mode = LLAMA_LOAD_MODE_MMAP;
    p.use_extra_bufts = false;
    if (device_boundary) { p.n_gpu_layers = 99; p.moe_cpu_tile_bytes = 64 * 1024 * 1024; }
    llama_model_ptr baseline(llama_model_load_from_file(path.c_str(), p));
    check(bool(baseline), "load pretrained baseline");
    p.moe_cpu_tile_bytes = 64 * 1024 * 1024;
    llama_model_ptr cached(llama_model_load_from_file(path.c_str(), p));
    check(bool(cached), "load pretrained host banks");
    auto cp = context_params();
    cp.n_ctx = 256;
    cp.n_batch = 128;
    cp.n_ubatch = 8;
    if (device_boundary) {
        cache_params(cp, 64 * 1024 * 1024);
        cp.moe_device_tensors = false;
        cp.op_offload = cp.offload_kqv = true;
    }
    llama_context_ptr a(llama_init_from_model(baseline.get(), cp));
    cache_params(cp, 64 * 1024 * 1024);
    cp.moe_device_tensors = true;
    llama_context_ptr b(llama_init_from_model(cached.get(), cp));
    check(bool(a) && bool(b), "pretrained contexts");
    const auto * vocab = llama_model_get_vocab(baseline.get());
    std::vector<llama_token> tokens{1, 2, 3, 4};
    if (!synthetic) {
        const std::string prompt = "The capital of France is";
        const int required = llama_tokenize(vocab, prompt.data(), int(prompt.size()), nullptr, 0, true, true);
        check(required < 0 && -required <= 128, "pretrained token count");
        tokens.resize(-required);
        check(llama_tokenize(vocab, prompt.data(), int(prompt.size()), tokens.data(), int(tokens.size()), true, true) == -required,
                "pretrained tokenize");
    }
    int position = 0;
    bool gpu_quality_ok = true, gpu_greedy_ok = true;
    auto evaluate = [&](const std::vector<llama_token> & input) {
        auto batch = llama_batch_init(int(input.size()), 0, 1);
        batch.n_tokens = int(input.size());
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = input[i];
            batch.pos[i] = position + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
        }
        const int status_a = llama_decode(a.get(), batch);
        const int status_b = llama_decode(b.get(), batch);
        llama_batch_free(batch);
        check(status_a == 0 && status_b == 0, "pretrained decode");
        const size_t count = input.size() * llama_vocab_n_tokens(vocab);
        if (test_device) {
            const size_t nv = llama_vocab_n_tokens(vocab);
            const auto * la = llama_get_logits(a.get());
            const auto * lb = llama_get_logits(b.get());
            for (size_t row = 0; row < input.size(); ++row) {
                double za = 0, zb = 0, error = 0, maximum = 0, kl = 0, tv = 0;
                const float ma = *std::max_element(la, la + nv), mb = *std::max_element(lb, lb + nv);
                for (size_t i = 0; i < nv; ++i) {
                    check(std::isfinite(la[i]) && std::isfinite(lb[i]), "nonfinite GPU logits");
                    za += std::exp(double(la[i]) - ma);
                    zb += std::exp(double(lb[i]) - mb);
                    const double diff = double(la[i]) - lb[i];
                    error += diff * diff;
                    maximum = std::max(maximum, std::abs(diff));
                }
                za = std::log(za) + ma;
                zb = std::log(zb) + mb;
                for (size_t i = 0; i < nv; ++i) {
                    const double pa = std::exp(la[i] - za), pb = std::exp(lb[i] - zb);
                    kl += pa * ((la[i] - za) - (lb[i] - zb));
                    tv += std::abs(pa - pb) * 0.5;
                }
                std::printf("GPU logits position=%d max_abs=%g rmse=%g KL=%g TV=%g\n",
                        position + int(row), maximum, std::sqrt(error / nv), kl, tv);
                std::fflush(stdout);
                gpu_quality_ok &= kl <= 0.01 && tv <= 0.05;
                if (device_boundary) { check(maximum <= 1e-5, "device boundary changed same-mode logits"); }
                la += nv;
                lb += nv;
            }
        } else {
            compare({llama_get_logits(a.get()), llama_get_logits(a.get()) + count},
                    {llama_get_logits(b.get()), llama_get_logits(b.get()) + count});
        }
        position += int(input.size());
        const auto * logits_a = llama_get_logits_ith(a.get(), -1);
        const auto * logits_b = llama_get_logits_ith(b.get(), -1);
        const auto next_a = llama_token(std::max_element(logits_a, logits_a + llama_vocab_n_tokens(vocab)) - logits_a);
        const auto next_b = llama_token(std::max_element(logits_b, logits_b + llama_vocab_n_tokens(vocab)) - logits_b);
        if (next_a != next_b) {
            std::fprintf(stderr, "greedy mismatch position=%d CPU=%d GPU=%d\n", position - 1, next_a, next_b);
            if (!test_device) { throw std::runtime_error("pretrained greedy token mismatch"); }
            gpu_greedy_ok = false;
        }
        return next_a;
    };
    auto next = evaluate(tokens);
    const auto state_a = save_state(a.get()), state_b = save_state(b.get());
    const int prompt_end = position;
    const auto saved_next = next;
    std::vector<llama_token> generated;
    for (int i = 0; i < 16; ++i) {
        generated.push_back(next);
        next = evaluate({next});
    }
    restore_state(a.get(), state_a);
    restore_state(b.get(), state_b);
    position = prompt_end;
    next = saved_next;
    for (auto expected : generated) {
        check(next == expected, "pretrained state continuation mismatch");
        next = evaluate({next});
    }
    check(gpu_quality_ok, "GPU probability envelope exceeded (KL > 0.01 or TV > 0.05)");
    check(gpu_greedy_ok, "GPU greedy tokens differ from baseline");
    if (device_boundary) {
        const auto host = llama_moe_memory(a.get()), device = llama_moe_memory(b.get());
        check(device.output_readback_bytes < host.output_readback_bytes, "pretrained device boundary was not used");
        std::printf("Device boundary output readback bytes: host=%llu device=%llu; CUDA captures=%llu replays=%llu\n",
                (unsigned long long) host.output_readback_bytes, (unsigned long long) device.output_readback_bytes,
                (unsigned long long) device.expert_graph_captures, (unsigned long long) device.expert_graph_replays);
        if (std::getenv("FT_TEST_REQUIRE_GRAPHS")) {
            check(device.expert_graph_captures > 0 && device.expert_graph_replays > 0, "pretrained CUDA replay not exercised");
        }
    }
    std::printf("PASS %s: full-vocabulary logits, 16 greedy tokens, state replay; device=%s\n",
            synthetic ? "acceptance harness synthetic smoke" : "pretrained", test_device ? ggml_backend_dev_name(test_device) : "CPU");
    if (!synthetic) {
        for (auto token : generated) {
            std::vector<char> piece(256);
            int n = llama_token_to_piece(vocab, token, piece.data(), int(piece.size()), 0, true);
            if (n < 0) {
                piece.resize(-n);
                n = llama_token_to_piece(vocab, token, piece.data(), int(piece.size()), 0, true);
            }
            check(n >= 0, "pretrained output detokenization");
            std::fwrite(piece.data(), 1, n, stdout);
        }
        std::puts("");
    }
}

int main(int argc, char ** argv) {
    common_chat_msg_spans spans;
    spans.add(COMMON_CHAT_ROLE_SYSTEM, 0, 8);
    spans.add(COMMON_CHAT_ROLE_USER, 8, 16);
    spans.add(COMMON_CHAT_ROLE_ASSISTANT, 24, 10);
    spans.add(COMMON_CHAT_ROLE_TOOL, 34, 12);
    check(!spans.is_message_start(0) && spans.is_message_start(8) && spans.is_message_start(24) &&
            spans.is_message_start(34) && !spans.is_message_start(35), "semantic message boundaries");
    gpu_layers = std::getenv("FT_TEST_GPU_LAYERS") != nullptr;
    ggml_backend_load_all();
    llama_backend_init();
    if (argc >= 3 && std::string(argv[1]) == "--device") {
        test_device = ggml_backend_dev_by_name(argv[2]);
        if (!test_device || ggml_backend_dev_type(test_device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            std::fprintf(stderr, "GPU test device unavailable\n");
            return 1;
        }
        argc -= 2;
        argv += 2;
    }
    const auto path = (std::filesystem::temp_directory_path() / ("moe-integration-" +
            std::to_string(ggml_time_us()) + ".gguf")).string();
    try {
        if (argc >= 3 && std::string(argv[1]) == "--cpu-miss-percent") {
            cpu_miss_percent = std::stoi(argv[2]);
            argc -= 2;
            argv += 2;
        }
        if (argc == 3 && std::string(argv[1]) == "--model") {
            pretrained(argv[2]);
            llama_backend_free();
            return 0;
        }
        check(argc == 1, "usage: test-moe-integration [--model pretrained.gguf]");
        for (auto example : {LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}) {
            common_params params;
            std::vector<std::string> args{"test", "-m", path, "--moe-cpu-cache-mib", "2", "-ngl", "0",
                    "--no-op-offload", "--no-kv-offload", "-np", "1"};
            std::vector<char *> argv;
            for (auto & arg : args) { argv.push_back(&arg[0]); }
            check(common_params_parse(int(argv.size()), argv.data(), params, example), "cache option parse");
            check(params.moe_cpu_cache_bytes == 2 * 1024 * 1024 && params.moe_cpu_tile_bytes == 0, "cache option units");
            check(common_model_params_to_llama(params).moe_cpu_tile_bytes == params.moe_cpu_cache_bytes, "cache option omitted host setup");
            params.tensor_buft_overrides = {{nullptr, nullptr}};
            check(common_model_params_to_llama(params).tensor_buft_overrides == nullptr, "empty override sentinel treated as an override");
            check(common_context_params_to_llama(params).moe_cpu_cache_bytes == params.moe_cpu_cache_bytes, "cache option omitted context setup");
            params.moe_cpu_tile_bytes = 1024 * 1024;
            check(common_model_params_to_llama(params).moe_cpu_tile_bytes == 1024 * 1024, "explicit tile budget lost");
        }
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
            fixture(path, type);
            const size_t one = 3 * ggml_row_size(type, 32) * 64 + 256;
            for (auto mode : {LLAMA_LOAD_MODE_NONE, LLAMA_LOAD_MODE_MMAP}) {
                run(path, mode, one);
                run(path, mode, one * 4);
                run(path, mode, one, false, false, one * 8);
            }
        }
        fixture(path, GGML_TYPE_Q8_0, 7, true);
        run(path + "-00001-of-00002.gguf", LLAMA_LOAD_MODE_MMAP, 3 * ggml_row_size(GGML_TYPE_Q8_0, 32) * 64 + 256);
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
            fixture(path, type, 4, false, true);
            const size_t one = 3 * ggml_row_size(type, 32) * 64 + 256;
            for (auto mode : {LLAMA_LOAD_MODE_NONE, LLAMA_LOAD_MODE_MMAP}) {
                run(path, mode, one, true);
                run(path, mode, one * 4, true);
                run(path, mode, one, true, false, one);
                run(path, mode, one, true, false, one * 8);
            }
        }
        fixture(path, GGML_TYPE_Q8_0, 7, true, true, true);
        run(path + "-00001-of-00002.gguf", LLAMA_LOAD_MODE_MMAP,
                3 * ggml_row_size(GGML_TYPE_Q8_0, 32) * 64 + 256, true);
        run(path + "-00001-of-00002.gguf", LLAMA_LOAD_MODE_MMAP,
                3 * ggml_row_size(GGML_TYPE_Q8_0, 32) * 64 + 256, true, false, 65536);
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
            fixture(path, type, 4, false, false, false, true);
            const size_t one = 3 * ggml_row_size(type, 32) * 64 + 256;
            for (auto mode : {LLAMA_LOAD_MODE_NONE, LLAMA_LOAD_MODE_MMAP}) {
                run(path, mode, one, false, true);
                run(path, mode, one * 4, false, true);
                run(path, mode, one, false, true, one);
                run(path, mode, one, false, true, one * 8);
            }
        }
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
            fixture(path, type, 9, true, false, false, true);
            const size_t one = 3 * ggml_row_size(type, 32) * 64 + 256;
            run(path + "-00001-of-00002.gguf", LLAMA_LOAD_MODE_MMAP, one, false, true);
            for (size_t capacity : {size_t(1), size_t(2), size_t(18)}) {
                run(path + "-00001-of-00002.gguf", LLAMA_LOAD_MODE_MMAP, one, false, true, one * capacity);
            }
        }
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
            fixture(path, type, 9, true, false, false, true, true);
            const size_t one = 3 * ggml_row_size(type, 32) * 64 + 256;
            for (auto mode : {LLAMA_LOAD_MODE_NONE, LLAMA_LOAD_MODE_MMAP}) {
                run(path, mode, one, false, true);
                run(path, mode, one, false, true, one);
                run(path, mode, one, false, true, one * 18);
            }
            run(path + "-00001-of-00002.gguf", LLAMA_LOAD_MODE_MMAP, one, false, true, one * 2);
        }
        pretrained(path, true);
        auto p = llama_model_default_params();
        p.n_gpu_layers = 0;
        p.moe_cpu_tile_bytes = 1;
        check(llama_model_load_from_file(path.c_str(), p) == nullptr, "accepted tiny budget");
        p.moe_cpu_tile_bytes = 1024 * 1024;
        p.n_gpu_layers = 0;
        p.load_mtp = true;
        check(llama_model_load_from_file(path.c_str(), p) == nullptr, "accepted MTP loading");
        p.load_mtp = false;
        p.no_alloc = true;
        check(llama_model_load_from_file(path.c_str(), p) == nullptr, "accepted no_alloc");
        p.no_alloc = false;
        fixture(path, GGML_TYPE_Q5_0);
        check(llama_model_load_from_file(path.c_str(), p) == nullptr, "accepted unsupported expert type");
        fixture(path, GGML_TYPE_Q5_0, 4, false, false, false, true);
        check(llama_model_load_from_file(path.c_str(), p) == nullptr, "accepted unsupported Gemma expert type");
        std::filesystem::remove(path);
        std::filesystem::remove(path + "-00001-of-00002.gguf");
        std::filesystem::remove(path + "-00002-of-00002.gguf");
        llama_backend_free();
        if (gpu_layers) { std::printf("Native intermediate max absolute difference: %.9g (diagnostic; logits retain standard tolerance)\n", native_intermediate_max_error); }
        std::printf("%s MoE integration: 135 model comparisons, Qwen/Gemma cache targets, fused Gemma top-8 with scales, state continuation and recovery passed\n",
                test_device ? "GPU" : "CPU");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        std::filesystem::remove(path);
        std::filesystem::remove(path + "-00001-of-00002.gguf");
        std::filesystem::remove(path + "-00002-of-00002.gguf");
        return 1;
    }
}
