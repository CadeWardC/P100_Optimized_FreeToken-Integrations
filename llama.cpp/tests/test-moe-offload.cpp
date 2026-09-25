#include "llama-moe-offload.h"
#include "ggml-cpu.h"
#include "ggml-moe-cache.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstring>

static void check(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

template<class F> static void rejects(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "invalid configuration was accepted");
}

struct fixture {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensors[3];
    ggml_tensor * scales = nullptr;

    fixture(ggml_type type, int64_t experts, bool mixed = false, int64_t embd = 32, int64_t ff = 32,
            bool scaled = false, bool fused = false, bool random_weights = false, ggml_type down_type = GGML_TYPE_COUNT) {
        ctx = ggml_init({ 8 * ggml_tensor_overhead(), nullptr, true });
        check(ctx != nullptr, "fixture context");
        for (size_t p = 0; p < 3; ++p) {
            tensors[p] = ggml_new_tensor_3d(ctx, p == 2 && down_type != GGML_TYPE_COUNT ? down_type : mixed && p == 2 ? GGML_TYPE_Q8_0 : type,
                                          p == 2 ? ff : embd, p == 2 ? embd : ff * (fused && p == 0 ? 2 : 1), experts);
        }
        if (scaled) { scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, experts); }
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        check(buffer != nullptr, "fixture allocation");
        if (scales) {
            std::vector<float> values(experts);
            for (int64_t i = 0; i < experts; ++i) { values[i] = float(i % 7 - 2) * 0.5f; }
            ggml_backend_tensor_set(scales, values.data(), 0, ggml_nbytes(scales));
        }
        for (size_t p = 0; p < 3; ++p) {
            auto * t = tensors[p];
            std::vector<float> values(size_t(ggml_nelements(t)));
            uint32_t seed = 1729 + uint32_t(p);
            for (size_t i = 0; i < values.size(); ++i) {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                values[i] = random_weights ? 0.15f * (float(seed % 65536) / 32768.0f - 1.0f) :
                        0.15f * std::sin(float(i) * 0.17f + float(p));
            }
            std::vector<uint8_t> bytes(ggml_nbytes(t));
            const std::vector<float> importance(size_t(t->ne[0]), 1.0f);
            const size_t written = ggml_quantize_chunk(t->type, values.data(), bytes.data(), 0, t->ne[1] * experts, t->ne[0], importance.data());
            check(written == bytes.size(), "quantized size");
            ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        }
    }

    ~fixture() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};

static llama_moe_reference_result ordinary(const llama_moe_host_bank & bank,
        const std::vector<float> & input, size_t tokens, size_t k,
        const std::vector<int32_t> & ids, const std::vector<float> & weights,
        bool gelu = false, ggml_backend_dev_t device = nullptr) {
    auto * backend = device ? ggml_backend_dev_init(device, nullptr) : ggml_backend_cpu_init();
    check(backend != nullptr, "ordinary backend");
    if (!device) { ggml_backend_cpu_set_n_threads(backend, 2); }
    auto * ctx = ggml_init({ 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true });
    check(ctx != nullptr, "ordinary context");
    auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, bank.n_embd(), 1, int64_t(tokens));
    auto * route = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, int64_t(k), int64_t(tokens));
    auto * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, int64_t(k), int64_t(tokens));
    ggml_set_input(x);
    ggml_set_input(route);
    ggml_set_input(w);
    ggml_tensor * projections[3];
    auto * weight_ctx = ggml_init({3 * ggml_tensor_overhead(), nullptr, true});
    ggml_backend_buffer_t weight_buffer = nullptr;
    for (size_t p = 0; p < 3; ++p) {
        projections[p] = device ? ggml_dup_tensor(weight_ctx, bank.tensor(p)) : const_cast<ggml_tensor *>(bank.tensor(p));
    }
    if (device) {
        weight_buffer = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);
        check(weight_buffer != nullptr, "ordinary GPU weights");
        for (size_t p = 0; p < 3; ++p) {
            ggml_backend_tensor_set(projections[p], bank.tensor(p)->data, 0, ggml_nbytes(projections[p]));
        }
    }
    auto * gate = ggml_mul_mat_id(ctx, projections[0], x, route);
    auto * up = ggml_mul_mat_id(ctx, projections[1], x, route);
    if (device) { ggml_set_output(gate); ggml_set_output(up); }
    auto * act = gelu ? ggml_geglu_split(ctx, gate, up) : ggml_swiglu_split(ctx, gate, up);
    auto * experts = ggml_mul_mat_id(ctx, projections[2], act, route);
    auto * scales = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, bank.n_expert());
    ggml_set_input(scales);
    experts = ggml_mul(ctx, experts, ggml_get_rows(ctx,
            ggml_repeat_4d(ctx, scales, 1, bank.n_expert(), int64_t(tokens), 1), route));
    ggml_set_output(experts);
    auto * weighted = ggml_mul(ctx, experts, w);
    auto * output = ggml_view_2d(ctx, weighted, bank.n_embd(), int64_t(tokens), weighted->nb[2], 0);
    for (size_t i = 1; i < k; ++i) {
        auto * next = ggml_view_2d(ctx, weighted, bank.n_embd(), int64_t(tokens), weighted->nb[2], i * weighted->nb[1]);
        output = ggml_add(ctx, output, next);
    }
    output = ggml_cont(ctx, output);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    auto * fallback = device ? ggml_backend_cpu_init() : nullptr;
    ggml_backend_t backends[] = {backend, fallback};
    auto * sched = ggml_backend_sched_new(backends, nullptr, device ? 2 : 1, 64, false, false);
    if (device) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            check(ggml_backend_supports_op(backend, node), "ordinary GPU operation unsupported");
            ggml_backend_sched_set_tensor_backend(sched, node, backend);
        }
    }
    check(sched && ggml_backend_sched_alloc_graph(sched, graph), "ordinary scheduler allocation");
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(route, ids.data(), 0, ggml_nbytes(route));
    ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
    std::vector<float> scale_values(bank.n_expert());
    for (int32_t e = 0; e < bank.n_expert(); ++e) { scale_values[e] = bank.output_scale(e); }
    ggml_backend_tensor_set(scales, scale_values.data(), 0, ggml_nbytes(scales));
    check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "ordinary compute");
    llama_moe_reference_result result;
    result.output.resize(input.size());
    result.expert_outputs.resize(input.size() * k);
    ggml_backend_tensor_get(output, result.output.data(), 0, ggml_nbytes(output));
    ggml_backend_tensor_get(experts, result.expert_outputs.data(), 0, ggml_nbytes(experts));
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(weight_ctx);
    ggml_free(ctx);
    ggml_backend_free(backend);
    if (fallback) { ggml_backend_free(fallback); }
    return result;
}

static void close(const std::vector<float> & actual, const std::vector<float> & expected) {
    check(actual.size() == expected.size(), "output size");
    for (size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 2e-5f + 2e-5f * std::abs(expected[i]);
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > tolerance) {
            std::fprintf(stderr, "mismatch at %zu: %.9g vs %.9g\n", i, actual[i], expected[i]);
            throw std::runtime_error("numerical disagreement");
        }
    }
}

static void scalar_f32(const llama_moe_host_bank & bank, const std::vector<float> & input,
        size_t k, const std::vector<int32_t> & ids, const std::vector<float> & actual, bool gelu = false) {
    std::vector<float> expected(actual.size());
    for (size_t a = 0; a < ids.size(); ++a) {
        float activation[32];
        const float * gate = static_cast<const float *>(bank.tensor(0)->data) + ids[a] * 1024;
        const float * up = static_cast<const float *>(bank.tensor(1)->data) + ids[a] * 1024;
        const float * down = static_cast<const float *>(bank.tensor(2)->data) + ids[a] * 1024;
        for (size_t f = 0; f < 32; ++f) {
            double g = 0, u = 0;
            for (size_t d = 0; d < 32; ++d) {
                g += double(gate[f * 32 + d]) * input[(a / k) * 32 + d];
                u += double(up[f * 32 + d]) * input[(a / k) * 32 + d];
            }
            // The CPU GELU table rounds its input and output through F16.
            if (gelu) { g = ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(g))); }
            double gate_act = gelu ? 0.5 * g * (1.0 + std::tanh(std::sqrt(2.0 / 3.141592653589793) *
                    (g + 0.044715 * g * g * g))) : g / (1 + std::exp(-g));
            if (gelu) { gate_act = ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(gate_act))); }
            activation[f] = float(gate_act * u);
        }
        for (size_t d = 0; d < 32; ++d) {
            double value = 0;
            for (size_t f = 0; f < 32; ++f) {
                value += double(down[d * 32 + f]) * activation[f];
            }
            expected[a * 32 + d] = float(value) * bank.output_scale(ids[a]);
        }
    }
    close(actual, expected);
}

static void test_routes() {
    const std::vector<int32_t> ids{4, 0, 4, 2, 0, 6, 2, 1, 6};
    for (size_t capacity : {size_t(1), size_t(2), size_t(7)}) {
        auto tiles = llama_moe_plan_routes(ids, 7, capacity);
        std::vector<int> seen(ids.size(), 0), expert_seen(7, 0);
        for (const auto & tile : tiles) {
            check(tile.experts.size() <= capacity, "capacity exceeded");
            for (auto id : tile.experts) { ++expert_seen[id]; }
            for (size_t a = 0; a < tile.assignments.size(); ++a) {
                const size_t original = tile.assignments[a];
                ++seen[original];
                check(tile.slots[a] >= 0 && size_t(tile.slots[a]) < tile.experts.size(), "invalid compact ID");
                check(tile.experts[tile.slots[a]] == ids[original], "remap changed expert identity");
            }
        }
        for (auto count : seen) { check(count == 1, "lost or duplicate assignment"); }
        for (auto count : expert_seen) { check(count <= 1, "duplicate expert copy"); }
    }
    check(llama_moe_plan_routes({}, 7, 1).empty(), "empty routes");
    rejects([] { llama_moe_plan_routes({-1}, 7, 1); });
    rejects([] { llama_moe_plan_routes({7}, 7, 1); });
    rejects([] { llama_moe_plan_routes({0}, 7, 0); });
}

static void test_invalid() {
    auto source = std::make_shared<fixture>(GGML_TYPE_F32, 7);
    auto * g = source->tensors[0];
    auto * u = source->tensors[1];
    auto * d = source->tensors[2];
    rejects([&] { llama_moe_host_bank bank({}, g, u, d); });
    rejects([&] { llama_moe_host_bank bank(source, nullptr, u, d); });
    rejects([&] { llama_moe_host_bank bank(source, g, g, d); });
    g->extra = source.get();
    rejects([&] { llama_moe_host_bank bank(source, g, u, d); });
    g->extra = nullptr;
    g->ne[2] = 6;
    rejects([&] { llama_moe_host_bank bank(source, g, u, d); });
    g->ne[2] = 7;
    g->nb[1] += 4;
    rejects([&] { llama_moe_host_bank bank(source, g, u, d); });
    g->nb[1] -= 4;
    void * data = g->data;
    g->data = static_cast<uint8_t *>(ggml_backend_buffer_get_base(source->buffer)) + ggml_backend_buffer_get_size(source->buffer);
    rejects([&] { llama_moe_host_bank bank(source, g, u, d); });
    g->data = data;
    auto unsupported = std::make_shared<fixture>(GGML_TYPE_BF16, 7);
    rejects([&] { llama_moe_host_bank bank(unsupported, unsupported->tensors[0], unsupported->tensors[1], unsupported->tensors[2]); });
    llama_moe_host_bank bank(source, g, u, d);
    const std::vector<float> input(32, 1.0f);
    rejects([&] { llama_moe_reference_execute(bank, input, 1, 1, {0}, {1}, 0); });
    rejects([&] { llama_moe_reference_execute(bank, input, 1, 1, {0}, {1}, bank.expert_bytes() - 1); });
    rejects([&] { llama_moe_reference_execute(bank, input, 1, 1, {7}, {1}, bank.host_bytes()); });
    rejects([&] { llama_moe_reference_execute(bank, {}, 1, 1, {0}, {1}, bank.host_bytes()); });
    rejects([&] { llama_moe_reference_execute(bank, input, 1, 1, {0}, {std::numeric_limits<float>::quiet_NaN()}, bank.host_bytes()); });
    rejects([&] { llama_moe_reference_execute(bank, input, 1, 1, {0}, {1}, bank.host_bytes(), 0); });
    rejects([&] { llama_moe_reference_execute(bank, input, 1, 1, {0}, {1}, bank.host_bytes(), 1,
            static_cast<llama_moe_activation>(-1)); });
    rejects([&] { bank.output_scale(-1); });
    rejects([&] { bank.output_scale(7); });
    auto scaled = std::make_shared<fixture>(GGML_TYPE_F32, 7, false, 32, 32, true);
    rejects([&] { llama_moe_host_bank invalid(source, g, u, d, g); });
    const float nan = std::numeric_limits<float>::quiet_NaN();
    ggml_backend_tensor_set(scaled->scales, &nan, 0, sizeof(nan));
    rejects([&] { llama_moe_host_bank invalid(scaled, scaled->tensors[0], scaled->tensors[1],
            scaled->tensors[2], scaled->scales); });
    bool overflow = false;
    try { llama_moe_reference_execute(bank, {}, std::numeric_limits<size_t>::max(), 2, {}, {}, bank.host_bytes()); }
    catch (const std::overflow_error &) { overflow = true; }
    check(overflow, "size overflow accepted");
}

static size_t test_execution(ggml_type type, bool mixed, int64_t n_expert, int64_t embd = 32, int64_t ff = 32,
        bool gelu_scaled = false) {
    const auto activation = gelu_scaled ? llama_moe_activation::gelu : llama_moe_activation::silu;
    auto source = std::make_shared<fixture>(type, n_expert, mixed, embd, ff, gelu_scaled);
    std::weak_ptr<fixture> lifetime = source;
    size_t peak = 0;
    {
        llama_moe_host_bank bank(source, source->tensors[0], source->tensors[1], source->tensors[2], source->scales);
        source.reset();
        check(!lifetime.expired(), "bank did not retain owner");
        check(bank.host_bytes() == (bank.expert_bytes() + (gelu_scaled ? sizeof(float) : 0)) * size_t(n_expert), "host accounting");
        for (size_t tokens : {size_t(1), size_t(5)}) {
            for (size_t k : {size_t(1), size_t(3)}) {
                std::vector<float> input(tokens * size_t(embd)), weights(tokens * k);
                std::vector<int32_t> ids(tokens * k);
                for (size_t i = 0; i < input.size(); ++i) { input[i] = std::cos(float(i) * 0.3f); }
                for (size_t i = 0; i < ids.size(); ++i) {
                    ids[i] = int32_t((i * 3 + i / 4) % 7);
                    weights[i] = i % 5 == 4 ? 0.0f : float(1 + i % 3) * 0.25f;
                }
                const auto expected = ordinary(bank, input, tokens, k, ids, weights, gelu_scaled);
                for (size_t capacity : {size_t(1), size_t(2), size_t(7)}) {
                    size_t budget = 0;
                    const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
                    for (size_t p = 0; p < 3; ++p) {
                        const size_t bytes = bank.tensor(p)->nb[2] * capacity;
                        budget += ((bytes + alignment - 1) / alignment) * alignment;
                    }
                    const auto actual = llama_moe_reference_execute(bank, input, tokens, k, ids, weights, budget, 2, activation);
                    close(actual.expert_outputs, expected.expert_outputs);
                    close(actual.output, expected.output);
                    llama_moe_cached_executor cached({std::make_shared<llama_moe_host_bank>(bank)}, budget);
                    const auto cached_result = cached.execute(0, input, tokens, k, ids, weights, 2, activation);
                    close(cached_result.expert_outputs, expected.expert_outputs);
                    close(cached_result.output, expected.output);
                    const auto again = cached.execute(0, input, tokens, k, ids, weights, 1, activation);
                    close(again.output, expected.output);
                    check(cached.metrics().weights <= budget && cached_result.peak_tile_weight_bytes == 0, "cached weight accounting");
                    if (type == GGML_TYPE_F32 && !mixed && embd == 32 && ff == 32) {
                        scalar_f32(bank, input, k, ids, actual.expert_outputs, gelu_scaled);
                    }
                    auto unique = ids;
                    std::sort(unique.begin(), unique.end());
                    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
                    check(actual.copied_weight_bytes == unique.size() * bank.expert_bytes(), "copied bytes");
                    check(actual.peak_tile_weight_bytes <= budget, "weight allocation is not bounded");
                    check(actual.tiles == (unique.size() + capacity - 1) / capacity, "tile count");
                    if (capacity == 1) { peak = std::max(peak, actual.peak_tile_weight_bytes); }
                }
            }
        }
        const std::vector<float> repeated_input(3 * size_t(embd), 0.2f);
        const std::vector<int32_t> repeated_ids(9, 6);
        const std::vector<float> repeated_weights(9, 1.0f / 3.0f);
        const auto repeated = llama_moe_reference_execute(bank, repeated_input, 3, 3, repeated_ids, repeated_weights, peak, 1, activation);
        const auto expected = ordinary(bank, repeated_input, 3, 3, repeated_ids, repeated_weights, gelu_scaled);
        close(repeated.output, expected.output);
        close(repeated.expert_outputs, expected.expert_outputs);
        check(repeated.tiles == 1 && repeated.copied_weight_bytes == bank.expert_bytes(), "repeated routes copied more than one expert");
    }
    check(lifetime.expired(), "bank leaked owner");
    return peak;
}

struct delayed_backend {
    struct copy { ggml_tensor * tensor; const void * data; size_t offset, size; };
    ggml_backend backend{};
    ggml_backend_device device{};
    ggml_backend_buffer_type buft{};
    std::vector<copy> pending;
    size_t records = 0, waits = 0, syncs = 0, events = 0;
    bool fail_allocation = false;
    size_t submissions = 0, fail_submission = SIZE_MAX, largest_copy = 0;

    void drain() {
        for (const auto & c : pending) { ggml_backend_tensor_set(c.tensor, c.data, c.offset, c.size); }
        pending.clear();
    }
    delayed_backend() {
        buft = *ggml_backend_cpu_buffer_type();
        buft.context = this;
        buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t b, size_t size) -> ggml_backend_buffer_t {
            if (static_cast<delayed_backend *>(b->context)->fail_allocation) { return nullptr; }
            return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
        };
        device.context = this;
        device.iface.get_buffer_type = [](ggml_backend_dev_t d) { return &static_cast<delayed_backend *>(d->context)->buft; };
        device.iface.event_new = [](ggml_backend_dev_t d) {
            ++static_cast<delayed_backend *>(d->context)->events;
            return new ggml_backend_event{d, nullptr};
        };
        device.iface.event_free = [](ggml_backend_dev_t d, ggml_backend_event_t e) {
            --static_cast<delayed_backend *>(d->context)->events;
            delete e;
        };
        device.iface.event_synchronize = [](ggml_backend_dev_t d, ggml_backend_event_t) {
            auto & s = *static_cast<delayed_backend *>(d->context);
            ++s.waits;
            s.drain();
        };
        backend.device = &device;
        backend.context = this;
        backend.iface.set_tensor_async = [](ggml_backend_t b, ggml_tensor * t, const void * data, size_t offset, size_t size) {
            auto & s = *static_cast<delayed_backend *>(b->context);
            s.pending.push_back({t, data, offset, size});
            s.largest_copy = std::max(s.largest_copy, size);
            if (++s.submissions == s.fail_submission) { throw std::runtime_error("injected copy failure"); }
        };
        backend.iface.synchronize = [](ggml_backend_t b) {
            auto & s = *static_cast<delayed_backend *>(b->context);
            ++s.syncs;
            s.drain();
        };
        backend.iface.event_record = [](ggml_backend_t b, ggml_backend_event_t) {
            ++static_cast<delayed_backend *>(b->context)->records;
        };
    }
};

static void test_cached_hit_priority() {
    auto first = std::make_shared<fixture>(GGML_TYPE_F32, 7);
    auto second = std::make_shared<fixture>(GGML_TYPE_F32, 7);
    ggml_backend_tensor_memset(second->tensors[2], 0, 0, ggml_nbytes(second->tensors[2]));
    auto bank = std::make_shared<llama_moe_host_bank>(first, first->tensors[0], first->tensors[1], first->tensors[2]);
    auto other = std::make_shared<llama_moe_host_bank>(second, second->tensors[0], second->tensors[1], second->tensors[2]);
    llama_moe_cached_executor cache({bank, other}, 2 * llama_moe_min_tile_bytes(*bank));
    const std::vector<float> input(32, 0.2f);
    cache.execute(0, input, 1, 2, {1, 2}, {0.5f, 0.5f}, 1, llama_moe_activation::silu);
    const auto before = cache.metrics();
    const auto result = cache.execute(0, input, 1, 3, {0, 1, 2}, {0.2f, 0.3f, 0.5f}, 1, llama_moe_activation::silu);
    close(result.output, ordinary(*bank, input, 1, 3, {0, 1, 2}, {0.2f, 0.3f, 0.5f}).output);
    check(cache.metrics().loads == before.loads + 1 && cache.metrics().hits == before.hits + 2,
            "miss evicted a resident route from a later tile");
    rejects([&] { cache.execute(0, input, 1, 2, {1, 99}, {0.5f, 0.5f}, 1, llama_moe_activation::silu); });
    const auto zero = cache.execute(1, input, 1, 1, {0}, {1.0f}, 1, llama_moe_activation::silu);
    close(zero.output, std::vector<float>(32, 0.0f));
    const auto recovered = cache.execute(0, input, 1, 1, {0}, {1.0f}, 1, llama_moe_activation::silu);
    close(recovered.output, ordinary(*bank, input, 1, 1, {0}, {1.0f}).output);
    const auto warm = cache.metrics();
    cache.execute(0, input, 1, 1, {0}, {1.0f}, 1, llama_moe_activation::silu);
    check(cache.metrics().copied_bytes == warm.copied_bytes, "warm executor copied weights");
    auto incompatible = std::make_shared<fixture>(GGML_TYPE_F16, 7, false, 32, 64);
    auto bad = std::make_shared<llama_moe_host_bank>(incompatible, incompatible->tensors[0], incompatible->tensors[1], incompatible->tensors[2]);
    rejects([&] { llama_moe_cached_executor rejected({bank, bad}, 65536); });
}

static void test_cache_completion() {
    delayed_backend backend;
    const std::array<ggml_type, 3> types{GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32};
    backend.fail_allocation = true;
    bool failed = false;
    try { ggml_moe_cache_storage cache(&backend.backend, 12288, 7, 32, 32, types); }
    catch (const std::bad_alloc &) { failed = true; }
    check(failed && backend.events == 0 && backend.pending.empty(), "allocation failure cleanup");
    backend.fail_allocation = false;
    auto owner = std::make_shared<std::vector<float>>(1024, 0.25f);
    ggml_moe_cache_source src{owner, {owner->data(), owner->data(), owner->data()}, {4096, 4096, 4096}};
    const ggml_moe_cache_key key{1, 7, 0, 0, 0};
    {
        ggml_moe_cache_storage cache(&backend.backend, 12288, 7, 32, 32, types);
        const auto r = cache.acquire(key, src);
        check(backend.pending.empty() && backend.records == 2 && backend.waits == 2, "load published before event wait");
        auto * t = cache.tensor(r.ticket, key, 0);
        float actual = 0;
        ggml_backend_tensor_get(t, &actual, 0, sizeof(actual));
        check(actual == 0.25f, "delayed copy not completed");
        float replacement = 0.75f;
        ggml_backend_tensor_set_async(&backend.backend, t, &replacement, 0, sizeof(replacement));
        cache.release(r.ticket, key);
        check(backend.pending.empty() && backend.records == 3 && backend.waits == 3, "release before backend completion");
        const auto warm = cache.acquire(key, src);
        check(warm.status == ggml_moe_cache_status::hit && backend.records == 3, "hit submitted transfer");
        ggml_backend_tensor_get(cache.tensor(warm.ticket, key, 0), &actual, 0, sizeof(actual));
        check(actual == replacement, "release did not drain submitted work");
        ggml_backend_tensor_set_async(&backend.backend, t, owner->data(), 0, sizeof(float));
    }
    check(backend.pending.empty() && backend.events == 0 && backend.syncs > 0, "destruction did not drain and free events");
}

static void test_staged_transfer(bool events) {
    delayed_backend backend;
    if (!events) { backend.device.iface.event_new = nullptr; }
    auto owner = std::make_shared<std::vector<uint8_t>>(4096);
    for (size_t i = 0; i < owner->size(); ++i) { (*owner)[i] = uint8_t(i * 17 + i / 53); }
    ggml_moe_cache_source src{owner, {owner->data(), owner->data(), owner->data()}, {4096, 4096, 4096}};
    size_t remaining = SIZE_MAX;
    ggml_moe_transfer_options options{127, [](void * p) { return (*static_cast<size_t *>(p))-- == 0; }, &remaining};
    ggml_moe_cache_storage cache(&backend.backend, 12288, 7, 32, 32,
            {GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32}, options);
    const ggml_moe_cache_key key{1, 7, 0, 0, 0};
    for (bool cancellation : {false, true}) {
        remaining = cancellation ? 2 : SIZE_MAX;
        backend.fail_submission = cancellation ? SIZE_MAX : backend.submissions + 2;
        bool failed = false;
        try { cache.acquire(key, src); } catch (const std::runtime_error &) { failed = true; }
        check(failed && backend.pending.empty(), "failed transfer was not drained");
        check(cache.acquire_existing(key).status == ggml_moe_cache_status::miss, "partial transfer became resident");
    }
    remaining = SIZE_MAX;
    backend.fail_submission = SIZE_MAX;
    const auto r = cache.acquire(key, src);
    for (size_t p = 0; p < 3; ++p) {
        std::vector<uint8_t> actual(4096);
        ggml_backend_tensor_get(cache.tensor(r.ticket, key, p), actual.data(), 0, actual.size());
        check(actual == *owner, "staging reused before completion or lost a tail");
    }
    cache.release(r.ticket, key);
    check(cache.staging_bytes() == 127 && backend.largest_copy <= 127, "staging budget exceeded");
    check(cache.stats().failed_loads == 2, "failed loads not counted");
}

static void device_close(const std::vector<float> & actual, const std::vector<float> & expected) {
    check(actual.size() == expected.size(), "device output shape");
    double error = 0, signal = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        check(std::isfinite(actual[i]), "nonfinite device output");
        error += double(actual[i] - expected[i]) * (actual[i] - expected[i]);
        signal += double(expected[i]) * expected[i];
        check(std::abs(actual[i] - expected[i]) <= 0.003f + 0.05f * std::abs(expected[i]), "device maximum error");
    }
    if (std::sqrt(error / actual.size()) > 1e-4 + 0.05 * std::sqrt(signal / actual.size())) {
        std::fprintf(stderr, "device error L2=%g signal L2=%g first=%g expected=%g\n",
                std::sqrt(error), std::sqrt(signal), actual.front(), expected.front());
        throw std::runtime_error("device relative L2 error");
    }
}

static void test_resident_geglu(ggml_backend_dev_t device) {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    std::vector<float> table(65536), gate;
    for (size_t i = 0; i < table.size(); ++i) { table[i] = ggml_fp16_to_fp32(ggml_fp16_t(i)); }
    gate = table;
    ggml_cpu_gelu_table_f32(table.data());
    for (int i = 0; i < 0x7bff; ++i) {
        const float midpoint = (ggml_fp16_to_fp32(ggml_fp16_t(i)) + ggml_fp16_to_fp32(ggml_fp16_t(i + 1))) * 0.5f;
        for (float x : {midpoint, std::nextafter(midpoint, -INFINITY), std::nextafter(midpoint, INFINITY)}) {
            gate.push_back(x);
            gate.push_back(-x);
        }
    }
    for (float x : {-10.0f, 10.0f, -0.001f, 0.001f, -1.0f, 1.0f}) {
        gate.push_back(std::nextafter(x, -INFINITY));
        gate.push_back(x);
        gate.push_back(std::nextafter(x, INFINITY));
    }
    std::vector<float> up(gate.size()), expected(gate.size()), actual(gate.size());
    for (size_t i = 0; i < up.size(); ++i) { up[i] = float(int(i % 17) - 8) * 0.13f; }
    ggml_cpu_geglu_f32(gate.data(), up.data(), expected.data(), int64_t(gate.size()));
    ggml_init_params params{16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    auto * g = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, int64_t(gate.size()));
    auto * u = ggml_dup_tensor(ctx.get(), g);
    auto * t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, int64_t(table.size()));
    auto * out = ggml_geglu_split(ctx.get(), g, u);
    out->src[2] = t;
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    check(bool(buffer), "GeGLU device allocation");
    ggml_backend_tensor_set(g, gate.data(), 0, ggml_nbytes(g));
    ggml_backend_tensor_set(u, up.data(), 0, ggml_nbytes(u));
    ggml_backend_tensor_set(t, table.data(), 0, ggml_nbytes(t));
    check(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "resident GeGLU graph");
    ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!((std::isnan(expected[i]) && std::isnan(actual[i])) || std::memcmp(&expected[i], &actual[i], sizeof(float)) == 0)) { std::fprintf(stderr, "GeGLU index=%zu gate=%a up=%a expected=%a actual=%a\n", i, gate[i], up[i], expected[i], actual[i]); }
        check((std::isnan(expected[i]) && std::isnan(actual[i])) ||
                std::memcmp(&expected[i], &actual[i], sizeof(float)) == 0, "resident GeGLU differs from CPU table");
    }
    std::puts("PASS resident GeGLU: all half inputs and rounding midpoints, signed zero, infinities, NaNs and threshold neighbors");
}

static void test_persistent_workspace(ggml_backend_dev_t device) {
    auto owner = std::make_shared<fixture>(GGML_TYPE_Q4_0, 7, false, 256, 256, true, true, true);
    auto bank = std::make_shared<llama_moe_host_bank>(owner, owner->tensors[0], nullptr, owner->tensors[2], owner->scales);
    llama_moe_executor_options options;
    options.device = device;
    const size_t budget = llama_moe_min_tile_bytes(*bank, ggml_backend_dev_buffer_type(device));
    llama_moe_cached_executor executor({bank, bank}, budget, options);
    for (int step = 0; step < 12; ++step) {
        std::vector<float> input(256);
        for (size_t d = 0; d < input.size(); ++d) { input[d] = 0.1f * std::cos(float(d + step)); }
        const std::vector<int32_t> ids{step % 7};
        const std::vector<float> weights{0.3f + step * 0.01f};
        device_close(executor.execute(step % 2, input, 1, 1, ids, weights, 2, llama_moe_activation::gelu).output,
                ordinary(*bank, input, 1, 1, ids, weights, true).output);
    }
    const auto warm = executor.metrics();
    check(warm.workspace_builds == 1 && warm.workspace_reuses == 11, "decode rebuilt persistent workspace");
    std::printf("Persistent FFN CUDA captures=%llu replays=%llu\n",
            (unsigned long long) warm.graph_captures, (unsigned long long) warm.graph_replays);
    if (std::getenv("FT_TEST_REQUIRE_GRAPHS")) {
        check(warm.graph_captures > 0 && warm.graph_replays > 0, "persistent expert FFN did not replay");
    }
    check(warm.retained_compute_bytes > 0 && warm.retained_compute_bytes + warm.activation_table <= options.compute_bytes,
            "persistent compute accounting");
    rejects([&] { executor.resize(1); });
    check(executor.metrics().retained_compute_bytes == warm.retained_compute_bytes, "failed resize discarded workspace");
    const std::vector<float> input(3 * 256, 0.1f);
    for (auto activation : {llama_moe_activation::silu, llama_moe_activation::gelu}) {
        device_close(executor.execute(0, input, 3, 1, {2, 2, 2}, {0.2f, 0.3f, 0.4f}, 2, activation).output,
                ordinary(*bank, input, 3, 1, {2, 2, 2}, {0.2f, 0.3f, 0.4f}, activation == llama_moe_activation::gelu).output);
    }
    check(executor.metrics().workspace_builds == 3, "shape/activation workspace invalidation");
    executor.resize(2 * budget);
    check(executor.metrics().retained_compute_bytes == 0, "resize retained stale cache pointers");
    device_close(executor.execute(1, input, 3, 1, {2, 2, 2}, {0.2f, 0.3f, 0.4f}, 2, llama_moe_activation::gelu).output,
            ordinary(*bank, input, 3, 1, {2, 2, 2}, {0.2f, 0.3f, 0.4f}, true).output);
    check(executor.metrics().workspace_builds == 4, "resize did not rebuild workspace");
    std::puts("PASS persistent workspace: changing inputs/routes/layers, eviction, shape/activation changes and resize");
}

static void test_device_boundary(ggml_backend_dev_t device) {
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
            for (size_t d = 0; d < actual.size(); ++d) {
                check(std::abs(actual[d] - expected.output[d]) <= 1e-6f, "device merge changed same-mode arithmetic");
            }
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
        const auto graphs = executor.metrics();
        std::printf("GPU boundary fast=%d CUDA captures=%llu replays=%llu\n", int(fast),
                (unsigned long long) graphs.graph_captures, (unsigned long long) graphs.graph_replays);
        if (std::getenv("FT_TEST_REQUIRE_GRAPHS")) {
            check(graphs.graph_captures > 0 && graphs.graph_replays > 0, "CUDA graph replay not exercised");
        }
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
        bool rejected = false;
        try { limited.execute_device(0, x, ids, w, out, llama_moe_activation::silu); }
        catch (const std::runtime_error &) { rejected = true; }
        check(rejected, "device boundary ignored compute budget");
    }
    std::puts("PASS GPU tensor boundary: scales, duplicate routes, ordered merge, eviction, resize, cancellation and budget");
}

static void test_device_executor(const char * device_name) {
    ggml_backend_load_all();
    const auto device = ggml_backend_dev_by_name(device_name);
    check(device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU, "requested GPU unavailable");
    test_device_boundary(device);
    test_persistent_workspace(device);
    test_resident_geglu(device);
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
      for (bool fused : {false, true}) {
        auto owner = std::make_shared<fixture>(type, 7, false, 256, 256, true, fused, true);
        auto bank = std::make_shared<llama_moe_host_bank>(owner, owner->tensors[0], fused ? nullptr : owner->tensors[1], owner->tensors[2], owner->scales);
        for (size_t capacity : {1u, 2u, 7u}) {
            std::printf("RUN %s fused=%d capacity=%zu\n", ggml_type_name(type), int(fused), capacity);
            bool abort = false;
            llama_moe_executor_options options{device, 4093, 8 * 1024 * 1024,
                    [](void * p) { return *static_cast<bool *>(p); }, &abort};
            options.pipeline = true;
            llama_moe_cached_executor cache({bank}, capacity * llama_moe_min_tile_bytes(*bank, ggml_backend_dev_buffer_type(device)), options);
            for (size_t tokens : {1u, 5u}) {
                const size_t k = 3;
                std::vector<float> input(tokens * 256), weights(tokens * k, 1.0f / k);
                std::vector<int32_t> ids(tokens * k);
                for (size_t i = 0; i < input.size(); ++i) { input[i] = 0.1f * std::cos(float(i) * 0.13f); }
                for (size_t i = 0; i < ids.size(); ++i) { ids[i] = int32_t((i * 3 + i / 3) % 7); }
                for (auto activation : {llama_moe_activation::silu, llama_moe_activation::gelu}) {
                    const auto expected = ordinary(*bank, input, tokens, k, ids, weights, activation == llama_moe_activation::gelu);
                    for (size_t repeat = 0; repeat < 2; ++repeat) {
                        const auto before = cache.metrics();
                        const auto result = cache.execute(0, input, tokens, k, ids, weights, 2, activation);
                        device_close(result.output, expected.output);
                        device_close(result.expert_outputs, expected.expert_outputs);
                        check(result.peak_compute_bytes <= options.compute_bytes && cache.metrics().staging == options.staging_bytes,
                                "device allocation budget");
                        if (capacity == 7 && repeat == 1) { check(cache.metrics().copied_bytes == before.copied_bytes, "warm device copied weights"); }
                    }
                    abort = true;
                    bool cancelled = false;
                    try { cache.execute(0, input, tokens, k, ids, weights, 2, activation); }
                    catch (const std::runtime_error &) { cancelled = true; }
                    check(cancelled, "device cancellation ignored");
                    abort = false;
                    device_close(cache.execute(0, input, tokens, k, ids, weights, 2, activation).output, expected.output);
                }
            }
        }
        std::printf("PASS device %s %s: capacities 1/2/7, decode/prefill, SiLU/GELU, scales, warm reuse, cancellation\n",
                device_name, ggml_type_name(type));
      }
    }
    auto owner = std::make_shared<fixture>(GGML_TYPE_F32, 7, false, 256, 256, false, false, true);
    auto bank = std::make_shared<llama_moe_host_bank>(owner, owner->tensors[0], owner->tensors[1], owner->tensors[2]);
    int remaining = 5;
    llama_moe_executor_options options{device, 4093, 8 * 1024 * 1024,
            [](void * p) { auto & n = *static_cast<int *>(p); return n >= 0 && n-- == 0; }, &remaining};
    llama_moe_cached_executor cache({bank}, 4 * llama_moe_min_tile_bytes(*bank), options);
    const std::vector<float> input(256, 0.1f);
    bool failed = false;
    try { cache.execute(0, input, 1, 1, {3}, {1}, 1, llama_moe_activation::silu); }
    catch (const std::runtime_error &) { failed = true; }
    check(failed && cache.metrics().copied_bytes > 0, "device partial-transfer cancellation not exercised");
    remaining = -1;
    device_close(cache.execute(0, input, 1, 1, {3}, {1}, 1, llama_moe_activation::silu).output,
            ordinary(*bank, input, 1, 1, {3}, {1}).output);
    options.compute_bytes = 1;
    llama_moe_cached_executor limited({bank}, 4 * llama_moe_min_tile_bytes(*bank), options);
    failed = false;
    try { limited.execute(0, input, 1, 1, {0}, {1}, 1, llama_moe_activation::silu); }
    catch (const std::runtime_error &) { failed = true; }
    check(failed, "device compute limit ignored");
    auto cancellation_owner = std::make_shared<fixture>(GGML_TYPE_F32, 7, false, 256, 256, true);
    auto cancellation_bank = std::make_shared<llama_moe_host_bank>(cancellation_owner, cancellation_owner->tensors[0],
            cancellation_owner->tensors[1], cancellation_owner->tensors[2], cancellation_owner->scales);
    options.compute_bytes = 8 * 1024 * 1024;
    llama_moe_cached_executor cancellation_cache({cancellation_bank}, llama_moe_min_tile_bytes(*cancellation_bank,
            ggml_backend_dev_buffer_type(device)), options);
    std::vector<float> cancellation_input(256);
    for (size_t i = 0; i < cancellation_input.size(); ++i) { cancellation_input[i] = 0.1f * std::cos(float(i) * 0.13f); }
    const auto cancellation_expected = ordinary(*cancellation_bank, cancellation_input, 1, 3, {0, 3, 6}, {1.0f/3, 1.0f/3, 1.0f/3});
    const auto cancellation_result = cancellation_cache.execute(0, cancellation_input, 1, 3, {0, 3, 6},
            {1.0f/3, 1.0f/3, 1.0f/3}, 2, llama_moe_activation::silu);
    device_close(cancellation_result.output, cancellation_expected.output);
    device_close(cancellation_result.expert_outputs, cancellation_expected.expert_outputs);
    std::puts("PASS device partial-transfer cancellation/retry, compute budget rejection and cancellation-heavy numerical fixture");

    auto zero_owner = std::make_shared<fixture>(GGML_TYPE_Q4_0, 7, false, 256, 256, true);
    auto * zero_gate = zero_owner->tensors[0];
    std::vector<uint8_t> zero_bytes(ggml_nbytes(zero_gate), 0x88);
    const ggml_fp16_t scale = ggml_fp32_to_fp16(1.0f);
    for (size_t offset = 0; offset < zero_bytes.size(); offset += ggml_type_size(GGML_TYPE_Q4_0)) {
        std::memcpy(zero_bytes.data() + offset, &scale, sizeof(scale));
    }
    ggml_backend_tensor_set(zero_gate, zero_bytes.data(), 0, zero_bytes.size());
    auto zero_bank = std::make_shared<llama_moe_host_bank>(zero_owner, zero_gate,
            zero_owner->tensors[1], zero_owner->tensors[2]);
    for (size_t capacity : {1u, 7u}) {
        llama_moe_cached_executor zero_cache({zero_bank}, capacity * llama_moe_min_tile_bytes(*zero_bank,
                ggml_backend_dev_buffer_type(device)), options);
        for (size_t tokens : {1u, 5u}) {
            std::vector<float> x(256 * tokens);
            for (size_t i = 0; i < x.size(); ++i) { x[i] = 0.1f * std::cos(float(i) * 0.13f); }
            std::vector<int32_t> ids(tokens * 3);
            for (size_t i = 0; i < ids.size(); ++i) { ids[i] = int32_t(i % 7); }
            const auto result = zero_cache.execute(0, x, tokens, 3, ids,
                    std::vector<float>(ids.size(), 1.0f / 3), 1, llama_moe_activation::silu);
            for (float value : result.expert_outputs) {
                check(value == 0.0f, "Q4 zero-point correction produced a nonzero expert");
            }
        }
    }
    std::puts("PASS Q4 zero-point cancellation across cache capacities and decode/prefill");
    for (int split : {-1, 1, 50, 100}) {
        options.cpu_miss_percent = split;
        for (size_t capacity : {1u, 7u}) {
            llama_moe_cached_executor hybrid({bank}, capacity * llama_moe_min_tile_bytes(*bank,
                    ggml_backend_dev_buffer_type(device)), options);
            for (auto activation : {llama_moe_activation::silu, llama_moe_activation::gelu}) {
                const std::vector<int32_t> routes{0, 3, 6, 0, 3, 0};
                const std::vector<float> x(512, 0.1f), w{0.2f, 0.3f, 0.5f, 0.5f, 0.2f, 0.3f};
                const auto expected = ordinary(*bank, x, 2, 3, routes, w, activation == llama_moe_activation::gelu);
                for (int repeat = 0; repeat < 2; ++repeat) {
                    const auto result = hybrid.execute(0, x, 2, 3, routes, w, 2, activation);
                    device_close(result.output, expected.output);
                    device_close(result.expert_outputs, expected.expert_outputs);
                    check(result.total_us >= result.compute_us, "invalid timing accounting");
                    if (split == 100) { check(result.cpu_assignments == routes.size(), "all misses did not run on CPU"); }
                }
            }
        }
    }
    options.cpu_miss_percent = -1;
    options.pipeline = true;
    llama_moe_cached_executor calibrated({bank}, 2 * llama_moe_min_tile_bytes(*bank,
            ggml_backend_dev_buffer_type(device)), options);
    device_close(calibrated.execute(0, input, 1, 3, {0, 3, 6}, {0.2f, 0.3f, 0.5f}, 2,
            llama_moe_activation::silu).output, ordinary(*bank, input, 1, 3, {0, 3, 6}, {0.2f, 0.3f, 0.5f}).output);
    check(calibrated.metrics().cpu_bytes_per_us > 0 && calibrated.metrics().transfer_bytes_per_us > 0,
            "bandwidth calibration missing");
    const auto old_budget = calibrated.metrics().budget;
    rejects([&] { calibrated.resize(1); });
    check(calibrated.metrics().budget == old_budget, "failed resize changed cache");
    calibrated.resize(3 * llama_moe_min_tile_bytes(*bank, ggml_backend_dev_buffer_type(device)));
    device_close(calibrated.execute(0, input, 1, 1, {0}, {1.0f}, 1, llama_moe_activation::silu).output,
            ordinary(*bank, input, 1, 1, {0}, {1.0f}).output);
    std::puts("PASS bandwidth calibration, thread recalibration, transactional cache resize");
    options.cpu_miss_percent = 50;
    llama_moe_cached_executor concurrent({bank}, 2 * llama_moe_min_tile_bytes(*bank, ggml_backend_dev_buffer_type(device)), options);
    const size_t tokens = 32;
    std::vector<int32_t> routes(tokens * 3);
    for (size_t i = 0; i < routes.size(); ++i) { routes[i] = int32_t(i % 7); }
    const std::vector<float> x(tokens * 256, 0.1f), w(routes.size(), 1.0f / 3);
    const auto measured = concurrent.execute(0, x, tokens, 3, routes, w, 2, llama_moe_activation::gelu);
    device_close(measured.output, ordinary(*bank, x, tokens, 3, routes, w, true).output);
    check(measured.cpu_assignments && measured.gpu_assignments && measured.cpu_assignments + measured.gpu_assignments == routes.size(), "hybrid assignment coverage");
    check(measured.prefetched_experts > 0, "pipeline did not prefetch any experts");
    options.cpu_miss_percent = 0;
    options.fast = true;
    llama_moe_cached_executor fast({bank}, 2 * llama_moe_min_tile_bytes(*bank,
            ggml_backend_dev_buffer_type(device)), options);
    check(fast.metrics().activation_table == 0, "fast path allocated compatibility table");
    device_close(fast.execute(0, x, tokens, 3, routes, w, 2, llama_moe_activation::silu).output,
            ordinary(*bank, x, tokens, 3, routes, w).output);
    std::printf("hybrid timing: cpu=%llu gpu=%llu total=%llu overlap-lower-bound=%llu us\n",
            (unsigned long long) measured.cpu_branch_us, (unsigned long long) measured.gpu_branch_us,
            (unsigned long long) measured.total_us, (unsigned long long) measured.overlap_us);
    remaining = 5;
    failed = false;
    try { concurrent.execute(0, x, tokens, 3, routes, w, 2, llama_moe_activation::gelu); }
    catch (const std::runtime_error &) { failed = true; }
    check(failed, "hybrid cancellation was ignored");
    remaining = -1;
    device_close(concurrent.execute(0, x, tokens, 3, routes, w, 2, llama_moe_activation::gelu).output,
            ordinary(*bank, x, tokens, 3, routes, w, true).output);
    std::puts("PASS hybrid fixed/adaptive splits, duplicate routes, warm reuse and eviction");
}

static void test_cache_storage(ggml_type type, bool mixed = false, int64_t embd = 32, int64_t ff = 32) {
    auto owner = std::make_shared<fixture>(type, 7, mixed, embd, ff);
    llama_moe_host_bank bank(owner, owner->tensors[0], owner->tensors[1], owner->tensors[2]);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    check(bool(backend), "cache backend");
    ggml_backend_cpu_set_n_threads(backend.get(), 2);
    const std::array<ggml_type, 3> types{bank.tensor(0)->type, bank.tensor(1)->type, bank.tensor(2)->type};
    auto source = [&](int32_t expert) {
        ggml_moe_cache_source src;
        src.owner = owner;
        for (size_t p = 0; p < 3; ++p) {
            src.bytes[p] = bank.tensor(p)->nb[2];
            src.data[p] = static_cast<const char *>(bank.tensor(p)->data) + size_t(expert) * src.bytes[p];
        }
        return src;
    };
    auto key = [](int32_t expert) { return ggml_moe_cache_key{1, 7, 0, 3, uint32_t(expert)}; };
    const size_t unit = llama_moe_min_tile_bytes(bank);
    rejects([&] { ggml_moe_cache_storage c(backend.get(), unit - 1, 7, embd, ff, types); });
    rejects([&] { ggml_moe_cache_storage c(nullptr, unit, 7, embd, ff, types); });
    rejects([&] { ggml_moe_cache_storage c(backend.get(), unit, 0, embd, ff, types); });
    rejects([&] { ggml_moe_cache_storage c(backend.get(), unit, 7, INT64_MAX, ff, types); });
    rejects([&] { ggml_moe_cache_storage c(backend.get(), unit, 7, embd, ff, {GGML_TYPE_I32, type, type}); });
    const std::vector<int32_t> ids{4, 0, 4, 2, 0, 6, 2, 1, 6};
    const size_t k = 3, tokens = 3;
    std::vector<float> input(size_t(embd) * tokens), weights(ids.size(), 1.0f / float(k));
    for (size_t i = 0; i < input.size(); ++i) { input[i] = std::sin(float(i) * 0.31f); }
    const auto expected = ordinary(bank, input, tokens, k, ids, weights);
    for (size_t capacity : {size_t(1), size_t(2), size_t(7)}) {
        ggml_moe_cache_storage cache(backend.get(), unit * capacity, 7, embd, ff, types);
        check(cache.capacity() == capacity && cache.allocated_bytes() <= unit * capacity, "cache budget");
        auto invalid = source(0);
        invalid.owner.reset();
        rejects([&] { cache.acquire(key(0), invalid); });
        invalid = source(0);
        --invalid.bytes[2];
        rejects([&] { cache.acquire(key(0), invalid); });
        check(cache.stats().loads == 0, "invalid source mutated cache");
        // Warm in a different order so route-local indices cannot stand in for cache slots.
        for (int32_t expert : {0, 4}) {
            const auto r = cache.acquire(key(expert), source(expert));
            cache.release(r.ticket, key(expert));
            rejects([&] { cache.tensor(r.ticket, key(expert), 0); });
            rejects([&] { cache.release(r.ticket, key(expert)); });
        }
        uint64_t after_first = 0;
        for (int repeat = 0; repeat < 2; ++repeat) {
            std::vector<float> actual(expected.expert_outputs.size());
            for (const auto & tile : llama_moe_plan_routes(ids, bank.n_expert(), capacity)) {
                std::vector<ggml_moe_cache_ticket> tickets;
                for (int32_t expert : tile.experts) {
                    const auto src = source(expert);
                    const auto r = cache.acquire(key(expert), src);
                    check(r.status == ggml_moe_cache_status::hit || r.status == ggml_moe_cache_status::load, "cache acquire");
                    tickets.push_back(r.ticket);
                    check(cache.acquire(key(expert), src).status == ggml_moe_cache_status::busy, "active expert eviction");
                    for (size_t p = 0; p < 3; ++p) {
                        auto * tensor = cache.tensor(r.ticket, key(expert), p);
                        check(size_t(tensor->ne[2]) == capacity, "full bank leaked into cache graph");
                        std::vector<uint8_t> bytes(src.bytes[p]);
                        ggml_backend_tensor_get(tensor, bytes.data(), r.ticket.slot * tensor->nb[2], bytes.size());
                        check(std::memcmp(bytes.data(), src.data[p], bytes.size()) == 0, "cache changed canonical bytes");
                    }
                }
                if (tickets.size() == capacity) {
                    check(cache.acquire(key(5), source(5)).status == ggml_moe_cache_status::full, "active slots replaced");
                }
                std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({
                        64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true}), ggml_free);
                check(bool(ctx), "cache graph context");
                const size_t count = tile.assignments.size();
                auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, embd, 1, int64_t(count));
                auto * route = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, int64_t(count));
                ggml_set_input(x);
                ggml_set_input(route);
                auto * gate = ggml_mul_mat_id(ctx.get(), cache.tensor(tickets[0], key(tile.experts[0]), 0), x, route);
                auto * up = ggml_mul_mat_id(ctx.get(), cache.tensor(tickets[0], key(tile.experts[0]), 1), x, route);
                auto * act = ggml_swiglu_split(ctx.get(), gate, up);
                auto * out = ggml_mul_mat_id(ctx.get(), cache.tensor(tickets[0], key(tile.experts[0]), 2), act, route);
                ggml_set_output(out);
                auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
                ggml_build_forward_expand(graph, out);
                ggml_backend_t backends[] = {backend.get()};
                std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> sched(
                        ggml_backend_sched_new(backends, nullptr, 1, 64, false, false), ggml_backend_sched_free);
                check(sched && ggml_backend_sched_alloc_graph(sched.get(), graph), "cache graph allocation");
                std::vector<float> packed(count * size_t(embd));
                std::vector<int32_t> remapped(count);
                for (size_t a = 0; a < count; ++a) {
                    std::copy_n(input.data() + tile.assignments[a] / k * size_t(embd), size_t(embd), packed.data() + a * size_t(embd));
                    remapped[a] = int32_t(tickets[tile.slots[a]].slot);
                }
                ggml_backend_tensor_set(x, packed.data(), 0, ggml_nbytes(x));
                ggml_backend_tensor_set(route, remapped.data(), 0, ggml_nbytes(route));
                check(ggml_backend_sched_graph_compute_async(sched.get(), graph) == GGML_STATUS_SUCCESS, "cache graph compute");
                for (size_t e = 0; e < tickets.size(); ++e) { cache.release(tickets[e], key(tile.experts[e])); }
                ggml_backend_sched_synchronize(sched.get());
                ggml_backend_tensor_get(out, packed.data(), 0, ggml_nbytes(out));
                for (size_t a = 0; a < count; ++a) {
                    std::copy_n(packed.data() + a * size_t(embd), size_t(embd), actual.data() + tile.assignments[a] * size_t(embd));
                }
            }
            close(actual, expected.expert_outputs);
            std::vector<float> merged(input.size(), 0.0f);
            for (size_t a = 0; a < ids.size(); ++a) {
                for (size_t d = 0; d < size_t(embd); ++d) {
                    merged[a / k * size_t(embd) + d] += actual[a * size_t(embd) + d] * weights[a];
                }
            }
            close(merged, expected.output);
            if (repeat == 0) { after_first = cache.storage_stats().copied_bytes; }
            if (repeat == 1 && capacity == 7) {
                check(cache.storage_stats().copied_bytes == after_first, "warm routes copied weights");
            }
        }
        check(cache.storage_stats().copied_bytes == cache.stats().loads * bank.expert_bytes(), "actual transfer bytes");
        const auto r = cache.acquire(key(0), source(0));
        auto other = key(0);
        ++other.layer;
        rejects([&] { cache.tensor(r.ticket, other, 0); });
        rejects([&] { cache.release(r.ticket, other); });
        // Destruction must drain even if the caller abandons an active ticket.
    }
}

static void test_mixed_layouts(ggml_backend_dev_t device = nullptr) {
    std::vector<std::shared_ptr<llama_moe_host_bank>> banks;
    size_t budget = 0;
    const std::array<ggml_type, 2> layouts[] = {
        {GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_S}, {GGML_TYPE_Q2_K, GGML_TYPE_Q2_K},
        {GGML_TYPE_Q3_K, GGML_TYPE_Q4_K}, {GGML_TYPE_Q3_K, GGML_TYPE_Q5_K},
        {GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_S}};
    for (const auto & types : layouts) {
        auto f = std::make_shared<fixture>(types[0], 4, false, 256, 256, false, false, true, types[1]);
        auto bank = std::make_shared<llama_moe_host_bank>(f, f->tensors[0], f->tensors[1], f->tensors[2]);
        budget += 4 * llama_moe_min_tile_bytes(*bank, device ? ggml_backend_dev_buffer_type(device) : nullptr);
        banks.push_back(bank);
    }
    llama_moe_executor_options options;
    options.device = device;
    options.fast = device != nullptr;
    llama_moe_cached_executor executor(banks, budget, options);
    const std::vector<float> input(256, 0.1f);
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (size_t layer = 0; layer < banks.size(); ++layer) {
            const auto expected = ordinary(*banks[layer], input, 1, 2, {0, 1}, {0.4f, 0.6f}, false, device);
            const auto result = executor.execute(layer, input, 1, 2, {0, 1}, {0.4f, 0.6f}, 1, llama_moe_activation::silu);
            close(result.output, expected.output);
        }
    }
    check(executor.metrics().hits >= 10, "mixed layout warm hits lost");
    check(executor.metrics().weights <= budget, "mixed layout budget exceeded");
    rejects([&] { executor.resize(1); });
    executor.resize(budget / 4);
    for (size_t layer = 0; layer < banks.size(); ++layer) {
        for (auto ids : {std::vector<int32_t>{2, 3}, std::vector<int32_t>{0, 1}}) {
            close(executor.execute(layer, input, 1, 2, ids, {0.4f, 0.6f}, 1, llama_moe_activation::silu).output,
                    ordinary(*banks[layer], input, 1, 2, ids, {0.4f, 0.6f}, false, device).output);
        }
    }
    check(executor.metrics().evictions > 0, "mixed layout eviction not exercised");
    const std::vector<float> batch(512, 0.1f);
    for (size_t layer = 0; layer < banks.size(); ++layer) {
        close(executor.execute(layer, batch, 2, 2, {0, 1, 1, 2}, {0.4f, 0.6f, 0.3f, 0.7f}, 1, llama_moe_activation::silu).output,
                ordinary(*banks[layer], batch, 2, 2, {0, 1, 1, 2}, {0.4f, 0.6f, 0.3f, 0.7f}, false, device).output);
    }
    std::puts("PASS K/I quantization and mixed layouts: decode/prefill parity, warm hits, eviction, resize, budget");
}

static void scheduler_boundary() {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
            ggml_init({16384 + ggml_graph_overhead(), nullptr, true}), ggml_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    auto * raw = backend.get();
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> sched(
            ggml_backend_sched_new(&raw, nullptr, 1, 64, false, false), ggml_backend_sched_free);
    auto * x = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_set_input(x);
    auto * y = ggml_scale(ctx.get(), x, 2);
    auto * z = ggml_add(ctx.get(), y, x);
    ggml_set_output(z);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, z);
    check(ggml_backend_sched_alloc_graph(sched.get(), graph), "boundary graph allocation");
    struct state { ggml_tensor * node; int calls = 0, observations = 0; bool fail = false; } data{y};
    ggml_backend_sched_set_node_executor(sched.get(),
        [](ggml_tensor * t, void * p) { return t == static_cast<state *>(p)->node; },
        [](ggml_tensor * t, void * p) {
            auto & state = *static_cast<struct state *>(p);
            ++state.calls;
            if (state.fail) { return GGML_STATUS_FAILED; }
            float value;
            ggml_backend_tensor_get(t->src[0], &value, 0, sizeof(value));
            value *= 3;
            ggml_backend_tensor_set(t, &value, 0, sizeof(value));
            return GGML_STATUS_SUCCESS;
        }, &data);
    ggml_backend_sched_set_eval_callback(sched.get(), [](ggml_tensor * t, bool ask, void * p) {
        auto & state = *static_cast<struct state *>(p);
        if (t != state.node) { return !ask; }
        if (!ask) { ++state.observations; }
        return true;
    }, &data);
    for (float input : {2.0f, 3.0f}) {
        ggml_backend_tensor_set(x, &input, 0, sizeof(input));
        check(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS, "external execution");
        float output;
        ggml_backend_tensor_get(z, &output, 0, sizeof(output));
        check(output == 4 * input, "external node was run by backend or dependency was not ready");
    }
    check(data.calls == 2 && data.observations == 2, "external boundary callback ordering");
    data.fail = true;
    check(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_FAILED, "external failure swallowed");
    data.fail = false;
    ggml_backend_sched_set_eval_callback(sched.get(), nullptr, nullptr);
    check(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS, "external retry without observer");
    ggml_backend_sched_set_node_executor(sched.get(), nullptr, nullptr, nullptr);
    check(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS, "ordinary scheduler restored");
    float output;
    ggml_backend_tensor_get(z, &output, 0, sizeof(output));
    check(output == 9, "ordinary backend did not execute restored node");
}

int main(int argc, char ** argv) {
    scheduler_boundary();
    try {
        if (argc == 3 && std::string(argv[1]) == "--device") {
            ggml_backend_load_all();
            test_mixed_layouts(ggml_backend_dev_by_name(argv[2]));
            test_device_executor(argv[2]);
            return 0;
        }
        check(argc == 1, "usage: test-moe-offload [--device CUDA0]");
        test_mixed_layouts();
        test_routes();
        test_invalid();
        test_cache_completion();
        test_staged_transfer(true);
        test_staged_transfer(false);
        test_cached_hit_priority();
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
            test_cache_storage(type);
            const size_t small = test_execution(type, false, 7);
            const size_t large = test_execution(type, false, 19);
            check(small == large, "tile allocation grew with host bank");
            const size_t gelu_small = test_execution(type, false, 7, 32, 32, true);
            const size_t gelu_large = test_execution(type, false, 19, 32, 32, true);
            check(gelu_small == gelu_large, "GELU tile allocation grew with host bank");
            std::printf("PASS %s: 7/19 experts, decode/batch, top-1/top-3, capacities 1/2/7\n", ggml_type_name(type));
        }
        test_execution(GGML_TYPE_Q4_0, true, 7);
        test_execution(GGML_TYPE_F32, false, 7, 13, 17);
        test_execution(GGML_TYPE_F16, false, 7, 32, 64);
        test_execution(GGML_TYPE_Q4_0, true, 7, 32, 64);
        test_cache_storage(GGML_TYPE_Q4_0, true, 32, 64);
        test_cache_storage(GGML_TYPE_F32, false, 13, 17);
        std::puts("PASS backend cache storage: canonical copies, bounded allocation, remapped FFN, eviction, warm reuse");
        std::puts("PASS GELU/output scales, mixed quantization, routing, ownership, accounting, invalid inputs");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
