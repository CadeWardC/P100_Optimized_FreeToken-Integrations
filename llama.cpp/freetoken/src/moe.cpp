#include "freetoken/moe.h"

#include "ggml-cpu.h"
#include "freetoken/cache.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <future>
#include <atomic>
#include <unordered_set>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace freetoken {
namespace {

size_t cpu_work_size(ggml_backend_t backend) {
#ifdef FREETOKEN_PATCHED_GGML
    return ggml_backend_cpu_get_work_size(backend);
#else
    (void) backend;
    return 0;
#endif
}

ggml_backend_buffer_type_t mapped_cpu_buffer_type() {
    static auto type = [] {
        alignas(64) static char storage[64];
        auto * buffer = ggml_backend_cpu_buffer_from_ptr(storage, sizeof(storage));
        if (!buffer) { throw std::bad_alloc(); }
        auto result = ggml_backend_buffer_get_type(buffer);
        ggml_backend_buffer_free(buffer);
        return result;
    }();
    return type;
}

using moe_clock = std::chrono::steady_clock;
uint64_t elapsed_us(moe_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(moe_clock::now() - start).count();
}

size_t checked_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::overflow_error("MoE byte count overflow");
    }
    return a + b;
}

size_t checked_mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error("MoE element count overflow");
    }
    return a * b;
}

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using scheduler_ptr = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
using allocator_ptr = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;

struct backend_drain {
    ggml_backend_t backend;
    ~backend_drain() { ggml_backend_synchronize(backend); }
};

void check_cancel(const executor_options & options) {
    if (options.abort_callback && options.abort_callback(options.abort_data)) {
        throw std::runtime_error("MoE execution cancelled");
    }
}

context_ptr make_context(size_t nodes = 64) {
    ggml_init_params params = { (nodes + 32) * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true };
    context_ptr ctx(ggml_init(params), ggml_free);
    if (!ctx) {
        throw std::bad_alloc();
    }
    return ctx;
}

bool supported_type(ggml_type type) {
    return ggml_moe_supported_type(type);
}

void validate_tensor(const ggml_tensor * t, bool split = false) {
    if (!t || !supported_type(t->type) || !t->data || !t->buffer || t->extra ||
        t->view_src || t->op != GGML_OP_NONE || (!split && !ggml_is_contiguous(t)) ||
        (ggml_backend_buffer_get_type(t->buffer) != ggml_backend_cpu_buffer_type() &&
         ggml_backend_buffer_get_type(t->buffer) != mapped_cpu_buffer_type())) {
        throw std::invalid_argument("MoE banks require canonical, contiguous ordinary CPU tensors");
    }
    if (t->ne[0] <= 0 || t->ne[1] <= 0 || t->ne[2] <= 0 || t->ne[3] != 1 ||
        t->ne[0] % ggml_blck_size(t->type) != 0 || t->ne[2] > INT32_MAX) {
        throw std::invalid_argument("Unsupported MoE bank dimensions");
    }
    const size_t slice = checked_mul(ggml_row_size(t->type, t->ne[0]), size_t(t->ne[1]));
    const size_t span = checked_add(checked_mul(t->nb[2], size_t(t->ne[2] - 1)), slice);
    if (t->nb[0] != ggml_type_size(t->type) || t->nb[1] != ggml_row_size(t->type, t->ne[0]) ||
            t->nb[2] != checked_mul(slice, split ? 2 : 1) || span != ggml_nbytes(t)) {
        throw std::invalid_argument("Unsupported MoE expert stride");
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(t->buffer));
    const uintptr_t data = reinterpret_cast<uintptr_t>(t->data);
    const size_t size = ggml_backend_buffer_get_size(t->buffer);
    if (data < base || data - base > size || span > size - (data - base)) {
        throw std::invalid_argument("MoE bank exceeds its retained CPU buffer");
    }
}

size_t tile_bytes(const host_bank & bank, size_t capacity, ggml_backend_buffer_type_t buft = nullptr) {
    auto ctx = make_context();
    for (size_t p = 0; p < 3; ++p) {
        const auto * t = bank.tensor(p);
        checked_mul(t->nb[2], capacity);
        ggml_new_tensor_3d(ctx.get(), t->type, t->ne[0], t->ne[1], int64_t(capacity));
    }
    return ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft ? buft : ggml_backend_cpu_buffer_type());
}

} // namespace

size_t min_tile_bytes(const host_bank & bank, ggml_backend_buffer_type_t buft) {
    return tile_bytes(bank, 1, buft);
}

size_t min_tile_bytes(const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down) {
    auto ctx = make_context();
    if (!gate || !down || (!up && gate->ne[1] % 2)) { throw std::invalid_argument("Invalid MoE projection geometry"); }
    for (const auto * t : {gate, up ? up : gate, down}) {
        ggml_new_tensor_3d(ctx.get(), t->type, t->ne[0], !up && t == gate ? t->ne[1] / 2 : t->ne[1], 1);
    }
    return ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), ggml_backend_cpu_buffer_type());
}

static void moe_compute(ggml_tensor * dst, const ggml_tensor * input,
        const ggml_tensor * ids, const ggml_tensor * weights, void * userdata, bool external) {
    auto & op = *static_cast<cpu_op *>(userdata);
    try {
        if (op.eager && !external) { throw std::runtime_error("GPU MoE must execute on the scheduler thread"); }
        if (external && op.executor && op.executor->supports_device(input->ne[1]) &&
                !ggml_backend_buffer_is_host(dst->buffer)) {
            op.metrics = op.executor->execute_device(op.layer, input, ids, weights, dst, op.activation);
            return;
        }
        std::vector<float> x(ggml_nelements(input));
        std::vector<int32_t> routes(ggml_nelements(ids));
        std::vector<float> w(ggml_nelements(weights));
        ggml_backend_tensor_get(input, x.data(), 0, ggml_nbytes(input));
        ggml_backend_tensor_get(ids, routes.data(), 0, ggml_nbytes(ids));
        ggml_backend_tensor_get(weights, w.data(), 0, ggml_nbytes(weights));
        auto result = op.executor ? op.executor->execute(op.layer, x, input->ne[1], ids->ne[0],
                routes, w, op.n_threads, op.activation) : reference_execute(*op.bank, x, input->ne[1], ids->ne[0],
                routes, w, op.budget, op.n_threads, op.activation);
        ggml_backend_tensor_set(dst, result.output.data(), 0, ggml_nbytes(dst));
        op.metrics = result;
        const size_t boundary_bytes = checked_add(checked_mul(x.capacity(), sizeof(float)),
                checked_add(checked_mul(routes.capacity(), sizeof(int32_t)), checked_mul(w.capacity(), sizeof(float))));
        op.metrics.peak_host_vector_bytes = checked_add(op.metrics.peak_host_vector_bytes, boundary_bytes);
        op.metrics.peak_temporary_bytes = checked_add(op.metrics.peak_temporary_bytes, boundary_bytes);
    } catch (...) {
        op.error = std::current_exception();
        ggml_backend_tensor_memset(dst, 0, 0, ggml_nbytes(dst));
    }
}

static void moe_cpu_compute(ggml_tensor * dst, const ggml_tensor * input,
        const ggml_tensor * ids, const ggml_tensor * weights, int ith, int, void * userdata) {
    if (ith == 0) { moe_compute(dst, input, ids, weights, userdata, false); }
}

ggml_status eager_compute(ggml_tensor * dst, cpu_op & op) {
    moe_compute(dst, dst->src[0], dst->src[1], dst->src[2], &op, true);
    return op.error ? GGML_STATUS_FAILED : GGML_STATUS_SUCCESS;
}

ggml_tensor * cpu_graph(ggml_context * ctx, ggml_tensor * input,
        ggml_tensor * ids, ggml_tensor * weights, cpu_op * op) {
    if (!op || !op->bank || input->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        weights->type != GGML_TYPE_F32 || input->ne[0] != op->bank->n_embd() ||
        input->ne[2] != 1 || input->ne[3] != 1 || ids->ne[1] != input->ne[1] ||
        ids->ne[2] != 1 || ids->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != ids->ne[0] ||
        weights->ne[2] != input->ne[1] || weights->ne[3] != 1) {
        throw std::invalid_argument("Invalid CPU MoE graph boundary");
    }
    op->output = ggml_map_custom3(ctx, ggml_cont(ctx, input), ggml_cont(ctx, ids),
            ggml_cont(ctx, weights), moe_cpu_compute, 1, op);
    return op->output;
}

host_bank::host_bank(std::shared_ptr<const void> owner,
        const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
        const ggml_tensor * down_scale)
    : owner(std::move(owner)), tensors{gate, up, down}, down_scale(down_scale) {
    if (!this->owner) {
        throw std::invalid_argument("MoE host bank requires a lifetime owner");
    }
    if (!up) {
        validate_tensor(gate);
        if (gate->ne[1] % 2) { throw std::invalid_argument("Fused MoE gate/up width must be even"); }
        split_tensors = std::make_shared<std::array<ggml_tensor, 2>>();
        for (size_t p = 0; p < 2; ++p) {
            auto & t = (*split_tensors)[p];
            t = *gate;
            t.ne[1] /= 2;
            t.data = static_cast<uint8_t *>(gate->data) + p * checked_mul(t.nb[1], size_t(t.ne[1]));
            tensors[p] = &t;
        }
        gate = tensors[0];
        up = tensors[1];
    }
    for (size_t p = 0; p < 3; ++p) {
        validate_tensor(tensors[p], split_tensors && p < 2);
        bytes_per_expert = checked_add(bytes_per_expert, projection_bytes(p));
    }
    if (gate == up || gate == down || up == down ||
        gate->ne[0] != up->ne[0] || gate->ne[1] != up->ne[1] ||
        down->ne[0] != gate->ne[1] || down->ne[1] != gate->ne[0] ||
        up->ne[2] != gate->ne[2] || down->ne[2] != gate->ne[2]) {
        throw std::invalid_argument("MoE gate/up/down shapes do not match");
    }
    checked_mul(bytes_per_expert, size_t(n_expert()));
    if (down_scale) {
        validate_tensor(down_scale);
        if (down_scale->type != GGML_TYPE_F32 || down_scale->ne[0] != n_expert() ||
            down_scale->ne[1] != 1 || down_scale->ne[2] != 1) {
            throw std::invalid_argument("MoE output scales require one F32 scalar per expert");
        }
        for (int32_t expert = 0; expert < n_expert(); ++expert) {
            if (!std::isfinite(output_scale(expert))) {
                throw std::invalid_argument("MoE output scale must be finite");
            }
        }
    }
}

int64_t host_bank::n_embd() const { return tensors[0]->ne[0]; }
int64_t host_bank::n_ff() const { return tensors[0]->ne[1]; }
int64_t host_bank::n_expert() const { return tensors[0]->ne[2]; }
size_t host_bank::host_bytes() const {
    return checked_add(bytes_per_expert * size_t(n_expert()), down_scale ? ggml_nbytes(down_scale) : 0);
}
size_t host_bank::expert_bytes() const { return bytes_per_expert; }
size_t host_bank::projection_bytes(size_t projection) const {
    const auto * t = tensor(projection);
    return checked_mul(t->nb[1], size_t(t->ne[1]));
}

float host_bank::output_scale(int32_t expert) const {
    if (expert < 0 || expert >= n_expert()) {
        throw std::invalid_argument("MoE output scale expert out of range");
    }
    return down_scale ? static_cast<const float *>(down_scale->data)[expert] : 1.0f;
}

const ggml_tensor * host_bank::tensor(size_t projection) const {
    if (projection >= 3) {
        throw std::out_of_range("Invalid MoE projection");
    }
    return tensors[projection];
}

std::vector<route_tile> plan_routes(
        const std::vector<int32_t> & ids, int64_t n_expert, size_t capacity) {
    if (capacity == 0 || n_expert <= 0 || n_expert > INT32_MAX) {
        throw std::invalid_argument("Invalid MoE route capacity or expert count");
    }
    std::vector<route_tile> tiles;
    std::unordered_map<int32_t, std::pair<size_t, int32_t>> locations;
    for (size_t a = 0; a < ids.size(); ++a) {
        const int32_t id = ids[a];
        if (id < 0 || id >= n_expert) {
            throw std::invalid_argument("MoE expert ID is out of range");
        }
        auto found = locations.find(id);
        if (found == locations.end()) {
            if (tiles.empty() || tiles.back().experts.size() == capacity) {
                tiles.emplace_back();
            }
            const int32_t slot = int32_t(tiles.back().experts.size());
            tiles.back().experts.push_back(id);
            found = locations.emplace(id, std::make_pair(tiles.size() - 1, slot)).first;
        }
        auto & tile = tiles[found->second.first];
        tile.assignments.push_back(a);
        tile.slots.push_back(found->second.second);
    }
    return tiles;
}

namespace {

struct cache_leases {
    struct entry { int32_t expert; ggml_moe_cache_ticket ticket; };
    ggml_moe_cache_storage * cache;
    uint32_t layer;
    std::vector<entry> entries;
    ggml_moe_cache_key key(int32_t expert) const { return {1, 1, 0, layer, uint32_t(expert)}; }
    ~cache_leases() {
        for (auto & e : entries) {
            if (e.ticket.owner) { cache->release(e.ticket, key(e.expert)); }
        }
    }
    entry & find(int32_t expert) {
        for (auto & e : entries) { if (e.expert == expert) { return e; } }
        throw std::logic_error("MoE cache route missing lease");
    }
};

void prioritize_hits(std::vector<route_tile> & tiles, const std::vector<int32_t> & ids, cache_leases & leases, size_t capacity) {
    size_t count = 0;
    for (const auto & tile : tiles) { count += tile.experts.size(); }
    leases.entries.reserve(count);
    for (const auto & tile : tiles) {
        for (int32_t expert : tile.experts) {
            auto r = leases.cache->acquire_existing(leases.key(expert));
            if (r.status != ggml_moe_cache_status::hit && r.status != ggml_moe_cache_status::miss) {
                throw std::logic_error("MoE cache has outstanding work");
            }
            leases.entries.push_back({expert, r.ticket});
        }
    }
    tiles.clear();
    std::unordered_map<int32_t, std::pair<size_t, int32_t>> locations;
    for (bool hits : {true, false}) {
        for (const auto & e : leases.entries) {
            if (bool(e.ticket.owner) != hits) { continue; }
            if (tiles.empty() || tiles.back().experts.size() == capacity) { tiles.emplace_back(); }
            locations.emplace(e.expert, std::make_pair(tiles.size() - 1, int32_t(tiles.back().experts.size())));
            tiles.back().experts.push_back(e.expert);
        }
    }
    for (size_t a = 0; a < ids.size(); ++a) {
        const auto found = locations.find(ids[a]);
        if (found == locations.end()) { continue; }
        const auto loc = found->second;
        tiles[loc.first].assignments.push_back(a);
        tiles[loc.first].slots.push_back(loc.second);
    }
}

// Copy contiguous rows without moving tensor contents through host memory.
void copy_device_rows(ggml_backend_t backend, const ggml_tensor * src, size_t src_row, ggml_tensor * dst, size_t dst_row, size_t width) {
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
    ggml_backend_tensor_copy_async(backend, backend, &a, &b);
}

void release_device_graph(ggml_backend_t backend, ggml_cgraph * graph) {
    if (!backend || !graph) { return; }
    using release_fn = void (*)(ggml_backend_t, const ggml_cgraph *);
    auto release = reinterpret_cast<release_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "ggml_backend_cuda_graph_release"));
    if (release) { release(backend, graph); }
}

struct device_merge_workspace {
    ggml_backend_t backend = nullptr;
    ~device_merge_workspace() { release_device_graph(backend, graph); }
    context_ptr ctx{nullptr, ggml_free};
    allocator_ptr alloc{nullptr, ggml_gallocr_free};
    ggml_tensor * contributions = nullptr, * scales = nullptr, * weights = nullptr, * out = nullptr, * checked_ids = nullptr;
    ggml_cgraph * graph = nullptr;
    size_t top_k = 0, compute_bytes = 0;
};

struct expert_workspace {
    ggml_backend_t backend = nullptr;
    ~expert_workspace() { release_device_graph(backend, graph); }
    context_ptr ctx{nullptr, ggml_free};
    scheduler_ptr sched{nullptr, ggml_backend_sched_free};
    allocator_ptr alloc{nullptr, ggml_gallocr_free};
    ggml_tensor * x = nullptr;
    ggml_tensor * route = nullptr;
    ggml_tensor * out = nullptr;
    ggml_cgraph * graph = nullptr;
    size_t count = 0, compute_bytes = 0;
    freetoken::activation activation = activation::silu;
    std::vector<float> packed;
};

struct expert_workspace_cache {
    // One arena bounds retained memory independently of the number of layers and shapes.
    std::unique_ptr<expert_workspace> workspace;
    uint64_t builds = 0, reuses = 0;
};

reference_result execute_ffn(
        const host_bank & bank,
        const std::vector<float> & input, size_t n_tokens, size_t top_k,
        const std::vector<int32_t> & ids, const std::vector<float> & weights,
        size_t tile_weight_budget, int n_threads, freetoken::activation activation,
        ggml_moe_cache_storage * cache, ggml_backend_t cached_backend, uint32_t layer,
        const std::shared_ptr<host_bank> & bank_owner,
        const executor_options & options = {}, ggml_tensor * gelu_table = nullptr,
        const std::unordered_set<int32_t> * selected = nullptr, expert_workspace_cache * workspaces = nullptr,
        const ggml_tensor * device_input = nullptr, device_merge_workspace * merge = nullptr) {
    const auto execution_start = moe_clock::now();
    check_cancel(options);
    if (activation != activation::silu && activation != activation::gelu) {
        throw std::invalid_argument("Unsupported MoE activation");
    }
    if (!n_tokens || !top_k || top_k > size_t(bank.n_expert()) || n_threads <= 0) {
        throw std::invalid_argument("Invalid MoE token, top-k, or thread count");
    }
    const size_t assignments = checked_mul(n_tokens, top_k);
    const size_t embd = size_t(bank.n_embd());
    if (assignments > size_t(INT64_MAX) || (!device_input && input.size() != checked_mul(n_tokens, embd)) ||
        ids.size() != assignments || (!device_input && weights.size() != assignments)) {
        throw std::invalid_argument("MoE input or routing shape mismatch");
    }
    for (float w : weights) {
        if (!std::isfinite(w)) {
            throw std::invalid_argument("MoE routing weight must be finite");
        }
    }
    if (!cache && tile_bytes(bank, 1) > tile_weight_budget) {
        throw std::invalid_argument("MoE weight budget cannot hold one complete expert");
    }
    size_t lo = 1, hi = size_t(bank.n_expert());
    while (!cache && lo < hi) {
        const size_t mid = lo + (hi - lo + 1) / 2;
        if (tile_bytes(bank, mid) <= tile_weight_budget) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const size_t capacity = cache ? (options.pipeline ? std::max(size_t(1), cache->capacity() / 2) : cache->capacity()) : lo;
    auto tiles = plan_routes(ids, bank.n_expert(), capacity);
    if (selected) {
        std::vector<route_tile> filtered;
        for (const auto & tile : tiles) {
            route_tile kept;
            for (size_t a = 0; a < tile.assignments.size(); ++a) {
                const auto expert = ids[tile.assignments[a]];
                if (!selected->count(expert)) { continue; }
                auto it = std::find(kept.experts.begin(), kept.experts.end(), expert);
                if (it == kept.experts.end()) { kept.experts.push_back(expert); it = kept.experts.end() - 1; }
                kept.slots.push_back(int32_t(it - kept.experts.begin()));
                kept.assignments.push_back(tile.assignments[a]);
            }
            if (!kept.experts.empty()) { filtered.push_back(std::move(kept)); }
        }
        tiles = std::move(filtered);
    }
    cache_leases leases{cache, layer, {}};
    if (cache) { prioritize_hits(tiles, ids, leases, capacity); }
    reference_result result;
    result.host_bank_bytes = bank.host_bytes();
    if (!device_input) {
        result.output.resize(input.size(), 0.0f);
        result.expert_outputs.resize(checked_mul(assignments, embd));
    }
    size_t vector_bytes = checked_add(checked_mul(result.output.capacity(), sizeof(float)),
            checked_mul(result.expert_outputs.capacity(), sizeof(float)));
    vector_bytes = checked_add(vector_bytes, checked_mul(tiles.capacity(), sizeof(route_tile)));
    vector_bytes = checked_add(vector_bytes, checked_mul(leases.entries.capacity(), sizeof(cache_leases::entry)));
    for (const auto & tile : tiles) {
        vector_bytes = checked_add(vector_bytes, checked_mul(tile.experts.capacity(), sizeof(int32_t)));
        vector_bytes = checked_add(vector_bytes, checked_mul(tile.assignments.capacity(), sizeof(size_t)));
        vector_bytes = checked_add(vector_bytes, checked_mul(tile.slots.capacity(), sizeof(int32_t)));
    }
    backend_ptr owned_backend(cache ? nullptr : ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_t backend = cache ? cached_backend : owned_backend.get();
    if (!backend) {
        throw std::bad_alloc();
    }
    const bool cpu = ggml_backend_is_cpu(backend);
    if (cpu) { ggml_backend_cpu_set_n_threads(backend, n_threads); }

    auto load_expert = [&](int32_t expert, bool optional) {
        auto & e = leases.find(expert);
        if (e.ticket.owner) { return; }
        ggml_moe_cache_source source;
        source.owner = bank_owner;
        for (size_t p = 0; p < 3; ++p) {
            source.bytes[p] = bank.projection_bytes(p);
            source.data[p] = static_cast<const uint8_t *>(bank.tensor(p)->data) + size_t(expert) * bank.tensor(p)->nb[2];
        }
        const auto r = cache->acquire(leases.key(expert), source);
        if (optional && r.status == ggml_moe_cache_status::full) { return; }
        if (r.status != ggml_moe_cache_status::load) { throw std::logic_error("MoE cache miss reservation failed"); }
        e.ticket = r.ticket;
        result.copied_weight_bytes = checked_add(result.copied_weight_bytes, bank.expert_bytes());
        if (optional) { ++result.prefetched_experts; }
    };
    for (size_t tile_index = 0; tile_index < tiles.size(); ++tile_index) {
        const auto & tile = tiles[tile_index];
        check_cancel(options);
        context_ptr weight_ctx(nullptr, ggml_free);
        buffer_ptr weight_buffer(nullptr, ggml_backend_buffer_free);
        ggml_tensor * compact[3];
        const auto load_start = moe_clock::now();
        size_t allocated = 0;
        std::vector<int32_t> remapped;
        if (cache) {
            for (int32_t expert : tile.experts) { load_expert(expert, false); }
            const auto & e = leases.find(tile.experts.front());
            for (size_t p = 0; p < 3; ++p) { compact[p] = cache->tensor(e.ticket, leases.key(e.expert), p); }
            remapped.reserve(tile.slots.size());
            for (int32_t slot : tile.slots) { remapped.push_back(int32_t(leases.find(tile.experts[slot]).ticket.slot)); }
        } else {
            weight_ctx = make_context();
            for (size_t p = 0; p < 3; ++p) {
                const auto * src = bank.tensor(p);
                compact[p] = ggml_new_tensor_3d(weight_ctx.get(), src->type, src->ne[0], src->ne[1], int64_t(tile.experts.size()));
            }
            weight_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx.get(), ggml_backend_cpu_buffer_type()));
            if (!weight_buffer) {
                throw std::bad_alloc();
            }
            allocated = ggml_backend_buffer_get_size(weight_buffer.get());
            if (allocated > tile_weight_budget) {
                throw std::runtime_error("MoE backend allocation exceeded weight budget");
            }
            ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_clear(weight_buffer.get(), 0);
            result.peak_tile_weight_bytes = std::max(result.peak_tile_weight_bytes, allocated);
            for (size_t p = 0; p < 3; ++p) {
                const auto * src = bank.tensor(p);
                for (size_t slot = 0; slot < tile.experts.size(); ++slot) {
                    const auto * bytes = static_cast<const uint8_t *>(src->data) + size_t(tile.experts[slot]) * src->nb[2];
                    ggml_backend_tensor_set(compact[p], bytes, slot * compact[p]->nb[2], bank.projection_bytes(p));
                    result.copied_weight_bytes = checked_add(result.copied_weight_bytes, bank.projection_bytes(p));
                }
            }
        }

        result.weight_load_us += elapsed_us(load_start);
        const size_t batch_limit = cpu ? tile.assignments.size() : 8;
        for (size_t begin = 0; begin < tile.assignments.size(); begin += batch_limit) {
            const size_t count = std::min(batch_limit, tile.assignments.size() - begin);
            expert_workspace_cache local;
            auto & owner = workspaces && !cpu ? *workspaces : local;
            const size_t table_bytes = gelu_table ? ggml_backend_buffer_get_size(gelu_table->buffer) : 0;
            if (!owner.workspace || owner.workspace->count != count || owner.workspace->activation != activation) {
                // Previous work is drained before reaching this point. Free the old arena before reserving another.
                owner.workspace.reset();
                auto replacement = std::make_unique<expert_workspace>();
                auto & ws = *replacement;
                ws.backend = cpu ? nullptr : backend;
                ws.ctx = make_context();
                ws.count = count;
                ws.activation = activation;
                auto & ctx = ws.ctx;
                auto & sched = ws.sched;
                auto & alloc = ws.alloc;
                auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, bank.n_embd(), 1, int64_t(count));
                auto * route = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, int64_t(count));
                ggml_set_input(x);
                ggml_set_input(route);
                auto * gate = ggml_mul_mat_id(ctx.get(), compact[0], x, route);
                auto * up = ggml_mul_mat_id(ctx.get(), compact[1], x, route);
                if (!cpu && !options.fast) {
                    // Keep projections materialized; fused CUDA FFNs are not validated for this eager boundary.
                    ggml_set_output(gate);
                    ggml_set_output(up);
                }
                auto * act = activation == activation::gelu ? ggml_geglu_split(ctx.get(), gate, up) :
                    ggml_swiglu_split(ctx.get(), gate, up);
                if (!cpu && !options.fast && activation == activation::gelu) {
                    if (!gelu_table) { throw std::runtime_error("MoE GPU GeGLU requires the compatibility table"); }
                    act->src[2] = gelu_table;
                }
                auto * out = ggml_mul_mat_id(ctx.get(), compact[2], act, route);
                for (auto * projection : {gate, up, out}) {
                    if (!cpu && !options.fast) {
                        ggml_mul_mat_set_prec(projection, GGML_PREC_F32);
                    }
                }
                ggml_set_output(out);
                auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
                ggml_build_forward_expand(graph, out);
                auto & compute_bytes = ws.compute_bytes;
                if (cpu) {
                    sched.reset(ggml_backend_sched_new(&backend, nullptr, 1, 64, false, false));
                    if (!sched || !ggml_backend_sched_alloc_graph(sched.get(), graph)) { throw std::bad_alloc(); }
                    compute_bytes = ggml_backend_sched_get_buffer_size(sched.get(), backend);
                } else {
                    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                        if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) {
                            throw std::invalid_argument("MoE device does not support the complete expert graph");
                        }
                    }
                    alloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
                    if (!alloc) { throw std::bad_alloc(); }
                    ggml_gallocr_reserve_n_size(alloc.get(), graph, nullptr, nullptr, &compute_bytes);
                    if (checked_add(compute_bytes, table_bytes) > options.compute_bytes) {
                        throw std::runtime_error("MoE device compute budget exceeded");
                    }
                    if (!ggml_gallocr_reserve(alloc.get(), graph) || !ggml_gallocr_alloc_graph(alloc.get(), graph)) {
                        throw std::bad_alloc();
                    }
                    compute_bytes = ggml_gallocr_get_buffer_size(alloc.get(), 0);
                    if (checked_add(compute_bytes, table_bytes) > options.compute_bytes) { throw std::runtime_error("MoE device allocation exceeded budget"); }
                }
                ws.x = x;
                ws.route = route;
                ws.out = out;
                ws.graph = graph;
                ws.packed.resize(checked_mul(count, embd));
                owner.workspace = std::move(replacement);
                ++owner.builds;
            } else {
                ++owner.reuses;
            }
            auto & ws = *owner.workspace;
            auto & ctx = ws.ctx;
            auto & sched = ws.sched;
            auto * x = ws.x;
            auto * route = ws.route;
            auto * out = ws.out;
            auto * graph = ws.graph;
            const size_t compute_bytes = ws.compute_bytes;
            backend_drain drain{backend};
            result.peak_compute_bytes = std::max(result.peak_compute_bytes, checked_add(compute_bytes, table_bytes));
            auto & packed = ws.packed;
            for (size_t a = 0; !device_input && a < count; ++a) {
                const size_t token = tile.assignments[begin + a] / top_k;
                std::copy_n(input.data() + token * embd, embd, packed.data() + a * embd);
            }
            const auto upload_start = moe_clock::now();
            if (device_input) {
                for (size_t a = 0; a < count; ++a) { copy_device_rows(backend, device_input, 0, x, a, embd); }
            } else {
                ggml_backend_tensor_set(x, packed.data(), 0, ggml_nbytes(x));
            }
            ggml_backend_tensor_set(route, (cache ? remapped.data() : tile.slots.data()) + begin, 0, ggml_nbytes(route));
            check_cancel(options);
            result.input_upload_us += elapsed_us(upload_start);
            result.input_upload_bytes += (device_input ? 0 : ggml_nbytes(x)) + ggml_nbytes(route);
            const auto compute_start = moe_clock::now();
            const auto status = cpu ? ggml_backend_sched_graph_compute(sched.get(), graph) : ggml_backend_graph_compute_async(backend, graph);
            if (!cpu) {
                if (status == GGML_STATUS_SUCCESS && options.pipeline && begin == 0 && tile_index + 1 < tiles.size()) {
                    const auto prefetch_start = moe_clock::now();
                    for (int32_t expert : tiles[tile_index + 1].experts) { load_expert(expert, true); }
                    result.weight_load_us += elapsed_us(prefetch_start);
                }
                ggml_backend_synchronize(backend);
            }
            result.compute_us += elapsed_us(compute_start);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("MoE reference graph execution failed");
            }
            check_cancel(options);
            const auto readback_start = moe_clock::now();
            if (device_input) {
                for (size_t a = 0; a < count; ++a) {
                    copy_device_rows(backend, out, a, merge->contributions, tile.assignments[begin + a], embd);
                }
            } else {
                ggml_backend_tensor_get(out, packed.data(), 0, ggml_nbytes(out));
                result.output_readback_us += elapsed_us(readback_start);
                result.output_readback_bytes += ggml_nbytes(out);
            }
            const size_t host_bytes = checked_add(vector_bytes, checked_add(checked_mul(packed.capacity(), sizeof(float)),
                    checked_mul(remapped.capacity(), sizeof(int32_t))));
            const size_t metadata_bytes = checked_add(weight_ctx ? ggml_get_mem_size(weight_ctx.get()) : 0, ggml_get_mem_size(ctx.get()));
            const size_t workspace_bytes = cache ? 0 : cpu_work_size(backend);
            size_t temporary_bytes = checked_add(allocated, compute_bytes);
            temporary_bytes = checked_add(temporary_bytes, checked_add(host_bytes, checked_add(metadata_bytes, workspace_bytes)));
            result.peak_host_vector_bytes = std::max(result.peak_host_vector_bytes, host_bytes);
            result.peak_metadata_bytes = std::max(result.peak_metadata_bytes, metadata_bytes);
            result.peak_workspace_bytes = std::max(result.peak_workspace_bytes, workspace_bytes);
            result.peak_temporary_bytes = std::max(result.peak_temporary_bytes, temporary_bytes);
            for (size_t a = 0; !device_input && a < count; ++a) {
                const float scale = bank.output_scale(ids[tile.assignments[begin + a]]);
                for (size_t e = 0; e < embd; ++e) { packed[a * embd + e] *= scale; }
                std::copy_n(packed.data() + a * embd, embd, result.expert_outputs.data() + tile.assignments[begin + a] * embd);
            }
            if (cpu) { result.cpu_assignments += count; } else { result.gpu_assignments += count; }
            ++result.tiles;
        }
        if (cache) {
            for (int32_t expert : tile.experts) {
                auto & e = leases.find(expert);
                cache->release(e.ticket, leases.key(expert));
                e.ticket = {};
            }
        }
    }
    const auto merge_start = moe_clock::now();
    // Match the ordinary FFN's routing order, independent of tile order.
    for (size_t a = 0; !device_input && a < assignments; ++a) {
        for (size_t d = 0; d < embd; ++d) {
            result.output[(a / top_k) * embd + d] += result.expert_outputs[a * embd + d] * weights[a];
        }
    }
    result.merge_us = elapsed_us(merge_start);
    result.total_us = elapsed_us(execution_start);
    if (cpu) { result.cpu_branch_us = result.total_us; } else { result.gpu_branch_us = result.total_us; }
    return result;
}

} // namespace

reference_result reference_execute(const host_bank & bank,
        const std::vector<float> & input, size_t n_tokens, size_t top_k,
        const std::vector<int32_t> & ids, const std::vector<float> & weights,
        size_t tile_weight_budget, int n_threads, freetoken::activation activation) {
    return execute_ffn(bank, input, n_tokens, top_k, ids, weights, tile_weight_budget, n_threads, activation,
            nullptr, nullptr, 0, {});
}

struct cached_executor::impl {
    std::vector<std::shared_ptr<host_bank>> banks;
    size_t budget;
    executor_options options;
    backend_ptr backend{nullptr, ggml_backend_free};
    backend_ptr transfer_backend{nullptr, ggml_backend_free};
    std::vector<std::unique_ptr<ggml_moe_cache_storage>> caches;
    std::vector<size_t> layer_cache, representatives, layout_layers;
    ggml_moe_cache_storage * cache = nullptr;

    void select(size_t layer) {
        auto * next = caches.at(layer_cache.at(layer)).get();
        if (cache != next) {
            ggml_backend_synchronize(backend.get());
            workspaces.workspace.reset();
            cache = next;
        }
    }

    std::vector<std::unique_ptr<ggml_moe_cache_storage>> allocate(size_t bytes) {
        const auto buft = ggml_backend_get_default_buffer_type(backend.get());
        std::vector<size_t> minimum;
        size_t required = 0;
        for (size_t layer : representatives) {
            minimum.push_back(min_tile_bytes(*banks[layer], buft));
            required = checked_add(required, minimum.back());
        }
        if (bytes < required) { throw std::invalid_argument("MoE cache budget cannot hold one expert per weight layout"); }
        std::vector<std::unique_ptr<ggml_moe_cache_storage>> result;
        size_t extra = bytes - required, remaining_weight = 0;
        for (size_t i = 0; i < minimum.size(); ++i) {
            remaining_weight = checked_add(remaining_weight, checked_mul(minimum[i], layout_layers[i]));
        }
        for (size_t i = 0; i < representatives.size(); ++i) {
            const auto & b = *banks[representatives[i]];
            const size_t weight = checked_mul(minimum[i], layout_layers[i]);
            const size_t share = checked_add(checked_mul(extra / remaining_weight, weight),
                    checked_mul(extra % remaining_weight, weight) / remaining_weight);
            extra -= share;
            remaining_weight -= weight;
            result.emplace_back(new ggml_moe_cache_storage(transfer_backend ? transfer_backend.get() : backend.get(),
                    minimum[i] + share, 1, b.n_embd(), b.n_ff(),
                    {b.tensor(0)->type, b.tensor(1)->type, b.tensor(2)->type},
                    {options.device ? options.staging_bytes / representatives.size() : 0, options.abort_callback, options.abort_data}));
        }
        return result;
    }
    context_ptr activation_ctx{nullptr, ggml_free};
    buffer_ptr activation_buffer{nullptr, ggml_backend_buffer_free};
    ggml_tensor * gelu_table = nullptr;
    expert_workspace_cache workspaces;
    std::unique_ptr<device_merge_workspace> merge;
    double cpu_bw = 0, transfer_bw = 0;
    int calibrated_threads = 0;
    uint64_t calibration_loads = 0, calibration_evictions = 0, calibration_bytes = 0, calibration_wait_us = 0;
};

cached_executor::cached_executor(
        const std::vector<std::shared_ptr<host_bank>> & banks, size_t budget,
        executor_options options) : state(new impl) {
    if (options.cpu_miss_percent < -1 || options.cpu_miss_percent > 100 || (options.cpu_miss_percent && !options.device)) {
        throw std::invalid_argument("MoE CPU miss percent requires a device and a value from -1 to 100");
    }
    if (banks.empty() || banks.size() > UINT32_MAX || !banks.front()) {
        throw std::invalid_argument("MoE cache requires retained layer banks");
    }
    const auto & first = *banks.front();
    for (const auto & bank : banks) {
        if (!bank) { throw std::invalid_argument("MoE cache layer is missing"); }
        for (size_t p = 0; p < 3; ++p) {
            const auto * a = bank->tensor(p);
            const auto * b = first.tensor(p);
            if (a->ne[0] != b->ne[0] || a->ne[1] != b->ne[1]) {
                throw std::invalid_argument("MoE cache requires uniform projection layouts across layers");
            }
        }
    }
    state->banks = banks;
    for (size_t layer = 0; layer < banks.size(); ++layer) {
        size_t layout = 0;
        for (; layout < state->representatives.size(); ++layout) {
            const auto & other = banks[state->representatives[layout]];
            bool same = true;
            for (size_t p = 0; p < 3; ++p) { same &= banks[layer]->tensor(p)->type == other->tensor(p)->type; }
            if (same) { break; }
        }
        if (layout == state->representatives.size()) {
            state->representatives.push_back(layer);
            state->layout_layers.push_back(0);
        }
        ++state->layout_layers[layout];
        state->layer_cache.push_back(layout);
    }
    state->budget = budget;
    state->options = options;
    if (options.device && (!options.staging_bytes || !options.compute_bytes)) {
        throw std::invalid_argument("MoE device execution requires transfer and compute budgets");
    }
    state->backend.reset(options.device ? ggml_backend_dev_init(options.device, nullptr) : ggml_backend_cpu_init());
    if (!state->backend) { throw std::bad_alloc(); }
#ifndef FREETOKEN_PATCHED_GGML
    if (!options.fast && !ggml_backend_is_cpu(state->backend.get())) {
        throw std::invalid_argument("FreeToken compatibility GPU execution requires patched GGML; select fast=true for native backend arithmetic");
    }
#else
    if (!options.fast && options.compute_bytes >= 65536 * sizeof(float) && !ggml_backend_is_cpu(state->backend.get()) && std::strstr(ggml_backend_name(state->backend.get()), "CUDA")) {
        state->activation_ctx = make_context();
        state->gelu_table = ggml_new_tensor_1d(state->activation_ctx.get(), GGML_TYPE_F32, 65536);
        state->activation_buffer.reset(ggml_backend_alloc_ctx_tensors(state->activation_ctx.get(), state->backend.get()));
        if (!state->activation_buffer) { throw std::bad_alloc(); }
        std::vector<float> table(65536);
        ggml_cpu_gelu_table_f32(table.data());
        ggml_backend_tensor_set(state->gelu_table, table.data(), 0, ggml_nbytes(state->gelu_table));
    }
#endif
    if (options.device && options.pipeline) {
        state->transfer_backend.reset(ggml_backend_dev_init(options.device, nullptr));
        if (!state->transfer_backend) { throw std::bad_alloc(); }
    }
    state->caches = state->allocate(budget);
    state->select(0);
}

cached_executor::~cached_executor() = default;

bool cached_executor::supports_device(size_t n_tokens) const {
    return state->options.device_tensors && state->options.device && !state->options.cpu_miss_percent && n_tokens == 1 &&
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(state->options.device),
                "ggml_backend_cuda_moe_read_routes") != nullptr;
}

ggml_backend_dev_t cached_executor::device() const { return state->options.device; }

execution_stats cached_executor::execute_device(size_t layer, const ggml_tensor * input,
        const ggml_tensor * ids, const ggml_tensor * weights, ggml_tensor * dst, freetoken::activation activation) {
    const auto start = moe_clock::now();
    const auto & bank = state->banks.at(layer);
    state->select(layer);
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
    std::vector<float> scales(top_k);
    if (!state->merge || state->merge->top_k != top_k) {
        state->merge.reset();
        auto ws = std::make_unique<device_merge_workspace>();
        ws->backend = backend;
        const size_t nodes = checked_add(32, checked_mul(top_k, 4));
        ws->ctx = make_context(nodes);
        auto * ctx = ws->ctx.get();
        ws->top_k = top_k;
        ws->contributions = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, bank->n_embd(), int64_t(top_k));
        ws->scales = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, int64_t(top_k));
        ws->weights = ggml_dup_tensor(ctx, ws->scales);
        ws->checked_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, int64_t(top_k));
        ggml_set_input(ws->checked_ids);
        ggml_set_output(ws->checked_ids);
        auto * zero = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bank->n_embd());
        for (auto * t : {ws->contributions, ws->scales, ws->weights, zero}) { ggml_set_input(t); }
        ggml_set_output(zero); // Keep the additive identity immutable across calls.
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
        ggml_build_forward_expand(ws->graph, ws->checked_ids);
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
    using read_routes_fn = void (*)(ggml_backend_t, const ggml_tensor *, const ggml_tensor *, ggml_tensor *, int32_t *);
    auto read_routes = reinterpret_cast<read_routes_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(state->options.device), "ggml_backend_cuda_moe_read_routes"));
    read_routes(backend, ids, weights, merge.checked_ids, routes.data());
    for (size_t a = 0; a < top_k; ++a) { scales[a] = bank->output_scale(routes[a]); }
    copy_device_rows(backend, weights, 0, merge.weights, 0, top_k);
    ggml_backend_tensor_set(merge.scales, scales.data(), 0, ggml_nbytes(merge.scales));
    auto options = state->options;
    options.compute_bytes -= merge.compute_bytes;
    auto result = execute_ffn(*bank, {}, 1, top_k, routes, {}, 0, 1, activation,
            state->cache, backend, uint32_t(layer), bank, options, state->gelu_table,
            nullptr, &state->workspaces, input, &merge);
    const auto merge_start = moe_clock::now();
    if (ggml_backend_graph_compute(backend, merge.graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("MoE device merge failed");
    }
    check_cancel(options);
    copy_device_rows(backend, merge.out, 0, dst, 0, size_t(bank->n_embd()));
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

reference_result cached_executor::execute(size_t layer, const std::vector<float> & input,
        size_t n_tokens, size_t top_k, const std::vector<int32_t> & ids, const std::vector<float> & weights,
        int n_threads, freetoken::activation activation) {
    state->merge.reset();
    const auto & bank = state->banks.at(layer);
    state->select(layer);
    if (!state->options.cpu_miss_percent) {
        return execute_ffn(*bank, input, n_tokens, top_k, ids, weights, 0, n_threads, activation,
                state->cache, state->backend.get(), uint32_t(layer), bank, state->options, state->gelu_table, nullptr, &state->workspaces);
    }
    if (state->options.cpu_miss_percent < 0 && state->calibrated_threads != n_threads) {
        if (n_threads <= 0) { throw std::invalid_argument("Invalid calibration thread count"); }
        check_cancel(state->options);
        const auto before = state->cache->stats();
        const auto before_storage = state->cache->storage_stats();
        state->cache->clear();
        auto account_calibration = [&] {
            state->calibration_loads += state->cache->stats().loads - before.loads;
            state->calibration_evictions += state->cache->stats().evictions - before.evictions;
            state->calibration_bytes += state->cache->storage_stats().copied_bytes - before_storage.copied_bytes;
            state->calibration_wait_us += state->cache->storage_stats().load_wait_us - before_storage.load_wait_us;
        };
        std::atomic<bool> stop{false}, ready{false};
        const std::vector<float> probe(size_t(bank->n_embd()), 0.01f);
        auto worker = std::async(std::launch::async, [&] {
            uint64_t count = 0;
            auto run = [&] {
                const auto & b = bank;
                const int32_t expert = int32_t((count / state->banks.size()) % size_t(b->n_expert()));
                return reference_execute(*b, probe, 1, 1, {expert}, {1.0f},
                        min_tile_bytes(*b), n_threads, activation);
            };
            run();
            ready.store(true);
            const auto start = moe_clock::now();
            do { run(); ++count; } while (!stop.load());
            return double(count) * bank->expert_bytes() / std::max(uint64_t(1), elapsed_us(start));
        });
        uint64_t count = 0, transfer_us = 0;
        try {
            while (!ready.load()) {
                check_cancel(state->options);
                if (worker.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready) {
                    worker.get();
                    throw std::runtime_error("MoE calibration worker stopped");
                }
            }
            const auto start = moe_clock::now();
            do {
                check_cancel(state->options);
                const size_t layer_id = layer;
                const auto & b = state->banks[layer_id];
                const uint32_t expert = uint32_t((count / state->banks.size()) % size_t(b->n_expert()));
                ggml_moe_cache_source source;
                source.owner = b;
                for (size_t p = 0; p < 3; ++p) {
                    source.bytes[p] = b->projection_bytes(p);
                    source.data[p] = static_cast<const uint8_t *>(b->tensor(p)->data) + expert * b->tensor(p)->nb[2];
                }
                const ggml_moe_cache_key key{1, 1, 0, uint32_t(layer_id), expert};
                const auto r = state->cache->acquire(key, source);
                if (r.status != ggml_moe_cache_status::load) { throw std::logic_error("Calibration requires a cold slot"); }
                state->cache->release(r.ticket, key);
                state->cache->clear();
                ++count;
            } while (count < 3 || elapsed_us(start) < 20000);
            transfer_us = std::max(uint64_t(1), elapsed_us(start));
            stop.store(true);
            const double cpu_bw = worker.get();
            state->cpu_bw = cpu_bw;
            state->transfer_bw = double(count) * bank->expert_bytes() / transfer_us;
            state->calibrated_threads = n_threads;
            account_calibration();
        } catch (...) {
            stop.store(true);
            if (worker.valid()) { worker.wait(); }
            state->cache->clear();
            account_calibration();
            throw;
        }
    }
    if (state->options.cpu_miss_percent < 0 && n_tokens > 1) {
        return execute_ffn(*bank, input, n_tokens, top_k, ids, weights, 0, n_threads, activation,
                state->cache, state->backend.get(), uint32_t(layer), bank, state->options, state->gelu_table, nullptr, &state->workspaces);
    }
    const auto start = moe_clock::now();
    const auto plan = plan_routes(ids, bank->n_expert(), size_t(bank->n_expert()));
    std::unordered_set<int32_t> cpu_experts, gpu_experts;
    std::vector<int32_t> misses;
    for (const auto & tile : plan) {
        for (int32_t expert : tile.experts) {
            if (state->cache->resident({1, 1, 0, uint32_t(layer), uint32_t(expert)})) { gpu_experts.insert(expert); }
            else { misses.push_back(expert); }
        }
    }
    const double fraction = state->options.cpu_miss_percent < 0 ?
            state->cpu_bw / (state->cpu_bw + state->transfer_bw) : state->options.cpu_miss_percent / 100.0;
    const size_t cpu_count = size_t(std::round(misses.size() * fraction));
    for (size_t i = 0; i < misses.size(); ++i) {
        (i < cpu_count ? cpu_experts : gpu_experts).insert(misses[i]);
    }
    if (cpu_experts.empty()) {
        auto result = execute_ffn(*bank, input, n_tokens, top_k, ids, weights, 0, n_threads, activation,
                state->cache, state->backend.get(), uint32_t(layer), bank, state->options, state->gelu_table, nullptr, &state->workspaces);
        return result;
    }
    check_cancel(state->options);
    // The worker owns a separate CPU backend. Only the scheduler accesses the GPU cache and abort callback.
    std::atomic<bool> stop{false};
    executor_options cpu_options;
    cpu_options.abort_callback = [](void * p) { return static_cast<std::atomic<bool> *>(p)->load(); };
    cpu_options.abort_data = &stop;
    auto worker = std::async(std::launch::async, [&] {
        return execute_ffn(*bank, input, n_tokens, top_k, ids, weights, min_tile_bytes(*bank),
                n_threads, activation, nullptr, nullptr, 0, {}, cpu_options, nullptr, &cpu_experts);
    });
    reference_result result;
    reference_result cpu;
    try {
        result = execute_ffn(*bank, input, n_tokens, top_k, ids, weights, 0, n_threads, activation,
                state->cache, state->backend.get(), uint32_t(layer), bank, state->options, state->gelu_table, &gpu_experts, &state->workspaces);
        while (worker.wait_for(std::chrono::milliseconds(1)) != std::future_status::ready) { check_cancel(state->options); }
        cpu = worker.get();
    } catch (...) {
        stop.store(true);
        if (worker.valid()) { worker.wait(); }
        throw;
    }
    check_cancel(state->options);
    result.cpu_branch_us = cpu.total_us;
    result.gpu_branch_us = result.total_us;
    const auto branches_us = elapsed_us(start);
    result.overlap_us = result.cpu_branch_us + result.gpu_branch_us > branches_us ?
            result.cpu_branch_us + result.gpu_branch_us - branches_us : 0;
    const size_t embd = size_t(bank->n_embd());
    result.cpu_assignments = result.gpu_assignments = 0;
    for (size_t a = 0; a < ids.size(); ++a) {
        if (cpu_experts.count(ids[a])) {
            std::copy_n(cpu.expert_outputs.data() + a * embd, embd, result.expert_outputs.data() + a * embd);
            ++result.cpu_assignments;
        } else { ++result.gpu_assignments; }
    }
    std::fill(result.output.begin(), result.output.end(), 0.0f);
    const auto merge_start = moe_clock::now();
    for (size_t a = 0; a < ids.size(); ++a) {
        for (size_t d = 0; d < embd; ++d) {
            result.output[(a / top_k) * embd + d] += result.expert_outputs[a * embd + d] * weights[a];
        }
    }
    result.merge_us += elapsed_us(merge_start);
    result.total_us = elapsed_us(start);
    result.copied_weight_bytes += cpu.copied_weight_bytes;
    result.tiles += cpu.tiles;
    result.peak_compute_bytes += cpu.peak_compute_bytes;
    result.peak_tile_weight_bytes = std::max(result.peak_tile_weight_bytes, cpu.peak_tile_weight_bytes);
    result.peak_host_vector_bytes += cpu.peak_host_vector_bytes;
    result.peak_metadata_bytes += cpu.peak_metadata_bytes;
    result.peak_workspace_bytes += cpu.peak_workspace_bytes;
    result.peak_temporary_bytes += cpu.peak_temporary_bytes;
    return result;
}

cache_metrics cached_executor::metrics() const {
    ggml_moe_cache_stats stats{};
    uint64_t copied_bytes = 0, load_wait_us = 0;
    size_t allocated = 0, metadata = 0, staging = 0;
    for (const auto & cache : state->caches) {
        stats.hits += cache->stats().hits;
        stats.loads += cache->stats().loads;
        stats.evictions += cache->stats().evictions;
        copied_bytes += cache->storage_stats().copied_bytes;
        load_wait_us += cache->storage_stats().load_wait_us;
        allocated += cache->allocated_bytes();
        metadata += cache->metadata_bytes();
        staging += cache->staging_bytes();
    }
    uint64_t captures = 0, replays = 0;
    using stats_fn = void (*)(ggml_backend_t, uint64_t *, uint64_t *);
    auto graph_stats = reinterpret_cast<stats_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(state->backend.get())), "ggml_backend_cuda_graph_stats"));
    if (graph_stats) { graph_stats(state->backend.get(), &captures, &replays); }
    return {state->budget, allocated, metadata,
            ggml_backend_is_cpu(state->backend.get()) ? cpu_work_size(state->backend.get()) : 0,
            stats.hits, stats.loads - state->calibration_loads, stats.evictions - state->calibration_evictions,
            copied_bytes - state->calibration_bytes, load_wait_us - state->calibration_wait_us,
            staging, state->activation_buffer ? ggml_backend_buffer_get_size(state->activation_buffer.get()) : 0, state->cpu_bw, state->transfer_bw,
            (state->workspaces.workspace ? state->workspaces.workspace->compute_bytes : 0) +
                    (state->merge ? state->merge->compute_bytes : 0),
            state->workspaces.builds, state->workspaces.reuses, captures, replays};
}

void cached_executor::resize(size_t budget) {
    ggml_backend_synchronize(state->backend.get());
    auto replacement = state->allocate(budget);
    state->workspaces.workspace.reset();
    state->merge.reset();
    state->caches.swap(replacement);
    state->cache = state->caches.front().get();
    state->budget = budget;
    state->calibration_loads = state->calibration_evictions = state->calibration_bytes = state->calibration_wait_us = 0;
}

} // namespace freetoken
