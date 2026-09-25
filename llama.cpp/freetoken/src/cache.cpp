#include "freetoken/cache.h"
#include "ggml-alloc.h"

#include <chrono>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cstring>

static void add_counter(uint64_t & counter, uint64_t value = 1) {
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    counter = value > max - counter ? max : counter + value;
}

ggml_moe_cache::ggml_moe_cache(size_t budget, size_t slot_bytes, uint64_t layout) :
    bytes_per_slot(slot_bytes), layout(layout) {
    if (slot_bytes == 0 || budget < slot_bytes || layout == 0 ||
            budget / slot_bytes > size_t(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("moe cache: invalid uniform arena geometry");
    }
    slots.resize(budget / slot_bytes);
}

uint64_t ggml_moe_cache::next_serial() {
    if (serial == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("moe cache: lease serial exhausted");
    }
    return ++serial;
}

ggml_moe_cache_request ggml_moe_cache::acquire(const ggml_moe_cache_key & key, bool existing_only) {
    if (key.model == 0 || key.layout != layout) {
        throw std::invalid_argument("moe cache: incompatible expert identity");
    }
    for (size_t i = 0; i < slots.size(); ++i) {
        auto & s = slots[i];
        if (s.status == state::empty || !(s.key == key)) {
            continue;
        }
        if (s.status == state::loading) {
            return { ggml_moe_cache_status::pending, {} };
        }
        if (s.status == state::in_use) {
            return { ggml_moe_cache_status::busy, {} };
        }
        s.lease = next_serial();
        s.status = state::in_use;
        s.last_use = s.lease;
        add_counter(counters.hits);
        return { ggml_moe_cache_status::hit, { this, i, s.generation, s.lease } };
    }

    if (existing_only) {
        return { ggml_moe_cache_status::miss, {} };
    }
    size_t victim = slots.size();
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto & s = slots[i];
        if (s.status == state::empty) {
            victim = i;
            break;
        }
        if (s.status == state::ready && (victim == slots.size() || s.last_use < slots[victim].last_use)) {
            victim = i;
        }
    }
    if (victim == slots.size()) {
        return { ggml_moe_cache_status::full, {} };
    }
    auto & s = slots[victim];
    if (s.generation == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("moe cache: content generation exhausted");
    }
    const uint64_t lease = next_serial();
    if (s.status == state::ready) {
        add_counter(counters.evictions);
    }
    s.key = key;
    s.status = state::loading;
    ++s.generation;
    s.lease = lease;
    s.last_use = lease;
    add_counter(counters.loads);
    return { ggml_moe_cache_status::load, { this, victim, s.generation, s.lease } };
}

ggml_moe_cache::slot & ggml_moe_cache::checked(const ggml_moe_cache_ticket & ticket, state expected) {
    if (ticket.owner != this || ticket.slot >= slots.size()) {
        throw std::invalid_argument("moe cache: foreign slot ticket");
    }
    auto & s = slots[ticket.slot];
    if (s.generation != ticket.generation || s.lease != ticket.lease || s.status != expected) {
        throw std::invalid_argument("moe cache: stale ticket or invalid transition");
    }
    return s;
}

void ggml_moe_cache::complete_load(const ggml_moe_cache_ticket & ticket, bool success) {
    auto & s = checked(ticket, state::loading);
    s.status = success ? state::in_use : state::empty;
    if (success) {
        add_counter(counters.completed_slot_bytes, uint64_t(bytes_per_slot));
    } else {
        add_counter(counters.failed_loads);
    }
}

void ggml_moe_cache::release(const ggml_moe_cache_ticket & ticket) {
    auto & s = checked(ticket, state::in_use);
    s.last_use = next_serial();
    s.status = state::ready;
}

bool ggml_moe_cache::usable(const ggml_moe_cache_ticket & ticket, const ggml_moe_cache_key & key) const {
    if (ticket.owner != this || ticket.slot >= slots.size()) {
        return false;
    }
    const auto & s = slots[ticket.slot];
    return s.status == state::in_use && s.generation == ticket.generation &&
           s.lease == ticket.lease && s.key == key;
}

struct ggml_moe_cache_storage::impl {
    ggml_backend_t backend;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
    std::unique_ptr<ggml_backend_event, decltype(&ggml_backend_event_free)> event{nullptr, ggml_backend_event_free};
    std::unique_ptr<ggml_moe_cache> cache;
    std::array<ggml_tensor *, 3> tensors{};
    ggml_moe_cache_storage_stats counters;
    ggml_moe_transfer_options transfer;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> staging{nullptr, ggml_backend_buffer_free};
    std::unique_ptr<uint8_t[]> pageable;
    uint8_t * staging_data = nullptr;

    explicit impl(ggml_backend_t backend) : backend(backend) {}
    ~impl() {
        if (buffer) {
            ggml_backend_synchronize(backend);
        }
    }

    void drain() {
        if (event) {
            ggml_backend_event_record(event.get(), backend);
            ggml_backend_event_synchronize(event.get());
        } else {
            ggml_backend_synchronize(backend);
        }
    }

    void describe(int64_t n_embd, int64_t n_ff, const std::array<ggml_type, 3> & types, size_t capacity) {
        ctx.reset(ggml_init({3 * ggml_tensor_overhead(), nullptr, true}));
        if (!ctx) {
            throw std::bad_alloc();
        }
        for (size_t p = 0; p < 3; ++p) {
            tensors[p] = ggml_new_tensor_3d(ctx.get(), types[p], p == 2 ? n_ff : n_embd,
                    p == 2 ? n_embd : n_ff, int64_t(capacity));
        }
    }
};

ggml_moe_cache_storage::ggml_moe_cache_storage(ggml_backend_t backend, size_t budget, uint64_t layout,
        int64_t n_embd, int64_t n_ff, const std::array<ggml_type, 3> & types,
        ggml_moe_transfer_options transfer) : state(new impl(backend)) {
    const uint64_t max_elements = std::min(uint64_t(SIZE_MAX), uint64_t(INT64_MAX)) / (3 * sizeof(float));
    if (!backend || !layout || n_embd <= 0 || n_ff <= 0 || uint64_t(n_embd) > max_elements / uint64_t(n_ff)) {
        throw std::invalid_argument("moe cache storage: invalid geometry or backend");
    }
    for (size_t p = 0; p < 3; ++p) {
        const auto type = types[p];
        if (!ggml_moe_supported_type(type) ||
                (p == 2 ? n_ff : n_embd) % ggml_blck_size(type) != 0) {
            throw std::invalid_argument("moe cache storage: unsupported projection type or shape");
        }
    }
    auto & s = *state;
    s.transfer = transfer;
    if (transfer.staging_bytes) {
        const auto host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
        if (host && ggml_backend_buft_is_host(host)) {
            s.staging.reset(ggml_backend_buft_alloc_buffer(host, transfer.staging_bytes));
        }
        if (s.staging && ggml_backend_buffer_get_size(s.staging.get()) > transfer.staging_bytes) {
            s.staging.reset();
        }
        if (s.staging) {
            s.staging_data = static_cast<uint8_t *>(ggml_backend_buffer_get_base(s.staging.get()));
        } else {
            s.pageable.reset(new uint8_t[transfer.staging_bytes]);
            s.staging_data = s.pageable.get();
        }
    }
    const auto buft = ggml_backend_get_default_buffer_type(backend);
    s.describe(n_embd, n_ff, types, 1);
    const size_t slot_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(s.ctx.get(), buft);
    s.cache.reset(new ggml_moe_cache(budget, slot_bytes, layout));
    for (const auto * t : s.tensors) {
        if (uint64_t(t->ne[0]) * uint64_t(t->ne[1]) > uint64_t(INT64_MAX) / s.cache->capacity()) {
            throw std::invalid_argument("moe cache storage: tensor size overflow");
        }
    }
    s.describe(n_embd, n_ff, types, s.cache->capacity());
    const size_t required = ggml_backend_alloc_ctx_tensors_from_buft_size(s.ctx.get(), buft);
    if (!required || required > budget) {
        throw std::invalid_argument("moe cache storage: backend padding exceeds budget");
    }
    s.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(s.ctx.get(), buft));
    if (!s.buffer) {
        throw std::bad_alloc();
    }
    if (ggml_backend_buffer_get_size(s.buffer.get()) > budget) {
        throw std::runtime_error("moe cache storage: backend allocation exceeds budget");
    }
    ggml_backend_buffer_set_usage(s.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(s.buffer.get(), 0); // Include quantized kernel tail padding.
    s.event.reset(ggml_backend_event_new(ggml_backend_get_device(backend)));
    s.drain();
}

ggml_moe_cache_storage::~ggml_moe_cache_storage() = default;

ggml_moe_cache_request ggml_moe_cache_storage::acquire_existing(const ggml_moe_cache_key & key) {
    return state->cache->acquire(key, true);
}

ggml_moe_cache_request ggml_moe_cache_storage::acquire(const ggml_moe_cache_key & key,
        const ggml_moe_cache_source & source) {
    auto & s = *state;
    if (!source.owner) {
        throw std::invalid_argument("moe cache storage: source requires lifetime owner");
    }
    for (size_t p = 0; p < 3; ++p) {
        if (!source.data[p] || source.bytes[p] != s.tensors[p]->nb[2]) {
            throw std::invalid_argument("moe cache storage: invalid canonical expert slice");
        }
    }
    const auto result = s.cache->acquire(key);
    if (result.status != ggml_moe_cache_status::load) {
        return result;
    }
    const auto owner = source.owner;
    const auto start = std::chrono::steady_clock::now();
    try {
        for (size_t p = 0; p < 3; ++p) {
            for (size_t offset = 0; offset < source.bytes[p];) {
                if (s.transfer.abort_callback && s.transfer.abort_callback(s.transfer.abort_data)) {
                    throw std::runtime_error("moe cache: transfer cancelled");
                }
                const size_t count = s.staging_data ? std::min(s.transfer.staging_bytes, source.bytes[p] - offset) : source.bytes[p];
                const auto * data = static_cast<const uint8_t *>(source.data[p]) + offset;
                if (s.staging_data) {
                    std::memcpy(s.staging_data, data, count);
                    data = s.staging_data;
                }
                ggml_backend_tensor_set_async(s.backend, s.tensors[p], data,
                        result.ticket.slot * s.tensors[p]->nb[2] + offset, count);
                add_counter(s.counters.copied_bytes, count);
                if (s.staging_data) { s.drain(); } // Do not overwrite a pending DMA source.
                offset += count;
            }
        }
        s.drain();
    } catch (...) {
        ggml_backend_synchronize(s.backend);
        s.cache->complete_load(result.ticket, false);
        throw;
    }
    add_counter(s.counters.load_wait_us, uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count()));
    s.cache->complete_load(result.ticket, true);
    return result;
}

ggml_tensor * ggml_moe_cache_storage::tensor(const ggml_moe_cache_ticket & ticket,
        const ggml_moe_cache_key & key, size_t projection) const {
    if (projection >= 3 || !state->cache->usable(ticket, key)) {
        throw std::invalid_argument("moe cache storage: invalid projection or inactive ticket");
    }
    return state->tensors[projection];
}

void ggml_moe_cache_storage::release(const ggml_moe_cache_ticket & ticket, const ggml_moe_cache_key & key) {
    if (!state->cache->usable(ticket, key)) {
        throw std::invalid_argument("moe cache storage: inactive ticket");
    }
    state->drain();
    state->cache->release(ticket);
}

size_t ggml_moe_cache_storage::capacity() const { return state->cache->capacity(); }
size_t ggml_moe_cache_storage::staging_bytes() const { return state->transfer.staging_bytes; }
size_t ggml_moe_cache_storage::allocated_bytes() const { return ggml_backend_buffer_get_size(state->buffer.get()); }
size_t ggml_moe_cache_storage::metadata_bytes() const {
    return ggml_get_mem_size(state->ctx.get()) + state->cache->metadata_bytes();
}
const ggml_moe_cache_stats & ggml_moe_cache_storage::stats() const { return state->cache->stats(); }
const ggml_moe_cache_storage_stats & ggml_moe_cache_storage::storage_stats() const { return state->counters; }

bool ggml_moe_cache::resident(const ggml_moe_cache_key & key) const {
    for (const auto & slot : slots) {
        if ((slot.status == state::ready || slot.status == state::in_use) && slot.key == key) { return true; }
    }
    return false;
}

bool ggml_moe_cache_storage::resident(const ggml_moe_cache_key & key) const {
    return state->cache->resident(key);
}

void ggml_moe_cache::clear() {
    for (const auto & s : slots) {
        if (s.status == state::loading || s.status == state::in_use) {
            throw std::invalid_argument("moe cache: cannot clear outstanding leases");
        }
    }
    for (auto & s : slots) { s.status = state::empty; }
}

void ggml_moe_cache_storage::clear() {
    state->drain();
    state->cache->clear();
}
