#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <array>


inline bool ggml_moe_supported_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: case GGML_TYPE_IQ2_S:
            return true;
        default: return false;
    }
}

struct ggml_moe_cache_key {
    uint64_t model;
    uint64_t layout;
    uint64_t adapter;
    uint32_t layer;
    uint32_t expert;

    bool operator==(const ggml_moe_cache_key & other) const {
        return model == other.model && layout == other.layout && adapter == other.adapter &&
               layer == other.layer && expert == other.expert;
    }
};

class ggml_moe_cache;

struct ggml_moe_cache_ticket {
    const ggml_moe_cache * owner = nullptr;
    size_t slot = 0;
    uint64_t generation = 0;
    uint64_t lease = 0;
};

enum class ggml_moe_cache_status { hit, load, pending, busy, full, miss };

struct ggml_moe_cache_request {
    ggml_moe_cache_status status;
    ggml_moe_cache_ticket ticket;
};

struct ggml_moe_cache_stats {
    uint64_t hits = 0;
    uint64_t loads = 0;
    uint64_t evictions = 0;
    uint64_t completed_slot_bytes = 0; // padded slot bytes, not measured DMA traffic
    uint64_t failed_loads = 0;
};

// Fixed uniform slots, metadata only. The caller owns the arena and serializes all access.
// layout identifies the complete gate/up/down shape, types, strides and padding.
// Drain backend work before destruction. Tickets cannot outlive this object.
class ggml_moe_cache {
public:
    ggml_moe_cache(size_t budget, size_t slot_bytes, uint64_t layout);
    ggml_moe_cache(const ggml_moe_cache &) = delete;
    ggml_moe_cache & operator=(const ggml_moe_cache &) = delete;

    // A hit pins the slot. A load reserves it until transfer completion is reported.
    // Repeated keys in one route plan must share one ticket, not acquire twice.
    ggml_moe_cache_request acquire(const ggml_moe_cache_key & key, bool existing_only = false);

    // Call only after ALL bundle transfers complete, or after failed work is drained.
    // Success pins the loaded bundle; failure empties the slot. Pending DMA cannot be cancelled here.
    void complete_load(const ggml_moe_cache_ticket & ticket, bool success);

    // Call only after every kernel using the ticket completes.
    void release(const ggml_moe_cache_ticket & ticket);
    bool usable(const ggml_moe_cache_ticket & ticket, const ggml_moe_cache_key & key) const;

    bool resident(const ggml_moe_cache_key & key) const;
    void clear();
    size_t capacity() const { return slots.size(); }
    size_t arena_bytes() const { return slots.size() * bytes_per_slot; }
    size_t metadata_bytes() const { return slots.capacity() * sizeof(slot); }
    const ggml_moe_cache_stats & stats() const { return counters; }

private:
    enum class state { empty, loading, ready, in_use };
    struct slot {
        ggml_moe_cache_key key = {};
        state status = state::empty;
        uint64_t generation = 0;
        uint64_t lease = 0;
        uint64_t last_use = 0;
    };
    slot & checked(const ggml_moe_cache_ticket & ticket, state expected);
    uint64_t next_serial();
    size_t bytes_per_slot;
    uint64_t layout;
    uint64_t serial = 0;
    std::vector<slot> slots;
    ggml_moe_cache_stats counters;
};

struct ggml_moe_cache_source {
    // Immutable canonical GGUF slices for gate, up and down, retained through transfer completion.
    std::shared_ptr<const void> owner;
    std::array<const void *, 3> data;
    std::array<size_t, 3> bytes;
};

struct ggml_moe_cache_storage_stats {
    uint64_t copied_bytes = 0;
    uint64_t load_wait_us = 0; // host submission and completion wait, not device-only timing
};

struct ggml_moe_transfer_options {
    size_t staging_bytes = 0; // Zero copies directly from retained source memory.
    ggml_abort_callback abort_callback = nullptr;
    void * abort_data = nullptr;
};

// Eager, serialized backend storage. The backend must outlive this object.
// Its default buffer type must support canonical contiguous tensors (CPU/CUDA target).
// Submit all kernels on that backend and release tickets only after their last submission.
// Returned tensors describe the whole pool; every referenced slot must have a live ticket.
// Source identity/layout and canonical bytes are the caller's responsibility; no repacked weights.
class ggml_moe_cache_storage {
public:
    ggml_moe_cache_storage(ggml_backend_t backend, size_t budget, uint64_t layout,
            int64_t n_embd, int64_t n_ff, const std::array<ggml_type, 3> & types,
            ggml_moe_transfer_options transfer = {});
    ~ggml_moe_cache_storage();
    ggml_moe_cache_storage(const ggml_moe_cache_storage &) = delete;
    ggml_moe_cache_storage & operator=(const ggml_moe_cache_storage &) = delete;

    ggml_moe_cache_request acquire(const ggml_moe_cache_key & key, const ggml_moe_cache_source & source);
    // Pin a resident bundle, or return miss without reserving or evicting anything.
    ggml_moe_cache_request acquire_existing(const ggml_moe_cache_key & key);
    ggml_tensor * tensor(const ggml_moe_cache_ticket & ticket, const ggml_moe_cache_key & key, size_t projection) const;
    void release(const ggml_moe_cache_ticket & ticket, const ggml_moe_cache_key & key);
    bool resident(const ggml_moe_cache_key & key) const;
    void clear();
    size_t capacity() const;
    size_t allocated_bytes() const;
    size_t metadata_bytes() const;
    size_t staging_bytes() const;
    const ggml_moe_cache_stats & stats() const;
    const ggml_moe_cache_storage_stats & storage_stats() const;

private:
    struct impl;
    std::unique_ptr<impl> state;
};


