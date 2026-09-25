#include "ggml-moe-cache.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

using status = ggml_moe_cache_status;

static void check(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

template<class F> static void rejects(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "invalid operation accepted");
}

static ggml_moe_cache_key key(uint32_t expert) {
    return { 1, 7, 0, 0, expert };
}

static ggml_moe_cache_ticket load(ggml_moe_cache & cache, ggml_moe_cache_key k) {
    const auto r = cache.acquire(k);
    check(r.status == status::load, "expected miss");
    check(!cache.usable(r.ticket, k), "pending bundle usable");
    cache.complete_load(r.ticket, true);
    check(cache.usable(r.ticket, k), "completed bundle unavailable");
    return r.ticket;
}

static void lifecycle() {
    ggml_moe_cache cache(257, 128, 7);
    check(cache.acquire(key(0), true).status == status::miss && cache.stats().loads == 0, "probe reserved a miss");
    check(cache.capacity() == 2 && cache.arena_bytes() == 256, "budget rounding");
    const auto a = cache.acquire(key(0));
    const auto b = cache.acquire(key(1));
    check(a.status == status::load && b.status == status::load, "empty slots");
    check(cache.acquire(key(0)).status == status::pending, "duplicate pending load");
    check(cache.acquire(key(2)).status == status::full, "evicted pending DMA");
    rejects([&] { cache.release(a.ticket); });
    cache.complete_load(b.ticket, true); // Transfers complete out of order.
    check(!cache.usable(a.ticket, key(0)), "partial completion exposed another bundle");
    check(cache.acquire(key(1)).status == status::busy, "active slot acquired twice");
    check(cache.acquire(key(2)).status == status::full, "evicted active kernel");
    rejects([&] { cache.complete_load(b.ticket, true); });
    cache.release(b.ticket);
    const auto c = cache.acquire(key(2));
    check(c.status == status::load && c.ticket.slot == b.ticket.slot, "wrong eviction victim");
    check(c.ticket.generation != b.ticket.generation, "unchanged content generation");
    rejects([&] { cache.complete_load(b.ticket, false); });
    rejects([&] { cache.release(b.ticket); });
    cache.complete_load(c.ticket, false); // Caller has drained the failed transfer.
    rejects([&] { cache.complete_load(c.ticket, true); });
    const auto retry = load(cache, key(2));
    cache.release(retry);
    cache.complete_load(a.ticket, true);
    cache.release(a.ticket);
    const auto warm = cache.acquire(key(0));
    check(warm.status == status::hit && warm.ticket.generation == a.ticket.generation, "warm generation");
    check(warm.ticket.lease != a.ticket.lease, "lease reused");
    rejects([&] { cache.release(a.ticket); });
    check(!cache.usable(a.ticket, key(0)) && cache.usable(warm.ticket, key(0)), "stale lease usable");
    cache.release(warm.ticket);
    check(cache.acquire(key(99), true).status == status::miss && cache.stats().evictions == 1, "probe evicted a ready slot");
    const auto d = cache.acquire(key(3));
    check(d.ticket.slot == retry.slot, "LRU did not preserve recent hit");
    cache.complete_load(d.ticket, false);
    check(cache.stats().hits == 1 && cache.stats().loads == 5 && cache.stats().evictions == 2 &&
          cache.stats().failed_loads == 2 && cache.stats().completed_slot_bytes == 384, "lifecycle counters");

    ggml_moe_cache other(128, 128, 7);
    const auto foreign = load(other, key(0));
    rejects([&] { cache.release(foreign); });
    check(!cache.usable(foreign, key(0)), "foreign ticket usable");
    other.release(foreign);
    auto invalid = warm.ticket;
    invalid.slot = cache.capacity();
    rejects([&] { cache.release(invalid); });
    rejects([&] { cache.release({}); });
}

static void identities() {
    ggml_moe_cache cache(4 * 32, 32, 7);
    auto model = key(0);
    model.model = 2;
    auto layer = key(0);
    layer.layer = 1;
    auto adapter = key(0);
    adapter.adapter = 1;
    const std::vector<ggml_moe_cache_key> keys = { key(0), model, layer, adapter };
    std::vector<ggml_moe_cache_ticket> tickets;
    for (const auto & k : keys) {
        tickets.push_back(load(cache, k));
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        for (size_t j = 0; j < keys.size(); ++j) {
            check(cache.usable(tickets[i], keys[j]) == (i == j), "identity alias");
        }
        cache.release(tickets[i]);
    }
    auto bad = key(0);
    bad.layout = 8;
    rejects([&] { cache.acquire(bad); });
    bad = key(0);
    bad.model = 0;
    rejects([&] { cache.acquire(bad); });
    for (const auto & k : keys) {
        const auto r = cache.acquire(k);
        check(r.status == status::hit, "identity not retained");
        cache.release(r.ticket);
    }
}

static void lru_reference() {
    std::mt19937 random(100);
    for (size_t capacity : { 1u, 2u, 7u }) {
        ggml_moe_cache cache(capacity * 64, 64, 7);
        std::map<uint32_t, size_t> reference;
        uint64_t misses = 0;
        for (size_t step = 1; step <= 10000; ++step) {
            const uint32_t id = random() % 19;
            const bool hit = reference.count(id) != 0;
            const auto r = cache.acquire(key(id));
            check(r.status == (hit ? status::hit : status::load), "reference LRU mismatch");
            if (!hit) {
                ++misses;
                if (reference.size() == capacity) {
                    auto oldest = std::min_element(reference.begin(), reference.end(),
                        [](const auto & a, const auto & b) { return a.second < b.second; });
                    reference.erase(oldest);
                }
                cache.complete_load(r.ticket, true);
            }
            reference[id] = step;
            check(cache.usable(r.ticket, key(id)), "reference route unavailable");
            cache.release(r.ticket);
            check(cache.stats().completed_slot_bytes == misses * 64, "transfer accounting mismatch");
        }
    }
}

// Simulate compact slot contents and scatter duplicate token assignments across capacity-one tiles.
static void routes() {
    for (size_t capacity : { 1u, 2u, 7u }) {
        ggml_moe_cache cache(capacity * 32, 32, 7);
        std::vector<int> contents(capacity, -1);
        const std::vector<uint32_t> ids = { 6, 1, 6, 2, 1, 0, 5, 3, 4, 5, 0 };
        for (size_t repeat = 0; repeat < 3; ++repeat) {
            const auto bytes_before = cache.stats().completed_slot_bytes;
            std::vector<int> outputs(ids.size(), -1);
            for (uint32_t id = 0; id < 7; ++id) {
                const auto r = cache.acquire(key(id));
                if (r.status == status::load) {
                    contents[r.ticket.slot] = int(id) * 13;
                    cache.complete_load(r.ticket, true);
                } else {
                    check(r.status == status::hit, "route blocked");
                }
                check(cache.usable(r.ticket, key(id)), "invalid compact ID");
                for (size_t i = 0; i < ids.size(); ++i) {
                    if (ids[i] == id) {
                        outputs[i] = contents[r.ticket.slot] + int(i);
                    }
                }
                cache.release(r.ticket);
            }
            for (size_t i = 0; i < ids.size(); ++i) {
                check(outputs[i] == int(ids[i]) * 13 + int(i), "assignment lost or wrong slot");
            }
            if (capacity == 7 && repeat > 0) {
                check(cache.stats().completed_slot_bytes == bytes_before, "warm routes transferred weights");
            }
        }
    }
}

int main() {
    try {
        rejects([] { ggml_moe_cache c(0, 1, 7); });
        rejects([] { ggml_moe_cache c(1, 0, 7); });
        rejects([] { ggml_moe_cache c(31, 32, 7); });
        rejects([] { ggml_moe_cache c(32, 32, 0); });
        rejects([] { ggml_moe_cache c(std::numeric_limits<size_t>::max(), 1, 7); });
        {
            ggml_moe_cache cache(64, 32, 7);
            const auto r = cache.acquire(key(1));
            rejects([&] { cache.clear(); });
            cache.complete_load(r.ticket, true);
            rejects([&] { cache.clear(); });
            cache.release(r.ticket);
            cache.clear();
            check(!cache.resident(key(1)), "clear retained mapping");
            const auto next = cache.acquire(key(1));
            check(next.status == status::load && !cache.usable(r.ticket, key(1)), "clear revived stale lease");
            cache.complete_load(next.ticket, true);
            cache.release(next.ticket);
        }
        lifecycle();
        identities();
        lru_reference();
        routes();
        std::puts("moe cache: lifecycle, identity, 30000 LRU steps and compact route checks passed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "moe cache: %s\n", e.what());
        return 1;
    }
}
