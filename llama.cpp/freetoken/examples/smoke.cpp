#include "freetoken/moe.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

struct weights_owner {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~weights_owner() {
        if (buffer) { ggml_backend_buffer_free(buffer); }
        if (ctx) { ggml_free(ctx); }
    }
};

static void check(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

int main() {
    try {
        auto owner = std::make_shared<weights_owner>();
        owner->ctx = ggml_init({8 * ggml_tensor_overhead(), nullptr, true});
        check(owner->ctx != nullptr, "context allocation");
        auto * gate = ggml_new_tensor_3d(owner->ctx, GGML_TYPE_F32, 32, 32, 4);
        auto * up = ggml_new_tensor_3d(owner->ctx, GGML_TYPE_F32, 32, 32, 4);
        auto * down = ggml_new_tensor_3d(owner->ctx, GGML_TYPE_F32, 32, 32, 4);
        owner->buffer = ggml_backend_alloc_ctx_tensors_from_buft(owner->ctx, ggml_backend_cpu_buffer_type());
        check(owner->buffer != nullptr, "weight allocation");
        for (auto * tensor : {gate, up, down}) {
            std::vector<float> data(ggml_nelements(tensor));
            for (size_t i = 0; i < data.size(); ++i) { data[i] = float(int(i % 17) - 8) * 0.01f; }
            ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
        }
        auto bank = std::make_shared<freetoken::host_bank>(owner, gate, up, down);
        owner.reset(); // The bank retains the tensor storage.
        const auto unit = freetoken::min_tile_bytes(*bank);
        freetoken::cached_executor executor({bank}, 2 * unit);
        const std::vector<float> input(64, 0.25f);
        const std::vector<int32_t> ids{0, 1, 1, 0};
        const std::vector<float> routes{0.25f, 0.75f, 0.5f, 0.5f};
        for (auto activation : {freetoken::activation::silu, freetoken::activation::gelu}) {
            auto expected = freetoken::reference_execute(*bank, input, 2, 2, ids, routes, unit, 1, activation);
            auto actual = executor.execute(0, input, 2, 2, ids, routes, 1, activation);
            check(actual.output.size() == expected.output.size(), "output shape");
            for (size_t i = 0; i < actual.output.size(); ++i) {
                check(std::isfinite(actual.output[i]) && std::fabs(actual.output[i] - expected.output[i]) < 1e-6f, "cached/reference parity");
            }
        }
        check(executor.metrics().hits > 0, "warm cache reuse");
        executor.resize(unit);
        check(executor.metrics().budget == unit, "cache resize");
        bool rejected = false;
        try { executor.execute(0, input, 2, 2, {0, 4, 1, 0}, routes, 1, freetoken::activation::silu); }
        catch (const std::exception &) { rejected = true; }
        check(rejected, "invalid expert must be rejected");
        std::puts("PASS standalone FreeToken: SiLU/GeGLU parity, ownership, warm reuse, resize, invalid routes");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
