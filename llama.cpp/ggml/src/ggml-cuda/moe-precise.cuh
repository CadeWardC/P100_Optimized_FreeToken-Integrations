#pragma once

#include "common.cuh"

static __global__ void moe_quantize_q8_0_precise(const float * x, block_q8_0 * y, int k, int rows) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * (k / 32)) { return; }
    x += index * 32;
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) { amax = fmaxf(amax, fabsf(x[j])); }
    const float d = __fdiv_rn(amax, 127.0f);
    const float inv = d == 0.0f ? 0.0f : __fdiv_rn(1.0f, d);
    y[index].d = __float2half_rn(d);
    for (int j = 0; j < 32; ++j) { y[index].qs[j] = int8_t(roundf(__fmul_rn(x[j], inv))); }
}

template <bool q4>
static __global__ void moe_dot_precise(const char * weights, const block_q8_0 * x, const int32_t * ids,
        float * out, int k, int n, int rows, size_t expert_stride, size_t row_stride) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (row >= n || token >= rows) { return; }
    const char * w = weights + size_t(ids[token]) * expert_stride + size_t(row) * row_stride;
    x += token * (k / 32);
    float sum = 0.0f;
    for (int b = 0; b < k / 32; ++b) {
        int dot = 0;
        float d;
        if constexpr (q4) {
            const auto & block = reinterpret_cast<const block_q4_0 *>(w)[b];
            d = __half2float(block.d);
            for (int j = 0; j < 16; ++j) {
                dot += (int(block.qs[j] & 15) - 8) * int(x[b].qs[j]);
                dot += (int(block.qs[j] >> 4) - 8) * int(x[b].qs[j + 16]);
            }
        } else {
            const auto & block = reinterpret_cast<const block_q8_0 *>(w)[b];
            d = __half2float(block.d);
            for (int j = 0; j < 32; ++j) { dot += int(block.qs[j]) * int(x[b].qs[j]); }
        }
        const float value = q4 ? __fmul_rn(__fmul_rn(float(dot), d), __half2float(x[b].d)) :
                __fmul_rn(float(dot), __fmul_rn(d, __half2float(x[b].d)));
        sum = __fadd_rn(sum, value);
    }
    out[token * n + row] = sum;
}

static bool moe_precise_supported(const ggml_tensor * dst) {
    return dst->op_params[0] == GGML_PREC_F32 &&
        (dst->src[0]->type == GGML_TYPE_Q4_0 || dst->src[0]->type == GGML_TYPE_Q8_0) &&
        dst->src[1]->type == GGML_TYPE_F32 && dst->src[1]->ne[1] == 1 && dst->src[2]->ne[0] == 1 &&
        dst->src[0]->ne[3] == 1 && dst->src[1]->ne[3] == 1 && dst->ne[3] == 1 &&
        ggml_is_contiguous(dst->src[1]) && ggml_is_contiguous(dst->src[2]) && ggml_is_contiguous(dst);
}

static void moe_precise(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * w = dst->src[0];
    const auto * x = dst->src[1];
    const int k = int(w->ne[0]), n = int(w->ne[1]), rows = int(x->ne[2]);
    ggml_cuda_pool_alloc<block_q8_0> quantized(ctx.pool(), size_t(rows) * (k / 32));
    moe_quantize_q8_0_precise<<<(rows * (k / 32) + 127) / 128, 128, 0, ctx.stream()>>>(
            static_cast<const float *>(x->data), quantized.get(), k, rows);
    const dim3 grid((n + 127) / 128, rows);
    if (w->type == GGML_TYPE_Q4_0) {
        moe_dot_precise<true><<<grid, 128, 0, ctx.stream()>>>(static_cast<const char *>(w->data), quantized.get(),
                static_cast<const int32_t *>(dst->src[2]->data), static_cast<float *>(dst->data), k, n, rows, w->nb[2], w->nb[1]);
    } else {
        moe_dot_precise<false><<<grid, 128, 0, ctx.stream()>>>(static_cast<const char *>(w->data), quantized.get(),
                static_cast<const int32_t *>(dst->src[2]->data), static_cast<float *>(dst->data), k, n, rows, w->nb[2], w->nb[1]);
    }
}
