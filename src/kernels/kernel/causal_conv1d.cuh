#pragma once

// qus::kernels - causal_conv1d kernel: depthwise causal k=4 with fused SiLU.
// SiLU is computed as x / (1 + exp(-x)) in fp32, with no polynomial approximation.

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace qus::kernels {

__device__ __forceinline__ float causal_conv1d_silu_f32(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ void causal_conv1d_acc_pair(__nv_bfloat162 w, __nv_bfloat162 x,
                                                       float& acc0, float& acc1) {
    acc0 += __low2float(w) * __low2float(x);
    acc1 += __high2float(w) * __high2float(x);
}

__global__ void causal_conv1d_prefill_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                             const __nv_bfloat16* conv_state, __nv_bfloat16* out,
                                             std::int32_t C, std::int32_t T) {
    const std::int64_t C64      = static_cast<std::int64_t>(C);
    const std::int64_t c_blocks = (C64 + static_cast<std::int64_t>(blockDim.x) - 1) / blockDim.x;
    const std::int64_t block    = static_cast<std::int64_t>(blockIdx.x);
    const std::int32_t t        = static_cast<std::int32_t>(block / c_blocks);
    const std::int64_t c_base   = (block - static_cast<std::int64_t>(t) * c_blocks) * blockDim.x;
    const std::int64_t c64      = c_base + threadIdx.x;
    if (t >= T || c64 >= C64) { return; }

    const std::int32_t c       = static_cast<std::int32_t>(c64);
    const std::int64_t out_idx = static_cast<std::int64_t>(t) * C64 + c64;
    const __nv_bfloat16 x0     = (t >= 3) ? x[static_cast<std::int64_t>(t - 3) * C64 + c64]
                                          : conv_state[static_cast<std::int64_t>(t) * C64 + c64];
    const __nv_bfloat16 x1     = (t >= 2) ? x[static_cast<std::int64_t>(t - 2) * C64 + c64]
                                          : conv_state[static_cast<std::int64_t>(t + 1) * C64 + c64];
    const __nv_bfloat16 x2     = (t >= 1) ? x[static_cast<std::int64_t>(t - 1) * C64 + c64]
                                          : conv_state[static_cast<std::int64_t>(t + 2) * C64 + c64];
    const __nv_bfloat16 x3     = x[out_idx];

    float acc = 0.0f;
    acc += __bfloat162float(weight[c]) * __bfloat162float(x0);
    acc += __bfloat162float(weight[C64 + c]) * __bfloat162float(x1);
    acc += __bfloat162float(weight[2 * C64 + c]) * __bfloat162float(x2);
    acc += __bfloat162float(weight[3 * C64 + c]) * __bfloat162float(x3);
    out[out_idx] = __float2bfloat16_rn(causal_conv1d_silu_f32(acc));
}

__global__ void causal_conv1d_prefill_pairs_kernel(const __nv_bfloat16* x,
                                                   const __nv_bfloat16* weight,
                                                   const __nv_bfloat16* conv_state,
                                                   __nv_bfloat16* out, std::int32_t C,
                                                   std::int32_t T) {
    const std::int64_t C2        = static_cast<std::int64_t>(C / 2);
    const std::int64_t c_blocks  = (C2 + static_cast<std::int64_t>(blockDim.x) - 1) / blockDim.x;
    const std::int64_t block     = static_cast<std::int64_t>(blockIdx.x);
    const std::int32_t t         = static_cast<std::int32_t>(block / c_blocks);
    const std::int64_t pair_base = (block - static_cast<std::int64_t>(t) * c_blocks) * blockDim.x;
    const std::int64_t p         = pair_base + threadIdx.x;
    if (t >= T || p >= C2) { return; }

    const auto* x2      = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* weight2 = reinterpret_cast<const __nv_bfloat162*>(weight);
    const auto* state2  = reinterpret_cast<const __nv_bfloat162*>(conv_state);
    auto* out2          = reinterpret_cast<__nv_bfloat162*>(out);

    const std::int64_t out_idx = static_cast<std::int64_t>(t) * C2 + p;
    const __nv_bfloat162 x0    = (t >= 3) ? x2[static_cast<std::int64_t>(t - 3) * C2 + p]
                                          : state2[static_cast<std::int64_t>(t) * C2 + p];
    const __nv_bfloat162 x1    = (t >= 2) ? x2[static_cast<std::int64_t>(t - 2) * C2 + p]
                                          : state2[static_cast<std::int64_t>(t + 1) * C2 + p];
    const __nv_bfloat162 x2v   = (t >= 1) ? x2[static_cast<std::int64_t>(t - 1) * C2 + p]
                                          : state2[static_cast<std::int64_t>(t + 2) * C2 + p];
    const __nv_bfloat162 x3    = x2[out_idx];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    causal_conv1d_acc_pair(weight2[p], x0, acc0, acc1);
    causal_conv1d_acc_pair(weight2[C2 + p], x1, acc0, acc1);
    causal_conv1d_acc_pair(weight2[2 * C2 + p], x2v, acc0, acc1);
    causal_conv1d_acc_pair(weight2[3 * C2 + p], x3, acc0, acc1);
    out2[out_idx] =
        __floats2bfloat162_rn(causal_conv1d_silu_f32(acc0), causal_conv1d_silu_f32(acc1));
}

// Writes the trailing width-3 conv window after consuming the T input columns.
// The initial window is read from `conv_state_in` and the new window is written
// to `conv_state_out`; passing the same pointer for both is the in-place form.
// Separating them lets prefix-append prefill read the committed state from a
// selected GDN snapshot slot while always publishing the running state to slot 0.
// The initial values are loaded into registers before any store, so overlapping
// in/out (in == out) has no read/write hazard.
__global__ void causal_conv1d_prefill_state_kernel(const __nv_bfloat16* x,
                                                   const __nv_bfloat16* conv_state_in,
                                                   __nv_bfloat16* conv_state_out, std::int32_t C,
                                                   std::int32_t T) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t C64    = static_cast<std::int64_t>(C);

    for (std::int64_t c64 = start; c64 < C64; c64 += stride) {
        const std::int32_t c     = static_cast<std::int32_t>(c64);
        const __nv_bfloat16 old0 = conv_state_in[c];
        const __nv_bfloat16 old1 = conv_state_in[C64 + c];
        const __nv_bfloat16 old2 = conv_state_in[2 * C64 + c];

        for (std::int32_t s = 0; s < 3; ++s) {
            const std::int32_t seq_pos = T + s;
            __nv_bfloat16 v;
            if (seq_pos == 0) {
                v = old0;
            } else if (seq_pos == 1) {
                v = old1;
            } else if (seq_pos == 2) {
                v = old2;
            } else {
                v = x[static_cast<std::int64_t>(seq_pos - 3) * C64 + c];
            }
            conv_state_out[static_cast<std::int64_t>(s) * C64 + c] = v;
        }
    }
}

__global__ void causal_conv1d_decode_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                            __nv_bfloat16* conv_state, __nv_bfloat16* out,
                                            std::int32_t C) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t C64    = static_cast<std::int64_t>(C);

    for (std::int64_t c64 = start; c64 < C64; c64 += stride) {
        const std::int32_t c   = static_cast<std::int32_t>(c64);
        const __nv_bfloat16 s0 = conv_state[c];
        const __nv_bfloat16 s1 = conv_state[C64 + c];
        const __nv_bfloat16 s2 = conv_state[2 * C64 + c];
        const __nv_bfloat16 x0 = x[c];

        float acc = 0.0f;
        acc += __bfloat162float(weight[c]) * __bfloat162float(s0);
        acc += __bfloat162float(weight[C64 + c]) * __bfloat162float(s1);
        acc += __bfloat162float(weight[2 * C64 + c]) * __bfloat162float(s2);
        acc += __bfloat162float(weight[3 * C64 + c]) * __bfloat162float(x0);

        out[c]                  = __float2bfloat16_rn(causal_conv1d_silu_f32(acc));
        conv_state[c]           = s1;
        conv_state[C64 + c]     = s2;
        conv_state[2 * C64 + c] = x0;
    }
}

__global__ void causal_conv1d_sequence_snapshot_kernel(const __nv_bfloat16* x,
                                                       const __nv_bfloat16* weight,
                                                       __nv_bfloat16* conv_states,
                                                       const std::int32_t* initial_slot,
                                                       __nv_bfloat16* out, std::int32_t C,
                                                       std::int32_t T, std::int32_t slots,
                                                       std::int64_t slot_stride) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t C64    = static_cast<std::int64_t>(C);
    const std::int32_t slot = *initial_slot;
    const std::int32_t safe_slot = (slot >= 0 && slot < slots) ? slot : 0;
    const __nv_bfloat16* init =
        conv_states + static_cast<std::int64_t>(safe_slot) * slot_stride;

    for (std::int64_t c64 = start; c64 < C64; c64 += stride) {
        const std::int32_t c = static_cast<std::int32_t>(c64);
        __nv_bfloat16 s0     = init[c64];
        __nv_bfloat16 s1     = init[C64 + c64];
        __nv_bfloat16 s2     = init[2 * C64 + c64];

        for (std::int32_t t = 0; t < T; ++t) {
            const std::int64_t out_idx = static_cast<std::int64_t>(t) * C64 + c64;
            const __nv_bfloat16 x0     = x[out_idx];

            float acc = 0.0f;
            acc += __bfloat162float(weight[c]) * __bfloat162float(s0);
            acc += __bfloat162float(weight[C64 + c]) * __bfloat162float(s1);
            acc += __bfloat162float(weight[2 * C64 + c]) * __bfloat162float(s2);
            acc += __bfloat162float(weight[3 * C64 + c]) * __bfloat162float(x0);

            out[out_idx] = __float2bfloat16_rn(causal_conv1d_silu_f32(acc));
            s0           = s1;
            s1           = s2;
            s2           = x0;

            __nv_bfloat16* slot = conv_states + static_cast<std::int64_t>(t) * slot_stride;
            slot[c64]           = s0;
            slot[C64 + c64]     = s1;
            slot[2 * C64 + c64] = s2;
        }
    }
}

} // namespace qus::kernels
