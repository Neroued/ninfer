#pragma once

// ninfer::ops - causal_conv1d kernel: depthwise causal k=4 with fused SiLU.
// SiLU is computed as x / (1 + exp(-x)) in fp32, with no polynomial approximation.

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ void causal_conv1d_acc_pair(__nv_bfloat162 w, __nv_bfloat162 x,
                                                       float& acc0, float& acc1) {
    acc0 += __low2float(w) * __low2float(x);
    acc1 += __high2float(w) * __high2float(x);
}

inline constexpr int kCausalConvChannelTile = 32;

__global__ void causal_conv1d_prefill_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                             const __nv_bfloat16* conv_state, __nv_bfloat16* out,
                                             std::int32_t C, std::int32_t T) {
    const std::int64_t C64      = static_cast<std::int64_t>(C);
    const std::int64_t c_blocks = div_up(C64, static_cast<std::int64_t>(blockDim.x));
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
    out[out_idx] = __float2bfloat16_rn(silu(acc));
}

__global__ void causal_conv1d_prefill_pairs_kernel(const __nv_bfloat16* x,
                                                   const __nv_bfloat16* weight,
                                                   const __nv_bfloat16* conv_state,
                                                   __nv_bfloat16* out, std::int32_t C,
                                                   std::int32_t T) {
    const std::int64_t C2        = static_cast<std::int64_t>(C / 2);
    const std::int64_t c_blocks  = div_up(C2, static_cast<std::int64_t>(blockDim.x));
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
    out2[out_idx] = __floats2bfloat162_rn(silu(acc0), silu(acc1));
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

// Small-T ordinary form. One thread owns a channel for the complete sequence, so the initial
// state and four weights are loaded once and the final state is published in the same launch.
// conv_state_in and conv_state_out may be identical or disjoint.
__global__ void causal_conv1d_sequence_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                              const __nv_bfloat16* conv_state_in,
                                              __nv_bfloat16* conv_state_out, __nv_bfloat16* out,
                                              std::int32_t C, std::int32_t T) {
    const std::int64_t c64 = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t C64 = static_cast<std::int64_t>(C);
    if (c64 >= C64) { return; }

    __nv_bfloat16 s0 = conv_state_in[c64];
    __nv_bfloat16 s1 = conv_state_in[C64 + c64];
    __nv_bfloat16 s2 = conv_state_in[2 * C64 + c64];
    const float w0   = __bfloat162float(weight[c64]);
    const float w1   = __bfloat162float(weight[C64 + c64]);
    const float w2   = __bfloat162float(weight[2 * C64 + c64]);
    const float w3   = __bfloat162float(weight[3 * C64 + c64]);

    for (std::int32_t t = 0; t < T; ++t) {
        const std::int64_t out_idx = static_cast<std::int64_t>(t) * C64 + c64;
        const __nv_bfloat16 x0     = x[out_idx];

        float acc = 0.0f;
        acc += w0 * __bfloat162float(s0);
        acc += w1 * __bfloat162float(s1);
        acc += w2 * __bfloat162float(s2);
        acc += w3 * __bfloat162float(x0);

        out[out_idx] = __float2bfloat16_rn(silu(acc));
        s0           = s1;
        s1           = s2;
        s2           = x0;
    }

    conv_state_out[c64]           = s0;
    conv_state_out[C64 + c64]     = s1;
    conv_state_out[2 * C64 + c64] = s2;
}

// Small-T ordinary form parallelized across both channels and tokens. Each CTA owns one channel
// tile for the full sequence. Loading history into shared memory before any state publication makes
// the exact-alias state form safe without a second kernel.
__global__ void causal_conv1d_smallt_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                            const __nv_bfloat16* conv_state_in,
                                            __nv_bfloat16* conv_state_out, __nv_bfloat16* out,
                                            std::int32_t C, std::int32_t T) {
    __shared__ __nv_bfloat16 history[3][kCausalConvChannelTile];
    __shared__ __nv_bfloat16 weights[4][kCausalConvChannelTile];

    const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t t    = static_cast<std::int32_t>(threadIdx.y);
    const std::int64_t c64  = static_cast<std::int64_t>(blockIdx.x) * kCausalConvChannelTile + lane;
    const std::int64_t C64  = static_cast<std::int64_t>(C);
    const bool valid        = c64 < C64;

    if (t == 0) {
        for (std::int32_t s = 0; s < 3; ++s) {
            history[s][lane] = valid ? conv_state_in[static_cast<std::int64_t>(s) * C64 + c64]
                                     : __float2bfloat16(0.0f);
        }
        for (std::int32_t r = 0; r < 4; ++r) {
            weights[r][lane] =
                valid ? weight[static_cast<std::int64_t>(r) * C64 + c64] : __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    if (!valid) { return; }

    const std::int32_t p0 = t - 3;
    const std::int32_t p1 = t - 2;
    const std::int32_t p2 = t - 1;
    const __nv_bfloat16 x0 =
        p0 < 0 ? history[p0 + 3][lane] : x[static_cast<std::int64_t>(p0) * C64 + c64];
    const __nv_bfloat16 x1 =
        p1 < 0 ? history[p1 + 3][lane] : x[static_cast<std::int64_t>(p1) * C64 + c64];
    const __nv_bfloat16 x2 =
        p2 < 0 ? history[p2 + 3][lane] : x[static_cast<std::int64_t>(p2) * C64 + c64];
    const std::int64_t out_idx = static_cast<std::int64_t>(t) * C64 + c64;
    const __nv_bfloat16 x3     = x[out_idx];

    float acc = 0.0f;
    acc += __bfloat162float(weights[0][lane]) * __bfloat162float(x0);
    acc += __bfloat162float(weights[1][lane]) * __bfloat162float(x1);
    acc += __bfloat162float(weights[2][lane]) * __bfloat162float(x2);
    acc += __bfloat162float(weights[3][lane]) * __bfloat162float(x3);
    out[out_idx] = __float2bfloat16_rn(silu(acc));

    if (t == 0) {
        for (std::int32_t s = 0; s < 3; ++s) {
            const std::int32_t pos = T - 3 + s;
            conv_state_out[static_cast<std::int64_t>(s) * C64 + c64] =
                pos < 0 ? history[pos + 3][lane] : x[static_cast<std::int64_t>(pos) * C64 + c64];
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

        out[c]                  = __float2bfloat16_rn(silu(acc));
        conv_state[c]           = s1;
        conv_state[C64 + c]     = s2;
        conv_state[2 * C64 + c] = x0;
    }
}

__global__ void causal_conv1d_decode_distinct_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ conv_state_in, __nv_bfloat16* __restrict__ conv_state_out,
    __nv_bfloat16* __restrict__ out, std::int32_t C) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t C64    = static_cast<std::int64_t>(C);

    for (std::int64_t c64 = start; c64 < C64; c64 += stride) {
        const std::int32_t c   = static_cast<std::int32_t>(c64);
        const __nv_bfloat16 s0 = conv_state_in[c];
        const __nv_bfloat16 s1 = conv_state_in[C64 + c];
        const __nv_bfloat16 s2 = conv_state_in[2 * C64 + c];
        const __nv_bfloat16 x0 = x[c];

        float acc = 0.0f;
        acc += __bfloat162float(weight[c]) * __bfloat162float(s0);
        acc += __bfloat162float(weight[C64 + c]) * __bfloat162float(s1);
        acc += __bfloat162float(weight[2 * C64 + c]) * __bfloat162float(s2);
        acc += __bfloat162float(weight[3 * C64 + c]) * __bfloat162float(x0);

        out[c]                      = __float2bfloat16_rn(silu(acc));
        conv_state_out[c]           = s1;
        conv_state_out[C64 + c]     = s2;
        conv_state_out[2 * C64 + c] = x0;
    }
}

__global__ void causal_conv1d_snapshot_decode_kernel(const __nv_bfloat16* __restrict__ x,
                                                     const __nv_bfloat16* __restrict__ weight,
                                                     __nv_bfloat16* __restrict__ conv_states,
                                                     const std::int32_t* __restrict__ initial_slot,
                                                     __nv_bfloat16* __restrict__ out,
                                                     std::int32_t C, std::int64_t slot_stride) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t C64    = static_cast<std::int64_t>(C);
    const __nv_bfloat16* init =
        conv_states + static_cast<std::int64_t>(*initial_slot) * slot_stride;

    for (std::int64_t c64 = start; c64 < C64; c64 += stride) {
        const std::int32_t c   = static_cast<std::int32_t>(c64);
        const __nv_bfloat16 s0 = init[c];
        const __nv_bfloat16 s1 = init[C64 + c];
        const __nv_bfloat16 s2 = init[2 * C64 + c];
        const __nv_bfloat16 x0 = x[c];

        float acc = 0.0f;
        acc += __bfloat162float(weight[c]) * __bfloat162float(s0);
        acc += __bfloat162float(weight[C64 + c]) * __bfloat162float(s1);
        acc += __bfloat162float(weight[2 * C64 + c]) * __bfloat162float(s2);
        acc += __bfloat162float(weight[3 * C64 + c]) * __bfloat162float(x0);

        out[c]                   = __float2bfloat16_rn(silu(acc));
        conv_states[c]           = s1;
        conv_states[C64 + c]     = s2;
        conv_states[2 * C64 + c] = x0;
    }
}

__global__ void causal_conv1d_sequence_snapshot_kernel(const __nv_bfloat16* x,
                                                       const __nv_bfloat16* weight,
                                                       __nv_bfloat16* conv_states,
                                                       const std::int32_t* initial_slot,
                                                       __nv_bfloat16* out, std::int32_t C,
                                                       std::int32_t T, std::int64_t slot_stride) {
    const std::int64_t c64 = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t C64 = static_cast<std::int64_t>(C);
    if (c64 >= C64) { return; }

    const std::int32_t slot   = *initial_slot;
    const __nv_bfloat16* init = conv_states + static_cast<std::int64_t>(slot) * slot_stride;
    __nv_bfloat16 s0          = init[c64];
    __nv_bfloat16 s1          = init[C64 + c64];
    __nv_bfloat16 s2          = init[2 * C64 + c64];
    const float w0            = __bfloat162float(weight[c64]);
    const float w1            = __bfloat162float(weight[C64 + c64]);
    const float w2            = __bfloat162float(weight[2 * C64 + c64]);
    const float w3            = __bfloat162float(weight[3 * C64 + c64]);

    for (std::int32_t t = 0; t < T; ++t) {
        const std::int64_t out_idx = static_cast<std::int64_t>(t) * C64 + c64;
        const __nv_bfloat16 x0     = x[out_idx];

        float acc = 0.0f;
        acc += w0 * __bfloat162float(s0);
        acc += w1 * __bfloat162float(s1);
        acc += w2 * __bfloat162float(s2);
        acc += w3 * __bfloat162float(x0);

        out[out_idx] = __float2bfloat16_rn(silu(acc));
        s0           = s1;
        s1           = s2;
        s2           = x0;

        __nv_bfloat16* snapshot = conv_states + static_cast<std::int64_t>(t) * slot_stride;
        snapshot[c64]           = s0;
        snapshot[C64 + c64]     = s1;
        snapshot[2 * C64 + c64] = s2;
    }
}

// Small-T snapshot form. A CTA owns all token outputs for one channel tile. The selected history
// is cached before any snapshot slot is written, so initial_slot may name any slot, including one
// overwritten by this call.
__global__ void causal_conv1d_snapshot_smallt_kernel(const __nv_bfloat16* x,
                                                     const __nv_bfloat16* weight,
                                                     __nv_bfloat16* conv_states,
                                                     const std::int32_t* initial_slot,
                                                     __nv_bfloat16* out, std::int32_t C,
                                                     std::int32_t T, std::int64_t slot_stride) {
    __shared__ __nv_bfloat16 history[3][kCausalConvChannelTile];
    __shared__ __nv_bfloat16 weights[4][kCausalConvChannelTile];

    const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t t    = static_cast<std::int32_t>(threadIdx.y);
    const std::int64_t c64  = static_cast<std::int64_t>(blockIdx.x) * kCausalConvChannelTile + lane;
    const std::int64_t C64  = static_cast<std::int64_t>(C);
    const bool valid        = c64 < C64;

    if (t == 0) {
        const std::int32_t slot   = *initial_slot;
        const __nv_bfloat16* init = conv_states + static_cast<std::int64_t>(slot) * slot_stride;
        for (std::int32_t s = 0; s < 3; ++s) {
            history[s][lane] =
                valid ? init[static_cast<std::int64_t>(s) * C64 + c64] : __float2bfloat16(0.0f);
        }
        for (std::int32_t r = 0; r < 4; ++r) {
            weights[r][lane] =
                valid ? weight[static_cast<std::int64_t>(r) * C64 + c64] : __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    if (!valid) { return; }

    const std::int32_t p0 = t - 3;
    const std::int32_t p1 = t - 2;
    const std::int32_t p2 = t - 1;
    const __nv_bfloat16 x0 =
        p0 < 0 ? history[p0 + 3][lane] : x[static_cast<std::int64_t>(p0) * C64 + c64];
    const __nv_bfloat16 x1 =
        p1 < 0 ? history[p1 + 3][lane] : x[static_cast<std::int64_t>(p1) * C64 + c64];
    const __nv_bfloat16 x2 =
        p2 < 0 ? history[p2 + 3][lane] : x[static_cast<std::int64_t>(p2) * C64 + c64];
    const std::int64_t out_idx = static_cast<std::int64_t>(t) * C64 + c64;
    const __nv_bfloat16 x3     = x[out_idx];

    float acc = 0.0f;
    acc += __bfloat162float(weights[0][lane]) * __bfloat162float(x0);
    acc += __bfloat162float(weights[1][lane]) * __bfloat162float(x1);
    acc += __bfloat162float(weights[2][lane]) * __bfloat162float(x2);
    acc += __bfloat162float(weights[3][lane]) * __bfloat162float(x3);
    out[out_idx] = __float2bfloat16_rn(silu(acc));

    __nv_bfloat16* snapshot = conv_states + static_cast<std::int64_t>(t) * slot_stride;
    for (std::int32_t s = 0; s < 3; ++s) {
        const std::int32_t pos = t - 2 + s;
        snapshot[static_cast<std::int64_t>(s) * C64 + c64] =
            pos < 0 ? history[pos + 3][lane] : x[static_cast<std::int64_t>(pos) * C64 + c64];
    }
}

} // namespace ninfer::ops
