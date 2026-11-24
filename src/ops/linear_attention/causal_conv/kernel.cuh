#pragma once

#include "ops/linear_attention/kernels/causal_conv.cuh"

namespace ninfer::ops::detail {
inline constexpr int kCausalConvChannelTile = 32;

template <class Output>
__global__ void causal_conv1d_prefill_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                             const __nv_bfloat16* conv_state, Output out,
                                             std::int32_t C, std::int32_t T) {
    const std::int64_t C64      = static_cast<std::int64_t>(C);
    const std::int64_t c_blocks = common::div_up(C64, static_cast<std::int64_t>(blockDim.x));
    const std::int64_t block    = static_cast<std::int64_t>(blockIdx.x);
    const std::int32_t t        = static_cast<std::int32_t>(block / c_blocks);
    const std::int64_t c_base   = (block - static_cast<std::int64_t>(t) * c_blocks) * blockDim.x;
    const std::int64_t c64      = c_base + threadIdx.x;
    if (t >= T || c64 >= C64) { return; }

    const std::int32_t c      = static_cast<std::int32_t>(c64);
    const std::int64_t in_idx = static_cast<std::int64_t>(t) * C64 + c64;
    const __nv_bfloat16 x0    = (t >= 3) ? x[static_cast<std::int64_t>(t - 3) * C64 + c64]
                                         : conv_state[static_cast<std::int64_t>(t) * C64 + c64];
    const __nv_bfloat16 x1    = (t >= 2) ? x[static_cast<std::int64_t>(t - 2) * C64 + c64]
                                         : conv_state[static_cast<std::int64_t>(t + 1) * C64 + c64];
    const __nv_bfloat16 x2    = (t >= 1) ? x[static_cast<std::int64_t>(t - 1) * C64 + c64]
                                         : conv_state[static_cast<std::int64_t>(t + 2) * C64 + c64];
    const __nv_bfloat16 x3    = x[in_idx];

    const __nv_bfloat16 result = causal_conv4_silu(
        __bfloat162float(weight[c]), __bfloat162float(weight[C64 + c]),
        __bfloat162float(weight[2 * C64 + c]), __bfloat162float(weight[3 * C64 + c]),
        __bfloat162float(x0), __bfloat162float(x1), __bfloat162float(x2), __bfloat162float(x3));
    *out.column(c64).at(t) = result;
}

template <class Output>
__global__ void causal_conv1d_prefill_pairs_kernel(const __nv_bfloat16* x,
                                                   const __nv_bfloat16* weight,
                                                   const __nv_bfloat16* conv_state, Output out,
                                                   std::int32_t C, std::int32_t T) {
    const std::int64_t C2        = static_cast<std::int64_t>(C / 2);
    const std::int64_t c_blocks  = common::div_up(C2, static_cast<std::int64_t>(blockDim.x));
    const std::int64_t block     = static_cast<std::int64_t>(blockIdx.x);
    const std::int32_t t         = static_cast<std::int32_t>(block / c_blocks);
    const std::int64_t pair_base = (block - static_cast<std::int64_t>(t) * c_blocks) * blockDim.x;
    const std::int64_t p         = pair_base + threadIdx.x;
    if (t >= T || p >= C2) { return; }

    const auto* x2      = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* weight2 = reinterpret_cast<const __nv_bfloat162*>(weight);
    const auto* state2  = reinterpret_cast<const __nv_bfloat162*>(conv_state);

    const std::int64_t in_idx = static_cast<std::int64_t>(t) * C2 + p;
    const __nv_bfloat162 x0   = (t >= 3) ? x2[static_cast<std::int64_t>(t - 3) * C2 + p]
                                         : state2[static_cast<std::int64_t>(t) * C2 + p];
    const __nv_bfloat162 x1   = (t >= 2) ? x2[static_cast<std::int64_t>(t - 2) * C2 + p]
                                         : state2[static_cast<std::int64_t>(t + 1) * C2 + p];
    const __nv_bfloat162 x2v  = (t >= 1) ? x2[static_cast<std::int64_t>(t - 1) * C2 + p]
                                         : state2[static_cast<std::int64_t>(t + 2) * C2 + p];
    const __nv_bfloat162 x3   = x2[in_idx];

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    causal_conv1d_acc_pair(weight2[p], x0, acc0, acc1);
    causal_conv1d_acc_pair(weight2[C2 + p], x1, acc0, acc1);
    causal_conv1d_acc_pair(weight2[2 * C2 + p], x2v, acc0, acc1);
    causal_conv1d_acc_pair(weight2[3 * C2 + p], x3, acc0, acc1);
    *out.column(p).at(t) = __floats2bfloat162_rn(common::silu(acc0), common::silu(acc1));
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

// Small-T ordinary form parallelized across both channels and tokens. Each CTA owns one channel
// tile for the full sequence. Loading history into shared memory before any state publication makes
// the exact-alias state form safe without a second kernel.
template <class Output>
__global__ void causal_conv1d_smallt_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                            const __nv_bfloat16* conv_state_in,
                                            __nv_bfloat16* conv_state_out, Output out,
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
    const std::int64_t x_idx = static_cast<std::int64_t>(t) * C64 + c64;
    const __nv_bfloat16 x3   = x[x_idx];

    const __nv_bfloat16 result = causal_conv4_silu(
        __bfloat162float(weights[0][lane]), __bfloat162float(weights[1][lane]),
        __bfloat162float(weights[2][lane]), __bfloat162float(weights[3][lane]),
        __bfloat162float(x0), __bfloat162float(x1), __bfloat162float(x2), __bfloat162float(x3));
    *out.column(c64).at(t) = result;

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

        const __nv_bfloat16 result = causal_conv4_silu(
            __bfloat162float(weight[c]), __bfloat162float(weight[C64 + c]),
            __bfloat162float(weight[2 * C64 + c]), __bfloat162float(weight[3 * C64 + c]),
            __bfloat162float(s0), __bfloat162float(s1), __bfloat162float(s2), __bfloat162float(x0));

        out[c]                  = result;
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

        const __nv_bfloat16 result = causal_conv4_silu(
            __bfloat162float(weight[c]), __bfloat162float(weight[C64 + c]),
            __bfloat162float(weight[2 * C64 + c]), __bfloat162float(weight[3 * C64 + c]),
            __bfloat162float(s0), __bfloat162float(s1), __bfloat162float(s2), __bfloat162float(x0));

        out[c]                      = result;
        conv_state_out[c]           = s1;
        conv_state_out[C64 + c]     = s2;
        conv_state_out[2 * C64 + c] = x0;
    }
}

__global__ void causal_conv1d_snapshot_decode_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* source, __nv_bfloat16* snapshots,
    const std::int64_t* __restrict__ source_offsets,
    const std::int32_t* __restrict__ snapshot_base_slot, __nv_bfloat16* __restrict__ out,
    std::int32_t C, std::int64_t slot_stride) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t C64    = static_cast<std::int64_t>(C);
    const __nv_bfloat16* init = source + *source_offsets / 2;

    for (std::int64_t c64 = start; c64 < C64; c64 += stride) {
        const std::int32_t c   = static_cast<std::int32_t>(c64);
        const __nv_bfloat16 s0 = init[c];
        const __nv_bfloat16 s1 = init[C64 + c];
        const __nv_bfloat16 s2 = init[2 * C64 + c];
        const __nv_bfloat16 x0 = x[c];

        const __nv_bfloat16 result = causal_conv4_silu(
            __bfloat162float(weight[c]), __bfloat162float(weight[C64 + c]),
            __bfloat162float(weight[2 * C64 + c]), __bfloat162float(weight[3 * C64 + c]),
            __bfloat162float(s0), __bfloat162float(s1), __bfloat162float(s2), __bfloat162float(x0));

        out[c] = result;
        __nv_bfloat16* snapshot =
            snapshots + static_cast<std::int64_t>(*snapshot_base_slot) * slot_stride;
        snapshot[c]           = s1;
        snapshot[C64 + c]     = s2;
        snapshot[2 * C64 + c] = x0;
    }
}

__global__ void
causal_conv1d_sequence_snapshot_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                       const __nv_bfloat16* source, __nv_bfloat16* snapshots,
                                       const std::int64_t* source_offsets,
                                       const std::int32_t* snapshot_base_slot, __nv_bfloat16* out,
                                       std::int32_t C, std::int32_t T, std::int64_t slot_stride) {
    const std::int64_t c64 = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t C64 = static_cast<std::int64_t>(C);
    if (c64 >= C64) { return; }

    const __nv_bfloat16* init = source + *source_offsets / 2;
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

        const __nv_bfloat16 result =
            causal_conv4_silu(w0, w1, w2, w3, __bfloat162float(s0), __bfloat162float(s1),
                              __bfloat162float(s2), __bfloat162float(x0));

        out[out_idx] = result;
        s0           = s1;
        s1           = s2;
        s2           = x0;

        __nv_bfloat16* snapshot =
            snapshots + static_cast<std::int64_t>(*snapshot_base_slot + t) * slot_stride;
        snapshot[c64]           = s0;
        snapshot[C64 + c64]     = s1;
        snapshot[2 * C64 + c64] = s2;
    }
}

// Batched serial-column form. One block row owns one request and one thread owns a channel, so
// request-local state is loaded before that row publishes any snapshot. It is used for W=1 and
// for masked B=1 widths outside the parallel small-W domain.
template <bool Masked>
__global__ void causal_conv1d_batched_sequence_snapshot_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* source,
    __nv_bfloat16* snapshots, const std::int32_t* valid_columns, const std::int64_t* source_offsets,
    const std::int32_t* snapshot_base_slots, __nv_bfloat16* out, std::int32_t C, std::int32_t width,
    std::int64_t slot_stride) {
    const std::int32_t batch = static_cast<std::int32_t>(blockIdx.y);
    const std::int64_t c64   = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t C64   = static_cast<std::int64_t>(C);
    if (c64 >= C64) { return; }

    const std::int32_t valid    = Masked ? valid_columns[batch] : width;
    const __nv_bfloat16* init   = source + source_offsets[batch] / 2;
    __nv_bfloat16 s0            = init[c64];
    __nv_bfloat16 s1            = init[C64 + c64];
    __nv_bfloat16 s2            = init[2 * C64 + c64];
    const float w0              = __bfloat162float(weight[c64]);
    const float w1              = __bfloat162float(weight[C64 + c64]);
    const float w2              = __bfloat162float(weight[2 * C64 + c64]);
    const float w3              = __bfloat162float(weight[3 * C64 + c64]);
    const std::int64_t row_base = static_cast<std::int64_t>(batch) * width * C64;

    for (std::int32_t column = 0; column < width; ++column) {
        const std::int64_t out_idx = row_base + static_cast<std::int64_t>(column) * C64 + c64;
        if (column >= valid) {
            out[out_idx] = __float2bfloat16(0.0f);
            continue;
        }

        const __nv_bfloat16 x0 = x[out_idx];
        const __nv_bfloat16 result =
            causal_conv4_silu(w0, w1, w2, w3, __bfloat162float(s0), __bfloat162float(s1),
                              __bfloat162float(s2), __bfloat162float(x0));
        out[out_idx] = result;
        s0           = s1;
        s1           = s2;
        s2           = x0;

        __nv_bfloat16* snapshot =
            snapshots +
            static_cast<std::int64_t>(snapshot_base_slots[batch] + column) * slot_stride;
        snapshot[c64]           = s0;
        snapshot[C64 + c64]     = s1;
        snapshot[2 * C64 + c64] = s2;
    }
}

// A CTA caches the source field before publishing any snapshot for its channel tile.
__global__ void
causal_conv1d_snapshot_smallt_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                     const __nv_bfloat16* source, __nv_bfloat16* snapshots,
                                     const std::int64_t* source_offsets,
                                     const std::int32_t* snapshot_base_slot, __nv_bfloat16* out,
                                     std::int32_t C, std::int32_t T, std::int64_t slot_stride) {
    __shared__ __nv_bfloat16 history[3][kCausalConvChannelTile];
    __shared__ __nv_bfloat16 weights[4][kCausalConvChannelTile];

    const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t t    = static_cast<std::int32_t>(threadIdx.y);
    const std::int64_t c64  = static_cast<std::int64_t>(blockIdx.x) * kCausalConvChannelTile + lane;
    const std::int64_t C64  = static_cast<std::int64_t>(C);
    const bool valid        = c64 < C64;

    if (t == 0) {
        const __nv_bfloat16* init = source + *source_offsets / 2;
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

    const __nv_bfloat16 result = causal_conv4_silu(
        __bfloat162float(weights[0][lane]), __bfloat162float(weights[1][lane]),
        __bfloat162float(weights[2][lane]), __bfloat162float(weights[3][lane]),
        __bfloat162float(x0), __bfloat162float(x1), __bfloat162float(x2), __bfloat162float(x3));
    out[out_idx] = result;

    __nv_bfloat16* snapshot =
        snapshots + static_cast<std::int64_t>(*snapshot_base_slot + t) * slot_stride;
    for (std::int32_t s = 0; s < 3; ++s) {
        const std::int32_t pos = t - 2 + s;
        snapshot[static_cast<std::int64_t>(s) * C64 + c64] =
            pos < 0 ? history[pos + 3][lane] : x[static_cast<std::int64_t>(pos) * C64 + c64];
    }
}

template <bool Masked>
__global__ void causal_conv1d_batched_snapshot_smallt_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* source,
    __nv_bfloat16* snapshots, const std::int32_t* valid_columns, const std::int64_t* source_offsets,
    const std::int32_t* snapshot_base_slots, __nv_bfloat16* out, std::int32_t C, std::int32_t width,
    std::int64_t slot_stride) {
    __shared__ __nv_bfloat16 history[3][kCausalConvChannelTile];
    __shared__ __nv_bfloat16 weights[4][kCausalConvChannelTile];

    const std::int32_t batch  = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t lane   = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t column = static_cast<std::int32_t>(threadIdx.y);
    const std::int64_t c64 = static_cast<std::int64_t>(blockIdx.x) * kCausalConvChannelTile + lane;
    const std::int64_t C64 = static_cast<std::int64_t>(C);
    const bool channel_valid = c64 < C64;

    if (column == 0) {
        const __nv_bfloat16* init = source + source_offsets[batch] / 2;
        for (std::int32_t s = 0; s < 3; ++s) {
            history[s][lane] = channel_valid ? init[static_cast<std::int64_t>(s) * C64 + c64]
                                             : __float2bfloat16(0.0f);
        }
        for (std::int32_t r = 0; r < 4; ++r) {
            weights[r][lane] = channel_valid ? weight[static_cast<std::int64_t>(r) * C64 + c64]
                                             : __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    if (!channel_valid) { return; }

    const std::int32_t valid    = Masked ? valid_columns[batch] : width;
    const std::int64_t row_base = static_cast<std::int64_t>(batch) * width * C64;
    const std::int64_t out_idx  = row_base + static_cast<std::int64_t>(column) * C64 + c64;
    if (column >= valid) {
        out[out_idx] = __float2bfloat16(0.0f);
        return;
    }

    const std::int32_t p0 = column - 3;
    const std::int32_t p1 = column - 2;
    const std::int32_t p2 = column - 1;
    const __nv_bfloat16 x0 =
        p0 < 0 ? history[p0 + 3][lane] : x[row_base + static_cast<std::int64_t>(p0) * C64 + c64];
    const __nv_bfloat16 x1 =
        p1 < 0 ? history[p1 + 3][lane] : x[row_base + static_cast<std::int64_t>(p1) * C64 + c64];
    const __nv_bfloat16 x2 =
        p2 < 0 ? history[p2 + 3][lane] : x[row_base + static_cast<std::int64_t>(p2) * C64 + c64];
    const __nv_bfloat16 x3 = x[out_idx];

    const __nv_bfloat16 result = causal_conv4_silu(
        __bfloat162float(weights[0][lane]), __bfloat162float(weights[1][lane]),
        __bfloat162float(weights[2][lane]), __bfloat162float(weights[3][lane]),
        __bfloat162float(x0), __bfloat162float(x1), __bfloat162float(x2), __bfloat162float(x3));
    out[out_idx] = result;

    __nv_bfloat16* snapshot =
        snapshots + static_cast<std::int64_t>(snapshot_base_slots[batch] + column) * slot_stride;
    for (std::int32_t s = 0; s < 3; ++s) {
        const std::int32_t pos = column - 2 + s;
        snapshot[static_cast<std::int64_t>(s) * C64 + c64] =
            pos < 0 ? history[pos + 3][lane]
                    : x[row_base + static_cast<std::int64_t>(pos) * C64 + c64];
    }
}

} // namespace ninfer::ops::detail
