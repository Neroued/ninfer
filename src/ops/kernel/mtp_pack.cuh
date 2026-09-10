#pragma once

#include "ops/kernel/rmsnorm.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kMtpAttnRows = 14336;
inline constexpr int kMtpQRows    = 6144;
inline constexpr int kMtpKvRows   = 1024;

__global__ void mtp_pack_fc_input_kernel(const __nv_bfloat16* embedding_norm,
                                         const __nv_bfloat16* hidden_norm, __nv_bfloat16* out,
                                         std::int32_t rows) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) { return; }

    const int token             = static_cast<int>(blockIdx.y);
    const std::int64_t in_idx   = static_cast<std::int64_t>(token) * rows + row;
    const std::int64_t out_base = static_cast<std::int64_t>(token) * (2 * rows);
    out[out_base + row]         = embedding_norm[in_idx];
    out[out_base + rows + row]  = hidden_norm[in_idx];
}

// Both stem norms in one launch, written straight into the packed FC input. blockIdx.y picks
// the half: 0 is the embedding, 1 is the hidden state. A block does exactly what the standalone
// RMSNorm CTA kernel did for its row -- same block width, same pair decomposition, same reduction,
// same epilogue -- so the bytes are the bytes the two separate launches wrote, and the copy that
// used to move them into place is gone with the two launches.
template <RmsEpilogue Epilogue, int Block, int MaxPairsPerThread, bool Prefetch>
__launch_bounds__(Block) __global__
    void mtp_norm_pack_fc_input_kernel(const __nv_bfloat162* embedding,
                                       const __nv_bfloat162* embedding_weight,
                                       const __nv_bfloat162* hidden,
                                       const __nv_bfloat162* hidden_weight, __nv_bfloat162* out,
                                       std::int32_t d, std::int64_t rows, float eps) {
    static_assert(Block % kWarpSize == 0);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }
    const int half          = static_cast<int>(blockIdx.y);
    const __nv_bfloat162* x = half == 0 ? embedding : hidden;
    const __nv_bfloat162* w = half == 0 ? embedding_weight : hidden_weight;

    const int pairs             = d / 2;
    const int pairs_per_thread  = pairs / Block;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    const std::int64_t out_base =
        row * static_cast<std::int64_t>(2 * pairs) + static_cast<std::int64_t>(half) * pairs;
    __nv_bfloat162 values[MaxPairsPerThread];
    __nv_bfloat162 weights[MaxPairsPerThread];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair = static_cast<int>(threadIdx.x) + k * Block;
            values[k]      = x[row_base + pair];
            if constexpr (Prefetch) { weights[k] = w[pair]; }
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    __shared__ float warp_sums[Block / kWarpSize];
    __shared__ float inv_shared;
    const float block_sum = block_reduce_sum<Block>(sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(d) + eps); }
    __syncthreads();
    const float inv = inv_shared;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair              = static_cast<int>(threadIdx.x) + k * Block;
            const float2 xf             = __bfloat1622float2(values[k]);
            const __nv_bfloat162 w_pair = Prefetch ? weights[k] : w[pair];
            const float2 wf             = __bfloat1622float2(w_pair);
            out[out_base + pair] =
                __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, 0.0f),
                                      rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, 0.0f));
        }
    }
}

// The residual update and the norm that reads it back are one pass over the same 2048 values.
// A block adds its row exactly as residual_add does -- pairwise, FP32, round to nearest -- stores
// the result, and normalises the stored values with the same reduction the standalone RMSNorm CTA
// kernel uses, so both outputs are the bytes the two Ops wrote.
template <RmsEpilogue Epilogue, int Block, int MaxPairsPerThread, bool Prefetch>
__launch_bounds__(Block) __global__
    void mtp_residual_norm_kernel(const __nv_bfloat162* delta, __nv_bfloat162* residual,
                                  const __nv_bfloat162* weight, __nv_bfloat162* out, std::int32_t d,
                                  std::int64_t rows, float eps) {
    static_assert(Block % kWarpSize == 0);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const int pairs_per_thread  = pairs / Block;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[MaxPairsPerThread];
    __nv_bfloat162 weights[MaxPairsPerThread];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair                  = static_cast<int>(threadIdx.x) + k * Block;
            const __nv_bfloat162 delta_pair = delta[row_base + pair];
            const __nv_bfloat162 x_pair     = residual[row_base + pair];
            const float low                 = __low2float(x_pair) + __low2float(delta_pair);
            const float high                = __high2float(x_pair) + __high2float(delta_pair);
            values[k]                       = __floats2bfloat162_rn(low, high);
            residual[row_base + pair]       = values[k];
            if constexpr (Prefetch) { weights[k] = weight[pair]; }
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    __shared__ float warp_sums[Block / kWarpSize];
    __shared__ float inv_shared;
    const float block_sum = block_reduce_sum<Block>(sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(d) + eps); }
    __syncthreads();
    const float inv = inv_shared;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair              = static_cast<int>(threadIdx.x) + k * Block;
            const float2 xf             = __bfloat1622float2(values[k]);
            const __nv_bfloat162 w_pair = Prefetch ? weights[k] : weight[pair];
            const float2 wf             = __bfloat1622float2(w_pair);
            out[row_base + pair] =
                __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, 0.0f),
                                      rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, 0.0f));
        }
    }
}

__global__ void mtp_split_attn_in_kernel(const __nv_bfloat16* attn_in, __nv_bfloat16* q,
                                         __nv_bfloat16* k, __nv_bfloat16* gate, __nv_bfloat16* v,
                                         std::int32_t tokens) {
    const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t n   = static_cast<std::int64_t>(kMtpAttnRows) * tokens;
    if (idx >= n) { return; }

    const int row             = static_cast<int>(idx % kMtpAttnRows);
    const int token           = static_cast<int>(idx / kMtpAttnRows);
    const __nv_bfloat16 value = attn_in[idx];

    if (row < kMtpQRows) {
        q[static_cast<std::int64_t>(token) * kMtpQRows + row] = value;
        return;
    }
    if (row < kMtpQRows + kMtpKvRows) {
        const int local                                          = row - kMtpQRows;
        k[static_cast<std::int64_t>(token) * kMtpKvRows + local] = value;
        return;
    }
    if (row < kMtpQRows + kMtpKvRows + kMtpQRows) {
        const int local                                            = row - kMtpQRows - kMtpKvRows;
        gate[static_cast<std::int64_t>(token) * kMtpQRows + local] = value;
        return;
    }

    const int local = row - kMtpQRows - kMtpKvRows - kMtpQRows;
    v[static_cast<std::int64_t>(token) * kMtpKvRows + local] = value;
}

} // namespace ninfer::ops
