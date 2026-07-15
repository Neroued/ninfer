#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kMtpHiddenRows = 5120;
inline constexpr int kMtpFcRows     = 10240;
inline constexpr int kMtpAttnRows   = 14336;
inline constexpr int kMtpQRows      = 6144;
inline constexpr int kMtpKvRows     = 1024;

__global__ void mtp_pack_fc_input_kernel(const __nv_bfloat16* embedding_norm,
                                         const __nv_bfloat16* hidden_norm, __nv_bfloat16* out,
                                         std::int32_t tokens) {
    const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t n   = static_cast<std::int64_t>(kMtpHiddenRows) * tokens;
    if (idx >= n) { return; }

    const int row                        = static_cast<int>(idx % kMtpHiddenRows);
    const int token                      = static_cast<int>(idx / kMtpHiddenRows);
    const std::int64_t out_base          = static_cast<std::int64_t>(token) * kMtpFcRows;
    out[out_base + row]                  = embedding_norm[idx];
    out[out_base + kMtpHiddenRows + row] = hidden_norm[idx];
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
