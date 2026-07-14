#pragma once

#include "kernels/kernel/gdn_common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#define NINFER_GDN_PROPAGATE(expr)                                                                    \
    do {                                                                                           \
        const cudaError_t ninfer_gdn_err = (expr);                                                    \
        if (ninfer_gdn_err != cudaSuccess) { return ninfer_gdn_err; }                                    \
    } while (0)

namespace ninfer::kernels::detail::gdn_chunked {

inline constexpr std::int64_t kChunkSize      = 64;
inline constexpr std::int64_t kSubChunkSize   = 16;
inline constexpr std::int64_t kWorkspaceAlign = 256;

static_assert(kChunkSize % kSubChunkSize == 0, "kChunkSize must be a multiple of kSubChunkSize");

struct workspace_layout {
    std::int64_t g_cumsum_off   = 0;
    std::int64_t g_cumsum_bytes = 0;
    std::int64_t W_off          = 0;
    std::int64_t W_bytes        = 0;
    std::int64_t U_off          = 0;
    std::int64_t U_bytes        = 0;
    std::int64_t v_new_off      = 0;
    std::int64_t v_new_bytes    = 0;
    std::int64_t h_chunk_off    = 0;
    std::int64_t h_chunk_bytes  = 0;
    std::int64_t total_bytes    = 0;
};

inline std::int64_t reserve(std::int64_t& cursor, std::int64_t bytes) {
    if (bytes == 0) { return cursor; }
    const std::int64_t off = cursor;
    cursor                 = ninfer::kernels::align_up<kWorkspaceAlign>(off + bytes);
    return off;
}

inline workspace_layout compute_workspace_layout(std::int64_t S, std::int64_t H_qk,
                                                 std::int64_t H_v, std::int64_t L, std::int64_t B) {
    (void)H_qk;
    constexpr std::int64_t f    = static_cast<std::int64_t>(sizeof(float));
    constexpr std::int64_t bf16 = static_cast<std::int64_t>(sizeof(__nv_bfloat16));
    const std::int64_t T     = L;
    const std::int64_t NT    = div_up(T, kChunkSize);

    const std::int64_t per_token_g   = B * T * H_v;
    const std::int64_t per_token_S   = B * T * H_v * S;
    const std::int64_t per_chunk_SxS = B * NT * H_v * S * S;

    workspace_layout w{};
    w.g_cumsum_bytes = per_token_g * f;
    w.W_bytes        = per_token_S * bf16;
    w.U_bytes        = per_token_S * bf16;
    w.v_new_bytes    = per_token_S * bf16;
    w.h_chunk_bytes  = per_chunk_SxS * bf16;

    std::int64_t cur = 0;
    w.g_cumsum_off   = reserve(cur, w.g_cumsum_bytes);
    w.W_off          = reserve(cur, w.W_bytes);
    w.U_off          = reserve(cur, w.U_bytes);
    w.v_new_off      = reserve(cur, w.v_new_bytes);
    w.h_chunk_off    = reserve(cur, w.h_chunk_bytes);
    w.total_bytes    = cur;
    return w;
}

inline std::int64_t workspace_bytes(std::int64_t S, std::int64_t H_qk, std::int64_t H_v,
                                    std::int64_t L, std::int64_t B) {
    return compute_workspace_layout(S, H_qk, H_v, L, B).total_bytes;
}

struct prepare_wy_wu_config {
    std::int64_t S    = 0;
    std::int64_t H_qk = 0;
    std::int64_t H_v  = 0;
    std::int64_t L    = 0;
    std::int64_t B    = 0;

    const __nv_bfloat16* k = nullptr;
    const __nv_bfloat16* v = nullptr;
    const float* g_in = nullptr;
    const float* beta = nullptr;

    __nv_bfloat16* W   = nullptr;
    __nv_bfloat16* U   = nullptr;
    float* g_cumsum_out = nullptr;

    std::int64_t k_stride_t_floats = 0;
    std::int64_t v_stride_t_floats = 0;

    cudaStream_t stream = nullptr;
};

struct state_passing_config {
    std::int64_t S    = 0;
    std::int64_t H_qk = 0;
    std::int64_t H_v  = 0;
    std::int64_t L    = 0;
    std::int64_t B    = 0;

    const __nv_bfloat16* W = nullptr;
    const __nv_bfloat16* U = nullptr;
    const __nv_bfloat16* k = nullptr;
    const float* g_cumsum = nullptr;
    const float* state_in = nullptr;

    __nv_bfloat16* v_new = nullptr;
    __nv_bfloat16* h_chunk = nullptr;
    float* state_out = nullptr;

    std::int64_t k_stride_t_floats = 0;

    cudaStream_t stream = nullptr;
};

struct chunk_output_config {
    std::int64_t S    = 0;
    std::int64_t H_qk = 0;
    std::int64_t H_v  = 0;
    std::int64_t L    = 0;
    std::int64_t B    = 0;

    const __nv_bfloat16* q = nullptr;
    const __nv_bfloat16* k = nullptr;
    const __nv_bfloat16* v_new = nullptr;
    const float* g_cumsum = nullptr;
    const __nv_bfloat16* h_chunk = nullptr;

    __nv_bfloat16* attn_out = nullptr;

    std::int64_t q_stride_t_floats = 0;
    std::int64_t k_stride_t_floats = 0;
    float scale                    = 0.0f;

    cudaStream_t stream = nullptr;
};

struct bh_decode_t {
    int b;
    int h_v;

    static __device__ __forceinline__ bh_decode_t of(int bh, int H_v) {
        bh_decode_t r{};
        r.b   = bh / H_v;
        r.h_v = bh - r.b * H_v;
        return r;
    }

    static __device__ __forceinline__ bh_decode_t of(int bh, ninfer::kernels::head_map qk_map) {
        bh_decode_t r{};
        const int H_v   = qk_map.H_v;
        r.b             = bh / H_v;
        const int cta_h = bh - r.b * H_v;
        r.h_v           = qk_map.cta_h_v(cta_h);
        return r;
    }
};

template <int TILES, int N>
__device__ __forceinline__ void zero_frag(float (&frag)[TILES][N]) {
#pragma unroll
    for (int t = 0; t < TILES; ++t) {
#pragma unroll
        for (int e = 0; e < N; ++e) { frag[t][e] = 0.0f; }
    }
}

inline constexpr int BT    = static_cast<int>(kChunkSize);
inline constexpr int BC    = static_cast<int>(kSubChunkSize);
inline constexpr int MMA_M = 16;
inline constexpr int MMA_N = 8;
inline constexpr int MMA_K = 8;

static_assert(BT % BC == 0, "BT must be a multiple of BC");
static_assert(BT % MMA_M == 0, "BT must be a multiple of MMA_M");

struct stage_validator {
    const char* name;
    std::int64_t S;
    std::int64_t H_qk;
    std::int64_t H_v;
    std::int64_t T;
    std::int64_t B;
    bool require_h_qk = true;

    cudaError_t check_shape() const {
        const bool bad_shape =
            S <= 0 || H_v <= 0 || T <= 0 || B <= 0 || (require_h_qk && H_qk <= 0);
        if (bad_shape) {
            std::fprintf(stderr, "%s: invalid shape (S=%lld H_qk=%lld H_v=%lld T=%lld B=%lld)\n",
                         name, static_cast<long long>(S), static_cast<long long>(H_qk),
                         static_cast<long long>(H_v), static_cast<long long>(T),
                         static_cast<long long>(B));
            return cudaErrorInvalidValue;
        }
        if (!ninfer::kernels::is_supported_gdn_head_dim(S)) {
            std::fprintf(stderr, "%s: unsupported S=%lld (allowed: 16, 32, 64, 128)\n", name,
                         static_cast<long long>(S));
            return cudaErrorInvalidValue;
        }
        if (require_h_qk && !ninfer::kernels::are_gdn_head_counts_valid(H_qk, H_v)) {
            std::fprintf(stderr,
                         "%s: invalid head counts H_qk=%lld H_v=%lld "
                         "(need H_qk >= 1, H_v >= H_qk, H_v %% H_qk == 0)\n",
                         name, static_cast<long long>(H_qk), static_cast<long long>(H_v));
            return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }

    cudaError_t check_gdn_full_chunks() const {
        if ((T % BT) != 0) {
            std::fprintf(stderr,
                         "%s: GDN chunked path requires T to be a multiple of %d; "
                         "route tail tokens through AR instead (T=%lld)\n",
                         name, BT, static_cast<long long>(T));
            return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }

    cudaError_t check_grid(std::int64_t grid_x, std::int64_t grid_y,
                           std::int64_t grid_z = 1) const {
        if (grid_x > static_cast<std::int64_t>(0xffffffff)) {
            std::fprintf(stderr, "%s: grid.x too large (%lld)\n", name,
                         static_cast<long long>(grid_x));
            return cudaErrorInvalidConfiguration;
        }
        if (grid_y > static_cast<std::int64_t>(0xffff)) {
            std::fprintf(stderr, "%s: grid.y too large (%lld)\n", name,
                         static_cast<long long>(grid_y));
            return cudaErrorInvalidConfiguration;
        }
        if (grid_z > static_cast<std::int64_t>(0xffff)) {
            std::fprintf(stderr, "%s: grid.z too large (%lld)\n", name,
                         static_cast<long long>(grid_z));
            return cudaErrorInvalidConfiguration;
        }
        return cudaSuccess;
    }
};

} // namespace ninfer::kernels::detail::gdn_chunked

namespace ninfer::kernels::detail::gdn_prepare_wy_wu {
cudaError_t launch_prepare_wy_wu(const gdn_chunked::prepare_wy_wu_config& cfg);
} // namespace ninfer::kernels::detail::gdn_prepare_wy_wu

namespace ninfer::kernels::detail::gdn_state_passing {
cudaError_t launch_state_passing(const gdn_chunked::state_passing_config& cfg);
} // namespace ninfer::kernels::detail::gdn_state_passing

namespace ninfer::kernels::detail::gdn_chunk_output {
cudaError_t launch_chunk_output(const gdn_chunked::chunk_output_config& cfg);
} // namespace ninfer::kernels::detail::gdn_chunk_output
