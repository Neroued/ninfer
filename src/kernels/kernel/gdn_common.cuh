#pragma once

// GDN-specific head mapping and shared-memory layouts. Generic CUDA primitives
// live under kernels/common and are included only where this header uses them.

#include "kernels/common/math.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef __CUDACC__
#    include "kernels/common/math.cuh"
#    include "kernels/common/memory.cuh"
#    include "kernels/common/warp.cuh"
#    include <cuda_bf16.h>
#    define QUS_KERNELS_HOST_DEVICE __host__ __device__
#else
#    define QUS_KERNELS_HOST_DEVICE
#endif

namespace qus::kernels {

inline uint3 init_fastdiv_values(std::uint64_t d64) {
    if (d64 == 0 || d64 > static_cast<std::uint64_t>(0xffffffffu)) {
        std::fprintf(stderr, "qus::kernels::init_fastdiv_values: invalid divisor %llu\n",
                     static_cast<unsigned long long>(d64));
        std::abort();
    }

    const auto d    = static_cast<std::uint32_t>(d64);
    std::uint32_t L = 0;
    while (L < 32 && (static_cast<std::uint32_t>(1) << L) < d) { ++L; }
    const auto mp = static_cast<std::uint32_t>(
        ((static_cast<std::uint64_t>(1) << 32) * ((static_cast<std::uint64_t>(1) << L) - d)) / d +
        1);
    return make_uint3(mp, L, d);
}

inline bool is_supported_gdn_head_dim(std::int64_t S) {
    return S == 16 || S == 32 || S == 64 || S == 128;
}

inline bool are_gdn_head_counts_valid(std::int64_t H_qk, std::int64_t H_v) {
    return H_qk > 0 && H_v >= H_qk && (H_v % H_qk) == 0;
}

#ifdef __CUDACC__

static __device__ __forceinline__ std::uint32_t fastdiv(std::uint32_t n, uint3 fastdiv_values) {
    const std::uint32_t hi = __umulhi(n, fastdiv_values.x);
    return (hi + n) >> fastdiv_values.y;
}

template <int STRIDE>
struct SmemTile {
    float* __restrict__ base;
    static_assert(STRIDE == 16 || STRIDE >= 32,
                  "SmemTile: only STRIDE in {16, 32, 64, 128, ...} supported");

    __device__ __forceinline__ int swz_xor(int row) const {
        if constexpr (STRIDE >= 32) {
            return ((row & 3) << 3) | (row & 4);
        } else {
            return ((row >> 1) & 3) << 2;
        }
    }

    __device__ __forceinline__ float& at(int row, int col) const {
        return base[row * STRIDE + (col ^ swz_xor(row))];
    }

    __device__ __forceinline__ float4& vec4_at(int row, int col) const {
        return *reinterpret_cast<float4*>(&base[row * STRIDE + (col ^ swz_xor(row))]);
    }
};

template <int ROWS, int STRIDE, int THREADS, class View>
static __device__ __forceinline__ void
issue_async_load_vec4(View view, const float* __restrict__ gmem_base_row0,
                      std::int64_t gmem_row_stride_floats, int cl, int tid) {
    static_assert(STRIDE % 4 == 0, "issue_async_load_vec4: STRIDE must be a multiple of 4");
    constexpr int VEC_PER_ROW = STRIDE / 4;
    constexpr int N_VEC       = ROWS * VEC_PER_ROW;
#    pragma unroll
    for (int v = tid; v < N_VEC; v += THREADS) {
        const int row       = v / VEC_PER_ROW;
        const int col4      = v - row * VEC_PER_ROW;
        float* smem_ptr     = reinterpret_cast<float*>(&view.vec4_at(row, col4 * 4));
        const bool in_range = row < cl;
        if (in_range) {
            const float* gmem_ptr =
                gmem_base_row0 + static_cast<std::int64_t>(row) * gmem_row_stride_floats + col4 * 4;
            cp_async<16>(smem_ptr, gmem_ptr);
        } else {
            store_vec(smem_ptr, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        }
    }
}

static __device__ __forceinline__ float4 load_bf16_vec4_as_float4(
    const __nv_bfloat16* __restrict__ src) {
    const auto* src2 = reinterpret_cast<const __nv_bfloat162*>(src);
    const float2 lo  = __bfloat1622float2(src2[0]);
    const float2 hi  = __bfloat1622float2(src2[1]);
    return make_float4(lo.x, lo.y, hi.x, hi.y);
}

template <int ROWS, int STRIDE, int THREADS, class View>
static __device__ __forceinline__ void
issue_load_bf16_to_float_vec4(View view, const __nv_bfloat16* __restrict__ gmem_base_row0,
                              std::int64_t gmem_row_stride_elems, int cl, int tid) {
    static_assert(STRIDE % 4 == 0, "issue_load_bf16_to_float_vec4: STRIDE must be a multiple of 4");
    constexpr int VEC_PER_ROW = STRIDE / 4;
    constexpr int N_VEC       = ROWS * VEC_PER_ROW;
#    pragma unroll
    for (int v = tid; v < N_VEC; v += THREADS) {
        const int row       = v / VEC_PER_ROW;
        const int col4      = v - row * VEC_PER_ROW;
        float4 val          = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const bool in_range = row < cl;
        if (in_range) {
            const __nv_bfloat16* gmem_ptr =
                gmem_base_row0 + static_cast<std::int64_t>(row) * gmem_row_stride_elems + col4 * 4;
            val = load_bf16_vec4_as_float4(gmem_ptr);
        }
        view.vec4_at(row, col4 * 4) = val;
    }
}

struct mma_lane_t {
    int lane;
    int warp;
    int lane_g;
    int lane_t;

    static __device__ __forceinline__ mma_lane_t decode(int tid) {
        mma_lane_t L{};
        L.lane   = tid & (kWarpSize - 1);
        L.warp   = tid >> 5;
        L.lane_g = L.lane >> 2;
        L.lane_t = L.lane & 3;
        return L;
    }
};

struct chunk_bounds_t {
    std::int64_t cs;
    std::int64_t ce;
    int cl;

    static __device__ __forceinline__ chunk_bounds_t of(int chunk_idx, std::int64_t T, int BT) {
        chunk_bounds_t b{};
        b.cs                    = static_cast<std::int64_t>(chunk_idx) * BT;
        const std::int64_t ce64 = b.cs + BT;
        b.ce                    = (ce64 < T) ? ce64 : T;
        b.cl                    = static_cast<int>(b.ce - b.cs);
        return b;
    }
};

inline constexpr float RCP_LN2_F = 1.4426950408889634f;

#endif // __CUDACC__

struct head_map {
    int H_qk;
    int H_v;
    uint3 group_magic;

    static head_map of(int H_qk_, int H_v_) {
        const int G = H_v_ / H_qk_;
        return head_map{H_qk_, H_v_, init_fastdiv_values(static_cast<std::uint64_t>(G))};
    }

    QUS_KERNELS_HOST_DEVICE int group_size() const { return H_v / H_qk; }

    QUS_KERNELS_HOST_DEVICE int qk_head(int h_v) const {
#if defined(__CUDA_ARCH__)
        return static_cast<int>(fastdiv(static_cast<std::uint32_t>(h_v), group_magic));
#else
        return h_v / group_size();
#endif
    }

    QUS_KERNELS_HOST_DEVICE int cta_h_v(int cta_h) const { return cta_h; }
};

} // namespace qus::kernels

#undef QUS_KERNELS_HOST_DEVICE
