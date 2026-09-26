#pragma once

// Gated DeltaNet head mapping and shared-memory layouts. Generic CUDA primitives
// live under ops/common and are included only where this header uses them.

#include "ops/common/bf16_vector.cuh"
#include "ops/common/math.h"
#include "ops/linear_attention/gated_delta_net/common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef __CUDACC__
#    include "ops/common/math.cuh"
#    include "ops/common/memory.cuh"
#    include "ops/common/warp.cuh"
#    include <cuda_bf16.h>
#    define NINFER_KERNELS_HOST_DEVICE __host__ __device__
#else
#    define NINFER_KERNELS_HOST_DEVICE
#endif

namespace ninfer::ops::detail::gated_delta_net {

inline uint3 init_fastdiv_values(std::uint64_t d64) {
    if (d64 == 0 || d64 > static_cast<std::uint64_t>(0xffffffffu)) {
        std::fprintf(stderr,
                     "ninfer::ops::detail::gated_delta_net::init_fastdiv_values: "
                     "invalid divisor %llu\n",
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
issue_load_bf16_to_float_vec4(View view, const __nv_bfloat16* __restrict__ gmem_base_row0,
                              std::int64_t gmem_row_stride_elems, int tid) {
    static_assert(STRIDE % 4 == 0, "issue_load_bf16_to_float_vec4: STRIDE must be a multiple of 4");
    constexpr int VEC_PER_ROW = STRIDE / 4;
    constexpr int N_VEC       = ROWS * VEC_PER_ROW;
#    pragma unroll
    for (int v = tid; v < N_VEC; v += THREADS) {
        const int row  = v / VEC_PER_ROW;
        const int col4 = v - row * VEC_PER_ROW;
        const __nv_bfloat16* gmem_ptr =
            gmem_base_row0 + static_cast<std::int64_t>(row) * gmem_row_stride_elems + col4 * 4;
        const Bf16x4Pack packed     = load_vec<Bf16x4Pack>(gmem_ptr);
        const float2 lo             = bf16x2_to_float2(packed.pair[0]);
        const float2 hi             = bf16x2_to_float2(packed.pair[1]);
        view.vec4_at(row, col4 * 4) = make_float4(lo.x, lo.y, hi.x, hi.y);
    }
}

#endif // __CUDACC__

struct head_map {
    int H_qk;
    int H_v;
    uint3 group_magic;

    static head_map of(int H_qk_, int H_v_) {
        const int G = H_v_ / H_qk_;
        return head_map{H_qk_, H_v_, init_fastdiv_values(static_cast<std::uint64_t>(G))};
    }

    NINFER_KERNELS_HOST_DEVICE int group_size() const { return H_v / H_qk; }

    NINFER_KERNELS_HOST_DEVICE int qk_head(int h_v) const {
#if defined(__CUDA_ARCH__)
        return static_cast<int>(fastdiv(static_cast<std::uint32_t>(h_v), group_magic));
#else
        return h_v / group_size();
#endif
    }
};

} // namespace ninfer::ops::detail::gated_delta_net

#undef NINFER_KERNELS_HOST_DEVICE
