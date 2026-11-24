#pragma once

#include "ops/moe/kernels/cuda/block_scaled/fp8/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/output.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/quant/fp8.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled::fp8 {
using common::load_vec;
using common::warp_reduce_sum;
using common::warp_max;
using common::silu;
using common::bf16x2_bits_to_float2;
using common::cp_async;
using common::cp_async_zfill;
using common::cp_commit;
using common::cp_wait;
using common::smem_addr;
using common::mma_fp8_e4m3;
using common::ldmatrix_x2;
using common::ldmatrix_x4;
using common::fp8_code2_to_float2;
using common::fp8_fp32_scale;
using common::fp8_quant_fp32_pair;

struct Matrix {
    const std::uint8_t* codes;
    const __nv_bfloat16* scales;
};

__device__ __forceinline__ Matrix resolve(const ExpertMatrixDeviceView& matrix, int expert) {
    const auto offset = matrix.offsets ? matrix.offsets[expert] : 0;
    return {reinterpret_cast<const std::uint8_t*>(matrix.codes) +
                (matrix.offsets ? offset : expert * matrix.code_stride),
            reinterpret_cast<const __nv_bfloat16*>(
                matrix.scales + (matrix.offsets ? offset : expert * matrix.scale_stride))};
}

template <int Columns, bool Gate>
struct Schedule {
    static constexpr int columns = Columns, bk = 128;
    static constexpr int warp_rows    = Gate ? 8 : 4;
    static constexpr int warp_groups  = Columns >= 64 ? 2 : 1;
    static constexpr int threads      = warp_rows * warp_groups * 32;
    static constexpr int warp_columns = Columns / warp_groups;
    static constexpr int mma_columns  = warp_columns / 8;
    static constexpr int k            = Gate ? kHidden : kIntermediate;
    static constexpr int weight_rows  = Gate ? 256 : 128;
};

template <class S>
struct TileStorage {
    union {
        struct {
            alignas(16) std::uint8_t weights[2][S::weight_rows * S::bk];
            alignas(16) std::uint8_t input[2][S::columns * S::bk];
        } stage;

        float product[S::weight_rows == 256 ? S::columns : 1][128];
    };

    int sources[S::columns];
};

template <class S>
__device__ __forceinline__ int swizzle(int row, int byte) {
    return byte ^ ((row & (S::bk / 16 - 1)) << 4);
}

__device__ __forceinline__ bool includes(ExpertSubsetKind kind, const int* mask, int expert) {
    return kind == ExpertSubsetKind::All || mask[expert] == (kind == ExpertSubsetKind::MaskOnes);
}

__device__ __forceinline__ float dot_four(unsigned codes, float4 x) {
    const float2 lo = fp8_code2_to_float2(codes), hi = fp8_code2_to_float2(codes >> 16);
    return fmaf(hi.y, x.w, fmaf(hi.x, x.z, fmaf(lo.y, x.y, lo.x * x.x)));
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::fp8
