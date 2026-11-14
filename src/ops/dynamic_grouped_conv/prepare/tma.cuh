#pragma once

#include "ops/dynamic_grouped_conv/prepare/launch.h"
#include "ops/linear/kernels/bf16/bf16_a16_tma_register_b.cuh"

namespace ninfer::ops::detail::dynamic_grouped_conv::prepare {

// A 4D view of the existing [plane,group,K] weights loads all four coefficients
// of a contiguous group interval into one tile, without materializing a new layout.
struct NativeCoefficientAccess {
    template <class S>
    static CUtensorMap map(const Bf16A16Operands& p) {
        CUtensorMap result{};
        std::uint64_t dimensions[]{64, std::uint64_t(p.k / 64), std::uint64_t(p.rows / 4), 4};
        std::uint64_t strides[]{128, std::uint64_t(p.k) * 2, std::uint64_t(p.rows / 4) * p.k * 2};
        std::uint32_t box[]{64, std::uint32_t(S::kBlockK / 64), std::uint32_t(S::kBlockRows / 4),
                            4};
        const std::uint32_t steps[]{1, 1, 1, 1};
        const auto status = cuTensorMapEncodeTiled(
            &result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<__nv_bfloat16*>(p.weight),
            dimensions, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (status != CUDA_SUCCESS) {
            const char* name = nullptr;
            (void)cuGetErrorName(status, &name);
            throw std::runtime_error(std::string("dynamic grouped conv prepare TensorMap: ") +
                                     (name ? name : "CUDA error"));
        }
        return result;
    }

    template <class S>
    __device__ static void load(void* destination, const CUtensorMap* map, int k_sector, int row,
                                std::uint64_t* barrier) {
        common::tma_load_4d(destination, map, 0, k_sector, row / 4, 0, barrier);
    }
};

struct FusedOutputs {
    const __nv_bfloat16 *residual, *gain, *base;
    const float* inverse;
    __nv_bfloat16 *prepared, *finish;
    int width;

    template <int Hidden, bool Streaming = false>
    __device__ float normalized(int token, int hidden) const {
        const auto* input = residual + std::int64_t(token) * Hidden + hidden;
        __nv_bfloat16 value;
        if constexpr (Streaming) {
            __nv_bfloat16_raw bits;
            asm volatile("ld.global.cs.b16 %0, [%1];" : "=h"(bits.x) : "l"(input));
            value = __nv_bfloat16(bits);
        } else {
            value = *input;
        }
        return __bfloat162float(value) * inverse[token] * __bfloat162float(gain[hidden]);
    }
};

template <int Hidden, bool Streaming = false>
struct CoefficientConvolution {
    static constexpr bool kTmaCollective = true;
    template <class S>
    static constexpr int kSharedBytes = S::kBlockRows * S::kBlockTokens * sizeof(float);

    template <class S, bool FullTokens, class Output>
    __device__ void finish_tile(const Output& out, unsigned char* scratch,
                                float (&accum)[S::kMmaRows][S::kMmaTokens][4], int row_begin,
                                int token_begin, int, int tokens) const {
        constexpr int rows = S::kBlockRows, groups = rows / 4;
        static_assert(rows >= 32 && (rows & (rows - 1)) == 0);
        auto* tile    = reinterpret_cast<float*>(scratch);
        const auto at = [&](int row, int token) {
            return tile + token * rows + (row ^ (((token / 2) & (rows / 8 - 1)) * 8));
        };
        if (threadIdx.x >= S::kProducerThreads) {
            const int tid  = threadIdx.x - S::kProducerThreads;
            const int warp = tid / 32, lane = tid % 32;
            const int warp_row = warp / S::kWarpsTokens, warp_token = warp % S::kWarpsTokens;
#pragma unroll
            for (int mi = 0; mi < S::kMmaRows; ++mi) {
                const int row = warp_row * S::kWarpRows + mi * 16 + lane / 4;
#pragma unroll
                for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                    const int token         = warp_token * S::kWarpTokens + ni * 8 + 2 * (lane % 4);
                    *at(row, token)         = accum[mi][ni][0];
                    *at(row, token + 1)     = accum[mi][ni][1];
                    *at(row + 8, token)     = accum[mi][ni][2];
                    *at(row + 8, token + 1) = accum[mi][ni][3];
                }
            }
        }
        __syncthreads();
        const int columns =
            FullTokens ? S::kBlockTokens : min(S::kBlockTokens, tokens - token_begin);
        for (int item = threadIdx.x; item < columns * groups * 16; item += S::kThreads) {
            const int column = item / (groups * 16), group = item / 16 % groups;
            const int hidden  = row_begin / 4 * 16 + item % (groups * 16);
            const int token   = token_begin + column;
            const auto offset = std::int64_t(token) * Hidden + hidden;
            float value       = (__bfloat162float(out.base[hidden]) + *at(group, column)) *
                          out.template normalized<Hidden, Streaming>(token, hidden);
            if (token % out.width)
                value =
                    fmaf(__bfloat162float(out.base[Hidden + hidden]) + *at(groups + group, column),
                         out.template normalized<Hidden, Streaming>(token - 1, hidden), value);
            out.prepared[offset] = __float2bfloat16_rn(value);
        }
        for (int item = threadIdx.x; item < columns * groups * 2; item += S::kThreads) {
            const int column = item / (groups * 2), tap = item / groups % 2, group = item % groups;
            const auto offset = (std::int64_t(token_begin + column) * 2 + tap) * (Hidden / 16) +
                                row_begin / 4 + group;
            out.finish[offset] = __float2bfloat16_rn(*at((2 + tap) * groups + group, column));
        }
    }
};

template <int Hidden, int Rows, int Tokens, int K, int WarpRows, int WarpTokens, int Stages,
          bool RegisterB = false, Bf16MmaRaster Raster = Bf16MmaRaster::TokenFast,
          bool Streaming = false>
void launch_tma_fused(const NormSource& source, const Tensor& input, const Weight& weight,
                      const Tensor& base, Tensor& prepared, Tensor& finish, cudaStream_t stream) {
    using Base = std::conditional_t<
        RegisterB, Bf16A16TmaRegisterBSchedule<Rows, Tokens, K, WarpRows, WarpTokens, 2, Raster>,
        Bf16A16TmaMmaSchedule<Rows, Tokens, K, WarpRows, WarpTokens, Stages>>;
    using S = Bf16ScheduleInstance<Base, Hidden>;
    const Bf16A16Operands operands{static_cast<const __nv_bfloat16*>(input.data),
                                   static_cast<const __nv_bfloat16*>(weight.qdata), Hidden / 4,
                                   Hidden, input.ne[1] * input.ne[2]};
    const FusedOutputs output{static_cast<const __nv_bfloat16*>(source.residual.data),
                              static_cast<const __nv_bfloat16*>(source.weight.data),
                              static_cast<const __nv_bfloat16*>(base.data),
                              static_cast<const float*>(source.inverse.data),
                              static_cast<__nv_bfloat16*>(prepared.data),
                              static_cast<__nv_bfloat16*>(finish.data),
                              input.ne[1]};
    if constexpr (RegisterB)
        launch_bf16_a16_tma_register_b<S, FusedOutputs, CoefficientConvolution<Hidden, Streaming>,
                                       NativeCoefficientAccess>(operands, output, {}, stream);
    else
        launch_bf16_a16_tma_mma<S, 1, FusedOutputs, CoefficientConvolution<Hidden, Streaming>,
                                NativeCoefficientAccess>(operands, output, {}, stream);
}

} // namespace ninfer::ops::detail::dynamic_grouped_conv::prepare
