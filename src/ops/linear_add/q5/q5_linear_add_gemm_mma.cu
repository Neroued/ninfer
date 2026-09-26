#include "ops/linear/q5/q5_instances.cuh"
#include "core/weight.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/bf16_vector.cuh"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q5/q5_mma_launch.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {


// This Op keeps its vectorized collective finish and its BF16 projection
// materialization. The contraction only lends scratch and fully reduced fragments.
struct Q5LinearAddTileEpilogue {
    template <class Schedule, bool Full>
    __device__ __forceinline__ void
    finish_tile(LinearBf16StridedOutput output, __nv_bfloat16* scratch,
                const float (&acc)[Schedule::kMmaRows][Schedule::kMmaTokens][4], int row_begin,
                int token_begin, int rows, int token_end) const {
        constexpr int R = Schedule::kBlockRows;
        constexpr int T = Schedule::kBlockTokens;
        static_assert(R <= Schedule::kBlockK && R % 8 == 0);
        const int tid        = int(threadIdx.x);
        const int warp       = tid >> 5;
        const int lane       = tid & 31;
        const int warp_row   = warp / Schedule::kWarpGridTokens;
        const int warp_token = warp % Schedule::kWarpGridTokens;
#pragma unroll
        for (int mi = 0; mi < Schedule::kMmaRows; ++mi) {
            const int row = warp_row * Schedule::kWarpRows + mi * 16 + (lane >> 2);
#pragma unroll
            for (int ni = 0; ni < Schedule::kMmaTokens; ++ni) {
                const int token = warp_token * Schedule::kWarpTokens + ni * 8 + 2 * (lane & 3);
                scratch[token * R + row]           = __float2bfloat16_rn(acc[mi][ni][0]);
                scratch[(token + 1) * R + row]     = __float2bfloat16_rn(acc[mi][ni][1]);
                scratch[token * R + row + 8]       = __float2bfloat16_rn(acc[mi][ni][2]);
                scratch[(token + 1) * R + row + 8] = __float2bfloat16_rn(acc[mi][ni][3]);
            }
        }
        __syncthreads();

        union alignas(16) Pack {
            int4 raw;
            Bf16x8Pack values;
        };

        for (int item = tid; item < T * (R / 8); item += Schedule::kThreads) {
            const int local_token = item / (R / 8);
            const int local_row   = (item % (R / 8)) * 8;
            const int row         = row_begin + local_row;
            const int token       = token_begin + local_token;
            if (Full || (row < rows && token < token_end)) {
                auto* dst =
                    output.data + std::int64_t(token) * output.leading_dim + output.row_begin + row;
                const auto* projected = scratch + local_token * R + local_row;
                if (Full || row + 8 <= rows) {
                    Pack value, residual;
                    value.raw    = load_vec<int4>(projected);
                    residual.raw = load_vec<int4>(dst);
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair)
                        residual.values.pair[pair] =
                            __floats2bfloat162_rn(__low2float(residual.values.pair[pair]) +
                                                      __low2float(value.values.pair[pair]),
                                                  __high2float(residual.values.pair[pair]) +
                                                      __high2float(value.values.pair[pair]));
                    store_vec(dst, residual.raw);
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        if (row + i < rows)
                            dst[i] = __float2bfloat16_rn(__bfloat162float(dst[i]) +
                                                         __bfloat162float(projected[i]));
                }
            }
        }
    }
};

template <class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    launch_q5_a16_mma<Schedule>(
        q5_linear_operands(x, w),
        LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(residual_out.data),
                                std::int64_t(residual_out.nb[1] / sizeof(__nv_bfloat16)), 0},
        Q5LinearAddTileEpilogue{}, stream);
}

template <class Schedule>
void launch_pointwise(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    auto* data        = static_cast<__nv_bfloat16*>(residual.data);
    const auto stride = std::int64_t(residual.nb[1] / sizeof(__nv_bfloat16));
    launch_q5_a16_mma<Schedule>(q5_linear_operands(x, w), LinearBf16StridedOutput{data, stride, 0},
                                LinearResidualAddEpilogue{{data, stride, 0}}, stream);
}
} // namespace

void q5_linear_add_mma_r32_t32_k128_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                           cudaStream_t stream) {
    launch_pointwise<q5_instances::MmaR32T32K128S2A2>(x, w, residual, stream);
}

void q5_linear_add_mma_r32_t128_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                       cudaStream_t stream) {
    launch_pointwise<q5_instances::MmaR32T128>(x, w, residual, stream);
}

void q5_linear_add_mma_r64_t128_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                       cudaStream_t stream) {
    launch_route<q5_instances::MmaR64T128>(x, w, residual, stream);
}
} // namespace ninfer::ops::detail
