#include "core/weight.h"
#include "ops/linear_add/bf16/bf16_linear_add_plan.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemm_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

struct Bf16LinearAddMmaOutputTile {
    __nv_bfloat16* residual;
    std::int32_t leading_dim;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token,
                                          float accumulator) const {
        __nv_bfloat16* destination =
            residual + static_cast<std::int64_t>(token) * leading_dim + row;
        const float residual_value = __bfloat162float(*destination);
        *destination               = __float2bfloat16_rn(accumulator + residual_value);
    }
};

struct Bf16LinearAddMmaOutput {
    __nv_bfloat16* residual;
    std::int32_t leading_dim;

    __device__ __forceinline__ Bf16LinearAddMmaOutputTile tile(std::int32_t) const {
        return {residual, leading_dim};
    }
};

template <class Schedule, bool FullTokens>
void launch_variant(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Geometry = Bf16Geometry<5120, 6144>;
    static_assert((Geometry::kOutputRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);

    constexpr int tiles_m = Geometry::kOutputRows / Schedule::kBlockRows;
    const int tiles_n     = div_up(x.ne[1], Schedule::kBlockCols);
    const int blocks      = tiles_m * tiles_n;
    const Bf16LinearAddMmaOutput output{static_cast<__nv_bfloat16*>(residual.data),
                                        Geometry::kOutputRows};

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attr = cudaFuncSetAttribute(
            bf16_gemm_mma_kernel<Geometry, Schedule, FullTokens, Bf16LinearAddMmaOutput>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        CUDA_CHECK(attr);
    }
    bf16_gemm_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void bf16_linear_add_aggregate_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                          cudaStream_t stream) {
    using UpTo32 = Bf16MmaSchedule<32, 32, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                                   Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
    using UpTo48 = Bf16MmaSchedule<32, 32, 192, 16, 8, 2, 2, Cache::cg, Cache::ca,
                                   Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

    if (x.ne[1] <= 32) {
        if ((x.ne[1] % UpTo32::kBlockCols) == 0) {
            launch_variant<UpTo32, true>(x, weight, residual, stream);
        } else {
            launch_variant<UpTo32, false>(x, weight, residual, stream);
        }
        return;
    }
    launch_variant<UpTo48, false>(x, weight, residual, stream);
}

void bf16_linear_add_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                cudaStream_t stream) {
    using Geometry = Bf16Geometry<5120, 6144>;
    using Schedule = Bf16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                                     Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
    if ((x.ne[1] % Schedule::kBlockCols) == 0) {
        launch_variant<Schedule, true>(x, weight, residual, stream);
    } else {
        launch_variant<Schedule, false>(x, weight, residual, stream);
    }
}

} // namespace ninfer::ops::detail
