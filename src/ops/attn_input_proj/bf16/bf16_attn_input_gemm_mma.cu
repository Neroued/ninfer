#include "core/weight.h"
#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemm_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

struct Bf16AttentionInputMmaOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ Bf16MmaOutputTile tile(std::int32_t parent_row) const {
        constexpr std::int32_t kQueryRows  = 6144;
        constexpr std::int32_t kKeyRows    = 1024;
        constexpr std::int32_t kGateRows   = 6144;
        constexpr std::int32_t kKeyBegin   = kQueryRows;
        constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
        constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;

        if (parent_row < kKeyBegin) { return {query, kQueryRows, 0}; }
        if (parent_row < kGateBegin) { return {key, kKeyRows, kKeyBegin}; }
        if (parent_row < kValueBegin) { return {gate, kGateRows, kGateBegin}; }
        return {value, kKeyRows, kValueBegin};
    }
};

template <bool FullTokens>
void launch_variant(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                    Tensor& v, cudaStream_t stream) {
    using Geometry = Bf16Geometry<14336, 5120>;
    using Schedule = Bf16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                                     Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
    static_assert((Geometry::kOutputRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    static_assert((6144 % Schedule::kBlockRows) == 0);
    static_assert((1024 % Schedule::kBlockRows) == 0);

    constexpr int tiles_m = Geometry::kOutputRows / Schedule::kBlockRows;
    const int tiles_n     = div_up(x.ne[1], Schedule::kBlockCols);
    const int blocks      = tiles_m * tiles_n;
    const Bf16AttentionInputMmaOutput output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attr = cudaFuncSetAttribute(
            bf16_gemm_mma_kernel<Geometry, Schedule, FullTokens, Bf16AttentionInputMmaOutput>,
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

void bf16_attn_input_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                Tensor& k, Tensor& v, cudaStream_t stream) {
    using Geometry = Bf16Geometry<14336, 5120>;
    using Schedule = Bf16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                                     Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
    if ((x.ne[1] % Schedule::kBlockCols) == 0) {
        launch_variant<true>(x, weight, q, gate, k, v, stream);
    } else {
        launch_variant<false>(x, weight, q, gate, k, v, stream);
    }
}

} // namespace ninfer::ops::detail
