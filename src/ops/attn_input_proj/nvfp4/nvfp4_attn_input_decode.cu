#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

struct Nvfp4AttentionInputOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t,
                                          float result) const {
        constexpr std::int32_t kQueryRows  = 6144;
        constexpr std::int32_t kKeyRows    = 1024;
        constexpr std::int32_t kGateRows   = 6144;
        constexpr std::int32_t kKeyBegin   = kQueryRows;
        constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
        constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;

        const __nv_bfloat16 result_bf16 = __float2bfloat16_rn(result);
        if (parent_row < kKeyBegin) {
            query[parent_row] = result_bf16;
        } else if (parent_row < kGateBegin) {
            key[parent_row - kKeyBegin] = result_bf16;
        } else if (parent_row < kValueBegin) {
            gate[parent_row - kGateBegin] = result_bf16;
        } else {
            value[parent_row - kValueBegin] = result_bf16;
        }
    }
};

} // namespace

void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    using Geometry = Nvfp4N14336K5120;
    using Schedule =
        Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
    static_assert((6144 % 128) == 0);
    static_assert((1024 % 128) == 0);

    const Nvfp4AttentionInputOutput output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    constexpr int kBlocks              = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
        Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
