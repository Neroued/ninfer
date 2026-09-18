#include "core/weight.h"
#include "ops/linear_add/bf16/bf16_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_gemv.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

struct Bf16LinearAddDecodeOutput {
    __nv_bfloat16* residual;
};

struct Bf16LinearAddDecodeEpilogue {
    __device__ __forceinline__ void operator()(const Bf16LinearAddDecodeOutput& output,
                                               std::int32_t row, float accumulator) const {
        const float residual = __bfloat162float(output.residual[row]);
        output.residual[row] = __float2bfloat16_rn(accumulator + residual);
    }
};

} // namespace

void bf16_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream) {
    using Geometry = Bf16Geometry<5120, 6144>;
    using Schedule =
        Bf16GemvSchedule<8, 2, 2, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                         Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 1>;

    const Bf16LinearAddDecodeOutput output{static_cast<__nv_bfloat16*>(residual.data)};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_gemv_kernel<Geometry, Schedule, Bf16LinearAddDecodeOutput, Bf16LinearAddDecodeEpilogue>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
