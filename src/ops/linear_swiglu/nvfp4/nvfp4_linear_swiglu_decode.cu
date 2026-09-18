#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4N34816K5120;
using Schedule =
    Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 2>;

constexpr int kIntermediate = Geometry::kOutputRows / 2;
static_assert(Schedule::kRowsPerWarp == 2);
static_assert((Schedule::kWarpsPerCta % 4) == 0);
static_assert((128 % Schedule::kWarpsPerCta) == 0);
static_assert((kIntermediate % Schedule::kWarpsPerCta) == 0);

__global__ __launch_bounds__(
    Schedule::kThreads,
    Schedule::
        kMinBlocksPerSm) void nvfp4_linear_swiglu_decode_kernel(const __nv_bfloat16* __restrict__ x,
                                                                const std::
                                                                    uint8_t* __restrict__ codes,
                                                                const std::
                                                                    uint8_t* __restrict__ scales,
                                                                float inverse_weight_divisor,
                                                                __nv_bfloat16* __restrict__ out) {
    __shared__ Nvfp4GemvSharedStorage<Geometry, Schedule> shared;
    constexpr int kCtasPerM128                    = 128 / Schedule::kWarpsPerCta;
    const int block                               = static_cast<int>(blockIdx.x);
    const int m_tile                              = block / kCtasPerM128;
    const int cta_in_tile                         = block - m_tile * kCtasPerM128;
    const int lane                                = static_cast<int>(threadIdx.x) & 31;
    const int warp                                = static_cast<int>(threadIdx.x) >> 5;
    const int flat_pair                           = cta_in_tile * Schedule::kWarpsPerCta + warp;
    const int rmod                                = flat_pair >> 2;
    const int quartile                            = flat_pair & 3;
    const int gate_row                            = m_tile * 128 + rmod + quartile * 32;
    const int parent_rows[Schedule::kRowsPerWarp] = {gate_row, gate_row + kIntermediate};

    float accumulators[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};
    compute_nvfp4_rows<Geometry, Schedule>(x, codes, scales, shared, inverse_weight_divisor,
                                           parent_rows, warp * Schedule::kRowsPerWarp, lane,
                                           accumulators);

    float gate = 0.0F;
    float up   = 0.0F;
#pragma unroll
    for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
        gate += accumulators[0][chain];
        up += accumulators[1][chain];
    }
    gate = warp_reduce_sum(gate);
    up   = warp_reduce_sum(up);
    if (lane == 0) { out[gate_row] = __float2bfloat16_rn(silu(gate) * up); }
}

} // namespace

void nvfp4_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream) {
    constexpr int kBlocks = kIntermediate / Schedule::kWarpsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    nvfp4_linear_swiglu_decode_kernel<<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
