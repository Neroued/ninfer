#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_simt.cuh"

#include <cuda_bf16.h>

#include <array>
#include <cstddef>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Geometry              = Nvfp4N34816K5120;
constexpr int kIntermediate = Geometry::kOutputRows / 2;

template <int ActiveTokens, class Schedule>
__global__ __launch_bounds__(
    Schedule::kThreads,
    Schedule::
        kMinBlocksPerSm) void nvfp4_linear_swiglu_small_t_kernel(const __nv_bfloat16* __restrict__ x,
                                                                 const std::
                                                                     uint8_t* __restrict__ codes,
                                                                 const std::
                                                                     uint8_t* __restrict__ scales,
                                                                 float inverse_weight_divisor,
                                                                 __nv_bfloat16* __restrict__ out) {
    static_assert(Schedule::kWarpsPerRow == 1);
    static_assert(Schedule::kRowsPerWarp == 2);
    static_assert(Schedule::kTokenTile == ActiveTokens);
    static_assert((Schedule::kWarpsPerCta % 4) == 0);
    static_assert((128 % Schedule::kWarpsPerCta) == 0);

    __shared__ Nvfp4SimtSharedStorage<Geometry, ActiveTokens, Schedule> shared;
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

    float accumulators[Schedule::kRowsPerWarp][Schedule::kTokenTile][Schedule::kAccumulatorChains] =
        {};
    compute_nvfp4_simt_rows<Geometry, ActiveTokens, Schedule>(
        x, codes, scales, shared, inverse_weight_divisor, parent_rows,
        warp * Schedule::kRowsPerWarp, 0, 0, lane, accumulators);

#pragma unroll
    for (int token = 0; token < ActiveTokens; ++token) {
        float gate = 0.0F;
        float up   = 0.0F;
#pragma unroll
        for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
            gate += accumulators[0][token][chain];
            up += accumulators[1][token][chain];
        }
        gate = warp_reduce_sum(gate);
        up   = warp_reduce_sum(up);
        if (lane == 0) {
            out[static_cast<std::int64_t>(token) * kIntermediate + gate_row] =
                __float2bfloat16_rn(silu(gate) * up);
        }
    }
}

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SimtActivationAccess::SharedPhase
                                                  : Nvfp4SimtActivationAccess::TokenPacked;
    static constexpr int kWarpsPerCta       = ActiveTokens >= 13 ? 16 : (ActiveTokens >= 5 ? 4 : 8);
    using Schedule = Nvfp4SimtSchedule<kWarpsPerCta, 1, 2, 16, ActiveTokens, 1, kActivationAccess,
                                       Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                       Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    static_assert((kIntermediate % Schedule::kWarpsPerCta) == 0);
    constexpr int kBlocks = kIntermediate / Schedule::kWarpsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    nvfp4_linear_swiglu_small_t_kernel<ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), inverse,
            static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{&launch_exact<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<16 - 2 + 1>{});

} // namespace

void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - 2);
    kLaunchers[index](x, weight, out, stream);
}

} // namespace ninfer::ops::detail
