#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<2, 1, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    6144>;
using Sliced = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>, 6144>;
using HotMma = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    6144>;
using HotTma     = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 128, 16, 16, 2, 2>, 6144>;
using Prefill128 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 6144>;
using Prefill192 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<160, 192, 64, 80, 48, 2, 1>, 6144>;
using Large128   = Bf16ScheduleInstance<
      Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 12, 3>, 6144>;
using Large192 = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<160, 192, 64, 80, 48, 2, 1, Bf16MmaRaster::Grouped, 8>, 6144>;

template <class S>
void launch_warp_projection(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                               Bf16MmaWarpStoreEpilogue{}, stream);
}

void launch_vector_projection(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Large128>(bf16_a16_operands(x, w),
                                      LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                      Bf16TmaVectorEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n2560_k6144(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Sliced>;
    if (tokens <= 96) return launch_bf16_mma<HotMma>;
    if (tokens <= 128) return launch_bf16_tma_mma<HotTma>;

    // Both prefill tiles reside at one CTA/SM on this 48-SM GB10. Include the
    // last partial CTA wave when comparing padded work; use 64-bit host arithmetic
    // throughout the positive token domain. The measured midrange favors the
    // larger tile, while grouped traversal bounds the large-T cache working set.
    const auto t        = static_cast<std::int64_t>(tokens);
    const auto waves128 = (20 * ((t + 127) / 128) + 47) / 48;
    const auto waves192 = (16 * ((t + 191) / 192) + 47) / 48;
    const bool wide =
        waves192 * (160 * 192) < waves128 * (128 * 128) || (tokens > 1344 && tokens <= 1920);
    if (wide)
        return tokens <= 2112 ? launch_warp_projection<Prefill192>
                              : launch_warp_projection<Large192>;
    return tokens <= 1920 ? launch_warp_projection<Prefill128> : launch_vector_projection;
}
} // namespace ninfer::ops::detail
