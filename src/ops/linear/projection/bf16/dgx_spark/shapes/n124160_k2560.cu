#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
using Small = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<4, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 8>,
    2560>;
using Sliced = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>, 2560>;
using Sliced32 =
    Bf16ScheduleInstance<Bf16KTailSchedule<Bf16A16SlicedKMmaSchedule<16, 32, 8, 128, 1>>, 2560>;
using Hot256 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
using Hot128 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 128, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
using Medium  = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 2560>;
using Bulk    = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 2560>;
using Window  = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<160, 192, 64, 80, 48, 2, 1>, 2560>;
using Prefill = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1>, 2560>;
using Wide    = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1>, 2560>;

template <class Schedule, class Epilogue = Bf16MmaWarpStoreEpilogue>
void warp_store(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(bf16_a16_operands(x, w),
                                      LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                                      Epilogue{}, stream);
}

void streaming(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    const auto operands = bf16_a16_operands(x, w);
    for (int offset = 0; offset < operands.tokens;) {
        const int remaining = operands.tokens - offset;
        // Merge a short tail instead of rereading the complete weight for it.
        const int count = remaining < 2304 ? remaining : 2048;
        auto slice      = operands;
        slice.x += static_cast<std::int64_t>(offset) * w.k;
        slice.tokens = count;
        const LinearBf16Output output{
            static_cast<__nv_bfloat16*>(y.data) + static_cast<std::int64_t>(offset) * w.n, w.n};
        launch_bf16_a16_tma_mma<Wide>(slice, output, Bf16MmaStreamingWarpStoreEpilogue{}, stream);
        offset += count;
    }
}
} // namespace

Bf16Launch select_bf16_n124160_k2560(std::int32_t tokens) {
    if (tokens <= 8) return launch_bf16_simt<Small>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Sliced>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<Sliced32>;
    if (tokens <= 96) return launch_bf16_mma<Hot256>;
    if (tokens <= 128) return launch_bf16_mma<Hot128>;
    if (tokens <= 256) return launch_bf16_tma_mma<Medium>;
    if (tokens >= 2432) return streaming;
    if ((tokens > 512 && tokens <= 576) || (tokens > 1024 && tokens <= 1152))
        return launch_bf16_tma_mma<Window>;
    const int tail = tokens % 256;
    if (tokens >= 385 && (tail == 0 || tail > 128)) {
        if (tokens >= 2048) return warp_store<Wide, Bf16MmaStreamingWarpStoreEpilogue>;
        return warp_store<Wide>;
    }
    if (tokens < 512) return launch_bf16_tma_mma<Bulk>;
    return launch_bf16_tma_mma<Prefill>;
}
} // namespace ninfer::ops::detail
