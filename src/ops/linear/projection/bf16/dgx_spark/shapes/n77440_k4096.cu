#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<16, 16, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    4096>;
using Small = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<8, 8, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 8>,
    4096>;
using Sliced = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>, 4096>;
using Hot256 = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    4096>;
using Medium = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 4096>;
using Mid128 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 2>, 4096>;
using Bulk   = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 4096>;
using Window = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<160, 192, 64, 80, 48, 2, 1>, 4096>;
using Prefill =
    Bf16ScheduleInstance<Bf16RowTailSchedule<Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1>>,
                         4096>;
using Wide = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1>, 4096>;

template <class Schedule, class Epilogue = Bf16MmaWarpStoreEpilogue>
void warp_store(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(bf16_a16_operands(x, w),
                                      LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                                      Epilogue{}, stream);
}

void streaming(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    const auto input   = bf16_a16_operands(x, w);
    constexpr int slab = 2048;
    for (std::int64_t offset = 0; offset < input.tokens;) {
        const int remaining = input.tokens - offset;
        // Keep a short final tile with its preceding slab instead of rereading the complete
        // weight bank for a separate one-column call at, for example, T=4097.
        const int count = remaining < slab + Wide::kBlockTokens ? remaining : slab;
        auto part       = input;
        part.x += offset * input.k;
        part.tokens = count;
        launch_bf16_a16_tma_mma<Wide>(
            part, LinearBf16Output{static_cast<__nv_bfloat16*>(y.data) + offset * w.n, w.n},
            Bf16MmaStreamingWarpStoreEpilogue{}, stream);
        offset += count;
    }
}
} // namespace

Bf16Launch select_bf16_n77440_k4096(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 8) return launch_bf16_simt<Small>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Sliced>;
    if (tokens <= 128) return launch_bf16_mma<Hot256>;
    if (tokens <= 384) return launch_bf16_tma_mma<Medium>;
    if (tokens <= 512) return launch_bf16_tma_mma<Mid128>;
    if (tokens >= 2432) return streaming;
    if ((tokens > 1024 && tokens <= 1152) || tokens > 2176) return launch_bf16_tma_mma<Window>;
    if (tokens <= 640 || (tokens > 768 && tokens <= 896)) return launch_bf16_tma_mma<Bulk>;
    const int tail = tokens % 256;
    if (tail == 0 || tail > 128) {
        if (tokens >= 2048) return warp_store<Wide, Bf16MmaStreamingWarpStoreEpilogue>;
        return warp_store<Wide>;
    }
    return launch_bf16_tma_mma<Prefill>;
}
} // namespace ninfer::ops::detail
