#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class S, int Tag>
struct Bf16LmHeadSchedule : S {};

using Original2 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 24, 4, 64, 2>, 2560>;
using Original3 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 32, 4, 64, 2>, 2560>;
using Original4 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 32, 128, 32, 16, 2, 1>, 2560>;
using Original5 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 48, 128, 32, 16, 2, 1>, 2560>;
using Original6 = Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>;
using S2        = Bf16LmHeadSchedule<Original2, 2>;
using S3        = Bf16LmHeadSchedule<Original3, 3>;
using S4        = Bf16LmHeadSchedule<Original4, 4>;
using S5        = Bf16LmHeadSchedule<Original5, 5>;
using S6        = Bf16LmHeadSchedule<Original6, 6>;
using Small     = Bf16LmHeadSchedule<
        Bf16KTailSchedule<Bf16ScheduleInstance<
            Bf16A16SimtSchedule<2, 2, 1, 16, 4, 2, Bf16SimtActivationAccess::DirectStream,
                                Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2, 8>,
            2560>>,
        10>;
using Medium =
    Bf16LmHeadSchedule<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 2560>,
                       12>;
using Window = Bf16LmHeadSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 192, 64, 64, 32, 2, 1>, 2560>, 13>;
using Prefill = Bf16LmHeadSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1>, 2560>, 14>;
using WideFull = Bf16LmHeadSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1>, 2560>, 16>;
using WarpWide = Bf16LmHeadSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1>, 2560>, 18>;
using WarpLarge = Bf16LmHeadSchedule<
    Bf16ScheduleInstance<
        Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1, Bf16MmaRaster::Grouped, 6, 1>, 2560>,
    19>;

template <class Schedule>
void launch_warp_store(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(bf16_a16_operands(x, w),
                                      LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                                      Bf16MmaWarpStoreEpilogue{}, stream);
}

void launch_window(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Window>(bf16_a16_operands(x, w),
                                    LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                                    Bf16TmaVectorEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n248320_k2560(std::int32_t tokens) {
    if (tokens <= 8) return launch_bf16_simt<Small>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<S3>;
    if (tokens <= 64) return launch_bf16_tma_mma<S4>;
    if (tokens <= 96) return launch_bf16_tma_mma<S5>;
    // The original hot route avoids the R32 candidate's median and p95 regressions.
    if (tokens <= 128) return launch_bf16_tma_mma<S6>;
    if (tokens <= 256) return launch_bf16_tma_mma<Medium>;
    if (tokens < 512) return launch_bf16_tma_mma<S6>;
    if ((tokens >= 513 && tokens <= 576) || (tokens >= 1025 && tokens <= 1152))
        return launch_window;
    if (tokens >= 3840) return launch_warp_store<WarpLarge>;
    // The upper half has the same padded work as T128, with fewer shared loads.
    const int tail = tokens % 256;
    if (tokens >= 641 && tokens <= 2560 && (tail == 0 || tail > 128))
        return launch_warp_store<WarpWide>;
    if (tail == 0) return launch_bf16_tma_mma<WideFull>;
    return launch_bf16_tma_mma<Prefill>;
}
} // namespace ninfer::ops::detail
