#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
template <class S, int Tag>
struct Bf16VisionMergerFc2Schedule : S {};

using Original0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<8, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Streaming,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    4608>;
using Original1 = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<1, 1, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>,
    4608>;
using Original2 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 3>, 4608>;
using Original4 = Bf16RowTailSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<48, 32, 192, 16, 16, 3, 1>, 4608>>;
using Original5 = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<64, 32, 128, 16, 16, 3, 1, Bf16MmaRaster::TokenFast, 1, 18>, 4608>;
using S0    = Bf16VisionMergerFc2Schedule<Original0, 0>;
using S1    = Bf16VisionMergerFc2Schedule<Original1, 1>;
using S2    = Bf16VisionMergerFc2Schedule<Original2, 2>;
using S4    = Bf16VisionMergerFc2Schedule<Original4, 4>;
using S5    = Bf16VisionMergerFc2Schedule<Original5, 5>;
using Small = Bf16VisionMergerFc2Schedule<
    Bf16ScheduleInstance<
        Bf16A16MmaSchedule<32, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                           Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
        4608>,
    20>;
using Medium = Bf16VisionMergerFc2Schedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 4608>, 21>;
using Transition = Bf16VisionMergerFc2Schedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 2, 2>, 4608>, 22>;
using Wide = Bf16VisionMergerFc2Schedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 4608>, 23>;
using Large = Bf16VisionMergerFc2Schedule<
    Bf16ScheduleInstance<
        Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 12, 1>, 4608>,
    24>;

using Narrow = Bf16VisionMergerFc2Schedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 3, 1>, 4608>, 25>;
using ScalarWindow = Bf16VisionMergerFc2Schedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 4608>, 9>;

template <class S>
void launch_vector_projection(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                               Bf16TmaVectorEpilogue{}, stream);
}

template <class S>
void launch_warp_projection(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                               Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n2560_k4608(std::int32_t tokens) {
    if (tokens > 32768) throw std::invalid_argument("bf16 linear: V must be in [1,32768]");
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 2) return launch_bf16_simt<S1>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 32) return launch_bf16_mma<Small>;
    if (tokens <= 64) return launch_bf16_tma_mma<S4>;
    if (tokens <= 96) return launch_bf16_tma_mma<S5>;
    if (tokens <= 128) return launch_bf16_tma_mma<Medium>;
    if (tokens <= 192) return launch_warp_projection<Transition>;
    // Keep the original two-token-tile interval: its matched warp-store control
    // did not show a stable gain across independent boundary measurements.
    if (tokens <= 256) return launch_vector_projection<Wide>;
    if (tokens > 256 && tokens <= 384) return launch_vector_projection<Narrow>;
    if (tokens > 2048 && tokens < 2112) return launch_bf16_tma_mma<ScalarWindow>;
    if ((tokens > 1984 && tokens <= 2048) || tokens >= 2112) return launch_warp_projection<Large>;
    return launch_warp_projection<Wide>;
}
} // namespace ninfer::ops::detail
