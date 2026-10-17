#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
template <class S, int Tag>
struct Bf16VisionMergerFc1Schedule : S {};

using Original0 = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<1, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    4608>;
using Original1 = Bf16KTailSchedule<Bf16ScheduleInstance<
    Bf16A16SimtSchedule<8, 1, 2, 8, 4, 2, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>,
    4608>>;
using Original2 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 2>, 4608>;
using Original3 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 2>, 4608>;
using Original4 = Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<32, 8, 4, 64, 2>, 4608>;
using Original5 = Bf16A16MmaSchedule<32, 16, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                                     Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using Original6 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 128, 16, 16, 3, 1>, 4608>;
using Original7 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<96, 32, 128, 16, 16, 3, 1>, 4608>;
using Original8 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 128, 32, 16, 3, 1>, 4608>;
using S0        = Bf16VisionMergerFc1Schedule<Original0, 0>;
using S1        = Bf16VisionMergerFc1Schedule<Original1, 1>;
using S2        = Bf16VisionMergerFc1Schedule<Original2, 2>;
using S3        = Bf16VisionMergerFc1Schedule<Original3, 3>;
using S4        = Bf16VisionMergerFc1Schedule<Original4, 4>;
using S5        = Bf16VisionMergerFc1Schedule<Original5, 5>;
using S6        = Bf16VisionMergerFc1Schedule<Original6, 6>;
using S7        = Bf16VisionMergerFc1Schedule<Original7, 7>;
using S8        = Bf16VisionMergerFc1Schedule<Original8, 8>;
using Small     = Bf16VisionMergerFc1Schedule<
        Bf16ScheduleInstance<
            Bf16A16MmaSchedule<32, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                               Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
            4608>,
        20>;
using Medium = Bf16VisionMergerFc1Schedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<192, 128, 64, 64, 32, 2, 1>, 4608>, 21>;
using Wide = Bf16VisionMergerFc1Schedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 4608>, 22>;
using Large = Bf16VisionMergerFc1Schedule<
    Bf16ScheduleInstance<
        Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 8, 3>, 4608>,
    23>;

template <class S>
void launch_vector_projection(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                               Bf16TmaVectorEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n4608_k4608(std::int32_t tokens) {
    if (tokens > 32768) throw std::invalid_argument("bf16 linear: V must be in [1,32768]");
    if (tokens <= 1) return launch_bf16_gemv<S0>;
    if (tokens <= 2) return launch_bf16_simt<S1>;
    if (tokens <= 3) return launch_bf16_sliced_k_mma<S2>;
    if (tokens <= 4) return launch_bf16_sliced_k_mma<S3>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<S4>;
    if (tokens <= 16) return launch_bf16_mma<Small>;
    if (tokens <= 32) return launch_bf16_mma<S5>;
    if (tokens <= 64) return launch_bf16_tma_mma<S6>;
    if (tokens <= 96) return launch_bf16_tma_mma<S7>;
    if (tokens <= 128) return launch_bf16_tma_mma<S8>;
    if (tokens <= 256) return launch_vector_projection<Medium>;
    if (tokens <= 2112) return launch_vector_projection<Wide>;
    return launch_vector_projection<Large>;
}
} // namespace ninfer::ops::detail
