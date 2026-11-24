#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
template <class S, int Tag>
struct Bf16VisionFc2Schedule : S {};
template <class S, int Tag>
using Fc2 = Bf16VisionFc2Schedule<Bf16KTailSchedule<Bf16ScheduleInstance<S, 4304>>, Tag>;
using Tiny4 =
    Fc2<Bf16A16SimtSchedule<4, 1, 2, 16, 4, 2, Bf16SimtActivationAccess::DirectStream,
                            Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2, 4>,
        20>;
using Tiny8 = Bf16VisionFc2Schedule<
    Bf16KTailSchedule<Bf16ScheduleInstance<
        Bf16A16SimtSchedule<4, 1, 2, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                            Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2, 8>,
        4304, 8, true>>,
    21>;
using Slice       = Fc2<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 2>, 22>;
using Small       = Fc2<Bf16A16MmaSchedule<32, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                                           Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
                        23>;
using OriginalHot = Fc2<Bf16A16MmaSchedule<32, 32, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                                           Bf16MmaFragmentPipeline::PingPong,
                                           Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>,
                        4>;
using Hot         = Fc2<Bf16A16TmaMmaSchedule<32, 32, 64, 16, 8, 3, 2>, 24>;
using Narrow      = Fc2<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>, 25>;
using Medium      = Fc2<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 26>;
using Wide96      = Fc2<Bf16A16TmaMmaSchedule<96, 128, 64, 32, 32, 3, 1>, 27>;
using Wide128     = Fc2<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 28>;
using Wide192     = Fc2<Bf16A16TmaMmaSchedule<192, 128, 64, 64, 32, 2, 1>, 29>;
using Row96  = Fc2<Bf16A16TmaMmaSchedule<96, 128, 64, 32, 32, 3, 1, Bf16MmaRaster::RowFast>, 30>;
using Row128 = Fc2<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>, 31>;
using Row192 = Fc2<Bf16A16TmaMmaSchedule<192, 128, 64, 64, 32, 2, 1, Bf16MmaRaster::RowFast>, 32>;

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

Bf16Launch select_bf16_n1152_k4304(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("bf16 linear: P must be a multiple of 4 in [4,131072]");
    if (tokens <= 4) return launch_bf16_simt<Tiny4>;
    if (tokens <= 8) return launch_bf16_simt<Tiny8>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Slice>;
    if (tokens <= 80) return launch_bf16_mma<Small>;
    if (tokens <= 96) return launch_warp_projection<Hot>;
    if (tokens > 96 && tokens <= 116) return launch_bf16_mma<OriginalHot>;
    if (tokens <= 128) return launch_bf16_tma_mma<Hot>;
    if (tokens <= 160) return launch_bf16_tma_mma<Narrow>;
    if (tokens <= 384) return launch_bf16_tma_mma<Medium>;
    if (tokens > 4096) return launch_vector_projection<Row128>;
    const auto tiles   = (static_cast<std::int64_t>(tokens) + 127) / 128;
    const auto work96  = 96 * ((12 * tiles + 47) / 48);
    const auto work128 = 128 * ((9 * tiles + 47) / 48);
    const auto work192 = 192 * ((6 * tiles + 47) / 48);
    if (tokens <= 1024) {
        if (work192 <= work96 && work192 <= work128) return launch_bf16_tma_mma<Wide192>;
        if (work128 < work96) return launch_warp_projection<Wide128>;
        return launch_warp_projection<Wide96>;
    }
    if (work128 <= work96 && work128 <= work192) return launch_warp_projection<Row128>;
    if (work192 <= work96) return launch_warp_projection<Row192>;
    return launch_warp_projection<Row96>;
}
} // namespace ninfer::ops::detail
