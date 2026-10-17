#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
template <class Base>
struct Bf16VisionPatchSchedule : Base {};
template <class Base>
using Patch = Bf16VisionPatchSchedule<Bf16ScheduleInstance<Base, 1536>>;

using Tiny    = Bf16VisionPatchSchedule<Bf16ScheduleInstance<
       Bf16A16SimtSchedule<4, 1, 2, 16, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                           Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>,
       1536, 4, true>>;
using Small8  = Patch<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 2>>;
using Small16 = Patch<Bf16A16SlicedKMmaSchedule<32, 8, 4, 64, 2>>;
using Middle =
    Patch<Bf16A16MmaSchedule<32, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                             Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Hot     = Patch<Bf16A16TmaMmaSchedule<32, 32, 64, 16, 8, 3, 2>>;
using Narrow  = Patch<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>>;
using Medium  = Patch<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>>;
using Wide128 = Patch<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::RowFast>>;
using Wide192 = Patch<Bf16A16TmaMmaSchedule<192, 128, 64, 64, 32, 2, 1, Bf16MmaRaster::RowFast>>;

template <class Schedule>
void launch_vector(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(bf16_a16_operands(x, w),
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

Bf16Launch select_bf16_n1152_k1536(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("bf16 linear: P must be a multiple of 4 in [4,131072]");
    if (tokens == 4) return launch_bf16_simt<Tiny>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<Small8>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Small16>;
    if (tokens <= 64) return launch_bf16_mma<Middle>;
    if (tokens <= 128) return launch_bf16_tma_mma<Hot>;
    if (tokens <= 192) return launch_bf16_tma_mma<Narrow>;
    if (tokens <= 320) return launch_warp_projection<Medium>;
    // The native weight fits L2; row-fast traversal retains it while sharing each X tile.
    if (tokens > 4096) return launch_vector<Wide128>;
    const auto tiles    = (static_cast<std::int64_t>(tokens) + 127) / 128;
    const auto waves128 = (9 * tiles + 47) / 48;
    const auto waves192 = (6 * tiles + 47) / 48;
    if (192 * waves192 <= 128 * waves128) return launch_warp_projection<Wide192>;
    return launch_vector<Wide128>;
}
} // namespace ninfer::ops::detail
