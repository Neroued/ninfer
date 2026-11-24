#include "ops/linear/bias/dispatch.h"
#include "ops/linear/bias/bf16_launch.cuh"

namespace ninfer::ops::detail::linear_bias {
namespace {
template <class S>
using Static = Bf16ScheduleInstance<S, 4096>;
using Gemv =
    Static<Bf16A16GemvSchedule<4, 4, 1, 8, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 4, 2>>;
using Small = Static<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 2>>;
using S32   = Static<Bf16A16SlicedKMmaSchedule<16, 32, 8, 64, 2>>;
using Hot   = Static<Bf16A16MmaSchedule<32, 64, 256, 16, 16, 2, 1, Cache::cg, Cache::cg,
                                        Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Mid256 =
    Static<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 2, 2, Bf16MmaRaster::TokenFast, 1, 1, 1>>;
using Mid512 =
    Static<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 2, Bf16MmaRaster::TokenFast, 1, 1, 1>>;
using Large = Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>>;
using Bulk =
    Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 12, 3>>;

} // namespace

Launch select_n4096_k4096(std::int32_t tokens) {
    if (tokens == 1) return gemv<Gemv>;
    if (tokens <= 16) return sliced<Small>;
    if (tokens <= 32) return sliced<S32>;
    if (tokens <= 192) return mma<Hot>;
    if (tokens <= 256) return warp<Mid256>;
    if (tokens <= 512) return warp<Mid512>;
    if (tokens <= 2560) return warp<Large>;
    return warp<Bulk>;
}
} // namespace ninfer::ops::detail::linear_bias
