#include "ops/linear/bias/dispatch.h"
#include "ops/linear/bias/bf16_launch.cuh"

namespace ninfer::ops::detail::linear_bias {
namespace {
template <class Schedule>
using Static = Bf16KTailSchedule<Bf16ScheduleInstance<Schedule, 1176>>;
using Gemv1 =
    Static<Bf16A16GemvSchedule<4, 1, 1, 8, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using S8     = Static<Bf16A16SlicedKMmaSchedule<16, 8, 4, 64, 2>>;
using M32    = Static<Bf16A16MmaSchedule<32, 32, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                                         Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using T32    = Static<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3, 2>>;
using T64    = Static<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>>;
using Mid128 = Static<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3, 1>>;
using T128   = Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>>;
using T256   = Static<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1>>;
using G128T128R12 =
    Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 12, 3>>;
using G256T128R4 =
    Static<Bf16A16TmaMmaSchedule<256, 128, 64, 128, 32, 2, 1, Bf16MmaRaster::Grouped, 4, 3>>;

} // namespace

Launch select_n1024_k1176(std::int32_t tokens) {
    if (tokens <= 1) return gemv<Gemv1>;
    if (tokens <= 16) return sliced<S8>;
    if (tokens <= 32) return mma<M32>;
    if (tokens <= 128) return tma<T32>;
    if (tokens <= 192) return tma<T64>;
    if (tokens <= 256) return warp<Mid128>;
    if (tokens <= 512) return warp<T128>;
    if (tokens <= 1536) return warp<T256>;
    if (tokens <= 2304) return warp<G128T128R12>;
    return warp<G256T128R4>;
}
} // namespace ninfer::ops::detail::linear_bias
