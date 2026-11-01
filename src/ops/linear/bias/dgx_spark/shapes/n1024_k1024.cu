#include "ops/linear/bias/dispatch.h"
#include "ops/linear/bias/bf16_launch.cuh"

namespace ninfer::ops::detail::linear_bias {
namespace {
template <class S>
using Static = Bf16ScheduleInstance<S, 1024>;
using Gemv =
    Static<Bf16A16GemvSchedule<8, 1, 1, 8, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 2, 2>>;
using S8    = Static<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 1>>;
using M16   = Static<Bf16A16MmaSchedule<16, 16, 256, 16, 8, 2, 2, Cache::cg, Cache::cg,
                                        Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using M32   = Static<Bf16A16MmaSchedule<32, 32, 256, 32, 8, 2, 1, Cache::cg, Cache::cg,
                                        Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using M64   = Static<Bf16A16MmaSchedule<32, 64, 256, 16, 16, 2, 1, Cache::cg, Cache::cg,
                                        Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using T64   = Static<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>>;
using Mid   = Static<Bf16A16TmaMmaSchedule<64, 128, 128, 32, 32, 2, 1>>;
using Large = Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>>;
using Bulk  = Static<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1, Bf16MmaRaster::Grouped, 8>>;
using Balanced =
    Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1, Bf16MmaRaster::Grouped, 12, 3>>;

} // namespace

Launch select_n1024_k1024(std::int32_t tokens) {
    if (tokens == 1) return gemv<Gemv>;
    if (tokens <= 8) return sliced<S8>;
    if (tokens <= 16) return mma<M16>;
    if (tokens <= 32) return mma<M32>;
    if (tokens <= 64) return mma<M64>;
    if (tokens <= 256) return warp<T64>;
    if (tokens <= 384) return warp<Mid>;
    if (tokens <= 768) return warp<Large>;
    if (tokens <= 1536) return warp<Bulk>;
    if (tokens <= 2304) return warp<Balanced>;
    return warp<Bulk>;
}
} // namespace ninfer::ops::detail::linear_bias
