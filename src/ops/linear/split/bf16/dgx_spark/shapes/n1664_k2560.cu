#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/split/dispatch.h"
#include "ops/linear/split/bf16/launch.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {
// The geometry owns its CUDA entries, even where constants match another Op.
template <class Base>
struct Bf16KvIndexSplitSchedule : Base {};

using Gemv = Bf16KvIndexSplitSchedule<Bf16ScheduleInstance<
    Bf16A16GemvSchedule<4, 2, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    2560>>;
template <int Stages>
using Hot32  = Bf16KvIndexSplitSchedule<Bf16ScheduleInstance<
     Bf16A16MmaSchedule<32, 32, 128, 16, 8, Stages, 1, Cache::cg, Cache::cg,
                        Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
     2560>>;
using HotTma = Bf16KvIndexSplitSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>, 2560>>;
using Mid48 = Bf16KvIndexSplitSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 2>, 2560>>;
using Bulk64 = Bf16KvIndexSplitSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2>, 2560>>;
using Bulk80 = Bf16KvIndexSplitSchedule<
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 80, 64, 32, 16, 3>, 2560>>;
template <Bf16MmaRaster Raster>
using Bulk96 = Bf16KvIndexSplitSchedule<Bf16ScheduleInstance<
    Bf16RowTailSchedule<Bf16A16TmaMmaSchedule<96, 128, 64, 32, 32, 3, 1, Raster>>, 2560>>;

template <Bf16MmaRaster Raster>
using Bulk192 = Bf16KvIndexSplitSchedule<Bf16ScheduleInstance<
    Bf16RowTailSchedule<Bf16A16TmaMmaSchedule<192, 128, 64, 64, 32, 2, 1, Raster>>, 2560>>;

template <class Routes>
auto select_route(std::int32_t tokens, const Routes& routes) {
    if (tokens == 1) return routes.template gemv<Gemv>();
    if (tokens <= 32) return routes.template mma<Hot32<3>>();
    if (tokens <= 64) return routes.template mma<Hot32<2>>();
    if (tokens <= 96) return routes.template mma<Hot32<3>>();
    if (tokens <= 128) return routes.template tma<HotTma>();
    if (tokens <= 160) return routes.template tma<Bulk80>();
    if (tokens <= 256) return routes.template tma<Bulk96<Bf16MmaRaster::TokenFast>>();
    if (tokens <= 336) return routes.template tma<Mid48>();
    if (tokens <= 448) return routes.template tma<Bulk64>();
    if (tokens <= 560) return routes.template tma<Bulk80>();
    // Eighteen 80-token tiles finish in five waves here. The wider, row-padded
    // 96x128 tile regressed across this interval in the complete Op comparison.
    if (tokens > 1408 && tokens <= 1440) return routes.template tma<Bulk80>();
    const std::int64_t t = tokens;
    const auto waves96   = (18 * ((t + 127) / 128) + 47) / 48;
    const auto waves192  = (9 * ((t + 127) / 128) + 47) / 48;
    const auto cost96    = waves96 * 96 * 128;
    const auto cost192   = waves192 * 192 * 128;
    if (tokens <= 2048) {
        const auto waves64 = (13 * ((t + 63) / 64) + 95) / 96;
        const auto cost64  = waves64 * 2 * 128 * 64;
        // Keep wider fragment reuse only when complete CTA waves add no work.
        if (cost192 <= cost96 && (cost192 < cost64 || (tokens > 1152 && cost192 == cost64)))
            return routes.template tma<Bulk192<Bf16MmaRaster::TokenFast>>();
        if (cost96 < cost64) return routes.template tma<Bulk96<Bf16MmaRaster::TokenFast>>();
        return routes.template tma<Bulk64>();
    }
    if (cost192 <= cost96) return routes.template tma<Bulk192<Bf16MmaRaster::RowFast>>();
    return routes.template tma<Bulk96<Bf16MmaRaster::RowFast>>();
}

template <class Output>
struct Bf16KvWarpRoutes : Bf16SplitRoutes<Output> {
    template <class S>
    void tma() const {
        if constexpr (!bf16_predicated_rows<S> && S::kBlockRows == 128)
            launch_bf16_a16_tma_mma<S>(this->operands, this->output, Bf16MmaWarpStoreEpilogue{},
                                       this->stream);
        else
            Bf16SplitRoutes<Output>::template tma<S>();
    }
};

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> out, MemoryArena*,
                  cudaStream_t stream) {
    const auto& weight = weights.first;
    using Output       = LinearBf16SegmentedOutput<512, 512, 512, 128>;
    const Output output{
        {static_cast<__nv_bfloat16*>(out[0].data), static_cast<__nv_bfloat16*>(out[1].data),
         static_cast<__nv_bfloat16*>(out[2].data), static_cast<__nv_bfloat16*>(out[3].data)}};
    select_route(x.ne[1], Bf16KvWarpRoutes<Output>{Bf16SplitRoutes<Output>{
                              bf16_a16_operands(x, weight), output, stream}});
}
} // namespace

const SplitShape kBf16SplitN1664K2560{1664, 2560, kSplitRows<512, 512, 512, 128>, launch_split};

} // namespace ninfer::ops::detail::linear_split
