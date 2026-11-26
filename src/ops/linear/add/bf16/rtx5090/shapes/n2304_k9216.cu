#include "ops/linear/add/bf16/rtx5090/routes.cuh"
#include "ops/linear/projection/bf16/rtx5090/shapes/n2304_k9216.cuh"

namespace ninfer::ops::detail {
namespace {

namespace base = bf16_n2304_k9216;

using TmaR32T32 = base::StaticK<Bf16A16TmaMmaSchedule<32, 32, 128, 16, 16, 3, 1>>;
using TmaR128T32 =
    base::StaticK<Bf16A16TmaMmaSchedule<128, 32, 64, 32, 16, 3, 1, Bf16MmaRaster::Grouped, 4, 4>>;
using TmaR96T96 =
    base::StaticK<Bf16A16TmaMmaSchedule<96, 96, 128, 48, 24, 2, 1, Bf16MmaRaster::RowFast>>;

} // namespace

LinearAddLaunch select_bf16_linear_add_n2304_k9216(std::int32_t tokens) {
    const Bf16AddRoutes routes;
    if (tokens == 1) return base::select(tokens, routes);
    if (tokens <= 8) return routes.sliced<base::Sliced8, Bf16ScalarResidualAddWarpStoreEpilogue>();
    if (tokens <= 32) return base::select(tokens, routes);
    if (tokens <= 64) return routes.tma<TmaR32T32>();
    if (tokens <= 72)
        return routes.sliced<base::Sliced24, Bf16ScalarResidualAddWarpStoreEpilogue>();
    if (tokens <= 80)
        return routes.sliced<base::Sliced40, Bf16ScalarResidualAddWarpStoreEpilogue>();
    if (tokens <= 192) return base::select(tokens, routes);
    if (tokens <= 224) return routes.tma<base::TmaR96T32>();
    if (tokens <= 256) return base::select(tokens, routes);
    if (tokens <= 288) return routes.tma<TmaR128T32>();
    if (tokens <= 448) return routes.tma<base::TmaR96T64>();
    if (tokens <= 576) return base::select(tokens, routes);
    if (tokens <= 672) return routes.tma<TmaR96T96>();
    if (tokens <= 896) return routes.tma<base::TmaR96T64>();
    if (tokens <= 1152) return base::select(tokens, routes);
    if (tokens <= 1344) return routes.tma<TmaR96T96>();
    if (tokens <= 1792) return routes.tma<base::TmaR128T64>();
    if (tokens <= 2304) return routes.tma<base::TmaR256T64>();
    if (tokens <= 3136) return routes.tma<base::TmaR96T64>();
    if (tokens <= 3584) return routes.tma<base::TmaR256T64>();
    return base::select(tokens, routes);
}

} // namespace ninfer::ops::detail
