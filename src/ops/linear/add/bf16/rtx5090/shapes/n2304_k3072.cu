#include "ops/linear/add/bf16/rtx5090/routes.cuh"
#include "ops/linear/projection/bf16/rtx5090/shapes/n2304_k3072.cuh"

namespace ninfer::ops::detail {
namespace {

namespace base = bf16_n2304_k3072;

using TmaR64T16   = base::StaticK<Bf16A16TmaMmaSchedule<64, 16, 128, 16, 8, 3, 1>>;
using TmaR128T32 =
    base::StaticK<Bf16A16TmaMmaSchedule<128, 32, 64, 32, 16, 3, 1, Bf16MmaRaster::TokenFast, 1, 2>>;
using TmaR64T128  = base::StaticK<Bf16A16TmaMmaSchedule<64, 128, 64, 64, 16, 3, 1>>;
using TmaR96T96   = base::StaticK<Bf16A16TmaMmaSchedule<96, 96, 64, 48, 24, 3, 1>>;

} // namespace

LinearAddLaunch select_bf16_linear_add_n2304_k3072(std::int32_t tokens) {
    const Bf16AddRoutes routes;
    if (tokens <= 16) return base::select(tokens, routes);
    if (tokens <= 64) return routes.tma<TmaR64T16>();
    if (tokens <= 128) return routes.tma<base::TmaR64T32>();
    if (tokens <= 224) return routes.tma<base::TmaR96T32>();
    if (tokens <= 256) return base::select(tokens, routes);
    if (tokens <= 288) return routes.tma<TmaR128T32>();
    if (tokens <= 448) return routes.tma<base::TmaR96T64>();
    if (tokens <= 512) return routes.tma<TmaR64T128>();
    if (tokens <= 576) return base::select(tokens, routes);
    if (tokens <= 672) return routes.tma<TmaR96T96>();
    if (tokens <= 896) return routes.tma<base::TmaR96T64>();
    if (tokens <= 1152) return base::select(tokens, routes);
    if (tokens <= 1344) return routes.tma<TmaR96T96>();
    return routes.tma_with_epilogue<base::TmaR128T64, Bf16ResidualAddWarpStoreEpilogue>();
}

} // namespace ninfer::ops::detail
