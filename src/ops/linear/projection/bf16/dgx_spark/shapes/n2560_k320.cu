#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Base>
struct N2560K320 : Bf16ScheduleInstance<Base, 320> {};

using Small =
    N2560K320<Bf16A16MmaSchedule<64, 16, 64, 16, 8, 3, 2, Cache::cg, Cache::cg,
                                 Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Medium = N2560K320<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>>;
using Wide   = N2560K320<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2>>;
using Bulk   = N2560K320<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 2>>;
using BulkRows =
    N2560K320<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2, Bf16MmaRaster::RowFast>>;

template <class S>
void warp_store(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                               Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n2560_k320(std::int32_t tokens) {
    if (tokens <= 32) return launch_bf16_mma<Small>;
    if (tokens <= 64) return warp_store<Medium>;
    if (tokens <= 128) return warp_store<Wide>;
    if (tokens <= 2048) return warp_store<Bulk>;
    return warp_store<BulkRows>;
}
} // namespace ninfer::ops::detail
