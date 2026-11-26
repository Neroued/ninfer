#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/projection/bf16/rtx5090/shapes/n32_k2304.cuh"

namespace ninfer::ops::detail {
namespace {

using bf16_n32_k2304::StaticK;
using Sliced8 = StaticK<Bf16A16SlicedKMmaSchedule<16, 8, 6, 128, 2, Cache::cg, Cache::cg, 1, 1>>;
using Tma16   = StaticK<Bf16A16TmaMmaSchedule<16, 16, 256, 16, 8, 3>>;
using Tma32   = StaticK<Bf16A16TmaMmaSchedule<16, 32, 128, 16, 8, 3>>;
using Tma64   = StaticK<Bf16A16TmaMmaSchedule<32, 64, 128, 16, 16, 3>>;

} // namespace

Bf16Launch select_bf16_n32_k2304(std::int32_t tokens) {
    if (tokens <= 16) return launch_bf16_simt<bf16_n32_k2304::DecodeSimt>;
    if (tokens <= 384) return launch_bf16_sliced_k_mma<Sliced8>;
    if (tokens <= 768) return launch_bf16_sliced_k_mma<bf16_n32_k2304::Sliced16>;
    if (tokens <= 1152) return launch_bf16_tma_mma<Tma16, Bf16MmaWarpStoreEpilogue>;
    if (tokens <= 3584) return launch_bf16_tma_mma<Tma32>;
    // One row tile per CTA avoids loading each activation tile twice at large T.
    if (tokens <= 8192) return launch_bf16_tma_mma<Tma64>;
    return launch_bf16_tma_mma<bf16_n32_k2304::Tma128>;
}

} // namespace ninfer::ops::detail
