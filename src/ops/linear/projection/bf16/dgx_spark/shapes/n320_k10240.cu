#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Base>
struct Bf16HcFinalDownSchedule : Base {};
template <class S>
using Native = Bf16HcFinalDownSchedule<Bf16ScheduleInstance<S, 10240>>;
template <class S>
using Tail = Bf16HcFinalDownSchedule<Bf16RowTailSchedule<Bf16ScheduleInstance<S, 10240>>>;
using Gemv =
    Native<Bf16A16GemvSchedule<2, 2, 1, 16, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using Simt = Native<
    Bf16A16SimtSchedule<2, 2, 1, 16, 4, 2, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2, 2>>;
using Hot8   = Native<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 3>>;
using Hot16  = Native<Bf16A16SlicedKMmaSchedule<16, 16, 4, 128, 3>>;
using Hot32  = Native<Bf16A16SlicedKMmaSchedule<16, 32, 8, 64, 2>>;
using Late32 = Native<Bf16A16SlicedKMmaSchedule<16, 32, 4, 64, 2>>;
using Rows64Deep =
    Native<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 6, 1, Bf16MmaRaster::RowFast, 1>>;
using Rows64Two =
    Native<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 1, Bf16MmaRaster::RowFast, 1>>;
using Rows128Deep =
    Tail<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 16, 4, 1, Bf16MmaRaster::RowFast, 1>>;
using Rows128Two =
    Tail<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 16, 2, 1, Bf16MmaRaster::RowFast, 1>>;

template <class S>
void launch_warp(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    static_assert(!bf16_predicated_rows<S>);
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                               Bf16MmaWarpStoreEpilogue{}, stream);
}

template <class S>
void launch_hot_warp(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    static_assert(!bf16_predicated_rows<S>);
    launch_bf16_a16_sliced_k_mma<S>(bf16_a16_operands(x, w),
                                    LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                    Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n320_k10240(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 4) return launch_bf16_simt<Simt>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Hot8>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<Hot16>;
    if (tokens <= 64) return launch_hot_warp<Hot32>;
    if (tokens <= 128) return launch_bf16_sliced_k_mma<Late32>;
    if (tokens <= 576) return launch_bf16_tma_mma<Rows64Deep>;
    if (tokens <= 768) return launch_bf16_tma_mma<Rows128Deep>;
    if (tokens <= 1216) return launch_warp<Rows64Two>;
    const std::int64_t token_tiles = (static_cast<std::int64_t>(tokens) + 63) / 64;
    const auto work64              = ((5 * token_tiles + 47) / 48) * 64 * 64;
    const auto work128_deep        = ((3 * token_tiles + 47) / 48) * 128 * 64;
    const auto work128_two         = ((3 * token_tiles + 95) / 96) * 2 * 128 * 64;
    // Prefer a complete single-CTA wave when two residents round up more work.
    if (work128_deep < work128_two && work128_deep <= work64)
        return launch_bf16_tma_mma<Rows128Deep>;
    // Predicating 384 physical rows adds work for N320; retain the stricter wave guard.
    if (work128_two <= work64) return launch_bf16_tma_mma<Rows128Two>;
    return launch_bf16_tma_mma<Rows64Deep>;
}
} // namespace ninfer::ops::detail
