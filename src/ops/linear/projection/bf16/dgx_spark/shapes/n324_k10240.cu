#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Base>
struct Bf16HcDownInjectSchedule : Base {};
template <class S>
using Native = Bf16HcDownInjectSchedule<Bf16ScheduleInstance<S, 10240>>;
template <class S>
using Tail = Bf16HcDownInjectSchedule<Bf16RowTailSchedule<Bf16ScheduleInstance<S, 10240>>>;
using Gemv =
    Native<Bf16A16GemvSchedule<2, 2, 1, 16, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>;
using SimtTwo = Native<
    Bf16A16SimtSchedule<8, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>>;
using Simt = Native<
    Bf16A16SimtSchedule<2, 2, 1, 16, 4, 2, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2, 2>>;
using Hot8       = Tail<Bf16A16SlicedKMmaSchedule<16, 8, 4, 128, 3>>;
using Hot16      = Tail<Bf16A16SlicedKMmaSchedule<16, 16, 4, 128, 3>>;
using Hot32      = Tail<Bf16A16SlicedKMmaSchedule<16, 32, 8, 64, 2>>;
using Late32     = Tail<Bf16A16SlicedKMmaSchedule<16, 32, 4, 64, 2>>;
using HotTma     = Tail<Bf16A16TmaMmaSchedule<16, 32, 128, 16, 8, 3, 1>>;
using Rows64Deep = Tail<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 6, 1, Bf16MmaRaster::RowFast, 1>>;
using Rows128Deep =
    Tail<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 16, 4, 1, Bf16MmaRaster::RowFast, 1>>;
using Rows128Two =
    Tail<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 16, 2, 1, Bf16MmaRaster::RowFast, 1>>;
} // namespace

Bf16Launch select_bf16_n324_k10240(std::int32_t tokens) {
    if (tokens <= 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 2) return launch_bf16_simt<SimtTwo>;
    if (tokens <= 4) return launch_bf16_simt<Simt>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Hot8>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<Hot16>;
    if (tokens <= 64) return launch_bf16_sliced_k_mma<Hot32>;
    if (tokens <= 96) return launch_bf16_sliced_k_mma<Late32>;
    if (tokens <= 128) return launch_bf16_tma_mma<HotTma>;
    if (tokens <= 512) return launch_bf16_tma_mma<Rows64Deep>;
    if (tokens <= 1024) return launch_bf16_tma_mma<Rows128Deep>;
    const std::int64_t token_tiles = (static_cast<std::int64_t>(tokens) + 63) / 64;
    const auto work64              = ((6 * token_tiles + 47) / 48) * 64 * 64;
    const auto work128_deep        = ((3 * token_tiles + 47) / 48) * 128 * 64;
    const auto work128_two         = ((3 * token_tiles + 95) / 96) * 2 * 128 * 64;
    // Prefer a complete single-CTA wave when two residents round up more work.
    if (work128_deep < work128_two && work128_deep <= work64)
        return launch_bf16_tma_mma<Rows128Deep>;
    // Matched traffic controls show about 25% fewer L2 sectors for the wider tile.
    // Keep its wave penalty within that measured reuse benefit; public curves cover seams.
    if (3 * work128_two <= 4 * work64) return launch_bf16_tma_mma<Rows128Two>;
    return launch_bf16_tma_mma<Rows64Deep>;
}
} // namespace ninfer::ops::detail
