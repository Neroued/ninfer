#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_tma_register_b.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

#include <type_traits>

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 1536>;

using Gemv =
    Static<Bf16A16GemvSchedule<4, 1, 2, 16, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 4, 2>>;
using Short  = Static<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 2>>;
using Small  = Static<Bf16A16SlicedKMmaSchedule<16, 16, 8, 64, 1>>;
using Medium = Static<Bf16A16SlicedKMmaSchedule<16, 32, 8, 64, 1>>;
using Near   = Static<Bf16A16TmaMmaSchedule<32, 48, 64, 16, 16, 3, 3, Bf16MmaRaster::RowFast>>;
using Large  = Static<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3, 2, Bf16MmaRaster::RowFast>>;
using Larger = Static<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2, Bf16MmaRaster::RowFast>>;
using Wide   = Static<Bf16A16TmaMmaSchedule<256, 128, 64, 64, 32, 2, 1, Bf16MmaRaster::RowFast>>;
using Transition =
    Static<Bf16A16TmaRegisterBSchedule<128, 128, 64, 128, 16, 2, Bf16MmaRaster::RowFast>>;
using Bulk = Static<Bf16A16TmaRegisterBSchedule<128, 128, 64, 64, 32, 2, Bf16MmaRaster::RowFast>>;

template <class S, bool Streaming = false>
void tma(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<S>(bf16_a16_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                               std::conditional_t<Streaming, Bf16MmaStreamingWarpStoreEpilogue,
                                                  Bf16MmaWarpStoreEpilogue>{},
                               stream);
}

template <class S, bool Streaming = false>
void register_b(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_bf16_a16_tma_register_b<S>(
        bf16_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
        std::conditional_t<Streaming, Bf16MmaStreamingWarpStoreEpilogue,
                           Bf16MmaWarpStoreEpilogue>{},
        stream);
}
} // namespace

Bf16Launch select_bf16_n4096_k1536(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<Short>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<Small>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<Medium>;
    if (tokens <= 48) return tma<Near>;
    if (tokens <= 64) return tma<Large>;
    if (tokens <= 192) return tma<Larger>;
    // Sixteen row tiles fill the 48 SMs for every three token tiles. Avoid
    // a final wave occupying only one third of the SMs.
    if (tokens <= 1536 && ((tokens + 127) / 128) % 3 != 1) return tma<Wide, true>;
    if (tokens <= 512) return register_b<Transition>;
    if (tokens <= 1536) return register_b<Bulk>;
    if (tokens <= 2048) return register_b<Bulk, true>;
    return register_b<Bulk>;
}
} // namespace ninfer::ops::detail
