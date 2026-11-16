#include "ops/linear/swiglu/dispatch.h"
#include "ops/linear/swiglu/bf16/paired.cuh"

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = Bf16ScheduleInstance<S, 4096>;
using Small  = SwiGluBf16SlicedSchedule<
     Static<Bf16A16SlicedKMmaSchedule<16, 32, 16, 32, 2, Cache::cg, Cache::cg, 1, 7,
                                      Bf16MmaRaster::TokenFast>>,
     6144>;
using Medium = SwiGluBf16SlicedSchedule<
    Static<Bf16A16SlicedKMmaSchedule<64, 32, 8, 32, 2, Cache::cg, Cache::cg, 1, 7,
                                     Bf16MmaRaster::TokenFast>>,
    6144>;
using Wide = SwiGluBf16SlicedSchedule<
    Static<Bf16A16SlicedKMmaSchedule<32, 64, 8, 32, 2, Cache::cg, Cache::cg, 1, 3,
                                     Bf16MmaRaster::TokenFast>>,
    6144>;
using Bulk = Static<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>>;
} // namespace

void launch_bf16_swiglu_n12288_k4096(const Tensor& x, const Weight& w, Tensor& y,
                                     cudaStream_t stream) {
    const auto p = bf16_a16_operands(x, w);
    const LinearBf16Output out{static_cast<__nv_bfloat16*>(y.data), 6144};
    if (p.tokens <= 64) {
        launch_bf16_a16_sliced_k_mma<Small>(p, out, SwiGluBf16FragmentEpilogue<SwiGluProduct>{},
                                            stream);
    } else if (p.tokens <= 96) {
        launch_bf16_a16_sliced_k_mma<Medium>(p, out, SwiGluBf16FragmentEpilogue<SwiGluProduct>{},
                                             stream);
    } else if (p.tokens <= 128) {
        launch_bf16_a16_sliced_k_mma<Wide>(p, out, SwiGluBf16FragmentEpilogue<SwiGluProduct>{},
                                           stream);
    } else {
        launch_swiglu_bf16_tma<Bulk, 6144>(p, out, SwiGluProduct{}, stream);
    }
}
} // namespace ninfer::ops::detail
