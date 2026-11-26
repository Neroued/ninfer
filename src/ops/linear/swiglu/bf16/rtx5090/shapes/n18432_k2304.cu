#include "ops/linear/swiglu/bf16/paired.cuh"
#include "ops/linear/swiglu/dispatch.h"

namespace ninfer::ops::detail {
namespace {

constexpr int kInputRows    = 2304;
constexpr int kIntermediate = 9216;

template <class Schedule>
using StaticK = Bf16ScheduleInstance<Schedule, kInputRows>;

using Sliced32 = SwiGluBf16SlicedSchedule<
    StaticK<Bf16A16SlicedKMmaSchedule<16, 32, 4, 64, 2, Cache::cg, Cache::cg, 1, 1,
                                      Bf16MmaRaster::TokenFast>>,
    kIntermediate>;
using Tma16 =
    StaticK<Bf16A16TmaMmaSchedule<64, 16, 128, 16, 8, 2, 2, Bf16MmaRaster::TokenFast, 1, 4>>;
using Tma32 = StaticK<Bf16A16TmaMmaSchedule<128, 32, 64, 16, 16, 3, 1>>;
using Tma48 =
    StaticK<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 2, 1, Bf16MmaRaster::RowFast, 1, 2>>;
using Tma128Wide =
    StaticK<Bf16A16TmaMmaSchedule<128, 128, 64, 128, 16, 3, 1, Bf16MmaRaster::RowFast, 1, 2>>;
using Tma144    = StaticK<Bf16A16TmaMmaSchedule<128, 144, 64, 32, 48, 2, 1>>;
using Tma96Wide = StaticK<Bf16A16TmaMmaSchedule<256, 96, 64, 64, 32, 2, 1>>;
using Tma96 =
    StaticK<Bf16A16TmaMmaSchedule<64, 96, 64, 64, 24, 2, 2, Bf16MmaRaster::Grouped, 4, 2>>;
using Tma64Wide =
    StaticK<Bf16A16TmaMmaSchedule<128, 64, 64, 128, 16, 2, 1, Bf16MmaRaster::RowFast, 1, 2>>;
using Tma128 =
    StaticK<Bf16A16TmaMmaSchedule<64, 128, 64, 64, 32, 2, 1, Bf16MmaRaster::Grouped, 4, 2>>;

using TmaLaunch = void (*)(const Bf16A16Operands&, LinearBf16Output, SwiGluProduct, cudaStream_t);

TmaLaunch select_tma(int tokens) {
    if (tokens <= 32) return launch_swiglu_bf16_tma<Tma16, kIntermediate>;
    if (tokens <= 64) return launch_swiglu_bf16_tma<Tma32, kIntermediate>;
    if (tokens <= 96) return launch_swiglu_bf16_tma<Tma48, kIntermediate>;
    if (tokens <= 128) return launch_swiglu_bf16_tma<Tma128Wide, kIntermediate>;
    if (tokens <= 144) return launch_swiglu_bf16_tma<Tma144, kIntermediate>;
    if (tokens <= 192) return launch_swiglu_bf16_tma<Tma96Wide, kIntermediate>;
    if (tokens <= 256) return launch_swiglu_bf16_tma<Tma128, kIntermediate>;
    if (tokens <= 384) return launch_swiglu_bf16_tma<Tma96, kIntermediate>;
    if (tokens <= 640) return launch_swiglu_bf16_tma<Tma64Wide, kIntermediate>;
    if (tokens <= 768) return launch_swiglu_bf16_tma<Tma96, kIntermediate>;
    if (tokens <= 960) return launch_swiglu_bf16_tma<Tma64Wide, kIntermediate>;
    if (tokens <= 1024) return launch_swiglu_bf16_tma<Tma128, kIntermediate>;
    if (tokens <= 1056) return launch_swiglu_bf16_tma<Tma96, kIntermediate>;
    if (tokens <= 1152) return launch_swiglu_bf16_tma<Tma64Wide, kIntermediate>;
    return launch_swiglu_bf16_tma<Tma128, kIntermediate>;
}

} // namespace

void launch_bf16_swiglu_n18432_k2304(const Tensor& x, const Weight& w, Tensor& y,
                                     cudaStream_t stream) {
    const auto p = bf16_a16_operands(x, w);
    const LinearBf16Output out{static_cast<__nv_bfloat16*>(y.data), kIntermediate};
    if (p.tokens == 1) {
        swiglu_bf16_paired_gemv<kInputRows, kIntermediate>
            <<<kIntermediate / 4, 128, 0, stream>>>(p.x, p.weight, out, SwiGluProduct{});
        CUDA_CHECK(cudaGetLastError());
    } else if (p.tokens <= 16) {
        launch_bf16_a16_sliced_k_mma<Sliced32>(p, out, SwiGluBf16FragmentEpilogue<SwiGluProduct>{},
                                               stream);
    } else {
        select_tma(p.tokens)(p, out, SwiGluProduct{}, stream);
    }
}

} // namespace ninfer::ops::detail
