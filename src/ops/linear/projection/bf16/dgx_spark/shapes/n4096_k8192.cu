#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_tma_register_b.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 8192>;
template <class Schedule>
using Predicated = Bf16KTailSchedule<Static<Schedule>>;
using Gemv =
    Static<Bf16A16GemvSchedule<8, 4, 1, 16, 4, Bf16ActivationAccess::Direct,
                               Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 4, 2>>;
using Decode = Static<Bf16A16SlicedKMmaSchedule<16, 8, 16, 64, 2>>;
using S16    = Static<Bf16A16SlicedKMmaSchedule<16, 16, 8, 128, 1>>;
using S32    = Static<Bf16A16SlicedKMmaSchedule<16, 32, 8, 64, 2>>;
using M64    = Static<Bf16A16MmaSchedule<32, 64, 256, 16, 16, 2, 1, Cache::cg, Cache::cg,
                                         Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;

template <class Base>
struct ShiftTokenTail : Base {
    static constexpr bool kShiftTokenTail = true;
};

using Medium = Static<ShiftTokenTail<
    Bf16A16TmaRegisterBSchedule<128, 128, 64, 64, 32, 2, Bf16MmaRaster::Grouped, 4>>>;
using Long   = Static<ShiftTokenTail<
      Bf16A16TmaRegisterBSchedule<128, 128, 64, 128, 16, 2, Bf16MmaRaster::Grouped, 4>>>;
using Bridge = Static<ShiftTokenTail<
    Bf16A16TmaRegisterBSchedule<128, 96, 64, 128, 16, 2, Bf16MmaRaster::Grouped, 4>>>;
using Window = Static<
    ShiftTokenTail<Bf16A16TmaRegisterBSchedule<64, 128, 64, 64, 32, 3, Bf16MmaRaster::Grouped, 4>>>;
using T256 = Predicated<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1>>;
using Bulk =
    Predicated<Bf16A16TmaMmaSchedule<128, 256, 64, 64, 64, 2, 1, Bf16MmaRaster::Grouped, 12, 3>>;

struct TokenWindowEpilogue : Bf16MmaWarpStoreEpilogue {
    template <class Schedule, bool Full, class Output>
    __device__ __forceinline__ void
    finish_token_window(const Output& output, unsigned char* scratch,
                        float (&accum)[Schedule::kMmaRows][Schedule::kMmaTokens][4], int row_begin,
                        int token_begin, int rows, int tokens, int first_token) const {
        Bf16MmaCtaWarpStoreEpilogue<>{}.template finish_token_window<Schedule, Full>(
            output, scratch, accum, row_begin, token_begin, rows, tokens, first_token);
    }
};

template <class Schedule>
void register_b(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_register_b<Schedule>(
        bf16_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, TokenWindowEpilogue{},
        stream);
}

template <class Schedule>
void warp(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(
        bf16_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
        Bf16MmaWarpStoreEpilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n4096_k8192(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Gemv>;
    if (tokens <= 8) return launch_bf16_sliced_k_mma<Decode>;
    if (tokens <= 16) return launch_bf16_sliced_k_mma<S16>;
    if (tokens <= 32) return launch_bf16_sliced_k_mma<S32>;
    if (tokens <= 192) return launch_bf16_mma<M64>;
    if (tokens <= 384) return register_b<Medium>;
    if (tokens <= 512) return register_b<Window>;
    if (tokens <= 576) return register_b<Bridge>;
    if (tokens <= 640) return warp<T256>;
    if (tokens <= 768) return register_b<Long>;
    if (tokens <= 1024) return warp<T256>;
    if (tokens <= 1536) return register_b<Long>;
    return warp<Bulk>;
}
} // namespace ninfer::ops::detail
