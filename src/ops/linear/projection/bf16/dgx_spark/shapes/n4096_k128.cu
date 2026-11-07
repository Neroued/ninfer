#include "ops/linear/projection/bf16/bf16_shapes.h"
#include "ops/linear/projection/bf16/bf16_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Base>
struct Bf16N4096K128Schedule : Base {};
template <class Schedule>
using Static = Bf16ScheduleInstance<Schedule, 128>;

// A complete K tile has no refill: one staging buffer is sufficient.
template <class Base>
struct WholeK : Base {
    static_assert(Base::kBlockK == 128);
    static constexpr int kStages         = 1;
    static constexpr int kSharedElements = (Base::kBlockRows + Base::kBlockTokens) * 128;
    static constexpr int kSharedBytes    = kSharedElements * sizeof(__nv_bfloat16);
};

using Decode =
    Static<WholeK<Bf16A16MmaSchedule<64, 16, 128, 16, 8, 2, 2, Cache::cg, Cache::cg,
                                     Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>>;
using Small =
    Static<Bf16A16MmaSchedule<64, 32, 64, 16, 16, 2, 2, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;
using Medium = Bf16N4096K128Schedule<Static<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2>>>;
using Bulk   = Bf16N4096K128Schedule<
      Static<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 2, Bf16MmaRaster::Grouped, 8, 3>>>;

void decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_mma<Decode>(bf16_a16_operands(x, weight),
                                LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
                                Bf16MmaWarpStoreEpilogue{}, stream);
}

struct TokenTileVectorEpilogue : Bf16TmaVectorEpilogue {
    template <class Schedule, bool Full, class Output>
    __device__ __forceinline__ void
    finish_tile(const Output& output, unsigned char* scratch,
                float (&accum)[Schedule::kMmaRows][Schedule::kMmaTokens][4], int row_begin,
                int token_begin, int rows, int tokens) const {
        if constexpr (Full) {
            Bf16TmaVectorEpilogue::template finish_tile<Schedule, true>(
                output, scratch, accum, row_begin, token_begin, rows, tokens);
        } else if (token_begin + Schedule::kBlockTokens <= tokens) {
            Bf16TmaVectorEpilogue::template finish_tile<Schedule, true>(
                output, scratch, accum, row_begin, token_begin, rows, tokens);
        } else {
            Bf16TmaVectorEpilogue::template finish_tile<Schedule, false>(
                output, scratch, accum, row_begin, token_begin, rows, tokens);
        }
    }
};

template <class Schedule, class Epilogue = Bf16TmaVectorEpilogue>
void vector_store(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule>(
        bf16_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, Epilogue{}, stream);
}
} // namespace

Bf16Launch select_bf16_n4096_k128(std::int32_t tokens) {
    if (tokens <= 32) return decode;
    if (tokens <= 128) return launch_bf16_mma<Small>;
    if (tokens < 4096) {
        if (tokens % Medium::kBlockTokens) return vector_store<Medium, TokenTileVectorEpilogue>;
        return vector_store<Medium>;
    }
    return vector_store<Bulk>;
}
} // namespace ninfer::ops::detail
