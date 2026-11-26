#include "ops/linear/pair/dispatch.h"
#include "ops/linear/pair/bf16/launch.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
using Small = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 32, 64, 16, 16, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    128>;

// K=128 has one load group and needs no refill buffer.
template <class Base>
struct WholeK : Base {
    static_assert(Base::kBlockK == 128);
    static constexpr int kStages         = 1;
    static constexpr int kSharedElements = (Base::kBlockRows + Base::kBlockTokens) * 128;
    static constexpr int kSharedBytes    = kSharedElements * sizeof(__nv_bfloat16);
};

template <int Tokens, int WarpRows, int WarpTokens>
using CompleteK = Bf16ScheduleInstance<
    WholeK<Bf16A16MmaSchedule<64, Tokens, 128, WarpRows, WarpTokens, 2, 2, Cache::cg, Cache::cg,
                              Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>,
    128>;

struct StreamingOutput : LinearBf16Output {
    __device__ __forceinline__ void store_vector(int row, int token, uint4 value) const {
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(at(row, token)), "r"(value.x),
                     "r"(value.y), "r"(value.z), "r"(value.w)
                     : "memory");
    }
};

struct TokenTileVectorEpilogue : Bf16TmaVectorEpilogue {
    template <class S, bool Full, class Output>
    __device__ __forceinline__ void finish_tile(const Output& output, unsigned char* scratch,
                                                float (&accum)[S::kMmaRows][S::kMmaTokens][4],
                                                int row, int token, int rows, int tokens) const {
        if (Full || tokens - token >= S::kBlockTokens)
            Bf16TmaVectorEpilogue::template finish_tile<S, true>(output, scratch, accum, row, token,
                                                                 rows, tokens);
        else
            Bf16TmaVectorEpilogue::template finish_tile<S, false>(output, scratch, accum, row,
                                                                  token, rows, tokens);
    }
};

template <class S, bool Streaming>
void tma(Bf16A16Operands a, Bf16A16Operands b, LinearBf16Output ya, LinearBf16Output yb,
         cudaStream_t stream) {
    using Output = std::conditional_t<Streaming, StreamingOutput, LinearBf16Output>;
    launch_bf16_pair_tma<S, TokenTileVectorEpilogue, Output>(a, b, ya, yb, stream);
}
} // namespace

void launch_bf16_pair_k128(const Tensor& x0, const Tensor& x1, const Weight& w0, const Weight& w1,
                           Tensor& y0, Tensor& y1, cudaStream_t stream) {
    const auto a = bf16_a16_operands(x0, w0), b = bf16_a16_operands(x1, w1);
    const LinearBf16Output first{static_cast<__nv_bfloat16*>(y0.data), w0.n},
        second{static_cast<__nv_bfloat16*>(y1.data), w1.n};
    if (a.rows == 4096) {
        if (a.tokens <= 32)
            return launch_bf16_pair_mma<CompleteK<16, 16, 8>, Bf16MmaWarpStoreEpilogue>(
                a, b, first, second, stream);
        if (a.tokens <= 160)
            return launch_bf16_pair_mma<CompleteK<32, 16, 16>, Bf16MmaWarpStoreEpilogue>(
                a, b, first, second, stream);
        if (a.tokens <= 256)
            return launch_bf16_pair_mma<CompleteK<64, 32, 16>, Bf16MmaWarpStoreEpilogue>(
                a, b, first, second, stream);
        if (a.tokens <= 2048)
            return tma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2,
                                                                  Bf16MmaRaster::Grouped, 1, 3>,
                                            128>,
                       true>(a, b, first, second, stream);
        return tma<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 2,
                                                              Bf16MmaRaster::Grouped, 8, 3>,
                                        128>,
                   false>(a, b, first, second, stream);
    }
    if (a.tokens <= 128)
        return launch_bf16_pair_mma<Small, LinearIdentityEpilogue>(a, b, first, second, stream);
    const auto run = [&]<class S>() {
        launch_bf16_a16_tma_mma<S>(a, first, Bf16TmaVectorEpilogue{}, stream);
        launch_bf16_a16_tma_mma<S>(b, second, Bf16TmaVectorEpilogue{}, stream);
    };
    if (a.tokens < 4096)
        run.template
        operator()<Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2>, 128>>();
    else
        run.template operator()<Bf16ScheduleInstance<
            Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 2, Bf16MmaRaster::Grouped, 8, 3>, 128>>();
}
} // namespace ninfer::ops::detail
