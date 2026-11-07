#include "ops/linear/pair/dispatch.h"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"
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

struct PairDescriptors {
    Bf16TmaDescriptors first, second;
};

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

template <class S, bool Full, class Epilogue>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void bf16_pair_mma_kernel(
    Bf16A16Operands a, Bf16A16Operands b, LinearBf16Output ya, LinearBf16Output yb, int offset,
    int count) {
    const auto& p = blockIdx.y ? b : a;
    bf16_a16_mma_body<S, Full>(p.x, p.weight, blockIdx.y ? yb : ya, Epilogue{}, p.rows, p.k, offset,
                               count);
}

template <class S, class Epilogue>
void mma(Bf16A16Operands a, Bf16A16Operands b, LinearBf16Output ya, LinearBf16Output yb,
         cudaStream_t stream) {
    common::for_each_token_slice(a.tokens, S::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(a.rows / S::kBlockRows * common::div_up(count, S::kBlockTokens), 2);
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = bf16_pair_mma_kernel<S, Full, Epilogue>;
            const auto bytes =
                common::prepare_shared<linear_mma_shared_bytes<S, Epilogue>, kernel, true>();
            kernel<<<grid, S::kThreads, bytes, stream>>>(a, b, ya, yb, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % S::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

template <class S, bool Full, class Output>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void bf16_pair_tma_kernel(
    const __grid_constant__ PairDescriptors descriptors, Output ya, Output yb, int rows, int offset,
    int count) {
    // Keep both descriptors in launch-owned parameter storage; each CTA borrows one pair.
    const auto& maps = blockIdx.y ? descriptors.second : descriptors.first;
    bf16_a16_tma_mma_body<S, Full>(maps, blockIdx.y ? yb : ya, TokenTileVectorEpilogue{}, rows, 128,
                                   offset, count);
}

template <class S, bool Streaming>
void tma(Bf16A16Operands a, Bf16A16Operands b, LinearBf16Output ya, LinearBf16Output yb,
         cudaStream_t stream) {
    using Output = std::conditional_t<Streaming, StreamingOutput, LinearBf16Output>;
    Output first, second;
    first.data  = ya.data;
    first.rows  = ya.rows;
    second.data = yb.data;
    second.rows = yb.rows;
    const PairDescriptors maps{make_bf16_tma_descriptors<S>(a), make_bf16_tma_descriptors<S>(b)};
    common::for_each_token_slice(a.tokens, S::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(a.rows / S::kBlockRows * common::div_up(count, S::kBlockTokens), 2);
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = bf16_pair_tma_kernel<S, Full, Output>;
            constexpr int bytes =
                bf16_tma_scratch_bytes<S, TokenTileVectorEpilogue> + S::kBarrierBytes;
            common::prepare_shared<bytes, kernel, true>();
            kernel<<<grid, S::kThreads, bytes, stream>>>(maps, first, second, a.rows, offset,
                                                         count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % S::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}
} // namespace

void launch_bf16_pair(const Tensor& x0, const Tensor& x1, const Weight& w0, const Weight& w1,
                      Tensor& y0, Tensor& y1, cudaStream_t stream) {
    const auto a = bf16_a16_operands(x0, w0), b = bf16_a16_operands(x1, w1);
    const LinearBf16Output first{static_cast<__nv_bfloat16*>(y0.data), w0.n},
        second{static_cast<__nv_bfloat16*>(y1.data), w1.n};
    if (a.rows == 4096) {
        if (a.tokens <= 32)
            return mma<CompleteK<16, 16, 8>, Bf16MmaWarpStoreEpilogue>(a, b, first, second, stream);
        if (a.tokens <= 160)
            return mma<CompleteK<32, 16, 16>, Bf16MmaWarpStoreEpilogue>(a, b, first, second,
                                                                        stream);
        if (a.tokens <= 256)
            return mma<CompleteK<64, 32, 16>, Bf16MmaWarpStoreEpilogue>(a, b, first, second,
                                                                        stream);
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
    if (a.tokens <= 128) return mma<Small, LinearIdentityEpilogue>(a, b, first, second, stream);
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
