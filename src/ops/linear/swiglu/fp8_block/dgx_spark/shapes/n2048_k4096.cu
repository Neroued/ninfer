#include "ops/linear/swiglu/fp8_block/dispatch.h"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"
#include "ops/linear/kernels/fp8_block/operands.h"
#include "ops/linear/swiglu/row_major_bf16_epilogue.cuh"
#include "ops/linear/kernels/fp8_block/a16_tma.cuh"
#include <limits>

namespace ninfer::ops::detail {
namespace {
template <class S>
struct PairWarps : S {
    static constexpr bool kPairAcrossWarps = true;
};

template <class S>
struct RegisterWeights : S {
    static constexpr bool kRegisterWeights = true;
};

template <class S>
struct BlockSwiGluN2048K4096Schedule : NativeFp8ScheduleInstance<S, 4096> {};

template <class S>
using Static = BlockSwiGluN2048K4096Schedule<S>;

template <class S>
void simt(const Tensor& x, const Weight& weight, Tensor& out, float limit, bool clamped,
          cudaStream_t stream) {
    const auto launch = [&]<class Product>(Product product) {
        launch_native_fp8_a16_simt<Static<S>>(
            fp8_block_a16_operands(x, weight),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2},
            BasicSwiGluPairEpilogue<Product>{product}, stream, SwiGluBranchRows<1>{});
    };
    if (clamped)
        launch(ClampedSwiGluProduct{limit});
    else
        launch(SwiGluProduct{});
}

template <class S>
void mma(const Tensor& x, const Weight& weight, Tensor& out, float limit, bool clamped,
         cudaStream_t stream) {
    const auto launch = [&]<class Product>(Product product) {
        launch_native_fp8_a16_mma<Static<S>>(
            fp8_block_a16_operands(x, weight),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2},
            BasicSwiGluInterleavedMmaEpilogue<Product>{product}, stream,
            SwiGluInterleavedRows<8>{});
    };
    if (clamped)
        launch(ClampedSwiGluProduct{limit});
    else
        launch(SwiGluProduct{});
}

template <class S>
void sliced(const Tensor& x, const Weight& weight, Tensor& out, float limit, bool clamped,
            cudaStream_t stream) {
    const auto launch = [&]<class Product>(Product product) {
        launch_native_fp8_a16_sliced_k<Static<S>>(
            fp8_block_a16_operands(x, weight),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2},
            BasicSwiGluPairEpilogue<Product>{product}, stream,
            SwiGluBranchRows<S::kBlockRows / 2>{});
    };
    if (clamped)
        launch(ClampedSwiGluProduct{limit});
    else
        launch(SwiGluProduct{});
}

// Full token tiles use the ordinary vector stores. Only the final tile needs
// masked output and finite dummy products for the discarded columns.
template <class Product>
struct TailEpilogue
    : BasicSwiGluPaddedBf16Epilogue<Product, 1024, BasicSwiGluRowMajorBf16VectorEpilogue> {
    template <class S, bool Full, class Output>
    __device__ __forceinline__ void finish_tile(Output output, unsigned char* scratch,
                                                float (&accum)[S::kMmaRows][S::kMmaTokens][4],
                                                int row, int token, int rows, int tokens) const {
        if (this->logical_tokens - token >= S::kBlockTokens)
            BasicSwiGluRowMajorBf16VectorEpilogue<Product>{this->product}
                .template finish_tile<S, true>(output, scratch, accum, row, token, rows, tokens);
        else
            BasicSwiGluPaddedBf16Epilogue<Product, 1024, BasicSwiGluRowMajorBf16VectorEpilogue>::
                template finish_tile<S, Full>(output, scratch, accum, row, token, rows, tokens);
    }
};

template <class Schedule, int TokenAlignment = Schedule::kBlockTokens>
void tma(const Tensor& x, const Weight& weight, Tensor& out, float limit, bool clamped,
         cudaStream_t stream) {
    using S = Static<Schedule>;
    static_assert(TokenAlignment % S::kBlockTokens == 0);
    static_assert(2048 % S::kBlockRows == 0);
    auto p            = fp8_block_a16_operands(x, weight);
    const int logical = p.tokens;
    const auto output = LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), 1024};
    const auto run    = [&]<class Product>(Product product) {
        if (logical > std::numeric_limits<int>::max() / TokenAlignment * TokenAlignment) {
            launch_fp8_block_a16_tma<S, true>(
                p, output, BasicSwiGluRowMajorBf16VectorEpilogue<Product>{product}, stream);
            return;
        }
        // The descriptor retains the real input extent: out-of-range TMA rows
        // supply zero without allocating or reading an activation padding buffer.
        const auto maps = fp8_block_a16_tma_descriptors<S, true>(p);
        p.tokens =
            int((std::int64_t(logical) + TokenAlignment - 1) / TokenAlignment * TokenAlignment);
        using E             = TailEpilogue<Product>;
        constexpr auto fn   = fp8_block_a16_tma_kernel<S, true, LinearBf16Output, E, true>;
        constexpr int bytes = fp8_block_a16_tma_bytes<S, E>;
        static_assert(bytes <= 99 * 1024);
        const auto blocks = std::int64_t(p.rows / S::kBlockRows) * (p.tokens / S::kBlockTokens);
        if (blocks > std::numeric_limits<int>::max())
            throw std::invalid_argument("FP8 block SwiGLU grid exceeds CUDA grid.x");
        common::prepare_shared<bytes, fn, true>();
        fn<<<unsigned(blocks), S::kThreads, bytes, stream>>>(maps, p, output, E{{product, logical}},
                                                                0);
        CUDA_CHECK(cudaGetLastError());
    };
    if (clamped)
        run(ClampedSwiGluProduct{limit});
    else
        run(SwiGluProduct{});
}
} // namespace

Fp8BlockSwiGluLaunch select_fp8_block_swiglu_n2048_k4096(int tokens) {
    if (tokens <= 2) return simt<PairWarps<NativeFp8A16SimtSchedule<2, 4, 1, 16>>>;
    if (tokens <= 16) return sliced<NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 1>>;
    if (tokens <= 64)
        return mma<RegisterWeights<NativeFp8MmaSchedule<48, 32, 128, 16, 16, 2, 2, 2>>>;
    if (tokens <= 96) return tma<Fp8BlockA16TmaSchedule<64, 32, 128, 32, 16, 3, 2>>;
    if (tokens <= 128)
        return mma<RegisterWeights<NativeFp8MmaSchedule<96, 64, 128, 32, 16, 2, 1, 2>>>;
    if (tokens <= 192)
        return mma<RegisterWeights<NativeFp8MmaSchedule<128, 64, 128, 32, 32, 2, 1, 2>>>;
    if (tokens <= 256) return tma<Fp8BlockA16TmaSchedule<64, 64, 64, 32, 16, 3, 2>>;
    if (tokens <= 384) return tma<Fp8BlockA16TmaSchedule<128, 128, 64, 64, 32, 2, 2>>;
    if (tokens <= 576 || (tokens > 768 && tokens <= 960))
        return tma<Fp8BlockA16TmaSchedule<128, 64, 64, 64, 16, 3, 8>>;
    if (tokens <= 768) return tma<Fp8BlockA16TmaSchedule<128, 128, 64, 64, 32, 2, 8>, 384>;
    return tma<Fp8BlockA16TmaSchedule<128, 128, 64, 64, 32, 2, 8>>;
}
} // namespace ninfer::ops::detail
