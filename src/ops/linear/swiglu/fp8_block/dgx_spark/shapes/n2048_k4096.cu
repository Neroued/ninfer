#include "ops/linear/swiglu/fp8_block/dispatch.h"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"
#include "ops/linear/kernels/fp8_block/operands.h"
#include "ops/linear/swiglu/product_epilogue.cuh"

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
using Static = NativeFp8ScheduleInstance<S, 4096>;

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
} // namespace

Fp8BlockSwiGluLaunch select_fp8_block_swiglu_n2048_k4096(int tokens) {
    if (tokens <= 2) return simt<PairWarps<NativeFp8A16SimtSchedule<2, 4, 1, 16>>>;
    if (tokens <= 16) return sliced<NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 1>>;
    if (tokens <= 64) return mma<NativeFp8MmaSchedule<32, 32, 128, 16, 16, 2>>;
    if (tokens <= 256) return mma<RegisterWeights<NativeFp8MmaSchedule<64, 64, 128, 16, 32, 2, 2>>>;
    return mma<RegisterWeights<NativeFp8MmaSchedule<128, 64, 128, 32, 32, 1, 2>>>;
}
} // namespace ninfer::ops::detail
