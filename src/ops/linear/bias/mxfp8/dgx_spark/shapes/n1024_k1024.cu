#include "ops/linear/bias/mxfp8/dispatch.h"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/mxfp8/a8.cuh"
#include "ops/linear/kernels/mxfp8/a8_tma.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

#include <array>
#include <limits>

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = NativeFp8ScheduleInstance<S, 1024>;

template <class S>
void simt(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& y, Mxfp8A8Workspace,
          cudaStream_t stream) {
    launch_native_fp8_a16_simt<Static<S>>(
        mxfp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
        LinearBiasEpilogue{static_cast<const __nv_bfloat16*>(bias.data)}, stream);
}

template <class S>
void a16(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& y, Mxfp8A8Workspace,
         cudaStream_t stream) {
    launch_native_fp8_a16_mma<Static<S>>(
        mxfp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
        LinearBiasEpilogue{static_cast<const __nv_bfloat16*>(bias.data)}, stream);
}

template <class S>
void a8(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& y, Mxfp8A8Workspace scratch,
        cudaStream_t stream) {
    launch_mxfp8_a8_quantize(static_cast<const __nv_bfloat16*>(x.data), w.k, x.ne[1], scratch,
                             stream);
    launch_mxfp8_a8_mma<Static<S>>(mxfp8_a8_operands(w, scratch, x.ne[1]),
                                   LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
                                   LinearBiasEpilogue{static_cast<const __nv_bfloat16*>(bias.data)},
                                   stream);
}

template <class S>
void sliced(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& y, Mxfp8A8Workspace,
            cudaStream_t stream) {
    launch_native_fp8_a16_sliced_k<Static<S>>(
        mxfp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
        LinearBiasEpilogue{static_cast<const __nv_bfloat16*>(bias.data)}, stream);
}

template <class S>
void tma(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& y, Mxfp8A8Workspace scratch,
         cudaStream_t stream) {
    launch_mxfp8_a8_quantize(static_cast<const __nv_bfloat16*>(x.data), w.k, x.ne[1], scratch,
                             stream);
    launch_mxfp8_a8_tma<Static<S>>(
        mxfp8_a8_operands(w, scratch, x.ne[1]),
        LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
        Bf16BiasWarpStoreEpilogue{{}, static_cast<const __nv_bfloat16*>(bias.data)}, stream);
}

constexpr int kAllTokens = std::numeric_limits<int>::max();
constexpr std::array<Mxfp8BiasRoute, 5> kA16{{
    {8, sliced<NativeFp8A16SlicedKSchedule<32, 8, 4, 128, 2>>, false},
    {32, sliced<NativeFp8A16SlicedKSchedule<16, 16, 2, 128, 2>>, false},
    {256, a16<NativeFp8MmaSchedule<32, 32, 128, 16, 16, 2>>, false},
    {512, a16<NativeFp8MmaSchedule<64, 64, 128, 32, 16, 2>>, false},
    {kAllTokens, a16<NativeFp8MmaSchedule<64, 128, 128, 32, 32, 2>>, false},
}};
constexpr std::array<Mxfp8BiasRoute, 5> kA8{{
    kA16[0],
    {32, a8<NativeFp8MmaSchedule<32, 8, 128, 16, 8, 2>>, true},
    {64, a8<NativeFp8MmaSchedule<64, 16, 128, 32, 8, 3>>, true},
    {256, a8<NativeFp8MmaSchedule<64, 32, 128, 32, 16, 2>>, true},
    {kAllTokens, tma<Mxfp8A8TmaSchedule<128, 128, 128, 64, 64, 2>>, true},
}};
} // namespace

const Mxfp8BiasShape kMxfp8BiasN1024K1024{1024, 1024, kA16, kA8};

} // namespace ninfer::ops::detail
