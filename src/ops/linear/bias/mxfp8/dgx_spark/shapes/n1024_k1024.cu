#include "ops/linear/bias/mxfp8/dispatch.h"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/mxfp8/a8.cuh"
#include "ops/linear/kernels/mxfp8/a8_tma.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_bias_vector_epilogue.cuh"

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
void sliced(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& y, Mxfp8A8Workspace,
            cudaStream_t stream) {
    launch_native_fp8_a16_sliced_k<Static<S>>(
        mxfp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
        LinearBiasEpilogue{static_cast<const __nv_bfloat16*>(bias.data)}, stream);
}

struct VectorOutput : LinearBf16Output {
    __device__ __forceinline__ void store_vector(int row, int token, uint4 v) const {
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(at(row, token)), "r"(v.x),
                     "r"(v.y), "r"(v.z), "r"(v.w)
                     : "memory");
    }
};

using VectorBias = Bf16BiasVectorEpilogue<1024>;

template <class S, bool Padded = false, bool Vector = false>
void tma(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& y, Mxfp8A8Workspace scratch,
         cudaStream_t stream) {
    using C   = Static<S>;
    int codes = Padded ? mxfp8_a8_padded_tokens(x.ne[1], C::kBlockTokens) : x.ne[1];
    launch_mxfp8_a8_quantize<__nv_bfloat16, Padded>(static_cast<const __nv_bfloat16*>(x.data), w.k,
                                                    x.ne[1], scratch, stream, codes);
    if constexpr (Vector)
        launch_mxfp8_a8_tma<C, Padded>(mxfp8_a8_operands(w, scratch, x.ne[1]),
                                       VectorOutput{{static_cast<__nv_bfloat16*>(y.data), w.n}},
                                       VectorBias{{}, static_cast<const __nv_bfloat16*>(bias.data)},
                                       stream);
    else
        launch_mxfp8_a8_tma<C, Padded>(
            mxfp8_a8_operands(w, scratch, x.ne[1]),
            LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
            Bf16BiasWarpStoreEpilogue{{}, static_cast<const __nv_bfloat16*>(bias.data)}, stream);
}

template <class S>
struct RegisterWeights : S {
    static constexpr bool kRegisterWeights = true;
};

template <class S>
void warp_mma(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& y, Mxfp8A8Workspace,
              cudaStream_t stream) {
    launch_native_fp8_a16_mma<Static<S>>(
        mxfp8_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
        Bf16BiasWarpStoreEpilogue{{}, static_cast<const __nv_bfloat16*>(bias.data)}, stream);
}

template <class S, bool Vector = false>
constexpr Mxfp8BiasRoute padded(int last) {
    return {last, tma<S, true, Vector>, true, S::kBlockTokens};
}

template <class S, class P>
__global__
__launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void mxfp8_bias_prefetch_sliced_kernel(
    P p, LinearBf16Output out, LinearBiasEpilogue epi) {
    int tt  = (p.tokens + S::kBlockTokens - 1) / S::kBlockTokens;
    int row = blockIdx.x / tt, token = blockIdx.x % tt * S::kBlockTokens;
    if (threadIdx.x == 0)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(epi.bias + row * S::kBlockRows));
    using Storage = NativeFp8SlicedKStorage<S, __nv_bfloat16, std::uint8_t>;
    native_fp8_a16_sliced_k_body<S>(p.x, p.codes, p.scales, p.rows, p.k, p.tokens, out, epi,
                                    Fp8IdentityRows{}, row, token,
                                    common::shared_storage<Storage::kBytes>());
}

template <class Schedule>
void prefetch_sliced(const Tensor& x, const Weight& w, const Tensor& bias, Tensor& y,
                     Mxfp8A8Workspace, cudaStream_t stream) {
    using S               = Static<Schedule>;
    auto p                = mxfp8_a16_operands(x, w);
    constexpr auto kernel = mxfp8_bias_prefetch_sliced_kernel<S, decltype(p)>;
    constexpr int bytes   = NativeFp8SlicedKStorage<S, __nv_bfloat16, std::uint8_t>::kBytes;
    auto smem             = common::prepare_shared<bytes, kernel>();
    kernel<<<(w.n / S::kBlockRows) * ((x.ne[1] + S::kBlockTokens - 1) / S::kBlockTokens),
             S::kThreads, smem, stream>>>(
        p, LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n},
        LinearBiasEpilogue{static_cast<const __nv_bfloat16*>(bias.data)});
    CUDA_CHECK(cudaGetLastError());
}

constexpr int kAllTokens = std::numeric_limits<int>::max();
constexpr std::array<Mxfp8BiasRoute, 5> kA16{{
    {8, sliced<NativeFp8A16SlicedKSchedule<32, 8, 4, 128, 2>>, false},
    {32, sliced<NativeFp8A16SlicedKSchedule<16, 16, 2, 128, 2>>, false},
    {256, a16<NativeFp8MmaSchedule<32, 32, 128, 16, 16, 2>>, false},
    {512, a16<NativeFp8MmaSchedule<64, 64, 128, 32, 16, 2>>, false},
    {kAllTokens, a16<NativeFp8MmaSchedule<64, 128, 128, 32, 32, 2>>, false},
}};
using Decode1 = NativeFp8A16SimtSchedule<1, 4, 1, 16, 4, 2>;
using Decode2 = NativeFp8A16SimtSchedule<2, 4, 1, 16, 2, 2>;
using Decode3 = NativeFp8A16SimtSchedule<3, 4, 1, 16, 1, 2>;
using A8Near  = Mxfp8A8TmaSchedule<64, 64, 256, 32, 16, 2, 1, 2>;
using A8Mid   = Mxfp8A8TmaSchedule<64, 128, 128, 32, 32, 2, 1, 2>;
using A8Wide  = Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 2, true, 1>;
using A8Bulk  = Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2, 1, 8>;
constexpr std::array<Mxfp8BiasRoute, 12> kA8{{
    {1, simt<Decode1>, false, 1, false},
    {2, simt<Decode2>, false, 1, false},
    {3, simt<Decode3>, false, 1, false},
    {8, prefetch_sliced<NativeFp8A16SlicedKSchedule<8, 8, 4, 128, 2>>, false},
    {32, sliced<NativeFp8A16SlicedKSchedule<16, 16, 2, 128, 2>>, false},
    {64, warp_mma<RegisterWeights<NativeFp8MmaSchedule<16, 32, 128, 16, 8, 2, 2, 8>>>, false},
    padded<A8Near, true>(192),
    padded<A8Mid, true>(384),
    padded<A8Bulk, true>(768),
    {1024, tma<A8Wide>, true},
    padded<A8Bulk, true>(kAllTokens / A8Bulk::kBlockTokens * A8Bulk::kBlockTokens),
    {kAllTokens, tma<A8Bulk, false, true>, true},
}};
} // namespace

const Mxfp8BiasShape kMxfp8BiasN1024K1024{1024, 1024, kA16, kA8};
} // namespace ninfer::ops::detail
