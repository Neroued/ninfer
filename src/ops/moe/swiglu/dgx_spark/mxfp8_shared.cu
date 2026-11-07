#include "ops/moe/swiglu/dispatch.h"
#include "ops/moe/swiglu/fp8_plan.h"
#include "ops/moe/swiglu/kernel/fp8_product.cuh"
#include "ops/moe/swiglu/kernel/output.cuh"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"
#include "ops/linear/kernels/mxfp8/a8.cuh"
#include "ops/linear/kernels/mxfp8/a8_tma.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail::swiglu_ffn {
namespace {
template <class Base, int K>
struct SharedSchedule : NativeFp8ScheduleInstance<Base, K> {};

struct DownEpilogue : Bf16MmaWarpStoreEpilogue {
    int logical_tokens;

    template <class Output>
    __device__ __forceinline__ void store_fragment(Output output, int row, int token, float4 value,
                                                   int rows, int tokens) const {
        const int end = min(tokens, logical_tokens);
        if constexpr (requires { output.combine(row, token, value.x); }) {
            // A padding column must not inspect the routed validity or value planes.
            if (token < end) {
                value.x = output.combine(row, token, value.x);
                value.z = output.combine(row + 8, token, value.z);
            }
            if (token + 1 < end) {
                value.y = output.combine(row, token + 1, value.y);
                value.w = output.combine(row + 8, token + 1, value.w);
            }
            Bf16MmaWarpStoreEpilogue::store_fragment(
                LinearBf16Output{output.destination, Output::kRows}, row, token, value, rows, end);
        } else {
            Bf16MmaWarpStoreEpilogue::store_fragment(output, row, token, value, rows, end);
        }
    }
};

template <class G, class D, class Output>
void float_product(const Tensor& x, const SwiGLUWeights& weights, Output output, float* product,
                   float limit, cudaStream_t stream) {
    launch_native_fp8_a16_sliced_k<SharedSchedule<G, 4096>>(
        mxfp8_a16_operands(x, weights.gate_up.weight), SwiGluFloatOutput{product, 1024},
        BasicSwiGluPairEpilogue<ClampedSwiGluProduct>{{limit}}, stream,
        SwiGluBranchRows<G::kBlockRows / 2>{});
    const auto& w = weights.down.weight;
    const Mxfp8A16Operands<float> down{product,
                                       static_cast<const std::uint8_t*>(w.qdata),
                                       static_cast<const std::uint8_t*>(w.scales),
                                       w.n,
                                       w.k,
                                       int(x.ne[1])};
    launch_native_fp8_a16_sliced_k<SharedSchedule<D, 1024>>(down, output, LinearIdentityEpilogue{},
                                                            stream, Fp8IdentityRows{});
}

struct RowProductEpilogue {
    ClampedSwiGluProduct product;
    int logical;
    template <class C>
    static constexpr int kSharedBytes = C::kBlockRows / 2 * C::kBlockTokens * 4;

    template <class C, bool Full>
    __device__ void finish_tile(Mxfp8ProductOutput output, unsigned char* scratch,
                                float (&acc)[C::kMmaRows][C::kMmaTokens][4], int rb, int tb,
                                int rows, int tokens) const {
        constexpr int P = C::kBlockRows / 2;
        static_assert(C::kWarpsRows == 2 && P % 32 == 0);
        auto* tile = (float*)scratch;
        int tid = threadIdx.x, warp = tid / 32, lane = tid & 31, wr = warp / C::kWarpsTokens,
            wt    = warp % C::kWarpsTokens;
        auto at   = [&](int r, int t) -> float& { return tile[t * P + (r ^ ((t & 6) << 2))]; };
        auto calc = [&](float g, float u) {
            g = fminf(g, product.limit);
            u = fmaxf(-product.limit, fminf(u, product.limit));
            return __fdividef(g, 1.F + __expf(-g)) * u;
        };
        if (wr == 1)
            for (int mi = 0; mi < C::kMmaRows; ++mi)
                for (int ni = 0; ni < C::kMmaTokens; ++ni) {
                    int r            = mi * 16 + (lane >> 2),
                        t            = wt * C::kWarpTokens + ni * 8 + 2 * (lane & 3);
                    auto& a          = acc[mi][ni];
                    at(r, t)         = a[0];
                    at(r, t + 1)     = a[1];
                    at(r + 8, t)     = a[2];
                    at(r + 8, t + 1) = a[3];
                }
        __syncthreads();
        if (wr == 0)
            for (int mi = 0; mi < C::kMmaRows; ++mi)
                for (int ni = 0; ni < C::kMmaTokens; ++ni) {
                    int r            = mi * 16 + (lane >> 2),
                        t            = wt * C::kWarpTokens + ni * 8 + 2 * (lane & 3);
                    auto& a          = acc[mi][ni];
                    at(r, t)         = tb + t < logical ? calc(a[0], at(r, t)) : 1.F;
                    at(r, t + 1)     = tb + t + 1 < logical ? calc(a[1], at(r, t + 1)) : 1.F;
                    at(r + 8, t)     = tb + t < logical ? calc(a[2], at(r + 8, t)) : 1.F;
                    at(r + 8, t + 1) = tb + t + 1 < logical ? calc(a[3], at(r + 8, t + 1)) : 1.F;
                }
        __syncthreads();
        for (int i = tid; i < C::kBlockTokens * (P / 4); i += C::kThreads) {
            int t = i / (P / 4), r = i % (P / 4) * 4;
            auto v  = common::load_vec<float4>(&at(r, t));
            float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
            for (int d = 4; d; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d, 8));
            auto sf = common::fp8_ue8m0_scale(m);
            if (rb + r + 3 < rows / 2 && (Full || tb + t < tokens)) {
                unsigned word =
                    common::fp8_quant_code2(v.x, v.y, sf.inverse_scale) |
                    (unsigned(common::fp8_quant_code2(v.z, v.w, sf.inverse_scale)) << 16);
                common::store_vec(output.values.codes + std::int64_t(tb + t) * output.rows + rb + r,
                                  word);
                if (!(lane & 7))
                    output.values
                        .scales[std::int64_t(tb + t) * (output.rows / 32) + (rb + r) / 32] =
                        sf.scale;
            }
        }
    }
};

struct StreamingOutput : LinearBf16Output {
    __device__ void store_vector(int r, int t, uint4 v) const {
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(at(r, t)), "r"(v.x), "r"(v.y),
                     "r"(v.z), "r"(v.w)
                     : "memory");
    }
};

struct VectorDownEpilogue : Bf16TmaVectorEpilogue {
    int logical_tokens;
    template <class C>
    static constexpr int kSharedBytes = C::kBlockRows * C::kBlockTokens * sizeof(float);

    template <class C, bool Full, class Output>
    __device__ __forceinline__ void finish_tile(Output output, unsigned char* scratch,
                                                float (&acc)[C::kMmaRows][C::kMmaTokens][4], int rb,
                                                int tb, int rows, int) const {
        if constexpr (requires { output.combine(0, 0, 0.F); }) {
            constexpr int BR = C::kBlockRows, BT = C::kBlockTokens;
            auto* routed  = reinterpret_cast<float*>(scratch);
            const auto at = [&](int r, int t) { return routed + t * BR + (r ^ ((t & 6) << 2)); };
            const bool aligned = (reinterpret_cast<std::uintptr_t>(output.routed.values) & 15) == 0;
            // A contiguous asynchronous read per vector hides the contribution's cold-read
            // latency. Invalid columns remain unreadable, including a null values plane.
            for (int i = threadIdx.x; i < BT * (BR / 4); i += C::kThreads) {
                const int t = i / (BR / 4), r = (i % (BR / 4)) * 4;
                const int token = tb + t, row = rb + r;
                const bool valid =
                    token < logical_tokens && row + 3 < rows &&
                    (!output.routed.valid_tokens || output.routed.valid_tokens[token]);
                const void* source =
                    valid ? static_cast<const void*>(output.routed.values +
                                                     std::int64_t(token) * Output::kRows + row)
                          : static_cast<const void*>(output.destination);
                if (aligned)
                    common::cp_async_zfill<16, Cache::cg>(at(r, t), source, valid ? 16 : 0);
                else {
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        common::cp_async_zfill<4>(at(r, t) + j,
                                                  static_cast<const unsigned char*>(source) +
                                                      j * sizeof(float),
                                                  valid ? 4 : 0);
                }
            }
            common::cp_commit();
            common::cp_wait<0>();
            __syncthreads();
            const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
            const int wr = warp / C::kWarpsTokens, wt = warp % C::kWarpsTokens;
#pragma unroll
            for (int mi = 0; mi < C::kMmaRows; ++mi) {
#pragma unroll
                for (int ni = 0; ni < C::kMmaTokens; ++ni) {
                    const int row   = wr * C::kWarpRows + mi * 16 + (lane >> 2);
                    const int token = wt * C::kWarpTokens + ni * 8 + 2 * (lane & 3);
                    auto& a         = acc[mi][ni];
                    a[0] += *at(row, token);
                    a[1] += *at(row, token + 1);
                    a[2] += *at(row + 8, token);
                    a[3] += *at(row + 8, token + 1);
                }
            }
            __syncthreads();
            Bf16TmaVectorEpilogue::template finish_tile<C, false, true>(
                StreamingOutput{{output.destination, Output::kRows}}, scratch, acc, rb, tb, rows,
                logical_tokens);
        } else {
            Bf16TmaVectorEpilogue::template finish_tile<C, false, true>(
                StreamingOutput{output}, scratch, acc, rb, tb, rows, logical_tokens);
        }
    }
};

template <class Output>
void float_product(const Tensor& x, const SwiGLUWeights& weights, Output output, void* storage,
                   float limit, cudaStream_t stream) {
    auto* product = static_cast<float*>(storage);
    if (x.ne[1] <= 4)
        return float_product<NativeFp8A16SlicedKSchedule<8, 2, 8, 256, 2>,
                             NativeFp8A16SlicedKSchedule<8, 2, 8, 128, 1>>(x, weights, output,
                                                                           product, limit, stream);
    return float_product<NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 1>,
                         NativeFp8A16SlicedKSchedule<16, 8, 4, 128, 2>>(x, weights, output, product,
                                                                        limit, stream);
}

template <class G, class D, bool GuTma, bool DownTma = true, bool RowProduct = false,
          bool Vector = false, class Output>
void quantized(const Tensor& x, const SwiGLUWeights& weights, Output output,
               const ScaledFp8Storage& storage, int physical, float limit, cudaStream_t stream) {
    const int logical = x.ne[1];
    launch_mxfp8_a8_quantize_padded(static_cast<const __nv_bfloat16*>(x.data), 4096, logical,
                                    storage.input_a8, stream, physical);
    const auto gu = mxfp8_a8_operands(weights.gate_up.weight, storage.input_a8, physical);
    const Mxfp8ProductOutput product{storage.product_a8, 1024};
    if constexpr (RowProduct) {
        static_assert(GuTma);
        launch_mxfp8_a8_tma<SharedSchedule<G, 4096>>(gu, product,
                                                     RowProductEpilogue{{limit}, logical}, stream,
                                                     SwiGluTmaBranchRows<G::kBlockRows / 2>{});
    } else if constexpr (GuTma)
        launch_mxfp8_a8_tma<SharedSchedule<G, 4096>>(gu, product,
                                                     Mxfp8PackedProductEpilogue{{limit}, logical},
                                                     stream, SwiGluInterleavedRows<8>{});
    else
        launch_mxfp8_a8_mma<SharedSchedule<G, 4096>>(gu, product,
                                                     Mxfp8PackedProductEpilogue{{limit}, logical},
                                                     stream, SwiGluInterleavedRows<8>{});
    const auto down = mxfp8_a8_operands(weights.down.weight, storage.product_a8, physical);
    if constexpr (Vector) {
        static_assert(DownTma);
        launch_mxfp8_a8_tma<SharedSchedule<D, 1024>>(down, output, VectorDownEpilogue{{}, logical},
                                                     stream);
    } else if constexpr (DownTma)
        launch_mxfp8_a8_tma<SharedSchedule<D, 1024>>(down, output, DownEpilogue{{}, logical},
                                                     stream);
    else
        launch_mxfp8_a8_mma<SharedSchedule<D, 1024>>(down, output, DownEpilogue{{}, logical},
                                                     stream);
}

template <class Output>
constexpr bool has_routed_values(Output output) {
    if constexpr (requires { output.routed.values; }) return output.routed.values != nullptr;
    return false;
}

template <class G, class D, bool GuTma, class Output>
void quantized_medium(const Tensor& x, const SwiGLUWeights& weights, Output output,
                      const ScaledFp8Storage& storage, int physical, float limit,
                      cudaStream_t stream) {
    // A null contribution plane is valid only when every mask entry is zero. Its
    // scalar epilogue avoids staging an all-zero tile just to add it to the result.
    if (has_routed_values(output))
        return quantized<G, D, GuTma, true, false, true>(x, weights, output, storage, physical,
                                                       limit, stream);
    quantized<G, D, GuTma>(x, weights, output, storage, physical, limit, stream);
}

template <class Output>
void run(const Tensor& x, const SwiGLUWeights& weights, Output output, float limit,
         MemoryArena& workspace, cudaStream_t stream) {
    auto scope            = workspace.scope();
    const auto profile    = swiglu_profile(weights);
    const int tokens      = x.ne[1];
    constexpr bool joined = requires { output.combine(0, 0, 0.F); };
    const auto plan       = select_scaled_fp8(profile, tokens, joined);
    const auto storage    = allocate_scaled_fp8(workspace, profile, tokens, joined);
    const int physical    = plan.storage_tokens(tokens);
    switch (plan.kernel) {
    case ScaledFp8Kernel::MtpFloat:
        return float_product(x, weights, output, storage.product, limit, stream);
    case ScaledFp8Kernel::MtpSmall:
        return quantized<NativeFp8MmaSchedule<64, 16, 128, 32, 8, 2>,
                         NativeFp8MmaSchedule<32, 32, 128, 16, 16, 3>, false, false>(
            x, weights, output, storage, physical, limit, stream);
    case ScaledFp8Kernel::MtpPadded:
        return quantized_medium<NativeFp8MmaSchedule<64, 64, 128, 32, 16, 3>,
                                Mxfp8A8TmaSchedule<64, 64, 128, 32, 16, 2, 1, 2, true>, false>(
            x, weights, output, storage, physical, limit, stream);
    case ScaledFp8Kernel::MtpTransition:
        return quantized_medium<Mxfp8A8TmaSchedule<64, 128, 128, 32, 32, 3, 1, 8, true>,
                                Mxfp8A8TmaSchedule<64, 128, 128, 32, 32, 3, 1, 8, true>, true>(
            x, weights, output, storage, physical, limit, stream);
    case ScaledFp8Kernel::MtpBulkMedium:
        return quantized<Mxfp8A8TmaSchedule<192, 128, 128, 96, 32, 2, 1, 8, true, 1>,
                         Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 2, true, 1>, true, true,
                         true, true>(x, weights, output, storage, physical, limit, stream);
    case ScaledFp8Kernel::MtpBulk: {
        // GB10 has 48 SMs. Compare the work per occupied SM across complete CTA waves.
        const int columns      = physical / 128;
        const int wide_waves   = common::div_up(11 * columns, 48);
        const int narrow_waves = common::div_up(16 * columns, 48);
        if (wide_waves * 192 <= narrow_waves * 128)
            return quantized<Mxfp8A8TmaSchedule<192, 128, 128, 96, 32, 2, 1, 8, true, 1>,
                             Mxfp8A8TmaSchedule<128, 128, 128, 64, 64, 2, 1, 8, true>, true, true, true, true>(
                x, weights, output, storage, physical, limit, stream);
        [[fallthrough]];
    }
    case ScaledFp8Kernel::MtpLarge:
        return quantized<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2, 1, 4, true, 1>,
                         Mxfp8A8TmaSchedule<128, 128, 128, 64, 64, 2, 1, 8, true>, true, true, true, true>(
            x, weights, output, storage, physical, limit, stream);
    default:
        throw std::logic_error("shared MXFP8 FFN requires its selected kernel plan");
    }
}
} // namespace

void launch_mxfp8_shared_ffn(const Tensor& x, const SwiGLUWeights& weights, Tensor& output,
                             float limit, MemoryArena& workspace, cudaStream_t stream) {
    run(x, weights, LinearBf16Output{static_cast<__nv_bfloat16*>(output.data), 4096}, limit,
        workspace, stream);
}

void launch_mxfp8_shared_ffn_join(const Tensor& x, const SwiGLUWeights& weights, Tensor& output,
                                  RoutedSwiGLUContribution routed, float limit,
                                  MemoryArena& workspace, cudaStream_t stream) {
    run(x, weights, SwiGluRoutedOutput<4096>{static_cast<__nv_bfloat16*>(output.data), routed},
        limit, workspace, stream);
}
} // namespace ninfer::ops::detail::swiglu_ffn
