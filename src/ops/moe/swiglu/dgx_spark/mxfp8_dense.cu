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
struct DenseSchedule : NativeFp8ScheduleInstance<Base, K> {};

template <class Base>
struct RegisterWeights : Base {
    static constexpr bool kRegisterWeights = true;
};

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
    launch_native_fp8_a16_sliced_k<DenseSchedule<G, 4096>>(
        mxfp8_a16_operands(x, weights.gate_up.weight), SwiGluFloatOutput{product, 6144},
        BasicSwiGluPairEpilogue<ClampedSwiGluProduct>{{limit}}, stream,
        SwiGluBranchRows<G::kBlockRows / 2>{});
    const auto& w = weights.down.weight;
    const Mxfp8A16Operands<float> down{product,
                                       static_cast<const std::uint8_t*>(w.qdata),
                                       static_cast<const std::uint8_t*>(w.scales),
                                       w.n,
                                       w.k,
                                       int(x.ne[1])};
    launch_native_fp8_a16_sliced_k<DenseSchedule<D, 6144>>(down, output, LinearIdentityEpilogue{},
                                                           stream, Fp8IdentityRows{});
}

template <class Output>
void float_product(const Tensor& x, const SwiGLUWeights& weights, Output output, void* storage,
                   float limit, cudaStream_t stream) {
    using One     = NativeFp8A16SlicedKSchedule<8, 1, 16, 128, 2>;
    using Four    = NativeFp8A16SlicedKSchedule<8, 4, 8, 256, 2>;
    using Eight   = NativeFp8A16SlicedKSchedule<8, 4, 8, 256, 1>;
    auto* product = static_cast<float*>(storage);
    if (x.ne[1] == 1) return float_product<One, Four>(x, weights, output, product, limit, stream);
    if (x.ne[1] == 2) return float_product<One, One>(x, weights, output, product, limit, stream);
    if (x.ne[1] <= 4) return float_product<Four, Four>(x, weights, output, product, limit, stream);
    if (x.ne[1] <= 8)
        return float_product<Eight, Eight>(x, weights, output, product, limit, stream);
    float_product<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 2>,
                  NativeFp8A16SlicedKSchedule<32, 16, 4, 128, 1>>(x, weights, output, product,
                                                                  limit, stream);
}

template <class Output>
void compensated_product(const Tensor& x, const SwiGLUWeights& weights, Output output,
                         void* storage, float limit, cudaStream_t stream) {
    const int tokens = x.ne[1];
    auto* planes     = static_cast<__nv_bfloat16*>(storage);
    using G          = DenseSchedule<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 2>, 4096>;
    launch_native_fp8_a16_sliced_k<G>(mxfp8_a16_operands(x, weights.gate_up.weight),
                                      SwiGluCompensatedOutput{planes, 6144, tokens},
                                      BasicSwiGluPairEpilogue<ClampedSwiGluProduct>{{limit}},
                                      stream, SwiGluBranchRows<8>{});
    const auto& w = weights.down.weight;
    const Mxfp8CompensatedA16Operands down{{planes, static_cast<const std::uint8_t*>(w.qdata),
                                            static_cast<const std::uint8_t*>(w.scales), w.n, w.k,
                                            tokens},
                                           planes + std::size_t(6144) * tokens};
    using D = DenseSchedule<RegisterWeights<NativeFp8MmaSchedule<32, 16, 128, 16, 8, 2>>, 6144>;
    launch_native_fp8_a16_mma<D>(down, output, DownEpilogue{{}, tokens}, stream);
}

template <class G, class D, bool GuTma, bool DownTma = true, class Output>
void quantized_product(const Tensor& x, const SwiGLUWeights& weights, Output output,
                       const ScaledFp8Storage& storage, int physical, float limit,
                       cudaStream_t stream) {
    const int logical = x.ne[1];
    launch_mxfp8_a8_quantize_padded(static_cast<const __nv_bfloat16*>(x.data), 4096, logical,
                                    storage.input_a8, stream, physical);
    const auto gu = mxfp8_a8_operands(weights.gate_up.weight, storage.input_a8, physical);
    const Mxfp8ProductOutput product{storage.product_a8, 6144};
    const Mxfp8PackedProductEpilogue epilogue{{limit}, logical};
    if constexpr (GuTma)
        launch_mxfp8_a8_tma<DenseSchedule<G, 4096>>(gu, product, epilogue, stream,
                                                    SwiGluInterleavedRows<8>{});
    else
        launch_mxfp8_a8_mma<DenseSchedule<G, 4096>>(gu, product, epilogue, stream,
                                                    SwiGluInterleavedRows<8>{});
    const auto down = mxfp8_a8_operands(weights.down.weight, storage.product_a8, physical);
    if constexpr (DownTma)
        launch_mxfp8_a8_tma<DenseSchedule<D, 6144>>(down, output, DownEpilogue{{}, logical},
                                                    stream);
    else
        launch_mxfp8_a8_mma<DenseSchedule<D, 6144>>(down, output, DownEpilogue{{}, logical},
                                                    stream);
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
    switch (plan.kernel) {
    case ScaledFp8Kernel::DenseFloat:
        return float_product(x, weights, output, storage.product, limit, stream);
    case ScaledFp8Kernel::DenseCompensated:
        return compensated_product(x, weights, output, storage.product, limit, stream);
    case ScaledFp8Kernel::DensePaddedSmall:
        return quantized_product<NativeFp8MmaSchedule<64, 16, 128, 32, 8, 3>,
                                 NativeFp8MmaSchedule<64, 16, 128, 32, 8, 3>, false, false>(
            x, weights, output, storage, plan.storage_tokens(tokens), limit, stream);
    case ScaledFp8Kernel::DenseSmall:
        return quantized_product<NativeFp8MmaSchedule<64, 16, 128, 32, 8, 2>,
                                 NativeFp8MmaSchedule<32, 32, 128, 16, 16, 3>, false, false>(
            x, weights, output, storage, tokens, limit, stream);
    case ScaledFp8Kernel::DenseMma:
        return quantized_product<NativeFp8MmaSchedule<64, 64, 128, 32, 16, 3>,
                                 Mxfp8A8TmaSchedule<64, 64, 256, 32, 16, 2>, false>(
            x, weights, output, storage, plan.storage_tokens(tokens), limit, stream);
    case ScaledFp8Kernel::DenseTransition:
        return quantized_product<Mxfp8A8TmaSchedule<64, 128, 128, 32, 32, 3, 1, 8, true>,
                                 Mxfp8A8TmaSchedule<64, 128, 128, 32, 32, 3, 1, 8, true>, true>(
            x, weights, output, storage, plan.storage_tokens(tokens), limit, stream);
    case ScaledFp8Kernel::DenseBulk:
        if (tokens <= 2048)
            return quantized_product<Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 8, true, 1>,
                                     Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 4, true, 1>,
                                     true>(x, weights, output, storage, plan.storage_tokens(tokens),
                                           limit, stream);
        return quantized_product<Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 2, true, 1>,
                                 Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 8, true, 1>, true>(
            x, weights, output, storage, plan.storage_tokens(tokens), limit, stream);
    case ScaledFp8Kernel::Default:
        throw std::logic_error("dense MXFP8 FFN requires its selected kernel plan");
    }
}
} // namespace

void launch_mxfp8_dense_ffn(const Tensor& x, const SwiGLUWeights& weights, Tensor& output,
                            float limit, MemoryArena& workspace, cudaStream_t stream) {
    run(x, weights, LinearBf16Output{static_cast<__nv_bfloat16*>(output.data), 4096}, limit,
        workspace, stream);
}

void launch_mxfp8_dense_ffn_join(const Tensor& x, const SwiGLUWeights& weights, Tensor& output,
                                 RoutedSwiGLUContribution routed, float limit,
                                 MemoryArena& workspace, cudaStream_t stream) {
    run(x, weights, SwiGluRoutedOutput<4096>{static_cast<__nv_bfloat16*>(output.data), routed},
        limit, workspace, stream);
}
} // namespace ninfer::ops::detail::swiglu_ffn
