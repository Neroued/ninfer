#include "ops/moe/swiglu/dispatch.h"
#include "ops/moe/swiglu/fp8_plan.h"
#include "ops/moe/swiglu/kernel/fp8_product.cuh"
#include "ops/moe/swiglu/kernel/output.cuh"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/fp8_block/operands.h"
#include "ops/linear/kernels/mxfp8/a8.cuh"
#include "ops/linear/kernels/mxfp8/a8_tma.cuh"

#include <type_traits>

namespace ninfer::ops::detail::swiglu_ffn {
namespace {

template <class Schedule>
struct PairWarpSchedule : Schedule {
    static constexpr bool kPairAcrossWarps = true;
};

template <class Schedule>
struct RegisterWeightSchedule : Schedule {
    static constexpr bool kRegisterWeights = true;
};

template <int Intermediate, class Operands, class Output>
void gate_up_a16(const Operands& p, Output out, float limit, cudaStream_t stream) {
    const BasicSwiGluInterleavedMmaEpilogue<ClampedSwiGluProduct> product{{limit}};
    if constexpr (std::is_same_v<Output, SwiGluFloatOutput>) {
        const auto vector = [&]<int Tokens, int Chains>() {
            using S = PairWarpSchedule<NativeFp8ScheduleInstance<
                NativeFp8A16SimtSchedule<Tokens, 4, 1, 16, Chains>, 4096>>;
            launch_native_fp8_a16_simt<S>(p, out, product, stream,
                                          SwiGluBranchRows<1, Intermediate>{});
        };
        if (p.tokens == 1) return vector.template operator()<1, 1>();
        if (p.tokens <= 2) return vector.template operator()<2, 4>();
        return vector.template operator()<4, 1>();
    } else {
        if constexpr (std::is_same_v<typename Operands::ScaleType, float>) {
            if (p.tokens >= 33) {
                using S = RegisterWeightSchedule<NativeFp8ScheduleInstance<
                    NativeFp8MmaSchedule<64, 64, 128, 16, 32, 2, 2>, 4096>>;
                return launch_native_fp8_a16_mma<S>(p, out, product, stream,
                                                    SwiGluInterleavedRows<8>{});
            }
        }
        if (p.tokens <= 16) {
            using S = NativeFp8ScheduleInstance<NativeFp8MmaSchedule<32, 16, 128, 16, 8, 3>, 4096>;
            return launch_native_fp8_a16_mma<S>(p, out, product, stream,
                                                SwiGluInterleavedRows<8>{});
        }
        if (p.tokens <= 32) {
            using S = NativeFp8ScheduleInstance<NativeFp8MmaSchedule<64, 32, 128, 32, 16, 3>, 4096>;
            return launch_native_fp8_a16_mma<S>(p, out, product, stream,
                                                SwiGluInterleavedRows<8>{});
        }
        using S = NativeFp8ScheduleInstance<NativeFp8MmaSchedule<64, 64, 128, 32, 16, 2>, 4096>;
        launch_native_fp8_a16_mma<S>(p, out, product, stream, SwiGluInterleavedRows<8>{});
    }
}

template <int Intermediate, class Operands, class Output>
void down_a16(const Operands& p, Output out, cudaStream_t stream) {
    if constexpr (!requires { p.x_low; }) {
        const auto vector = [&]<int Tokens, int Chains>() {
            using S = NativeFp8ScheduleInstance<
                NativeFp8A16SimtSchedule<Tokens, 4, 1, (Tokens <= 2 ? 16 : 8), Chains>,
                Intermediate>;
            launch_native_fp8_a16_simt<S>(p, out, LinearIdentityEpilogue{}, stream,
                                          Fp8IdentityRows{});
        };
        if (p.tokens == 1) return vector.template operator()<1, 1>();
        if (p.tokens <= 2) return vector.template operator()<2, 1>();
        return vector.template operator()<4, 4>();
    } else {
        if constexpr (std::is_same_v<typename Operands::ScaleType, float>) {
            if (p.tokens >= 33) {
                using S = RegisterWeightSchedule<NativeFp8ScheduleInstance<
                    NativeFp8MmaSchedule<128, 64, 128, 32, 32, 1, 2>, Intermediate>>;
                return launch_native_fp8_a16_mma<S>(p, out, LinearIdentityEpilogue{}, stream,
                                                    Fp8IdentityRows{});
            }
        }
        if (p.tokens <= 8) {
            using S =
                NativeFp8ScheduleInstance<NativeFp8MmaSchedule<32, 8, 128, 16, 8, 3>, Intermediate>;
            return launch_native_fp8_a16_mma<S>(p, out, LinearIdentityEpilogue{}, stream,
                                                Fp8IdentityRows{});
        }
        if (p.tokens <= 16) {
            using S = NativeFp8ScheduleInstance<NativeFp8MmaSchedule<32, 16, 128, 16, 8, 3>,
                                                Intermediate>;
            return launch_native_fp8_a16_mma<S>(p, out, LinearIdentityEpilogue{}, stream,
                                                Fp8IdentityRows{});
        }
        if (p.tokens <= 32) {
            using S = NativeFp8ScheduleInstance<NativeFp8MmaSchedule<128, 32, 128, 32, 16, 2>,
                                                Intermediate>;
            return launch_native_fp8_a16_mma<S>(p, out, LinearIdentityEpilogue{}, stream,
                                                Fp8IdentityRows{});
        }
        if (p.tokens <= 256) {
            using S = NativeFp8ScheduleInstance<NativeFp8MmaSchedule<128, 32, 128, 32, 16, 2>,
                                                Intermediate>;
            return launch_native_fp8_a16_mma<S>(p, out, LinearIdentityEpilogue{}, stream,
                                                Fp8IdentityRows{});
        }
        using S =
            NativeFp8ScheduleInstance<NativeFp8MmaSchedule<128, 64, 128, 32, 32, 1>, Intermediate>;
        launch_native_fp8_a16_mma<S>(p, out, LinearIdentityEpilogue{}, stream, Fp8IdentityRows{});
    }
}

template <int Intermediate>
void gate_up_a8(const Mxfp8A8Operands& p, Mxfp8A8Workspace out, float limit, cudaStream_t stream) {
    const Mxfp8ProductOutput output{out, Intermediate};
    const Mxfp8ProductEpilogue epilogue{{limit}};
    if (p.tokens <= 64) {
        using S = NativeFp8ScheduleInstance<NativeFp8MmaSchedule<64, 16, 128, 32, 8, 3>, 4096>;
        return launch_mxfp8_a8_mma<S>(p, output, epilogue, stream, SwiGluInterleavedRows<8>{});
    }
    if (p.tokens <= 128) {
        using S = NativeFp8ScheduleInstance<NativeFp8MmaSchedule<64, 32, 128, 32, 16, 3>, 4096>;
        return launch_mxfp8_a8_mma<S>(p, output, epilogue, stream, SwiGluInterleavedRows<8>{});
    }
    using S = NativeFp8ScheduleInstance<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2>, 4096>;
    launch_mxfp8_a8_tma<S>(p, output, epilogue, stream, SwiGluInterleavedRows<8>{});
}

template <int Intermediate, class Output>
void down_a8(const Mxfp8A8Operands& p, Output out, cudaStream_t stream) {
    if (p.tokens <= 64) {
        using S =
            NativeFp8ScheduleInstance<NativeFp8MmaSchedule<64, 16, 128, 32, 8, 3>, Intermediate>;
        return launch_mxfp8_a8_mma<S>(p, out, LinearIdentityEpilogue{}, stream);
    }
    if (p.tokens <= 128) {
        using S =
            NativeFp8ScheduleInstance<NativeFp8MmaSchedule<64, 32, 128, 32, 16, 3>, Intermediate>;
        return launch_mxfp8_a8_mma<S>(p, out, LinearIdentityEpilogue{}, stream);
    }
    if constexpr (Intermediate == 6144) {
        using S =
            NativeFp8ScheduleInstance<Mxfp8A8TmaSchedule<128, 64, 128, 64, 32, 3>, Intermediate>;
        launch_mxfp8_a8_tma<S>(p, out, LinearIdentityEpilogue{}, stream);
    } else {
        if (p.tokens <= 256) {
            using S =
                NativeFp8ScheduleInstance<Mxfp8A8TmaSchedule<64, 64, 128, 32, 32, 3>, Intermediate>;
            return launch_mxfp8_a8_tma<S>(p, out, LinearIdentityEpilogue{}, stream);
        }
        using S =
            NativeFp8ScheduleInstance<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2>, Intermediate>;
        launch_mxfp8_a8_tma<S>(p, out, LinearIdentityEpilogue{}, stream);
    }
}

template <int Intermediate, bool Mxfp8, class Output>
void run(const Tensor& x, const SwiGLUWeights& weights, Output out, float limit,
         MemoryArena& workspace, cudaStream_t stream) {
    auto scope            = workspace.scope();
    const auto profile    = swiglu_profile(weights);
    const int tokens      = x.ne[1];
    constexpr bool joined = requires { out.combine(0, 0, 0.F); };
    const auto plan       = select_scaled_fp8(profile, tokens, joined);
    const auto storage    = allocate_scaled_fp8(workspace, profile, tokens, joined);
    if constexpr (Mxfp8) {
        if (plan.product == ScaledFp8Product::Mxfp8) {
            launch_mxfp8_a8_quantize(static_cast<const __nv_bfloat16*>(x.data), 4096, tokens,
                                     storage.input_a8, stream);
            gate_up_a8<Intermediate>(
                mxfp8_a8_operands(weights.gate_up.weight, storage.input_a8, tokens),
                storage.product_a8, limit, stream);
            down_a8<Intermediate>(
                mxfp8_a8_operands(weights.down.weight, storage.product_a8, tokens), out, stream);
            return;
        }
    }
    const auto operands = [&]<class Input>(const Input* data, const Weight& weight) {
        using P = std::conditional_t<Mxfp8, Mxfp8A16Operands<Input>, Fp8BlockA16Operands<Input>>;
        return P{data,
                 static_cast<const std::uint8_t*>(weight.qdata),
                 static_cast<const typename P::ScaleType*>(weight.scales),
                 weight.n,
                 weight.k,
                 tokens};
    };
    const auto gu = operands(static_cast<const __nv_bfloat16*>(x.data), weights.gate_up.weight);
    if (plan.product == ScaledFp8Product::Float) {
        auto* product = static_cast<float*>(storage.product);
        gate_up_a16<Intermediate>(gu, SwiGluFloatOutput{product, Intermediate}, limit, stream);
        down_a16<Intermediate>(operands(product, weights.down.weight), out, stream);
    } else {
        auto* planes = static_cast<__nv_bfloat16*>(storage.product);
        gate_up_a16<Intermediate>(gu, SwiGluCompensatedOutput{planes, Intermediate, tokens}, limit,
                                  stream);
        using P =
            std::conditional_t<Mxfp8, Mxfp8CompensatedA16Operands, Fp8BlockCompensatedA16Operands>;
        const P down{operands(planes, weights.down.weight),
                     planes + std::size_t(Intermediate) * tokens};
        down_a16<Intermediate>(down, out, stream);
    }
}

template <class Output>
void dispatch(const Tensor& x, const SwiGLUWeights& weights, Output out, float limit,
              MemoryArena& workspace, cudaStream_t stream) {
    if (weights.gate_up.weight.qtype == QType::FP8_E4M3FN_BLOCK128X128_FP32)
        return run<1024, false>(x, weights, out, limit, workspace, stream);
    if (weights.gate_up.weight.n == 12288)
        return run<6144, true>(x, weights, out, limit, workspace, stream);
    run<1024, true>(x, weights, out, limit, workspace, stream);
}
} // namespace

void launch_scaled_fp8_ffn(const Tensor& x, const SwiGLUWeights& weights, Tensor& out, float limit,
                           MemoryArena& workspace, cudaStream_t stream) {
    if (select_scaled_fp8(swiglu_profile(weights), x.ne[1]).kernel != ScaledFp8Kernel::Default) {
        if (weights.gate_up.weight.qtype == QType::FP8_E4M3FN_BLOCK128X128_FP32)
            return launch_fp8_block_shared_ffn(x, weights, out, limit, workspace, stream);
        if (weights.down.weight.k == 1024)
            return launch_mxfp8_shared_ffn(x, weights, out, limit, workspace, stream);
        return launch_mxfp8_dense_ffn(x, weights, out, limit, workspace, stream);
    }
    dispatch(x, weights, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), 4096}, limit,
             workspace, stream);
}

void launch_scaled_fp8_ffn_join(const Tensor& x, const SwiGLUWeights& weights, Tensor& out,
                                RoutedSwiGLUContribution routed, float limit,
                                MemoryArena& workspace, cudaStream_t stream) {
    if (select_scaled_fp8(swiglu_profile(weights), x.ne[1], true).kernel !=
        ScaledFp8Kernel::Default) {
        if (weights.gate_up.weight.qtype == QType::FP8_E4M3FN_BLOCK128X128_FP32)
            return launch_fp8_block_shared_ffn_join(x, weights, out, routed, limit, workspace,
                                                    stream);
        if (weights.down.weight.k == 1024)
            return launch_mxfp8_shared_ffn_join(x, weights, out, routed, limit, workspace, stream);
        return launch_mxfp8_dense_ffn_join(x, weights, out, routed, limit, workspace, stream);
    }
    dispatch(x, weights, SwiGluRoutedOutput<4096>{static_cast<__nv_bfloat16*>(out.data), routed},
             limit, workspace, stream);
}

} // namespace ninfer::ops::detail::swiglu_ffn
