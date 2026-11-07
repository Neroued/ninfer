#include "ops/moe/swiglu/dispatch.h"
#include "ops/moe/swiglu/fp8_plan.h"
#include "ops/moe/swiglu/kernel/output.cuh"
#include "ops/moe/swiglu/kernel/fp8_block_scaled_product.cuh"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail::swiglu_ffn {
namespace {
template <class B, int K>
struct SharedBlockSchedule : NativeFp8ScheduleInstance<B, K> {};
template <class B, int K>
using S = SharedBlockSchedule<B, K>;

template <class B>
struct Reg : B {
    static constexpr bool kRegisterWeights = true;
};

template <class B, int Group>
struct InputGroup : B {
    static constexpr int kInputGroup = Group;
};

// Both row warps own one complete physical N128 scale group, gate and up respectively.
struct WidePairedGuSchedule : Fp8BlockA16TmaSchedule<128, 64, 64, 64, 16, 3, 8> {
    static constexpr int kBlockRows = 256, kWarpRows = 128, kMmaRows = 8;
    static constexpr int kStorageBytes = kStages * (kBlockRows + 2 * kBlockTokens) * kBlockK;
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
void sliced(const Tensor& x, const SwiGLUWeights& w, Output output, void* storage, float limit,
            cudaStream_t stream) {
    int t   = x.ne[1];
    auto* p = static_cast<float*>(storage);
    launch_native_fp8_a16_sliced_k<S<G, 4096>>(
        fp8_block_a16_operands(x, w.gate_up.weight), SwiGluFloatOutput{p, 1024},
        BasicSwiGluPairEpilogue<ClampedSwiGluProduct>{{limit}}, stream,
        SwiGluBranchRows<G::kBlockRows / 2>{});
    const auto& d = w.down.weight;
    Fp8BlockA16Operands<float> dn{p, (const std::uint8_t*)d.qdata, (const float*)d.scales, d.n, d.k,
                                  t};
    launch_native_fp8_a16_sliced_k<S<D, 1024>>(dn, output, LinearIdentityEpilogue{}, stream);
}

template <class G, class D, class Output>
void mma(const Tensor& x, const SwiGLUWeights& w, Output output, void* storage, float limit,
         cudaStream_t stream) {
    int t   = x.ne[1];
    auto* p = static_cast<__nv_bfloat16*>(storage);
    launch_native_fp8_a16_mma<S<Reg<G>, 4096>>(
        fp8_block_a16_operands(x, w.gate_up.weight), SwiGluCompensatedOutput{p, 1024, t},
        BasicSwiGluInterleavedMmaEpilogue<ClampedSwiGluProduct>{{limit}}, stream,
        SwiGluInterleavedRows<8>{});
    const auto& d = w.down.weight;
    Fp8BlockCompensatedA16Operands dn{
        {p, (const std::uint8_t*)d.qdata, (const float*)d.scales, d.n, d.k, t},
        p + std::size_t(1024) * t};
    launch_native_fp8_a16_mma<S<Reg<D>, 1024>>(dn, output, DownEpilogue{{}, t}, stream);
}

template <class G, class D, class Output>
void scaled_f16(const Tensor& x, const SwiGLUWeights& w, Output output,
                const ScaledFp8Storage& storage, const ScaledFp8Plan& plan, float limit,
                cudaStream_t stream) {
    const int logical = x.ne[1], physical = plan.storage_tokens(logical);
    constexpr int Group    = G::kBlockRows / 2;
    using Product          = ScaledF16ProductEpilogue<Group>;
    auto* values           = static_cast<__half*>(storage.product);
    auto* scales           = storage.product_scales;
    using GS               = S<G, 4096>;
    auto gu                = fp8_block_a16_operands(x, w.gate_up.weight);
    const auto maps        = fp8_block_a16_tma_descriptors<GS, true>(gu);
    gu.tokens              = physical;
    constexpr auto gkernel = fp8_block_a16_tma_kernel<GS, true, ScaledF16Output, Product, true>;
    constexpr int gb       = fp8_block_a16_tma_bytes<GS, Product>;
    static_assert(gb <= 99 * 1024);
    common::prepare_shared<gb, gkernel, true>();
    const auto gblocks = std::int64_t(2048 / GS::kBlockRows) * (physical / GS::kBlockTokens);
    using DS           = S<InputGroup<D, Group>, 1024>;
    const auto dblocks = std::int64_t(4096 / DS::kBlockRows) * (physical / DS::kBlockTokens);
    if (gblocks > std::numeric_limits<int>::max() || dblocks > std::numeric_limits<int>::max())
        throw std::invalid_argument("scaled F16 shared FFN grid exceeds CUDA grid.x");
    gkernel<<<unsigned(gblocks), GS::kThreads, gb, stream>>>(
        maps, gu, ScaledF16Output{values, scales}, Product{{limit}}, 0);
    CUDA_CHECK(cudaGetLastError());
    const auto& dw = w.down.weight;
    const ScaledF16BlockOperands down{values,
                                      static_cast<const std::uint8_t*>(dw.qdata),
                                      static_cast<const float*>(dw.scales),
                                      scales,
                                      dw.n,
                                      dw.k,
                                      physical};
    const auto dm     = scaled_f16_block_maps<DS>(down);
    constexpr auto dk = scaled_f16_block_tma_kernel<DS, Output, DownEpilogue>;
    constexpr int db  = fp8_block_a16_tma_bytes<DS, DownEpilogue>;
    static_assert(db <= 99 * 1024);
    common::prepare_shared<db, dk, true>();
    dk<<<unsigned(dblocks), DS::kThreads, db, stream>>>(dm, down, output,
                                                        DownEpilogue{{}, logical});
    CUDA_CHECK(cudaGetLastError());
}

template <class Output>
void run(const Tensor& x, const SwiGLUWeights& weights, Output output, float limit,
         MemoryArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto profile = swiglu_profile(weights);
    const int tokens   = x.ne[1];
    const auto plan    = select_scaled_fp8(profile, tokens);
    const auto storage = allocate_scaled_fp8(workspace, profile, tokens);
    switch (plan.kernel) {
    case ScaledFp8Kernel::BlockFloat:
        return sliced<NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 1>,
                      NativeFp8A16SlicedKSchedule<16, 8, 8, 128, 1>>(
            x, weights, output, storage.product, limit, stream);
    case ScaledFp8Kernel::BlockCompensated:
        return mma<NativeFp8MmaSchedule<48, 32, 128, 16, 16, 2, 2, 2>,
                   NativeFp8MmaSchedule<32, 32, 128, 16, 16, 2, 2, 2>>(
            x, weights, output, storage.product, limit, stream);
    case ScaledFp8Kernel::BlockScaledF16Small:
        return scaled_f16<Fp8BlockA16TmaSchedule<64, 32, 128, 32, 16, 3, 2>,
                          Fp8BlockA16TmaSchedule<64, 32, 128, 32, 16, 3, 2>>(
            x, weights, output, storage, plan, limit, stream);
    case ScaledFp8Kernel::BlockScaledF16Medium:
        return scaled_f16<Fp8BlockA16TmaSchedule<64, 64, 64, 32, 16, 3, 2>,
                          Fp8BlockA16TmaSchedule<128, 64, 64, 32, 32, 3, 2>>(
            x, weights, output, storage, plan, limit, stream);
    case ScaledFp8Kernel::BlockScaledF16Wide:
        return scaled_f16<WidePairedGuSchedule, Fp8BlockA16TmaSchedule<128, 128, 64, 32, 32, 3, 8>>(
            x, weights, output, storage, plan, limit, stream);
    case ScaledFp8Kernel::BlockScaledF16Bulk:
        return scaled_f16<Fp8BlockA16TmaSchedule<128, 128, 64, 64, 32, 2, 8>,
                          Fp8BlockA16TmaSchedule<128, 128, 64, 32, 64, 2, 8>>(
            x, weights, output, storage, plan, limit, stream);
    default:
        throw std::logic_error("block shared FFN requires its selected kernel plan");
    }
}
} // namespace

void launch_fp8_block_shared_ffn(const Tensor& x, const SwiGLUWeights& weights, Tensor& output,
                                 float limit, MemoryArena& workspace, cudaStream_t stream) {
    run(x, weights, LinearBf16Output{static_cast<__nv_bfloat16*>(output.data), 4096}, limit,
        workspace, stream);
}

void launch_fp8_block_shared_ffn_join(const Tensor& x, const SwiGLUWeights& weights, Tensor& output,
                                      RoutedSwiGLUContribution routed, float limit,
                                      MemoryArena& workspace, cudaStream_t stream) {
    run(x, weights, SwiGluRoutedOutput<4096>{static_cast<__nv_bfloat16*>(output.data), routed},
        limit, workspace, stream);
}
} // namespace ninfer::ops::detail::swiglu_ffn
