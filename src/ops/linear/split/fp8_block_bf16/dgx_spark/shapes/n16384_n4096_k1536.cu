#include "ops/linear/split/dispatch.h"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"
#include "ops/linear/kernels/fp8_block/operands.h"
#include "ops/linear/kernels/fp8_block/a16_tma.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_tma_register_b.cuh"
#include "ops/linear/split/bf16/launch.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {
template <class S>
using Fp8 = NativeFp8ScheduleInstance<S, 1536>;
template <class S>
using Bf16 = Bf16ScheduleInstance<S, 1536>;

template <class S>
struct RegisterWeights : S {
    static constexpr bool kRegisterWeights = true;
};

using Fp8Medium = RegisterWeights<Fp8<NativeFp8MmaSchedule<64, 32, 128, 32, 16, 2, 2, 2>>>;
using Fp8Wide   = RegisterWeights<Fp8<NativeFp8MmaSchedule<64, 64, 128, 32, 32, 2, 2, 8>>>;
using Fp8Large  = Fp8<Fp8BlockA16TmaSchedule<128, 64, 128, 32, 32, 2, 8>>;
using Fp8Bulk   = Fp8<Fp8BlockA16TmaSchedule<128, 128, 64, 16, 128, 3, 8>>;
using Bf16Large = Bf16<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2, Bf16MmaRaster::RowFast>>;
using Bf16Wide  = Bf16<Bf16A16TmaMmaSchedule<256, 128, 64, 64, 32, 2, 1, Bf16MmaRaster::RowFast>>;
using Bf16Transition =
    Bf16<Bf16A16TmaRegisterBSchedule<128, 128, 64, 128, 16, 2, Bf16MmaRaster::RowFast>>;
using Bf16Bulk = Bf16<Bf16A16TmaRegisterBSchedule<128, 128, 64, 64, 32, 2, Bf16MmaRaster::RowFast>>;

template <class F, class B>
constexpr int kMixedSharedBytes =
    NativeFp8SlicedKStorage<F, __nv_bfloat16, float>::kBytes > B::kSharedBytes
        ? NativeFp8SlicedKStorage<F, __nv_bfloat16, float>::kBytes
        : B::kSharedBytes;

// Each CTA contracts one native parent directly into its consumer's output.
// The uniform branch shares the launch and staging allocation, not weight precision.
template <class F, class B>
__global__ __launch_bounds__(F::kThreads, 2) void fp8_block_bf16_split_sliced_kernel(
    const __nv_bfloat16* x, const std::uint8_t* codes, const float* scales,
    const __nv_bfloat16* weight, __nv_bfloat16* first, __nv_bfloat16* second, int tokens) {
    static_assert(F::kThreads == B::kThreads && F::kBlockTokens == B::kBlockTokens);
    constexpr int first_tiles = 16384 / F::kBlockRows;
    const int token_tiles     = (tokens + F::kBlockTokens - 1) / F::kBlockTokens;
    const int tile            = blockIdx.x / token_tiles;
    const int token           = blockIdx.x % token_tiles * F::kBlockTokens;
    auto* shared              = common::shared_storage<kMixedSharedBytes<F, B>>();
    if (tile < first_tiles) {
        native_fp8_a16_sliced_k_body<F>(x, codes, scales, 16384, 1536, tokens,
                                        LinearBf16Output{first, 16384}, LinearIdentityEpilogue{},
                                        Fp8IdentityRows{}, tile, token, shared);
    } else {
        bf16_a16_sliced_k_mma_body<B>(x, weight, LinearBf16Output{second, 4096},
                                      LinearIdentityEpilogue{}, 4096, 1536, tokens,
                                      tile - first_tiles, token, shared);
    }
}

template <int Rows, int Tokens, int Warps, int Bf16Stages>
void launch_sliced(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
                   cudaStream_t stream) {
    using F               = Fp8<NativeFp8A16SlicedKSchedule<Rows, Tokens, Warps, 128, 1, 2>>;
    using B               = Bf16<Bf16A16SlicedKMmaSchedule<16, Tokens, Warps, 64, Bf16Stages>>;
    constexpr auto kernel = fp8_block_bf16_split_sliced_kernel<F, B>;
    const auto shared     = common::prepare_shared<kMixedSharedBytes<F, B>, kernel>();
    const int blocks      = (16384 / Rows + 4096 / 16) * common::div_up(x.ne[1], Tokens);
    kernel<<<blocks, F::kThreads, shared, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weights.first.qdata),
        static_cast<const float*>(weights.first.scales),
        static_cast<const __nv_bfloat16*>(weights.second->qdata),
        static_cast<__nv_bfloat16*>(outputs[0].data), static_cast<__nv_bfloat16*>(outputs[1].data),
        x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void launch_first(const Tensor& x, const Weight& weight, LinearBf16Output out,
                  cudaStream_t stream) {
    const auto p = fp8_block_a16_operands(x, weight);
    if (p.tokens <= 96)
        return launch_native_fp8_a16_mma<Fp8Medium>(p, out, Bf16MmaStreamingWarpStoreEpilogue{},
                                                    stream);
    if (p.tokens <= 128)
        return launch_native_fp8_a16_mma<Fp8Wide>(p, out, Bf16MmaStreamingWarpStoreEpilogue{},
                                                  stream);
    if (p.tokens <= 192)
        return launch_fp8_block_a16_tma<Fp8Large>(p, out, Bf16MmaStreamingWarpStoreEpilogue{},
                                                  stream);
    launch_fp8_block_a16_tma<Fp8Bulk>(p, out, Bf16MmaStreamingWarpStoreEpilogue{}, stream);
}

template <class S, class Epilogue = Bf16MmaWarpStoreEpilogue>
void register_b(const Bf16SplitRoutes<LinearBf16Output>& routes) {
    launch_bf16_a16_tma_register_b<S>(routes.operands, routes.output, Epilogue{}, routes.stream);
}

void launch_second(const Bf16SplitRoutes<LinearBf16Output>& routes) {
    const int tokens = routes.operands.tokens;
    if (tokens <= 192) return routes.tma<Bf16Large, Bf16MmaWarpStoreEpilogue>();
    if (tokens <= 1536 && ((tokens + 127) / 128) % 3 != 1)
        return routes.tma<Bf16Wide, Bf16MmaStreamingWarpStoreEpilogue>();
    if (tokens <= 512) return register_b<Bf16Transition>(routes);
    if (tokens <= 1536) return register_b<Bf16Bulk>(routes);
    if (tokens <= 2048) return register_b<Bf16Bulk, Bf16MmaStreamingWarpStoreEpilogue>(routes);
    register_b<Bf16Bulk>(routes);
}

void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs, MemoryArena*,
            cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 8) return launch_sliced<16, 8, 8, 2>(x, weights, outputs, stream);
    if (tokens <= 16) return launch_sliced<16, 8, 8, 1>(x, weights, outputs, stream);
    if (tokens <= 48) return launch_sliced<32, 16, 4, 2>(x, weights, outputs, stream);
    if (tokens <= 64) return launch_sliced<32, 32, 4, 1>(x, weights, outputs, stream);
    launch_first(x, weights.first, {static_cast<__nv_bfloat16*>(outputs[0].data), 16384}, stream);
    launch_second({bf16_a16_operands(x, *weights.second),
                   {static_cast<__nv_bfloat16*>(outputs[1].data), 4096},
                   stream});
}
} // namespace

const SplitShape kFp8BlockBf16SplitN16384N4096K1536{16384,  1536,    kSplitRows<16384, 4096>,
                                                    launch, nullptr, 4096};
} // namespace ninfer::ops::detail::linear_split
