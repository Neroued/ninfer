#include "ops/moe/swiglu/dispatch.h"
#include "ops/moe/swiglu/q8_plan.h"
#include "ops/moe/swiglu/kernel/output.cuh"
#include "ops/moe/swiglu/kernel/q8_down.cuh"
#include "ops/linear/swiglu/q8/paired.cuh"

namespace ninfer::ops::detail::swiglu_ffn {
namespace {
using Small     = Q8A16SlicedKMmaSchedule<16, 8, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 4096>;
using Hot       = Q8A16SlicedKMmaSchedule<32, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 4096>;
using S48       = Q8A16SlicedKMmaSchedule<48, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 4096>;
using T64       = Q8A16TmaMmaSchedule<16, 64, 16, 16, 3, 8, 4096, 64, true, false, true, false, 2>;
using T64Rows32 = Q8A16TmaMmaSchedule<32, 64, 32, 16, 3, 8, 4096, 64, true, false, true, false, 2>;
using Medium    = Q8A16TmaMmaSchedule<64, 64, 64, 16, 2, 8, 4096, 64, true, false, true, false, 2>;
using T96       = Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 4096, 64, true, false, true, false, 2>;
using T128      = Q8A16TmaMmaSchedule<32, 128, 32, 32, 2, 8, 4096, 64, true, false, true, false, 2>;
using Bulk   = Q8A16TmaMmaSchedule<128, 128, 64, 32, 1, 8, 4096, 64, true, false, true, false, 2>;
using Down   = Q8A16TmaMmaSchedule<64, 64, 32, 16, 2, 8, 1024, 64, true, true, true, false, 1, 2>;
using Down8  = Q8A16TmaMmaSchedule<32, 8, 16, 8, 2, 8, 1024, 128, true, true, true, false, 1, 2>;
using Down16 = Q8A16TmaMmaSchedule<64, 16, 32, 8, 1, 8, 1024, 128, true, true, true, false, 1, 2>;

template <class ProductOutput>
void gate_up(const Q8LinearOperands& p, ProductOutput output, float limit, cudaStream_t stream) {
    const ClampedSwiGluProduct product{limit};
    if (p.tokens == 1) return launch_q8_swiglu_vector<4096, 1024, 1, 8>(p, output, product, stream);
    // Sliced-K applies each stored FP16 scale to an FP32 dot of exact int8 codes.
    // It preserves GU precision without a second weight-component contraction.
    if (p.tokens <= 16) return launch_q8_swiglu_sliced<Small, 1024>(p, output, product, stream);
    if (p.tokens <= 32) return launch_q8_swiglu_sliced<Hot, 1024>(p, output, product, stream);
    if (p.tokens <= 48) return launch_q8_swiglu_sliced<S48, 1024>(p, output, product, stream);
    if (p.tokens <= 64) return launch_q8_swiglu_interleaved_tma<T64>(p, output, product, stream);
    if (p.tokens <= 128) return launch_q8_swiglu_tma<T64Rows32>(p, output, product, stream);
    if (p.tokens <= 192) return launch_q8_swiglu_tma<Medium>(p, output, product, stream);
    if (p.tokens <= 288) return launch_q8_swiglu_tma<T96>(p, output, product, stream);
    if (p.tokens <= 2048) return launch_q8_swiglu_tma<T128>(p, output, product, stream);
    if (p.tokens <= 4096) return launch_q8_swiglu_tma<T96>(p, output, product, stream);
    return launch_q8_swiglu_tma<Bulk>(p, output, product, stream);
}

template <class Output>
void run(const Tensor& x, const SwiGLUWeights& weights, Output output, float limit,
         MemoryArena& workspace, cudaStream_t stream) {
    auto scope                         = workspace.scope();
    const int tokens                   = x.ne[1];
    const auto [plane_tokens, storage] = allocate_q8_product(workspace, tokens);
    const auto gu                      = q8_linear_operands(x, weights.gate_up.weight);
    if (tokens == 1) {
        auto* product = static_cast<float*>(storage.data);
        gate_up(gu, SwiGluFloatOutput{product, 1024}, limit, stream);
        const auto& w = weights.down.weight;
        const Q8DownWeightView view{static_cast<const std::uint8_t*>(w.qdata),
                                    static_cast<const std::uint8_t*>(w.scales)};
        q8_down_vector_kernel<<<4096 / 4, 128, 0, stream>>>(product, view, output);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    auto* planes = static_cast<__nv_bfloat16*>(storage.data);
    if (tokens <= 32)
        gate_up(gu, SwiGluPaddedCompensatedOutput{planes, 1024, tokens, plane_tokens}, limit,
                stream);
    else
        gate_up(gu, SwiGluCompensatedOutput{planes, 1024, tokens}, limit, stream);
    if (tokens <= 8)
        q8_compensated_down_tma<Down8>(planes, weights.down.weight, tokens, output, stream,
                                       plane_tokens);
    else if (tokens <= 16)
        q8_compensated_down_tma<Down16>(planes, weights.down.weight, tokens, output, stream,
                                        plane_tokens);
    else
        q8_compensated_down_tma<Down>(planes, weights.down.weight, tokens, output, stream,
                                      plane_tokens);
}
} // namespace

void launch_q8_ffn(const Tensor& x, const SwiGLUWeights& weights, Tensor& out, float limit,
                   MemoryArena& workspace, cudaStream_t stream) {
    run(x, weights, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), 4096}, limit, workspace,
        stream);
}

void launch_q8_ffn_join(const Tensor& x, const SwiGLUWeights& weights, Tensor& out,
                        RoutedSwiGLUContribution routed, float limit, MemoryArena& workspace,
                        cudaStream_t stream) {
    run(x, weights, SwiGluRoutedOutput<4096>{static_cast<__nv_bfloat16*>(out.data), routed}, limit,
        workspace, stream);
}
} // namespace ninfer::ops::detail::swiglu_ffn
