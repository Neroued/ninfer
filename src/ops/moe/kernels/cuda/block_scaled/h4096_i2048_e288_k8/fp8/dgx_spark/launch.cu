#include "ops/moe/kernels/cuda/block_scaled/h4096_i2048_e288_k8/fp8/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/h4096_i2048_e288_k8/fp8/compute.cuh"
#include "ops/moe/kernels/cuda/block_scaled/h4096_i2048_e288_k8/fp8/vector_pair.cuh"
#include "ops/common/host/launch.h"

#include <algorithm>

namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8::fp8 {
namespace {
__global__ void quantize_input(const __nv_bfloat16* input, std::uint8_t* codes, float* scales,
                               int tokens, int* counts, float* routed_sum, bool clear_counts,
                               const int* ids, const int* mask, ExpertSubsetKind kind,
                               int* active_tokens) {
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = tid & 31, group = tid / 32;
    if (counts && clear_counts && tid < kExperts) { counts[tid] = 0; }
    // Eight warps handle eight K128 groups. Each token has32 groups/four CTAs;
    // token liveness is identical for all its CTAs, while output clear ranges are disjoint.
    const int token = group / (kHidden / 128);
    bool hit        = false;
    if (lane < kTopK) hit = includes(kind, mask, ids[std::int64_t(token) * kTopK + lane]);
    const bool live = __any_sync(0xffffffff, hit);
    if (!lane && group % (kHidden / 128) == 0) active_tokens[token] = live;
    if (!live) return;
    if (routed_sum) {
        const int local_tid = tid % (kHidden / 4);
        for (int row = local_tid; row < kHidden; row += kHidden / 4)
            routed_sum[std::int64_t(token) * kHidden + row] = 0;
    }
    if (group < tokens * (kHidden / 128)) {
        const auto bits = load_vec<uint2>(input + std::int64_t(group) * 128 + lane * 4);
        const float2 lo = bf16x2_bits_to_float2(bits.x), hi = bf16x2_bits_to_float2(bits.y);
        const float maximum =
            warp_max(fmaxf(fmaxf(fabsf(lo.x), fabsf(lo.y)), fmaxf(fabsf(hi.x), fabsf(hi.y))));
        const float scale = fp8_fp32_scale(maximum);
        const unsigned packed =
            fp8_quant_fp32_pair(lo, scale) | (unsigned(fp8_quant_fp32_pair(hi, scale)) << 16);
        *reinterpret_cast<unsigned*>(codes + std::int64_t(group) * 128 + lane * 4) = packed;
        if (!lane) { scales[group] = scale; }
    }
}

template <bool Gate, int Columns, PopulationRoute Population>
void launch_mma(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                const Storage& s, const Plan& plan, CudaExecutionView execution) {
    using S                = Schedule<Columns, Gate>;
    constexpr int rows     = Gate ? kIntermediate : kHidden;
    const int job_columns  = plan.job_columns();
    const int subtiles     = job_columns / Columns;
    const int max_jobs     = (kTopK * x.ne[1] / job_columns + kExperts) * subtiles;
    const int blocks       = std::min(max_jobs * (rows / 128),
                                      execution.multiprocessor_count() * plan.queued_ctas_per_sm);
    const int shared_bytes = common::prepare_shared<sizeof(TileStorage<S>),
                                                    grouped_mma<Columns, Gate, Population>, true>();
    grouped_mma<Columns, Gate, Population>
        <<<blocks, S::threads, shared_bytes, execution.stream()>>>(
            Gate ? s.input_codes : s.product_codes,
            static_cast<const float*>(Gate ? s.input_scales : s.product_scales),
            Gate ? experts.bank.gate_up : experts.bank.down, s.groups, s.product_codes,
            static_cast<float*>(s.product_scales),
            static_cast<const float*>(routes.coefficients.data), s.routed_sum, subtiles,
            plan.clamp_limit, std::min(blocks, execution.multiprocessor_count() * 2), plan.fp8,
            2 * plan.gate_columns);
    CUDA_CHECK(cudaGetLastError());
}

template <bool Gate, class Function>
void with_columns(int columns, Function&& function) {
    switch (columns) {
    case 8:
        function.template operator()<8>();
        break;
    case 16:
        function.template operator()<16>();
        break;
    case 32:
        function.template operator()<32>();
        break;
    case 64:
        function.template operator()<64>();
        break;
    case 128:
        if constexpr (Gate) {
            function.template operator()<128>();
            break;
        }
        [[fallthrough]];
    default:
        throw std::logic_error("invalid block FP8 MoE column tile");
    }
}
} // namespace

void launch_input(const Tensor& x, const CudaSwiGLUExperts&, const ExpertRoutes& routes,
                  ExpertSubset subset, const Storage& s, CudaExecutionView execution,
                  bool clear_counts) {
    quantize_input<<<(std::max(x.ne[1] * (kHidden / 4), kExperts) + 255) / 256, 256, 0,
                     execution.stream()>>>(
        static_cast<const __nv_bfloat16*>(x.data), s.input_codes,
        static_cast<float*>(s.input_scales), x.ne[1], s.groups.counts, s.routed_sum, clear_counts,
        static_cast<const int*>(routes.ids.data), static_cast<const int*>(subset.mask.data),
        subset.kind, s.active_tokens);
    CUDA_CHECK(cudaGetLastError());
}

void launch_experts(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                    const Plan& plan, const Storage& s, CudaExecutionView execution) {
    // Population facts are produced on Device and consumed on every replay.
    // Both down families share the same packed assignments and GU product.
    with_columns<true>(plan.gate_columns, [&]<int C>() {
        launch_mma<true, C, PopulationRoute::Gate>(x, experts, routes, s, plan, execution);
    });
    with_columns<false>(plan.down_columns, [&]<int C>() {
        launch_mma<false, C, PopulationRoute::Down>(x, experts, routes, s, plan, execution);
    });
    if (plan.fp8.wide_down)
        launch_mma<false, Fp8PopulationTuning::kWideDownColumns, PopulationRoute::WideDown>(
            x, experts, routes, s, plan, execution);
}

void launch_vector(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, MoeEpilogue mode, Tensor& destination, const Storage& s,
                   const Plan& plan, CudaExecutionView execution) {
    const auto stream        = execution.stream();
    const auto* ids          = static_cast<const int*>(routes.ids.data);
    const auto* mask         = static_cast<const int*>(subset.mask.data);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    if (x.ne[1] == 2) {
        pair_gate<<<dim3(kIntermediate / 4, 2 * kTopK), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), experts.bank.gate_up, ids, subset.kind, mask,
            s.vector_activation, plan.clamp_limit);
    } else {
        vector_gate<<<dim3(kIntermediate / 4, x.ne[1] * kTopK), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), experts.bank.gate_up, ids, subset.kind, mask,
            s.vector_activation, plan.clamp_limit);
    }
    CUDA_CHECK(cudaGetLastError());
    const auto down = [&](auto output) {
        if (x.ne[1] == 2) {
            pair_down<4><<<kHidden / 4, kTopK * 32, 0, stream>>>(
                s.vector_activation, experts.bank.down, ids, coefficients, subset.kind, mask,
                output);
        } else {
            vector_down<4><<<dim3(kHidden / 4, x.ne[1]), kTopK * 32, 0, stream>>>(
                s.vector_activation, experts.bank.down, ids, coefficients, subset.kind, mask,
                output);
        }
        CUDA_CHECK(cudaGetLastError());
    };
    if (s.partial_output)
        down(DenseContributionWithValidity{static_cast<float*>(destination.data), s.active_tokens});
    else
        with_output(destination, mode, down);
}
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8::fp8
