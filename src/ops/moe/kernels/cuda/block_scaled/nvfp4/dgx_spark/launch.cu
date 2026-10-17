#include "ops/moe/kernels/cuda/block_scaled/nvfp4/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/nvfp4/compute.cuh"

#include <algorithm>

namespace ninfer::ops::detail::moe_cuda::block_scaled::nvfp4 {
namespace {
__global__ void quantize_input(const __nv_bfloat16* x, std::uint8_t* codes, std::uint8_t* scales,
                               float divisor, int tokens, int* counts, float* routed_sum,
                               bool clear_counts) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (routed_sum) {
        for (int i = group; i < tokens * kHidden; i += gridDim.x * blockDim.x) {
            routed_sum[i] = 0;
        }
    }
    if (counts && clear_counts && group < kExperts) { counts[group] = 0; }
    if (group < tokens * (kHidden / 16)) {
        const auto packed = common::quantize_nvfp4_k16(x + std::int64_t(group) * 16, divisor);
        *reinterpret_cast<uint2*>(codes + std::int64_t(group) * 8) =
            make_uint2(packed.codes_lo, packed.codes_hi);
        scales[group] = packed.scale;
    }
}

template <bool Gate, int Columns, bool Atomic = false>
void launch_mma(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                const Storage& s, const Plan& plan, CudaExecutionView execution) {
    using S               = Schedule<Columns, Gate>;
    constexpr int rows    = Gate ? kIntermediate : kHidden;
    const int job_columns = plan.job_columns();
    const int subtiles    = job_columns / Columns;
    const int max_jobs    = (kTopK * x.ne[1] / job_columns + kExperts) * subtiles;
    const int blocks      = std::min(max_jobs * (rows / S::rows),
                                     execution.multiprocessor_count() * plan.queued_ctas_per_sm);
    grouped_mma<Columns, Gate, Atomic><<<blocks, S::threads, 0, execution.stream()>>>(
        Gate ? s.input_codes : s.product_codes,
        static_cast<const std::uint8_t*>(Gate ? s.input_scales : s.product_scales),
        Gate ? experts.bank.gate_up : experts.bank.down,
        Gate ? *experts.activation_divisors.gate_up : *experts.activation_divisors.down,
        *experts.activation_divisors.down, s.groups, s.product_codes,
        static_cast<std::uint8_t*>(s.product_scales), s.contributions,
        static_cast<const float*>(routes.coefficients.data), s.routed_sum, subtiles);
    CUDA_CHECK(cudaGetLastError());
}

template <bool Gate, class Function>
void with_columns(int columns, Function&& function) {
    switch (columns) {
    case 8:
        function.template operator()<8>();
        break;
    case 32:
        function.template operator()<32>();
        break;
    case 64:
        function.template operator()<64>();
        break;
    case 128:
        function.template operator()<128>();
        break;
    default:
        throw std::logic_error("invalid NVFP4 MoE column tile");
    }
}
} // namespace

void launch_input(const Tensor& x, const CudaSwiGLUExperts& experts, const Storage& s,
                  CudaExecutionView execution, bool clear_counts) {
    const auto stream = execution.stream();
    quantize_input<<<(std::max(x.ne[1] * (kHidden / 16), kExperts) + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), s.input_codes,
        static_cast<std::uint8_t*>(s.input_scales), *experts.activation_divisors.gate_up, x.ne[1],
        s.groups.counts, s.routed_sum, clear_counts);
    CUDA_CHECK(cudaGetLastError());
}

void launch_experts(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                    const Plan& plan, const Storage& s, CudaExecutionView execution) {
    with_columns<true>(plan.gate_columns, [&]<int C>() {
        launch_mma<true, C>(x, experts, routes, s, plan, execution);
    });
    if (plan.column_major_down) {
        launch_column_jobs(s.groups, x.ne[1], plan.job_columns(), execution);
    }
    with_columns<false>(plan.down_columns, [&]<int C>() {
        if constexpr (C == 128) {
            if (!plan.atomic_output) {
                throw std::logic_error("NVFP4 down128 requires atomic output");
            }
            launch_mma<false, C, true>(x, experts, routes, s, plan, execution);
        } else if (plan.atomic_output) {
            launch_mma<false, C, true>(x, experts, routes, s, plan, execution);
        } else {
            launch_mma<false, C>(x, experts, routes, s, plan, execution);
        }
    });
}

void launch_vector(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, bool shared, MoeEpilogue mode, Tensor& destination,
                   const Storage& s, CudaExecutionView execution) {
    const auto stream        = execution.stream();
    const auto* ids          = static_cast<const int*>(routes.ids.data);
    const auto* mask         = static_cast<const int*>(subset.mask.data);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    vector_gate<<<dim3(kIntermediate / 4, x.ne[1] * kTopK), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), experts.bank.gate_up, ids, subset.kind, mask,
        s.vector_activation);
    CUDA_CHECK(cudaGetLastError());
    const auto down = [&](auto output) {
        vector_down<4><<<dim3(kHidden / 4, x.ne[1]), kTopK * 32, 0, stream>>>(
            s.vector_activation, experts.bank.down, ids, coefficients, subset.kind, mask, output);
        CUDA_CHECK(cudaGetLastError());
    };
    if (shared) {
        down(FloatContribution{s.routed_sum});
    } else {
        with_output(destination, mode, down);
    }
}
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::nvfp4
