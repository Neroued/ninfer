#include "ops/moe/kernels/cuda/block_scaled/nvfp4/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/nvfp4/compute.cuh"

#include "ops/moe/kernels/cuda/block_scaled/nvfp4/dgx_spark/launch.cuh"

#include <algorithm>
#include <climits>

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

template <int Intermediate, bool Gate, int Columns, bool Atomic, int Stages, int KStep,
          int Eviction, bool CachedExpertMajor = false>
void launch_mma(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                const Storage& s, const Plan& plan, CudaExecutionView execution,
                int minimum_expert_count = 0, int maximum_expert_count = 0x7fffffff) {
    constexpr int reduction   = Gate ? kHidden : Intermediate;
    constexpr int stage_limit = (reduction + KStep - 1) / KStep;
    constexpr int stages      = Stages < stage_limit ? Stages : stage_limit;
    using S                   = Schedule<Columns, Gate, stages, KStep, Eviction, Intermediate>;
    constexpr int rows        = Gate ? Intermediate : kHidden;
    const int job_columns     = plan.job_columns();
    const int subtiles        = job_columns / Columns;
    const int max_jobs        = (kTopK * x.ne[1] / job_columns + kExperts) * subtiles;
    const int blocks          = std::min(max_jobs * (rows / S::rows),
                                         execution.multiprocessor_count() * plan.queued_ctas_per_sm);
    constexpr auto kernel     = grouped_mma<Columns, Gate, Atomic, stages, KStep, Eviction,
                                            CachedExpertMajor, Intermediate>;
    constexpr int bytes       = sizeof(TileStorage<S>) > 48 * 1024 ? sizeof(TileStorage<S>) : 0;
    if constexpr (bytes) {
        static const auto status =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        CUDA_CHECK(status);
    }
    kernel<<<blocks, S::threads, bytes, execution.stream()>>>(
        Gate ? s.input_codes : s.product_codes,
        static_cast<const std::uint8_t*>(Gate ? s.input_scales : s.product_scales),
        Gate ? experts.bank.gate_up : experts.bank.down,
        Gate ? *experts.activation_divisors.gate_up : *experts.activation_divisors.down,
        *experts.activation_divisors.down, s.groups, s.product_codes,
        static_cast<std::uint8_t*>(s.product_scales), s.contributions,
        static_cast<const float*>(routes.coefficients.data), s.routed_sum, subtiles,
        plan.nvfp4.down_row_window, static_cast<int>(plan.nvfp4.cache), minimum_expert_count,
        maximum_expert_count);
    CUDA_CHECK(cudaGetLastError());
}

template <int Intermediate, bool Gate, int Columns, bool Atomic = false>
void launch_tuned_mma(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                      const Storage& s, const Plan& plan, CudaExecutionView execution,
                      int minimum_expert_count = 0, int maximum_expert_count = 0x7fffffff) {
    if (plan.nvfp4.pipeline == Nvfp4Pipeline::K128 &&
        plan.nvfp4.cache == Nvfp4CachePolicy::Cached && (Gate || plan.nvfp4.down_row_window == 0)) {
        launch_mma<Intermediate, Gate, Columns, Atomic, 2, 128, 0, true>(
            x, experts, routes, s, plan, execution, minimum_expert_count, maximum_expert_count);
        return;
    }
    with_nvfp4_tuning<Columns, true>(
        plan.nvfp4.pipeline, plan.nvfp4.cache, [&]<int Stages, int KStep, int Eviction>() {
            launch_mma<Intermediate, Gate, Columns, Atomic, Stages, KStep, Eviction>(
                x, experts, routes, s, plan, execution, minimum_expert_count, maximum_expert_count);
        });
}

template <class Function>
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

template <int Intermediate>
void launch_experts_impl(const Tensor& x, const CudaSwiGLUExperts& experts,
                         const ExpertRoutes& routes, const Plan& plan, const Storage& s,
                         CudaExecutionView execution) {
    const int limit  = plan.dense.gate_columns ? plan.small_expert_limit : INT_MAX;
    const auto phase = [&]<bool Gate>() {
        const auto project = [&](const Plan& tuning, int first, int last) {
            if (first >= x.ne[1]) return;
            with_columns(Gate ? tuning.gate_columns : tuning.down_columns, [&]<int C>() {
                const auto launch = [&]<bool Atomic>() {
                    launch_tuned_mma<Intermediate, Gate, C, Atomic>(x, experts, routes, s, tuning,
                                                                    execution, first, last);
                };
                if constexpr (Gate) {
                    launch.template operator()<false>();
                } else if constexpr (C == 128) {
                    if (!plan.atomic_output)
                        throw std::logic_error("NVFP4 down128 requires atomic output");
                    launch.template operator()<true>();
                } else if (plan.atomic_output) {
                    launch.template operator()<true>();
                } else {
                    launch.template operator()<false>();
                }
            });
        };
        const auto load_class = [&](const Nvfp4LoadSchedule& schedule, int first, int last) {
            auto tuning               = plan;
            tuning.group_columns      = plan.job_columns();
            tuning.gate_columns       = schedule.gate_columns;
            tuning.down_columns       = schedule.down_columns;
            tuning.nvfp4              = schedule.tuning;
            tuning.queued_ctas_per_sm = schedule.queued_ctas_per_sm;
            project(tuning, first, last);
        };
        project(plan, 0, limit);
        if (plan.medium.gate_columns) load_class(plan.medium, limit, plan.medium.max_load);
        if (plan.dense.gate_columns)
            load_class(plan.dense, plan.medium.gate_columns ? plan.medium.max_load : limit,
                       INT_MAX);
    };
    phase.template operator()<true>();
    if (plan.column_major_down) {
        launch_column_jobs(s.groups, x.ne[1], plan.job_columns(), plan.preserve_balanced_down,
                           execution);
    }
    phase.template operator()<false>();
}

template <int Intermediate>
void launch_vector_impl(const Tensor& x, const CudaSwiGLUExperts& experts,
                        const ExpertRoutes& routes, ExpertSubset subset, bool shared,
                        MoeEpilogue mode, Tensor& destination, const Storage& s,
                        CudaExecutionView execution) {
    const auto stream        = execution.stream();
    const auto* ids          = static_cast<const int*>(routes.ids.data);
    const auto* mask         = static_cast<const int*>(subset.mask.data);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    vector_gate<Intermediate><<<dim3(Intermediate / 4, x.ne[1] * kTopK), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), experts.bank.gate_up, ids, subset.kind, mask,
        s.vector_activation);
    CUDA_CHECK(cudaGetLastError());
    const auto down = [&](auto output) {
        vector_down<Intermediate, 8, 2><<<dim3(kHidden / 8, x.ne[1]), 2 * 32, 0, stream>>>(
            s.vector_activation, experts.bank.down, ids, coefficients, subset.kind, mask, output);
        CUDA_CHECK(cudaGetLastError());
    };
    if (shared) {
        down(FloatContribution{s.routed_sum});
    } else {
        with_output(destination, mode, down);
    }
}

void launch_experts(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                    const Plan& plan, const Storage& s, CudaExecutionView execution) {
    if (plan.intermediate == 320)
        launch_experts_impl<320>(x, experts, routes, plan, s, execution);
    else
        launch_experts_impl<640>(x, experts, routes, plan, s, execution);
}

void launch_vector(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, bool shared, MoeEpilogue mode, Tensor& destination,
                   const Storage& s, const Plan& plan, CudaExecutionView execution) {
    if (plan.intermediate == 320)
        launch_vector_impl<320>(x, experts, routes, subset, shared, mode, destination, s,
                                execution);
    else
        launch_vector_impl<640>(x, experts, routes, subset, shared, mode, destination, s,
                                execution);
}
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::nvfp4
