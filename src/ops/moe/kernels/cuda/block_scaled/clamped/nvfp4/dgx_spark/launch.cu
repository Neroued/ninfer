#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/compute.cuh"

#include "ops/moe/kernels/cuda/block_scaled/nvfp4/dgx_spark/launch.cuh"

#include <algorithm>

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4 {
namespace {
__global__ void quantize_input(const __nv_bfloat16* x, std::uint8_t* codes, std::uint8_t* scales,
                               float divisor, int tokens, int* counts, float* routed_sum,
                               bool clear_counts, const int* ids, const int* mask,
                               ExpertSubsetKind kind, int* active_tokens) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    // H4096/G16 gives one 256-thread CTA per token. Publish liveness before any
    // activation load; inactive columns have no grouped assignments and no scratch consumer.
    __shared__ int live;
    const int token = blockIdx.x;
    // Count initialization may need one more CTA than the token grid at T=1.
    // That CTA owns counts only; it must not read routes or publish a token.
    if (counts && clear_counts && group < kExperts) counts[group] = 0;
    if (token >= tokens) return;
    if (threadIdx.x < 32) {
        bool hit = false;
        if (threadIdx.x < kTopK)
            hit = includes(kind, mask, ids[std::int64_t(token) * kTopK + threadIdx.x]);
        const unsigned votes = __ballot_sync(0xffffffff, hit);
        if (threadIdx.x == 0) {
            live                 = votes != 0;
            active_tokens[token] = votes != 0;
        }
    }
    __syncthreads();
    if (!live) return;
    if (routed_sum)
        for (int row = threadIdx.x; row < kHidden; row += blockDim.x)
            routed_sum[std::int64_t(token) * kHidden + row] = 0;
    if (group < tokens * (kHidden / 16)) {
        const auto packed = common::quantize_nvfp4_k16(x + std::int64_t(group) * 16, divisor);
        *reinterpret_cast<uint2*>(codes + std::int64_t(group) * 8) =
            make_uint2(packed.codes_lo, packed.codes_hi);
        scales[group] = packed.scale;
    }
}

template <int Intermediate, bool Gate, int Columns, bool Atomic, int Stages, int KStep,
          int Eviction, bool CachedExpertMajor = false, bool FilterLoads = false>
void launch_mma(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                const Storage& s, const Plan& plan, CudaExecutionView execution,
                int minimum_expert_count = 0, int maximum_expert_count = 0x7fffffff,
                ExpertLoadRegime regime = ExpertLoadRegime::All) {
    using S               = Schedule<Columns, Gate, Stages, KStep, Eviction, Intermediate>;
    constexpr int rows    = Gate ? Intermediate : kHidden;
    const int job_columns = plan.job_columns();
    const int subtiles    = job_columns / Columns;
    const int max_jobs    = (kTopK * x.ne[1] / job_columns + kExperts) * subtiles;
    const int blocks      = std::min(max_jobs * (rows / S::rows),
                                     execution.multiprocessor_count() * plan.queued_ctas_per_sm);
    constexpr auto kernel = grouped_mma<Columns, Gate, Atomic, Stages, KStep, Eviction,
                                        CachedExpertMajor, FilterLoads, Intermediate>;
    constexpr int bytes   = sizeof(TileStorage<S>) > 48 * 1024 ? sizeof(TileStorage<S>) : 0;
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
        static_cast<const float*>(routes.coefficients.data), s.routed_sum,
        std::countr_zero(unsigned(subtiles)), plan.clamp_limit, plan.nvfp4.down_row_window,
        static_cast<int>(plan.nvfp4.cache), minimum_expert_count, maximum_expert_count, regime,
        plan.broad_short_limit, plan.broad_long_limit, plan.broad.max_load, x.ne[1],
        Intermediate == 1024 && plan.dense_down_columns != 0);
    CUDA_CHECK(cudaGetLastError());
}

template <int Intermediate, bool Gate, int Columns, bool Atomic = false>
void launch_tuned_mma(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                      const Storage& s, const Plan& plan, CudaExecutionView execution,
                      int minimum_expert_count = 0, int maximum_expert_count = 0x7fffffff,
                      ExpertLoadRegime regime = ExpertLoadRegime::All) {
    const auto launch = [&]<int Stages, int KStep, int Eviction, bool Cached>() {
        if constexpr (Columns == 128) {
            if (regime != ExpertLoadRegime::All) {
                launch_mma<Intermediate, Gate, Columns, Atomic, Stages, KStep, Eviction, Cached,
                           true>(x, experts, routes, s, plan, execution, minimum_expert_count,
                                 maximum_expert_count, regime);
                return;
            }
        }
        launch_mma<Intermediate, Gate, Columns, Atomic, Stages, KStep, Eviction, Cached>(
            x, experts, routes, s, plan, execution, minimum_expert_count, maximum_expert_count);
    };
    if (plan.nvfp4.pipeline == Nvfp4Pipeline::K128 &&
        plan.nvfp4.cache == Nvfp4CachePolicy::Cached && (Gate || plan.nvfp4.down_row_window == 0)) {
        launch.template operator()<2, 128, 0, true>();
        return;
    }
    with_nvfp4_tuning<Columns, true>(
        plan.nvfp4.pipeline, plan.nvfp4.cache, [&]<int Stages, int KStep, int Eviction>() {
            launch.template operator()<Stages, KStep, Eviction, false>();
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

void launch_input(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                  ExpertSubset subset, const Storage& s, CudaExecutionView execution,
                  bool clear_counts) {
    const auto stream = execution.stream();
    quantize_input<<<(std::max(x.ne[1] * (kHidden / 16), kExperts) + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), s.input_codes,
        static_cast<std::uint8_t*>(s.input_scales), *experts.activation_divisors.gate_up, x.ne[1],
        s.groups.counts, s.routed_sum, clear_counts, static_cast<const int*>(routes.ids.data),
        static_cast<const int*>(subset.mask.data), subset.kind, s.active_tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <int Intermediate>
void launch_experts_impl(const Tensor& x, const CudaSwiGLUExperts& experts,
                         const ExpertRoutes& routes, const Plan& plan, const Storage& s,
                         CudaExecutionView execution) {
    if (plan.dense_down_columns && !plan.atomic_output)
        throw std::logic_error("dense token down requires a summed FP32 contribution");
    const int limit  = plan.dense.gate_columns ? plan.small_expert_limit : 0x7fffffff;
    const bool broad = plan.broad.gate_columns != 0;
    const auto phase = [&]<bool Gate>() {
        const auto project = [&](const Plan& tuning, int first, int last, ExpertLoadRegime regime) {
            with_columns(Gate ? tuning.gate_columns : tuning.down_columns, [&]<int C>() {
                const auto launch = [&]<bool Atomic>() {
                    launch_tuned_mma<Intermediate, Gate, C, Atomic>(x, experts, routes, s, tuning,
                                                                    execution, first, last, regime);
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
        const auto load_class = [&](const Nvfp4LoadSchedule& schedule, int first, int last,
                                    ExpertLoadRegime regime) {
            auto tuning               = plan;
            tuning.group_columns      = plan.job_columns();
            tuning.gate_columns       = schedule.gate_columns;
            tuning.down_columns       = schedule.down_columns;
            tuning.nvfp4              = schedule.tuning;
            tuning.queued_ctas_per_sm = schedule.queued_ctas_per_sm;
            project(tuning, first, last, regime);
        };
        const auto primary = broad ? ExpertLoadRegime::ExcludeBroad : ExpertLoadRegime::All;
        project(plan, 0, limit, primary);
        if (plan.medium.gate_columns) load_class(plan.medium, limit, plan.medium.max_load, primary);
        if (plan.dense.gate_columns)
            load_class(plan.dense, plan.medium.gate_columns ? plan.medium.max_load : limit,
                       0x7fffffff,
                       broad ? ExpertLoadRegime::DenseWhenBroad : ExpertLoadRegime::All);
        if (broad) load_class(plan.broad, 0, plan.broad.max_load, ExpertLoadRegime::OnlyBroad);
    };
    phase.template operator()<true>();
    if constexpr (Intermediate == 1024) {
        if (plan.dense_down_columns) {
            const auto launch_dense = [&]<int C>() {
                const int tasks  = (kHidden / 128) * ((x.ne[1] + C - 1) / C);
                const int blocks = std::min(tasks, execution.multiprocessor_count() * 3);
                dense_token_down<C><<<blocks, Schedule<C, false>::threads, 0, execution.stream()>>>(
                    s.product_codes, static_cast<const std::uint8_t*>(s.product_scales),
                    experts.bank.down, *experts.activation_divisors.down, s.groups,
                    static_cast<const int*>(routes.ids.data),
                    static_cast<const float*>(routes.coefficients.data), s.routed_sum, x.ne[1]);
                CUDA_CHECK(cudaGetLastError());
            };
            switch (plan.dense_down_columns) {
            case 16:
                launch_dense.template operator()<16>();
                break;
            case 32:
                launch_dense.template operator()<32>();
                break;
            case 64:
                launch_dense.template operator()<64>();
                break;
            case 128:
                launch_dense.template operator()<128>();
                break;
            default:
                throw std::logic_error("invalid dense token down column tile");
            }
        }
    }
    if (plan.column_major_down) {
        launch_column_jobs(s.groups, x.ne[1], plan.job_columns(), plan.preserve_balanced_down,
                           execution);
    }
    phase.template operator()<false>();
}

template <int Intermediate>
void launch_vector_impl(const Tensor& x, const CudaSwiGLUExperts& experts,
                        const ExpertRoutes& routes, ExpertSubset subset, MoeEpilogue mode,
                        Tensor& destination, const Storage& s, const Plan& plan,
                        CudaExecutionView execution) {
    const auto stream        = execution.stream();
    const auto* ids          = static_cast<const int*>(routes.ids.data);
    const auto* mask         = static_cast<const int*>(subset.mask.data);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    vector_gate<Intermediate><<<dim3(Intermediate / 4, x.ne[1] * kTopK), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), experts.bank.gate_up, ids, subset.kind, mask,
        s.direct_product, plan.clamp_limit);
    CUDA_CHECK(cudaGetLastError());
    const auto down = [&](auto output) {
        vector_down<Intermediate, 8, 4><<<dim3(kHidden / 8, x.ne[1]), 4 * 32, 0, stream>>>(
            s.direct_product, experts.bank.down, ids, coefficients, subset.kind, mask, output);
        CUDA_CHECK(cudaGetLastError());
    };
    if (s.partial_output)
        down(DenseContributionWithValidity{static_cast<float*>(destination.data), s.active_tokens});
    else
        with_output(destination, mode, down);
}

void launch_experts(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                    const Plan& plan, const Storage& storage, CudaExecutionView execution) {
    if (plan.intermediate == 1024)
        launch_experts_impl<1024>(x, experts, routes, plan, storage, execution);
    else
        launch_experts_impl<2048>(x, experts, routes, plan, storage, execution);
}

void launch_vector(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, MoeEpilogue mode, Tensor& destination,
                   const Storage& storage, const Plan& plan, CudaExecutionView execution) {
    if (plan.intermediate == 1024)
        launch_vector_impl<1024>(x, experts, routes, subset, mode, destination, storage, plan,
                                 execution);
    else
        launch_vector_impl<2048>(x, experts, routes, subset, mode, destination, storage, plan,
                                 execution);
}
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4
