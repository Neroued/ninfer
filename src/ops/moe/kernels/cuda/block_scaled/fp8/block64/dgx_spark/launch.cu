#include "ops/moe/kernels/cuda/block_scaled/fp8/block64/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/fp8/block64/compute.cuh"
#include "ops/common/host/launch.h"

#include <algorithm>
#include <bit>

namespace ninfer::ops::detail::moe_cuda::block_scaled::fp8::block64 {
namespace {
template <bool Gate, int Columns, bool Atomic = false>
void launch_mma(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                const Storage& storage, const Plan& plan, CudaExecutionView execution,
                int minimum_count = 0, int maximum_count = 0x7fffffff) {
    using S                 = Schedule<Columns, Gate>;
    constexpr int row_tiles = Gate ? (kLocalIntermediate + 127) / 128 : kHidden / 128;
    const int job_columns = plan.job_columns(), subtiles = job_columns / Columns;
    const int max_jobs = (kTopK * x.ne[1] / job_columns + kExperts) * subtiles;
    const int blocks =
        std::min(max_jobs * row_tiles, execution.multiprocessor_count() * plan.queued_ctas_per_sm);
    const int bytes =
        common::prepare_shared<sizeof(TileStorage<S>), grouped_mma<Columns, Gate, Atomic>, true>();
    grouped_mma<Columns, Gate, Atomic><<<blocks, S::threads, bytes, execution.stream()>>>(
        Gate ? storage.input_codes : storage.product_codes,
        static_cast<const float*>(Gate ? storage.input_scales : storage.product_scales),
        Gate ? experts.bank.gate_up : experts.bank.down, storage.groups, storage.product_codes,
        static_cast<float*>(storage.product_scales), storage.contributions,
        static_cast<const float*>(routes.coefficients.data), storage.routed_sum,
        std::countr_zero(unsigned(subtiles)), minimum_count, maximum_count);
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
        throw std::logic_error("invalid block64 FP8 column tile");
    }
}
} // namespace

void launch_experts(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                    const Plan& plan, const Storage& storage, CudaExecutionView execution) {
    const int limit  = plan.fp8.maximum_small_load ? plan.fp8.maximum_small_load : 0x7fffffff;
    const auto phase = [&]<bool Gate>() {
        const auto project = [&](const Plan& tuning, int first, int last) {
            with_columns<Gate>(Gate ? tuning.gate_columns : tuning.down_columns, [&]<int C>() {
                if constexpr (Gate)
                    launch_mma<true, C>(x, experts, routes, storage, tuning, execution, first,
                                        last);
                else if (plan.atomic_output)
                    launch_mma<false, C, true>(x, experts, routes, storage, tuning, execution,
                                               first, last);
                else
                    launch_mma<false, C>(x, experts, routes, storage, tuning, execution, first,
                                         last);
            });
        };
        project(plan, 0, limit);
        if (plan.fp8.maximum_small_load) {
            auto large               = plan;
            large.gate_columns       = plan.fp8.large_gate_columns;
            large.down_columns       = plan.fp8.large_down_columns;
            large.queued_ctas_per_sm = plan.fp8.large_queued_ctas_per_sm;
            const int maximum =
                plan.fp8.maximum_large_load ? plan.fp8.maximum_large_load : 0x7fffffff;
            project(large, limit, maximum);
            if (plan.fp8.maximum_large_load) {
                large.queued_ctas_per_sm = plan.fp8.dense_queued_ctas_per_sm;
                project(large, maximum, 0x7fffffff);
            }
        }
    };
    phase.template operator()<true>();
    if (plan.column_major_down)
        launch_column_jobs(storage.groups, x.ne[1], plan.job_columns(), plan.preserve_balanced_down,
                           execution);
    phase.template operator()<false>();
}

void launch_vector(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, bool shared, MoeEpilogue mode, Tensor& destination,
                   const Storage& storage, CudaExecutionView execution) {
    const auto* ids          = static_cast<const int*>(routes.ids.data);
    const auto* mask         = static_cast<const int*>(subset.mask.data);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    vector_gate<<<dim3(kLocalIntermediate / 4, x.ne[1] * kTopK), 128, 0, execution.stream()>>>(
        static_cast<const __nv_bfloat16*>(x.data), experts.bank.gate_up, ids, subset.kind, mask,
        storage.vector_activation);
    CUDA_CHECK(cudaGetLastError());
    const auto down = [&](auto output) {
        vector_down<4><<<dim3(kHidden / 4, x.ne[1]), kTopK * 32, 0, execution.stream()>>>(
            storage.vector_activation, experts.bank.down, ids, coefficients, subset.kind, mask,
            output);
        CUDA_CHECK(cudaGetLastError());
    };
    if (shared)
        down(FloatContribution{storage.routed_sum});
    else
        with_output(destination, mode, down);
}
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::fp8::block64
