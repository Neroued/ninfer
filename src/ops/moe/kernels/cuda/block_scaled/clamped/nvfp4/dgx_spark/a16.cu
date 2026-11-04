#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/a16.cuh"
#include "ops/common/host/launch.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4 {
namespace {

template <int I, bool Gate, int Rows, int Columns, int KStep, int Stages, bool Compressed>
void mma(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
         const Plan& p, const Storage& s, CudaExecutionView execution, int minimum, int maximum) {
    using S             = a16::Schedule<I, Gate, Rows, Columns, KStep, Stages>;
    constexpr int bytes = sizeof(a16::Tile<S>);
    if constexpr (bytes > 96 * 1024) {
        throw std::logic_error("NVFP4 A16 tile exceeds shared memory");
    } else {
        constexpr auto entry    = a16::grouped<I, Gate, Rows, Columns, KStep, Stages, Compressed>;
        const int shared        = common::prepare_shared<bytes, entry>();
        const int possible_jobs = kTopK * x.ne[1] / p.job_columns() + kExperts;
        const int row_tiles     = (Gate ? I : kHidden) / S::feature_rows;
        const int maximum_load  = std::min(maximum, x.ne[1]);
        const int subtiles      = (std::min(p.job_columns(), maximum_load) + Columns - 1) / Columns;
        const int blocks        = std::min(possible_jobs * row_tiles * subtiles,
                                           execution.multiprocessor_count() * p.queued_ctas_per_sm);
        entry<<<blocks, S::threads, shared, execution.stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            Gate ? experts.bank.gate_up : experts.bank.down, s.groups, s.compensated_product,
            s.contributions, static_cast<const float*>(routes.coefficients.data), s.routed_sum,
            p.job_columns(), p.clamp_limit, minimum, maximum_load, p.nvfp4.down_row_window);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <int I, bool Gate, int Columns, bool Compressed>
void tuned(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
           const Plan& p, const Storage& s, CudaExecutionView execution, int minimum, int maximum,
           const Nvfp4A16Tile& tuning) {
    const auto rows = [&]<int Rows>() {
        const auto k = [&]<int KStep>() {
            const auto stages = [&]<int Stages>() {
                mma<I, Gate, Rows, Columns, KStep, Stages, Compressed>(x, experts, routes, p, s,
                                                                       execution, minimum, maximum);
            };
            if (tuning.stages == 1)
                stages.template operator()<1>();
            else if (tuning.stages == 2)
                stages.template operator()<2>();
            else
                throw std::logic_error("invalid NVFP4 A16 stage count");
        };
        if (tuning.k_step == 64)
            k.template operator()<64>();
        else if (tuning.k_step == 128)
            k.template operator()<128>();
        else
            throw std::logic_error("invalid NVFP4 A16 K tile");
    };
    const int count = Gate ? tuning.gate_rows : tuning.down_rows;
    if (count == 64)
        rows.template operator()<64>();
    else if (count == 128)
        rows.template operator()<128>();
    else
        throw std::logic_error("invalid NVFP4 A16 row tile");
}

template <int I, bool Gate, bool Compressed>
void project(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
             const Plan& p, const Storage& s, CudaExecutionView execution, int columns, int minimum,
             int maximum, const Nvfp4A16Tile& tuning) {
    switch (columns) {
    case 8:
        tuned<I, Gate, 8, Compressed>(x, experts, routes, p, s, execution, minimum, maximum,
                                      tuning);
        break;
    case 16:
        tuned<I, Gate, 16, Compressed>(x, experts, routes, p, s, execution, minimum, maximum,
                                       tuning);
        break;
    case 32:
        tuned<I, Gate, 32, Compressed>(x, experts, routes, p, s, execution, minimum, maximum,
                                       tuning);
        break;
    case 64:
        tuned<I, Gate, 64, Compressed>(x, experts, routes, p, s, execution, minimum, maximum,
                                       tuning);
        break;
    case 128:
        tuned<I, Gate, 128, Compressed>(x, experts, routes, p, s, execution, minimum, maximum,
                                        tuning);
        break;
    default:
        throw std::logic_error("invalid NVFP4 A16 column tile");
    }
}

template <int I, bool Compressed>
void launch(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
            const Plan& p, const Storage& s, CudaExecutionView execution) {
    const int vector_limit = Compressed ? 0 : p.a16.vector_max_load;
    if (vector_limit < 0 || vector_limit > 4)
        throw std::logic_error("NVFP4 A16 vector load must be in [0,4]");
    const auto phase = [&]<bool Gate>() {
        if constexpr (!Compressed) {
            if (vector_limit) {
                constexpr int rows   = Gate ? I : kHidden;
                const int jobs       = kTopK * x.ne[1] / p.job_columns() + kExperts;
                const int block_rows = Gate ? 4 : 8;
                const int blocks =
                    std::min(jobs * (rows / block_rows),
                             execution.multiprocessor_count() * p.queued_ctas_per_sm);
                const auto vector = [&]<int Columns>() {
                    if constexpr (Gate) {
                        a16::vector_gate<I, Columns><<<blocks, 128, 0, execution.stream()>>>(
                            static_cast<const __nv_bfloat16*>(x.data), experts.bank.gate_up,
                            s.groups, s.compensated_product, p.clamp_limit, vector_limit);
                    } else {
                        a16::vector_down<I, Columns><<<blocks, 128, 0, execution.stream()>>>(
                            experts.bank.down, s.groups, s.compensated_product, s.contributions,
                            static_cast<const float*>(routes.coefficients.data), s.routed_sum,
                            vector_limit);
                    }
                    CUDA_CHECK(cudaGetLastError());
                };
                const int columns = std::min(vector_limit, x.ne[1]);
                if (columns == 1)
                    vector.template operator()<1>();
                else if (columns == 2)
                    vector.template operator()<2>();
                else
                    vector.template operator()<4>();
            }
        }
        const int columns = Gate ? p.gate_columns : p.down_columns;
        const int medium  = Gate ? p.medium.gate_columns : p.medium.down_columns;
        const int dense   = Gate ? p.dense.gate_columns : p.dense.down_columns;
        const int hot     = Gate ? p.broad.gate_columns : p.broad.down_columns;
        if (medium) {
            if (p.small_expert_limit > vector_limit && x.ne[1] > vector_limit)
                project<I, Gate, Compressed>(x, experts, routes, p, s, execution, columns,
                                             vector_limit, p.small_expert_limit, p.a16.small);
            if (x.ne[1] > p.small_expert_limit)
                project<I, Gate, Compressed>(x, experts, routes, p, s, execution, medium,
                                             std::max(vector_limit, p.small_expert_limit),
                                             dense ? p.medium.max_load : INT32_MAX, p.a16.medium);
            if (dense && x.ne[1] > p.medium.max_load)
                project<I, Gate, Compressed>(x, experts, routes, p, s, execution, dense,
                                             std::max(vector_limit, p.medium.max_load),
                                             hot ? p.dense.max_load : INT32_MAX, p.a16.dense);
            if (hot && x.ne[1] > p.dense.max_load)
                project<I, Gate, Compressed>(x, experts, routes, p, s, execution, hot,
                                             std::max(vector_limit, p.dense.max_load), INT32_MAX,
                                             p.a16.hot);
        } else if (x.ne[1] > vector_limit) {
            project<I, Gate, Compressed>(x, experts, routes, p, s, execution, columns, vector_limit,
                                         INT32_MAX, p.a16.small);
        }
    };
    phase.template operator()<true>();
    if (p.column_major_down)
        launch_column_jobs(s.groups, x.ne[1], p.job_columns(), p.preserve_balanced_down, execution);
    phase.template operator()<false>();
}

} // namespace

void launch_a16_experts(const Tensor& x, const CudaSwiGLUExperts& experts,
                        const ExpertRoutes& routes, const Plan& p, const Storage& s,
                        CudaExecutionView execution) {
    const bool csf = experts.bank.gate_up.layout == QuantLayout::Csf32;
    if (p.intermediate == 1024) {
        if (csf)
            launch<1024, true>(x, experts, routes, p, s, execution);
        else
            launch<1024, false>(x, experts, routes, p, s, execution);
    } else if (p.intermediate == 2048) {
        if (csf)
            launch<2048, true>(x, experts, routes, p, s, execution);
        else
            launch<2048, false>(x, experts, routes, p, s, execution);
    } else {
        throw std::logic_error("unregistered NVFP4 A16 expert width");
    }
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4
