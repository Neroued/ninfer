#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/a16.cuh"
#include "ops/common/host/launch.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4 {
namespace {

template <int I, bool Gate, int Rows, int Columns, int KStep, int Stages, bool Compressed,
          bool Fp16>
void mma(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
         const Plan& p, const Storage& s, CudaExecutionView execution, int minimum, int maximum,
         const Nvfp4A16Tile& tuning) {
    const auto run = [&]<bool RegisterPipeline>() {
        using S = a16::Schedule<I, Gate, Rows, Columns, KStep, Stages, Fp16, RegisterPipeline>;
        constexpr int bytes = sizeof(a16::Tile<S>);
        if constexpr (bytes > 96 * 1024) {
            throw std::logic_error("NVFP4 A16 tile exceeds shared memory");
        } else {
            constexpr auto entry =
                a16::grouped<I, Gate, Rows, Columns, KStep, Stages, Compressed, Fp16, RegisterPipeline>;
            const int shared        = common::prepare_shared<bytes, entry>();
            const int possible_jobs = kTopK * x.ne[1] / p.job_columns() + kExperts;
            const int row_tiles     = (Gate ? I : kHidden) / S::feature_rows;
            const int maximum_load  = std::min(maximum, x.ne[1]);
            const int subtiles      = (std::min(p.job_columns(), maximum_load) + Columns - 1) / Columns;
            const int queued        = tuning.queued_ctas_per_sm ? tuning.queued_ctas_per_sm
                                                              : p.queued_ctas_per_sm;
            int blocks              = std::min(possible_jobs * row_tiles * subtiles,
                                               execution.multiprocessor_count() * queued);
            // Rotate persistent CTAs through partial column jobs. An even stride
            // can otherwise leave whole CTAs assigned only to empty tail subtiles.
            if constexpr (Fp16) {
                if (subtiles > 1 && blocks > 1 && !(blocks & 1)) --blocks;
            }
            auto dispatch = p.a16.dispatch(Gate);
            dispatch.full_rows_when_mixed = tuning.full_rows_when_mixed;
            entry<<<blocks, S::threads, shared, execution.stream()>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                Gate ? experts.bank.gate_up : experts.bank.down, s.groups, s.a16_product,
                s.contributions, static_cast<const float*>(routes.coefficients.data), s.routed_sum,
                p.job_columns(), p.clamp_limit, minimum, maximum_load,
                Gate ? 0 : p.nvfp4.down_row_window,
                dispatch);
            CUDA_CHECK(cudaGetLastError());
        }
    };
    if constexpr (Gate) {
        if (tuning.gate_register_pipeline) {
            if constexpr (Rows == 128 && Columns == 64 && KStep == 128 && Stages == 2 &&
                          !Compressed && Fp16)
                return run.template operator()<true>();
            else
                throw std::logic_error("unregistered NVFP4 A16 register pipeline");
        }
    }
    run.template operator()<false>();
}

template <int I, bool Gate, int Columns, bool Compressed, bool Fp16>
void tuned(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
           const Plan& p, const Storage& s, CudaExecutionView execution, int minimum, int maximum,
           const Nvfp4A16Tile& tuning) {
    const auto rows = [&]<int Rows>() {
        const auto k = [&]<int KStep>() {
            const auto stages = [&]<int Stages>() {
                mma<I, Gate, Rows, Columns, KStep, Stages, Compressed, Fp16>(
                    x, experts, routes, p, s, execution, minimum, maximum, tuning);
            };
            const int stage_count = Gate && tuning.gate_register_pipeline ? 2 : tuning.stages;
            if (stage_count == 1)
                stages.template operator()<1>();
            else if (stage_count == 2)
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

template <int I, bool Gate, bool Compressed, bool Fp16>
void project(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
             const Plan& p, const Storage& s, CudaExecutionView execution, int columns, int minimum,
             int maximum, const Nvfp4A16Tile& tuning) {
    switch (columns) {
    case 8:
        tuned<I, Gate, 8, Compressed, Fp16>(x, experts, routes, p, s, execution, minimum, maximum,
                                            tuning);
        break;
    case 16:
        tuned<I, Gate, 16, Compressed, Fp16>(x, experts, routes, p, s, execution, minimum, maximum,
                                             tuning);
        break;
    case 32:
        tuned<I, Gate, 32, Compressed, Fp16>(x, experts, routes, p, s, execution, minimum, maximum,
                                             tuning);
        break;
    case 64:
        tuned<I, Gate, 64, Compressed, Fp16>(x, experts, routes, p, s, execution, minimum, maximum,
                                             tuning);
        break;
    case 128:
        tuned<I, Gate, 128, Compressed, Fp16>(x, experts, routes, p, s, execution, minimum, maximum,
                                              tuning);
        break;
    default:
        throw std::logic_error("invalid NVFP4 A16 column tile");
    }
}

template <int I, bool Compressed, bool Fp16>
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
                        a16::vector_gate<I, Columns, Fp16><<<blocks, 128, 0, execution.stream()>>>(
                            static_cast<const __nv_bfloat16*>(x.data), experts.bank.gate_up,
                            s.groups, s.a16_product, p.clamp_limit, vector_limit,
                            p.a16.dispatch(true));
                    } else {
                        a16::vector_down<I, Columns, Fp16><<<blocks, 128, 0, execution.stream()>>>(
                            experts.bank.down, s.groups, s.a16_product, s.contributions,
                            static_cast<const float*>(routes.coefficients.data), s.routed_sum,
                            vector_limit, p.a16.dispatch(false));
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
        int first = vector_limit;
        for (const auto& bucket : p.a16.buckets) {
            const int columns = Gate ? bucket.gate_columns : bucket.down_columns;
            if (!columns) continue;
            if (x.ne[1] > first)
                project<I, Gate, Compressed, Fp16>(x, experts, routes, p, s, execution, columns,
                                                   first, bucket.maximum_load, bucket.tile);
            first = bucket.maximum_load;
        }
    };
    phase.template operator()<true>();
    if constexpr (Fp16) {
        if (x.ne[1] > vector_limit) {
            const int blocks = std::min(kTopK * x.ne[1], execution.multiprocessor_count() * 6);
            a16::normalize_product<I><<<blocks, 256, 0, execution.stream()>>>(
                s.a16_product, s.groups, static_cast<const int*>(routes.ids.data), vector_limit,
                p.a16.dispatch(true));
            CUDA_CHECK(cudaGetLastError());
        }
    }
    if (p.column_major_down)
        launch_column_jobs(s.groups, x.ne[1], p.job_columns(), p.preserve_balanced_down, execution);
    phase.template operator()<false>();
}

} // namespace

void launch_a16_experts(const Tensor& x, const CudaSwiGLUExperts& experts,
                        const ExpertRoutes& routes, const Plan& p, const Storage& s,
                        CudaExecutionView execution) {
    const bool csf = experts.bank.gate_up.layout == QuantLayout::Csf32;
    const auto run = [&]<int I>() {
        if (p.a16.fp16_product) {
            if (csf || p.clamp_limit != 10.F)
                throw std::logic_error("scaled FP16 product requires ordinary scales and clamp 10");
            launch<I, false, true>(x, experts, routes, p, s, execution);
        } else if (csf) {
            launch<I, true, false>(x, experts, routes, p, s, execution);
        } else {
            launch<I, false, false>(x, experts, routes, p, s, execution);
        }
    };
    if (p.intermediate == 1024)
        run.template operator()<1024>();
    else if (p.intermediate == 2048)
        run.template operator()<2048>();
    else
        throw std::logic_error("unregistered NVFP4 A16 expert width");
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4
