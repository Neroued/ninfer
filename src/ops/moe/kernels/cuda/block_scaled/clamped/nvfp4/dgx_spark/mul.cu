#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/mul.h"
#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/compute.cuh"
#include "ops/moe/kernels/cuda/block_scaled/pool_down.cuh"

#include <algorithm>
#include <bit>

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4 {
namespace {
ActivationScale calibration(const ActivationCalibrationView& view) {
    if (view.operation != ActivationScaleOperation::Multiply || view.empty())
        throw std::invalid_argument("native NVFP4 requires activation multipliers");
    if (view.values().size() == 1) return {nullptr, view.values().front()};
    return {static_cast<const float*>(view.device_values.data), 1};
}

template <bool Quantize>
__global__ void prepare_input(const __nv_bfloat16* x, std::uint8_t* codes, std::uint8_t* scales,
                              ActivationScale input_scale, const int* ids, int tokens,
                              ExpertSubsetKind kind, const int* mask, int* counts,
                              float* routed_sum, int* active_tokens) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (counts && index < kExperts) counts[index] = 0;
    const int assignment = blockIdx.x;
    const int token      = Quantize ? assignment / kTopK : assignment;
    if (token >= tokens) return;
    if (!Quantize || assignment % kTopK == 0) {
        if (threadIdx.x < 32) {
            bool live = false;
            if (threadIdx.x < kTopK)
                live = includes(kind, mask, ids[std::int64_t(token) * kTopK + threadIdx.x]);
            const auto bits = __ballot_sync(0xffffffff, live);
            if (!threadIdx.x && active_tokens) active_tokens[token] = bits != 0;
        }
        if (routed_sum)
            for (int row = threadIdx.x; row < kHidden; row += blockDim.x)
                routed_sum[std::int64_t(token) * kHidden + row] = 0;
    }
    if constexpr (Quantize) {
        const int expert = ids[assignment];
        if (!includes(kind, mask, expert)) return;
        const int group    = threadIdx.x;
        const auto encoded = common::quantize_nvfp4_mul_k16(
            x + std::int64_t(token) * kHidden + group * 16, input_scale.get(expert));
        *reinterpret_cast<uint2*>(codes + std::int64_t(assignment) * (kHidden / 2) + group * 8) =
            make_uint2(encoded.codes_lo, encoded.codes_hi);
        scales[std::int64_t(assignment) * (kHidden / 16) + group] = encoded.scale;
    }
}

template <int I, bool Gate, int Columns>
void launch_columns(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                    const Plan& plan, const Storage& storage, CudaExecutionView execution,
                    int minimum_load, int maximum_load) {
    using S               = Schedule<Columns, Gate, 2, 128, 0, I, true>;
    constexpr auto kernel = grouped_mma<Columns, Gate, true, 2, 128, 0, true, false, I, true>;
    const int assignments = kTopK * x.ne[1];
    const int covered     = std::min({plan.job_columns(), maximum_load, x.ne[1]});
    const int subtiles    = std::bit_ceil(unsigned((covered + Columns - 1) / Columns));
    const int groups      = std::min(kExperts, assignments / (minimum_load + 1));
    const int jobs        = (assignments / plan.job_columns() + groups) * subtiles;
    constexpr int rows    = Gate ? I : kHidden;
    const int blocks      = std::min(jobs * (rows / S::rows),
                                     execution.multiprocessor_count() * plan.queued_ctas_per_sm);
    constexpr int bytes   = sizeof(TileStorage<S>) > 48 * 1024 ? sizeof(TileStorage<S>) : 0;
    if constexpr (bytes) {
        static const auto status =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        CUDA_CHECK(status);
    }
    kernel<<<blocks, S::threads, bytes, execution.stream()>>>(
        Gate ? storage.input_codes : storage.product_codes,
        static_cast<const std::uint8_t*>(Gate ? storage.input_scales : storage.product_scales),
        Gate ? experts.bank.gate_up : experts.bank.down,
        calibration(Gate ? experts.activation_calibration.gate_up
                         : experts.activation_calibration.down),
        calibration(experts.activation_calibration.down), storage.groups, storage.product_codes,
        static_cast<std::uint8_t*>(storage.product_scales), storage.contributions,
        static_cast<const float*>(routes.coefficients.data), storage.routed_sum,
        std::countr_zero(unsigned(subtiles)), plan.clamp_limit, 0, 0, minimum_load, maximum_load,
        ExpertLoadRegime::All, 0, 0, 0, x.ne[1], plan.dense_down_columns != 0,
        plan.pool_expert_limit, plan.pool_min_experts);
    CUDA_CHECK(cudaGetLastError());
}

template <int I>
void launch_bank(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                 const Plan& plan, const Storage& storage, CudaExecutionView execution) {
    const auto phase = [&]<bool Gate>() {
        const auto run = [&](const Plan& selected, int first, int last) {
            if (x.ne[1] <= first) return;
            const int columns = Gate ? selected.gate_columns : selected.down_columns;
            switch (columns) {
            case 8:
                return launch_columns<I, Gate, 8>(x, experts, routes, selected, storage, execution,
                                                  first, last);
            case 16:
                return launch_columns<I, Gate, 16>(x, experts, routes, selected, storage, execution,
                                                   first, last);
            case 32:
                return launch_columns<I, Gate, 32>(x, experts, routes, selected, storage, execution,
                                                   first, last);
            case 64:
                return launch_columns<I, Gate, 64>(x, experts, routes, selected, storage, execution,
                                                   first, last);
            case 128:
                return launch_columns<I, Gate, 128>(x, experts, routes, selected, storage,
                                                    execution, first, last);
            default:
                throw std::logic_error("invalid native NVFP4 column tile");
            }
        };
        if (!plan.dense.gate_columns) {
            run(plan, 0, INT_MAX);
            return;
        }
        run(plan, 0, plan.small_expert_limit);
        int dense_begin = plan.small_expert_limit;
        if (plan.medium.gate_columns) {
            auto medium               = plan;
            medium.gate_columns       = plan.medium.gate_columns;
            medium.down_columns       = plan.medium.down_columns;
            medium.queued_ctas_per_sm = plan.medium.queued_ctas_per_sm;
            run(medium, dense_begin, plan.medium.max_load);
            dense_begin = plan.medium.max_load;
        }
        auto dense               = plan;
        dense.gate_columns       = plan.dense.gate_columns;
        dense.down_columns       = plan.dense.down_columns;
        dense.queued_ctas_per_sm = plan.dense.queued_ctas_per_sm;
        run(dense, dense_begin, INT_MAX);
    };
    phase.template operator()<true>();
    if (plan.dense_down_columns) {
        const auto run = [&]<int C>() {
            const int blocks = std::min((kHidden / 128) * ((x.ne[1] + C - 1) / C),
                                        execution.multiprocessor_count() * 3);
            dense_token_down<C, I, true>
                <<<blocks, Schedule<C, false>::threads, 0, execution.stream()>>>(
                    storage.product_codes, static_cast<const std::uint8_t*>(storage.product_scales),
                    experts.bank.down, calibration(experts.activation_calibration.down),
                    storage.groups, static_cast<const int*>(routes.ids.data),
                    static_cast<const float*>(routes.coefficients.data), storage.routed_sum,
                    x.ne[1], plan.pool_expert_limit, plan.pool_min_experts);
            CUDA_CHECK(cudaGetLastError());
        };
        switch (plan.dense_down_columns) {
        case 16:
            run.template operator()<16>();
            break;
        case 32:
            run.template operator()<32>();
            break;
        case 64:
            run.template operator()<64>();
            break;
        case 128:
            run.template operator()<128>();
            break;
        default:
            throw std::logic_error("invalid native dense-down column tile");
        }
    }
    if (plan.pool_expert_limit) {
        if (plan.pool_columns != 128)
            throw std::logic_error("invalid native expert pool column tile");
        launch_pool_down<kHidden, kExperts, kTopK, 128>(
            PoolContraction<128, I, true>{
                storage.product_codes, static_cast<const std::uint8_t*>(storage.product_scales),
                experts.bank.down, calibration(experts.activation_calibration.down)},
            storage.groups, static_cast<const int*>(routes.ids.data),
            static_cast<const float*>(routes.coefficients.data), storage.routed_sum, x.ne[1],
            plan.pool_min_experts, plan.pool_expert_limit, plan.pool_row_window,
            plan.pool_queued_ctas_per_sm, execution);
    }
    if (plan.column_major_down)
        launch_column_jobs(storage.groups, x.ne[1], plan.job_columns(), plan.preserve_balanced_down,
                           execution);
    phase.template operator()<false>();
}
} // namespace

void launch_mul_input(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                      ExpertSubset subset, const Plan& plan, const Storage& storage,
                      CudaExecutionView execution) {
    const bool quantize = plan.arithmetic == Nvfp4Arithmetic::MultiplyA4;
    const auto launch   = [&]<bool Quantize>() {
        const int count = x.ne[1] * (Quantize ? kTopK : 1);
        prepare_input<Quantize>
            <<<std::max(count, (kExperts + 255) / 256), 256, 0, execution.stream()>>>(
                static_cast<const __nv_bfloat16*>(x.data), storage.input_codes,
                static_cast<std::uint8_t*>(storage.input_scales),
                Quantize ? calibration(experts.activation_calibration.gate_up) : ActivationScale{},
                static_cast<const int*>(routes.ids.data), x.ne[1], subset.kind,
                static_cast<const int*>(subset.mask.data), storage.groups.counts,
                storage.routed_sum, storage.active_tokens);
    };
    if (quantize)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
    CUDA_CHECK(cudaGetLastError());
}

void launch_mul_experts(const Tensor& x, const CudaSwiGLUExperts& experts,
                        const ExpertRoutes& routes, const Plan& plan, const Storage& storage,
                        CudaExecutionView execution) {
    if (plan.intermediate == 1024)
        launch_bank<1024>(x, experts, routes, plan, storage, execution);
    else
        launch_bank<2048>(x, experts, routes, plan, storage, execution);
}
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4
