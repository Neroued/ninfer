#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/mul.h"
#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/compute.cuh"
#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/grouped_tma.cuh"
#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/grouped_scale_cache.cuh"
#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/short_expert.cuh"
#include "ops/moe/kernels/cuda/block_scaled/pool_down.cuh"
#include "ops/moe/kernels/cuda/block_scaled/grouping.cuh"

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

// Small native batches publish grouping in the input launch. Counts and job
// prefixes fit in separate 16-bit fields for at most 256*8 assignments.
__device__ void group_decode(const int* ids, ExpertSubsetKind kind, const int* mask, Groups g,
                             int tokens, int columns, bool ordered) {
    constexpr int warps = kExperts / 32;
    __shared__ int counts[kExperts], ranks[kTopK * 256];
    __shared__ unsigned prefixes[warps];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    counts[tid] = 0;
    __syncthreads();
    for (int a = tid; a < tokens * kTopK; a += kExperts) {
        const int expert = ids[a];
        ranks[a]         = includes(kind, mask, expert) ? atomicAdd(counts + expert, 1) : -1;
    }
    __syncthreads();
    const int count = counts[tid], jobs = (count + columns - 1) / columns;
    publish_load_range<kExperts>(g.load_range, count);
    unsigned prefix = unsigned(count) | (unsigned(jobs) << 16);
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
        const unsigned preceding = __shfl_up_sync(0xffffffff, prefix, d);
        if (lane >= d) prefix += preceding;
    }
    if (lane == 31) prefixes[warp] = prefix;
    __syncthreads();
    if (warp == 0) {
        unsigned total = lane < warps ? prefixes[lane] : 0;
#pragma unroll
        for (int d = 1; d < 32; d *= 2) {
            const unsigned preceding = __shfl_up_sync(0xffffffff, total, d);
            if (lane >= d) total += preceding;
        }
        if (lane < warps) prefixes[lane] = total;
    }
    __syncthreads();
    prefix += warp ? prefixes[warp - 1] : 0;
    const int begin = int(prefix & 65535) - count, first = int(prefix >> 16) - jobs;
    g.offsets[tid] = begin;
    for (int j = 0; j < jobs; ++j) {
        g.job_experts[first + j] = tid;
        g.job_columns[first + j] = j * columns;
    }
    if (tid == kExperts - 1) {
        g.offsets[kExperts] = prefix & 65535;
        *g.jobs             = prefix >> 16;
    }
    __syncthreads();
    const bool token_order =
        fully_covered_groups<kExperts, kTopK>(g, tokens * kTopK, false, ordered);
    for (int a = tid; a < tokens * kTopK; a += kExperts) {
        const int packed =
            ranks[a] >= 0 ? g.offsets[ids[a]] + (token_order ? a / kTopK : ranks[a]) : -1;
        if (g.inverse) g.inverse[a] = packed;
        if (packed >= 0) g.assignments[packed] = a;
    }
}

template <bool Quantize, bool Group, bool SkipInactive>
__global__ void prepare_input(const __nv_bfloat16* x, std::uint8_t* codes, std::uint8_t* scales,
                              ActivationScale input_scale, const int* ids, int tokens,
                              ExpertSubsetKind kind, const int* mask, int* counts,
                              float* routed_sum, int* active_tokens, Groups groups, int columns,
                              bool ordered) {
    if constexpr (Group) {
        if (blockIdx.x == 0) group_decode(ids, kind, mask, groups, tokens, columns, ordered);
    }
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (counts && index < kExperts) counts[index] = 0;
    const int assignment = blockIdx.x;
    const int token      = Quantize ? assignment / kTopK : assignment;
    if (token >= tokens) return;
    if (!Quantize || assignment % kTopK == 0) {
        bool clear = true;
        if (SkipInactive || threadIdx.x < 32) {
            bool live      = false;
            const int lane = threadIdx.x % 32;
            if (lane < kTopK) live = includes(kind, mask, ids[std::int64_t(token) * kTopK + lane]);
            const auto bits = __ballot_sync(0xffffffff, live);
            if (!threadIdx.x && active_tokens) active_tokens[token] = bits != 0;
            if constexpr (SkipInactive) clear = bits != 0 || !active_tokens;
        }
        if (routed_sum && clear)
            for (int row = threadIdx.x; row < kHidden; row += blockDim.x)
                routed_sum[std::int64_t(token) * kHidden + row] = 0;
    }
    if constexpr (Quantize) {
        if (threadIdx.x >= kHidden / 16) return;
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

template <bool Index, bool Persistent>
__global__ void
quantize_grouped_input(const __nv_bfloat16* x, const int* ids, ExpertSubsetKind kind,
                       const int* mask, Groups groups, int tokens, bool ordered,
                       ActivationScale calibration, std::uint8_t* codes, std::uint8_t* scales,
                       std::uint8_t* product, std::uint8_t* product_scales, int intermediate) {
    const int tid = threadIdx.x;
    if constexpr (Persistent) {
        if (!*groups.jobs) {
            if constexpr (Index) {
                if (!tid && groups.inverse)
                    for (int a = blockIdx.x; a < tokens * kTopK; a += gridDim.x)
                        groups.inverse[a] = -1;
            }
            return;
        }
    }
    for (int task = blockIdx.x; !Persistent || task < tokens * kTopK; task += gridDim.x) {
        // TMA fetches complete column tiles. Only the final partial tile can reach
        // unowned rows; initialize its in-bounds tail without another allocation or
        // launch. The map itself supplies zero for rows beyond the allocation.
        const int live = groups.offsets[kExperts], tail = min(255, tokens * kTopK - live);
        if (task < tail) {
            const int row = live + task;
            *reinterpret_cast<uint2*>(codes + std::int64_t(row) * (kHidden / 2) + tid * 8) =
                make_uint2(0, 0);
            scales[std::int64_t(row) * (kHidden / 16) + tid] = 0;
            if (tid < intermediate / 16) {
                *reinterpret_cast<uint2*>(product + std::int64_t(row) * (intermediate / 2) +
                                          tid * 8)                            = make_uint2(0, 0);
                product_scales[std::int64_t(row) * (intermediate / 16) + tid] = 0;
            }
        }
        int assignment;
        __shared__ int packed;
        if constexpr (Index) {
            assignment       = task;
            const int expert = ids[assignment];
            if (!includes(kind, mask, expert)) {
                if (!tid && groups.inverse) groups.inverse[assignment] = -1;
                if constexpr (Persistent)
                    continue;
                else
                    return;
            }
            if (!tid) {
                packed =
                    fully_covered_groups<kExperts, kTopK>(groups, tokens * kTopK, false, ordered)
                        ? groups.offsets[expert] + assignment / kTopK
                        : atomicAdd(groups.cursors + expert, 1);
                groups.assignments[packed] = assignment;
                if (groups.inverse) groups.inverse[assignment] = packed;
            }
            __syncthreads();
        } else {
            if (task >= groups.offsets[kExperts]) break;
            assignment = groups.assignments[task];
            if (!tid) packed = task;
            __syncthreads();
        }
        const int expert = ids[assignment], token = assignment / kTopK;
        const auto encoded = common::quantize_nvfp4_mul_k16(
            x + std::int64_t(token) * kHidden + tid * 16, calibration.get(expert));
        *reinterpret_cast<uint2*>(codes + std::int64_t(packed) * (kHidden / 2) + tid * 8) =
            make_uint2(encoded.codes_lo, encoded.codes_hi);
        scales[std::int64_t(packed) * (kHidden / 16) + tid] = encoded.scale;
        if constexpr (Persistent)
            __syncthreads();
        else
            break;
    }
}

template <int I, bool Gate, int Columns, bool Csf>
void launch_columns(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                    const Plan& plan, const Storage& storage, CudaExecutionView execution,
                    int minimum_load, int maximum_load) {
    using S               = Schedule<Columns, Gate, 2, 128, 0, I, true, Csf>;
    constexpr auto kernel = grouped_mma<Columns, Gate, true, 2, 128, 0, true, false, I, true, Csf>;
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
        plan.pool_expert_limit, plan.pool_min_experts, plan.native_a4_grouped_input);
    CUDA_CHECK(cudaGetLastError());
}

template <int I, bool Csf>
void launch_bank(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                 const Plan& plan, const Storage& storage, CudaExecutionView execution) {
    const auto phase = [&]<bool Gate>() {
        const int vector_limit = plan.native_a4_vector_max_load;
        if constexpr (Csf) {
            if (vector_limit) {
                if (vector_limit != 2) throw std::logic_error("invalid native A4 vector load");
                const auto short_kernel = [&]<int Columns>() {
                    using V = short_expert::Shape<I, Gate, Columns>;
                    const int blocks =
                        std::min(kTopK * x.ne[1] * ((Gate ? I : kHidden) / V::features),
                                 execution.multiprocessor_count() *
                                     (I == 2048 ? plan.queued_ctas_per_sm : 8));
                    short_expert::contract<I, Gate, Columns>
                        <<<blocks, V::threads, 0, execution.stream()>>>(
                            Gate ? storage.input_codes : storage.product_codes,
                            static_cast<const std::uint8_t*>(Gate ? storage.input_scales
                                                                  : storage.product_scales),
                            Gate ? experts.bank.gate_up : experts.bank.down,
                            calibration(Gate ? experts.activation_calibration.gate_up
                                             : experts.activation_calibration.down),
                            calibration(experts.activation_calibration.down), storage.groups,
                            storage.product_codes,
                            static_cast<std::uint8_t*>(storage.product_scales),
                            static_cast<const float*>(routes.coefficients.data), storage.routed_sum,
                            plan.clamp_limit, plan.pool_min_experts, plan.pool_expert_limit,
                            plan.native_a4_grouped_input);
                    CUDA_CHECK(cudaGetLastError());
                };
                short_kernel.template operator()<2>();
            }
        }
        if (x.ne[1] <= vector_limit) return;
        {
            const auto bank = Gate ? experts.bank.gate_up : experts.bank.down;
            constexpr int k = Gate ? kHidden : I;
            if (plan.native_a4_grouped_input && !bank.offsets && bank.code_stride % (k / 2) == 0) {
                const auto run_tma = [&]<int C>(int first, int last) {
                    if (x.ne[1] <= first) return;
                    if constexpr (I == 2048 && Csf && !Gate && C == 128) {
                        // A single column tile uses two resident CTAs with a
                        // full-K scale decoder. Longer groups favor the staged tile.
                        grouped_scale_cache::launch<I, Gate, C, Csf>(
                            x, experts, routes, plan, storage, execution,
                            std::max(first, vector_limit), 128);
                        if (x.ne[1] > 128)
                            grouped_tma::launch<I, Gate, C, Csf>(x, experts, routes, plan, storage,
                                                                 execution, 128, last);
                    } else if constexpr (Gate && C == 256) {
                        grouped_scale_cache::launch<I, Gate, C, Csf>(
                            x, experts, routes, plan, storage, execution,
                            std::max(first, vector_limit), last);
                    } else {
                        grouped_tma::launch<I, Gate, C, Csf>(x, experts, routes, plan, storage,
                                                             execution,
                                                             std::max(first, vector_limit), last);
                    }
                };
                run_tma.template operator()<16>(0, 16);
                run_tma.template operator()<64>(16, 64);
                run_tma.template operator()<128>(64, 512);
                run_tma.template operator()<256>(512, INT_MAX);
                return;
            }
        }
        const auto run = [&](const Plan& selected, int first, int last) {
            if (x.ne[1] <= first) return;
            const int columns = Gate ? selected.gate_columns : selected.down_columns;
            switch (columns) {
            case 8:
                return launch_columns<I, Gate, 8, Csf>(x, experts, routes, selected, storage,
                                                       execution, std::max(vector_limit, first),
                                                       last);
            case 16:
                return launch_columns<I, Gate, 16, Csf>(x, experts, routes, selected, storage,
                                                        execution, std::max(vector_limit, first),
                                                        last);
            case 32:
                return launch_columns<I, Gate, 32, Csf>(x, experts, routes, selected, storage,
                                                        execution, std::max(vector_limit, first),
                                                        last);
            case 64:
                return launch_columns<I, Gate, 64, Csf>(x, experts, routes, selected, storage,
                                                        execution, std::max(vector_limit, first),
                                                        last);
            case 128:
                return launch_columns<I, Gate, 128, Csf>(x, experts, routes, selected, storage,
                                                         execution, std::max(vector_limit, first),
                                                         last);
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
            dense_token_down<C, I, true, Csf>
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
            PoolContraction<128, I, true, Csf>{
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
    const bool quantize =
        plan.arithmetic == Nvfp4Arithmetic::MultiplyA4 && !plan.native_a4_grouped_input;
    const auto launch = [&]<bool Quantize, bool SkipInactive>() {
        const auto run = [&]<bool Group>() {
            constexpr int threads = Group ? kExperts : 256;
            const int count       = x.ne[1] * (Quantize ? kTopK : 1);
            prepare_input<Quantize, Group, SkipInactive>
                <<<std::max(count, (kExperts + threads - 1) / threads), threads, 0,
                   execution.stream()>>>(
                    static_cast<const __nv_bfloat16*>(x.data), storage.input_codes,
                    static_cast<std::uint8_t*>(storage.input_scales),
                    Quantize ? calibration(experts.activation_calibration.gate_up)
                             : ActivationScale{},
                    static_cast<const int*>(routes.ids.data), x.ne[1], subset.kind,
                    static_cast<const int*>(subset.mask.data), storage.groups.counts,
                    storage.routed_sum, storage.active_tokens, storage.groups, plan.job_columns(),
                    plan.token_ordered_groups);
        };
        if (plan.single_block_grouping)
            run.template operator()<true>();
        else
            run.template operator()<false>();
    };
    const auto select = [&]<bool SkipInactive>() {
        if (quantize)
            launch.template operator()<true, SkipInactive>();
        else
            launch.template operator()<false, SkipInactive>();
    };
    if (plan.intermediate == 2048)
        select.template operator()<true>();
    else
        select.template operator()<false>();
    CUDA_CHECK(cudaGetLastError());
}

void launch_mul_grouped_input(const Tensor& x, const CudaSwiGLUExperts& experts,
                              const ExpertRoutes& routes, ExpertSubset subset, const Plan& plan,
                              const Storage& storage, CudaExecutionView execution) {
    const auto run = [&]<bool Index, bool Persistent>() {
        const int blocks = Persistent
                               ? std::min(kTopK * x.ne[1], execution.multiprocessor_count() * 8)
                               : kTopK * x.ne[1];
        quantize_grouped_input<Index, Persistent><<<blocks, kHidden / 16, 0, execution.stream()>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const int*>(routes.ids.data),
            subset.kind, static_cast<const int*>(subset.mask.data), storage.groups, x.ne[1],
            plan.token_ordered_groups, calibration(experts.activation_calibration.gate_up),
            storage.input_codes, static_cast<std::uint8_t*>(storage.input_scales),
            storage.product_codes, static_cast<std::uint8_t*>(storage.product_scales),
            plan.intermediate);
    };
    const auto select = [&]<bool Persistent>() {
        if (plan.single_block_grouping)
            run.template operator()<false, Persistent>();
        else
            run.template operator()<true, Persistent>();
    };
    if (plan.intermediate == 2048)
        select.template operator()<true>();
    else
        select.template operator()<false>();
    CUDA_CHECK(cudaGetLastError());
}

void launch_mul_experts(const Tensor& x, const CudaSwiGLUExperts& experts,
                        const ExpertRoutes& routes, const Plan& plan, const Storage& storage,
                        CudaExecutionView execution) {
    const auto launch = [&]<bool Csf>() {
        if (plan.intermediate == 1024)
            launch_bank<1024, Csf>(x, experts, routes, plan, storage, execution);
        else
            launch_bank<2048, Csf>(x, experts, routes, plan, storage, execution);
    };
    if (experts.bank.gate_up.layout == QuantLayout::Csf32)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4
