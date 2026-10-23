#include "ops/moe/kernels/cuda/block_scaled/grouping.cuh"
#include "ops/moe/kernels/cuda/block_scaled/h4096_i2048_e288_k8/nvfp4/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/h4096_i2048_e288_k8/fp8/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/h4096_i2048_e288_k8/output.cuh"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8 {
namespace {

__device__ __forceinline__ bool selected(ExpertSubsetKind kind, const int* mask, int e) {
    return kind == ExpertSubsetKind::All || mask[e] == (kind == ExpertSubsetKind::MaskOnes);
}

__global__ void count_assignments(const int* ids, ExpertSubsetKind kind, const int* mask,
                                  int* counts, int assignments) {
    __shared__ int histogram[kExperts];
    for (int e = threadIdx.x; e < kExperts; e += blockDim.x) { histogram[e] = 0; }
    __syncthreads();
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < assignments) {
        const int e = ids[a];
        if (selected(kind, mask, e)) { atomicAdd(histogram + e, 1); }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < kExperts; e += blockDim.x) {
        if (histogram[e]) { atomicAdd(counts + e, histogram[e]); }
    }
}

// Publish FP8 population facts once, before either contraction consumes groups.
__device__ void publish_density(Groups g, int count, int columns, int* maxima, int* actives) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int maximum = count;
#pragma unroll
    for (int delta = 16; delta; delta /= 2)
        maximum = max(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
    const int active = __popc(__ballot_sync(0xffffffff, count > 0));
    if (!lane) {
        maxima[warp]  = maximum;
        actives[warp] = active;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int result = 0, active_total = 0;
#pragma unroll
        for (int w = 0; w < (kExperts + 31) / 32; ++w) {
            result = max(result, maxima[w]);
            active_total += actives[w];
        }
        g.jobs[1] = (result + columns - 1) / columns;
        g.jobs[2] = active_total;
        g.jobs[3] = result;
    }
}

__global__ void scan_assignments(Groups g, int columns, bool density) {
    __shared__ int scan[kExperts], maxima[(kExperts + 31) / 32], actives[(kExperts + 31) / 32];
    const int e     = threadIdx.x;
    const int count = g.counts[e];
    publish_load_range<kExperts>(g.load_range, count);
    scan[e] = count;
    __syncthreads();
#pragma unroll
    for (int delta = 1; delta < kExperts; delta *= 2) {
        const int preceding = e >= delta ? scan[e - delta] : 0;
        __syncthreads();
        scan[e] += preceding;
        __syncthreads();
    }
    g.offsets[e] = e ? scan[e - 1] : 0;
    g.cursors[e] = g.offsets[e];
    if (e == kExperts - 1) { g.offsets[kExperts] = scan[e]; }
    __syncthreads();
    const int jobs = (count + columns - 1) / columns;
    if (density) publish_density(g, count, columns, maxima, actives);
    scan[e] = jobs;
    __syncthreads();
#pragma unroll
    for (int delta = 1; delta < kExperts; delta *= 2) {
        const int preceding = e >= delta ? scan[e - delta] : 0;
        __syncthreads();
        scan[e] += preceding;
        __syncthreads();
    }
    const int first = e ? scan[e - 1] : 0;
    for (int j = 0; j < jobs; ++j) {
        g.job_experts[first + j] = e;
        g.job_columns[first + j] = j * columns;
    }
    if (e == kExperts - 1) { *g.jobs = scan[e]; }
}

__global__ void index_assignments(const int* ids, ExpertSubsetKind kind, const int* mask, Groups g,
                                  int assignments) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= assignments) { return; }
    const int expert = ids[a];
    if (!selected(kind, mask, expert)) {
        if (g.inverse) { g.inverse[a] = -1; }
        return;
    }
    const int packed      = atomicAdd(g.cursors + expert, 1);
    g.assignments[packed] = a;
    if (g.inverse) { g.inverse[a] = packed; }
}

__global__ void group_small(const int* ids, ExpertSubsetKind kind, const int* mask, Groups g,
                            int assignments, int columns, bool density) {
    __shared__ int counts[kExperts], ranks[kTopK * 64], scan[kExperts],
        maxima[(kExperts + 31) / 32], actives[(kExperts + 31) / 32];
    const int tid = threadIdx.x;
    counts[tid]   = 0;
    __syncthreads();
    for (int a = tid; a < assignments; a += kExperts) {
        const int e = ids[a];
        ranks[a]    = selected(kind, mask, e) ? atomicAdd(counts + e, 1) : -1;
    }
    __syncthreads();
    const int count = counts[tid];
    publish_load_range<kExperts>(g.load_range, count);
    scan[tid] = count;
    __syncthreads();
#pragma unroll
    for (int delta = 1; delta < kExperts; delta *= 2) {
        const int preceding = tid >= delta ? scan[tid - delta] : 0;
        __syncthreads();
        scan[tid] += preceding;
        __syncthreads();
    }
    const int begin = tid ? scan[tid - 1] : 0;
    g.offsets[tid]  = begin;
    if (tid == kExperts - 1) { g.offsets[kExperts] = scan[tid]; }
    const int jobs = (count + columns - 1) / columns;
    if (density) publish_density(g, count, columns, maxima, actives);
    __syncthreads();
    scan[tid] = jobs;
    __syncthreads();
#pragma unroll
    for (int delta = 1; delta < kExperts; delta *= 2) {
        const int preceding = tid >= delta ? scan[tid - delta] : 0;
        __syncthreads();
        scan[tid] += preceding;
        __syncthreads();
    }
    const int first = tid ? scan[tid - 1] : 0;
    for (int j = 0; j < jobs; ++j) {
        g.job_experts[first + j] = tid;
        g.job_columns[first + j] = j * columns;
    }
    if (tid == kExperts - 1) { *g.jobs = scan[tid]; }
    for (int a = tid; a < assignments; a += kExperts) {
        const int packed = ranks[a] >= 0 ? g.offsets[ids[a]] + ranks[a] : -1;
        if (g.inverse) { g.inverse[a] = packed; }
        if (packed >= 0) { g.assignments[packed] = a; }
    }
}

template <class Output>
__global__ void reduce(Storage storage, const float* coefficients, Output output, int elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements) {
        if (storage.partial_output && !storage.active_tokens[i / kHidden]) return;
        output.store(i, output.initial(i) +
                            routed_value(i / kHidden, i % kHidden, storage, coefficients));
    }
}

// One CTA emits column layer j across all experts. Its first output
// index is sum_e min(j, jobs_e), followed by the prefix of experts with jobs_e > j.
// Only the job metadata changes; packed assignments and GU products keep their offsets.
template <bool PreserveBalanced>
__global__ void order_jobs(Groups g, int columns) {
    __shared__ int warp_offsets[9], warp_before[9], base;
    const int e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const int jobs = (g.counts[e] + columns - 1) / columns;
    const int j    = blockIdx.x;
    if constexpr (PreserveBalanced) {
        int maximum = jobs;
#pragma unroll
        for (int d = 16; d; d /= 2)
            maximum = max(maximum, __shfl_down_sync(0xffffffff, maximum, d));
        if (!lane) warp_before[warp] = maximum;
        __syncthreads();
        if (!warp) {
            int total = lane < 9 ? warp_before[lane] : 0;
#pragma unroll
            for (int d = 16; d; d /= 2) total = max(total, __shfl_down_sync(0xffffffff, total, d));
            if (!lane) base = total;
        }
        __syncthreads();
        // GU consumed expert-major jobs. Keeping adjacent pairs reuses their
        // weights; larger hotspot groups retain column ordering for scheduling.
        // The predicate is uniform and derived solely from Device group counts.
        if (base <= 2) return;
    }
    const int value = int(jobs > j);
    int prefix = value, before = min(j, jobs);
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
        const int preceding = __shfl_up_sync(0xffffffff, prefix, d);
        if (lane >= d) { prefix += preceding; }
    }
#pragma unroll
    for (int d = 16; d; d /= 2) { before += __shfl_down_sync(0xffffffff, before, d); }
    if (lane == 31) { warp_offsets[warp] = prefix; }
    if (!lane) { warp_before[warp] = before; }
    __syncthreads();
    if (!warp) {
        int total     = lane < 9 ? warp_offsets[lane] : 0;
        const int own = total;
#pragma unroll
        for (int d = 1; d < 32; d *= 2) {
            const int preceding = __shfl_up_sync(0xffffffff, total, d);
            if (lane >= d) { total += preceding; }
        }
        if (lane < 9) { warp_offsets[lane] = total - own; }
        int sum = lane < 9 ? warp_before[lane] : 0;
#pragma unroll
        for (int d = 16; d; d /= 2) { sum += __shfl_down_sync(0xffffffff, sum, d); }
        if (!lane) { base = sum; }
    }
    __syncthreads();
    if (value) {
        const int index      = base + warp_offsets[warp] + prefix - 1;
        g.job_experts[index] = e;
        g.job_columns[index] = j * columns;
    }
}

} // namespace

void launch_column_jobs(Groups groups, int tokens, int columns, bool preserve_balanced,
                        CudaExecutionView execution) {
    // Stream ordering is the lifetime boundary: GU must finish reading the
    // expert-major metadata before this launch overwrites it for down.
    if (tokens <= columns) { return; }
    if (!groups.counts) { throw std::logic_error("job ordering requires counted groups"); }
    if (preserve_balanced)
        order_jobs<true><<<(tokens + columns - 1) / columns, kExperts, 0, execution.stream()>>>(
            groups, columns);
    else
        order_jobs<false><<<(tokens + columns - 1) / columns, kExperts, 0, execution.stream()>>>(
            groups, columns);
    CUDA_CHECK(cudaGetLastError());
}

void launch(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
            ExpertSubset subset, MoeEpilogue mode, Tensor& destination, const Plan& plan,
            const Storage& storage, CudaExecutionView execution) {
    auto s = storage;
    if (s.partial_output && plan.atomic_output)
        s.routed_sum = static_cast<float*>(destination.data);
    const auto stream = execution.stream();
    if (plan.vector) {
        if (plan.format == QType::NVFP4)
            nvfp4::launch_vector(x, experts, routes, subset, mode, destination, s, plan, execution);
        else
            fp8::launch_vector(x, experts, routes, subset, mode, destination, s, plan, execution);
        return;
    }
    const int assignments = x.ne[1] * kTopK;
    const auto* ids       = static_cast<const int*>(routes.ids.data);
    const auto* mask      = static_cast<const int*>(subset.mask.data);
    if (plan.format == QType::NVFP4)
        nvfp4::launch_input(x, experts, routes, subset, s, execution, true);
    else
        fp8::launch_input(x, experts, routes, subset, s, execution, true);
    if (plan.single_block_grouping) {
        group_small<<<1, kExperts, 0, stream>>>(ids, subset.kind, mask, s.groups, assignments,
                                                plan.job_columns(),
                                                plan.format == QType::FP8_E4M3FN_BLOCK128X128_BF16);
        CUDA_CHECK(cudaGetLastError());
    } else {
        count_assignments<<<(assignments + 255) / 256, 256, 0, stream>>>(
            ids, subset.kind, mask, s.groups.counts, assignments);
        scan_assignments<<<1, kExperts, 0, stream>>>(
            s.groups, plan.job_columns(), plan.format == QType::FP8_E4M3FN_BLOCK128X128_BF16);
        index_assignments<<<(assignments + 255) / 256, 256, 0, stream>>>(ids, subset.kind, mask,
                                                                         s.groups, assignments);
        CUDA_CHECK(cudaGetLastError());
    }
    if (plan.format == QType::NVFP4)
        nvfp4::launch_experts(x, experts, routes, plan, s, execution);
    else
        fp8::launch_experts(x, experts, routes, plan, s, execution);
    // Atomic down already wrote each live FP32 column into the borrowed output.
    // Inactive columns remain unreadable under the private contribution view.
    if (s.partial_output && plan.atomic_output) return;
    with_output(destination, mode, [&](auto output) {
        const int elements = x.ne[1] * kHidden;
        reduce<<<(elements + 255) / 256, 256, 0, stream>>>(
            s, static_cast<const float*>(routes.coefficients.data), output, elements);
    });
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8
