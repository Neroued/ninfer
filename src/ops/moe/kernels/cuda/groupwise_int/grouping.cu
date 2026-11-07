#include "ops/moe/kernels/cuda/groupwise_int/launch.h"

namespace ninfer::ops::detail::moe_cuda::groupwise_int {
namespace {

template <class Shape, bool Masked>
__global__ void count_assignments(const int* ids, const int* mask, int member, int* ranks,
                                  int* counts_out, int assignments) {
    constexpr int Tile = 8 * Shape::top_k;
    __shared__ int counts[Shape::experts];
    const int tid = threadIdx.x;
    for (int e = tid; e < Shape::experts; e += blockDim.x) { counts[e] = 0; }
    __syncthreads();
    for (int item = tid; item < Tile; item += blockDim.x) {
        const int assignment = blockIdx.x * Tile + item;
        if (assignment < assignments) {
            const int expert = ids[assignment];
            bool selected    = true;
            if constexpr (Masked) { selected = mask[expert] == member; }
            ranks[assignment] = selected ? atomicAdd(counts + expert, 1) : -1;
        }
    }
    __syncthreads();
    for (int e = tid; e < Shape::experts; e += blockDim.x) {
        counts_out[std::int64_t(blockIdx.x) * Shape::experts + e] = counts[e];
    }
}

// One block of Shape::experts * kScanSplits threads: the tile axis of the per-tile counters is cut
// into kScanSplits contiguous slices that are summed (and later turned into bases) concurrently,
// so the scan latency does not grow with the token count.
inline constexpr int kScanSplits = 4;

template <class Shape>
__global__ __launch_bounds__(Shape::experts* kScanSplits) void scan_assignments(
    GroupedStorage out, int tiles, GroupedPlan plan) {
    __shared__ int scan[Shape::experts];
    __shared__ int slice_sum[kScanSplits][Shape::experts];
    const int e        = threadIdx.x % Shape::experts;
    const int slice    = threadIdx.x / Shape::experts;
    const int per      = (tiles + kScanSplits - 1) / kScanSplits;
    const int tile_end = min(tiles, (slice + 1) * per);
    int part           = 0;
#pragma unroll 8
    for (int tile = slice * per; tile < tile_end; ++tile) {
        part += out.tile_counts[tile * Shape::experts + e];
    }
    slice_sum[slice][e] = part;
    __syncthreads();
    int count = 0, before = 0;
#pragma unroll
    for (int s = 0; s < kScanSplits; ++s) {
        count += slice_sum[s][e];
        before += s < slice ? slice_sum[s][e] : 0;
    }
    const bool leader = slice == 0;
    if (leader) { scan[e] = count; }
    __syncthreads();
#pragma unroll
    for (int d = 1; d < Shape::experts; d <<= 1) {
        const int value = leader && e >= d ? scan[e - d] : 0;
        __syncthreads();
        if (leader) { scan[e] += value; }
        __syncthreads();
    }
    const int begin       = e ? scan[e - 1] : 0;
    const int assignments = scan[Shape::experts - 1];
    if (leader) {
        out.offsets[e] = begin;
        if (e == Shape::experts - 1) { out.offsets[Shape::experts] = assignments; }
    }
    int cursor = begin + before;
#pragma unroll 8
    for (int tile = slice * per; tile < tile_end; ++tile) {
        const int index       = tile * Shape::experts + e;
        const int tile_count  = out.tile_counts[index];
        out.tile_bases[index] = cursor;
        cursor += tile_count;
    }
    __syncthreads();
    const int jobs = (count + plan.columns - 1) / plan.columns;
    if (leader) { scan[e] = jobs; }
    __syncthreads();
#pragma unroll
    for (int d = 1; d < Shape::experts; d <<= 1) {
        const int value = leader && e >= d ? scan[e - d] : 0;
        __syncthreads();
        if (leader) { scan[e] += value; }
        __syncthreads();
    }
    const int active = __syncthreads_count(leader && count > 0);
    const int first  = e ? scan[e - 1] : 0;
    if (leader) {
        for (int job = 0; job < jobs; ++job) {
            out.job_experts[first + job] = e;
            out.job_columns[first + job] = job * plan.columns;
        }
        if (e == Shape::experts - 1) {
            const int total = scan[e];
            *out.info       = {total,
                               int(plan.adaptive &&
                             total * plan.direct_denominator > plan.direct_numerator * assignments),
                               int(total * plan.spread_denominator < plan.spread_numerator * active)};
        }
    }
}

template <class Shape>
__global__ void index_assignments(const int* ids, GroupedStorage out, int assignments) {
    if (out.info->direct) { return; }
    const int assignment = blockIdx.x * blockDim.x + threadIdx.x;
    if (assignment >= assignments) { return; }
    const int local = out.local_ranks[assignment];
    if (local < 0) {
        out.inverse[assignment] = -1;
        return;
    }
    const int expert        = ids[assignment];
    const int tile          = assignment / (8 * Shape::top_k);
    const int packed        = out.tile_bases[tile * Shape::experts + expert] + local;
    out.inverse[assignment] = packed;
    out.tokens[packed]      = assignment / Shape::top_k;
}
} // namespace

void launch_group_count(const ExpertRoutes& routes, ExpertSubset subset,
                        const GroupedStorage& storage, CudaExecutionView execution) {
    using S          = SwiGLUH2048I512E256K8;
    const int count  = routes.ids.ne[1] * S::top_k;
    const int tiles  = (count + 8 * S::top_k - 1) / (8 * S::top_k);
    const auto* ids  = static_cast<const int*>(routes.ids.data);
    const auto* mask = static_cast<const int*>(subset.mask.data);
    if (subset.kind == ExpertSubsetKind::All) {
        count_assignments<S, false><<<tiles, S::experts, 0, execution.stream()>>>(
            ids, nullptr, 0, storage.local_ranks, storage.tile_counts, count);
    } else {
        count_assignments<S, true><<<tiles, S::experts, 0, execution.stream()>>>(
            ids, mask, subset.kind == ExpertSubsetKind::MaskOnes ? 1 : 0, storage.local_ranks,
            storage.tile_counts, count);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_group_scan(const ExpertRoutes& routes, const GroupedPlan& plan,
                       const GroupedStorage& storage, CudaExecutionView execution) {
    using S = SwiGLUH2048I512E256K8;
    scan_assignments<S><<<1, S::experts * kScanSplits, 0, execution.stream()>>>(
        storage, (routes.ids.ne[1] + 7) / 8, plan);
    CUDA_CHECK(cudaGetLastError());
}

void launch_group_index(const ExpertRoutes& routes, const GroupedStorage& storage,
                        CudaExecutionView execution) {
    using S         = SwiGLUH2048I512E256K8;
    const int count = S::top_k * routes.ids.ne[1];
    index_assignments<S><<<(count + 255) / 256, 256, 0, execution.stream()>>>(
        static_cast<const int*>(routes.ids.data), storage, count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::moe_cuda::groupwise_int
