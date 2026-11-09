#pragma once

#include "ops/moe_grouped/launch.h"
#include <cub/block/block_scan.cuh>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

struct GroupedPrefix {
    int assignments;
    int jobs;
};

struct AddGroupedPrefix {
    __device__ GroupedPrefix operator()(GroupedPrefix a, GroupedPrefix b) const {
        return {a.assignments + b.assignments, a.jobs + b.jobs};
    }
};

// Block-local hierarchical scan (Merrill/Garland, 2016, section 3.1).
template <int Experts>
__global__ void __launch_bounds__(Experts)
    grouped_scan_kernel(const std::int32_t* __restrict__ counts,
                        const std::int32_t* __restrict__ table, std::int32_t* __restrict__ offsets,
                        std::int32_t* __restrict__ job_experts,
                        std::int32_t* __restrict__ job_columns,
                        std::int32_t* __restrict__ job_slots,
                        std::int32_t* __restrict__ job_count) {
    static_assert(Experts > 0 && Experts <= 1024);
    using Scan = cub::BlockScan<GroupedPrefix, Experts, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename Scan::TempStorage storage;
    const int expert = static_cast<int>(threadIdx.x);
    const int count  = counts[expert];
    const int jobs   = (count + kMoeGroupedJobColumns - 1) / kMoeGroupedJobColumns;
    const int slot   = jobs > 0 ? table[expert] : -1;
    GroupedPrefix prefix;
    Scan(storage).ExclusiveScan(GroupedPrefix{count, jobs}, prefix, GroupedPrefix{0, 0},
                                AddGroupedPrefix{});
    offsets[expert] = prefix.assignments;
    for (int job = 0; job < jobs; ++job) {
        job_experts[prefix.jobs + job] = expert;
        job_columns[prefix.jobs + job] = job * kMoeGroupedJobColumns;
        job_slots[prefix.jobs + job]   = slot;
    }
    if (expert == Experts - 1) {
        offsets[Experts] = prefix.assignments + count;
        *job_count       = prefix.jobs + jobs;
    }
}

} // namespace
} // namespace ninfer::ops::detail
