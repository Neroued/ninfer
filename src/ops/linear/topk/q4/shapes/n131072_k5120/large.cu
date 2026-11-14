#include "ops/linear/topk/dispatch.h"
#include "ops/linear/topk/epilogue.cuh"
#include "ops/linear/kernels/q4/q4_a16_tma_prefetch.cuh"

namespace ninfer::ops::detail {
namespace {
using Schedule = Q4A16TmaPrefetchSchedule<128, 128, 64, 32, 1, 5120, 1>;

void run(const Tensor& hidden, const Weight& head, const Tensor* map,
         const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    const TopKHeadOutput out{static_cast<std::uint64_t*>(workspace.partial_keys.data),
                             map ? static_cast<const int*>(map->data) : nullptr,
                             workspace.producer_groups, workspace.row_begin, kLinearTopKMappedRows};
    launch_q4_a16_tma_prefetch<Schedule>(q4_linear_operands(hidden, head), out,
                                         TopKPanelEpilogue<64, true>{}, stream);
}
} // namespace

LinearTopKPlan select_topk_q4_n131072_k5120_large() { return {run, Schedule::kBlockRows, 8192}; }
} // namespace ninfer::ops::detail
