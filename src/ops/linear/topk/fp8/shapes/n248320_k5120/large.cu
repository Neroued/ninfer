#include "ops/linear/topk/dispatch.h"
#include "ops/linear/topk/epilogue.cuh"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Schedule = Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, 5120>;

void run(const Tensor& hidden, const Weight& head, const Tensor* map,
         const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    const TopKHeadOutput out{static_cast<std::uint64_t*>(workspace.partial_keys.data),
                             map ? static_cast<const int*>(map->data) : nullptr,
                             workspace.producer_groups, workspace.row_begin,
                             kLinearTopKFullValidRows};
    launch_fp8_a16_mma<Schedule>(fp8_a16_operands(hidden, head), out, TopKPanelEpilogue<64, true>{},
                                 stream);
}
} // namespace

LinearTopKPlan select_topk_fp8_n248320_k5120_large() { return {run, Schedule::kBlockRows, 8192}; }
} // namespace ninfer::ops::detail
