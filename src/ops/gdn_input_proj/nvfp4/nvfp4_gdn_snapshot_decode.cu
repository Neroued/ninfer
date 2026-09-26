#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_conv_epilogue.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"

namespace ninfer::ops::detail {

void nvfp4_gdn_snapshot_decode_launch(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight, Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                      Tensor& value, Tensor& z, cudaStream_t stream) {
    using Geometry = Nvfp4N16384K5120;
    using Schedule =
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;

    launch_nvfp4_a16_gemv<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, weight),
        make_gdn_conv_output<1>(
            conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
            SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                   static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                   kGdnChannels}),
        Nvfp4GdnConvEpilogue<1>{}, stream);
}

} // namespace ninfer::ops::detail
