#include "ops/linear_attention/gated_delta_net/input_conv/q4_q5/launch.h"

namespace ninfer::ops::detail {

namespace {
Q4Q5GdnInputScheduleId select_grouped_projection(int columns) {
    if (columns <= 32) return Q4Q5GdnInputScheduleId::Mma32x32;
    if (columns <= 64) return Q4Q5GdnInputScheduleId::Mma32x64;
    return Q4Q5GdnInputScheduleId::Mma64x128;
}
} // namespace

void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              cudaStream_t stream) {
    if (x.ne[1] <= 12) {
        Tensor qk    = qkv.slice(0, 0, 4096);
        Tensor value = qkv.slice(0, 4096, 6144);
        launch_q4_q5_gdn_input_independent(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    const auto schedule = select_grouped_projection(x.ne[1]);
    launch_q4_q5_gdn_input_grouped_mma(x, qk_weight, value_z_weight, qkv, z, schedule, stream);
}

} // namespace ninfer::ops::detail
