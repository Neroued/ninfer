#include "core/weight.h"
#include "ops/linear_attention/gated_delta_net/input_conv/q8/launch.h"
#include "core/device.h"
#include "ops/linear/kernels/q8/q8_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {

using Output   = LinearBf16SegmentedOutput<8192, 4096>;
using Schedule = Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;

} // namespace

void launch_q8_gdn_input_mma_r64_c128(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      cudaStream_t stream) {
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    launch_q8_a16_mma<Schedule>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                                stream);
}
} // namespace ninfer::ops::detail
