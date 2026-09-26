#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Geometry = Fp8N34816K5120;
using Schedule = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;

constexpr int kIntermediate = Geometry::kOutputRows / 2;
static_assert(Schedule::kRowsPerWarp == 2);
static_assert((kIntermediate % Schedule::kWarpsPerCta) == 0);
using Rows = Fp8SwiGluRows<Schedule::kRowsPerWarp / 2, kIntermediate>;

} // namespace

void fp8_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream) {
    if (x.ne[0] != Geometry::kInputRows || x.ne[1] != 1 || out.ne[0] != kIntermediate ||
        out.ne[1] != 1 || weight.n != Geometry::kOutputRows || weight.k != Geometry::kInputRows) {
        throw std::invalid_argument("fp8 linear_swiglu decode: invalid exact problem");
    }
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, weight), output, Fp8SwiGluEpilogue{}, stream, Rows{});
}

} // namespace ninfer::ops::detail
