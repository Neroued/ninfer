#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Geometry = Fp8MlpGateUpGeometry;
using Schedule = typename Fp8LinearDecodeProductionSchedule<Geometry>::Type;

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
    constexpr int kBlocks = kIntermediate / Schedule::kWarpsPerCta;
    const Rows rows{};
    const Fp8SwiGluOutput output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    fp8_gemv_kernel<Geometry, Schedule, Fp8SwiGluOutput, Rows, true>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
