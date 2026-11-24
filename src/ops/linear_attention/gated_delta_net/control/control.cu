#include "ops/linear_attention/gated_delta_net/control/operands.h"
#include <array>
#include "ops/common/numeric/math.cuh"
#include "core/device.h"
#include <cuda_bf16.h>

namespace ninfer::ops::detail::gdn_control {
namespace {
__global__ void control_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, const float* a_log,
                               const float* bias, float* g, float* beta, std::int64_t elements) {
    const auto first  = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = std::int64_t(gridDim.x) * blockDim.x;
    for (auto i = first; i < elements; i += stride) {
        const int head = i % 48;
        g[i]           = -expf(a_log[head]) * common::softplus(__bfloat162float(a[i]) + bias[head]);
        beta[i]        = common::sigmoid(__bfloat162float(b[i]));
    }
}
} // namespace

} // namespace ninfer::ops::detail::gdn_control

namespace ninfer::ops {
void gdn_control(const Tensor& a, const Tensor& b, const Tensor& a_log, const Tensor& dt_bias,
                 GdnControlOutputs out, CudaExecutionView execution) {
    const auto stream = execution.stream();
    namespace control = detail::gdn_control;
    control::validate_control(a, b, a_log, dt_bias, out);
    if (a.ne[1] == 0) return;
    const std::array<const Tensor*, 6> tensors{&a, &b, &a_log, &dt_bias, &out.g, &out.beta};
    control::validate_storage(nullptr, tensors, execution);
    const auto elements   = out.g.numel();
    const unsigned blocks = static_cast<unsigned>((elements + 255) / 256);
    control::control_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(a.data), static_cast<const __nv_bfloat16*>(b.data),
        static_cast<const float*>(a_log.data), static_cast<const float*>(dt_bias.data),
        static_cast<float*>(out.g.data), static_cast<float*>(out.beta.data), elements);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
