#include "core/device.h"
#include "core/tensor.h"
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_qpn_split.cuh"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
constexpr std::int32_t kIntermediate = 17408; // Fp8MlpGateUpGeometry::kOutputRows / 2
} // namespace

bool fp8_linear_swiglu_qpn_split_supported(std::int32_t k, std::int32_t t) noexcept {
    return fp8_volta_qpn_supported(kIntermediate, k, t);
}

void fp8_linear_swiglu_qpn_split_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        float* gate_scratch, float* up_scratch,
                                        cudaStream_t stream) {
    const std::int32_t t = x.ne[1];

    Weight gate_weight = weight;
    gate_weight.n      = kIntermediate;

    Weight up_weight = weight;
    up_weight.n       = kIntermediate;
    up_weight.qdata   = static_cast<const std::uint8_t*>(weight.qdata) +
                       static_cast<std::int64_t>(kIntermediate) * weight.k;
    up_weight.scales = static_cast<const __nv_bfloat16*>(weight.scales) + kIntermediate;

    launch_fp8_volta_qpn_with_output(x, gate_weight,
                                     Fp8Fp32ContiguousOutput{gate_scratch, kIntermediate},
                                     kIntermediate, stream);
    launch_fp8_volta_qpn_with_output(x, up_weight, Fp8Fp32ContiguousOutput{up_scratch, kIntermediate},
                                     kIntermediate, stream);

    const std::int64_t elements = static_cast<std::int64_t>(kIntermediate) * t;
    const int threads           = 256;
    const int blocks = static_cast<int>(std::min<std::int64_t>((elements + threads - 1) / threads, 4096));
    fp8_swiglu_fp32_combine_kernel<<<blocks, threads, 0, stream>>>(
        gate_scratch, up_scratch, static_cast<__nv_bfloat16*>(out.data), elements);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
