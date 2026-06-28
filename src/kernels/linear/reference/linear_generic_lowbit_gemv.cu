// qus::kernels::detail - generic low-bit linear GEMV launcher.
#include "kernels/linear/reference/linear_generic.h"

#include "kernels/linear/reference/linear_generic_lowbit.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

constexpr int kBlock = 128;

int grid_for(std::int64_t work) {
    const std::int64_t grid = (work + kBlock - 1) / kBlock;
    if (grid > std::numeric_limits<int>::max()) {
        throw std::overflow_error("linear: generic low-bit GEMV launch grid exceeds CUDA limit");
    }
    return static_cast<int>(std::max<std::int64_t>(1, grid));
}

const std::uint8_t* payload_ptr(const Weight& w) {
    const void* payload = w.payload != nullptr ? w.payload : w.qdata;
    return static_cast<const std::uint8_t*>(payload);
}

} // namespace

void linear_generic_lowbit_gemv_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       LinearFormat fmt, cudaStream_t stream) {
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t padded_k = w.padded_shape[1];
    const std::uint8_t* payload = payload_ptr(w);

    switch (fmt) {
    case LinearFormat::Q4G64_N64K64:
        linear_generic_lowbit_gemv_kernel<Q4Codec><<<grid_for(n), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), payload, static_cast<__nv_bfloat16*>(out.data),
            n, k, padded_k);
        break;
    case LinearFormat::Q5G64_N64K64:
        linear_generic_lowbit_gemv_kernel<Q5Codec><<<grid_for(n), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), payload, static_cast<__nv_bfloat16*>(out.data),
            n, k, padded_k);
        break;
    case LinearFormat::Q6G64_N64K64:
        linear_generic_lowbit_gemv_kernel<Q6Codec><<<grid_for(n), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), payload, static_cast<__nv_bfloat16*>(out.data),
            n, k, padded_k);
        break;
    default: break;
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
