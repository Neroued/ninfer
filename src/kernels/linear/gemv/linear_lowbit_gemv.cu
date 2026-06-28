// qus::kernels::detail - tuned low-bit linear GEMV launcher.
#include "kernels/linear/gemv/linear_lowbit_gemv.h"

#include "kernels/linear/gemv/linear_lowbit_gemv.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kWarpSize        = 32;
constexpr int kWarpsPerBlock   = kThreadsPerBlock / kWarpSize;

int grid_for_rows(std::int64_t rows) {
    const std::int64_t grid = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (grid > std::numeric_limits<int>::max()) {
        throw std::overflow_error("linear: tuned low-bit GEMV launch grid exceeds CUDA limit");
    }
    return static_cast<int>(std::max<std::int64_t>(1, grid));
}

const std::uint8_t* payload_ptr(const Weight& w) {
    const void* payload = w.payload != nullptr ? w.payload : w.qdata;
    return static_cast<const std::uint8_t*>(payload);
}

template <class Codec>
void launch_tuned_lowbit_gemv(const Tensor& x, const std::uint8_t* payload, Tensor& out,
                              std::int32_t n, std::int32_t k, std::int32_t padded_k,
                              cudaStream_t stream) {
    linear_tuned_lowbit_gemv_kernel<Codec><<<grid_for_rows(n), kThreadsPerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), payload, static_cast<__nv_bfloat16*>(out.data),
        n, k, padded_k);
}

} // namespace

void linear_tuned_lowbit_gemv_launch(const Tensor& x, const Weight& w, Tensor& out,
                                     LinearFormat fmt, cudaStream_t stream) {
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t padded_k = w.padded_shape[1];
    const std::uint8_t* payload = payload_ptr(w);

    switch (fmt) {
    case LinearFormat::Q4G64_N64K64:
        launch_tuned_lowbit_gemv<Q4Codec>(x, payload, out, n, k, padded_k, stream);
        break;
    case LinearFormat::Q5G64_N64K64:
        launch_tuned_lowbit_gemv<Q5Codec>(x, payload, out, n, k, padded_k, stream);
        break;
    case LinearFormat::Q6G64_N64K64:
        launch_tuned_lowbit_gemv<Q6Codec>(x, payload, out, n, k, padded_k, stream);
        break;
    default:
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
