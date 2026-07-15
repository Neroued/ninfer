// ninfer::ops - l2norm launcher: grid/block/stream configuration + kernel launch.
#include "ops/launcher/l2norm.h"

#include "ops/common/math.h"
#include "ops/kernel/l2norm.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

void l2norm_launch(const Tensor& x, float eps, Tensor& out, cudaStream_t stream) {
    constexpr int kBlock = 512;
    const std::int32_t d = x.ne[0];
    if (d <= 0) { throw std::invalid_argument("l2norm: ne[0] must be positive"); }
    const std::int64_t rows = out.numel() / d;
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("l2norm: row count exceeds CUDA grid limit");
    }

    constexpr int kRowsPerBlock = kBlock / 32;
    const auto blocks =
        static_cast<unsigned int>(div_up(rows, static_cast<std::int64_t>(kRowsPerBlock)));
    l2norm_kernel<<<blocks, kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                 static_cast<__nv_bfloat16*>(out.data), d, rows,
                                                 eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
