// qus::kernels - rope launcher: grid/block/stream configuration + kernel launch.
#include "kernels/launcher/rope.h"

#include "kernels/kernel/rope.cuh"
#include "qus/core/device.h"  // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <limits>

namespace qus::kernels::detail {

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int64_t half = static_cast<std::int64_t>(rotary_dim / 2);
    const std::int64_t T = static_cast<std::int64_t>(positions.ne[0]);
    const std::int64_t q_pairs = static_cast<std::int64_t>(q.ne[1]) * T * half;
    const std::int64_t k_pairs = static_cast<std::int64_t>(k.ne[1]) * T * half;
    const std::int64_t total_pairs = q_pairs + k_pairs;
    const std::int64_t blocks =
        (total_pairs + static_cast<std::int64_t>(kBlock) - 1) / static_cast<std::int64_t>(kBlock);
    const int grid =
        static_cast<int>(std::min<std::int64_t>(blocks, std::numeric_limits<int>::max()));

    rope_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data), rotary_dim, theta, q.ne[1], k.ne[1], positions.ne[0],
        total_pairs);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
