#include "ops/launcher/position.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kernel/position.cuh"

namespace ninfer::ops::detail {

void fill_i32_positions_launch(Tensor& positions, std::int32_t start, cudaStream_t stream) {
    constexpr int block = 256;
    const int grid      = div_up(positions.ne[0], block);
    fill_i32_positions_kernel<<<grid, block, 0, stream>>>(
        static_cast<std::int32_t*>(positions.data), positions.ne[0], start);
    CUDA_CHECK(cudaGetLastError());
}

void offset_i32_positions_launch(const Tensor& source, const Tensor& delta, Tensor& destination,
                                 cudaStream_t stream) {
    offset_i32_positions_block_launch(source, delta, destination, 256, stream);
}

void offset_i32_positions_block_launch(const Tensor& source, const Tensor& delta,
                                       Tensor& destination, int block, cudaStream_t stream) {
    const int grid = div_up(source.ne[0], block);
    offset_i32_positions_kernel<<<grid, block, 0, stream>>>(
        static_cast<const std::int32_t*>(source.data), static_cast<const std::int32_t*>(delta.data),
        static_cast<std::int32_t*>(destination.data), source.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
