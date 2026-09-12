#include "ops/launcher/prepare_ragged_prefix.h"

#include "core/device.h"
#include "ops/kernel/prepare_ragged_prefix.cuh"

#include <algorithm>

namespace ninfer::ops::detail {

void prepare_ragged_prefix_launch(const Tensor& source, const Tensor& lanes, const Tensor& starts,
                                  const Tensor& ends, Tensor& destination, Tensor& positions,
                                  Tensor& counts, cudaStream_t stream) {
    const int vectors = source.ne[0] / 8;
    const int tiles   = (vectors + kRaggedPrefixVectorsPerBlock - 1) / kRaggedPrefixVectorsPerBlock;
    // Separate axes avoid per-CTA integer division. Grid stride preserves generic D when
    // feature tiles exceed CUDA's Z limit. Real feature widths use one vector per thread.
    const dim3 grid(source.ne[1], destination.ne[2], std::min(tiles, 65535));
    prepare_ragged_prefix_kernel<<<grid, kRaggedPrefixVectorsPerBlock, 0, stream>>>(
        static_cast<const uint4*>(source.data), static_cast<const std::int32_t*>(lanes.data),
        static_cast<const std::int32_t*>(starts.data), static_cast<const std::int32_t*>(ends.data),
        static_cast<uint4*>(destination.data), static_cast<std::int32_t*>(positions.data),
        static_cast<std::int32_t*>(counts.data), source.ne[0] / 8, source.ne[1],
        source.nb[1] / static_cast<std::int64_t>(sizeof(uint4)),
        source.nb[2] / static_cast<std::int64_t>(sizeof(uint4)));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
