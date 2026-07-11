// qus::kernels - argmax launcher: grid/block/stream configuration + kernel launch.
#include "kernels/launcher/argmax.h"

#include "kernels/common/math.h"
#include "kernels/kernel/argmax.cuh"
#include "qus/core/device.h"  // CUDA_CHECK

#include <cstdint>

namespace qus::kernels::detail {

void argmax_launch(const Tensor& logits, Tensor& out, cudaStream_t stream) {
    const std::int32_t vocab = logits.ne[0];
    const std::int32_t t_count = logits.ne[1];
    if (t_count == 0) { return; }

    constexpr int kTileElems = kArgmaxBlock * kArgmaxItemsPerThread;
    const int tiled_blocks = div_up(vocab, kTileElems);
    if (tiled_blocks < 2) {
        argmax_kernel<<<static_cast<unsigned int>(t_count), kArgmaxBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
            vocab);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    CUDA_CHECK(cudaMemsetAsync(out.data, 0, static_cast<std::size_t>(t_count) * sizeof(std::int32_t),
                               stream));
    const dim3 grid(static_cast<unsigned int>(tiled_blocks), static_cast<unsigned int>(t_count));
    argmax_tiled_atomic_kernel<<<grid, kArgmaxBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
        vocab);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
