#include "ops/softmax_attention/dense/packed/launch.h"
#include "ops/softmax_attention/dense/packed/plan.h"
#include "ops/softmax_attention/kernels/noncausal/kernel.cuh"

namespace ninfer::ops::detail {
namespace {
template <int Rows, int D>
void launch_descriptors(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens,
                        Tensor* tiles, Tensor& out, cudaStream_t stream) {
    const int count = tiles ? tiles->ne[1] : q.ne[2] / Rows + (q.ne[2] % Rows != 0);
    if (tiles) {
        packed_attention_prepare_tiles_kernel<Rows><<<1, 256, 0, stream>>>(
            static_cast<const int*>(cu_seqlens.data), cu_seqlens.ne[0] - 1,
            static_cast<NoncausalAttentionTile*>(tiles->data), count, q.ne[2]);
        CUDA_CHECK(cudaGetLastError());
    }
    const auto* descriptors =
        tiles ? static_cast<const NoncausalAttentionTile*>(tiles->data) : nullptr;
    if constexpr (D == 64)
        launch_flash<Rows, Rows, 64, Rows <= 64 ? 4 : 2, 64>(q, k, v, descriptors, 0, count, out,
                                                             stream);
    else
        launch_flash<64, 64>(q, k, v, descriptors, 0, count, out, stream);
}
} // namespace

void launch_packed_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& cu_seqlens, Tensor* tiles, Tensor& out,
                             cudaStream_t stream) {
    if (q.ne[0] == 72) return launch_descriptors<64, 72>(q, k, v, cu_seqlens, tiles, out, stream);
    if (!tiles) return launch_packed_attention_uniform(q, k, v, q.ne[2], out, stream);
    switch (packed_query_rows(64, q.ne[2], cu_seqlens.ne[0] - 1)) {
    case 16:
        return launch_descriptors<16, 64>(q, k, v, cu_seqlens, tiles, out, stream);
    case 32:
        return launch_descriptors<32, 64>(q, k, v, cu_seqlens, tiles, out, stream);
    case 64:
        return launch_descriptors<64, 64>(q, k, v, cu_seqlens, tiles, out, stream);
    case 128:
        return launch_descriptors<128, 64>(q, k, v, cu_seqlens, tiles, out, stream);
    }
}
} // namespace ninfer::ops::detail
