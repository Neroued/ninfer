#include "ops/launcher/mtp_pack.h"

#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/kernel/mtp_pack.cuh"
#include "core/device.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

void mtp_pack_fc_input_launch(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                              cudaStream_t stream) {
    constexpr int kBlock = 256;
    for_each_token_slice(embedding_norm.ne[1], 1, [&](int token_offset, int token_count) {
        const Tensor embedding_slice = embedding_norm.slice(1, token_offset, token_count);
        const Tensor hidden_slice    = hidden_norm.slice(1, token_offset, token_count);
        Tensor out_slice             = out.slice(1, token_offset, token_count);
        const dim3 grid(static_cast<unsigned int>(div_up(embedding_norm.ne[0], kBlock)),
                        static_cast<unsigned int>(token_count));
        mtp_pack_fc_input_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(embedding_slice.data),
            static_cast<const __nv_bfloat16*>(hidden_slice.data),
            static_cast<__nv_bfloat16*>(out_slice.data), embedding_norm.ne[0]);
        CUDA_CHECK(cudaGetLastError());
    });
}

void mtp_split_attn_in_launch(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
                              cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int64_t n = static_cast<std::int64_t>(attn_in.ne[0]) * attn_in.ne[1];
    const int grid =
        static_cast<int>(std::max<std::int64_t>(1, div_up(n, static_cast<std::int64_t>(kBlock))));
    mtp_split_attn_in_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(attn_in.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data), static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data), attn_in.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
