#include "core/weight.h"
#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

namespace ninfer::ops::detail {

void bf16_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        bf16_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (x.ne[1] <= kBf16AttnInputSmallTDispatchEnd) {
        bf16_attn_input_small_t_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    bf16_attn_input_mma_launch(x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
