#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/linear/fp8/fp8_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

// The attention projections are where the model's FP8 weights actually live -- two per layer for
// all 64 layers -- so the quadpair route matters here more than it does in plain Linear. The
// parent GEMM is identical; only the epilogue differs, and the output policy carries that.
void launch_fp8_attn_input_volta_qpn(const Tensor& x, const Weight& weight, Tensor& query,
                                     Tensor& gate, Tensor& key, Tensor& value,
                                     cudaStream_t stream) {
    const Fp8AttentionInputOutput output{static_cast<__nv_bfloat16*>(query.data),
                                         static_cast<__nv_bfloat16*>(key.data),
                                         static_cast<__nv_bfloat16*>(gate.data),
                                         static_cast<__nv_bfloat16*>(value.data)};
    launch_fp8_volta_qpn_with_output(x, weight, output, weight.n, stream);
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
