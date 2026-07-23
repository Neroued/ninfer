#pragma once

#include "core/tensor.h"
#include "ops/linear/w8/w8_rowsplit_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void w8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream);
void w8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                 Tensor& v, cudaStream_t stream);
void w8_companion_attn_input_decode_r4_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                              Tensor& k, Tensor& v, cudaStream_t stream);
void w8_companion_attn_input_decode_r16_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                               Tensor& k, Tensor& v, cudaStream_t stream);
void w8_attn_input_simt_r8_c4_launch(W8KernelVariant variant, const Tensor& x, const Weight& weight,
                                     Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                     cudaStream_t stream);
void w8_attn_input_simt_r8_c4_launch(W8KernelVariant variant, const Tensor& x, const Weight& weight,
                                     Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream);
void w8_attn_input_splitk_mma_launch(W8KernelVariant variant, const Tensor& x, const Weight& weight,
                                     Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                     cudaStream_t stream);
void w8_attn_input_splitk_mma_launch(W8KernelVariant variant, const Tensor& x, const Weight& weight,
                                     Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream);
void w8_attn_input_mma_r32_c128_launch(W8KernelVariant variant, const Tensor& x,
                                       const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                                       Tensor& v, cudaStream_t stream);
void w8_attn_input_mma_r32_c128_launch(W8KernelVariant variant, const Tensor& x,
                                       const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                                       cudaStream_t stream);
void w8_attn_input_mma_r64_c128_launch(W8KernelVariant variant, const Tensor& x,
                                       const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                                       Tensor& v, cudaStream_t stream);
void w8_attn_input_mma_r64_c128_launch(W8KernelVariant variant, const Tensor& x,
                                       const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                                       cudaStream_t stream);
void w8_companion_attn_input_mma_r32_c64_launch(W8KernelVariant variant, const Tensor& x,
                                                const Weight& weight, Tensor& q, Tensor& k,
                                                Tensor& v, cudaStream_t stream);
void w8_companion_attn_input_mma_r64_c64_launch(W8KernelVariant variant, const Tensor& x,
                                                const Weight& weight, Tensor& q, Tensor& k,
                                                Tensor& v, cudaStream_t stream);
void w8_companion_attn_input_mma_r32_c96_launch(W8KernelVariant variant, const Tensor& x,
                                                const Weight& weight, Tensor& q, Tensor& k,
                                                Tensor& v, cudaStream_t stream);
void w8_companion_attn_input_mma_r64_c96_launch(W8KernelVariant variant, const Tensor& x,
                                                const Weight& weight, Tensor& q, Tensor& k,
                                                Tensor& v, cudaStream_t stream);
void w8_companion_attn_input_mma_r128_c64_launch(W8KernelVariant variant, const Tensor& x,
                                                 const Weight& weight, Tensor& q, Tensor& k,
                                                 Tensor& v, cudaStream_t stream);
void w8_companion_attn_input_mma_r128_c80_launch(W8KernelVariant variant, const Tensor& x,
                                                 const Weight& weight, Tensor& q, Tensor& k,
                                                 Tensor& v, cudaStream_t stream);

} // namespace ninfer::ops::detail
