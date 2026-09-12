#pragma once

#include "core/tensor.h"

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
void w8_attn_input_simt_r8_c4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);
void w8_attn_input_simt_r8_c4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                     Tensor& v, cudaStream_t stream);
void w8_attn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);
void w8_attn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                     Tensor& v, cudaStream_t stream);
void w8_attn_input_mma_r32_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void w8_attn_input_mma_r32_c128_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                       Tensor& v, cudaStream_t stream);
void w8_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void w8_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                       Tensor& v, cudaStream_t stream);
void w8_companion_attn_input_mma_r32_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream);
void w8_companion_attn_input_mma_r64_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream);
void w8_companion_attn_input_mma_r32_c96_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream);
void w8_companion_attn_input_mma_r64_c96_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream);
void w8_companion_attn_input_mma_r128_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                 Tensor& k, Tensor& v, cudaStream_t stream);
void w8_companion_attn_input_mma_r128_c80_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                 Tensor& k, Tensor& v, cudaStream_t stream);
void w8_dflash2_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& k, Tensor& v, cudaStream_t stream);
void w8_dflash2_attn_input_mma_r32_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                              Tensor& k, Tensor& v, cudaStream_t stream);
void w8_dflash2_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                               Tensor& k, Tensor& v, cudaStream_t stream);

void w8_dflash2_attn_input_mma_r16_c64_k128_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                                   Tensor&, cudaStream_t);

void w8_dflash2_attn_input_mma_r32_c32_k128_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                                   Tensor&, cudaStream_t);

void w8_dflash2_attn_input_mma_r32_c64_k128_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                                   Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
