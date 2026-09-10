#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void mtp_pack_fc_input_launch(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                              cudaStream_t stream);

bool mtp_norm_pack_fc_input_admits(std::int32_t d, const Tensor& embedding,
                                   const Tensor& embedding_weight, const Tensor& hidden,
                                   const Tensor& hidden_weight, const Tensor& out);

void mtp_norm_pack_fc_input_launch(const Tensor& embedding, const Tensor& embedding_weight,
                                   const Tensor& hidden, const Tensor& hidden_weight, Tensor& out,
                                   float eps, cudaStream_t stream);

bool mtp_residual_norm_admits(std::int32_t d, const Tensor& delta, const Tensor& residual,
                              const Tensor& weight, const Tensor& out);

void mtp_residual_norm_launch(const Tensor& delta, Tensor& residual, const Tensor& weight,
                              Tensor& out, float eps, cudaStream_t stream);

void mtp_split_attn_in_launch(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
