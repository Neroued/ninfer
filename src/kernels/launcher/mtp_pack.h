#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void mtp_pack_fc_input_launch(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                              cudaStream_t stream);

void mtp_split_attn_in_launch(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
                              cudaStream_t stream);

} // namespace qus::kernels::detail

