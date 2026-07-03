#pragma once

// qus::kernels - MTP BF16 data movement helpers.
// mtp_pack_fc_input concatenates embedding and hidden norms along rows.
// mtp_split_attn_in copies fused attn_in rows into compact q/k/gate/v tensors.

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

void mtp_pack_fc_input(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                       cudaStream_t stream);

void mtp_split_attn_in(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
                       cudaStream_t stream);

} // namespace qus::kernels

