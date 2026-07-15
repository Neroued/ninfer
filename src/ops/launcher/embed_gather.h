#pragma once

// ninfer::ops::detail - private launch prototypes for embedding variants.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void embed_gather_dense_launch(const Tensor& ids, const Tensor& table, Tensor& out,
                               cudaStream_t stream);
void embed_gather_q6_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream);
void embed_gather_w8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
