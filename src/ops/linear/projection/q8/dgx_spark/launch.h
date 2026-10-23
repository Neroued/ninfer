#pragma once

#include "ops/linear/kernels/q8/q8_launch.h"

namespace ninfer::ops::detail {

void launch_q8_n2048_k4096_bulk(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_n4096_k1024_bulk(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q8_n4096_k4096_bulk_full(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream);

void launch_q8_n12288_k4096_bulk_full(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream);

void launch_q8_n2560_k3072_bulk_full(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream);

void launch_q8_n6144_k2560_bulk_full(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream);

void launch_q8_n8192_k2560_bulk_full(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream);

void launch_q8_n2560_k6144_prefill512(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream);

void launch_q8_n16384_k2560_bulk_full(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream);

void launch_q8_n12288_k2560_bulk_full(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
