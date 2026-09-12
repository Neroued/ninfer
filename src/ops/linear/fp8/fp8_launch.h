#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void launch_fp8_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_fp8_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_fp8_vocabulary_a16_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void launch_fp8_vocabulary_a16_gemm(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
