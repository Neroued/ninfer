#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q5Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q5_gemv_r16_s2_x(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_split4_c1_k5120(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_split4_c1_k6144(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_split4_c1_k17408(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_mma_r64_c16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_mma_r64_c32_s3(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_mma_r32_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
