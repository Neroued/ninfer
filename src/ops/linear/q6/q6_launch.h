#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q6Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q6_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_simt_r8_c5(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_simt_r8_c6(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_simt_r8_c7(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c16_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c24_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c32_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c40_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c48_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c56_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r32_c16_k256(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream);
void launch_q6_mma_r32_c24_k256(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream);
void launch_q6_mma_r32_c32_k256(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream);
void launch_q6_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c64_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c72_k128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c80(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c96(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c112(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
