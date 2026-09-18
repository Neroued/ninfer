#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q4Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q4_gemv_r4_w1_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q4_gemv_r1_q8_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q4_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c48(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c56(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c72(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c80(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c96(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c112(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c120(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
