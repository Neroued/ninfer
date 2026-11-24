#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void launch_q4_swiglu_gemv_pair(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_swiglu_mma_split_half_pair_r32_c128(const Tensor& x, const Weight& w, Tensor& out,
                                                   cudaStream_t stream);
// Wide folded tiles for the leading column blocks plus a narrow route for a small remainder.
void launch_q4_swiglu_mma_split_half_pair_r32_c128_tail(const Tensor& x, const Weight& w,
                                                        Tensor& out, cudaStream_t stream);
void launch_q4_swiglu_small_t_tiled(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);

void launch_q4_swiglu_materialized_projection(const Tensor&, const Weight&, Tensor&, cudaStream_t);
} // namespace ninfer::ops::detail
