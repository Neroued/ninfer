#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * @brief Applies plain RMSNorm to the non-anchor rows of each request block and packs
 * them into one dense matrix.
 *
 * @details `input` is contiguous BF16 `[5120,W,B]`, `weight` is contiguous BF16 `[5120]`, and
 * `output` is contiguous BF16 `[5120,(W-1)B]`, for `W` in `[2,16]` and `B` in `[1,8]`. Dimension
 * zero is stored fastest. For `b in [0,B)` and `i in [1,W-1]`, the ideal result is
 *
 * @f[
 *   s_{i,b} = \sum_{d=0}^{5119} \mathrm{FP32}(input_{d,i,b})^2,
 *   \qquad
 *   output_{d,(W-1)b+i-1} =
 *     \mathrm{FP32}(input_{d,i,b})
 *     \frac{\mathrm{FP32}(weight_d)}{\sqrt{s_{i,b}/5120 + 10^{-6}}}.
 * @f]
 *
 * This is plain multiplicative RMSNorm: the gain is `weight[d]`, not `1+weight[d]`. Anchor rows
 * `input[:,0,b]` are outside the computation and are not read. The oracle evaluates the complete
 * formula in FP64 from the represented BF16 inputs; the production reduction profile and BF16
 * output rounding are implementation effects covered by the Op's numerical criterion.
 *
 * All three tensors must be non-null, contiguous, 16-byte aligned, and pairwise non-overlapping.
 * The Op overwrites every output element, preserves both inputs, owns no workspace, and has no
 * persistent state effect.
 */
void rmsnorm_pack_tail(const Tensor& input, const Tensor& weight, Tensor& output,
                       cudaStream_t stream);

} // namespace ninfer::ops
