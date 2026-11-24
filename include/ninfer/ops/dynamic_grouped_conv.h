#pragma once

#include "core/device.h"
#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"


#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the transient capacity required by rmsnorm_dynamic_grouped_conv_prepare for every
 * width/batch pair in the inclusive intervals. Width endpoints lie in [2,16], batch endpoints
 * in [1,8]. Capacity covers every shape-selected route in the requested domain.
 */
[[nodiscard]] std::size_t rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
    std::int32_t min_width, std::int32_t max_width, std::int32_t min_batch_size,
    std::int32_t max_batch_size);

/**
 * Offset-free RMSNorm, dynamic coefficient projection and grouped two-tap convolution.
 *
 *   H=5120, W=2..16, group width 16, G=320, sides=2, taps=2. For h=16*g+j:
 *
 *     inv[i,b] = rsqrt(sum_h residual[h,i,b]^2 / H + eps)
 *     n[h,i,b] = residual[h,i,b] * inv[i,b] * norm_weight[h]
 *     row(s,t,g) = (s*2+t)*G+g
 *     d[s,t,g,i,b] = sum_h kernel_projection_weight[row(s,t,g),h] * n[h,i,b]
 *
 *     prepared[h,i,b] =
 *         (base_kernel[h,0,0] + d[0,0,g,i,b]) * n[h,i,b]
 *       + I(i>0) * (base_kernel[h,1,0] + d[0,1,g,i,b]) * n[h,i-1,b]
 *
 *     finish_delta[g,t,i,b] = d[1,t,g,i,b].
 *
 * Residual/prepared are contiguous BF16 [H,W,B], norm_weight [H], base_kernel
 * [H,2,2] with axes [channel,tap,side], and finish_delta [G,2,W,B]. The projection
 * weight is contiguous BF16 [1280,H]. W is in [2,16], B in [1,8]; tensor and weight
 * bases are 16-byte aligned. Eps is positive and finite. Position zero has no previous
 * contribution; the Op never reads another request or an earlier round.
 *
 * The normalized n and input-side coefficients are private intermediates, with no
 * observable cast boundary.
 *
 * Writes every element of prepared and finish_delta and preserves every input.
 * Inputs, weight payload, both outputs and live workspace are pairwise disjoint.
 * Workspace is caller-owned, call-scoped and sized by the query above. The Op does
 * not allocate or retain device memory and has no persistent state effect.
 */
void rmsnorm_dynamic_grouped_conv_prepare(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const Tensor& base_kernel,
                                          const Weight& kernel_projection_weight, Tensor& prepared,
                                          Tensor& finish_delta, MemoryArena& workspace,
                                          CudaExecutionView execution);

/**
 * Returns the transient capacity required by linear_dynamic_grouped_conv_add for every width/batch
 * pair in the inclusive intervals. input_rows is 4096 or 17408, widths lie in [2,16], and batch
 * sizes in [1,8]. Capacity follows every shape-selected production route.
 */
[[nodiscard]] std::size_t linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
    std::int32_t input_rows, std::int32_t min_width, std::int32_t max_width,
    std::int32_t min_batch_size, std::int32_t max_batch_size);

/**
 * Projection followed by grouped two-tap convolution and residual addition.
 *
 *   H=5120, W=2..16, group width 16, G=320. Let C be 4096 or 17408 and h=16*g+j:
 *
 *     z[h,i,b] = sum_c projection_weight[h,c] * x[c,i,b]
 *
 *     residual[h,i,b] +=
 *         (base_kernel[h,0,1] + finish_delta[g,0,i,b]) * z[h,i,b]
 *       + I(i>0) * (base_kernel[h,1,1] + finish_delta[g,1,i,b]) * z[h,i-1,b].
 *
 * Contiguous BF16 x [C,W,B], base_kernel [H,2,2] with axes [channel,tap,side],
 * finish_delta [G,2,W,B], and residual [H,W,B]. The projection weight is Q8_G32_FP16
 * RowSplit [H,C] without padding. W is in [2,16], B in [1,8]; tensor and weight-plane
 * bases are 16-byte aligned. Position zero has no previous contribution; the Op never
 * reads another request or an earlier round.
 *
 * Projected z is a private intermediate, with no observable cast boundary.
 *
 * Updates every residual element in place and preserves x, weights and finish_delta.
 * All operand storage, both weight planes and live workspace are pairwise disjoint.
 * Workspace is caller-owned, call-scoped and sized by the query above. The Op does
 * not allocate or retain device memory and has no persistent state effect.
 */
void linear_dynamic_grouped_conv_add(const Tensor& x, const Weight& projection_weight,
                                     const Tensor& base_kernel, const Tensor& finish_delta,
                                     Tensor& residual, MemoryArena& workspace,
                                     CudaExecutionView execution);

} // namespace ninfer::ops
