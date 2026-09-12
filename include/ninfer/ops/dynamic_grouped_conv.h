#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

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
 * Op: rmsnorm_dynamic_grouped_conv_prepare
 *
 * Math / indexing:
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
 * Logical shapes / supported domain:
 *   residual/prepared are contiguous BF16 [5120,W,B], W is in [2,16] and B in [1,8];
 *   norm_weight is contiguous BF16 [5120]. base_kernel is the runtime view BF16 [5120,2,2] with
 * axes [channel,tap,side]; kernel_projection_weight is contiguous BF16_CTRL [1280,5120]; and
 *   finish_delta is contiguous BF16 [320,2,W,B]. eps is positive and finite. Position zero has
 *   no previous-tap contribution: the Op never reads another request or an earlier round.
 *
 * Numeric:
 *   The oracle evaluates the complete formula naively in FP64 from the represented BF16 inputs.
 *   Both BF16 outputs are promoted and compared directly with those ideal values under their
 *   named reduction criteria. RMS reduction order, Tensor Core operand staging, projection
 *   factorization, split-K reduction, and private intermediate precision are implementation
 *   choices; neither n nor the input-side coefficient plane is an observable cast boundary.
 *
 * Effects:
 *   Writes every element of prepared and finish_delta and preserves every input. Inputs, the
 *   weight payload, both outputs, and live workspace must be pairwise non-overlapping. The Op has
 *   no persistent state side effect.
 *
 * Workspace:
 *   Caller-owned call-scoped storage sized by the capacity query above. The implementation does
 *   not allocate or retain device memory.
 */
void rmsnorm_dynamic_grouped_conv_prepare(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const Tensor& base_kernel,
                                          const Weight& kernel_projection_weight, Tensor& prepared,
                                          Tensor& finish_delta, WorkspaceArena& workspace,
                                          cudaStream_t stream);

/**
 * Returns the transient capacity required by linear_dynamic_grouped_conv_add for every width/batch
 * pair in the inclusive intervals. input_rows is 4096 or 17408, widths lie in [2,16], and batch
 * sizes in [1,8]. Capacity follows every shape-selected production route.
 */
[[nodiscard]] std::size_t linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
    std::int32_t input_rows, std::int32_t min_width, std::int32_t max_width,
    std::int32_t min_batch_size, std::int32_t max_batch_size);

/**
 * Op: linear_dynamic_grouped_conv_add
 *
 * Math / indexing:
 *   H=5120, W=2..16, group width 16, G=320. Let C be 4096 or 17408 and h=16*g+j:
 *
 *     z[h,i,b] = sum_c projection_weight[h,c] * x[c,i,b]
 *
 *     residual[h,i,b] +=
 *         (base_kernel[h,0,1] + finish_delta[g,0,i,b]) * z[h,i,b]
 *       + I(i>0) * (base_kernel[h,1,1] + finish_delta[g,1,i,b]) * z[h,i-1,b].
 *
 * Logical shapes / supported domain:
 *   x is contiguous BF16 [C,W,B]; projection_weight is W8G32_F16S RowSplit [5120,C] with
 *   unpadded C; base_kernel is contiguous BF16 [5120,2,2] with axes [channel,tap,side];
 *   finish_delta is contiguous BF16 [320,2,W,B]; and residual is contiguous BF16 [5120,W,B].
 *   W is in [2,16] and B in [1,8]. Position zero has no previous-tap contribution: the Op never
 * reads another request or an earlier round.
 *
 * Numeric:
 *   The oracle decodes each signed W8 value with its stored FP16 group scale and evaluates the
 *   complete formula in FP64 from represented inputs. The projection is a private intermediate;
 *   the contract does not prescribe its storage or arithmetic precision. The implementation
 *   writes the final residual in BF16.
 *
 * Effects:
 *   Updates every element of residual in place and preserves x, projection_weight, base_kernel,
 *   and finish_delta. All operand storage, both weight planes, and live workspace must be pairwise
 *   non-overlapping. The Op has no persistent state side effect.
 *
 * Workspace:
 *   Caller-owned call-scoped storage sized by the capacity query above. The implementation does
 *   not allocate or retain device memory.
 */
void linear_dynamic_grouped_conv_add(const Tensor& x, const Weight& projection_weight,
                                     const Tensor& base_kernel, const Tensor& finish_delta,
                                     Tensor& residual, WorkspaceArena& workspace,
                                     cudaStream_t stream);

} // namespace ninfer::ops
