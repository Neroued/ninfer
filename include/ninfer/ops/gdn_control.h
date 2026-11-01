#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/projection_weights.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct GdnControlOutputs {
    Tensor g, beta;
};

struct GdnNormControlOutputs {
    Tensor normalized;
    GdnControlOutputs control;
};

/**
 * GDN control preparation.
 *
 * Math / indexing:
 *   g[h,m] = -exp(a_log[h]) * softplus(a[h,m] + dt_bias[h])
 *   beta[h,m] = sigmoid(b[h,m]).
 *
 * Supported domain:
 *   Contiguous BF16 a/b [48,M], FP32 a_log/dt_bias [48], FP32 g/beta [48,M],
 *   with backing naturally aligned to its element type.
 *   M is nonnegative; an empty matrix performs no device work.
 *
 * Numeric / effects:
 *   The complete formula consumes represented inputs. All operands are disjoint; inputs remain
 *   unchanged and both outputs are fully written. No persistent state or workspace.
 *
 * Execution:
 *   Work is submitted to execution.stream(); capture/replay observes the current operand contents.
 */
void gdn_control(const Tensor& a, const Tensor& b, const Tensor& a_log, const Tensor& dt_bias,
                 GdnControlOutputs out, CudaExecutionView execution);

/**
 * GDN control projection.
 *
 * Math / indexing:
 *   a[h,m] = sum_k Wa[h,k] * x[k,m]
 *   b[h,m] = sum_k Wb[h,k] * x[k,m]
 *   g[h,m] = -exp(a_log[h]) * softplus(a[h,m] + dt_bias[h])
 *   beta[h,m] = sigmoid(b[h,m]).
 *
 * Supported domain:
 *   x is contiguous BF16 [K,M]; a_log/dt_bias are FP32 [H]; outputs are FP32 [H,M].
 *   Combined contiguous BF16 parents [2H,K] admit (H,K)=(48,5120),(32,2048),(48,2560).
 *   DGX Spark additionally admits (H,K)=(24,2560).
 *   Rows [0,H) are A and [H,2H) are B. Paired BF16 parents [H,K] admit (48,5120).
 *   BF16 x and weight backing must be 16-byte aligned; FP32 backing must be 4-byte aligned.
 *   Every valid activation permission admits this BF16 route. Every positive M is supported
 *   within represented views and available storage. A [K,T,B] block is presented as [K,M],
 *   M=T*B, column=t+T*b.
 *
 * Numeric / effects:
 *   The complete formula consumes represented x directly; there is no input normalization or
 *   public BF16 projection intermediate. Outputs approximate the complete fused formula.
 *   All inputs, outputs and allocated scratch are disjoint. Inputs remain unchanged; both outputs
 *   are fully written. Combined weights produce zero-copy row views, without runtime repacking.
 *
 * Workspace / execution:
 *   Caller-owned transient workspace is scoped to the call. The capacity query covers every M in
 *   an inclusive interval and rejects invalid profiles/intervals. It inspects metadata without
 *   requiring uploaded backing. execution supplies stream and the physical positive SM count;
 *   device facts may change launch decomposition, never mathematical behavior. Capture/replay
 *   reads current contents at the bound addresses. There is no persistent state effect.
 */
void gdn_control_proj(const Tensor& x, const ProjectionWeights& weights, const Tensor& a_log,
                      const Tensor& dt_bias, GdnControlOutputs out, MemoryArena& workspace,
                      CudaExecutionView execution);

[[nodiscard]] std::size_t
gdn_control_proj_workspace_capacity_bytes(const ProjectionProfile& weights, std::int32_t first,
                                          std::int32_t last);

/**
 * Input RMSNorm and GDN control projection.
 *
 * Math / indexing:
 *   n[k,m] = x[k,m] / sqrt(mean_j(x[j,m]^2) + eps) * (1 + norm_weight[k])
 *   normalized_ideal[k,m] = n[k,m]
 *   a[h,m] = sum_k Wa[h,k] * n[k,m]; b[h,m] = sum_k Wb[h,k] * n[k,m]
 *   g[h,m] = -exp(a_log[h]) * softplus(a[h,m] + dt_bias[h])
 *   beta[h,m] = sigmoid(b[h,m]).
 *
 * Supported domain:
 *   Projection operands/effects follow gdn_control_proj for (H,K)=(48,5120),(32,2048).
 *   norm_weight is contiguous BF16 [K], normalized is BF16 [K,M]; both are 16-byte aligned.
 *   eps is positive and finite.
 *
 * Numeric:
 *   normalized is the explicit BF16 output for other projections. Controls derive from n, without
 *   a required rounding through normalized. Both outputs approximate the complete fused formula.
 *   The additional norm input and output obey the same non-overlap and preservation
 *   rules. Workspace and capture/replay follow the projection contract above.
 */
void gdn_norm_control_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                           const ProjectionWeights& weights, const Tensor& a_log,
                           const Tensor& dt_bias, GdnNormControlOutputs out, MemoryArena& workspace,
                           CudaExecutionView execution);

[[nodiscard]] std::size_t
gdn_norm_control_proj_workspace_capacity_bytes(const ProjectionProfile& weights, std::int32_t first,
                                               std::int32_t last);

[[nodiscard]] inline std::size_t
gdn_control_proj_workspace_capacity_bytes(const ProjectionWeights& weights, std::int32_t first,
                                          std::int32_t last) {
    return gdn_control_proj_workspace_capacity_bytes(projection_profile(weights), first, last);
}

[[nodiscard]] inline std::size_t
gdn_norm_control_proj_workspace_capacity_bytes(const ProjectionWeights& weights, std::int32_t first,
                                               std::int32_t last) {
    return gdn_norm_control_proj_workspace_capacity_bytes(projection_profile(weights), first, last);
}
} // namespace ninfer::ops
