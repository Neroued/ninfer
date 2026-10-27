#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "core/arena.h"
#include "core/weight.h"
#include "ninfer/ops/projection_weights.h"

namespace ninfer::ops {

/**
 * Four-stream expansion and offset RMSNorm, with S=4 and a supported residual width H:
 *   residual[d,s,m] = embedding[d,m] (exact BF16 word copy, including signed zero)
 *   normalized[d,s,m] = embedding[d,m] * (1+norm_weight[d,s])
 *                         / sqrt(sum_d embedding[d,m]^2/H + epsilon).
 * The gain 1+w is not rounded to BF16 before multiplication.
 *
 * Contiguous BF16 embedding [H,M], norm_weight [H,S], outputs [H,S,M]; bases are
 * 16-byte aligned. M>0; a [T,B] block flattens as M=T*B, m=t+T*b. Epsilon is positive
 * and finite. Outputs are fully written, mutually disjoint and disjoint from inputs;
 * read-only inputs may share backing and remain unchanged.
 * Supported widths are owned by the norm/combine implementation catalog.
 *
 * All computation entries submit to execution.stream() with a positive physical SM count and
 * support Graph capture/replay with current contents at bound addresses. Norm/combine
 * require no workspace; no entry allocates hidden device storage or owns persistent state.
 */
void hc_expand_norm(const Tensor& embedding, const Tensor& norm_weight, float epsilon,
                    Tensor& residual, Tensor& normalized, CudaExecutionView execution);

/**
 * Independent offset RMSNorm for each residual stream:
 *   normalized[d,s,m] = residual[d,s,m] * (1+norm_weight[d,s])
 *                         / sqrt(sum_d residual[d,s,m]^2/H + epsilon).
 * Contiguous BF16 residual/normalized [H,S,M], norm_weight [H,S]. Each stream has its
 * own statistics and affine weights; 1+w is not rounded to BF16. Domain and alignment
 * follow hc_expand_norm. Writes all of normalized, disjoint from every input; read-only
 * inputs may share backing and remain unchanged.
 */
void hc_grouped_rmsnorm(const Tensor& residual, const Tensor& norm_weight, float epsilon,
                        Tensor& normalized, CudaExecutionView execution);

/**
 * updated_residual[d,s,m] = residual[d,s,m] + injection[s,m]*block_output[d,m].
 * Contiguous BF16 residual/updated_residual [H,S,M], block_output [H,M], and FP32
 * injection [S,M]; 16-byte-aligned bases and M>0. Injection holds finite, already
 * activated coefficients in [0,2], applied directly without further sigmoid or scaling.
 * Device coefficient values are a caller precondition.
 *
 * Writes the complete BF16 output. It may exactly alias the residual view or be disjoint;
 * all other output/input overlap is forbidden. Read-only inputs may share backing.
 * Preserves all non-aliased inputs.
 */
void hc_combine(const Tensor& residual, const Tensor& block_output, const Tensor& injection,
                Tensor& updated_residual, CudaExecutionView execution);

/**
 * hc_combine followed by hc_grouped_rmsnorm. Normalization consumes the actual BF16
 * updated_residual words. The offset gain 1+next_norm_weight is not rounded to BF16.
 *
 * Combine operands and alias rules follow hc_combine. Contiguous BF16 next_norm_weight
 * [H,S] and normalized [H,S,M] have 16-byte-aligned bases; epsilon is positive and finite.
 * Both outputs are fully written. Normalized is disjoint from updated_residual and all
 * inputs; next_norm_weight cannot overlap an in-place residual. Preserves non-aliased inputs.
 */
void hc_combine_norm(const Tensor& residual, const Tensor& block_output, const Tensor& injection,
                     const Tensor& next_norm_weight, float epsilon, Tensor& updated_residual,
                     Tensor& normalized, CudaExecutionView execution);

/**
 * Unit-weight broadcast-add followed by grouped offset RMSNorm. Math, BF16 residual
 * boundary, layouts and alias rules follow the injection overload with coefficient 1.
 * This overload neither reads nor creates an injection tensor.
 */
void hc_combine_norm(const Tensor& residual, const Tensor& block_output,
                     const Tensor& next_norm_weight, float epsilon, Tensor& updated_residual,
                     Tensor& normalized, CudaExecutionView execution);

struct HcBf16MixWeights {
    Weight down_injection; // BF16 Contiguous [324,10240], rows [320,324) are injection.
    Weight up;             // BF16 Contiguous [10240,320], row=s*2560+d.
};

struct HcBf16FinalMixWeights {
    Weight down; // BF16 Contiguous [320,10240].
    Weight up;   // BF16 Contiguous [10240,320], row=s*2560+d.
};

struct HcBf16MixProfile {
    LinearProfile down_injection, up;
};

struct HcBf16FinalMixProfile {
    LinearProfile down, up;
};

// Mixed representation: the injection matrix always consumes the original BF16
// normalized input, independently of the down/up activation policies.
struct HcFp8MixWeights {
    SingleProjectionWeight down, up; // [rank,4H], [4H,rank], row-scaled E4M3/BF16.
    Weight injection;               // BF16 Contiguous [4,4H].
};
struct HcFp8FinalMixWeights {
    SingleProjectionWeight down, up;
};
struct HcFp8MixProfile {
    LinearProfile down, up, injection;
};
struct HcFp8FinalMixProfile {
    LinearProfile down, up;
};

/**
 * Low-rank mixing, independently for each column:
 *   [a,b] = down_injection * flatten(normalized), with a:320 and b:4
 *   u = SiLU(a/4); g = up*u
 *   block_input[d,m] = sum_s sigmoid(g[s*H+d,m])*normalized[d,s,m]/4
 *   injection[s,m] = 2*sigmoid(b[s,m]/4).
 * SiLU(v)=v*sigmoid(v); stream gates are independent. Private a/b/u/g representations
 * are not observable cast boundaries. FP32 injection includes saturation endpoints 0,2.
 *
 * Contiguous BF16 normalized [H,S,M], block_input [H,M], FP32 injection [S,M], M>0;
 * tensor and weight bases are 16-byte aligned. Weights have exactly the declared native
 * BF16 geometry, without bias or padding; activation computation admits A16 only.
 * Writes both outputs completely. They are mutually disjoint and disjoint from inputs,
 * weights and live workspace. Read-only inputs may share backing; all inputs are preserved.
 * Workspace is disjoint from all operands, caller-owned, call-scoped and sized by the
 * matching capacity query.
 * This Op performs no normalization or residual combination.
 */
void hc_mix(const Tensor& normalized, const HcBf16MixWeights& weights, MemoryArena& workspace,
            Tensor& block_input, Tensor& injection, CudaExecutionView execution);

/**
 * Low-rank mix with down-only [320,10240] weights and no b/injection. Math and effects
 * follow the ordinary form; sample_hidden is BF16 [H,M]. Adds no normalization or
 * residual combination.
 */
void hc_mix(const Tensor& normalized, const HcBf16FinalMixWeights& weights, MemoryArena& workspace,
            Tensor& sample_hidden, CudaExecutionView execution);

/**
 * The same four-stream mixing formula with separate down and injection matrices:
 *   a=down*flatten(normalized), b=injection*flatten(normalized), u=SiLU(a/4).
 * The implementation catalog owns supported H/rank geometries.
 * Down/up use row-scaled FP8 with A16Only or AllowA8 independently. Injection is
 * BF16/A16Only and its output remains FP32. All other layouts, aliases, effects,
 * stream/capture and caller-owned workspace rules follow the BF16 form.
 * SiLU and sigmoid use expf and FP32 division, without approximate intrinsics.
 * Private low-rank staging is allowed; no BF16 gate-logit boundary is introduced.
 */
void hc_mix(const Tensor& normalized, const HcFp8MixWeights& weights, MemoryArena& workspace,
            Tensor& block_input, Tensor& injection, CudaExecutionView execution);
// Final form omits injection and adds no normalization.
void hc_mix(const Tensor& normalized, const HcFp8FinalMixWeights& weights, MemoryArena& workspace,
            Tensor& sample_hidden, CudaExecutionView execution);

/**
 * Minimum call-local workspace capacity for every M in inclusive [min_tokens,max_tokens],
 * with positive endpoints and the supplied weight metadata. Measured from
 * a 256-byte-aligned arena cursor, including internal alignment but excluding trailing
 * padding and public tensors. Inspects host metadata only; device backing is not required.
 * Invalid profiles or intervals throw.
 */
[[nodiscard]] std::size_t hc_mix_workspace_capacity_bytes(const HcFp8MixWeights& weights,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);
[[nodiscard]] std::size_t hc_mix_workspace_capacity_bytes(const HcFp8FinalMixWeights& weights,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);
[[nodiscard]] std::size_t hc_mix_workspace_capacity_bytes(const HcFp8MixProfile& profile,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);
[[nodiscard]] std::size_t hc_mix_workspace_capacity_bytes(const HcFp8FinalMixProfile& profile,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

[[nodiscard]] std::size_t hc_mix_workspace_capacity_bytes(const HcBf16MixWeights& weights,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);
[[nodiscard]] std::size_t hc_mix_workspace_capacity_bytes(const HcBf16FinalMixWeights& weights,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

// Address-free forms use the same admission and capacity rules as bound weights.
[[nodiscard]] std::size_t hc_mix_workspace_capacity_bytes(const HcBf16MixProfile& profile,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);
[[nodiscard]] std::size_t hc_mix_workspace_capacity_bytes(const HcBf16FinalMixProfile& profile,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

} // namespace ninfer::ops
