#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"

namespace ninfer::ops {

struct MhcGeometry {
    std::int32_t hidden;
    std::int32_t streams;
};

struct MhcPreParameters {
    float coefficient_epsilon;
    float branch_epsilon;
    float pre_epsilon;
    float sinkhorn_epsilon;
    float post_multiplier;
    std::int32_t sinkhorn_iterations;
};

struct MhcPreWeights {
    Weight projection; // BF16 Contiguous [N,S*D], N=S*(S+2), no padding or bias.
    Tensor base;       // BF16 [N].
    Tensor scale;      // BF16 [3].
    Tensor gamma;      // BF16 [D], direct gain.
};

struct MhcCoefficients {
    Tensor post; // FP32 [S,M].
    Tensor mix;  // FP32 [S_new,S_old,M]: mix[j,i,m] = C[i,j,m].
};

struct MhcPreOutputs {
    Tensor block_input; // BF16 [D,M], includes branch RMSNorm.
    MhcCoefficients coefficients;
};

/**
 * Manifold-constrained hyper-connections, independently for each physical column m:
 *   r = rsqrt(sum_{i,d} X[d,i,m]^2/(S*D) + coefficient_epsilon)
 *   z[a] = r * sum_{i,d} projection[a,i*D+d] * X[d,i,m]
 *   p[i] = sigmoid(scale[0]*z[i] + base[i]) + pre_epsilon
 *   q[j] = post_multiplier * sigmoid(scale[1]*z[S+j] + base[S+j])
 *   A[i,j] = scale[2]*z[2*S+S*i+j] + base[2*S+S*i+j]
 *   C = softmax_row(A) + sinkhorn_epsilon
 *   C[i,j] /= sum_i C[i,j] + sinkhorn_epsilon
 *   repeat sinkhorn_iterations-1 times: normalize rows, then columns, each denominator
 *     including sinkhorn_epsilon and computed from that step's complete row/column.
 *   v[d] = sum_i p[i]*X[d,i,m]
 *   block_input[d,m] = RN_BF16(gamma[d]*v[d]*rsqrt(sum_d v[d]^2/D + branch_epsilon)).
 * Writes all block_input, q and C. Pre gates are not normalized; C is only approximately
 * doubly stochastic. Gamma is a direct gain.
 *
 * Contiguous BF16 residual X [D,S,M]; native dimension zero is contiguous. M>0 and
 * registered D/S geometry are required. All tensor/weight bases are 16-byte aligned.
 * Epsilons and multiplier are positive normal FP32 values; iterations is a positive integer.
 * Device values are finite caller inputs; no value validation reads them back to Host.
 * Products, sums, coefficients, Sinkhorn, collapse and norms use FP32 computation.
 * Normalized X, logits and v have no intermediate BF16 representation boundary.
 *
 * All operands and workspace belong to execution.device(). Outputs are disjoint from
 * inputs, weights, one another and live scratch; read-only operands may share backing.
 * Non-aliased inputs are preserved. Entries submit on execution.stream(), support CUDA
 * Graph replay with current contents at fixed addresses, and perform no hidden allocation,
 * synchronization or Host readback. They own no persistent or cross-column state.
 * Scratch is borrowed within an arena scope, restoring its cursor on return or failure;
 * device lifetime and reuse follow stream ordering. Capacity queries below cover it.
 */
void mhc_pre(const Tensor& residual, const MhcPreWeights& weights,
             const MhcPreParameters& parameters, MhcPreOutputs outputs, MemoryArena& workspace,
             CudaExecutionView execution);

/**
 * updated_residual[d,j,m] = RN_BF16(sum_i C[i,j,m]*residual[d,i,m]
 *                                 + q[j,m]*block_output[d,m]).
 * Contiguous BF16 block_output [D,M], residual/updated_residual [D,S,M]. Coefficients
 * are already activated FP32 values, applied directly. All terms are combined in FP32
 * before a single round-to-nearest, ties-to-even BF16 store. C need not have unit sums.
 * Updated residual may exactly alias the residual view or be completely disjoint.
 * Partial overlap and every other output/input overlap are forbidden, including other
 * read-only operands overlapping an in-place residual. Reads all old streams before
 * overwriting them. Other domain, execution and alignment rules follow mhc_pre.
 * This entry has zero scratch.
 */
void mhc_post(const Tensor& residual, const Tensor& block_output,
              const MhcCoefficients& coefficients, Tensor& updated_residual,
              CudaExecutionView execution);

/**
 * Copies each BF16 embedding word [D,M] exactly to every stream of residual [D,S,M],
 * including signed zero, and computes the complete mhc_pre from those values.
 * Residual and pre outputs are mutually disjoint and disjoint from all inputs/weights.
 */
void mhc_expand_pre(const Tensor& embedding, const MhcPreWeights& weights,
                    const MhcPreParameters& parameters, Tensor& residual, MhcPreOutputs outputs,
                    MemoryArena& workspace, CudaExecutionView execution);

/**
 * Complete mhc_post followed by mhc_pre with next_weights/next_parameters. Publishes
 * BF16 updated_residual; the next projection, coefficient RMS and collapse all consume
 * its actual rounded BF16 values. Updated residual has mhc_post's exact-alias option;
 * next outputs are disjoint from every input, weight and other output. In particular,
 * old and new q/C cannot overlap. The fusion adds no other representation boundaries.
 */
void mhc_post_pre(const Tensor& residual, const Tensor& block_output,
                  const MhcCoefficients& coefficients, const MhcPreWeights& next_weights,
                  const MhcPreParameters& next_parameters, Tensor& updated_residual,
                  MhcPreOutputs next_outputs, MemoryArena& workspace, CudaExecutionView execution);

/**
 * Computes the complete mhc_post BF16 boundary on chip, then:
 *   mean[d,m] = sum_j updated_residual[d,j,m]/S
 *   hidden[d,m] = RN_BF16(gamma_final[d]*mean[d,m]
 *                        *rsqrt(sum_d mean[d,m]^2/D + final_epsilon)).
 * Mean remains FP32 through final RMSNorm. Does not assume unit row/column sums of C.
 * Only hidden (contiguous BF16 [D,M]) is written; all inputs are preserved. Gamma_final
 * is contiguous BF16 [D], a direct gain; final_epsilon is finite and positive.
 * Hidden is disjoint from all inputs and workspace. Other rules follow mhc_pre.
 */
void mhc_post_mean_norm(const Tensor& residual, const Tensor& block_output,
                        const MhcCoefficients& coefficients, const Tensor& gamma_final,
                        float final_epsilon, Tensor& hidden, MemoryArena& workspace,
                        CudaExecutionView execution);

/**
 * Minimum scratch high-water mark over inclusive [min_columns,max_columns], measured
 * from a 256-byte-aligned call-local cursor, including internal gaps but no trailing
 * padding. Geometry is registered, sm_count is the device's positive physical SM count,
 * and 0 < min_columns <= max_columns. Inspects metadata only; invalid queries throw.
 * Parameters do not change scratch. Public tensors are excluded.
 */
[[nodiscard]] std::size_t mhc_pre_workspace_capacity_bytes(const MhcGeometry& geometry,
                                                           std::int32_t sm_count,
                                                           std::int32_t min_columns,
                                                           std::int32_t max_columns);

[[nodiscard]] std::size_t mhc_expand_pre_workspace_capacity_bytes(const MhcGeometry& geometry,
                                                                  std::int32_t sm_count,
                                                                  std::int32_t min_columns,
                                                                  std::int32_t max_columns);
[[nodiscard]] std::size_t mhc_post_pre_workspace_capacity_bytes(const MhcGeometry& geometry,
                                                                std::int32_t sm_count,
                                                                std::int32_t min_columns,
                                                                std::int32_t max_columns);
[[nodiscard]] std::size_t mhc_post_mean_norm_workspace_capacity_bytes(const MhcGeometry& geometry,
                                                                      std::int32_t sm_count,
                                                                      std::int32_t min_columns,
                                                                      std::int32_t max_columns);

} // namespace ninfer::ops
