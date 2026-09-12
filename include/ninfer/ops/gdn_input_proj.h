#pragma once

// ninfer::ops - fused GDN Q/K/V/Z input projections.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: gdn_input_proj
 *
 * Math / indexing:
 *   qkv[:,t] = concat(qk_weight * x[:,t], value_z_weight[0:6144,:] * x[:,t])
 *   z[:,t]   = value_z_weight[6144:12288,:] * x[:,t]
 *
 * Logical shapes:
 *   x [5120,T], qk weight/output rows 4096, value/z rows 6144 each, qkv [10240,T],
 *   z [6144,T]. T may be any positive value. x, qkv, and z are contiguous BF16.
 *   qk_weight is Q4G64_F16S RowSplit [4096,5120] and value_z_weight is one
 *   Q5G64_F16S RowSplit parent [12288,5120] in [value,z] row order, both with FP16 scales.
 *
 * Numeric:
 *   The oracle exact-decodes both weight parents and evaluates all four logical projections
 *   naively in FP64 from the represented input. The BF16 qkv and z outputs are promoted and
 *   compared directly with those ideal values; final output storage rounding belongs to
 *   GdnInputProj's named A16 criterion, not the oracle. Production routes may choose their
 *   private precision independently; every registered route writes both final allocations.
 *
 * Effects:
 *   Writes the full qkv and z outputs; inputs and outputs must not alias.
 *
 * Workspace:
 *   No transient bytes are required.
 */
void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    Tensor& qkv, Tensor& z, cudaStream_t stream);

/**
 * Single-parent GDN projection. Registered parent forms are:
 *
 * - W8G32_F16S RowSplit [12288,2048], with stored row counts [2048,2048,4096,4096];
 * - NVFP4 BlockScaleK16M128x4 [16384,5120], with stored row counts [2048,2048,6144,6144].
 * - FP8_E4M3FN_ROW_BF16S RowScale [16384,5120], with stored row counts
 *   [2048,2048,6144,6144].
 *
 * The first three ranges are written contiguously to qkv and the final range is written to z.
 * W8 admits A16 only. NVFP4 admits A16Only and AllowA4; AllowA4 permits private activation
 * quantization at every positive T. FP8 admits A16Only and AllowA8 at every positive T; AllowA8
 * selects A16 through T=7 and private activation quantization followed by A8 Tensor Core
 * contraction at every T>=8. Every route writes the two independent final allocations directly.
 * The complete projection is evaluated against the same exact-decode/naive-FP64 oracle;
 * activation quantization and the production reduction profile are private effects covered by the
 * selected criterion. x, both persistent weight planes, qkv, z, and the live workspace must be
 * mutually non-overlapping.
 *
 * The policy-bearing form uses caller-owned call-scoped transient storage sized by
 * gdn_input_proj_workspace_capacity_bytes(). A16 requires zero bytes. The convenience overload
 * selects A16Only and requires no transient workspace.
 */
[[nodiscard]] std::size_t
gdn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                        std::int32_t input_rows, LinearPolicy policy,
                                        std::int32_t min_tokens, std::int32_t max_tokens);

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Applies the A16-only single-parent GDN projection without transient workspace.
 */
void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    cudaStream_t stream);

/**
 * Returns the transient capacity required by the registered two-parent Q4/Q5 or single-parent W8
 * snapshot profile. `batch_size` is exact and the query covers every W in the inclusive width
 * interval. B=1 preserves the format-specific fused/composed resolver. B=2..8 uses aggregate
 * projection plus one BF16 [C,B*W] projected plane. The query throws for an unregistered row
 * profile or unsupported B/W domain.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Returns the transient capacity for a registered [16384,5120] NVFP4 or row-scaled FP8 snapshot
 * profile. `batch_size` is exact and the query covers every W in the inclusive width interval.
 * B=1 preserves the format-specific fused/materialized resolver; B=2..8 covers its aggregate
 * projection mechanism plus any projected BF16 plane selected by the complete-Op plan.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Op: gdn_input_proj_conv_snapshot
 *
 * Math / indexing:
 *   For each row b, let p[:,j,b] be the concatenated q/k/value projection of x[:,j,b], while
 *   z[:,j,b] is the independent z projection. Starting from the BF16 width-three history selected
 *   by initial_state_slots[b], evaluate the width-four depthwise convolution over p, apply SiLU,
 *   and publish its channel ranges to query, key, and value. After valid column j, write the new
 *   width-three history to snapshot_base_slots[b]+j. Z bypasses convolution.
 *
 * Logical shapes:
 *   The 27B registered form has x [5120,W,B], Q4 q/k weight [4096,5120], one Q5 value/z parent
 *   [12288,5120], conv_weight [10240,4], conv_states [10240,3,Slots], query/key [2048,W,B],
 *   value/z [6144,W,B], and I32 selectors [B]. B=1 accepts every positive W; B=2..8 accepts
 *   W=1..16. `valid_columns` is empty for a dense invocation or I32 [B] for a mixed-width batch.
 *   A mixed-width invocation has B>=1 and every valid extent lies in [1,W].
 *
 * Numeric:
 *   The oracle exact-decodes packed weights and evaluates projection, convolution, SiLU, z, and
 *   every snapshot value naively in FP64 from represented inputs. BF16 query/key/value/z and
 *   snapshots are promoted and compared directly with those ideal values; their final storage
 *   rounding belongs to the Op's named A16 criterion, not the oracle. Former unfused projection
 *   tensors are not observable cast boundaries; production routes use their natural private
 *   accumulator and staging precision. This two-parent Q4/Q5 form does not quantize activation;
 *   the single-parent policy-bearing form below defines its own permitted compute profiles.
 *
 * Effects:
 *   Each row writes query/key/value through its valid prefix and exact zero to its invalid tail;
 *   z is projected for all B*W safe input columns. A row writes only its valid destination state
 *   prefix. The caller reserves disjoint complete [base,base+W) intervals, prevents one row from
 *   overwriting another row's initial slot, and may overlap a row's own initial slot with its
 *   destination after that initial history has been loaded. Other slots are unchanged. Newly
 *   projected convolution channels remain private to the call while published snapshots are BF16.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& qk_weight,
                                  const Weight& value_z_weight, const Tensor& conv_weight,
                                  Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream);

/**
 * Single-parent form of gdn_input_proj_conv_snapshot. Registered parents are W8G32_F16S RowSplit
 * [12288,2048], NVFP4 BlockScaleK16M128x4 [16384,5120], and FP8_E4M3FN_ROW_BF16S RowScale
 * [16384,5120], all in q/k/value/z row order. W8 admits A16Only, NVFP4 admits A16Only/AllowA4,
 * and FP8 admits A16Only/AllowA8. B=1 accepts every positive W for FP8; the batched domain is
 * B=2..8 and W=1..16. For FP8 B=1, A16 is fused at W=1..3 and W=7..10 and materialized
 * otherwise; AllowA8 uses the same winners through W=9 and A8 from W=10. Batched AllowA8 uses A8
 * when B*W>=9. Tensor operands, the complete FP8 parent, and live workspace must be mutually
 * non-overlapping, except that the read-only initial_state_slots and snapshot_base_slots selectors
 * may alias each other; same-row state-slot overlap remains governed by the snapshot state
 * contract.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& ws,
                                  cudaStream_t stream);

/**
 * Applies the A16-only single-parent form. FP8 accepts every positive W for dense B=1; the batched
 * domain is B=2..8 and W=1..16.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream);

/**
 * Returns the transient capacity for the registered Q4/Q5 or W8 record-producing profile.
 * `batch_size` is exact, and the inclusive T interval must lie within ReplaySSM's B=1..8,
 * T=2..16 execution domain. These profiles require no transient storage because materialized
 * projection writes directly to caller-owned conv_record.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Returns the transient capacity for a registered [16384,5120] NVFP4 or row-scaled FP8
 * record-producing profile. Fused and materialized A16 routes require no storage. AllowA4/AllowA8
 * returns only the activation-quantization workspace selected by this complete-Op route;
 * conv_record is caller-owned.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Op: gdn_input_proj_conv_record
 *
 * For each batch row, evaluates the registered projection, width-four causal convolution, SiLU,
 * and q/k/value/z split from the BF16 history selected by initial_state_slots. It writes each BF16
 * newest history column to conv_record [C,T,B]. Query, key,
 * and value are zero in each row's invalid tail; z is projected for every physical column.
 *
 * The execution domain is B=1..8 and T=2..16. valid_columns is empty for dense input or device
 * I32 [B], with each caller-supplied extent in [1,T]. conv_states is a read-only BF16 [C,3,S]
 * state-pool view, and initial_state_slots contains absolute slots in [0,S). Source state is not
 * modified. Only the valid prefix of conv_record is semantically defined. Outputs and valid
 * record columns are bit-identical to the corresponding snapshot execution from the same initial
 * history, inputs, physical T/B, format and policy; records equal its newest history columns.
 * A private convolution intermediate need not be rounded to BF16 before use.
 *
 * The two-parent form registers Q4 q/k [4096,5120] and the Q5 value/z parent [12288,5120]. All
 * tensor operands, outputs, conv_record, source state, and live workspace must be disjoint.
 */
void gdn_input_proj_conv_record(const Tensor& x, const Weight& qk_weight,
                                const Weight& value_z_weight, const Tensor& conv_weight,
                                const Tensor& conv_states, const Tensor& valid_columns,
                                const Tensor& initial_state_slots, Tensor& conv_record,
                                Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Single-parent record-producing form. Registered parents are W8G32_F16S [12288,2048], NVFP4
 * [16384,5120], and FP8_E4M3FN_ROW_BF16S [16384,5120]. W8 admits A16Only, NVFP4 admits
 * A16Only/AllowA4, and FP8 admits A16Only/AllowA8. Record and snapshot share arithmetic route
 * selection. Every tensor operand, the complete FP8 parent, and live workspace must be mutually
 * non-overlapping.
 */
void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream);

/** Applies the A16-only single-parent record-producing form. */
void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
