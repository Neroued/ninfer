#pragma once

#include "ninfer/ops/block_attn_res.h"
#include "ninfer/ops/ragged.h"
#include "core/record_view.h"

namespace ninfer::ops {

/**
 * Residual-scaled causal depthwise width-4 convolution with oldest tap first.
 * Initial history supplies u[c,-3..-1]. Input x supplies u[c,t] for t >= 0.
 *
 *   y_conv[c,t] = RN_BF16((sum_k(weight[c,k] * u[c,t-3+k]) + u[c,t]) / sqrt(5))
 *
 * The sum and scale use FP32. With addend, conv_out stores
 * RN_BF16((FP32(residual) + FP32(addend)) + FP32(y_conv)). An empty residual is
 * omitted from the sum. With empty addend, residual must also be empty and conv_out
 * stores y_conv directly. Residual may exactly alias conv_out.
 *
 * Batched operands x, conv_out, residual and addend are contiguous BF16 [C,W,B].
 * valid_columns is empty for dense input or Device I32 [B] with values V_b in [0,W].
 * Columns j >= V_b receive BF16 zero. A row with V_b == 0 skips all state offsets,
 * snapshot bases, history, x, residual and addend and has no state writes.
 * Direct form stores the last three inputs of the sequence window after V_b steps.
 * Snapshot form stores the window after live column j in snapshot_base_slots[b] + j.
 * snapshots is contiguous BF16 [C,3,slots] and snapshot_base_slots is Device I32 [B].
 *
 * Ragged operands x, conv_out, residual and addend are contiguous BF16 [C,N].
 * Each segment is convolved independently and stores its trailing window.
 *
 * weight is contiguous BF16 [C,4]. Each state view selects a contiguous BF16 [C,3]
 * field via Device I64 byte offsets, one per batch row or ragged segment. State views
 * may exactly alias for the same sequence. Each live snapshot row reserves W slots
 * and its source may exactly alias one complete slot in that reservation. A masked
 * batch with no live rows can supply a single slot.
 * The caller guarantees valid device metadata, in-bounds offsets and disjoint writes
 * across sequences. Accessed state fields are disjoint from other operands except
 * for the documented state aliases.
 *
 * All non-empty operands belong to execution.device(). State fields and snapshots
 * require 2-byte alignment. For C divisible by 8, x, weight, residual, addend and
 * conv_out require 16-byte alignment. Zero caller workspace. CUDA Graph capture and
 * replay are supported.
 */
void short_conv_residual(const Tensor& x, const Tensor& weight, ConstStateFieldView state_in,
                         StateFieldView state_out, const Tensor& valid_columns,
                         const Tensor& residual, const Tensor& addend, Tensor& conv_out,
                         CudaExecutionView execution);

void short_conv_residual_snapshot(const Tensor& x, const Tensor& weight,
                                  ConstStateFieldView state_in, Tensor& snapshots,
                                  const Tensor& valid_columns, const Tensor& snapshot_base_slots,
                                  const Tensor& residual, const Tensor& addend, Tensor& conv_out,
                                  CudaExecutionView execution);

void short_conv_residual(const Tensor& x, const Tensor& weight, const RaggedSequenceView& ragged,
                         ConstStateFieldView state_in, StateFieldView state_out,
                         const Tensor& residual, const Tensor& addend, Tensor& conv_out,
                         CudaExecutionView execution);

/**
 * Fused short_conv_residual + block_attn_res:
 *   1. Executes short_conv_residual, publishing BF16 conv_out and updating state_out / snapshots.
 *   2. Feeds the exact published BF16 conv_out as `partial` into block_attn_res(stack, conv_out,
 *      attnres_addend, queries, norm_weight, params, mixed_out), without re-reading conv_out from
 * HBM.
 *
 * Shapes:
 *   - Batched: x/conv_out/conv_residual/conv_addend/attnres_addend are BF16 [D, W, B],
 *     stack is empty or BF16 [D, W*B, N_stack] and mixed_out is BF16/FP32 [D, W*B, Q],
 *     where column m = b*W + j.
 *   - Ragged: x/conv_out/conv_residual/conv_addend/attnres_addend are BF16 [D, N],
 *     stack is empty or BF16 [D, N, N_stack], mixed_out is BF16/FP32 [D, N, Q].
 *   - Inactive batched columns (j >= valid_columns[b]) write exact zero to both conv_out and
 *     mixed_out, and do not read stack/attnres_addend or mutate conv state.
 *   - conv_residual may exactly alias conv_out. mixed_out is disjoint from inputs and conv_out.
 *   - Zero caller workspace; CUDA Graph capture/replay safe.
 */
void short_conv_block_attn_res(const Tensor& x, const Tensor& conv_weight,
                               ConstStateFieldView state_in, StateFieldView state_out,
                               const Tensor& valid_columns, const Tensor& conv_residual,
                               const Tensor& conv_addend, const Tensor& stack,
                               const Tensor& attnres_addend, const Tensor& queries,
                               const Tensor& norm_weight, const BlockAttnResParameters& params,
                               Tensor& conv_out, Tensor& mixed_out, CudaExecutionView execution);

void short_conv_block_attn_res_snapshot(
    const Tensor& x, const Tensor& conv_weight, ConstStateFieldView state_in, Tensor& snapshots,
    const Tensor& valid_columns, const Tensor& snapshot_base_slots, const Tensor& conv_residual,
    const Tensor& conv_addend, const Tensor& stack, const Tensor& attnres_addend,
    const Tensor& queries, const Tensor& norm_weight, const BlockAttnResParameters& params,
    Tensor& conv_out, Tensor& mixed_out, CudaExecutionView execution);

void short_conv_block_attn_res(const Tensor& x, const Tensor& conv_weight,
                               const RaggedSequenceView& ragged, ConstStateFieldView state_in,
                               StateFieldView state_out, const Tensor& conv_residual,
                               const Tensor& conv_addend, const Tensor& stack,
                               const Tensor& attnres_addend, const Tensor& queries,
                               const Tensor& norm_weight, const BlockAttnResParameters& params,
                               Tensor& conv_out, Tensor& mixed_out, CudaExecutionView execution);

} // namespace ninfer::ops
