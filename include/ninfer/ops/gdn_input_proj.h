#pragma once

// ninfer::ops - fused GDN Q/K/V input projections into one contiguous output.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Validates a registered problem and positive token capacity. The admitted row pairs are
 * (4096,6144) for the 27B Q4/Q5 route and (8192,4096) for the 35B W8 QKV/Z route. Every route
 * writes directly to final outputs, so the required transient capacity is zero. `max_tokens` is
 * a capacity query, not an execution limit.
 */
[[nodiscard]] std::size_t gdn_input_proj_workspace_bytes(std::int32_t qk_rows,
                                                         std::int32_t value_rows,
                                                         std::int32_t max_tokens);

/**
 * Op: gdn_input_proj
 *
 * Math / indexing:
 *   qkv[:,t] = concat(qk_weight * x[:,t], v_weight * x[:,t]) for every token t.
 *
 * Logical shapes:
 *   x [5120,T], qk weight/output rows 4096, v weight/output rows 6144, qkv [10240,T].
 *   T may be any positive value.
 *   x and qkv are contiguous BF16. qk_weight is Q4G64_F16S RowSplit [4096,5120] and
 *   v_weight is Q5G64_F16S RowSplit [6144,5120], both with FP16 scales.
 *
 * Numeric:
 *   The oracle exact-decodes both weights and evaluates both projections naively in FP64 before
 *   converting the observable concatenated output to BF16. Production routes may choose their
 *   private precision independently; every registered route writes the final qkv allocation
 *   directly.
 *
 * Effects:
 *   Writes the full qkv output; inputs and output must not alias.
 *
 * Workspace:
 *   No transient bytes are required. The retained arena boundary is caller-owned and is not used.
 */
void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& v_weight, Tensor& qkv,
                    WorkspaceArena& ws, cudaStream_t stream);

/**
 * Qwen3.6-35B W8 specialization. The one W8G32_F16S RowSplit parent has shape [12288,2048]
 * and stored row order [query 2048, key 2048, value 4096, z 4096]. `x` is contiguous BF16
 * [2048,T], qkv is contiguous BF16 [8192,T] in the exact causal-convolution channel order, and z
 * is an independent contiguous BF16 [4096,T] output. Every route writes both allocations directly
 * and requires no transient workspace. T may be any positive value.
 */
void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    WorkspaceArena& ws, cudaStream_t stream);

/**
 * Returns the transient capacity required by gdn_input_proj_conv_snapshot. The 35B W8 route writes
 * final outputs from projection epilogues for exact T=1..16; the 27B Q4/Q5 route may reserve
 * private BF16 staging for its small-T kernels. Extents outside a target route's optimized range
 * use the composed implementation and require two BF16 [C,T] intermediates.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_snapshot_workspace_bytes(std::int32_t query_rows,
                                                                       std::int32_t key_rows,
                                                                       std::int32_t value_rows,
                                                                       std::int32_t max_tokens);

/**
 * Op: gdn_input_proj_conv_snapshot
 *
 * Math / indexing:
 *   Let p[:,t] be concat(qk_weight*x[:,t], value_weight*x[:,t]). Starting from the BF16
 *   width-three history selected by device I32 initial_slot, evaluate the width-four depthwise
 *   convolution over p, apply SiLU, and write its three channel ranges directly to query, key,
 *   and value. After token t, write the resulting width-three projection history to state slot t.
 *
 * Logical shapes:
 *   The 27B registered form has x [5120,T], Q4 q/k weight [4096,5120], Q5 value weight
 *   [6144,5120], conv_weight [10240,4], conv_states [10240,3,Slots], query/key [2048,T],
 *   and value [6144,T]. T is positive, Slots>=T, and the device initial_slot value is in
 *   [0,Slots).
 *
 * Numeric:
 *   The oracle exact-decodes packed weights, evaluates projection, convolution, and SiLU naively
 *   in FP64 from represented inputs, then converts query/key/value and snapshots to BF16. Former
 *   unfused qkv tensors are not observable cast boundaries; production routes use their natural
 *   private accumulator and staging precision. Activation values are not quantized.
 *
 * Effects:
 *   Writes query/key/value and state slots [0,T); other slots are unchanged. Newly projected
 *   values remain private to the current call while each published snapshot is BF16.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& qk_weight,
                                  const Weight& value_weight, const Tensor& conv_weight,
                                  Tensor& conv_states, const Tensor& initial_slot, Tensor& query,
                                  Tensor& key, Tensor& value, WorkspaceArena& ws,
                                  cudaStream_t stream);

/**
 * W8 registered form of gdn_input_proj_conv_snapshot. The parent W8G32_F16S RowSplit weight is
 * [12288,2048] in q/k/value/z row order. The convolution has 8192 channels; z [4096,T] is the
 * direct BF16 projection output and does not participate in convolution or state updates.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& initial_slot, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream);

} // namespace ninfer::ops
