#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

struct LoopMessageWeights {
    Tensor down; // Contiguous BF16 [2*D, r, S], one [r, 2*D] row-major matrix per stream.
    Tensor up;   // Contiguous BF16 [r, 2*D, S], gain rows followed by forget rows per stream.
};

// Exact scratch high-water mark over the inclusive interval, from a 16-byte-aligned
// cursor. Public operands are excluded. Entry points reject unsupported build targets,
// invalid geometry, or unrepresentable operand extents.
[[nodiscard]] std::size_t
loop_message_blend_rmsnorm_workspace_capacity_bytes(std::int32_t hidden, std::int32_t rank,
                                                    std::int32_t streams, std::int32_t min_tokens,
                                                    std::int32_t max_tokens);

/**
 * Each token and stream computes:
 *   pair = RN_BF16(concat(rmsnorm(dest, pair_eps), rmsnorm(message, pair_eps)))
 *   h = RN_BF16(down @ pair)
 *   gain, forget = tanhf(up @ h) with FP32 logits
 *   blend = (1 - forget) * dest + gain * message with separate FP32 operations
 *   out = RN_BF16(rmsnorm(blend, norm_eps) * FP32(norm_weight)).
 * The RMSNorms inside the formula are weightless.
 *
 * Contiguous FP32 dest [D,M,S_dest] and message [D,M,S], BF16 norm_weight
 * [D,S_norm] and out [D,M,S]. S_dest and S_norm each independently equal 1
 * (broadcast) or S. D and r are positive multiples of 8 and 16 respectively.
 * M and S are positive. Epsilons are finite and positive. All bases are 16-byte
 * aligned and on execution.device(). Read-only inputs may overlap. The output
 * and live scratch are disjoint from all other operands.
 * Borrows workspace on execution.stream() without device allocation or synchronization.
 * The arena cursor is restored on return. Supports CUDA Graph capture/replay.
 */
void loop_message_blend_rmsnorm(const Tensor& dest, const Tensor& message,
                                const LoopMessageWeights& weights, const Tensor& norm_weight,
                                float pair_eps, float norm_eps, MemoryArena& workspace, Tensor& out,
                                CudaExecutionView execution);

} // namespace ninfer::ops
