#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Returns the caller-owned transient capacity for every legal patch/segment pair in the supplied
 * inclusive envelope. A pair is legal when 1 <= segments <= patches. An envelope with no legal
 * pair throws.
 *
 * Zero is returned only when the selected route needs no transient storage: the descriptor-based
 * route needs none for a single segment, whereas a route that restages q/k/v for a different
 * kernel geometry always does. Callers must size from this function rather than assume either.
 */
[[nodiscard]] std::size_t vision_attention_workspace_capacity_bytes(std::int32_t min_patches,
                                                                    std::int32_t max_patches,
                                                                    std::int32_t min_segments,
                                                                    std::int32_t max_segments);

/**
 * Packed, non-causal multi-head attention. For each segment [begin,end), head h, and
 * t in that segment:
 *
 *   score[j]    = dot(q[:,h,t], k[:,h,j]) / sqrt(72), begin <= j < end
 *   ideal[:,h,t] = sum_j softmax(score)[j] * v[:,h,j].
 *
 * q/k/v are BF16 [72,16,P] with contiguous feature and head dimensions; token strides may be
 * padded. out is contiguous BF16 [72,16,P]. cu_seqlens is contiguous I32 [S+1], begins at 0,
 * ends at P, and is strictly increasing. The oracle evaluates `ideal` naively in FP64 from the
 * represented inputs. The BF16 out is promoted and compared directly with that result; output
 * storage rounding belongs to the Op's numerical criterion, not the oracle. Inputs and output are
 * mutually non-overlapping.
 *
 * The Op allocates required opaque tile descriptors from `workspace` for the duration of the call;
 * no capacity is consumed for a single segment. There is no persistent state side effect.
 */
void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens,
                      WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/**
 * Equal-segment overload of the same packed, non-causal attention operation. P must be divisible
 * by `segment_length`; consecutive ranges `[s*segment_length,(s+1)*segment_length)` are the
 * independent segments. The q/k/v/out shape, layout, dtype, alias, and numerical semantics are
 * identical to the cu_seqlens overload, and `workspace` is reported by the same
 * vision_attention_workspace_capacity_bytes() with segments = P / segment_length.
 *
 * Segment bounds are derived arithmetically rather than from a descriptor, so routes that tile
 * directly consume no capacity; routes that stage q/k/v for a different kernel geometry do.
 */
void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                      std::int32_t segment_length, WorkspaceArena& workspace, Tensor& out,
                      cudaStream_t stream);

} // namespace ninfer::ops
