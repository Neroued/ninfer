#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

#include "core/device.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * @brief Permitted private activation-compute profiles for a linear projection.
 *
 * The policy constrains private route selection; it does not select a kernel or prescribe a
 * particular MMA instruction. The public activation and output tensors remain BF16 for every
 * policy.
 */
enum class LinearPolicy : std::uint8_t {
    A16Only, ///< Admit only A16 compute profiles.
    AllowA8, ///< Admit either A16 or A8 compute profiles.
    AllowA4, ///< Admit A16, A8 or A4 compute profiles.
};

[[nodiscard]] constexpr bool valid_linear_policy(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8 ||
           policy == LinearPolicy::AllowA4;
}

[[nodiscard]] constexpr bool allows_a8(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA8 || policy == LinearPolicy::AllowA4;
}

[[nodiscard]] constexpr bool allows_a4(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA4;
}

/**
 * Returns the caller-owned transient capacity required by Linear for every T in the inclusive
 * `[min_tokens,max_tokens]` interval. Invalid registered profiles, policies, or intervals throw;
 * a legal route that requires no transient storage returns zero.
 */
[[nodiscard]] std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                          std::int32_t input_rows,
                                                          LinearPolicy policy,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

/**
 * Y[n,t] = sum_k decode(W)[n,k] * X[k,t].
 *
 * X is BF16 [K,T], W represents [N,K], and Y is BF16 [N,T]. Activations and output are
 * contiguous, non-null and 16-byte aligned; T is positive. Output overlaps neither X nor
 * any weight plane. Inputs are preserved. There is no bias, activation or transpose.
 *
 * Supports registered weight geometries and column domains for contiguous BF16, RowSplit
 * Q4/Q5/Q6-G64 and Q8-G32 with FP16 scales, row-scaled FP8, and block-scaled NVFP4.
 * Unsupported format/geometry/layout combinations throw.
 *
 * policy permits private A16/A8/A4 computation; input and output remain BF16. Permission
 * does not require activation quantization. Workspace is caller-owned transient storage,
 * sized by linear_workspace_capacity_bytes(), disjoint from all operands. No hidden allocation.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            MemoryArena& workspace, CudaExecutionView execution);

/** A16-only projection with the same operand requirements and no transient workspace. */
void linear(const Tensor& x, const Weight& w, Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops
