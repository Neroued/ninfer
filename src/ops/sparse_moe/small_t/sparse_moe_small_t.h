#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kSparseMoeSmallTMin = 2;
// The fixed frontier covers the measured Q4+Q5 crossover. S2 assigns at most
// 32 resident warps and lets them cover the remaining tokens in a second turn.
inline constexpr std::int32_t kSparseMoeSmallTMax = 44;

enum class SparseMoeSmallTD3Schedule : std::uint8_t {
    Paths1,
    Paths3,
    Paths9,
};

enum class SparseMoeSmallTD4Schedule : std::uint8_t {
    Rows1,
    Rows2,
    Rows4,
};

struct SparseMoeSmallTPlan {
    std::int32_t tokens                   = 0;
    std::size_t workspace_bytes           = 0;
    SparseMoeSmallTD3Schedule d3_schedule = SparseMoeSmallTD3Schedule::Paths3;
    SparseMoeSmallTD4Schedule d4_schedule = SparseMoeSmallTD4Schedule::Rows1;
};

struct SparseMoeSmallTWorkspace {
    Tensor token_ids;
    Tensor token_alpha;
    Tensor shared_scale;
    Tensor scratch;
};

template <class Arena>
SparseMoeSmallTWorkspace allocate_sparse_moe_small_t_workspace(Arena& arena, std::int32_t tokens) {
    SparseMoeSmallTWorkspace out;
    const std::int32_t assignments = 8 * tokens;
    out.token_ids                  = arena.alloc(DType::I32, {assignments}, 16);
    out.token_alpha                = arena.alloc(DType::FP32, {assignments}, 16);
    out.shared_scale               = arena.alloc(DType::FP32, {tokens}, 16);
    // S1 uses [T,257,4] partial router scores. After S2, each token reuses
    // its [9,512] region for eight routed and one shared SwiGLU activations.
    out.scratch = arena.alloc(DType::FP32, {512, 9 * tokens}, 256);
    return out;
}

[[nodiscard]] bool sparse_moe_uses_small_t(std::int32_t tokens) noexcept;
[[nodiscard]] std::size_t sparse_moe_small_t_workspace_bytes(std::int32_t tokens);
[[nodiscard]] SparseMoeSmallTPlan
resolve_sparse_moe_small_t_plan(std::int32_t tokens, QType routed_gate_up, QType routed_down);

void sparse_moe_small_t_launch_s1(const Tensor& x, const Weight& router_shared_gate,
                                  const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream);
void sparse_moe_small_t_launch_s2(const SparseMoeSmallTPlan& plan,
                                  const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream);
void sparse_moe_small_t_launch_s3(const Tensor& x, const SparseMoeWeights& weights,
                                  const SparseMoeSmallTPlan& plan,
                                  const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream);
void sparse_moe_small_t_launch_s4(const SparseMoeWeights& weights, Tensor& destination,
                                  const SparseMoeSmallTPlan& plan,
                                  const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream);

void sparse_moe_small_t_launch(const Tensor& x, const SparseMoeWeights& weights,
                               Tensor& destination, const SparseMoeSmallTPlan& plan,
                               const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
