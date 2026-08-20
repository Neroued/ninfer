#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class SparseMoeSmallTD3Schedule : std::uint8_t;
enum class SparseMoeSmallTD4Schedule : std::uint8_t;

struct SparseMoeDecodePlan {
    std::size_t workspace_bytes = 0;
};

struct SparseMoeDecodeWorkspace {
    Tensor ids;
    Tensor alpha;
    Tensor shared_scale;
    Tensor scratch;
};

template <class Arena>
SparseMoeDecodeWorkspace allocate_sparse_moe_decode_workspace(Arena& arena) {
    SparseMoeDecodeWorkspace out;
    out.ids          = arena.alloc(DType::I32, {8}, 16);
    out.alpha        = arena.alloc(DType::FP32, {8}, 16);
    out.shared_scale = arena.alloc(DType::FP32, {1}, 4);
    // D1 uses the first 257 values as scores. D3 then reuses the same lifetime for [9,512]
    // natural FP32 SwiGLU results consumed by D4.
    out.scratch = arena.alloc(DType::FP32, {9, 512}, 256);
    return out;
}

[[nodiscard]] std::size_t sparse_moe_decode_workspace_bytes();
[[nodiscard]] SparseMoeDecodePlan resolve_sparse_moe_decode_plan(QType routed_gate_up,
                                                                 QType routed_down);

void sparse_moe_decode_launch_d3_small_t(const Tensor& x, const SparseMoeWeights& weights,
                                         const int* token_ids, float* token_activations,
                                         std::int32_t tokens, SparseMoeSmallTD3Schedule schedule,
                                         cudaStream_t stream,
                                         const int* adaptive_route_jobs = nullptr);
void sparse_moe_decode_launch_d4_small_t(const SparseMoeWeights& weights, Tensor& destination,
                                         const int* token_ids, const float* token_alpha,
                                         const float* shared_scale, const float* token_activations,
                                         std::int32_t tokens, SparseMoeSmallTD4Schedule schedule,
                                         cudaStream_t stream,
                                         const int* adaptive_route_jobs = nullptr);
void sparse_moe_decode_launch(const Tensor& x, const SparseMoeWeights& weights, Tensor& destination,
                              const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream,
                              const void* prefetch_data = nullptr, std::size_t prefetch_bytes = 0);

} // namespace ninfer::ops::detail
