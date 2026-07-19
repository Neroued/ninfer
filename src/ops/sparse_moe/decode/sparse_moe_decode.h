#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class SparseMoeD1Schedule : std::uint8_t {
    RowCta4,
    RowCta8,
};

enum class SparseMoeD2Schedule : std::uint8_t {
    SerialControl,
    WarpRegister,
};

enum class SparseMoeD3Schedule : std::uint8_t {
    NineWarp,
    BalancedEightWarp,
};

enum class SparseMoeD4Schedule : std::uint8_t {
    NineWarpRows1,
    BalancedEightWarpRows4,
};

struct SparseMoeDecodePlan {
    SparseMoeD1Schedule d1      = SparseMoeD1Schedule::RowCta8;
    SparseMoeD2Schedule d2      = SparseMoeD2Schedule::WarpRegister;
    SparseMoeD3Schedule d3      = SparseMoeD3Schedule::NineWarp;
    SparseMoeD4Schedule d4      = SparseMoeD4Schedule::NineWarpRows1;
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

void sparse_moe_decode_launch_d1(const Tensor& x, const Weight& router_shared_gate,
                                 const SparseMoeDecodeWorkspace& workspace,
                                 SparseMoeD1Schedule schedule, cudaStream_t stream);
void sparse_moe_decode_launch_d2(const SparseMoeDecodeWorkspace& workspace,
                                 SparseMoeD2Schedule schedule, cudaStream_t stream);
void sparse_moe_decode_launch_d3(const Tensor& x, const SparseMoeWeights& weights,
                                 const SparseMoeDecodeWorkspace& workspace,
                                 SparseMoeD3Schedule schedule, cudaStream_t stream);
void sparse_moe_decode_launch_d4(const SparseMoeWeights& weights, Tensor& destination,
                                 const SparseMoeDecodeWorkspace& workspace,
                                 SparseMoeD4Schedule schedule, cudaStream_t stream);

void sparse_moe_decode_launch_d3_small_t(const Tensor& x, const SparseMoeWeights& weights,
                                         const int* token_ids, float* token_activations,
                                         std::int32_t tokens, cudaStream_t stream);
void sparse_moe_decode_launch_d4_small_t(const SparseMoeWeights& weights, Tensor& destination,
                                         const int* token_ids, const float* token_alpha,
                                         const float* shared_scale, const float* token_activations,
                                         std::int32_t tokens, cudaStream_t stream);
void sparse_moe_decode_launch(const Tensor& x, const SparseMoeWeights& weights, Tensor& destination,
                              const SparseMoeDecodeWorkspace& workspace,
                              const SparseMoeDecodePlan& plan, cudaStream_t stream);

} // namespace ninfer::ops::detail
