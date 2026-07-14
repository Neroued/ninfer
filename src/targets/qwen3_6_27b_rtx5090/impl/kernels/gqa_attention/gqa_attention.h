#pragma once

// ninfer::kernels - grouped-query full attention with folded KV append.

#include "targets/qwen3_6_27b_rtx5090/impl/state/kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::kernels {

using targets::qwen3_6_27b_rtx5090::detail::KVCache;
using targets::qwen3_6_27b_rtx5090::detail::kKvQuantGroup;

[[nodiscard]] std::size_t gqa_attention_workspace_bytes(std::int32_t tokens);

void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   float scale, KVCache& kv, int layer, WorkspaceArena& ws, Tensor& out,
                   cudaStream_t stream);

void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions, KVCache& kv,
                   int layer, cudaStream_t stream);

void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale, KVCache& kv,
                          int layer, Tensor& out, cudaStream_t stream);

} // namespace ninfer::kernels
