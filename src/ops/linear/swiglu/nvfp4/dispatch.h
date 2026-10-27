#pragma once
#include "core/platform.h"

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>


#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {
struct Nvfp4A4Operands;

#if NINFER_TARGET_RTX5090
std::size_t nvfp4_swiglu_n28672_k5120_capacity(LinearPolicy policy, int min_tokens, int max_tokens);
void launch_nvfp4_swiglu_n28672_k5120(const Tensor& x, const Weight& w, Tensor& y,
                                       LinearPolicy policy, MemoryArena& workspace,
                                       cudaStream_t stream);
#endif

[[nodiscard]] std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                       std::int32_t min_tokens,
                                                                       std::int32_t max_tokens);

void launch_nvfp4_swiglu_decode(const Tensor& x, const Weight& weight, Tensor& out,
                                cudaStream_t stream);
void launch_nvfp4_swiglu_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                 cudaStream_t stream);
void launch_nvfp4_swiglu_a4(const Tensor& x, const Weight& weight, Tensor& out,
                            MemoryArena& workspace, cudaStream_t stream);

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, MemoryArena& workspace, cudaStream_t stream);

#if NINFER_TARGET_SPARK
std::size_t nvfp4_swiglu_n12288_k4096_capacity(LinearPolicy policy, int min_tokens, int max_tokens);
void launch_nvfp4_swiglu_n12288_k4096(const Tensor& x, const Weight& weight, Tensor& out,
                                      float limit, bool clamped, LinearPolicy policy,
                                      MemoryArena& workspace, cudaStream_t stream);
void launch_nvfp4_swiglu_n12288_k4096_tma(const Nvfp4A4Operands& operands, void* out, float limit,
                                          bool clamped, cudaStream_t stream);
#endif
} // namespace ninfer::ops::detail
