#pragma once
#include "ninfer/ops/moe.h"
#include "ops/moe/routed_swiglu/cuda/contribution.h"
#include <cuda_bf16.h>
#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"

namespace ninfer::ops::detail {
struct Nvfp4A4Operands;
}

namespace ninfer::ops::detail::swiglu_ffn {
struct QuantizedProductOutput;
void launch_nvfp4_gu_quantized_tma(const Nvfp4A4Operands&, const QuantizedProductOutput&,
                                   float limit, cudaStream_t);
void launch_nvfp4_gu_quantized(const Tensor&, const Weight&, Nvfp4A4Workspace, Nvfp4ScaleLayout,
                               float down_divisor, float limit, MemoryArena&, cudaStream_t);
void launch_nvfp4_quantized_ffn(const Tensor&, const SwiGLUWeights&, Tensor&, float limit,
                                MemoryArena&, cudaStream_t);

void launch_nvfp4_down(const float* product, const Weight& down, Tensor& output, int tokens,
                       MemoryArena& workspace, cudaStream_t stream);
void launch_a16_ffn_join(const Tensor&, const SwiGLUWeights&, Tensor&,
                         RoutedSwiGLUContribution routed, float limit, MemoryArena&, cudaStream_t);
void launch_q8_ffn(const Tensor&, const SwiGLUWeights&, Tensor&, float limit, MemoryArena&,
                   cudaStream_t);
void launch_q8_ffn_join(const Tensor&, const SwiGLUWeights&, Tensor&,
                        RoutedSwiGLUContribution routed, float limit, MemoryArena&, cudaStream_t);
void launch_bf16_ffn(const Tensor&, const SwiGLUWeights&, Tensor&, float limit, MemoryArena&,
                     cudaStream_t);
void launch_nvfp4_gu_float(const Tensor&, const Weight&, float*, float limit, MemoryArena&,
                           cudaStream_t);
void launch_bf16_gu_float(const Tensor&, const Weight&, float*, float limit, cudaStream_t);
void launch_bf16_gu_compensated(const Tensor&, const Weight&, __nv_bfloat16*, float limit,
                                cudaStream_t);
void launch_nvfp4_down_tma128(const Weight&, Tensor&, Nvfp4A4Workspace, int tokens, cudaStream_t);
void launch_nvfp4_down_tma256(const Weight&, Tensor&, Nvfp4A4Workspace, int tokens, cudaStream_t);

void launch_bf16_ffn_join(const Tensor&, const SwiGLUWeights&, Tensor&,
                          RoutedSwiGLUContribution routed, float limit, MemoryArena&, cudaStream_t);

} // namespace ninfer::ops::detail::swiglu_ffn
