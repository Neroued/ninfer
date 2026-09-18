#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

/**
 * Fused NVFP4 LinearSwiGLU over the W4A4 TMA route, for the 34816x5120 geometry. tokens may be any
 * positive count; the last M tile may be partial.
 *
 * activation_scales must be the tiled layout written over whole M tiles, that is a plane of
 * nvfp4_w4a4_padded_tokens(tokens) rows with the padding zeroed - what
 * allocate_nvfp4_w4a4_workspace sizes and launch_nvfp4_w4a4_quantize fills and checks. The
 * descriptor this builds covers that whole plane, so a caller that sizes the plane at tokens
 * instead has it read past the end.
 */
void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t* activation_codes,
                                         const std::uint8_t* activation_scales,
                                         const std::uint8_t* weight_codes,
                                         const std::uint8_t* weight_scales, __nv_bfloat16* output,
                                         std::int32_t tokens, float alpha, cudaStream_t stream);

} // namespace ninfer::ops::detail
