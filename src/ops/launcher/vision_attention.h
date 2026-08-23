#pragma once

#include "core/tensor.h"

#include <cstddef>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void vision_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& cu_seqlens, Tensor* tiles, Tensor& out,
                             cudaStream_t stream);

std::int32_t vision_attention_uniform_tile(std::int32_t segment_length);

void vision_attention_uniform_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     std::int32_t segment_length, Tensor& out, cudaStream_t stream);

void vision_attention_uniform_launch_with_tile(const Tensor& q, const Tensor& k, const Tensor& v,
                                               std::int32_t segment_length, std::int32_t tile_size,
                                               Tensor& out, cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
// Volta (sm_70) vision attention through the vendored llama.cpp flash-attention
// kernel. Defined in vision_attention_volta_flash.cu, the only vision translation
// unit that sees the vendored headers. See the V100 performance summary.
//
// The staging is linear in the patch count: Q is padded to head_dim 128 in FP32,
// K/V in FP16. Segments are one launch each rather than a block-diagonal mask
// over a single launch, so there is no O(P^2) term.
//
// Queries are processed in blocks of at most kVisionFlashQBlockPatches, which
// caps the two FP32 planes (Q and the kernel's output) independently of the
// patch count -- they are 8 KiB per patch each, twice what the FP16 K/V cost, so
// at the frontend's patch ceiling the block bounds ~2 GiB of reservation. K/V
// stay whole, because a query block attends its entire segment.
inline constexpr std::int32_t kVisionFlashPaddedHeadDim = 128;
inline constexpr std::int32_t kVisionFlashQBlockPatches = 8192;

// Rows of Q/output staging a launch needs: the query block bound, or the whole
// item when it is smaller.
std::int32_t vision_attention_volta_flash_staged_rows(std::int32_t patches);

std::size_t vision_attention_volta_flash_meta_elements(std::int32_t patches);

// segment_bounds is a HOST array of segments+1 ascending offsets, starting at 0
// and ending at q.ne[2] -- the same content cu_seqlens carries on the device. The
// launch geometry is per segment, so the bounds have to be readable here.
void vision_attention_volta_flash_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const std::int32_t* segment_bounds, std::int32_t segments,
                                         Tensor& q_f32, Tensor& k_f16, Tensor& v_f16,
                                         Tensor& out_f32, Tensor& dst_meta, Tensor& out,
                                         cudaStream_t stream);
#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
