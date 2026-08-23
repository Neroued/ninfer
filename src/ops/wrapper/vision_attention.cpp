#include "ninfer/ops/vision_attention.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/launcher/vision_attention.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 72;
constexpr std::int32_t kHeads   = 16;

std::int32_t scratch_tiles(std::int32_t patches, std::int32_t segments) {
    if (segments == 1) { return 0; }
    constexpr std::int64_t tile_rows = 64;
    const std::int64_t tiles =
        (static_cast<std::int64_t>(patches) + tile_rows - 1) / tile_rows + segments - 1LL;
    if (tiles > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("vision_attention: descriptor tile count exceeds int32");
    }
    return static_cast<std::int32_t>(tiles);
}

template <class Allocator>
Tensor allocate_workspace(Allocator& allocator, std::int32_t patches, std::int32_t segments) {
    const std::int32_t tiles = scratch_tiles(patches, segments);
    return tiles == 0 ? Tensor{} : allocator.alloc(DType::I32, {4, tiles});
}

#ifdef NINFER_VOLTA_BUILD
struct VisionFlashWorkspace {
    Tensor q_f32;
    Tensor k_f16;
    Tensor v_f16;
    Tensor out_f32;
    Tensor dst_meta;
};

// Staging for the Volta flash route: Q/K/V re-laid-out at the kernel's padded
// head dim, and the FP32 output the kernel writes before it is narrowed back to
// BF16. Every term is linear in the patch count -- segments are separate
// launches, so there is no mask, and the kernel clamps its last KV iteration to
// ne11, so the key extent is not padded either.
template <class Allocator>
VisionFlashWorkspace allocate_vision_flash_workspace(Allocator& allocator, std::int32_t patches) {
    constexpr std::int32_t d       = detail::kVisionFlashPaddedHeadDim;
    const std::int32_t staged_rows = detail::vision_attention_volta_flash_staged_rows(patches);

    return {
        allocator.alloc(DType::FP32, {d, kHeads, staged_rows}),
        allocator.alloc(DType::FP16, {d, kHeads, patches}),
        allocator.alloc(DType::FP16, {d, kHeads, patches}),
        allocator.alloc(DType::FP32, {d, kHeads, staged_rows}),
        allocator.alloc(DType::FP32,
                        {2, static_cast<std::int32_t>(
                                detail::vision_attention_volta_flash_meta_elements(patches))}),
    };
}

// Host-side segment bounds, which the per-segment launch geometry needs.
//
// The uniform overload has them arithmetically and a single segment is trivially
// {0, P}, so the only case that reads back from the device is a genuinely ragged
// cu_seqlens -- which the Qwen3.6 frontend never produces (segments are video
// frames of one grid, VisionItemControl::segment_length), leaving this to the op
// tests and any future ragged producer.
std::vector<std::int32_t> host_segment_bounds(const Tensor& cu_seqlens, std::int32_t segments,
                                              std::int32_t patches, cudaStream_t stream) {
    if (segments == 1) { return {0, patches}; }
    std::vector<std::int32_t> bounds(static_cast<std::size_t>(segments) + 1);
    CUDA_CHECK(cudaMemcpyAsync(bounds.data(), cu_seqlens.data,
                               bounds.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (bounds.front() != 0 || bounds.back() != patches) {
        throw std::invalid_argument("vision_attention: cu_seqlens must span [0, P]");
    }
    for (std::size_t i = 1; i < bounds.size(); ++i) {
        if (bounds[i] <= bounds[i - 1]) {
            throw std::invalid_argument("vision_attention: cu_seqlens must be strictly ascending");
        }
    }
    return bounds;
}

std::vector<std::int32_t> uniform_segment_bounds(std::int32_t patches,
                                                 std::int32_t segment_length) {
    std::vector<std::int32_t> bounds;
    bounds.reserve(static_cast<std::size_t>(patches / segment_length) + 1);
    for (std::int32_t begin = 0; begin <= patches; begin += segment_length) {
        bounds.push_back(begin);
    }
    return bounds;
}
#endif // NINFER_VOLTA_BUILD

void require_qkv(const Tensor& tensor, std::int32_t patches, const char* name) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != kHeadDim || tensor.ne[1] != kHeads ||
        tensor.ne[2] != patches || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("vision_attention: invalid ") + name + " shape");
    }
    constexpr std::int64_t elem = 2;
    if (tensor.nb[0] != elem || tensor.nb[1] != elem * kHeadDim ||
        tensor.nb[2] < elem * kHeadDim * kHeads || (tensor.nb[2] % elem) != 0) {
        throw std::invalid_argument(std::string("vision_attention: invalid ") + name + " strides");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("vision_attention: ") + name +
                                    " data must be non-null");
    }
}

} // namespace

std::size_t vision_attention_workspace_capacity_bytes(std::int32_t min_patches,
                                                      std::int32_t max_patches,
                                                      std::int32_t min_segments,
                                                      std::int32_t max_segments) {
    if (min_patches <= 0 || max_patches < min_patches || min_segments <= 0 ||
        max_segments < min_segments || min_segments > max_patches) {
        throw std::invalid_argument("vision_attention workspace: invalid execution envelope");
    }
    const std::int32_t segments = std::min(max_segments, max_patches);
    WorkspaceLayoutBuilder layout;
#ifdef NINFER_VOLTA_BUILD
    // The Volta route replaces the descriptor path outright rather than adding to
    // it, so the query must mirror that exactly: no tile descriptors, staging
    // only. Reporting both would over-declare and break the high-water contract
    // the caller checks.
    (void)segments;
    (void)allocate_vision_flash_workspace(layout, max_patches);
#else
    (void)allocate_workspace(layout, max_patches, segments);
#endif // NINFER_VOLTA_BUILD
    return layout.peak_bytes(1);
}

void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens,
                      WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    const std::int32_t patches  = q.ne[2];
    const std::int32_t segments = cu_seqlens.ne[0] - 1;
    if (patches <= 0) { throw std::invalid_argument("vision_attention: P must be positive"); }
    require_qkv(q, patches, "q");
    require_qkv(k, patches, "k");
    require_qkv(v, patches, "v");
    require_qkv(out, patches, "out");
    if (!out.is_contiguous()) {
        throw std::invalid_argument("vision_attention: out must be contiguous");
    }
    if (cu_seqlens.dtype != DType::I32 || segments <= 0 || segments > patches ||
        cu_seqlens.ne[1] != 1 || cu_seqlens.ne[2] != 1 || cu_seqlens.ne[3] != 1 ||
        !cu_seqlens.is_contiguous() || cu_seqlens.data == nullptr) {
        throw std::invalid_argument("vision_attention: cu_seqlens must be contiguous I32 [S+1]");
    }
    auto scratch_scope = workspace.scope();
#ifdef NINFER_VOLTA_BUILD
    // The Ampere+ kernel is ldmatrix/mma-based and traps below sm_80, so this is
    // the only vision attention path on Volta rather than a faster alternative.
    VisionFlashWorkspace staging = allocate_vision_flash_workspace(workspace, patches);
    const std::vector<std::int32_t> bounds =
        host_segment_bounds(cu_seqlens, segments, patches, stream);
    detail::vision_attention_volta_flash_launch(q, k, v, bounds.data(), segments, staging.q_f32,
                                                staging.k_f16, staging.v_f16, staging.out_f32,
                                                staging.dst_meta, out, stream);
#else
    Tensor tiles      = allocate_workspace(workspace, patches, segments);
    Tensor* tiles_ptr = tiles.data == nullptr ? nullptr : &tiles;
    detail::vision_attention_launch(q, k, v, cu_seqlens, tiles_ptr, out, stream);
#endif // NINFER_VOLTA_BUILD
}

void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                      std::int32_t segment_length, WorkspaceArena& workspace, Tensor& out,
                      cudaStream_t stream) {
    const std::int32_t patches = q.ne[2];
    if (patches <= 0) { throw std::invalid_argument("vision_attention: P must be positive"); }
    require_qkv(q, patches, "q");
    require_qkv(k, patches, "k");
    require_qkv(v, patches, "v");
    require_qkv(out, patches, "out");
    if (!out.is_contiguous()) {
        throw std::invalid_argument("vision_attention: out must be contiguous");
    }
    if (segment_length <= 0 || patches % segment_length != 0) {
        throw std::invalid_argument("vision_attention: invalid uniform segment length");
    }
#ifdef NINFER_VOLTA_BUILD
    auto scratch_scope           = workspace.scope();
    VisionFlashWorkspace staging = allocate_vision_flash_workspace(workspace, patches);
    const std::vector<std::int32_t> bounds = uniform_segment_bounds(patches, segment_length);
    detail::vision_attention_volta_flash_launch(q, k, v, bounds.data(),
                                                patches / segment_length, staging.q_f32,
                                                staging.k_f16, staging.v_f16, staging.out_f32,
                                                staging.dst_meta, out, stream);
#else
    (void)workspace;
    detail::vision_attention_uniform_launch(q, k, v, segment_length, out, stream);
#endif // NINFER_VOLTA_BUILD
}

} // namespace ninfer::ops
