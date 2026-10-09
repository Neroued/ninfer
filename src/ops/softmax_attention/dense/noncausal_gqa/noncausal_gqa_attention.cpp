#include "ninfer/ops/softmax_attention.h"

#include "core/layout.h"
#include "ops/softmax_attention/dense/noncausal_gqa/launch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim    = 256;
constexpr std::int32_t kQueryHeads = 24;
constexpr std::int32_t kKVHeads    = 4;
constexpr float kExpectedScale     = 0.0625f; // 1/sqrt(256)

void require_profile(AttentionHeadGeometry geometry, float scale, const char* op) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kHeadDim ||
        geometry.query_heads != kQueryHeads || geometry.kv_heads != kKVHeads) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-7f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
}

std::int32_t scratch_tiles(std::int32_t tokens, std::int32_t segments) {
    if (segments == 1) { return 0; }
    constexpr std::int64_t tile_rows = 64;
    const std::int64_t tiles =
        (static_cast<std::int64_t>(tokens) + tile_rows - 1) / tile_rows + segments - 1LL;
    if (tiles > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("noncausal_gqa_attention: descriptor tile count exceeds int32");
    }
    return static_cast<std::int32_t>(tiles);
}

template <class Allocator>
Tensor allocate_workspace(Allocator& allocator, std::int32_t tokens, std::int32_t segments) {
    const std::int32_t tiles = scratch_tiles(tokens, segments);
    return tiles == 0 ? Tensor{} : allocator.alloc(DType::I32, {4, tiles});
}

void require_qkv(const Tensor& tensor, std::int32_t tokens, std::int32_t heads, const char* op,
                 const char* name) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != kHeadDim || tensor.ne[1] != heads ||
        tensor.ne[2] != tokens || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name + " shape");
    }
    constexpr std::int64_t elem = 2;
    if (tensor.nb[0] != elem || tensor.nb[1] != elem * kHeadDim ||
        tensor.nb[2] < elem * kHeadDim * heads || (tensor.nb[2] % elem) != 0) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name + " strides");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

std::int32_t validate_qkv(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out,
                          AttentionHeadGeometry geometry, float scale, const char* op) {
    require_profile(geometry, scale, op);
    const std::int32_t tokens = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_qkv(q, tokens, kQueryHeads, op, "q");
    require_qkv(k, tokens, kKVHeads, op, "k");
    require_qkv(v, tokens, kKVHeads, op, "v");
    require_qkv(out, tokens, kQueryHeads, op, "out");
    if (!out.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": out must be contiguous");
    }
    return tokens;
}

} // namespace

std::size_t noncausal_gqa_attention_workspace_capacity_bytes(AttentionHeadGeometry geometry,
                                                             std::int32_t min_tokens,
                                                             std::int32_t max_tokens,
                                                             std::int32_t min_segments,
                                                             std::int32_t max_segments) {
    require_profile(geometry, kExpectedScale, "noncausal_gqa_attention workspace");
    if (min_tokens <= 0 || max_tokens < min_tokens || min_segments <= 0 ||
        max_segments < min_segments || min_segments > max_tokens) {
        throw std::invalid_argument(
            "noncausal_gqa_attention workspace: invalid execution envelope");
    }
    const std::int32_t segments = std::min(max_segments, max_tokens);
    WorkspaceLayoutBuilder layout;
    (void)allocate_workspace(layout, max_tokens, segments);
    return layout.peak_bytes(1);
}

void noncausal_gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                             AttentionHeadGeometry geometry, float scale, const Tensor& cu_seqlens,
                             WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    const std::int32_t tokens =
        validate_qkv(q, k, v, out, geometry, scale, "noncausal_gqa_attention");
    const std::int32_t segments = cu_seqlens.ne[0] - 1;
    if (cu_seqlens.dtype != DType::I32 || segments <= 0 || segments > tokens ||
        cu_seqlens.ne[1] != 1 || cu_seqlens.ne[2] != 1 || cu_seqlens.ne[3] != 1 ||
        !cu_seqlens.is_contiguous() || cu_seqlens.data == nullptr) {
        throw std::invalid_argument(
            "noncausal_gqa_attention: cu_seqlens must be contiguous I32 [S+1]");
    }
    auto scratch_scope = workspace.scope();
    Tensor tiles       = allocate_workspace(workspace, tokens, segments);
    Tensor* tiles_ptr  = tiles.data == nullptr ? nullptr : &tiles;
    detail::noncausal_gqa_attention_launch(q, k, v, cu_seqlens, tiles_ptr, out, stream);
}

} // namespace ninfer::ops
