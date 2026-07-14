#include "kernels/vision_attention/vision_attention.h"

#include "kernels/vision_attention/launch.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::kernels {
namespace {

constexpr std::int32_t kHeadDim = 72;
constexpr std::int32_t kHeads   = 16;

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

std::int32_t vision_attention_scratch_tiles(std::int32_t patches, std::int32_t segments) {
    if (patches <= 0 || segments <= 0) {
        throw std::invalid_argument("vision_attention: patches and segments must be positive");
    }
    if (segments == 1) { return 0; }
    constexpr std::int32_t tile_rows = 64;
    return (patches + tile_rows - 1) / tile_rows + segments - 1;
}

void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                      const Tensor& cu_seqlens, Tensor* scratch_tiles, Tensor& out,
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
    if (cu_seqlens.dtype != DType::I32 || cu_seqlens.ne[0] < 2 || cu_seqlens.ne[1] != 1 ||
        cu_seqlens.ne[2] != 1 || cu_seqlens.ne[3] != 1 || !cu_seqlens.is_contiguous() ||
        cu_seqlens.data == nullptr) {
        throw std::invalid_argument("vision_attention: cu_seqlens must be contiguous I32 [S+1]");
    }
    const std::int32_t required =
        vision_attention_scratch_tiles(patches, cu_seqlens.ne[0] - 1);
    if (required == 0) {
        if (scratch_tiles != nullptr && scratch_tiles->data != nullptr) {
            throw std::invalid_argument("vision_attention: one segment requires no scratch tiles");
        }
    } else if (scratch_tiles == nullptr || scratch_tiles->dtype != DType::I32 ||
               scratch_tiles->ne[0] != 4 || scratch_tiles->ne[1] < required ||
               !scratch_tiles->is_contiguous() || scratch_tiles->data == nullptr) {
        throw std::invalid_argument("vision_attention: invalid preplanned scratch tiles");
    }
    detail::vision_attention_launch(q, k, v, cu_seqlens, scratch_tiles, out, stream);
}

void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens,
                      WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    const std::int32_t patches  = q.ne[2];
    const std::int32_t segments      = cu_seqlens.ne[0] - 1;
    auto scratch_scope               = workspace.scope();
    Tensor tiles;
    Tensor* tiles_ptr = nullptr;
    const std::int32_t max_tiles = vision_attention_scratch_tiles(patches, segments);
    if (max_tiles != 0) {
        tiles                        = workspace.alloc(DType::I32, {4, max_tiles});
        tiles_ptr                    = &tiles;
    }
    vision_attention(q, k, v, cu_seqlens, tiles_ptr, out, stream);
}

} // namespace ninfer::kernels
