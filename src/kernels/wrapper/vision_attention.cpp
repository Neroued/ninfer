#include "qus/kernels/vision_attention.h"

#include "kernels/launcher/vision_attention.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace qus::kernels {
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

void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens,
                      WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
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
    (void)workspace;
    detail::vision_attention_launch(q, k, v, cu_seqlens, out, stream);
}

} // namespace qus::kernels
