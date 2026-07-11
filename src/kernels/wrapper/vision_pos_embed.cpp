#include "qus/kernels/vision_pos_embed.h"

#include "kernels/launcher/vision_pos_embed.h"

#include <stdexcept>

namespace qus::kernels {

void vision_pos_embed_add(const Tensor& table, const Tensor& indices, const Tensor& weights,
                          Tensor& x, cudaStream_t stream) {
    if (table.dtype != DType::BF16 || x.dtype != DType::BF16 || indices.dtype != DType::I32 ||
        weights.dtype != DType::FP32) {
        throw std::invalid_argument(
            "vision_pos_embed_add: table/x must be BF16, indices I32, weights FP32");
    }
    if (table.ne[2] != 1 || table.ne[3] != 1 || x.ne[2] != 1 || x.ne[3] != 1 ||
        indices.ne[0] != 4 || indices.ne[1] != x.ne[1] || indices.ne[2] != 1 ||
        indices.ne[3] != 1 || weights.ne[0] != 4 || weights.ne[1] != x.ne[1] ||
        weights.ne[2] != 1 || weights.ne[3] != 1 || table.ne[0] != x.ne[0]) {
        throw std::invalid_argument(
            "vision_pos_embed_add: expected table [D,R], controls [4,P], x [D,P]");
    }
    if (x.ne[0] < 0 || x.ne[1] < 0 || table.ne[1] <= 0) {
        throw std::invalid_argument("vision_pos_embed_add: invalid dimensions");
    }
    if (x.ne[0] == 0 || x.ne[1] == 0) { return; }
    if (!table.is_contiguous() || !indices.is_contiguous() || !weights.is_contiguous() ||
        !x.is_contiguous()) {
        throw std::invalid_argument("vision_pos_embed_add: tensors must be contiguous");
    }
    if (table.data == nullptr || indices.data == nullptr || weights.data == nullptr ||
        x.data == nullptr) {
        throw std::invalid_argument("vision_pos_embed_add: tensor data must be non-null");
    }
    detail::vision_pos_embed_add_launch(table, indices, weights, x, stream);
}

} // namespace qus::kernels
