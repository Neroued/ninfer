#include "kernels/scatter/scatter.h"

#include "kernels/scatter/launch.h"

#include <stdexcept>

namespace ninfer::kernels {

void scatter(const Tensor& src, const Tensor& indices, Tensor& dst, cudaStream_t stream) {
    if (src.dtype != DType::BF16 || dst.dtype != DType::BF16 || indices.dtype != DType::I32) {
        throw std::invalid_argument("scatter: src/dst must be BF16 and indices must be I32");
    }
    if (src.ne[2] != 1 || src.ne[3] != 1 || dst.ne[2] != 1 || dst.ne[3] != 1 ||
        indices.ne[1] != 1 || indices.ne[2] != 1 || indices.ne[3] != 1 || src.ne[0] != dst.ne[0] ||
        indices.ne[0] != src.ne[1]) {
        throw std::invalid_argument("scatter: expected src [D,V], indices [V], dst [D,T]");
    }
    if (src.ne[0] < 0 || src.ne[1] < 0 || dst.ne[1] < 0) {
        throw std::invalid_argument("scatter: negative dimension");
    }
    if (src.ne[0] == 0 || src.ne[1] == 0) { return; }
    if (!src.is_contiguous() || !indices.is_contiguous() || !dst.is_contiguous()) {
        throw std::invalid_argument("scatter: tensors must be contiguous");
    }
    if (src.data == nullptr || indices.data == nullptr || dst.data == nullptr) {
        throw std::invalid_argument("scatter: tensor data must be non-null");
    }
    detail::scatter_launch(src, indices, dst, stream);
}

} // namespace ninfer::kernels
