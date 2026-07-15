#include "ninfer/ops/scatter.h"

#include "core/device.h"
#include "ops/launcher/scatter.h"

#include <cstddef>
#include <stdexcept>

namespace ninfer::ops {

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

void extract_bf16_columns(const Tensor& source, std::int32_t source_column, Tensor& destination,
                          cudaStream_t stream) {
    if (source.dtype != DType::BF16 || destination.dtype != DType::BF16) {
        throw std::invalid_argument("extract_bf16_columns: tensors must be BF16");
    }
    if (source.ne[0] <= 0 || source.ne[1] <= 0 || destination.ne[0] <= 0 ||
        destination.ne[1] != source.ne[1] || source.ne[2] != 1 || source.ne[3] != 1 ||
        destination.ne[2] != 1 || destination.ne[3] != 1 || source_column < 0 ||
        source_column > source.ne[0] - destination.ne[0]) {
        throw std::invalid_argument("extract_bf16_columns: invalid rank-2 slice geometry");
    }
    if (!source.is_contiguous() || !destination.is_contiguous() || source.data == nullptr ||
        destination.data == nullptr) {
        throw std::invalid_argument(
            "extract_bf16_columns: tensors must be contiguous and non-null");
    }
    if (source.data == destination.data) {
        throw std::invalid_argument("extract_bf16_columns: source and destination must not alias");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width         = static_cast<std::size_t>(destination.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(source.ne[0]) * element_bytes;
    const std::size_t destination_pitch =
        static_cast<std::size_t>(destination.ne[0]) * element_bytes;
    const auto* source_ptr = static_cast<const unsigned char*>(source.data) +
                             static_cast<std::size_t>(source_column) * element_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(destination.data, destination_pitch, source_ptr, source_pitch,
                                 width, static_cast<std::size_t>(destination.ne[1]),
                                 cudaMemcpyDeviceToDevice, stream));
}

} // namespace ninfer::ops
