#include "models/qwen3_5/execution/visual_scatter.h"

#include "core/device.h"
#include "ninfer/ops/scatter.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

} // namespace

void scatter_shifted_visual_embeddings(Tensor& input_embeddings, const Tensor& visual_embeddings,
                                       const qwen3_5::MtpVisualOverlap& overlap,
                                       Tensor& destination_indices, cudaStream_t stream) {
    if (overlap.empty() || destination_indices.dtype != DType::I32 ||
        destination_indices.ne[0] != static_cast<std::int32_t>(overlap.size())) {
        throw std::invalid_argument("shifted visual scatter has invalid destination indices");
    }
    const auto count = static_cast<std::int32_t>(overlap.size());
    copy_i32(overlap.destination_columns.data(), destination_indices, stream);
    Tensor embeddings =
        visual_embeddings.slice(1, static_cast<std::int32_t>(overlap.source_begin), count);
    ops::scatter(embeddings, destination_indices, input_embeddings, stream);
}

} // namespace ninfer::models::qwen3_5::detail
