#include "qus/core/state_store.h"

#include "qus/core/device.h"

#include <limits>
#include <new>
#include <stdexcept>

namespace qus {
namespace {

std::size_t checked_mul_size(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("GdnState size overflow");
    }
    return a * b;
}

std::size_t checked_add_size(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error("GdnState size overflow");
    }
    return a + b;
}

void preflight_state_arena(DeviceArena& arena, std::uint32_t layers, std::size_t conv_bytes,
                           std::size_t ssm_bytes) {
    if (arena.used() > arena.capacity()) {
        throw std::runtime_error("state arena used exceeds capacity");
    }
    const std::size_t conv_with_align = checked_add_size(conv_bytes, 255);
    const std::size_t ssm_with_align  = checked_add_size(ssm_bytes, 255);
    const std::size_t per_layer       = checked_add_size(conv_with_align, ssm_with_align);
    const std::size_t required  = checked_mul_size(static_cast<std::size_t>(layers), per_layer);
    const std::size_t remaining = arena.capacity() - arena.used();
    if (required > remaining) { throw std::bad_alloc(); }
}

void validate_positive(std::int32_t value, const char* name) {
    if (value <= 0) { throw std::invalid_argument(name); }
}

} // namespace

GdnState::GdnState(DeviceArena& cache_arena, std::uint32_t gdn_layers, std::int32_t conv_dim_in,
                   std::int32_t conv_width_in, std::int32_t value_heads_in,
                   std::int32_t value_head_dim_in, std::int32_t key_head_dim_in,
                   DType conv_dtype_in)
    : conv_dim(conv_dim_in), conv_width(conv_width_in), value_heads(value_heads_in),
      value_head_dim(value_head_dim_in), key_head_dim(key_head_dim_in), conv_dtype(conv_dtype_in) {
    if (gdn_layers == 0) { throw std::invalid_argument("GdnState gdn_layers must be nonzero"); }
    validate_positive(conv_dim_in, "GdnState conv_dim must be positive");
    validate_positive(conv_width_in, "GdnState conv_width must be positive");
    validate_positive(value_heads_in, "GdnState value_heads must be positive");
    validate_positive(value_head_dim_in, "GdnState value_head_dim must be positive");
    validate_positive(key_head_dim_in, "GdnState key_head_dim must be positive");
    if (conv_dtype_in != DType::BF16 && conv_dtype_in != DType::FP32) {
        throw std::invalid_argument("GdnState conv_dtype must be BF16 or FP32");
    }

    const Tensor conv_shape(nullptr, conv_dtype_in, {conv_dim_in, conv_width_in});
    const Tensor ssm_shape(nullptr, DType::FP32,
                           {value_heads_in, value_head_dim_in, key_head_dim_in});
    preflight_state_arena(cache_arena, gdn_layers, conv_shape.bytes(), ssm_shape.bytes());

    conv.reserve(gdn_layers);
    ssm.reserve(gdn_layers);
    for (std::uint32_t layer = 0; layer < gdn_layers; ++layer) {
        conv.push_back(cache_arena.alloc(conv_dtype_in, {conv_dim_in, conv_width_in}));
        ssm.push_back(
            cache_arena.alloc(DType::FP32, {value_heads_in, value_head_dim_in, key_head_dim_in}));
    }
    reset();
}

std::uint32_t GdnState::layer_count() const noexcept {
    return static_cast<std::uint32_t>(conv.size());
}

void GdnState::reset(cudaStream_t stream) {
    for (const Tensor& tensor : conv) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), stream));
    }
    for (const Tensor& tensor : ssm) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace qus
