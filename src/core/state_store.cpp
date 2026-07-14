#include "ninfer/core/state_store.h"

#include "ninfer/core/device.h"

#include <limits>
#include <new>
#include <stdexcept>
#include <string>

namespace ninfer {
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

void validate_layer_slot(const GdnState& state, std::uint32_t layer, std::int32_t slot,
                         const char* label) {
    if (layer >= state.layer_count()) {
        throw std::out_of_range(std::string(label) + " layer out of range");
    }
    if (slot < 0 || slot >= state.snapshot_slots) {
        throw std::out_of_range(std::string(label) + " slot out of range");
    }
}

} // namespace

GdnState::GdnState(DeviceArena& cache_arena, std::uint32_t gdn_layers, std::int32_t conv_dim_in,
                   std::int32_t conv_width_in, std::int32_t value_heads_in,
                   std::int32_t value_head_dim_in, std::int32_t key_head_dim_in,
                   DType conv_dtype_in)
    : GdnState(cache_arena, gdn_layers, conv_dim_in, conv_width_in, value_heads_in,
               value_head_dim_in, key_head_dim_in, 1, conv_dtype_in) {}

GdnState::GdnState(DeviceArena& cache_arena, std::uint32_t gdn_layers, std::int32_t conv_dim_in,
                   std::int32_t conv_width_in, std::int32_t value_heads_in,
                   std::int32_t value_head_dim_in, std::int32_t key_head_dim_in,
                   std::int32_t snapshot_slots_in, DType conv_dtype_in)
    : conv_dim(conv_dim_in), conv_width(conv_width_in), value_heads(value_heads_in),
      value_head_dim(value_head_dim_in), key_head_dim(key_head_dim_in),
      snapshot_slots(snapshot_slots_in), conv_dtype(conv_dtype_in) {
    if (gdn_layers == 0) { throw std::invalid_argument("GdnState gdn_layers must be nonzero"); }
    validate_positive(conv_dim_in, "GdnState conv_dim must be positive");
    validate_positive(conv_width_in, "GdnState conv_width must be positive");
    validate_positive(value_heads_in, "GdnState value_heads must be positive");
    validate_positive(value_head_dim_in, "GdnState value_head_dim must be positive");
    validate_positive(key_head_dim_in, "GdnState key_head_dim must be positive");
    validate_positive(snapshot_slots_in, "GdnState snapshot_slots must be positive");
    if (conv_dtype_in != DType::BF16 && conv_dtype_in != DType::FP32) {
        throw std::invalid_argument("GdnState conv_dtype must be BF16 or FP32");
    }

    const Tensor conv_shape(nullptr, conv_dtype_in,
                            {conv_dim_in, conv_width_in, snapshot_slots_in});
    const Tensor ssm_shape(nullptr, DType::FP32,
                           {key_head_dim_in, value_head_dim_in, value_heads_in,
                            snapshot_slots_in});
    preflight_state_arena(cache_arena, gdn_layers, conv_shape.bytes(), ssm_shape.bytes());

    conv.reserve(gdn_layers);
    ssm.reserve(gdn_layers);
    for (std::uint32_t layer = 0; layer < gdn_layers; ++layer) {
        conv.push_back(
            cache_arena.alloc(conv_dtype_in, {conv_dim_in, conv_width_in, snapshot_slots_in}));
        ssm.push_back(cache_arena.alloc(
            DType::FP32,
            {key_head_dim_in, value_head_dim_in, value_heads_in, snapshot_slots_in}));
    }
    reset();
}

std::uint32_t GdnState::layer_count() const noexcept {
    return static_cast<std::uint32_t>(conv.size());
}

std::int64_t GdnState::conv_slot_stride_elements() const noexcept {
    return static_cast<std::int64_t>(conv_dim) * static_cast<std::int64_t>(conv_width);
}

std::int64_t GdnState::ssm_slot_stride_elements() const noexcept {
    return static_cast<std::int64_t>(key_head_dim) * static_cast<std::int64_t>(value_head_dim) *
           static_cast<std::int64_t>(value_heads);
}

Tensor GdnState::conv_slot(std::uint32_t layer, std::int32_t slot) const {
    validate_layer_slot(*this, layer, slot, "GdnState conv_slot");
    return conv.at(layer).slice(2, slot, 1).view({conv_dim, conv_width});
}

Tensor GdnState::ssm_slot(std::uint32_t layer, std::int32_t slot) const {
    validate_layer_slot(*this, layer, slot, "GdnState ssm_slot");
    return ssm.at(layer).slice(3, slot, 1).view({key_head_dim, value_head_dim, value_heads});
}

void GdnState::copy_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream) {
    if (src == dst) { return; }
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor s = conv_slot(layer, src);
        const Tensor d = conv_slot(layer, dst);
        CUDA_CHECK(cudaMemcpyAsync(d.data, s.data, s.bytes(), cudaMemcpyDeviceToDevice, stream));
    }
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor s = ssm_slot(layer, src);
        const Tensor d = ssm_slot(layer, dst);
        CUDA_CHECK(cudaMemcpyAsync(d.data, s.data, s.bytes(), cudaMemcpyDeviceToDevice, stream));
    }
}

void GdnState::reset(cudaStream_t stream) {
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor slot0 = conv_slot(layer, 0);
        CUDA_CHECK(cudaMemsetAsync(slot0.data, 0, slot0.bytes(), stream));
    }
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor slot0 = ssm_slot(layer, 0);
        CUDA_CHECK(cudaMemsetAsync(slot0.data, 0, slot0.bytes(), stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace ninfer
