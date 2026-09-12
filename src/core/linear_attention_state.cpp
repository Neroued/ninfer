#include "core/linear_attention_state.h"

#include "core/device.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr std::size_t kArenaAlign = 256;

void validate_positive(std::int32_t value, const char* message) {
    if (value <= 0) { throw std::invalid_argument(message); }
}

void validate_layer_slot(const LinearAttentionStatePool& pool, std::uint32_t layer,
                         std::int32_t slot, const char* label) {
    if (layer >= pool.layer_count()) {
        throw std::out_of_range(std::string(label) + " layer out of range");
    }
    if (slot < 0 || slot >= pool.slot_count()) {
        throw std::out_of_range(std::string(label) + " slot out of range");
    }
}

void validate_state_tensor(const Tensor& tensor, DType dtype,
                           std::initializer_list<std::int32_t> shape, const char* label) {
    const Tensor expected(nullptr, dtype, shape);
    if (tensor.data == nullptr || tensor.dtype != dtype || !tensor.is_contiguous() ||
        tensor.bytes() != expected.bytes()) {
        throw std::logic_error(std::string("LinearAttentionStatePool ") + label +
                               " tensor is inconsistent");
    }
    for (int dim = 0; dim < 4; ++dim) {
        if (tensor.ne[dim] != expected.ne[dim]) {
            throw std::logic_error(std::string("LinearAttentionStatePool ") + label +
                                   " shape is inconsistent");
        }
    }
}

std::int64_t layer_stride_bytes(const std::vector<Tensor>& tensors, const char* label) {
    if (tensors.empty()) {
        throw std::logic_error(std::string("LinearAttentionStatePool has no ") + label + " layers");
    }
    if (tensors.size() == 1) { return 0; }

    const auto first  = reinterpret_cast<std::uintptr_t>(tensors[0].data);
    const auto second = reinterpret_cast<std::uintptr_t>(tensors[1].data);
    if (second <= first ||
        second - first > static_cast<std::uintptr_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::logic_error(std::string("LinearAttentionStatePool ") + label +
                               " layer stride is invalid");
    }
    const auto stride = static_cast<std::int64_t>(second - first);
    if (static_cast<std::uint64_t>(stride) < tensors.front().bytes()) {
        throw std::logic_error(std::string("LinearAttentionStatePool ") + label +
                               " layers overlap");
    }
    for (std::size_t layer = 2; layer < tensors.size(); ++layer) {
        const auto previous = reinterpret_cast<std::uintptr_t>(tensors[layer - 1].data);
        const auto current  = reinterpret_cast<std::uintptr_t>(tensors[layer].data);
        if (current <= previous || current - previous != static_cast<std::uintptr_t>(stride)) {
            throw std::logic_error(std::string("LinearAttentionStatePool ") + label +
                                   " layer stride is not constant");
        }
    }
    return stride;
}

} // namespace

LinearAttentionStatePoolLayout
plan_linear_attention_state_pool(LayoutBuilder& builder, const LinearAttentionStatePoolSpec& spec) {
    if (spec.layers == 0) {
        throw std::invalid_argument("LinearAttentionStatePool layers must be nonzero");
    }
    if (spec.layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("LinearAttentionStatePool layer count exceeds int32");
    }
    validate_positive(spec.conv_channels,
                      "LinearAttentionStatePool conv_channels must be positive");
    validate_positive(spec.conv_width, "LinearAttentionStatePool conv_width must be positive");
    validate_positive(spec.value_heads, "LinearAttentionStatePool value_heads must be positive");
    validate_positive(spec.value_head_dim,
                      "LinearAttentionStatePool value_head_dim must be positive");
    validate_positive(spec.key_head_dim, "LinearAttentionStatePool key_head_dim must be positive");
    validate_positive(spec.slot_count, "LinearAttentionStatePool slot_count must be positive");
    if (spec.conv_dtype != DType::BF16 && spec.conv_dtype != DType::FP32) {
        throw std::invalid_argument("LinearAttentionStatePool conv_dtype must be BF16 or FP32");
    }

    const Tensor conv_shape(nullptr, spec.conv_dtype,
                            {spec.conv_channels, spec.conv_width, spec.slot_count});
    const Tensor recurrent_shape(
        nullptr, DType::FP32,
        {spec.key_head_dim, spec.value_head_dim, spec.value_heads, spec.slot_count});

    LinearAttentionStatePoolLayout layout;
    layout.spec = spec;
    layout.conv.reserve(spec.layers);
    layout.recurrent.reserve(spec.layers);
    for (std::uint32_t layer = 0; layer < spec.layers; ++layer) {
        const std::string prefix = "Linear Attention layer " + std::to_string(layer);
        layout.conv.push_back(builder.add(conv_shape.bytes(), kArenaAlign, prefix + " conv"));
        layout.recurrent.push_back(
            builder.add(recurrent_shape.bytes(), kArenaAlign, prefix + " recurrent"));
    }
    return layout;
}

LinearAttentionStatePool::LinearAttentionStatePool(DeviceSpan backing,
                                                   const LinearAttentionStatePoolLayout& layout)
    : spec_(layout.spec) {
    if (layout.conv.empty() || layout.recurrent.size() != layout.conv.size() ||
        layout.conv.size() != spec_.layers) {
        throw std::invalid_argument(
            "LinearAttentionStatePool layout layer counts are inconsistent");
    }

    const Tensor conv_shape(nullptr, spec_.conv_dtype,
                            {spec_.conv_channels, spec_.conv_width, spec_.slot_count});
    const Tensor recurrent_shape(
        nullptr, DType::FP32,
        {spec_.key_head_dim, spec_.value_head_dim, spec_.value_heads, spec_.slot_count});
    conv_.reserve(layout.conv.size());
    recurrent_.reserve(layout.recurrent.size());
    for (std::size_t layer = 0; layer < layout.conv.size(); ++layer) {
        if (layout.conv[layer].bytes != conv_shape.bytes() ||
            layout.recurrent[layer].bytes != recurrent_shape.bytes()) {
            throw std::logic_error(
                "LinearAttentionStatePool layout tensor byte size is inconsistent");
        }
        conv_.emplace_back(layout.conv[layer].bind(backing).data, spec_.conv_dtype,
                           std::initializer_list<std::int32_t>{spec_.conv_channels,
                                                               spec_.conv_width, spec_.slot_count});
        recurrent_.emplace_back(
            layout.recurrent[layer].bind(backing).data, DType::FP32,
            std::initializer_list<std::int32_t>{spec_.key_head_dim, spec_.value_head_dim,
                                                spec_.value_heads, spec_.slot_count});
    }

    // Validate the transfer geometry once while binding the immutable inventory.
    (void)all_layers_view();
}

std::uint32_t LinearAttentionStatePool::layer_count() const noexcept {
    return static_cast<std::uint32_t>(conv_.size());
}

std::int32_t LinearAttentionStatePool::slot_count() const noexcept { return spec_.slot_count; }

LinearAttentionStateLayerView LinearAttentionStatePool::layer_view(std::uint32_t layer) const {
    if (layer >= layer_count()) {
        throw std::out_of_range("LinearAttentionStatePool layer_view layer out of range");
    }
    return {.conv = conv_[layer], .recurrent = recurrent_[layer]};
}

LinearAttentionStateSlotView LinearAttentionStatePool::slot_view(std::int32_t slot) const {
    validate_layer_slot(*this, 0, slot, "LinearAttentionStatePool slot_view");
    const auto all   = all_layers_view();
    Tensor conv      = conv_slot(0, slot);
    Tensor recurrent = recurrent_slot(0, slot);
    return {
        .conv_layer0                 = conv,
        .recurrent_layer0            = recurrent,
        .conv_layer_bytes            = conv.bytes(),
        .recurrent_layer_bytes       = recurrent.bytes(),
        .conv_layer_pitch_bytes      = all.conv_layer_stride_bytes,
        .recurrent_layer_pitch_bytes = all.recurrent_layer_stride_bytes,
        .layers                      = layer_count(),
    };
}

LinearAttentionStateAllLayersView LinearAttentionStatePool::all_layers_view() const {
    if (conv_.size() != spec_.layers || recurrent_.size() != spec_.layers || conv_.empty()) {
        throw std::logic_error("LinearAttentionStatePool layer inventory is inconsistent");
    }
    for (std::size_t layer = 0; layer < conv_.size(); ++layer) {
        validate_state_tensor(conv_[layer], spec_.conv_dtype,
                              {spec_.conv_channels, spec_.conv_width, spec_.slot_count}, "conv");
        validate_state_tensor(
            recurrent_[layer], DType::FP32,
            {spec_.key_head_dim, spec_.value_head_dim, spec_.value_heads, spec_.slot_count},
            "recurrent");
    }
    return LinearAttentionStateAllLayersView{
        .conv_layer0                  = conv_.front(),
        .recurrent_layer0             = recurrent_.front(),
        .conv_layer_stride_bytes      = layer_stride_bytes(conv_, "conv"),
        .recurrent_layer_stride_bytes = layer_stride_bytes(recurrent_, "recurrent"),
        .spec                         = spec_,
    };
}

Tensor LinearAttentionStatePool::conv_slot(std::uint32_t layer, std::int32_t slot) const {
    validate_layer_slot(*this, layer, slot, "LinearAttentionStatePool conv_slot");
    return conv_.at(layer).slice(2, slot, 1).view({spec_.conv_channels, spec_.conv_width});
}

Tensor LinearAttentionStatePool::recurrent_slot(std::uint32_t layer, std::int32_t slot) const {
    validate_layer_slot(*this, layer, slot, "LinearAttentionStatePool recurrent_slot");
    return recurrent_.at(layer)
        .slice(3, slot, 1)
        .view({spec_.key_head_dim, spec_.value_head_dim, spec_.value_heads});
}

void LinearAttentionStatePool::copy_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream) {
    validate_layer_slot(*this, 0, src, "LinearAttentionStatePool copy_slot source");
    validate_layer_slot(*this, 0, dst, "LinearAttentionStatePool copy_slot destination");
    if (src == dst) { return; }
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor source      = conv_slot(layer, src);
        const Tensor destination = conv_slot(layer, dst);
        CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor source      = recurrent_slot(layer, src);
        const Tensor destination = recurrent_slot(layer, dst);
        CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

void LinearAttentionStatePool::zero_slot(std::int32_t slot, cudaStream_t stream) {
    validate_layer_slot(*this, 0, slot, "LinearAttentionStatePool zero_slot");
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor state = conv_slot(layer, slot);
        CUDA_CHECK(cudaMemsetAsync(state.data, 0, state.bytes(), stream));
    }
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor state = recurrent_slot(layer, slot);
        CUDA_CHECK(cudaMemsetAsync(state.data, 0, state.bytes(), stream));
    }
}

void LinearAttentionStatePool::zero_all(cudaStream_t stream) {
    for (const Tensor& state : conv_) {
        CUDA_CHECK(cudaMemsetAsync(state.data, 0, state.bytes(), stream));
    }
    for (const Tensor& state : recurrent_) {
        CUDA_CHECK(cudaMemsetAsync(state.data, 0, state.bytes(), stream));
    }
}

} // namespace ninfer
