#include "targets/qwen3_6_27b_rtx5090/impl/state/state_store.h"

#include "core/device.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {
namespace {

constexpr std::size_t kArenaAlign = 256;

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

GdnStateLayout plan_gdn_state(LayoutBuilder& builder, std::uint32_t gdn_layers,
                              std::int32_t conv_dim_in, std::int32_t conv_width_in,
                              std::int32_t value_heads_in, std::int32_t value_head_dim_in,
                              std::int32_t key_head_dim_in, std::int32_t snapshot_slots_in,
                              DType conv_dtype_in) {
    if (gdn_layers == 0) { throw std::invalid_argument("GdnState gdn_layers must be nonzero"); }
    if (gdn_layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("GdnState layer count exceeds int32");
    }
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
                           {key_head_dim_in, value_head_dim_in, value_heads_in, snapshot_slots_in});

    GdnStateLayout layout;
    layout.conv_dim       = conv_dim_in;
    layout.conv_width     = conv_width_in;
    layout.value_heads    = value_heads_in;
    layout.value_head_dim = value_head_dim_in;
    layout.key_head_dim   = key_head_dim_in;
    layout.snapshot_slots = snapshot_slots_in;
    layout.conv_dtype     = conv_dtype_in;
    layout.conv.reserve(gdn_layers);
    layout.ssm.reserve(gdn_layers);
    for (std::uint32_t layer = 0; layer < gdn_layers; ++layer) {
        const std::string prefix = "GDN layer " + std::to_string(layer);
        layout.conv.push_back(builder.add(conv_shape.bytes(), kArenaAlign, prefix + " conv"));
        layout.ssm.push_back(builder.add(ssm_shape.bytes(), kArenaAlign, prefix + " SSM"));
    }
    return layout;
}

GdnState::GdnState(DeviceSpan backing, const GdnStateLayout& layout)
    : conv_dim(layout.conv_dim), conv_width(layout.conv_width), value_heads(layout.value_heads),
      value_head_dim(layout.value_head_dim), key_head_dim(layout.key_head_dim),
      snapshot_slots(layout.snapshot_slots), conv_dtype(layout.conv_dtype) {
    if (layout.conv.empty() || layout.ssm.size() != layout.conv.size()) {
        throw std::invalid_argument("GdnState layout layer counts are inconsistent");
    }
    const Tensor conv_shape(nullptr, conv_dtype, {conv_dim, conv_width, snapshot_slots});
    const Tensor ssm_shape(nullptr, DType::FP32,
                           {key_head_dim, value_head_dim, value_heads, snapshot_slots});
    conv.reserve(layout.conv.size());
    ssm.reserve(layout.ssm.size());
    for (std::size_t layer = 0; layer < layout.conv.size(); ++layer) {
        if (layout.conv[layer].bytes != conv_shape.bytes() ||
            layout.ssm[layer].bytes != ssm_shape.bytes()) {
            throw std::logic_error("GdnState layout tensor byte size is inconsistent");
        }
        conv.emplace_back(
            layout.conv[layer].bind(backing).data, conv_dtype,
            std::initializer_list<std::int32_t>{conv_dim, conv_width, snapshot_slots});
        ssm.emplace_back(layout.ssm[layer].bind(backing).data, DType::FP32,
                         std::initializer_list<std::int32_t>{key_head_dim, value_head_dim,
                                                             value_heads, snapshot_slots});
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

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail
