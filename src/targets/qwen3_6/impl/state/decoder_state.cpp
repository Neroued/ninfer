#include <ninfer/targets/qwen3_6/decoder_state.h>

#include "core/device.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6 {
namespace {

constexpr std::size_t kArenaAlign = 256;

void validate_positive(std::int32_t value, const char* name) {
    if (value <= 0) { throw std::invalid_argument(name); }
}

void validate_layer_slot(const GdnStateStore& state, std::uint32_t layer, std::int32_t slot,
                         const char* label) {
    if (layer >= state.layer_count()) {
        throw std::out_of_range(std::string(label) + " layer out of range");
    }
    if (slot < 0 || slot >= state.spec.snapshot_slots) {
        throw std::out_of_range(std::string(label) + " slot out of range");
    }
}

} // namespace

GdnStateLayout plan_gdn_state(LayoutBuilder& builder, const GdnStateSpec& spec) {
    if (spec.layers == 0) { throw std::invalid_argument("GdnState layers must be nonzero"); }
    if (spec.layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("GdnState layer count exceeds int32");
    }
    validate_positive(spec.conv_dim, "GdnState conv_dim must be positive");
    validate_positive(spec.conv_width, "GdnState conv_width must be positive");
    validate_positive(spec.value_heads, "GdnState value_heads must be positive");
    validate_positive(spec.value_head_dim, "GdnState value_head_dim must be positive");
    validate_positive(spec.key_head_dim, "GdnState key_head_dim must be positive");
    validate_positive(spec.snapshot_slots, "GdnState snapshot_slots must be positive");
    if (spec.conv_dtype != DType::BF16 && spec.conv_dtype != DType::FP32) {
        throw std::invalid_argument("GdnState conv_dtype must be BF16 or FP32");
    }

    const Tensor conv_shape(nullptr, spec.conv_dtype,
                            {spec.conv_dim, spec.conv_width, spec.snapshot_slots});
    const Tensor ssm_shape(
        nullptr, DType::FP32,
        {spec.key_head_dim, spec.value_head_dim, spec.value_heads, spec.snapshot_slots});

    GdnStateLayout layout;
    layout.spec = spec;
    layout.conv.reserve(spec.layers);
    layout.ssm.reserve(spec.layers);
    for (std::uint32_t layer = 0; layer < spec.layers; ++layer) {
        const std::string prefix = "GDN layer " + std::to_string(layer);
        layout.conv.push_back(builder.add(conv_shape.bytes(), kArenaAlign, prefix + " conv"));
        layout.ssm.push_back(builder.add(ssm_shape.bytes(), kArenaAlign, prefix + " SSM"));
    }
    return layout;
}

GdnStateStore::GdnStateStore(DeviceSpan backing, const GdnStateLayout& layout) : spec(layout.spec) {
    if (layout.conv.empty() || layout.ssm.size() != layout.conv.size() ||
        layout.conv.size() != spec.layers) {
        throw std::invalid_argument("GdnState layout layer counts are inconsistent");
    }
    const Tensor conv_shape(nullptr, spec.conv_dtype,
                            {spec.conv_dim, spec.conv_width, spec.snapshot_slots});
    const Tensor ssm_shape(
        nullptr, DType::FP32,
        {spec.key_head_dim, spec.value_head_dim, spec.value_heads, spec.snapshot_slots});
    conv.reserve(layout.conv.size());
    ssm.reserve(layout.ssm.size());
    for (std::size_t layer = 0; layer < layout.conv.size(); ++layer) {
        if (layout.conv[layer].bytes != conv_shape.bytes() ||
            layout.ssm[layer].bytes != ssm_shape.bytes()) {
            throw std::logic_error("GdnState layout tensor byte size is inconsistent");
        }
        conv.emplace_back(layout.conv[layer].bind(backing).data, spec.conv_dtype,
                          std::initializer_list<std::int32_t>{spec.conv_dim, spec.conv_width,
                                                              spec.snapshot_slots});
        ssm.emplace_back(layout.ssm[layer].bind(backing).data, DType::FP32,
                         std::initializer_list<std::int32_t>{spec.key_head_dim, spec.value_head_dim,
                                                             spec.value_heads,
                                                             spec.snapshot_slots});
    }
    reset_running();
}

std::uint32_t GdnStateStore::layer_count() const noexcept {
    return static_cast<std::uint32_t>(conv.size());
}

std::int64_t GdnStateStore::conv_slot_stride_elements() const noexcept {
    return static_cast<std::int64_t>(spec.conv_dim) * static_cast<std::int64_t>(spec.conv_width);
}

std::int64_t GdnStateStore::ssm_slot_stride_elements() const noexcept {
    return static_cast<std::int64_t>(spec.key_head_dim) *
           static_cast<std::int64_t>(spec.value_head_dim) *
           static_cast<std::int64_t>(spec.value_heads);
}

Tensor GdnStateStore::conv_slot(std::uint32_t layer, std::int32_t slot) const {
    validate_layer_slot(*this, layer, slot, "GdnState conv_slot");
    return conv.at(layer).slice(2, slot, 1).view({spec.conv_dim, spec.conv_width});
}

Tensor GdnStateStore::ssm_slot(std::uint32_t layer, std::int32_t slot) const {
    validate_layer_slot(*this, layer, slot, "GdnState ssm_slot");
    return ssm.at(layer)
        .slice(3, slot, 1)
        .view({spec.key_head_dim, spec.value_head_dim, spec.value_heads});
}

void GdnStateStore::copy_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream) {
    if (src == dst) { return; }
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor source      = conv_slot(layer, src);
        const Tensor destination = conv_slot(layer, dst);
        CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor source      = ssm_slot(layer, src);
        const Tensor destination = ssm_slot(layer, dst);
        CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

void GdnStateStore::reset_running(cudaStream_t stream) {
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor slot = conv_slot(layer, 0);
        CUDA_CHECK(cudaMemsetAsync(slot.data, 0, slot.bytes(), stream));
    }
    for (std::uint32_t layer = 0; layer < layer_count(); ++layer) {
        const Tensor slot = ssm_slot(layer, 0);
        CUDA_CHECK(cudaMemsetAsync(slot.data, 0, slot.bytes(), stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv =
        plan_kv_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                      spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_kv_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                      spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group);
    }
    layout.gdn = plan_gdn_state(builder, spec.gdn);
    return layout;
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv), gdn(backing, layout.gdn) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

KVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const KVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6
