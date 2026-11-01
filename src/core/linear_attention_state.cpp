#include "core/linear_attention_state.h"

#include <stdexcept>

namespace ninfer {

LinearAttentionStateLayout plan_linear_attention_state(StateImageBuilder& builder,
                                                       const LinearAttentionStateSpec& spec) {
    if (spec.layers == 0 || spec.conv_channels <= 0 || spec.conv_width <= 0 ||
        spec.value_heads <= 0 || spec.value_head_dim <= 0 || spec.key_head_dim <= 0 ||
        (spec.conv_dtype != DType::BF16 && spec.conv_dtype != DType::FP32)) {
        throw std::invalid_argument("Linear Attention state geometry is invalid");
    }
    return {
        spec,
        builder.add(spec.conv_dtype, {spec.conv_channels, spec.conv_width}, spec.layers,
                    "Linear Attention convolution"),
        builder.add(DType::FP32, {spec.key_head_dim, spec.value_head_dim, spec.value_heads},
                    spec.layers, "Linear Attention recurrence"),
    };
}

LinearAttentionStateView::LinearAttentionStateView(const StateImageStorage& storage,
                                                   const LinearAttentionStateLayout& layout)
    : view_{storage.field(layout.conv, {}), storage.field(layout.recurrent, {}), layout.spec} {
    const auto& conv      = view_.conv.field;
    const auto& recurrent = view_.recurrent.field;
    const auto& spec      = layout.spec;
    if (conv.tensor.dtype != spec.conv_dtype ||
        conv.tensor.shape !=
            std::array<std::int32_t, 4>{spec.conv_channels, spec.conv_width, 1, 1} ||
        recurrent.tensor.dtype != DType::FP32 ||
        recurrent.tensor.shape != std::array<std::int32_t, 4>{spec.key_head_dim,
                                                              spec.value_head_dim, spec.value_heads,
                                                              1} ||
        conv.layers != spec.layers || recurrent.layers != spec.layers) {
        throw std::invalid_argument("Linear Attention state fields do not match their geometry");
    }
}

LinearAttentionStateLayerView
LinearAttentionStateView::layer_view(std::uint32_t layer, const Tensor& byte_offsets) const {
    auto all = all_layers_view(byte_offsets);
    return {all.conv.layer(layer), all.recurrent.layer(layer)};
}

LinearAttentionStateAllLayersView
LinearAttentionStateView::all_layers_view(const Tensor& byte_offsets) const {
    auto result                           = view_;
    result.conv.records.byte_offsets      = byte_offsets;
    result.recurrent.records.byte_offsets = byte_offsets;
    return result;
}

Tensor LinearAttentionStateView::conv_tensor(std::uint32_t layer, std::int64_t byte_offset) const {
    return view_.conv.tensor(byte_offset, layer);
}

Tensor LinearAttentionStateView::recurrent_tensor(std::uint32_t layer,
                                                  std::int64_t byte_offset) const {
    return view_.recurrent.tensor(byte_offset, layer);
}

} // namespace ninfer
