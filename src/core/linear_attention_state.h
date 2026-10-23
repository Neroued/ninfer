#pragma once

#include "core/state_image.h"

namespace ninfer {

struct LinearAttentionStateSpec {
    std::uint32_t layers        = 0;
    std::int32_t conv_channels  = 0;
    std::int32_t conv_width     = 0;
    std::int32_t value_heads    = 0;
    std::int32_t value_head_dim = 0;
    std::int32_t key_head_dim   = 0;
    DType conv_dtype            = DType::BF16;
};

struct LinearAttentionStateLayout {
    LinearAttentionStateSpec spec;
    StateField conv;
    StateField recurrent;
};

struct LinearAttentionStateAllLayersView {
    StateFieldView conv;
    StateFieldView recurrent;
    LinearAttentionStateSpec spec;
};

struct LinearAttentionStateLayerView {
    StateFieldView conv;
    StateFieldView recurrent;
};

[[nodiscard]] LinearAttentionStateLayout
plan_linear_attention_state(StateImageBuilder& builder, const LinearAttentionStateSpec& spec);

class LinearAttentionStateView {
public:
    LinearAttentionStateView(const StateImageStorage& storage,
                             const LinearAttentionStateLayout& layout);

    [[nodiscard]] const LinearAttentionStateSpec& spec() const noexcept { return view_.spec; }

    [[nodiscard]] std::uint32_t layer_count() const noexcept { return view_.spec.layers; }

    [[nodiscard]] LinearAttentionStateLayerView layer_view(std::uint32_t layer,
                                                           const Tensor& byte_offsets) const;
    [[nodiscard]] LinearAttentionStateAllLayersView
    all_layers_view(const Tensor& byte_offsets = {}) const;

    [[nodiscard]] Tensor conv_tensor(std::uint32_t layer, std::int64_t byte_offset) const;
    [[nodiscard]] Tensor recurrent_tensor(std::uint32_t layer, std::int64_t byte_offset) const;

private:
    LinearAttentionStateAllLayersView view_;
};

} // namespace ninfer
