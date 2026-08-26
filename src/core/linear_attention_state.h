#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer {

struct LinearAttentionStatePoolSpec {
    std::uint32_t layers        = 0;
    std::int32_t conv_channels  = 0;
    std::int32_t conv_width     = 0;
    std::int32_t value_heads    = 0;
    std::int32_t value_head_dim = 0;
    std::int32_t key_head_dim   = 0;
    std::int32_t slot_count     = 1;
    DType conv_dtype            = DType::BF16;
};

struct LinearAttentionStatePoolLayout {
    LinearAttentionStatePoolSpec spec;
    std::vector<LayoutRegion> conv;
    std::vector<LayoutRegion> recurrent;
};

struct LinearAttentionStateAllLayersView {
    Tensor conv_layer0;
    Tensor recurrent_layer0;
    std::int64_t conv_layer_stride_bytes      = 0;
    std::int64_t recurrent_layer_stride_bytes = 0;
    LinearAttentionStatePoolSpec spec;
};

struct LinearAttentionStateLayerView {
    Tensor conv;
    Tensor recurrent;
};

struct LinearAttentionStateSlotView {
    Tensor conv_layer0;
    Tensor recurrent_layer0;
    std::size_t conv_layer_bytes               = 0;
    std::size_t recurrent_layer_bytes          = 0;
    std::ptrdiff_t conv_layer_pitch_bytes      = 0;
    std::ptrdiff_t recurrent_layer_pitch_bytes = 0;
    std::uint32_t layers                       = 0;
};

[[nodiscard]] LinearAttentionStatePoolLayout
plan_linear_attention_state_pool(LayoutBuilder& builder, const LinearAttentionStatePoolSpec& spec);

/**
 * Fixed-capacity physical storage for model-level Linear Attention state images.
 *
 * One logical slot selects the same frontier across every layer's convolution and recurrent
 * component. The pool owns no slot roles, validity, request metadata, allocation policy, or CUDA
 * stream. Construction binds caller-owned backing without mutating it.
 */
class LinearAttentionStatePool {
public:
    LinearAttentionStatePool(DeviceSpan backing, const LinearAttentionStatePoolLayout& layout);

    LinearAttentionStatePool(const LinearAttentionStatePool&)            = delete;
    LinearAttentionStatePool& operator=(const LinearAttentionStatePool&) = delete;
    LinearAttentionStatePool(LinearAttentionStatePool&&)                 = delete;
    LinearAttentionStatePool& operator=(LinearAttentionStatePool&&)      = delete;

    [[nodiscard]] const LinearAttentionStatePoolSpec& spec() const noexcept { return spec_; }

    [[nodiscard]] std::uint32_t layer_count() const noexcept;
    [[nodiscard]] std::int32_t slot_count() const noexcept;
    [[nodiscard]] LinearAttentionStateLayerView layer_view(std::uint32_t layer) const;
    [[nodiscard]] LinearAttentionStateSlotView slot_view(std::int32_t slot) const;
    [[nodiscard]] LinearAttentionStateAllLayersView all_layers_view() const;
    [[nodiscard]] Tensor conv_slot(std::uint32_t layer, std::int32_t slot) const;
    [[nodiscard]] Tensor recurrent_slot(std::uint32_t layer, std::int32_t slot) const;

    void copy_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream = nullptr);
    void zero_slot(std::int32_t slot, cudaStream_t stream = nullptr);
    void zero_all(cudaStream_t stream = nullptr);

private:
    std::vector<Tensor> conv_;
    std::vector<Tensor> recurrent_;
    LinearAttentionStatePoolSpec spec_;
};

} // namespace ninfer
