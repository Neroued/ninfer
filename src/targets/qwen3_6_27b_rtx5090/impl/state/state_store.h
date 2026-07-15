#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {

struct GdnStateLayout {
    std::int32_t conv_dim       = 0;
    std::int32_t conv_width     = 0;
    std::int32_t value_heads    = 0;
    std::int32_t value_head_dim = 0;
    std::int32_t key_head_dim   = 0;
    std::int32_t snapshot_slots = 1;
    DType conv_dtype            = DType::BF16;
    std::vector<LayoutRegion> conv;
    std::vector<LayoutRegion> ssm;
};

[[nodiscard]] GdnStateLayout plan_gdn_state(LayoutBuilder& builder, std::uint32_t gdn_layers,
                                            std::int32_t conv_dim, std::int32_t conv_width,
                                            std::int32_t value_heads, std::int32_t value_head_dim,
                                            std::int32_t key_head_dim,
                                            std::int32_t snapshot_slots = 1,
                                            DType conv_dtype            = DType::BF16);

struct GdnState {
    std::vector<Tensor> conv;
    std::vector<Tensor> ssm;
    std::int32_t conv_dim       = 0;
    std::int32_t conv_width     = 0;
    std::int32_t value_heads    = 0;
    std::int32_t value_head_dim = 0;
    std::int32_t key_head_dim   = 0;
    std::int32_t snapshot_slots = 1;
    DType conv_dtype            = DType::BF16;

    GdnState() = default;
    GdnState(DeviceSpan backing, const GdnStateLayout& layout);

    std::uint32_t layer_count() const noexcept;
    std::int64_t conv_slot_stride_elements() const noexcept;
    std::int64_t ssm_slot_stride_elements() const noexcept;
    Tensor conv_slot(std::uint32_t layer, std::int32_t slot) const;
    Tensor ssm_slot(std::uint32_t layer, std::int32_t slot) const;
    // Copy the conv + SSM recurrent state from snapshot slot `src` into slot `dst` for every GDN
    // layer (device-to-device on `stream`). Used to stage a turn-boundary GDN snapshot for partial
    // prefix reuse and to restore it into the running slot. No-op when src == dst.
    void copy_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream = nullptr);
    void reset(cudaStream_t stream = nullptr);
};

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail
