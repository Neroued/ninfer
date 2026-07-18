#pragma once

#include "core/kv_cache.h"
#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::targets::qwen3_6 {

struct GdnStateSpec {
    std::uint32_t layers        = 0;
    std::int32_t conv_dim       = 0;
    std::int32_t conv_width     = 0;
    std::int32_t value_heads    = 0;
    std::int32_t value_head_dim = 0;
    std::int32_t key_head_dim   = 0;
    std::int32_t snapshot_slots = 1;
    DType conv_dtype            = DType::BF16;
};

struct GdnStateLayout {
    GdnStateSpec spec;
    std::vector<LayoutRegion> conv;
    std::vector<LayoutRegion> ssm;
};

[[nodiscard]] GdnStateLayout plan_gdn_state(LayoutBuilder& builder, const GdnStateSpec& spec);

struct GdnStateStore {
    std::vector<Tensor> conv;
    std::vector<Tensor> ssm;
    GdnStateSpec spec;

    GdnStateStore() = default;
    GdnStateStore(DeviceSpan backing, const GdnStateLayout& layout);

    [[nodiscard]] std::uint32_t layer_count() const noexcept;
    [[nodiscard]] std::int64_t conv_slot_stride_elements() const noexcept;
    [[nodiscard]] std::int64_t ssm_slot_stride_elements() const noexcept;
    [[nodiscard]] Tensor conv_slot(std::uint32_t layer, std::int32_t slot) const;
    [[nodiscard]] Tensor ssm_slot(std::uint32_t layer, std::int32_t slot) const;
    void copy_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream = nullptr);
    void reset_running(cudaStream_t stream = nullptr);
};

struct DecoderStateSpec {
    std::uint32_t full_attention_layers = 0;
    std::uint32_t mtp_layers            = 0;
    std::uint32_t capacity              = 0;
    std::int32_t kv_heads               = 0;
    std::int32_t attention_head_dim     = 0;
    DType kv_dtype                      = DType::BF16;
    std::int32_t kv_quant_group         = 0;
    bool enable_mtp                     = false;
    GdnStateSpec gdn;
};

struct DecoderStateLayout {
    KVCacheLayout text_kv;
    std::optional<KVCacheLayout> mtp_kv;
    GdnStateLayout gdn;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);

struct DecoderState {
    KVCache text_kv;
    std::optional<KVCache> mtp_kv;
    GdnStateStore gdn;

    DecoderState() = default;
    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout);

    [[nodiscard]] KVCache* mtp_cache() noexcept;
    [[nodiscard]] const KVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::targets::qwen3_6
