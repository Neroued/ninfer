#pragma once

#include "core/cyclic_kv_cache.h"
#include "core/linear_attention_state.h"
#include "core/state_image.h"

#include <optional>

namespace ninfer::models::qwen3_5 {

struct DFlashLocalStateSpec {
    std::uint32_t layers   = 0;
    std::uint32_t capacity = 0;
    std::int32_t kv_heads  = 0;
    std::int32_t head_dim  = 0;
};

struct StateImageSpec {
    std::int32_t slot_count = 0;
    LinearAttentionStateSpec linear;
    std::int32_t hidden = 0;
    std::optional<DFlashLocalStateSpec> dflash_local;
};

struct StateImageDeviceLayout {
    StateImageLayout images;
    LinearAttentionStateLayout linear;
    StateField continuation_hidden;
    std::optional<CyclicKVCacheLayout> dflash_local;
};

[[nodiscard]] StateImageDeviceLayout plan_state_image_device_pool(LayoutBuilder& builder,
                                                                  const StateImageSpec& spec);

// Qwen's field inventory over the common physical storage. It owns no slot policy or Host pool.
class StateImageViews {
public:
    StateImageViews(MemorySpan backing, const StateImageDeviceLayout& layout);

    [[nodiscard]] StateImageStorage& storage() noexcept { return storage_; }

    [[nodiscard]] const StateImageStorage& storage() const noexcept { return storage_; }

    [[nodiscard]] Tensor continuation_hidden_slot(std::int32_t slot) const;

    [[nodiscard]] LinearAttentionStateView& linear() noexcept { return linear_; }

    [[nodiscard]] const LinearAttentionStateView& linear() const noexcept { return linear_; }

    [[nodiscard]] const Tensor& continuation_hidden_store() const noexcept { return hidden_; }

    [[nodiscard]] Tensor& continuation_hidden_store() noexcept { return hidden_; }

    [[nodiscard]] CyclicKVCacheView* dflash_local() noexcept;
    [[nodiscard]] const CyclicKVCacheView* dflash_local() const noexcept;
    [[nodiscard]] TransferWork dflash_local_transfer_work() const;

    [[nodiscard]] LayoutRegion dflash_local_group() const { return dflash_group_.value(); }

private:
    StateImageStorage storage_;
    LinearAttentionStateView linear_;
    Tensor hidden_;
    std::optional<CyclicKVCacheView> dflash_local_;
    std::optional<LayoutRegion> dflash_group_;
};

} // namespace ninfer::models::qwen3_5
