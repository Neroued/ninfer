#pragma once

#include "core/cyclic_kv_cache.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/state_image.h"
#include "models/common/context/directory/history.h"
#include "models/common/context/directory/records.h"
#include "models/common/context/storage.h"
#include "models/inkling2/config.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::inkling2 {
namespace execution {
struct ModelProfile;
}

struct StateLayout {
    StateImageLayout image;
    std::optional<StateField> swa_k_conv, swa_v_conv;
    std::optional<StateField> full_k_conv, full_v_conv;
    StateField attn_sconv, mlp_sconv;
    StateField hidden;
    std::optional<CyclicKVCacheLayout> swa;
    std::optional<StateField> dspark_f_rnn;
    std::optional<CyclicKVCacheLayout> dspark_swa;
    // Fields whose entire image must be copied together on Fork/Capture.
    std::optional<LayoutRegion> persistent_copy;
};

struct StatePlacement {
    RecordSpec image;
    std::optional<LayoutRegion> persistent_copy;
    bool operator==(const StatePlacement&) const = default;
};

[[nodiscard]] StateLayout plan_state(const execution::ModelProfile&, KvCacheStorage swa_storage);
[[nodiscard]] StatePlacement describe_state(const StateLayout&);

struct CacheLayout {
    KVPageLayout page;
    KVExecutionTableLayout tables;
    PagedKVStorageLayout storage;
    std::uint32_t layers  = 0;
    std::int32_t head_dim = 0;
    std::int32_t kv_heads = 0;
};

[[nodiscard]] CacheLayout plan_cache(LayoutBuilder&, const AttentionGeometry&, std::uint32_t layers,
                                     std::uint32_t max_context, std::uint32_t rows,
                                     KvCacheStorage storage);

class StateViews {
public:
    StateViews(MemorySpan backing, const StateLayout& layout);

    [[nodiscard]] const StateImageStorage& image() const noexcept { return image_; }
    [[nodiscard]] StateImageStorage& image() noexcept { return image_; }

    [[nodiscard]] StateFieldView k_conv(bool is_swa, std::uint32_t slot,
                                        const Tensor& offsets) const;
    [[nodiscard]] StateFieldView v_conv(bool is_swa, std::uint32_t slot,
                                        const Tensor& offsets) const;
    [[nodiscard]] StateFieldView attn_sconv(std::uint32_t logical_layer,
                                            const Tensor& offsets) const;
    [[nodiscard]] StateFieldView mlp_sconv(std::uint32_t logical_layer,
                                           const Tensor& offsets) const;
    [[nodiscard]] StateFieldView hidden(const Tensor& offsets) const;
    [[nodiscard]] Tensor hidden(std::int64_t offset) const;
    [[nodiscard]] CyclicKVCacheLayerView swa_layer(std::uint32_t slot,
                                                   const Tensor& offsets) const;

    [[nodiscard]] StateFieldView dspark_f_rnn(const Tensor& offsets) const;
    [[nodiscard]] CyclicKVCacheLayerView dspark_swa_layer(std::uint32_t slot,
                                                          const Tensor& offsets) const;
    [[nodiscard]] static StateFieldView flatten_short_conv(StateFieldView);

private:
    StateLayout layout_;
    StateImageStorage image_;
    std::optional<CyclicKVCacheView> swa_, dspark_swa_;
};

class CacheViews {
public:
    CacheViews(MemorySpan context, MemorySpan fixed, const CacheLayout& layout);

    [[nodiscard]] PagedKVBatchLayerView layer(std::uint32_t layer) const;
    [[nodiscard]] context::ExecutionTableStorage& tables() noexcept { return tables_; }
    [[nodiscard]] const context::ExecutionTableStorage& tables() const noexcept { return tables_; }

private:
    CacheLayout layout_;
    MemorySpan context_;
    context::ExecutionTableStorage tables_;
};

[[nodiscard]] std::vector<context::RecordRangeCopy>
state_conversion(std::span<const StatePlacement>, std::uint32_t source_owner,
                 std::uint32_t destination_owner);

[[nodiscard]] context::HistoryCopyLayout history_conversion(const KVPageLayout&, std::uint32_t from,
                                                            std::uint32_t to, std::uint32_t tail);

} // namespace ninfer::models::inkling2
