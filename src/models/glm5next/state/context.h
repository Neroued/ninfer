#pragma once

#include "core/linear_attention_state.h"
#include "models/glm5next/parallel/attention.h"
#include "core/cyclic_kv_cache.h"
#include "models/common/context/directory/history.h"
#include "models/common/context/directory/records.h"
#include "models/common/context/storage.h"
#include "ninfer/ops/kpool_mla_layout.h"

namespace ninfer::models::glm5next {
namespace execution {
struct ModelProfile;
}
struct MlaConfig;

struct StateLayout {
    StateImageLayout common, owner;
    std::optional<LinearAttentionStateLayout> kda;
    StateField hidden;
    std::optional<CyclicKVCacheLayout> draft;
    std::optional<ops::KpoolTailFields> main;
    std::optional<ops::KpoolTailFields> mtp;
    LayoutRegion owner_fields;
};

// Directory geometry and record-relative migration ranges; rank-local field views stay
// with StateLayout and do not cross the preparation boundary.
struct StatePlacement {
    RecordSpec common, owner;
    LayoutRegion owner_fields;
    std::optional<LayoutRegion> draft_context;
    bool operator==(const StatePlacement&) const = default;
};

[[nodiscard]] StateLayout plan_state(const execution::ModelProfile&,
                                     LinearStateDType = LinearStateDType::Float32);
[[nodiscard]] StatePlacement describe_state(const StateLayout&);
[[nodiscard]] ops::KpoolMlaStorage latent_storage(KvCacheStorage);

struct CacheLayout {
    ops::KpoolMlaCacheLayout cache;
    KVExecutionTableLayout tables;
};

[[nodiscard]] CacheLayout plan_cache(LayoutBuilder&, const MlaConfig&, std::uint32_t layers,
                                     std::uint32_t max_context, std::uint32_t rows,
                                     ops::KpoolMlaStorage, KvCacheVectorFormat index_format);

class StateViews {
public:
    StateViews(MemorySpan, const StateLayout&);

    [[nodiscard]] const LinearAttentionStateView& kda() const { return kda_.value(); }

    [[nodiscard]] StateFieldView hidden(const Tensor& offsets) const;
    [[nodiscard]] CyclicKVCacheLayerView draft_layer(std::uint32_t, const Tensor& offsets) const;
    [[nodiscard]] ops::KpoolTailStateView main_tail(const Tensor& offsets) const;
    [[nodiscard]] ops::KpoolTailStateView mtp_tail(const Tensor& offsets) const;
    [[nodiscard]] const StateImageStorage& image(bool owner) const noexcept;

private:
    StateLayout layout_;
    StateImageStorage common_, owner_;
    std::optional<LinearAttentionStateView> kda_;
    std::optional<CyclicKVCacheView> draft_;
};

class CacheViews {
public:
    CacheViews(MemorySpan context, MemorySpan fixed, const CacheLayout&);
    [[nodiscard]] ops::KpoolMlaCacheView layer(std::uint32_t) const;

    [[nodiscard]] context::ExecutionTableStorage& tables() { return tables_.value(); }

    [[nodiscard]] const context::ExecutionTableStorage& tables() const { return tables_.value(); }

    bool has_pages() const noexcept { return tables_.has_value(); }

private:
    CacheLayout layout_;
    MemorySpan context_;
    std::optional<context::ExecutionTableStorage> tables_;
};

// Record-relative copies for a change of attention owner. Common KDA/hidden fields stay on
// their original rank; only the raw tails move. History placement is handled separately.
[[nodiscard]] std::vector<context::RecordRangeCopy>
state_conversion(std::span<const StatePlacement>, const AttentionOwners&,
                 std::uint32_t source_owner, std::uint32_t destination_owner);

[[nodiscard]] context::HistoryCopyLayout history_conversion(std::span<const KVPageLayout>,
                                                            const AttentionOwners&,
                                                            std::uint32_t from, std::uint32_t to,
                                                            std::uint32_t tail);

} // namespace ninfer::models::glm5next
