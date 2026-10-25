#include "models/glm5next/session/internal.h"

#include "models/glm5next/parallel/preparation.h"

namespace ninfer::models::glm5next {

runtime::ContextMetadataPlan session_metadata_plan(const context::ContextDirectorySpec& schema,
                                                   const ContextPoolGeometry& geometry,
                                                   std::uint32_t rank) {
    auto metadata      = context::ContextDirectory::metadata_plan(schema);
    const auto records = std::size_t(schema.state.logical_capacity) * 8;
    metadata.fixed_bytes += detail::Checkpoints::metadata_bytes(records) +
                            context::RecoveryIndex<>::metadata_bytes(records);
    const auto checkpoint_cost =
        detail::Checkpoints::metadata_bytes(8) - detail::Checkpoints::metadata_bytes(0);
    const auto recovery_cost =
        context::RecoveryIndex<>::metadata_bytes(8) - context::RecoveryIndex<>::metadata_bytes(0);
    const auto ranks        = schema.ranks.size();
    const auto history_cost = context::HistoryDirectory::metadata_bytes(4, 0, 0, ranks) -
                              context::HistoryDirectory::metadata_bytes(0, 0, 0, ranks);
    for (std::size_t i = 0; i < geometry.classes.size(); ++i) {
        const auto state_class = std::any_of(
            schema.state.layouts.begin(), schema.state.layouts.end(), [&](const auto& layout) {
                return std::any_of(layout.parts.begin(), layout.parts.end(), [&](const auto& part) {
                    return part.rank == rank &&
                           part.image_bytes == geometry.classes[i].spec.bytes &&
                           part.alignment == geometry.classes[i].spec.alignment;
                });
            });
        if (state_class)
            metadata.bytes_per_record.at(i) +=
                checkpoint_cost + recovery_cost + history_cost * schema.pools.size();
    }
    return metadata;
}

PreparedSession prepare_session(std::shared_ptr<const ModelDefinition> definition,
                                PreparedContext prepared, std::uint32_t interval) {
    SessionPlan plan;
    plan.shape               = prepared.shape;
    plan.context             = std::move(prepared.spec);
    plan.states              = std::move(prepared.states);
    plan.checkpoint_interval = interval;
    plan.vision_max_features = prepared.vision_max_features;
    for (const auto& geometry : prepared.caches)
        plan.caches.push_back(plan_kv_page_layout(geometry));
    return prepare_session(std::move(definition), std::move(plan), std::move(prepared.directory));
}

} // namespace ninfer::models::glm5next
