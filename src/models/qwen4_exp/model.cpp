#include "models/qwen4_exp/model.h"

namespace ninfer::models::qwen4_exp {

ModelShard::ModelShard(std::shared_ptr<const ModelDefinition> definition, RankPlan rank,
                       artifact::MaterializedArtifact backing, std::vector<BoundWeight> bound,
                       std::vector<ExpertBankSource> banks)
    : definition_(std::move(definition)), backing_(std::move(backing)), stats_(backing_.stats()),
      bound_(std::move(bound)), banks_(std::move(banks)), rank_(std::move(rank)) {}

ModelShard::~ModelShard() = default;

std::size_t ModelShard::ple_device_bytes() const {
    const auto* table = ple_ ? std::get_if<DevicePleTable>(&*ple_) : nullptr;
    return table ? table->backing.stats().device_capacity_bytes + table->decode.size() : 0;
}

std::size_t ModelShard::ple_host_bytes() const {
    const auto* table =
        ple_ ? std::get_if<std::shared_ptr<const ple::disk::Table>>(&*ple_) : nullptr;
    return table ? (*table)->memory_bytes() : 0;
}

void ModelShard::release_ple() {
    ple_.reset();
    stats_ = backing_.stats();
}

void ModelShard::prepare_ple(PleTablePlan&& plan, DeviceContext& device,
                             const StartupObserver* observer) {
    if (ple_) throw std::logic_error("PLE table already has a materialized owner");
    auto table = materialize_ple_table(plan, definition_->ple_parameters(), device, observer);
    rank_.ple_placement = plan.placement;
    rank_.ple_shards    = plan.shards;
    ple_.emplace(std::move(table));
    if (const auto* owned = std::get_if<DevicePleTable>(&*ple_)) {
        const auto& s = owned->backing.stats();
        stats_.device_capacity_bytes += s.device_capacity_bytes;
        stats_.read_bytes += s.read_bytes;
        stats_.h2d_bytes += s.h2d_bytes + owned->decode.size();
        stats_.upload_seconds += s.upload_seconds;
        stats_.device_object_count += s.device_object_count;
        stats_.peak_staging_bytes = std::max(stats_.peak_staging_bytes, s.peak_staging_bytes);
    }
}

std::size_t ModelShard::metadata_bytes() const {
    const auto* table = ple_ ? std::get_if<DevicePleTable>(&*ple_) : nullptr;
    auto bytes        = table ? table->decode.size() : 0;
    for (const auto& bank : local_banks_) bytes += bank.bytes();
    return bytes;
}

} // namespace ninfer::models::qwen4_exp
