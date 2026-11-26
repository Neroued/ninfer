#include "models/inkling2/model.h"

namespace ninfer::models::inkling2 {

ModelShard::ModelShard(std::shared_ptr<const ModelDefinition> definition, RankPlan rank,
                       artifact::MaterializedArtifact backing)
    : definition_(std::move(definition)), rank_(std::move(rank)), backing_(std::move(backing)) {}

std::size_t ModelShard::metadata_bytes() const {
    std::size_t bytes = 0;
    for (const auto& bank : experts_) bytes += bank.bytes();
    return bytes;
}

} // namespace ninfer::models::inkling2
