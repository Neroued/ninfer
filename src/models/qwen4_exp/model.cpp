#include "models/qwen4_exp/model.h"

namespace ninfer::models::qwen4_exp {

ModelShard::ModelShard(std::shared_ptr<const ModelDefinition> definition,
                       artifact::MaterializedArtifact backing, std::vector<BoundWeight> bound,
                       std::vector<ExpertBankSource> banks,
                       std::shared_ptr<const ple::disk::Table> ple)
    : definition_(std::move(definition)), backing_(std::move(backing)), ple_(std::move(ple)),
      bound_(std::move(bound)), banks_(std::move(banks)) {}

ModelShard::~ModelShard() = default;

std::size_t ModelShard::metadata_bytes() const {
    const auto* table = std::get_if<DevicePleTable>(&ple_);
    auto bytes        = table ? table->decode.size() : 0;
    for (const auto& bank : local_banks_) bytes += bank.bytes();
    return bytes;
}

} // namespace ninfer::models::qwen4_exp
