#include "models/qwen4_exp/model.h"

namespace ninfer::models::qwen4_exp {

ModelShard::ModelShard(std::shared_ptr<const ModelDefinition> definition,
                       artifact::MaterializedArtifact backing, std::vector<BoundWeight> bound,
                       std::vector<ExpertBankSource> banks,
                       std::shared_ptr<const ple::disk::Table> ple)
    : definition_(std::move(definition)), backing_(std::move(backing)), ple_(std::move(ple)),
      bound_(std::move(bound)), banks_(std::move(banks)) {}

ModelShard::~ModelShard() = default;

} // namespace ninfer::models::qwen4_exp
