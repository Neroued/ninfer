#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "models/common/moe/weights.h"
#include "models/glm5next/config.h"
#include "models/glm5next/frontend/resources.h"
#include "models/glm5next/weights.h"

namespace ninfer::models::glm5next {

struct InstanceInfo {
    std::string name;
};

class ModelDefinition {
public:
    ~ModelDefinition();
    ModelDefinition(const ModelDefinition&)            = delete;
    ModelDefinition& operator=(const ModelDefinition&) = delete;

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const LoadOptions& options() const;
    [[nodiscard]] const ModelWeights& weights() const;
    [[nodiscard]] const FrontendResources& resources() const;
    [[nodiscard]] const InstanceInfo& info() const;
    [[nodiscard]] const artifact::ArtifactId& artifact_id() const;
    [[nodiscard]] const artifact::MaterializationStats& resource_stats() const;
    [[nodiscard]] const artifact::ParameterReference& parameter(WeightId) const;
    [[nodiscard]] std::span<const artifact::ParameterReference> parameters() const;
    [[nodiscard]] std::span<const WeightUse> uses(WeightId) const;
    [[nodiscard]] std::span<const BoundWeight> weight_data() const;
    [[nodiscard]] std::span<const moe::BankDescription> expert_banks() const;
    [[nodiscard]] ops::WeightInput input(WeightId) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit ModelDefinition(std::unique_ptr<Impl>);
    friend std::shared_ptr<const ModelDefinition> read_definition(const artifact::Reader&,
                                                                  LoadOptions);
};

[[nodiscard]] std::shared_ptr<const ModelDefinition> read_definition(const artifact::Reader&,
                                                                     LoadOptions = {});

} // namespace ninfer::models::glm5next
