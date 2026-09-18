#pragma once

#include "artifact/binder.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/weights.h"

#include <map>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::loading {

[[nodiscard]] FrontendResources bind_resources(artifact::Binder& binder, const Config& config);

struct PendingWeight {
    artifact::ParameterReference reference;
    std::vector<WeightUse> uses;
    std::vector<std::string> source_objects;
};

class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder(binder) {}

    [[nodiscard]] WeightId parameter(std::string name, artifact::Shape shape,
                                     std::vector<std::string> inputs   = {},
                                     std::optional<QType> exact_format = {});
    [[nodiscard]] WeightId direct(std::string name, artifact::Shape shape,
                                  QType format = QType::BF16);

    [[nodiscard]] const PendingWeight& at(WeightId id) const { return weights.at(id.index); }

    [[nodiscard]] WeightUseId use(WeightId id, std::string_view input) const;

    artifact::Binder& binder;
    std::vector<PendingWeight> weights;

private:
    std::map<std::string, WeightId, std::less<>> parameters_;
};

[[nodiscard]] AttentionWeights bind_attention(Bindings& bindings, const TextConfig& config,
                                              const std::string& prefix);
[[nodiscard]] DenseWeights bind_dense(Bindings& bindings, std::uint64_t hidden,
                                      std::uint64_t intermediate, const std::string& prefix,
                                      bool draft = false);
[[nodiscard]] BlockWeights bind_block(Bindings& bindings, const TextConfig& config,
                                      const std::string& prefix, MixerKind mixer);
[[nodiscard]] TextWeights bind_text(Bindings& bindings, const TextConfig& config,
                                    const LoadOptions& options);
[[nodiscard]] VisionWeights bind_vision(Bindings& bindings, const VisionConfig& config,
                                        const TextConfig& target);
[[nodiscard]] MtpWeights bind_mtp(Bindings& bindings, const TextConfig& config,
                                  const TextWeights& target);
[[nodiscard]] DraftWeights bind_draft(Bindings& bindings, const DraftConfig& config,
                                      const TextConfig& target, const TextWeights& weights,
                                      const std::string& component);
void bind_dflash2(Bindings& bindings, DraftWeights& weights, const DraftConfig& config,
                  const TextConfig& target);
[[nodiscard]] ProposalWeights bind_proposal(Bindings& bindings, const artifact::Proposal& proposal,
                                            const TextConfig& target, const LoadOptions& options,
                                            std::uint32_t public_tokens);
[[nodiscard]] std::vector<BoundWeight>
resolve_weights(std::vector<PendingWeight>&& pending,
                const artifact::MaterializedArtifact& materialized);

} // namespace ninfer::models::qwen3_5::loading
