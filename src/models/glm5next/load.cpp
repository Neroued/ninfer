#include "models/glm5next/load.h"

namespace ninfer::models::glm5next {

ResolvedLoadPlan::ResolvedLoadPlan(const artifact::Reader& reader,
                                   std::shared_ptr<const ModelDefinition> definition, RankPlan rank,
                                   SelectedWeights selection,
                                   artifact::MaterializationPlan materialization, Device device)
    : reader_(&reader), definition_(std::move(definition)), rank_(std::move(rank)),
      selection_(std::move(selection)), materialization_(std::move(materialization)),
      device_(device) {
    parents_.reserve(materialization_.tensors.size());
    for (const auto& tensor : materialization_.tensors)
        parents_.push_back({tensor.geometry, device_, nullptr, {}});
    descriptions_ = bind_selected_weights(definition_->weight_data(), selection_,
                                          materialization_.requests, parents_);
}

ResolvedLoadPlan resolve_rank_load(const artifact::Reader& reader,
                                   std::shared_ptr<const ModelDefinition> definition, RankPlan rank,
                                   Device device) {
    if (!definition || definition->artifact_id() != reader.artifact_id())
        throw artifact::ArtifactError("GLM5Next definition belongs to another artifact");
    if (device.type() != DeviceType::CUDA)
        throw artifact::ArtifactError("GLM5Next rank weights require a CUDA device");
    if (rank.experts.size() != definition->expert_banks().size())
        throw artifact::ArtifactError("GLM5Next rank expert ownership differs from its definition");
    std::vector<ExpertWeightPartition> experts(definition->expert_banks().size());
    const auto append = [&](const MoeWeights& w) {
        experts.at(w.bank.index) = {definition->expert_banks()[w.bank.index],
                                    {w.gate, w.up, w.down},
                                    rank.experts[w.bank.index]};
    };
    for (const auto& layer : definition->weights().text.layers)
        if (const auto* m = std::get_if<MoeWeights>(&layer.ffn)) append(*m);
    if (definition->weights().mtp) append(definition->weights().mtp->moe);
    auto selection =
        select_weights(reader, definition->parameters(), rank.assemblies, experts, device);
    auto storage = artifact::plan_materialization(reader, selection.requests);
    return {
        reader, std::move(definition), std::move(rank), std::move(selection), std::move(storage),
        device};
}

std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&& plan, DeviceContext& device,
                                              const StartupObserver* observer) {
    if (!plan.definition_) throw artifact::ArtifactError("GLM5Next load plan was already consumed");
    if (plan.device_ != device.device())
        throw artifact::ArtifactError("GLM5Next materialization device differs from its plan");
    auto definition         = std::move(plan.definition_);
    const auto parent_count = plan.materialization_.tensors.size();
    auto requests           = std::move(plan.materialization_.requests);
    auto backing            = artifact::materialize(*plan.reader_, std::move(plan.materialization_),
                                                    device.transfer_view(), observer);
    auto out                = std::unique_ptr<ModelShard>(
        new ModelShard(std::move(definition), std::move(plan.rank_), std::move(backing)));
    out->parents_.reserve(parent_count);
    for (std::size_t i = 0; i < parent_count; ++i)
        out->parents_.push_back(out->backing_.parent({i}, device.device()));
    out->weights_     = bind_selected_weights(out->definition_->weight_data(), plan.selection_,
                                              requests, out->parents_);
    const auto parent = [&](std::size_t request) -> const WeightParent& {
        return out->parents_.at(requests.at(request).index);
    };
    for (const auto& bank : plan.selection_.experts) {
        std::vector<moe::ResidentExpertEntry> entries;
        entries.reserve(bank.entries.size());
        for (const auto& e : bank.entries)
            entries.push_back({e.expert, &parent(e.gate_up), &parent(e.down)});
        out->experts_.emplace_back(entries, out->definition_->config().text.moe.experts,
                                   device.device());
    }
    return out;
}

} // namespace ninfer::models::glm5next
