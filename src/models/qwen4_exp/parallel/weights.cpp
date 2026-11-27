#include "models/qwen4_exp/parallel/weights.h"

#include "artifact/reader.h"

namespace ninfer::models::qwen4_exp {

SelectedWeights select_weights(const artifact::Reader& reader, const ModelDefinition& definition,
                               const RankPlan& rank, Device device) {
    const auto banks = definition.expert_banks();
    if (rank.experts.size() != banks.size())
        throw artifact::ArtifactError("Qwen4Exp rank expert bank count differs from definition");
    std::vector<ExpertWeightPartition> partitions(banks.size());
    const auto append = [&](const MoeWeights& weights) {
        const auto index     = weights.bank.index;
        partitions.at(index) = {
            banks[index], {weights.gate, weights.up, weights.down}, rank.experts[index]};
    };
    for (const auto& layer : definition.weights().text.layers) append(layer.moe);
    if (definition.weights().mtp) append(definition.weights().mtp->layer.moe);
    return models::select_weights(reader, definition.parameters(), definition.weight_data(),
                                  rank.whole, rank.assemblies, partitions, rank.omitted, device);
}

} // namespace ninfer::models::qwen4_exp
