#include "models/qwen4_exp/parallel/weights.h"

#include "artifact/reader.h"

namespace ninfer::models::qwen4_exp {

SelectedWeights select_weights(const artifact::Reader& reader, const ModelDefinition& definition,
                               const RankPlan& rank, Device device) {
    const auto banks = definition.expert_banks();
    if (rank.layers.begin + rank.experts.size() > banks.size() ||
        rank.experts.size() != rank.layers.size + bool(definition.weights().mtp))
        throw artifact::ArtifactError("Qwen4Exp rank expert bank count differs from definition");
    std::vector<ExpertWeightPartition> partitions(rank.experts.size());
    const auto append = [&](const MoeWeights& weights) {
        const auto index     = weights.bank.index;
        partitions.at(rank.local_bank(weights.bank)) = {
            banks[index],
            {weights.gate, weights.up, weights.down},
            rank.experts[rank.local_bank(weights.bank)]};
    };
    for (const auto& layer :
         std::span(definition.weights().text.layers).subspan(rank.layers.begin, rank.layers.size))
        append(layer.moe);
    if (definition.weights().mtp) append(definition.weights().mtp->layer.moe);
    return models::select_weights(reader, definition.parameters(), definition.weight_data(),
                                  rank.whole, rank.assemblies, partitions, rank.omitted, device);
}

} // namespace ninfer::models::qwen4_exp
