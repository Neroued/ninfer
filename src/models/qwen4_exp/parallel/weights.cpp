#include "models/qwen4_exp/parallel/weights.h"

#include "artifact/reader.h"

namespace ninfer::models::qwen4_exp {

WeightSelection select_weights(const artifact::Reader& reader, const ModelDefinition& definition,
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
    WeightSelection out;
    out.weights = models::select_weights(reader, definition.parameters(), definition.weight_data(),
                                         rank.whole, rank.assemblies, partitions, rank.omitted,
                                         device);
    if (rank.ple_placement == ple::Placement::Device) {
        const auto& shards = definition.ple_parameters().shards;
        const auto range   = rank.ple_shards;
        if (!range.size || range.begin > shards.size() || range.size > shards.size() - range.begin)
            throw artifact::ArtifactError(
                "Qwen4Exp device PLE shard range differs from definition");
        for (const auto& shard : std::span(shards).subspan(range.begin, range.size)) {
            out.ple.push_back(out.weights.requests.size());
            out.weights.requests.push_back({artifact::whole_tensor(reader, shard.object), device,
                                            artifact::HostAccess::CpuOnly, 0});
        }
    }
    return out;
}

} // namespace ninfer::models::qwen4_exp
