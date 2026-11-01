#include "models/qwen4_exp/parallel/weights.h"

#include "artifact/reader.h"

namespace ninfer::models::qwen4_exp {

WeightSelection select_weights(const artifact::Reader& reader, const ModelDefinition& definition,
                               const RankPlan& rank, Device device) {
    std::vector<ExpertWeightPartition> partitions;
    if (rank.size > 1) {
        const auto append = [&](const MoeWeights& weights) {
            const auto index = weights.bank.index;
            partitions.push_back({definition.expert_banks()[index],
                                  {weights.gate, weights.up, weights.down},
                                  rank.experts.at(index)});
        };
        for (const auto& layer : definition.weights().text.layers) append(layer.moe);
        if (definition.weights().mtp) append(definition.weights().mtp->layer.moe);
    }
    WeightSelection out;
    out.weights = models::select_weights(reader, definition.parameters(), rank.assemblies,
                                         partitions, device);
    if (rank.size > 1) {
        for (std::uint32_t i = rank.ple_shards.begin;
             i < rank.ple_shards.begin + rank.ple_shards.size; ++i) {
            out.ple.push_back(out.weights.requests.size());
            out.weights.requests.push_back(
                {artifact::whole_tensor(reader, definition.ple_parameters().shards.at(i).object),
                 device, artifact::HostAccess::CpuOnly, 0});
        }
    }
    return out;
}

} // namespace ninfer::models::qwen4_exp
