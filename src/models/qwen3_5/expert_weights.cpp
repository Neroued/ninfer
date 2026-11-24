#include "models/qwen3_5/expert_weights.h"
#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5 {

std::vector<moe::BankDescription> describe_expert_weights(const artifact::Reader& reader,
                                                          ModelWeights& weights,
                                                          std::span<const PendingWeight> pending) {
    std::vector<moe::BankDescription> out;
    const auto describe = [&](MoeWeights& parameters) {
        std::vector<moe::ExpertBinding> bindings;
        bindings.reserve(parameters.experts.size());
        for (const auto& expert : parameters.experts)
            bindings.push_back({pending[expert.gate.index].reference,
                                pending[expert.up.index].reference,
                                pending[expert.down.index].reference});
        parameters.bank = {static_cast<std::uint32_t>(out.size())};
        out.push_back(moe::describe_bank(reader, bindings));
    };
    for (auto& layer : weights.text.layers)
        if (auto* parameters = std::get_if<MoeWeights>(&layer.ffn)) describe(*parameters);
    if (weights.mtp)
        if (auto* parameters = std::get_if<MoeWeights>(&weights.mtp->layer.ffn))
            describe(*parameters);
    return out;
}

} // namespace ninfer::models::qwen3_5
