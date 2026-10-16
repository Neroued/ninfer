#include "models/qwen3_5/program/planning/moe.h"

#include "models/qwen3_5/execution/profile.h"
#include "models/qwen3_5/load.h"

#include <numeric>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

std::vector<ExpertBankCacheSpec> bank_specs(const LoadPlan& source) {
    std::vector<ExpertBankCacheSpec> out;
    for (const auto& bank : source.expert_banks())
        out.push_back({bank.gate_up.layout, bank.down.layout, true});
    for (const auto& layer : source.weights().text.layers)
        if (const auto* parameters = std::get_if<MoeWeights>(&layer.ffn))
            out[parameters->bank.index].resident = false;
    return out;
}

} // namespace

std::shared_ptr<const moe::Plan> describe_moe_plan(const LoadPlan& source,
                                                   const execution::ModelProfile& profile,
                                                   const MoeOptions& options) {
    if (options.expert_priority && options.expert_priority->size() != profile.text.layers.size())
        throw std::invalid_argument("MoE expert priority must provide every Main layer");
    std::vector<moe::LayerSpec> layers;
    for (std::size_t i = 0; i < profile.text.layers.size(); ++i) {
        const auto* parameters =
            std::get_if<execution::MoeLayerProfile>(&profile.text.layers[i].ffn);
        if (!parameters) {
            if (options.expert_priority && !(*options.expert_priority)[i].empty())
                throw std::invalid_argument("dense layer has an expert priority");
            continue;
        }
        moe::LayerSpec layer{parameters->bank, parameters->math, ops::MoeEpilogue::AddResidual, {}};
        layer.hybrid = {64, 15};
        if (options.expert_priority) {
            layer.priority = (*options.expert_priority)[i];
        } else if (const auto priority = default_expert_priority(profile.config.text, i);
                   !priority.empty()) {
            layer.priority.assign(priority.begin(), priority.end());
        } else {
            layer.priority.resize(parameters->math.routed.experts);
            std::iota(layer.priority.begin(), layer.priority.end(), 0U);
        }
        layers.push_back(std::move(layer));
    }
    return moe::describe_plan(bank_specs(source), layers);
}

} // namespace ninfer::models::qwen3_5::detail
