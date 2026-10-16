#include "models/common/moe/offload/graph.h"
#include "models/common/moe/offload/plan.h"

#include <stdexcept>

namespace ninfer::models::moe::offload {
namespace {

std::size_t moe_graph_index(ops::ExpertSubsetKind kind) {
    switch (kind) {
    case ops::ExpertSubsetKind::All:
        return 0;
    case ops::ExpertSubsetKind::None:
        return 1;
    case ops::ExpertSubsetKind::MaskOnes:
        return 2;
    default:
        throw std::logic_error("invalid GPU subset for a MoE graph");
    }
}

void instantiate(DecodeGraphExecutable& executable, const DecodeGraphDefinition& definition) {
    if (definition.ready()) executable.instantiate(definition);
}

void update(DecodeGraphExecutable& executable, const DecodeGraphDefinition& definition) {
    if (executable.ready() != definition.ready())
        throw std::logic_error("MoE graph segment topology changed");
    if (definition.ready()) executable.update(definition);
}

void upload(DecodeGraphExecutable& executable, cudaStream_t stream) {
    if (executable.ready()) executable.upload(stream);
}
} // namespace

GraphPlan graph_plan(const Plan& schedule, ExpertBankId bank, std::int32_t tokens) {
    const auto& plan   = schedule.layer(bank);
    const auto experts = std::uint32_t(plan.spec.profile.routed.experts);
    using Kind         = ops::ExpertSubsetKind;
    GraphPlan out;
    if (plan.requires_full_bank(tokens)) {
        out.subsets[moe_graph_index(Kind::All)] = true;
    } else {
        out.subsets[moe_graph_index(Kind::All)]  = plan.cache_experts == experts;
        out.subsets[moe_graph_index(Kind::None)] = plan.fixed_experts == 0;
        out.subsets[moe_graph_index(Kind::MaskOnes)] =
            plan.cache_experts > 0 && plan.fixed_experts < experts;
    }
    if (!plan.requires_full_bank(tokens) || plan.stream_layout.empty())
        out.steady = plan.cache_experts == 0         ? Kind::None
                     : plan.cache_experts == experts ? Kind::All
                                                     : Kind::MaskOnes;
    return out;
}

LayerGraphExecutable::LayerGraphExecutable(const LayerGraphDefinition& source)
    : plan_(source.plan) {
    instantiate(mixer_, source.mixer);
    instantiate(steady_, source.steady);
    instantiate(feature_, source.feature);
    for (std::size_t kind = 0; kind < subsets_.size(); ++kind)
        instantiate(subsets_[kind], source.moe->subsets[kind]);
}

void LayerGraphExecutable::update(const LayerGraphDefinition& source) {
    if (plan_ != source.plan) throw std::logic_error("MoE graph plan changed");
    offload::update(mixer_, source.mixer);
    offload::update(steady_, source.steady);
    offload::update(feature_, source.feature);
}

void LayerGraphExecutable::upload(cudaStream_t stream) {
    offload::upload(mixer_, stream);
    offload::upload(steady_, stream);
    offload::upload(feature_, stream);
    for (auto& subset : subsets_) offload::upload(subset, stream);
}

void capture_layer(LayerGraphDefinition& definition, Resources& resources, ExpertBankId bank,
                   CudaExecutionView cuda, std::int32_t tokens, const std::function<void()>& mixer,
                   const std::function<void(const PreparedExecution&)>& gpu_moe,
                   const std::function<void()>& feature) {
    definition.plan = graph_plan(resources.plan(), bank, tokens);
    definition.mixer.capture(cuda.stream(), mixer);
    using Kind = ops::ExpertSubsetKind;
    for (auto kind : {Kind::All, Kind::None, Kind::MaskOnes}) {
        auto& part = definition.moe->subsets[moe_graph_index(kind)];
        if (!definition.plan.subsets[moe_graph_index(kind)] || part.ready()) continue;
        const auto call = resources.describe_layer(bank, tokens, kind);
        part.capture(cuda.stream(), [&] { gpu_moe(call); });
    }
    if (definition.plan.steady) {
        const auto kind = *definition.plan.steady;
        const auto call = resources.describe_layer(bank, tokens, kind);
        definition.steady.capture(cuda.stream(), [&] {
            mixer();
            gpu_moe(call);
        });
    }
    if (feature) definition.feature.capture(cuda.stream(), feature);
}

void LayerGraphExecutable::replay(Resources& resources, ExpertBankId bank, Tensor& output,
                                  CudaExecutionView cuda,
                                  const std::function<void()>& after_gpu_submit) {
    resources.before_layer(bank);
    if (resources.calibrating()) {
        mixer_.launch(cuda.stream());
        const auto run = [&](ops::ExpertSubsetKind kind, DecodeGraphExecutable& graph) {
            const auto call = resources.prepare_calibration_layer(bank, kind);
            try {
                graph.launch(cuda.stream());
                resources.gpu_submitted(bank);
                resources.finish_layer(complete(call, output));
            } catch (...) {
                resources.abort_layer(bank);
                throw;
            }
        };
        for (auto kind : {ops::ExpertSubsetKind::All, ops::ExpertSubsetKind::None,
                          ops::ExpertSubsetKind::MaskOnes}) {
            const auto index = moe_graph_index(kind);
            if (plan_.subsets[index]) run(kind, subsets_[index]);
        }
        if (plan_.steady) run(*plan_.steady, steady_);
        if (feature_.ready()) feature_.launch(cuda.stream());
        return;
    }
    const bool combined = resources.settled() && steady_.ready();
    if (!combined) mixer_.launch(cuda.stream());
    const auto call = resources.prepare_layer(bank);
    try {
        if (combined)
            steady_.launch(cuda.stream());
        else
            subsets_[moe_graph_index(call.partition.gpu_kind)].launch(cuda.stream());
        resources.gpu_submitted(bank);
        if (after_gpu_submit) after_gpu_submit();
        resources.finish_layer(complete(call, output));
    } catch (...) {
        resources.abort_layer(bank);
        throw;
    }
    if (feature_.ready()) feature_.launch(cuda.stream());
}

} // namespace ninfer::models::moe::offload
