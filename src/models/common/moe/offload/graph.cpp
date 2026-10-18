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
    if (pending_) throw std::logic_error("cannot update an active MoE layer graph");
    if (plan_ != source.plan) throw std::logic_error("MoE graph plan changed");
    offload::update(mixer_, source.mixer);
    offload::update(steady_, source.steady);
    offload::update(feature_, source.feature);
}

void LayerGraphExecutable::upload(cudaStream_t stream) {
    if (pending_) throw std::logic_error("cannot upload an active MoE layer graph");
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

bool LayerGraphExecutable::start(Resources& resources, ExpertBankId bank, Tensor& output,
                                 CudaExecutionView cuda, std::function<void()> after_gpu_submit) {
    if (pending_) throw std::logic_error("MoE layer graph is already active");
    resources.begin_call();
    pending_.emplace(Invocation{resources, bank, output, cuda, std::move(after_gpu_submit)});
    return poll();
}

bool LayerGraphExecutable::poll() {
    if (!pending_) return true;
    try {
        if (!advance()) return false;
        pending_->resources.end_call();
        pending_.reset();
        return true;
    } catch (...) {
        pending_->resources.fail_call();
        pending_.reset();
        throw;
    }
}

void LayerGraphExecutable::cancel() {
    if (!pending_) return;
    auto& resources = pending_->resources;
    if (pending_->call) enqueue_discard(*pending_->call);
    pending_.reset();
    resources.cancel();
}

bool LayerGraphExecutable::advance() {
    auto& invocation = *pending_;
    auto& resources  = invocation.resources;
    const auto bank  = invocation.bank;
    const auto cuda  = invocation.cuda;
    using Stage      = Invocation::Stage;
    resources.progress();
    if (invocation.stage == Stage::BeforeLayer) {
        if (!resources.try_before_layer(bank)) return false;
        invocation.combined = !resources.calibrating() && resources.settled() && steady_.ready();
        if (!invocation.combined) mixer_.launch(cuda.stream());
        invocation.stage = Stage::Prepare;
    }
    if (resources.calibrating()) {
        using Kind = ops::ExpertSubsetKind;
        constexpr std::array kinds{Kind::All, Kind::None, Kind::MaskOnes};
        // Calibration executes each finite variant; only its preparation may drain/repack.
        for (; invocation.calibration_variant < 4; ++invocation.calibration_variant) {
            const auto variant = invocation.calibration_variant;
            if (invocation.stage == Stage::Prepare) {
                if (variant < 3 ? !plan_.subsets[variant] : !plan_.steady) continue;
                const auto kind = variant < 3 ? kinds[variant] : *plan_.steady;
                invocation.call.emplace(resources.prepare_calibration_layer(bank, kind));
                auto& graph = variant < 3 ? subsets_[variant] : steady_;
                graph.launch(cuda.stream());
                resources.gpu_submitted(bank);
                invocation.stage = Stage::Complete;
            }
            const auto complete = try_complete(*invocation.call, invocation.output);
            if (!complete) return false;
            resources.finish_layer(*complete);
            invocation.call.reset();
            invocation.stage = Stage::Prepare;
        }
    } else {
        if (invocation.stage == Stage::Prepare) {
            auto call = resources.try_prepare_layer(bank);
            if (!call) return false;
            invocation.call.emplace(std::move(*call));
            if (invocation.combined)
                steady_.launch(cuda.stream());
            else
                subsets_[moe_graph_index(invocation.call->partition.gpu_kind)].launch(
                    cuda.stream());
            resources.gpu_submitted(bank);
            invocation.stage = Stage::Complete;
            if (invocation.after_gpu_submit) invocation.after_gpu_submit();
        }
        const auto complete = try_complete(*invocation.call, invocation.output);
        if (!complete) return false;
        resources.finish_layer(*complete);
    }
    if (feature_.ready()) feature_.launch(cuda.stream());
    return true;
}

} // namespace ninfer::models::moe::offload
