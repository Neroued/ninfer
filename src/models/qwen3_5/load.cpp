#include "models/qwen3_5/load.h"

#include "artifact/reader.h"
#include "models/qwen3_5/load/bindings.h"
#include "models/qwen3_5/load/sharding.h"
#include "models/qwen3_5/qwen4_layer_profile.h"

#include <charconv>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5 {

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    FrontendResources resources;
    InstanceInfo info;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const FrontendResources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

void LoadPlan::reserve_host_pinned_bytes(std::uint64_t bytes) {
    impl_->materialization.host_pinned_reserved_bytes = bytes;
}

std::size_t LoadPlan::parameter_count() const { return impl_->pending.size(); }

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

namespace {

void validate_tensor_parallel(const LoadOptions& options) {
    loading::validate_tensor_parallel_ranks(options);
    // The full output head is split by vocabulary rows and the DFlash2 candidate ranking has a
    // vocabulary-split form only for the optimized head (ops::linear_topk_split): the drafter
    // proposes through it.
    if (options.tp > 1 && options.dflash2() && !options.proposal_enabled()) {
        throw std::invalid_argument(
            "tensor-parallel DFlash2 requires the optimized proposal head (--lm-head-draft)");
    }
}

// The expert layer of a Qwen4Exp routed-expert parameter name: Text block i is layer i, MTP block
// i is layer text_layers + i; -1 for any other name.
std::int32_t qwen4_expert_layer(std::string_view name, std::uint32_t text_layers) {
    if (name.find("/moe/experts/") == std::string_view::npos) { return -1; }
    using Prefix = std::pair<std::string_view, std::uint32_t>;
    for (const auto& [prefix, base] :
         {Prefix{"text/layers/", 0U}, Prefix{"mtp/layers/", text_layers}}) {
        if (!name.starts_with(prefix)) { continue; }
        const auto digits   = name.substr(prefix.size());
        std::uint32_t index = 0;
        const auto [end, error] =
            std::from_chars(digits.data(), digits.data() + digits.size(), index);
        if (error != std::errc{} || end == digits.data() || *end != '/') { return -1; }
        return static_cast<std::int32_t>(base + index);
    }
    return -1;
}

// v0.7 pinned fallback: ranks the routed-expert banks (models/qwen3_5/qwen4_layer_profile.h) so the
// materializer may move the least important layers' banks to pageable memory when the page-locked
// allocation cannot hold them all; every other HostMapped parent (--embedding-host) keeps rank 0.
void rank_host_mapped(artifact::MaterializationPlan& plan,
                      std::span<const loading::PendingWeight> weights, const TextConfig& text) {
    std::unordered_map<std::size_t, std::int32_t> layer_of;
    std::int32_t layers = static_cast<std::int32_t>(text.num_hidden_layers);
    for (const auto& weight : weights) {
        const auto layer = qwen4_expert_layer(weight.reference.name, text.num_hidden_layers);
        if (layer < 0) { continue; }
        layers = std::max(layers, layer + 1);
        for (const auto& part : weight.reference.binding.parts) {
            layer_of[part.object.index] = layer;
        }
    }
    const auto order =
        qwen4_host_pin_order(layers, static_cast<std::int32_t>(text.num_hidden_layers));
    std::vector<std::uint32_t> rank(static_cast<std::size_t>(layers), 0);
    for (std::size_t i = 0; i < order.size(); ++i) {
        rank[static_cast<std::size_t>(order[i])] = static_cast<std::uint32_t>(i + 1);
    }
    for (auto& placement : plan.host_mapped_objects) {
        const auto found = layer_of.find(placement.object.index);
        if (found != layer_of.end()) {
            placement.pageable_rank = rank[static_cast<std::size_t>(found->second)];
        }
    }
}

} // namespace

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out = std::make_unique<LoadPlan::Impl>();
    validate_tensor_parallel(options);
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader, options.tp);
    out->resources = loading::bind_resources(binder, out->config);
    loading::Bindings bindings(binder);
    const auto& text  = out->config.text;
    out->weights.text = loading::bind_text(bindings, text, options);
    if (out->config.vision) {
        out->weights.vision = loading::bind_vision(bindings, *out->config.vision, text);
    }
    if (text.qwen4()) {
        if (out->config.draft || options.proposal_enabled()) {
            throw artifact::ArtifactError("Qwen4Exp supports MTP speculation only");
        }
        out->weights.qwen4 = loading::bind_qwen4_text(bindings, text);
        if (out->config.mtp) { out->weights.qwen4_mtp = loading::bind_qwen4_mtp(bindings, text); }
    } else if (out->config.mtp) {
        out->weights.mtp = loading::bind_mtp(bindings, text, out->weights.text);
    }
    if (out->config.draft) {
        out->weights.draft =
            loading::bind_draft(bindings, *out->config.draft, text, out->weights.text,
                                std::string(options.speculative_component()));
    }
    if (options.proposal_enabled()) {
        const auto& proposal = reader.directory().component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out->weights.proposal = loading::bind_proposal(bindings, *proposal, text, options,
                                                       out->resources.public_token_count);
        if (out->config.draft) {
            const auto domain =
                proposal->indexed ? proposal->rows : out->resources.public_token_count;
            if (out->config.draft->dflash2.selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out->weights.mtp) { out->weights.mtp->output_head = out->weights.proposal->head; }
        if (out->weights.draft) { out->weights.draft->output_head = out->weights.proposal->head; }
    }
    out->weights.text.output_head_use =
        bindings.use(out->weights.text.output_head, "text/final_hidden");
    if (out->weights.mtp) {
        out->weights.mtp->output_head_use =
            bindings.use(out->weights.mtp->output_head, "mtp/final_hidden");
    }
    if (out->weights.draft) {
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
    }
    loading::install_shard_resolver(binder, bindings.weights, out->config, options);
    out->pending                                 = std::move(bindings.weights);
    out->materialization                         = std::move(binder).finish();
    out->materialization.host_pinned_limit_bytes = options.host_pinned_max_bytes;
    if (text.qwen4_routed_experts()) { rank_host_mapped(out->materialization, out->pending, text); }
    out->info.name = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data    = std::move(plan.impl_);
    auto backing = artifact::materialize(*data->materialization.source,
                                         std::move(data->materialization), device, observer);
    Model::DeviceWeights bound;
    bound[0] = loading::resolve_weights(std::move(data->pending), backing);
    return std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound), 1,
        std::move(data->resources), std::move(data->info), std::move(backing)));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, ExecutionContext& execution,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    const int tp = plan.impl_->options.tp;
    if (execution.tp != tp) {
        throw std::invalid_argument(
            "ExecutionContext tensor parallelism differs from the load plan");
    }
    if (tp == 1) { return materialize_model(std::move(plan), execution.primary(), observer); }
    auto data    = std::move(plan.impl_);
    auto backing = artifact::materialize(*data->materialization.source,
                                         std::move(data->materialization), execution, observer);
    Model::DeviceWeights bound;
    for (int device = 0; device < tp; ++device) {
        bound[static_cast<std::size_t>(device)] =
            loading::resolve_weights(data->pending, backing, device);
    }
    return std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound), tp,
        std::move(data->resources), std::move(data->info), std::move(backing)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

} // namespace ninfer::models::qwen3_5
