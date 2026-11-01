#include "models/qwen4_exp/definition.h"

#include "models/qwen4_exp/load/bindings.h"

namespace ninfer::models::qwen4_exp {

struct ModelDefinition::Impl {
    artifact::PreparedResources prepared_resources;
    artifact::ArtifactId artifact_id;
    artifact::MaterializationStats resource_stats;
    Config config;
    LoadOptions options;
    ModelWeights weights;
    qwen::FrontendResources resources;
    InstanceInfo info;
    std::vector<artifact::ParameterReference> references;
    std::vector<moe::BankDescription> banks;
    std::vector<WeightParent> parents;
    std::vector<BoundWeight> descriptions;
    ple::LookupParameters ple;
};

ModelDefinition::ModelDefinition(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

ModelDefinition::~ModelDefinition() = default;

const Config& ModelDefinition::config() const { return impl_->config; }

const LoadOptions& ModelDefinition::options() const { return impl_->options; }

const ModelWeights& ModelDefinition::weights() const { return impl_->weights; }

const qwen::FrontendResources& ModelDefinition::resources() const { return impl_->resources; }

const InstanceInfo& ModelDefinition::info() const { return impl_->info; }

const artifact::ArtifactId& ModelDefinition::artifact_id() const { return impl_->artifact_id; }

const artifact::MaterializationStats& ModelDefinition::resource_stats() const {
    return impl_->resource_stats;
}

const artifact::ParameterReference& ModelDefinition::parameter(WeightId id) const {
    return impl_->references.at(id.index);
}

std::span<const artifact::ParameterReference> ModelDefinition::parameters() const {
    return impl_->references;
}

std::span<const WeightUse> ModelDefinition::uses(WeightId id) const {
    return impl_->descriptions.at(id.index).uses;
}

std::span<const BoundWeight> ModelDefinition::weight_data() const { return impl_->descriptions; }

std::span<const moe::BankDescription> ModelDefinition::expert_banks() const { return impl_->banks; }

ops::WeightInput ModelDefinition::input(WeightId id) const {
    return weight_input(impl_->descriptions, id);
}

const ple::LookupParameters& ModelDefinition::ple_parameters() const { return impl_->ple; }

std::shared_ptr<const ModelDefinition> read_definition(const artifact::Reader& reader,
                                                       LoadOptions options) {
    auto out         = std::make_unique<ModelDefinition::Impl>();
    out->artifact_id = reader.artifact_id();
    out->options     = options;
    out->config      = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    Bindings b(binder);
    const auto& c     = out->config.text;
    out->weights.text = loading::bind_text(b, c);
    if (out->config.mtp)
        out->weights.mtp = loading::bind_mtp(b, c, out->weights.text, options.proposal_head);
    if (out->config.vision)
        out->weights.vision = qwen::bind_vision(b, *out->config.vision, c.hidden);
    const auto table =
        binder.parameter("text/layers/" + std::to_string(c.ple.layer) + "/ple/table",
                         {std::uint64_t(c.ple.shards) * c.ple.rows_per_shard, c.ple.row_width},
                         QType::NF4_G16_E4M3FN);
    auto lookup = ple::load_lookup_parameters(
        reader, table.binding, binder.resource_bytes(binder.resource("text", "ple_hash_controls")),
        {c.ple.ngram_size, c.ple.heads_per_ngram});
    if (lookup.shards.size() != c.ple.shards)
        throw artifact::ArtifactError("PLE table shard count differs from config");
    for (const auto& shard : lookup.shards)
        if (shard.geometry.shape[0] != c.ple.rows_per_shard)
            throw artifact::ArtifactError("PLE shard rows differ from config");
    out->ple        = std::move(lookup);
    const auto bank = [&](MoeWeights& w) {
        w.bank = {static_cast<std::uint32_t>(out->banks.size())};
        out->banks.push_back(moe::describe_bank(
            reader, moe::ExpertBinding{b.at(w.gate).reference, b.at(w.up).reference,
                                       b.at(w.down).reference}));
    };
    for (auto& layer : out->weights.text.layers) bank(layer.moe);
    if (out->weights.mtp) bank(out->weights.mtp->layer.moe);
    auto pending            = std::move(b.weights);
    out->prepared_resources = std::move(binder).finish();
    out->info.name          = reader.directory().metadata.value("name", "Qwen4ExpForCausalLM");
    out->parents.resize(reader.directory().objects.size());
    out->references.reserve(pending.size());
    out->descriptions.reserve(pending.size());
    for (auto& parameter : pending) {
        WeightView view{parameter.reference.shape, {}};
        for (const auto& part : parameter.reference.binding.parts) {
            auto& parent = out->parents.at(part.object.index);
            if (!parent.geometry.elements) parent.geometry = reader.geometry(part.object);
            view.parts.push_back({&parent, part.begin, part.end});
        }
        out->descriptions.push_back({parameter.reference.name, std::move(parameter.source_objects),
                                     std::move(view), std::move(parameter.uses)});
        out->references.push_back(std::move(parameter.reference));
    }
    out->prepared_resources.source = nullptr;
    auto& stats                    = out->resource_stats;
    stats.file_bytes               = reader.file_bytes();
    stats.read_bytes               = out->prepared_resources.read_bytes;
    stats.owned_value_bytes        = out->prepared_resources.owned_value_bytes;
    stats.host_object_count        = out->prepared_resources.objects.size();
    for (const auto& resource : out->prepared_resources.objects)
        stats.retained_host_bytes += resource.data.size();
    stats.read_bytes += sizeof(float) * out->ple.shards.size();
    return std::shared_ptr<const ModelDefinition>(new ModelDefinition(std::move(out)));
}

} // namespace ninfer::models::qwen4_exp
