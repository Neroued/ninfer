#include "models/inkling2/definition.h"

#include "models/inkling2/load/bindings.h"

namespace ninfer::models::inkling2 {

struct ModelDefinition::Impl {
    artifact::PreparedResources prepared_resources;
    artifact::ArtifactId artifact_id;
    artifact::MaterializationStats resource_stats;
    Config config;
    LoadOptions options;
    ModelWeights weights;
    FrontendResources resources;
    InstanceInfo info;
    std::vector<artifact::ParameterReference> references;
    std::vector<moe::BankDescription> banks;
    std::vector<WeightParent> parents;
    std::vector<BoundWeight> descriptions;
};

ModelDefinition::ModelDefinition(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

ModelDefinition::~ModelDefinition() = default;

const Config& ModelDefinition::config() const { return impl_->config; }

const LoadOptions& ModelDefinition::options() const { return impl_->options; }

const ModelWeights& ModelDefinition::weights() const { return impl_->weights; }

const FrontendResources& ModelDefinition::resources() const { return impl_->resources; }

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

ops::WeightInput ModelDefinition::input(WeightUseId id) const {
    return weight_input(impl_->descriptions, id);
}

std::shared_ptr<const ModelDefinition> read_definition(const artifact::Reader& reader,
                                                       LoadOptions options) {
    auto out         = std::make_unique<ModelDefinition::Impl>();
    out->artifact_id = reader.artifact_id();
    out->options     = options;
    out->config      = parse_config(reader.directory(), options);

    artifact::Binder binder(reader);
    const auto& c  = out->config.text;
    out->resources = bind_resources(binder, c.vocabulary, out->config.vision.has_value());
    if (out->resources.text.public_token_count != c.unpadded_vocabulary)
        throw artifact::ArtifactError(
            "Inkling2 tokenizer public token count differs from unpadded_vocab_size");
    if (out->config.vision) {
        const auto& vision = *out->config.vision;
        if (!out->resources.processor)
            throw artifact::ArtifactError("Inkling2 Vision requires processor_config.json");
        const auto& proc = *out->resources.processor;
        const auto cell  = vision.native_patch_size * vision.merge_size;
        if (proc.native_patch_size != vision.native_patch_size ||
            proc.merge_size != vision.merge_size || proc.cell_height != cell ||
            proc.cell_width != cell || proc.features_per_patch() != vision.patch_width())
            throw artifact::ArtifactError("Inkling2 processor geometry differs from Vision config");
    }

    Bindings b(binder);
    out->weights.text = loading::bind_text(b, c);
    if (out->config.vision) out->weights.vision = loading::bind_vision(b, *out->config.vision);
    if (out->config.dspark)
        out->weights.dspark = loading::bind_dspark(b, *out->config.dspark, out->weights.text);

    out->banks.reserve(out->weights.text.layers.size());
    for (auto& layer : out->weights.text.layers) {
        layer.moe.bank = {static_cast<std::uint32_t>(out->banks.size())};
        out->banks.push_back(moe::describe_bank(reader, {b.at(layer.moe.gate).reference,
                                                         b.at(layer.moe.up).reference,
                                                         b.at(layer.moe.down).reference}));
    }

    out->info.name = reader.directory().metadata.value("name", "Inkling2ForCausalLM");
    out->parents.resize(reader.directory().objects.size());
    out->references.reserve(b.weights.size());
    out->descriptions.reserve(b.weights.size());
    for (auto& parameter : b.weights) {
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

    out->prepared_resources        = std::move(binder).finish();
    out->prepared_resources.source = nullptr;
    auto& stats                    = out->resource_stats;
    stats.file_bytes               = reader.file_bytes();
    stats.read_bytes               = out->prepared_resources.read_bytes;
    stats.owned_value_bytes        = out->prepared_resources.owned_value_bytes;
    stats.host_object_count        = out->prepared_resources.objects.size();
    for (const auto& resource : out->prepared_resources.objects)
        stats.retained_host_bytes += resource.data.size();
    return std::shared_ptr<const ModelDefinition>(new ModelDefinition(std::move(out)));
}

} // namespace ninfer::models::inkling2
