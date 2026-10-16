#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/load/bindings.h"

namespace ninfer::models::qwen4_exp {
struct LoadPlan::Impl {
    const artifact::Reader* reader;
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<PendingWeight> pending;
    artifact::PreparedResources prepared_resources;
    qwen::FrontendResources resources;
    std::unique_ptr<ple::disk::Table> ple;
    std::vector<moe::BankDescription> banks;
    std::vector<WeightParent> parents;
    std::vector<WeightView> views;
    std::string name;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const LoadOptions& LoadPlan::options() const { return impl_->options; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const qwen::FrontendResources& LoadPlan::resources() const { return impl_->resources; }

const ple::disk::Table& LoadPlan::ple_table() const { return *impl_->ple; }

std::span<const moe::BankDescription> LoadPlan::expert_banks() const { return impl_->banks; }

ops::WeightInput LoadPlan::input(WeightId id) const {
    const auto& uses = impl_->pending.at(id.index).uses;
    if (uses.size() != 1) throw artifact::ArtifactError("weight input requires one Use");
    return {impl_->views.at(id.index), uses[0].policy, uses[0].activation_input_divisor};
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->reader  = &reader;
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    Bindings b(binder);
    const auto& c     = out->config.text;
    out->weights.text = loading::bind_text(b, c);
    if (out->config.mtp) out->weights.mtp = loading::bind_mtp(b, c, out->weights.text);
    if (out->config.vision)
        out->weights.vision = qwen::bind_vision(b, *out->config.vision, c.hidden);
    const auto table =
        binder.parameter("text/layers/" + std::to_string(c.ple.layer) + "/ple/table",
                         {std::uint64_t(c.ple.shards) * c.ple.rows_per_shard, c.ple.row_width},
                         QType::NF4_G16_E4M3FN);
    auto lookup = ple::load_lookup_parameters(reader, table.binding,
                                              binder.resource("text", "ple_hash_controls"),
                                              {c.ple.ngram_size, c.ple.heads_per_ngram});
    if (lookup.shards.size() != c.ple.shards)
        throw artifact::ArtifactError("PLE table shard count differs from config");
    for (const auto& shard : lookup.shards)
        if (shard.geometry.shape[0] != c.ple.rows_per_shard)
            throw artifact::ArtifactError("PLE shard rows differ from config");
    out->ple        = std::make_unique<ple::disk::Table>(reader, std::move(lookup));
    const auto bank = [&](MoeWeights& w) {
        w.bank = {static_cast<std::uint32_t>(out->banks.size())};
        out->banks.push_back(moe::describe_bank(
            reader, moe::ExpertBinding{b.at(w.gate).reference, b.at(w.up).reference,
                                       b.at(w.down).reference}));
    };
    for (auto& layer : out->weights.text.layers) bank(layer.moe);
    if (out->weights.mtp) bank(out->weights.mtp->layer.moe);
    out->pending            = std::move(b.weights);
    out->prepared_resources = std::move(binder).finish();
    out->parents.resize(reader.directory().objects.size());
    out->views.reserve(out->pending.size());
    for (const auto& pending : out->pending) {
        WeightView view{pending.reference.shape, {}};
        for (const auto& part : pending.reference.binding.parts) {
            auto& parent = out->parents.at(part.object.index);
            if (!parent.geometry.elements) parent.geometry = reader.geometry(part.object);
            view.parts.push_back({&parent, part.begin, part.end});
        }
        out->views.push_back(std::move(view));
    }
    out->name = reader.directory().metadata.value("name", "Qwen4ExpForCausalLM");
    return LoadPlan(std::move(out));
}

namespace {
struct Placement {
    artifact::MaterializationPlan storage;
    std::vector<Device> weights, banks;
};

template <class Data>
Placement placement(const Data& data, Device device, Device main_experts) {
    if (device.type() != DeviceType::CUDA)
        throw artifact::ArtifactError("Qwen4Exp requires a CUDA device");
    if (main_experts != device && main_experts != Device::cpu())
        throw artifact::ArtifactError("Main experts require the model CUDA device or CPU backing");
    Placement out;
    out.weights.assign(data.pending.size(), device);
    out.banks.assign(data.banks.size(), device);
    for (const auto& layer : data.weights.text.layers) {
        const auto& w = layer.moe;
        for (auto id : {w.gate, w.up, w.down}) out.weights[id.index] = main_experts;
        out.banks[w.bank.index] = main_experts;
    }
    std::vector<artifact::PlacementRequest> requests;
    for (std::size_t i = 0; i < data.pending.size(); ++i)
        for (const auto& part : data.pending[i].reference.binding.parts)
            requests.push_back({part.object, out.weights[i],
                                out.weights[i] == Device::cpu() ? artifact::HostAccess::CudaTransfer
                                                                : artifact::HostAccess::CpuOnly});
    out.storage = artifact::plan_materialization(*data.reader, requests);
    return out;
}
} // namespace

artifact::MaterializationPlan LoadPlan::placement(Device device, Device main_experts) const {
    return qwen4_exp::placement(*impl_, device, main_experts).storage;
}

ResolvedLoadPlan::ResolvedLoadPlan(LoadPlan&& logical, artifact::MaterializationPlan storage,
                                   std::vector<Device> weights, std::vector<Device> banks)
    : logical_(std::move(logical)), placement_(std::move(storage)), weights_(std::move(weights)),
      banks_(std::move(banks)) {}

ResolvedLoadPlan resolve_load(LoadPlan&& plan, Device device, Device main_experts) {
    if (!plan.impl_) throw artifact::ArtifactError("Qwen4Exp load plan was consumed");
    auto placed              = placement(*plan.impl_, device, main_experts);
    placed.storage.resources = std::move(plan.impl_->prepared_resources);
    return {std::move(plan), std::move(placed.storage), std::move(placed.weights),
            std::move(placed.banks)};
}

std::unique_ptr<Model> materialize_model(ResolvedLoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.logical_.impl_) throw artifact::ArtifactError("Qwen4Exp load plan was consumed");
    auto data    = std::move(plan.logical_.impl_);
    auto backing = artifact::materialize(*data->reader, std::move(plan.placement_),
                                         device.transfer_view(), observer);
    auto bound   = resolve_weights(std::move(data->pending), backing, plan.weights_);
    auto banks   = moe::bind_banks(data->banks, backing, plan.banks_);
    return std::unique_ptr<Model>(
        new Model(std::move(data->config), data->options, std::move(data->weights),
                  std::move(bound), std::move(banks), std::move(data->resources),
                  std::move(data->name), std::move(backing), std::move(data->ple)));
}
} // namespace ninfer::models::qwen4_exp
