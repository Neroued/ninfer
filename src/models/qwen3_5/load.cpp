#include "models/qwen3_5/load.h"

#include "models/common/bindings.h"

namespace ninfer::models::qwen3_5 {

ResolvedLoadPlan::ResolvedLoadPlan(const artifact::Reader& reader,
                                   std::shared_ptr<const ModelDefinition> definition,
                                   artifact::MaterializationPlan materialization,
                                   std::vector<Device> devices, std::vector<Device> bank_devices)
    : reader_(&reader), definition_(std::move(definition)),
      materialization_(std::move(materialization)), devices_(std::move(devices)),
      bank_devices_(std::move(bank_devices)) {}

ResolvedLoadPlan resolve_load(const artifact::Reader& reader,
                              std::shared_ptr<const ModelDefinition> definition, Device cuda,
                              std::optional<Device> main_experts) {
    if (!definition || definition->artifact_id() != reader.artifact_id())
        throw artifact::ArtifactError("model definition belongs to another artifact");
    const auto expert_device = main_experts.value_or(cuda);
    if (cuda.type() != DeviceType::CUDA ||
        (expert_device != cuda && expert_device != Device::cpu()))
        throw artifact::ArtifactError("weight placement requires the selected GPU or CPU");
    std::vector<Device> devices(definition->weight_data().size(), cuda);
    std::vector<Device> banks(definition->expert_banks().size(), cuda);
    // Only Main experts participate in the offload traversal. MTP proposals use resident banks.
    for (const auto& layer : definition->weights().text.layers) {
        if (const auto* moe = std::get_if<MoeWeights>(&layer.ffn)) {
            banks[moe->bank.index] = expert_device;
            for (const auto& expert : moe->experts)
                for (const auto id : {expert.gate, expert.up, expert.down})
                    devices[id.index] = expert_device;
        }
    }
    std::vector<artifact::PlacementRequest> requests;
    for (std::size_t i = 0; i < devices.size(); ++i)
        for (const auto& part : definition->parameter({i}).binding.parts)
            requests.push_back({part.object, devices[i],
                                devices[i] == Device::cpu() ? artifact::HostAccess::CudaTransfer
                                                            : artifact::HostAccess::CpuOnly});
    auto storage = artifact::plan_materialization(reader, requests);
    return {reader, std::move(definition), std::move(storage), std::move(devices),
            std::move(banks)};
}

ResolvedLoadPlan resolve_rank_load(const artifact::Reader& reader,
                                   std::shared_ptr<const ModelDefinition> definition,
                                   const RankPlan& rank, Device device) {
    if (!definition || definition->artifact_id() != reader.artifact_id())
        throw artifact::ArtifactError("rank definition belongs to another artifact");
    if (rank.size != 1 || rank.rank != 0 || rank.ep_size != 1 || device.type() != DeviceType::CUDA)
        throw std::invalid_argument("Qwen3.5 rank loading requires one CUDA rank per replica");
    const auto banks = definition->expert_banks();
    if (rank.experts.size() != banks.size())
        throw artifact::ArtifactError("Qwen3.5 rank expert bank count differs from definition");

    std::vector<ExpertWeightPartition> experts(banks.size());
    const auto append = [&](const BlockWeights& block) {
        const auto* moe = std::get_if<MoeWeights>(&block.ffn);
        if (!moe) return;
        const auto index = moe->bank.index;
        auto& expert     = experts.at(index);
        expert.bank      = banks[index];
        expert.partition = rank.experts[index];
        expert.parameters.reserve(3 * moe->experts.size());
        for (const auto& e : moe->experts)
            expert.parameters.insert(expert.parameters.end(), {e.gate, e.up, e.down});
    };
    const auto& weights = definition->weights();
    for (const auto& block : weights.text.layers) append(block);
    if (weights.mtp) append(weights.mtp->layer);

    auto selection = select_weights(reader, definition->parameters(), definition->weight_data(),
                                    rank.whole, rank.assemblies, experts, rank.omitted, device);
    auto storage   = artifact::plan_materialization(reader, selection.requests);
    ResolvedLoadPlan result(reader, std::move(definition), std::move(storage), {}, {});
    result.selection_ = std::move(selection);
    return result;
}

std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&& plan, DeviceContext& device,
                                              const StartupObserver* observer) {
    if (!plan.definition_) throw artifact::ArtifactError("load plan was already consumed");
    auto definition = std::move(plan.definition_);
    auto requests           = plan.selection_ ? std::move(plan.materialization_.requests)
                                              : std::vector<artifact::StorageId>{};
    const auto parent_count = plan.materialization_.tensors.size();
    auto backing    = artifact::materialize(*plan.reader_, std::move(plan.materialization_),
                                            device.transfer_view(), observer);
    if (plan.selection_) {
        auto out =
            std::unique_ptr<ModelShard>(new ModelShard(definition, std::move(backing), {}, {}));
        out->local_parents_.reserve(parent_count);
        for (std::size_t i = 0; i < parent_count; ++i)
            out->local_parents_.push_back(out->backing_.parent({i}, device.device()));
        out->bound_ = bind_selected_weights(definition->weight_data(), *plan.selection_, requests,
                                            out->local_parents_);
        out->local_banks_ =
            bind_selected_experts(*plan.selection_, requests, out->local_parents_, device.device());
        out->expert_descriptions_ = std::move(plan.selection_->expert_descriptions);
        return out;
    }
    std::vector<PendingWeight> pending;
    pending.reserve(definition->weight_data().size());
    for (std::size_t i = 0; i < definition->weight_data().size(); ++i) {
        const auto& description = definition->weight_data()[i];
        pending.push_back(
            {definition->parameter({i}), description.uses, description.source_objects});
    }
    auto bound = resolve_weights(std::move(pending), backing, plan.devices_);
    auto banks = moe::bind_banks(definition->expert_banks(), backing, plan.bank_devices_);
    return std::unique_ptr<ModelShard>(new ModelShard(std::move(definition), std::move(backing),
                                                      std::move(bound), std::move(banks)));
}

} // namespace ninfer::models::qwen3_5
