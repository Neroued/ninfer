#include "models/qwen4_exp/load.h"

#include "models/common/bindings.h"

namespace ninfer::models::qwen4_exp {

ResolvedLoadPlan::ResolvedLoadPlan(const artifact::Reader& reader,
                                   std::shared_ptr<const ModelDefinition> definition,
                                   artifact::MaterializationPlan materialization,
                                   std::vector<Device> devices, std::vector<Device> bank_devices,
                                   std::shared_ptr<const ple::disk::Table> ple)
    : reader_(&reader), definition_(std::move(definition)),
      materialization_(std::move(materialization)), devices_(std::move(devices)),
      bank_devices_(std::move(bank_devices)), ple_(std::move(ple)) {}

ResolvedLoadPlan resolve_load(const artifact::Reader& reader,
                              std::shared_ptr<const ModelDefinition> definition, Device cuda,
                              std::optional<Device> main_experts,
                              std::shared_ptr<const ple::disk::Table> table) {
    if (!definition || definition->artifact_id() != reader.artifact_id())
        throw artifact::ArtifactError("model definition belongs to another artifact");
    const auto expert_device = main_experts.value_or(cuda);
    if (cuda.type() != DeviceType::CUDA ||
        (expert_device != cuda && expert_device != Device::cpu()))
        throw artifact::ArtifactError("weight placement requires the selected GPU or CPU");
    std::vector<Device> devices(definition->weight_data().size(), cuda);
    std::vector<Device> banks(definition->expert_banks().size(), cuda);
    for (const auto& layer : definition->weights().text.layers) {
        const auto& w = layer.moe;
        for (const auto id : {w.gate, w.up, w.down}) devices[id.index] = expert_device;
        banks[w.bank.index] = expert_device;
    }
    std::vector<artifact::PlacementRequest> requests;
    for (std::size_t i = 0; i < devices.size(); ++i)
        for (const auto& part : definition->parameter({i}).binding.parts)
            requests.push_back({part.object, devices[i],
                                devices[i] == Device::cpu() ? artifact::HostAccess::CudaTransfer
                                                            : artifact::HostAccess::CpuOnly});
    auto storage = artifact::plan_materialization(reader, requests);
    if (!table)
        table = std::make_shared<const ple::disk::Table>(reader, definition->ple_parameters());
    return {reader,           std::move(definition), std::move(storage), std::move(devices),
            std::move(banks), std::move(table)};
}

std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&& plan, DeviceContext& device,
                                              const StartupObserver* observer) {
    if (!plan.definition_) throw artifact::ArtifactError("load plan was already consumed");
    auto definition = std::move(plan.definition_);
    auto backing    = artifact::materialize(*plan.reader_, std::move(plan.materialization_),
                                            device.transfer_view(), observer);
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
                                                      std::move(bound), std::move(banks),
                                                      std::move(plan.ple_)));
}

} // namespace ninfer::models::qwen4_exp
