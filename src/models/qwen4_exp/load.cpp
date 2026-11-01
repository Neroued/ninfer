#include "models/qwen4_exp/load.h"

#include "models/common/bindings.h"
#include "core/nf4.h"

namespace ninfer::models::qwen4_exp {

ResolvedLoadPlan::ResolvedLoadPlan(const artifact::Reader& reader,
                                   std::shared_ptr<const ModelDefinition> definition,
                                   artifact::MaterializationPlan materialization,
                                   std::vector<Device> devices, std::vector<Device> bank_devices,
                                   std::shared_ptr<const ple::disk::Table> ple)
    : reader_(&reader), definition_(std::move(definition)),
      materialization_(std::move(materialization)), devices_(std::move(devices)),
      bank_devices_(std::move(bank_devices)), ple_(std::move(ple)),
      rank_(plan_single_rank(*definition_)) {}

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

ResolvedLoadPlan resolve_rank_load(const artifact::Reader& reader,
                                   std::shared_ptr<const ModelDefinition> definition, RankPlan rank,
                                   Device device) {
    if (rank.size == 1) return resolve_load(reader, std::move(definition), device);
    if (!definition || definition->artifact_id() != reader.artifact_id())
        throw artifact::ArtifactError("rank definition belongs to another artifact");
    auto selection = select_weights(reader, *definition, rank, device);
    auto storage   = artifact::plan_materialization(reader, selection.weights.requests);
    ResolvedLoadPlan out(reader, std::move(definition), std::move(storage), {}, {}, {});
    out.rank_      = std::move(rank);
    out.selection_ = std::move(selection);
    out.parents_.reserve(out.materialization_.tensors.size());
    for (const auto& tensor : out.materialization_.tensors)
        out.parents_.push_back({tensor.geometry, device, nullptr, {}});
    out.descriptions_ =
        bind_selected_weights(out.definition_->weight_data(), out.selection_->weights,
                              out.materialization_.requests, out.parents_);
    return out;
}

std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&& plan, DeviceContext& device,
                                              const StartupObserver* observer) {
    if (!plan.definition_) throw artifact::ArtifactError("load plan was already consumed");
    auto definition         = std::move(plan.definition_);
    auto requests           = std::move(plan.materialization_.requests);
    const auto parent_count = plan.materialization_.tensors.size();
    auto backing            = artifact::materialize(*plan.reader_, std::move(plan.materialization_),
                                                    device.transfer_view(), observer);
    if (plan.selection_) {
        auto out =
            std::unique_ptr<ModelShard>(new ModelShard(definition, std::move(backing), {}, {}, {}));
        out->rank_ = std::move(plan.rank_);
        out->local_parents_.reserve(parent_count);
        for (std::size_t i = 0; i < parent_count; ++i)
            out->local_parents_.push_back(out->backing_.parent({i}, device.device()));
        out->bound_ = bind_selected_weights(definition->weight_data(), plan.selection_->weights,
                                            requests, out->local_parents_);
        const auto parent = [&](std::size_t request) -> const WeightParent& {
            return out->local_parents_.at(requests.at(request).index);
        };
        out->local_banks_         = bind_selected_experts(plan.selection_->weights, requests,
                                                          out->local_parents_, device.device());
        out->expert_descriptions_ = std::move(plan.selection_->weights.expert_descriptions);
        auto& table = out->ple_.emplace<DevicePleTable>();
        table.decode =
            DeviceBuffer(device.device(), plan.selection_->ple.size() * sizeof(Nf4Bf16Lut));
        for (std::size_t i = 0; i < plan.selection_->ple.size(); ++i) {
            const auto shard_id = out->rank_.ple_shards.begin + i;
            const auto& shard   = definition->ple_parameters().shards.at(shard_id);
            const auto& p       = parent(plan.selection_->ple[i]);
            const auto lut      = make_nf4_bf16_lut(shard.divisor);
            const auto offset   = i * sizeof(lut);
            table.decode.copy_from_host(&lut, sizeof(lut), offset);
            table.shards.push_back(
                {{p.data, p.geometry.divisor_offset, device.device()},
                 Tensor(table.decode.span().subspan(offset, sizeof(lut)), DType::BF16, {16, 128}),
                 static_cast<std::uint32_t>(shard.row_end - shard.geometry.shape[0]),
                 static_cast<std::uint32_t>(shard.geometry.shape[0])});
        }
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
    auto out = std::unique_ptr<ModelShard>(new ModelShard(std::move(definition), std::move(backing),
                                                          std::move(bound), std::move(banks),
                                                          std::move(plan.ple_)));
    out->rank_ = std::move(plan.rank_);
    return out;
}

} // namespace ninfer::models::qwen4_exp
