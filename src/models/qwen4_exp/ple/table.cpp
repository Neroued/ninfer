#include "models/qwen4_exp/ple/table.h"

namespace ninfer::models::qwen4_exp {

PleTablePlan plan_ple_table(const artifact::Reader& reader, const ModelDefinition& definition,
                            ple::Placement placement, ParallelCoordinate consumer, Device device,
                            std::shared_ptr<const ple::disk::Table> disk) {
    if (reader.artifact_id() != definition.artifact_id())
        throw artifact::ArtifactError("PLE lookup parameters belong to another artifact");
    const auto& parameters = definition.ple_parameters();
    PleTablePlan plan{placement, ple::shard_range(placement, parameters.shards.size(), consumer)};
    if (placement == ple::Placement::Disk) {
        plan.disk =
            disk ? std::move(disk) : std::make_shared<const ple::disk::Table>(reader, parameters);
    } else {
        std::vector<artifact::PlacementRequest> requests;
        for (const auto& shard :
             std::span(parameters.shards).subspan(plan.shards.begin, plan.shards.size))
            requests.push_back({shard.object, device});
        plan.encoded = artifact::plan_materialization(reader, requests);
    }
    return plan;
}

PleTable materialize_ple_table(PleTablePlan& plan, const ple::LookupParameters& parameters,
                               DeviceContext& device, const StartupObserver* observer) {
    if (plan.disk) return std::move(plan.disk);
    DevicePleTable table;
    if (!plan.shards.size) return table;
    const auto& reader = *plan.encoded.source;
    table.backing =
        artifact::materialize(reader, std::move(plan.encoded), device.transfer_view(), observer);
    table.decode = DeviceBuffer(device.device(), plan.shards.size * sizeof(Nf4Bf16Lut));
    table.shards.reserve(plan.shards.size);
    std::vector<Nf4Bf16Lut> luts;
    luts.reserve(plan.shards.size);
    for (const auto& shard :
         std::span(parameters.shards).subspan(plan.shards.begin, plan.shards.size)) {
        const auto& parent = table.backing.device_parent(shard.object);
        const auto offset  = luts.size() * sizeof(Nf4Bf16Lut);
        luts.push_back(make_nf4_bf16_lut(shard.divisor));
        table.shards.push_back({{parent.data, parent.geometry.matrix_scale_offset, device.device()},
                                Tensor(table.decode.span().subspan(offset, sizeof(Nf4Bf16Lut)),
                                       DType::BF16, {16, 128}),
                                static_cast<std::uint32_t>(shard.row_end - shard.geometry.shape[0]),
                                static_cast<std::uint32_t>(shard.geometry.shape[0])});
    }
    table.decode.copy_from_host(luts.data(), table.decode.size());
    return table;
}

} // namespace ninfer::models::qwen4_exp
