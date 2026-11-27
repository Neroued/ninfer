#pragma once

#include "artifact/materializer.h"
#include "models/qwen4_exp/definition.h"
#include "models/qwen4_exp/ple/disk/table.h"
#include "models/qwen4_exp/ple/placement.h"
#include "ninfer/ops/ple.h"

namespace ninfer::models::qwen4_exp {

struct DevicePleTable {
    artifact::MaterializedArtifact backing;
    DeviceBuffer decode;
    std::vector<ops::Nf4EmbeddingShard> shards;
};

using PleTable = std::variant<std::shared_ptr<const ple::disk::Table>, DevicePleTable>;

struct PleTablePlan {
    ple::Placement placement;
    IndexRange shards;
    std::shared_ptr<const ple::disk::Table> disk;
    artifact::MaterializationPlan encoded;

    [[nodiscard]] std::size_t device_bytes() const {
        return encoded.device_capacity_bytes + std::size_t(shards.size) * sizeof(Nf4Bf16Lut);
    }

    [[nodiscard]] std::size_t host_bytes() const { return disk ? disk->memory_bytes() : 0; }

    [[nodiscard]] ple::disk::TableGeometry geometry() const {
        return disk ? disk->geometry() : ple::disk::TableGeometry{};
    }
};

[[nodiscard]] PleTablePlan plan_ple_table(const artifact::Reader&, const ModelDefinition&,
                                          ple::Placement, ParallelCoordinate consumer, Device,
                                          std::shared_ptr<const ple::disk::Table> = {});
[[nodiscard]] PleTable materialize_ple_table(PleTablePlan&, const ple::LookupParameters&,
                                             DeviceContext&, const StartupObserver*);

} // namespace ninfer::models::qwen4_exp
