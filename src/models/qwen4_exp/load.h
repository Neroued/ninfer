#pragma once

#include "models/qwen4_exp/model.h"

namespace ninfer::models::qwen4_exp {

// One local materialization. The Reader remains alive until materialize_model returns;
// the definition and all resulting storage survive independently of it.
class ResolvedLoadPlan {
public:
    ResolvedLoadPlan(ResolvedLoadPlan&&) noexcept            = default;
    ResolvedLoadPlan& operator=(ResolvedLoadPlan&&) noexcept = default;
    ResolvedLoadPlan(const ResolvedLoadPlan&)                = delete;
    ResolvedLoadPlan& operator=(const ResolvedLoadPlan&)     = delete;

    [[nodiscard]] const ModelDefinition& definition() const { return *definition_; }

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const {
        return materialization_;
    }

    [[nodiscard]] const ple::disk::Table& ple_table() const { return *ple_; }

    [[nodiscard]] const std::shared_ptr<const ple::disk::Table>& ple_table_owner() const {
        return ple_;
    }

private:
    friend ResolvedLoadPlan resolve_load(const artifact::Reader&,
                                         std::shared_ptr<const ModelDefinition>, Device,
                                         std::optional<Device>,
                                         std::shared_ptr<const ple::disk::Table>);
    friend std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                         const StartupObserver*);
    ResolvedLoadPlan(const artifact::Reader&, std::shared_ptr<const ModelDefinition>,
                     artifact::MaterializationPlan, std::vector<Device>, std::vector<Device>,
                     std::shared_ptr<const ple::disk::Table>);
    const artifact::Reader* reader_;
    std::shared_ptr<const ModelDefinition> definition_;
    artifact::MaterializationPlan materialization_;
    std::vector<Device> devices_, bank_devices_;
    std::shared_ptr<const ple::disk::Table> ple_;
};

[[nodiscard]] ResolvedLoadPlan resolve_load(const artifact::Reader&,
                                            std::shared_ptr<const ModelDefinition>, Device,
                                            std::optional<Device> main_experts            = {},
                                            std::shared_ptr<const ple::disk::Table> table = {});
[[nodiscard]] std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                            const StartupObserver* = nullptr);

} // namespace ninfer::models::qwen4_exp
